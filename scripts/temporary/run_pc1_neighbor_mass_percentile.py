"""Change only neighbor count across the fixed 4x4 PC1 mass/calibration grid."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
import torch
import run_pc1_extended_mass_percentile as extension
import run_pc1_neighbor_sweep as neighbor
from export_pc1_density_overview import SOURCES as ALL_SOURCES, read_csv
from run_years_pc1_density import sha256, write_csv

SOURCES=[source for source in ALL_SOURCES if source[2]==4]
MASSES=extension.MASSES
QUANTILES=extension.QUANTILES
NEIGHBORS=(5,10,20,40)
BASELINE=extension.BASELINE
OLD=extension.OLD
EXTENDED=Path('outputs/experiments/pc1_extended_mass_percentile/job_1934466')


def calibrate(reference,queries,k):
    if len(reference)<k or k<1:raise ValueError('Invalid neighbor count')
    distances=cdist(queries,reference)
    scores=np.partition(distances,k-1,axis=1)[:,k-1]
    thresholds=np.quantile(scores,QUANTILES,method='higher')
    if not np.isfinite(thresholds).all() or (thresholds<=0).any():raise ValueError('Invalid thresholds')
    return scores,thresholds


def validate_saved(part,qi,saved,old_qi):
    np.testing.assert_array_equal(part['directions'],saved['directions'].numpy())
    np.testing.assert_allclose(part['exact_coverage'][qi],saved['exact_coverage'][old_qi].numpy(),rtol=0,atol=1e-12)
    difference=float(np.max(np.abs(part['direction_coverage'][qi]-saved['direction_coverage'][old_qi].numpy())))
    if difference>float(part['max_grid_error'][qi])+float(saved['max_grid_error'][old_qi])+1e-12:
        raise ValueError('k20 reproduction exceeds discretization bounds')
    if part['mask_points']==saved['mask_points']:
        np.testing.assert_array_equal(part['acceptance_mask_packed'][qi],saved['acceptance_mask_packed'][old_qi].numpy())
    return difference


def aggregate(rows,config,k):
    result=[]
    for mass in MASSES:
        for q in QUANTILES:
            valid=[r for r in rows if r['run_id']==config['run_id'] and r['segment_mass']==mass and r['quantile']==q]
            summary=dict(run_id=config['run_id'],kind=config['kind'],K=int(config['K']),threshold=float(config['threshold']),
                neighbors=k,segment_mass=mass,quantile=q,eligible=len(valid),total=int(config['K']))
            for field in ['coverage_pc1','coverage_random','delta_coverage']:
                v=np.array([r[field] for r in valid])
                for label,value in [('mean',v.mean()),('median',np.median(v)),('q25',np.quantile(v,.25)),('q75',np.quantile(v,.75))]:
                    summary[label+'_'+field]=float(value)
            for name,field in [('fraction_full_pc1','coverage_pc1'),('fraction_full_random_mean','coverage_random')]:
                summary[name]=float(np.mean([r[field]==1. for r in valid]))
            summary['fraction_full_random_directions']=float(np.mean([r['random_full_fraction'] for r in valid]))
            result.append(summary)
    return result


def run_case(index,root,workers,smoke):
    k=NEIGHBORS[index//len(SOURCES)]
    dataset,_,layer,source,_=SOURCES[index%len(SOURCES)]
    name=f'{dataset}_l{layer:02}'
    old=BASELINE/name
    previous=json.loads((old/'manifest.json').read_text())
    calibration=torch.load(old/'calibration.pt',weights_only=True,map_location='cpu')
    assert calibration['calibration']==previous['calibration']
    reference=calibration['reference'].numpy()
    assert hashlib.sha256(reference.tobytes()).hexdigest()==previous['reference_sha256']
    queries,provenance=neighbor.recover_queries(previous,calibration)
    k20scores,k20thresholds=calibrate(reference,queries,20)
    np.testing.assert_array_equal(k20scores,calibration['calibration_scores'].numpy())
    np.testing.assert_array_equal(k20thresholds,json.loads((EXTENDED/name/'manifest.json').read_text())['thresholds'])
    scores,thresholds=calibrate(reference,queries,k)
    output=root/f'k{k}'/name
    output.mkdir(parents=True,exist_ok=True)
    protocol=dict(changed_parameter='neighbors only; thresholds recalibrated by the fixed rule',neighbors=k,masses=MASSES,quantiles=QUANTILES,
        thresholds=thresholds.tolist(),quantile_method='higher',
        calibration_source=str(old),calibration_sha256=sha256(old/'calibration.pt'),
        reference_sha256=previous['reference_sha256'],query_sha256=hashlib.sha256(queries.tobytes()).hexdigest(),
        old_manifest_sha256=sha256(old/'manifest.json'),reconstruction_provenance=provenance,
        calibration_acceptance=[float(np.mean(scores<=t)) for t in thresholds],
        grid_sizes=extension.previous.base.GRID_SIZES,grid_tolerance=extension.previous.base.GRID_TOLERANCE,
        random_directions=32,random_seed='42+component',smoke=smoke,job_id=os.environ['SLURM_JOB_ID'],
        source_sha256=sha256(Path(__file__)),scorer_sha256=sha256(Path(extension.previous.percentile.__file__)),geometry_bundle_sha256={})
    (output/'runner_at_compute.py').write_bytes(Path(__file__).read_bytes())
    (output/'manifest.json').write_text(json.dumps(protocol,indent=2)+'\n')
    torch.save(dict(reference=torch.from_numpy(reference),queries=torch.from_numpy(queries),calibration_scores=torch.from_numpy(scores),
        thresholds=torch.from_numpy(thresholds),neighbors=k,quantiles=QUANTILES,membership=calibration['calibration']),output/'calibration.pt')
    rows,summaries,cache,differences=[],[],{},[]
    configs=read_csv(source/'summary.csv')
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),
        initializer=extension.previous.percentile.initialize_worker,initargs=(reference,thresholds,k)) as executor:
        for config in configs[:1] if smoke else configs:
            run_id=config['run_id'];path=source/(run_id+'.pt')
            assert sha256(path)==previous['geometry_bundle_sha256'][run_id]
            bundle=torch.load(path,weights_only=True,map_location='cpu')
            geom=bundle['projected_geometry']
            old_components=torch.load(OLD/name/(run_id+'.pt'),weights_only=True,map_location='cpu')['components']
            old_q75=torch.load(EXTENDED/name/(run_id+'.pt'),weights_only=True,map_location='cpu')['components'] if k==20 else None
            eligible=sorted(bundle['components'])[:2] if smoke else sorted(bundle['components'])
            if not smoke:assert set(eligible)==set(old_components)
            signature=hashlib.sha256(b''.join(geom[key].contiguous().numpy().tobytes() for key in ['centers','vectors','half_lengths'])+str(eligible).encode()).hexdigest()
            results=cache.get(signature)
            if results is None:
                futures={i:executor.submit(extension.score_extension,geom['centers'][i].numpy(),geom['vectors'][i].numpy(),float(geom['half_lengths'][i]),i) for i in eligible}
                results={i:f.result() for i,f in futures.items()};cache[signature]=results
            for i,parts in results.items():
                old_parts=[old_components[i]['control']]+old_components[i]['by_mass']
                for mi,(mass,part) in enumerate(zip(MASSES,parts)):
                    np.testing.assert_array_equal(part['directions'],old_parts[mi]['directions'].numpy())
                    for qi,q in enumerate(QUANTILES):
                        if k==20:
                            saved,oi=(old_q75[i][mi],0) if qi==0 else (old_parts[mi],qi-1)
                            differences.append(validate_saved(part,qi,saved,oi))
                        row=dict(run_id=run_id,component=i,kind=config['kind'],K=int(config['K']),threshold=float(config['threshold']),
                            neighbors=k,segment_mass=mass,quantile=q,tau=float(thresholds[qi]),half_length80=float(geom['half_lengths'][i]),
                            half_length=float(geom['half_lengths'][i])*extension.previous.length_factor(mass),alpha=float(geom['alpha'][i]),mask_points=part['mask_points'])
                        for field in ['coverage_pc1','coverage_random','delta_coverage','random_full_fraction','center_supported','max_grid_error','max_refinement_change']:
                            row[field]=part[field][qi].item()
                        rows.append(row)
            torch.save(dict(run_id=run_id,neighbors=k,masses=MASSES,quantiles=QUANTILES,thresholds=torch.from_numpy(thresholds),
                components={i:[extension.previous.base.as_tensors(p) for p in parts] for i,parts in results.items()}),output/(run_id+'.pt'))
            protocol['geometry_bundle_sha256'][run_id]=sha256(path)
            (output/'manifest.json').write_text(json.dumps(protocol,indent=2)+'\n')
            summaries.extend(aggregate(rows,config,k))
            write_csv(output/'components.csv',rows);write_csv(output/'summary.csv',summaries)
            print(f'k{k} {name} {run_id}: {len(eligible)} components x16 settings passed',flush=True)
    if not smoke:
        assert len(summaries)==256 and len(rows)==16*len(read_csv(old/'components.csv'))
    validation=dict(passed=True,smoke=smoke,neighbors=k,summaries=len(summaries),component_rows=len(rows),
        max_k20_reproduction_difference=max(differences) if differences else None,
        max_grid_error=max(r['max_grid_error'] for r in rows),max_refinement_change=max(r['max_refinement_change'] for r in rows),
        original_geometry_population=True,original_R_Q=True,original_random_directions=True,q_masks_nested=True,physical_lengths_nested=True,
        cross_k_mask_nesting_assumed=False)
    (output/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    (output/'COMPLETED.json').write_text(json.dumps(dict(completed=not smoke,**validation),indent=2)+'\n')


def collect(root):
    import plot_pc1_delta_matrices as matrices
    all_rows=[]
    for k in NEIGHBORS:
        target=root/f'k{k}';combined=[];validations={}
        for dataset,_,layer,_,_ in SOURCES:
            case=target/f'{dataset}_l{layer:02}'
            status=json.loads((case/'COMPLETED.json').read_text())
            assert status['completed'] and status['passed']
            validations[case.name]=status
            components=read_csv(case/'components.csv')
            groups={}
            for row in components:groups.setdefault((row['run_id'],float(row['segment_mass']),float(row['quantile'])),[]).append(row)
            for summary in read_csv(case/'summary.csv'):
                v=groups[(summary['run_id'],float(summary['segment_mass']),float(summary['quantile']))]
                assert len(v)==int(summary['eligible'])
                for field in ['coverage_pc1','coverage_random','delta_coverage']:
                    assert math.isclose(float(summary['mean_'+field]),float(np.mean([float(r[field]) for r in v])),abs_tol=1e-12,rel_tol=0)
                combined.append(dict(dataset=dataset,layer=layer,**summary))
        assert len(combined)==512
        selected=[r for r in combined if float(r['threshold'])==.1];assert len(selected)==128
        write_csv(target/'tutte_le_512_configurazioni.csv',combined);write_csv(target/'tabelle_soglia_01.csv',selected)
        matrices.main(root=target,masses=MASSES,quantiles=QUANTILES,layer4_only=True,neighbors=k)
        (target/'COMPLETED.json').write_text(json.dumps(dict(completed=True,neighbors=k,configurations=512,validations=validations),indent=2)+'\n')
        all_rows.extend(combined)
    write_csv(root/'tutte_le_2048_configurazioni.csv',all_rows)
    (root/'COMPLETED.json').write_text(json.dumps(dict(completed=True,neighbors=NEIGHBORS,requested_neighbors=[5,10,40],
        control_neighbors=20,masses=MASSES,quantiles=QUANTILES,configurations=2048),indent=2)+'\n')
    (root/'report.md').write_text('# PC1 neighbor-count ablation\n\nOnly k changes; each k uses its own recalibrated thresholds with the same R/Q and quantile rule.\n\n'+
        '\n\n'.join(f'## k={k}'+(' (reproduction control)' if k==20 else '')+f'\n\n![Layer4](k{k}/delta_matrices/PC1_delta_layer4_K50_K100.png)' for k in NEIGHBORS)+'\n')
    print('Exported all four neighbor-count sweeps; k20 is a reproduction control',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--index',type=int,choices=range(8))
    parser.add_argument('--workers',type=int,default=6)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--collect-only',action='store_true')
    args=parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_STEP_ID') in [None,'batch','extern']:raise RuntimeError('Use srun')
    torch.set_num_threads(1);args.output.mkdir(parents=True,exist_ok=True)
    if args.collect_only:collect(args.output)
    elif args.index is not None:run_case(args.index,args.output,args.workers,args.smoke)
    else:parser.error('Specify index or collect-only')

if __name__=='__main__':main()
