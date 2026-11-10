"""Add q75 at m80/90/95/99; reuse audited q80/90/95 results in a 4x4 grid."""
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
import torch
import run_pc1_length_percentile_sweep as previous
from export_pc1_density_overview import SOURCES, read_csv
from run_years_pc1_density import sha256, write_csv

MASSES=(.80,.90,.95,.99)
QUANTILES=(.75,.80,.90,.95)
OLD=Path('outputs/experiments/pc1_length_percentile_sweep/job_1932529')
ANCHOR=previous.ANCHOR
BASELINE=previous.BASELINE


def score_extension(center,vector,half_length80,component):
    parts=[previous.percentile.score_percentiles(center,vector,half_length80*previous.length_factor(m),component) for m in MASSES]
    lengths=np.array([half_length80*previous.length_factor(m) for m in MASSES])
    exact=np.stack([p['exact_coverage'] for p in parts])
    accepted=2*lengths[:,None,None]*exact
    rejected=2*lengths[:,None,None]-accepted
    tol=1e-10*max(1.,lengths.max())
    if (np.diff(accepted,axis=0)<-tol).any() or (np.diff(rejected,axis=0)<-tol).any():
        raise ValueError('Accepted/unsupported physical length decreased')
    if any(not np.array_equal(p['directions'],parts[0]['directions']) for p in parts):
        raise ValueError('Random directions changed with mass')
    if any(not np.array_equal(p['center_supported'],parts[0]['center_supported']) for p in parts):
        raise ValueError('Center acceptance changed with mass')
    return parts


def validate_q80(part,saved):
    """New q75/q80 share a grid; q80 must reproduce its audited prior result."""
    np.testing.assert_array_equal(part['directions'],saved['directions'].numpy())
    np.testing.assert_allclose(part['exact_coverage'][1],saved['exact_coverage'][0].numpy(),rtol=0,atol=1e-12)
    difference=float(np.max(np.abs(part['direction_coverage'][1]-saved['direction_coverage'][0].numpy())))
    bound=float(part['max_grid_error'][1])+float(saved['max_grid_error'][0])+1e-12
    if difference>bound: raise ValueError('q80 reproduction exceeds numerical bound')
    if part['mask_points']==saved['mask_points']:
        np.testing.assert_array_equal(part['acceptance_mask_packed'][1],saved['acceptance_mask_packed'][0].numpy())
    return difference


def summaries_from_rows(rows,config):
    summaries=[]
    for mass in MASSES:
        valid=[r for r in rows if r['run_id']==config['run_id'] and r['segment_mass']==mass]
        summary=dict(run_id=config['run_id'],kind=config['kind'],K=int(config['K']),threshold=float(config['threshold']),
            segment_mass=mass,quantile=.75,eligible=len(valid),total=int(config['K']))
        for field in ['coverage_pc1','coverage_random','delta_coverage']:
            values=np.array([r[field] for r in valid])
            for label,func in [('mean',np.mean),('median',np.median),('q25',lambda v:np.quantile(v,.25)),('q75',lambda v:np.quantile(v,.75))]:
                summary[label+'_'+field]=float(func(values))
        summary['fraction_full_pc1']=float(np.mean([r['coverage_pc1']==1. for r in valid]))
        summary['fraction_full_random_mean']=float(np.mean([r['coverage_random']==1. for r in valid]))
        summary['fraction_full_random_directions']=float(np.mean([r['random_full_fraction'] for r in valid]))
        summaries.append(summary)
    return summaries


def run_case(index,root,workers,smoke):
    dataset,_,layer,source,_=SOURCES[index]
    name=f'{dataset}_l{layer:02}'
    old=BASELINE/name
    prior=OLD/name
    assert json.loads((prior/'COMPLETED.json').read_text())['completed']
    assert json.loads((prior/'validation.json').read_text())['passed']
    protocol_old=json.loads((old/'manifest.json').read_text())
    calibration=torch.load(old/'calibration.pt',weights_only=True,map_location='cpu')
    reference=calibration['reference'].numpy()
    assert hashlib.sha256(reference.tobytes()).hexdigest()==protocol_old['reference_sha256']
    assert calibration['calibration']==protocol_old['calibration']
    assert protocol_old['baseline_manifest']['probability']==.8
    scores=calibration['calibration_scores'].numpy()
    thresholds=np.quantile(scores,QUANTILES,method='higher')
    assert np.isfinite(thresholds).all() and np.all(thresholds>0) and np.all(np.diff(thresholds)>=0)
    np.testing.assert_array_equal(thresholds[1:],json.loads((prior/'manifest.json').read_text())['thresholds'])
    output=root/name
    output.mkdir(parents=True,exist_ok=True)
    protocol=dict(masses=MASSES,quantiles=QUANTILES,computed_quantile=.75,reproduction_quantile=.8,
        thresholds=thresholds.tolist(),calibration_acceptance=[float(np.mean(scores<=t)) for t in thresholds],
        quantile_order_statistics_1based=[math.ceil(q*(len(scores)-1))+1 for q in QUANTILES],
        calibration_source=str(old),calibration_sha256=sha256(old/'calibration.pt'),
        old_source=str(prior),old_manifest_sha256=sha256(prior/'manifest.json'),
        runner_sha256=sha256(Path(__file__)),scorer_sha256=sha256(Path(previous.percentile.__file__)),
        base_scorer_sha256=sha256(Path(previous.base.__file__)),
        grid_tolerance=previous.base.GRID_TOLERANCE,grid_sizes=previous.base.GRID_SIZES,
        k=20,random_directions=32,random_seed='42+component',smoke=smoke,job_id=os.environ['SLURM_JOB_ID'],
        geometry_bundle_sha256={})
    (output/'manifest.json').write_text(json.dumps(protocol,indent=2)+'\n')
    (output/'runner_at_compute.py').write_bytes(Path(__file__).read_bytes())
    rows,summaries,cache,differences=[],[],{},[]
    configs=read_csv(source/'summary.csv')
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),
        initializer=previous.percentile.initialize_worker,initargs=(reference,thresholds[:2])) as executor:
        for config in configs[:1] if smoke else configs:
            run_id=config['run_id']; bundle_path=source/(run_id+'.pt')
            assert sha256(bundle_path)==protocol_old['geometry_bundle_sha256'][run_id]
            bundle=torch.load(bundle_path,weights_only=True,map_location='cpu')
            geom=bundle['projected_geometry']
            saved=torch.load(prior/(run_id+'.pt'),weights_only=True,map_location='cpu')['components']
            eligible=sorted(bundle['components'])[:2] if smoke else sorted(bundle['components'])
            if not smoke: assert set(eligible)==set(saved)
            signature=hashlib.sha256(b''.join(geom[key].contiguous().numpy().tobytes() for key in ['centers','vectors','half_lengths'])+str(eligible).encode()).hexdigest()
            results=cache.get(signature)
            if results is None:
                futures={i:executor.submit(score_extension,geom['centers'][i].numpy(),geom['vectors'][i].numpy(),float(geom['half_lengths'][i]),i) for i in eligible}
                results={i:f.result() for i,f in futures.items()};cache[signature]=results
            for i,parts in results.items():
                old_parts=[saved[i]['control']]+saved[i]['by_mass']
                for mass,part,old_part in zip(MASSES,parts,old_parts):
                    differences.append(validate_q80(part,old_part))
                    row=dict(run_id=run_id,component=i,kind=config['kind'],K=int(config['K']),threshold=float(config['threshold']),
                        segment_mass=mass,quantile=.75,tau=float(thresholds[0]),half_length80=float(geom['half_lengths'][i]),
                        half_length=float(geom['half_lengths'][i])*previous.length_factor(mass),alpha=float(geom['alpha'][i]),mask_points=part['mask_points'])
                    for field in ['coverage_pc1','coverage_random','delta_coverage','random_full_fraction','center_supported','max_grid_error','max_refinement_change']:
                        row[field]=part[field][0].item()
                    rows.append(row)
            torch.save(dict(run_id=run_id,masses=MASSES,quantiles=(.75,.8),thresholds=torch.from_numpy(thresholds[:2]),
                components={i:[previous.base.as_tensors(p) for p in parts] for i,parts in results.items()}),output/(run_id+'.pt'))
            protocol['geometry_bundle_sha256'][run_id]=sha256(bundle_path)
            (output/'manifest.json').write_text(json.dumps(protocol,indent=2)+'\n')
            summaries.extend(summaries_from_rows(rows,config))
            write_csv(output/'components.csv',rows);write_csv(output/'summary.csv',summaries)
            print(f'{name} {run_id}: {len(eligible)} components x four masses at q75; q80 controls passed',flush=True)
    expected=len(read_csv(old/'components.csv'))*4
    if not smoke: assert len(summaries)==64 and len(rows)==expected
    validation=dict(passed=True,smoke=smoke,summaries=len(summaries),new_component_rows=len(rows),
        max_q80_reproduction_difference=max(differences),max_grid_error=max(r['max_grid_error'] for r in rows),
        max_refinement_change=max(r['max_refinement_change'] for r in rows),physical_lengths_nested=True,
        masks_q75_subset_q80=True,center_support_invariant=True,original_population=True)
    (output/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    (output/'COMPLETED.json').write_text(json.dumps(dict(completed=not smoke,**validation),indent=2)+'\n')


def collect(root):
    combined=[dict(r,provenance=str(OLD)) for r in read_csv(OLD/'tutte_le_576_configurazioni.csv')]
    combined.extend(dict(r,segment_mass=.8,provenance=str(ANCHOR)) for r in read_csv(ANCHOR/'tutte_le_192_configurazioni.csv'))
    validations={}
    for dataset,_,layer,_,_ in SOURCES:
        case=root/f'{dataset}_l{layer:02}'
        complete=json.loads((case/'COMPLETED.json').read_text())
        assert complete['completed'] and complete['passed']
        validations[case.name]=complete
        components=read_csv(case/'components.csv')
        for summary in read_csv(case/'summary.csv'):
            valid=[r for r in components if r['run_id']==summary['run_id'] and float(r['segment_mass'])==float(summary['segment_mass'])]
            assert len(valid)==int(summary['eligible'])
            for field in ['coverage_pc1','coverage_random','delta_coverage']:
                values=[float(r[field]) for r in valid]
                for label,func in [('mean',np.mean),('median',np.median),('q25',lambda v:np.quantile(v,.25)),('q75',lambda v:np.quantile(v,.75))]:
                    assert math.isclose(float(summary[label+'_'+field]),float(func(values)),abs_tol=1e-12,rel_tol=0)
            combined.append(dict(dataset=dataset,layer=layer,**summary,provenance=str(case)))
    keys=[(r['dataset'],int(r['layer']),r['run_id'],float(r['segment_mass']),float(r['quantile'])) for r in combined]
    assert len(keys)==1024 and len(set(keys))==1024
    grouped={}
    for r in combined:grouped.setdefault((r['dataset'],int(r['layer']),r['run_id'],float(r['segment_mass'])),[]).append(r)
    for group in grouped.values():
        ordered=sorted(group,key=lambda r:float(r['quantile']))
        assert [float(r['quantile']) for r in ordered]==list(QUANTILES)
        assert len({(int(r['eligible']),int(r['total'])) for r in ordered})==1
        for field in ['mean_coverage_pc1','mean_coverage_random','median_coverage_pc1','median_coverage_random']:
            assert np.all(np.diff([float(r[field]) for r in ordered])>=-.004), 'Coverage nesting exceeded discretization tolerance'
    write_csv(root/'tutte_le_1024_configurazioni.csv',combined)
    selected=[r for r in combined if float(r['threshold'])==.1]
    assert len(selected)==256
    write_csv(root/'tabelle_soglia_01.csv',selected)
    import plot_pc1_delta_matrices as matrices
    matrices.main(root=root,masses=MASSES,quantiles=QUANTILES)
    (root/'COMPLETED.json').write_text(json.dumps(dict(completed=True,masses=MASSES,quantiles=QUANTILES,
        configurations=1024,selected_rows=256,new_summaries=256,reused_summaries=768,case_validations=validations),indent=2)+'\n')
    print('Validated 1024 summaries; exported 4x4 matrices with all 256 selected values',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--experiment-index',type=int,choices=range(4))
    parser.add_argument('--workers',type=int,default=6)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--collect-only',action='store_true')
    args=parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_STEP_ID') in [None,'batch','extern']:
        raise RuntimeError('Run numerical work with srun on a compute node')
    args.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
    if args.collect_only:collect(args.output)
    elif args.experiment_index is not None:run_case(args.experiment_index,args.output,args.workers,args.smoke)
    else:parser.error('Specify case or collect-only')

if __name__=='__main__':main()
