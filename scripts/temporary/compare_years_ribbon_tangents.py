"""Compare saved HDDC and KMeans PCs with one fixed analytic spiral ribbon.

The wiki's tangent alignment/containment contracts are reused in ambient space.
This evaluates an estimated reference surface, not planted ground truth.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np
import torch

from dalg.evaluation.common import load_model, load_evaluation_data
from dalg.evaluation.subspaces import (effective_component_ranks, leading_covariance_eigenspaces,
    subspace_alignment, best_subset_alignment, leading_subspace_is_identifiable,
    PC_RELATIVE_EIGENGAP_THRESHOLD)
from dalg.evaluation.toy_manifold_geometry import _refined_grid_candidates, _distance_tied
import fit_years_spiral as curves
import fit_years_spiral_ribbon as ribbon

METRICS=('tangent_alignment','tangent_containment','tangent_partial_containment',
         'tangent_adjusted_alignment','sensitivity_containment_cap16')


def project_centers(model, centers, step=.25):
    """Eliminate radius analytically, search/refine separated year minima.

    QR compression is exact: the omitted orthogonal distance is constant in
    both surface coordinates. Grid-plus-local refinement is approximate global
    projection; candidate ambiguity and finer-grid checks are recorded.
    """
    q,_=np.linalg.qr(model['coef'].T,mode='reduced')
    small=dict(model,coef=model['coef']@q,origin=np.zeros(q.shape[1]))
    target=(centers-model['origin'])@q
    normal=np.maximum(0,np.sum((centers-model['origin'])**2,axis=1)-np.sum(target**2,axis=1))
    grid=np.linspace(1000,2000,int(round(1000/step))+1)
    centerline=ribbon.predict(small,grid)
    radial=ribbon.predict(small,grid,derivative='radius')
    energy=np.sum(radial**2,axis=1)
    cross=target@radial.T-np.sum(centerline*radial,axis=1)
    radii=np.clip(1+cross/energy,*model['radius_bounds'])
    distances=(np.sum(target**2,axis=1)[:,None]+np.sum(centerline**2,axis=1)
        -2*target@centerline.T-2*(radii-1)*cross+(radii-1)**2*energy)
    results=[]
    for k,z in enumerate(target):
        def value(year):
            y=np.array([year]);c=ribbon.predict(small,y)[0]
            d=ribbon.predict(small,y,derivative='radius')[0]
            radius=float(np.clip(1+np.dot(z-c,d)/np.dot(d,d),*model['radius_bounds']))
            residual=z-ribbon.predict(small,y,radius)[0]
            return float(np.dot(residual,residual)),radius
        curve=distances[k]
        locations=_refined_grid_candidates(torch.from_numpy(grid),torch.from_numpy(curve),
                                           lambda y:value(y)[0],periodic=False)
        candidates=[]
        for year in locations:
            error,radius=value(year)
            candidates.append((error,year,radius))
        candidates.sort()
        distinct=[]
        for item in candidates:
            if all(abs(item[1]-other[1])>1e-3 or abs(item[2]-other[2])>1e-5 for other in distinct):
                distinct.append(item)
        best=distinct[0]
        gap=float(distinct[1][0]-best[0]) if len(distinct)>1 else None
        ambiguous=len(distinct)>1 and _distance_tied(distinct[1][0]+normal[k],best[0]+normal[k])
        boundary_y=min(best[1]-1000,2000-best[1])<1e-4
        boundary_r=min(best[2]-model['radius_bounds'][0],model['radius_bounds'][1]-best[2])<1e-7
        results.append(dict(year=best[1],radius=best[2],distance=float(np.sqrt(max(0,best[0]+normal[k]))),
                            ambiguous=ambiguous,second_minimum_squared_gap=gap,
                            boundary_year=bool(boundary_y),boundary_radius=bool(boundary_r),
                            boundary=bool(boundary_y or boundary_r)))
    return results


def isotropic_eigenspaces(model):
    """Exact HDDC covariance PCs, retaining one noise eigenvalue for the boundary."""
    with torch.no_grad():
        w=model._W().detach().cpu().double()
        psi=model._psi().detach().cpu().double()
        if not torch.allclose(psi,psi[:,:1].expand_as(psi),rtol=1e-12,atol=0):
            raise ValueError('Require isotropic HDDC covariances')
        u,s,_=torch.linalg.svd(w,full_matrices=False)
        values=torch.cat((s.square()+psi[:,:1],psi[:,:1]),dim=1)
    return values,u


def component_scores(tangent,pcs,values,rank,dimension):
    """Same rank gates, boundary checks and subset rule as the wiki evaluator."""
    result={name:None for name in METRICS}
    if tangent is None:return result
    identifiable=lambda n:leading_subspace_is_identifiable(values,n,dimension,PC_RELATIVE_EIGENGAP_THRESHOLD)
    if rank==0:
        result['tangent_adjusted_alignment']=(0.,None)
        return result
    if rank<2:
        if identifiable(rank):
            overlap,worst=subspace_alignment(pcs[:,:rank],tangent)
            result['tangent_partial_containment']=(overlap,worst)
            result['tangent_adjusted_alignment']=(overlap*rank/2,None)
        return result
    if identifiable(2):
        overlap,worst=subspace_alignment(tangent,pcs[:,:2])
        result['tangent_alignment']=(overlap,worst)
        result['tangent_adjusted_alignment']=(overlap,None)
    if identifiable(rank):
        result['tangent_containment']=best_subset_alignment(tangent,pcs[:,:rank])
    capped=min(rank,16)
    if identifiable(capped):
        result['sensitivity_containment_cap16']=best_subset_alignment(tangent,pcs[:,:capped])
    return result


def summarize(rows):
    eligible=[row for row in rows if row['eligible']]
    result=dict(components=len(rows),eligible_components=len(eligible),excluded_components=len(rows)-len(eligible),
        ambiguous_projections=sum(row['ambiguous'] for row in eligible),
        boundary_projections=sum(row['boundary'] for row in eligible),
        mean_effective_rank=float(np.mean([row['rank'] for row in eligible])) if eligible else None,
        median_surface_distance=float(np.median([row['distance'] for row in eligible])) if eligible else None)
    for metric in METRICS:
        valid=[row for row in eligible if row[metric+'_overlap'] is not None]
        interior=[row for row in valid if not row['boundary']]
        result[metric]=dict(mean=float(np.mean([row[metric+'_overlap'] for row in valid])) if valid else None,
            worst_direction_cosine_mean=float(np.mean([row[metric+'_worst'] for row in valid])) if valid and metric!='tangent_adjusted_alignment' else None,
            valid_components=len(valid),undefined_components=len(eligible)-len(valid),
            interior_mean=float(np.mean([row[metric+'_overlap'] for row in interior])) if interior else None,
            interior_valid_components=len(interior))
    return result


def plots(output,summaries):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    for column,K in enumerate((50,100)):
        for line,metric in enumerate(('tangent_alignment','tangent_containment')):
            ax=axes[line,column]
            for kind in ('hddc','kmeans'):
                rows=sorted([s for s in summaries if s['K']==K and s['kind']==kind],key=lambda s:s['threshold'])
                ax.plot([s['threshold'] for s in rows],[s[metric]['mean'] for s in rows],'o-',label=kind.upper())
            ax.set(xscale='log',ylim=(0,1),xlabel='Saved rank threshold',ylabel='Mean tangent overlap',title=f'K={K}: {metric}')
            ax.legend()
    fig.savefig(output/'comparison.png',dpi=180);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--activation-dir',type=Path,required=True)
    parser.add_argument('--ribbon',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if not os.environ.get('SLURM_STEP_ID'):raise RuntimeError('Run through srun')
    output=args.output;output.mkdir(parents=True,exist_ok=False);(output/'source').mkdir()
    for source in (Path(__file__),Path(ribbon.__file__),Path(curves.__file__)):
        shutil.copy2(source,output/'source'/source.name)
    reference,_=ribbon.load_model(args.ribbon)
    reference_hash=curves.sha256(args.ribbon)
    summaries,all_rows,fingerprints=[],[],{str(args.ribbon.resolve()):reference_hash}
    projection_cache={}
    projection_audits=[]
    for kind,folder in [('hddc','sae_manifold_years_neutral_hddc_em'),('kmeans','sae_manifold_years_neutral_kmeans_pca')]:
        runs=sorted((args.activation_dir/folder).glob('*__l04__*'))
        if len(runs)!=8:raise ValueError(f'Expected eight layer-4 {kind} runs, found {len(runs)}')
        for run in runs:
            spec=json.loads((run/'run_spec.json').read_text());settings=spec['identity']['training_args']
            assert Path(settings['shard_dir']).resolve()==args.activation_dir.resolve()
            assert settings['seed']==42 and settings['split_seed']==42 and settings['val_frac']==.1
            threshold=float(settings['surgery_threshold'])
            model=load_model(run,kind)
            data=load_evaluation_data(run,shard_dir=args.activation_dir,layer=4,model=model,model_kind=kind)
            means=model.mu.detach().cpu().double().numpy()
            key=hashlib.sha256(means.tobytes()).hexdigest()
            if key not in projection_cache:
                projections=project_centers(reference,means)
                check=project_centers(reference,means[:5],step=.125)
                discrepancy=max(abs(a['distance']-b['distance']) for a,b in zip(projections,check))
                if discrepancy>1e-6:raise ValueError(f'Projection refinement disagrees by {discrepancy}')
                projection_audits.append(dict(run=run.name,checked_centers=len(check),maximum_distance_discrepancy=discrepancy))
                ys=np.array([p['year'] for p in projections]);rs=np.array([p['radius'] for p in projections])
                tangent=torch.from_numpy(ribbon.tangent_space(reference,ys,rs))
                projection_cache[key]=(projections,tangent)
            projections,tangents=projection_cache[key]
            ranks=effective_component_ranks(model,1.)
            values,pcs=(isotropic_eigenspaces(model) if kind=='hddc' else
                        leading_covariance_eigenspaces(model,max(1,int(ranks.max()))))
            rows=[]
            for k in range(model.K):
                live=bool(data.cluster_sizes[k]>0)
                valid_pca=kind!='kmeans' or bool(model.pca_valid[k])
                eligible=live and valid_pca
                scores=component_scores(tangents[k] if eligible and not projections[k]['ambiguous'] else None,
                                        pcs[k],values[k],int(ranks[k]),model.D)
                row=dict(run=run.name,kind=kind,K=model.K,threshold=threshold,component=k,
                    rank=int(ranks[k]),live=live,pca_valid=valid_pca,eligible=eligible,**projections[k])
                for metric,score in scores.items():
                    row[metric+'_overlap']=None if score is None else score[0]
                    row[metric+'_worst']=None if score is None else score[1]
                rows.append(row)
            summary=dict(run=run.name,kind=kind,K=model.K,threshold=threshold,**summarize(rows))
            summaries.append(summary);all_rows.extend(rows)
            filename='mfa_model' if kind=='hddc' else 'kmeans_model'
            for path in [run/(filename+'.pt'),run/(filename+'_assignments.pt'),run/'run_spec.json',run/'val_indices.json']:
                fingerprints[str(path.resolve())]=curves.sha256(path)
            print(kind,model.K,threshold,'rank',summary['mean_effective_rank'],
                  'alignment',summary['tangent_alignment'],'containment',summary['tangent_containment'],flush=True)
            curves.write_csv(output/'summary.csv',[dict(run=s['run'],kind=s['kind'],K=s['K'],threshold=s['threshold'],
                mean_rank=s['mean_effective_rank'],median_surface_distance=s['median_surface_distance'],
                **{m:s[m]['mean'] for m in METRICS}) for s in summaries])
            del model,data,pcs,values;gc.collect()
    comparisons=[]
    for K in (50,100):
        for threshold in (.0001,.001,.01,.1):
            pair={s['kind']:s for s in summaries if s['K']==K and s['threshold']==threshold}
            if set(pair)!=set(('hddc','kmeans')):raise ValueError('Missing matched comparison')
            comparisons.append(dict(K=K,threshold=threshold,**{m+'_hddc_minus_kmeans':
                (pair['hddc'][m]['mean']-pair['kmeans'][m]['mean']) if pair['hddc'][m]['mean'] is not None and pair['kmeans'][m]['mean'] is not None else None for m in METRICS}))
    curves.write_csv(output/'components.csv',all_rows)
    curves.write_csv(output/'matched_comparisons.csv',comparisons)
    result=dict(reference=str(args.ribbon.resolve()),reference_kind='estimated analytic spiral ribbon, not planted ground truth',
        metric_contract='docs/experiments/evaluation/toy-manifold-tiling.md',
        population='assignment-live components, excluding invalid KMeans PCA; unweighted valid-component means',
        boundary_policy='Unique boundary projections included in primary metrics; interior-only means supplied separately',
        summaries=summaries,matched_comparisons=comparisons,projection_audits=projection_audits)
    (output/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    (output/'manifest.json').write_text(json.dumps(dict(inputs=fingerprints,source_hash=curves.sha256(__file__),job_id=os.environ.get('SLURM_JOB_ID')),indent=2))
    plots(output,summaries)
    (output/'COMPLETED.json').write_text(json.dumps(dict(job_id=os.environ.get('SLURM_JOB_ID'))))
    print('Completed:',output,flush=True)


if __name__=='__main__':main()
