"""Explicit helical ribbons fitted to layer-4 years, in all ambient coordinates.

S(y,r) = c + v*t + sum_j (A_j cos(w_j*t) + B_j sin(w_j*t))
         + (r-1) (A_m cos(w_m*t) + B_m sin(w_m*t)), t=(y-1500)/500.
The selected pair m supplies the radial surface coordinate. No spline sheet.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil

import numpy as np
from sklearn.decomposition import PCA

import fit_years_spiral as curves


def design(years, periods, radius=None, radial_period=None, derivative=False):
    f = curves.basis(years, dict(kind='fourier', periods=periods), derivative)
    if radius is not None:
        pair = 2 + 2 * periods.index(radial_period)
        f[:, pair:pair+2] *= np.asarray(radius)[:, None]
    return f


def solve_coefficients(x, f, origin):
    gram = f.T @ f
    scale = np.sqrt(np.maximum(0, gram.diagonal()))
    active = scale > 1e-10 * scale.max()
    selected = np.flatnonzero(active)
    scale = scale[active]
    normalized = gram[np.ix_(selected, selected)] / scale[:, None] / scale[None, :]
    response = f[:, selected].T @ (x - origin)
    solution, _, rank, _ = np.linalg.lstsq(normalized, response / scale[:, None], rcond=1e-11)
    coef = np.zeros((f.shape[1], x.shape[1]))
    coef[selected] = solution / scale[:, None]
    return coef, int(rank)


def predict(model, years, radius=1., derivative=None):
    years = np.atleast_1d(years).astype(float)
    radius = np.broadcast_to(radius, years.shape)
    if np.any(radius <= 0):
        raise ValueError('Spiral radius must be positive')
    f = design(years, model['periods'], radius, model['radial_period'], derivative=='year')
    if derivative == 'radius':
        pair = 2 + 2 * model['periods'].index(model['radial_period'])
        return f[:, pair:pair+2] @ model['coef'][pair:pair+2] / radius[:, None]
    if derivative not in (None, 'year'):
        raise ValueError('Derivative must be year or radius')
    result = f @ model['coef']
    return result if derivative else result + model['origin']


def tangent_space(model, years, radius=1., relative_tolerance=1e-10):
    """Orthonormal ambient tangent bases, (N,D,2), from analytic partials.

    Normalize Jacobian columns before checking independence so year/radius units
    do not determine the rank decision. Reject points without a regular 2D tangent.
    """
    jacobian = np.stack((predict(model, years, radius, derivative='year'),
                         predict(model, years, radius, derivative='radius')), axis=-1)
    norms = np.linalg.norm(jacobian, axis=1, keepdims=True)
    if not np.isfinite(jacobian).all() or np.any(norms <= 1e-14):
        raise ValueError('Surface Jacobian has a zero or nonfinite derivative')
    basis, singular, _ = np.linalg.svd(jacobian / norms, full_matrices=False)
    if np.any(singular[:, 1] <= relative_tolerance * singular[:, 0]):
        raise ValueError('Surface Jacobian is rank deficient: no regular 2D tangent')
    return basis


def project_radius(model, x, years, bounds=None):
    """Exact ambient least-squares radius at the supplied year coordinate."""
    radial = predict(model, years, derivative='radius')
    energy = np.einsum('ij,ij->i', radial, radial)
    if np.any(energy < 1e-12):
        raise ValueError('Degenerate oscillation pair cannot define a radial sheet')
    radius = 1 + np.einsum('ij,ij->i', x - predict(model, years), radial) / energy
    return radius if bounds is None else np.clip(radius, *bounds)


def fit_ribbon(x, years, periods, radial_period, iterations=60, tolerance=1e-7):
    periods = sorted(float(p) for p in periods)
    if radial_period not in periods:
        raise ValueError('Radial period must be present in the curve')
    origin = x.mean(0, dtype=float)
    radius, history = np.ones(len(x)), []
    model = dict(periods=periods, radial_period=float(radial_period), origin=origin)
    for iteration in range(iterations):
        f = design(years, periods, radius, radial_period)
        model['coef'], model['rank'] = solve_coefficients(x, f, origin)
        radius = np.maximum(.001, project_radius(model, x, years))
        normalization = radius.mean()
        radius /= normalization
        pair = 2 + 2 * periods.index(radial_period)
        model['coef'][pair:pair+2] *= normalization
        mse = float(np.mean((x - predict(model, years, radius)) ** 2))
        history.append(mse)
        if iteration >= 4 and abs(history[-2] - mse) / max(mse, 1e-20) < tolerance:
            break
    # Store a coefficient solve exactly consistent with the saved training radii.
    model['coef'], model['rank'] = solve_coefficients(
        x, design(years, periods, radius, radial_period), origin)
    raw = project_radius(model, x, years)
    bounds = np.quantile(raw, [.005, .995])
    bounds[0] = max(.001, bounds[0])
    if bounds[1] <= bounds[0]:
        raise ValueError('Radial training range collapsed')
    model.update(radius_bounds=bounds.tolist(), history=history,
                 reached_iteration_budget=len(history)==iterations,
                 training_nonpositive_radius_fraction=float(np.mean(raw <= 0)))
    return model, radius


def evaluate(model, x, years, baseline, axes=None):
    raw_radius = project_radius(model, x, years)
    radius = np.clip(raw_radius, *model['radius_bounds'])
    residual = x - predict(model, years, radius)
    center_residual = x - predict(model, years)
    denominator = float(np.sum((x - baseline) ** 2))
    result = dict(n=len(x), mse=float(np.mean(residual ** 2)),
                  r2=1-float(np.sum(residual ** 2))/denominator,
                  centerline_r2=1-float(np.sum(center_residual ** 2))/denominator,
                  radius_edge_fraction=float(np.mean(radius != raw_radius)))
    if axes is not None:
        error, total = residual @ axes, (x-baseline) @ axes
        result['projected_r2'] = 1-float(np.sum(error**2))/float(np.sum(total**2))
        result['projected_centerline_r2'] = 1-float(np.sum((center_residual@axes)**2))/float(np.sum(total**2))
    return result


def save_model(path, model, radius):
    metadata = {k:v for k,v in model.items() if k not in ('coef','origin')}
    np.savez_compressed(path, coef=model['coef'], origin=model['origin'],
                        radius=radius, metadata=json.dumps(metadata))


def load_model(path):
    with np.load(path) as data:
        return dict(json.loads(str(data['metadata'])), coef=data['coef'], origin=data['origin']), data['radius']


def spiral_axes(model, x, years):
    pair = 2 + 2 * model['periods'].index(model['radial_period'])
    trend = ((years-years.mean())[:, None] * (x-x.mean(0))).sum(0)
    directions = np.column_stack((trend, model['coef'][pair:pair+2].T))
    if np.linalg.matrix_rank(directions) < 3:
        raise ValueError('Cannot display three independent spiral directions')
    return np.linalg.qr(directions, mode='reduced')[0]


def export(output, x, years, final):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    pca = PCA(3, svd_solver='randomized', random_state=42).fit(x)
    frames = {'pca':pca.components_.T, 'spiral':spiral_axes(final, x, years)}
    center = x.mean(0)
    take = np.sort(np.random.default_rng(42).choice(len(x), 4500, replace=False))
    year_grid = np.linspace(1000, 2000, 4001)
    radius_grid = np.array(final['radius_bounds'])
    vertices = predict(final, np.repeat(year_grid, 2), np.tile(radius_grid, len(year_grid)))
    triangles = np.array([[2*i,2*i+1,2*i+2] for i in range(len(year_grid)-1)]
                        + [[2*i+1,2*i+3,2*i+2] for i in range(len(year_grid)-1)])
    payload = dict(years=years[take].tolist(), periods=final['periods'], radial_period=final['radial_period'],
                   radius_bounds=final['radius_bounds'], frames={})
    fig = plt.figure(figsize=(13,6), layout='constrained')
    for j,(name,axes) in enumerate(frames.items(),1):
        cloud = (x[take]-center)@axes
        surface = (vertices-center)@axes
        line = (predict(final, year_grid)-center)@axes
        variance = float(np.sum(((x-center)@axes)**2)/np.sum((x-center)**2))
        payload['frames'][name] = dict(cloud=np.round(cloud,6).tolist(),
            coef=np.round(final['coef']@axes,10).tolist(),
            origin=np.round((final['origin']-center)@axes,10).tolist(), variance_fraction=variance)
        ax=fig.add_subplot(1,2,j,projection='3d')
        scatter=ax.scatter(*cloud.T,c=years[take],cmap='viridis',s=2,alpha=.35,linewidths=0)
        ax.plot_trisurf(*surface.T,triangles=triangles,color='orangered',alpha=.32,linewidth=0)
        ax.plot(*line.T,color='black',lw=1.2)
        ax.set(xlabel='PC1' if name=='pca' else 'Year trend',
               ylabel='PC2' if name=='pca' else 'Oscillation 1',
               zlabel='PC3' if name=='pca' else 'Oscillation 2',
               title=f'{name}: {variance:.1%} of ambient variance')
        ax.set_box_aspect(np.ptp(cloud,axis=0))
    fig.colorbar(scatter,ax=fig.axes,label='Year',shrink=.65)
    fig.savefig(output/'spiral_ribbon_overlay.png',dpi=180)
    plt.close(fig)
    (output/'visualization.json').write_text(json.dumps(payload,separators=(',',':'),allow_nan=False))
    np.savez_compressed(output/'display_axes.npz',center=center,**frames)
    return {name:frame['variance_fraction'] for name,frame in payload['frames'].items()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--activation-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if not os.environ.get('SLURM_STEP_ID'):
        raise RuntimeError('Run through srun on a compute node')
    output=args.output;output.mkdir(parents=True,exist_ok=False);(output/'source').mkdir()
    for source in (Path(__file__),Path(curves.__file__)):
        shutil.copy2(source,output/'source'/source.name)
    x,years,templates,index=curves.load_activations(args.activation_dir)
    x=x.astype(float)
    masks,membership=curves.split_development(years,templates)
    baseline=x[masks['fit']].mean(0)
    fit_stats=curves.statistics(x,years,templates,masks['fit'],baseline)
    validation_stats=curves.statistics(x,years,templates,masks['validation'],baseline)
    # Reuse the existing training-profile search to propose a slow period.
    _,proposals=curves.select_candidates(fit_stats,validation_stats,spline_knots=())
    slow=proposals['helix']['spec']['periods'][0]
    definitions=[('single_100',[100.],100.),('single_slow',[slow],slow),
                 ('two_period',[100.,slow],100.),('four_period',[5.,10.,100.,slow],100.)]
    if abs(np.log(slow/100.))<.01:
        raise ValueError('Slow and century periods are indistinguishable')
    train_pca=PCA(3,svd_solver='randomized',random_state=42).fit(x[masks['fit']])
    candidates=[]
    for name,periods,radial in definitions:
        model,_=fit_ribbon(x[masks['fit']],years[masks['fit']],periods,radial)
        score=evaluate(model,x[masks['validation']],years[masks['validation']],baseline,train_pca.components_.T)
        row=dict(name=name,periods=periods,radial_period=radial,validation=score)
        candidates.append(row);print('Candidate:',row,flush=True)
    selected=min(candidates,key=lambda r:r['validation']['mse'])
    curves.write_csv(output/'candidates.csv',candidates)
    (output/'selection.json').write_text(json.dumps(selected,indent=2))
    model,radius=fit_ribbon(x[masks['refit']],years[masks['refit']],selected['periods'],selected['radial_period'])
    save_model(output/'evaluation_ribbon.npz',model,radius)
    scores={name:evaluate(model,x[mask],years[mask],baseline,train_pca.components_.T)
            for name,mask in masks.items() if name.endswith('evaluation')}
    print('Evaluation:',scores,flush=True)
    final,final_radius=fit_ribbon(x,years,selected['periods'],selected['radial_period'])
    save_model(output/'final_ribbon_all_development.npz',final,final_radius)
    display_variance=export(output,x,years,final)
    manifest=dict(layer=4,dimension=4096,activation_dir=str(args.activation_dir.resolve()),
        population='development only; external test not read',membership=membership,
        source_row_ids=[r['source_row_id'] for r in index],script_hash=curves.sha256(__file__),
        inputs={str(p):curves.sha256(p) for p in [args.activation_dir/'config.json',
            args.activation_dir/'progress.json',args.activation_dir/'source/records.jsonl',
            *sorted((args.activation_dir/'layer04').glob('shard_*.pt'))]},
        job_id=os.environ.get('SLURM_JOB_ID'))
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    result=dict(selection=selected,evaluation=scores,display_variance=display_variance,
        radius_bounds=final['radius_bounds'],iterations=len(final['history']),
        reached_iteration_budget=final['reached_iteration_budget'],
        training_nonpositive_radius_fraction=final['training_nonpositive_radius_fraction'],
        interpretation='Explicit generalized helical ribbon; year is supplied and radius inferred. The transverse continuation is a model assumption, not identified ground-truth geometry.')
    (output/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    (output/'COMPLETED.json').write_text(json.dumps(dict(job_id=os.environ.get('SLURM_JOB_ID'))))
    print('Completed:',output,flush=True)


if __name__=='__main__':
    main()
