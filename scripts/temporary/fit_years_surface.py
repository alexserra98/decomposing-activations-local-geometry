"""Experimental 2D principal sheet in ambient layer-4 activations.

Alternating cubic B-spline regression and closest-point projection. Labels are
used for development splits and colours only, never as surface coordinates.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil

import numpy as np
from scipy.interpolate import BSpline
from scipy.linalg import solve
from scipy.spatial import cKDTree, Delaunay
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA

from fit_years_spiral import load_activations, sha256, split_development, write_csv


def features(uv, n, derivative=None):
    """Tensor-product clamped cubic basis, including analytic partials."""
    uv = np.asarray(uv, dtype=float)
    if uv.ndim != 2 or uv.shape[1] != 2 or not np.isfinite(uv).all():
        raise ValueError('Coordinates must be finite (N, 2)')
    if np.any((uv < 0) | (uv > 1)) or n < 4:
        raise ValueError('Require n >= 4 and coordinates in [0,1]^2')
    knots = np.r_[np.zeros(3), np.linspace(0, 1, n - 2), np.ones(3)]
    spline = BSpline(knots, np.eye(n), 3, extrapolate=False)
    a = spline(uv[:, 0], nu=int(derivative == 0))
    b = spline(uv[:, 1], nu=int(derivative == 1))
    return csr_matrix((a[:, :, None] * b[:, None, :]).reshape(len(uv), n * n))


def penalty(n):
    """Grid-scaled bending penalty and a small first-difference stabilizer."""
    first = np.diff(np.eye(n), axis=0) * (n - 1)
    second = np.diff(np.eye(n), n=2, axis=0) * (n - 1) ** 2
    d1 = np.vstack((np.kron(first, np.eye(n)), np.kron(np.eye(n), first)))
    d2 = np.vstack((np.kron(second, np.eye(n)), np.kron(np.eye(n), second)))
    return (d2.T @ d2 + .01 * d1.T @ d1) / (n * n)


def regress(x, uv, n, smoothing, origin):
    f = features(uv, n)
    gram = (f.T @ f).toarray() + len(x) * smoothing * penalty(n)
    coef = solve(gram, f.T @ (x - origin), assume_a='pos')
    return dict(n=n, smoothing=smoothing, origin=origin, coef=coef)


def predict(model, uv, derivative=None):
    result = features(uv, model['n'], derivative) @ model['coef']
    return result if derivative is not None else result + model['origin']


def compressed(model, x):
    """Exact projection arithmetic in the coefficient span; no PCA truncation."""
    q, _ = np.linalg.qr(model['coef'].T, mode='reduced')
    return (x - model['origin']) @ q, model['coef'] @ q


def optimize_coordinates(target, coef, n, uv, steps=20):
    """Bounded Gauss-Newton, accepting only rowwise error reductions."""
    uv = uv.copy()
    for _ in range(steps):
        residual = features(uv, n) @ coef - target
        ju, jv = (features(uv, n, j) @ coef for j in (0, 1))
        a = np.einsum('ij,ij->i', ju, ju) + 1e-10
        b = np.einsum('ij,ij->i', ju, jv)
        c = np.einsum('ij,ij->i', jv, jv) + 1e-10
        ru = np.einsum('ij,ij->i', ju, residual)
        rv = np.einsum('ij,ij->i', jv, residual)
        det = np.maximum(a * c - b * b, 1e-20)
        delta = np.column_stack(((c * ru - b * rv) / det, (a * rv - b * ru) / det))
        delta /= np.maximum(1, np.linalg.norm(delta, axis=1) / .15)[:, None]
        error = np.einsum('ij,ij->i', residual, residual)
        pending = np.ones(len(uv), dtype=bool)
        movement = np.zeros_like(uv)
        for factor in (1., .5, .25, .125, .0625):
            ids = np.flatnonzero(pending)
            if not len(ids):
                break
            trial = np.clip(uv[ids] - factor * delta[ids], 0, 1)
            r = features(trial, n) @ coef - target[ids]
            accepted = np.einsum('ij,ij->i', r, r) < error[ids]
            take = ids[accepted]
            movement[take] = trial[accepted] - uv[take]
            uv[take] = trial[accepted]
            pending[take] = False
        if np.max(np.linalg.norm(movement, axis=1), initial=0) < 1e-6:
            break
    r = features(uv, n) @ coef - target
    return uv, np.einsum('ij,ij->i', r, r)


def project(model, x, initial=None, steps=25, starts=3):
    """Approximate ambient nearest point, with separated coarse-grid starts."""
    target, coef = compressed(model, x)
    n = model['n']
    if initial is not None:
        return optimize_coordinates(target, coef, n, initial, steps)[0]
    grid = np.stack(np.meshgrid(np.linspace(0, 1, 25), np.linspace(0, 1, 25)), -1).reshape(-1, 2)
    distances = cdist(target, features(grid, n) @ coef, metric='sqeuclidean')
    best, best_error = None, np.full(len(x), np.inf)
    for _ in range(starts):
        nearest = distances.argmin(1)
        uv, error = optimize_coordinates(target, coef, n, grid[nearest], steps)
        better = error < best_error
        if best is None:
            best = uv.copy()
        best[better], best_error[better] = uv[better], error[better]
        separated = np.sum((grid[None, :, :] - uv[:, None, :]) ** 2, axis=2) > .15 ** 2
        distances[~separated] = np.inf
    return best


def fit_surface(x, n, smoothing, initialization=(0, 1), iterations=18):
    origin = x.mean(0)
    pca = PCA(n_components=3, svd_solver='randomized', random_state=42).fit(x)
    z = pca.transform(x)[:, list(initialization)]
    low, high = z.min(0), z.max(0)
    uv = .05 + .9 * (z - low) / (high - low)
    history = []
    for iteration in range(iterations):
        model = regress(x, uv, n, smoothing, origin)
        uv = project(model, x, initial=uv, steps=10)
        residual = x - predict(model, uv)
        mse = float(np.mean(residual ** 2))
        objective = mse * x.shape[1] + smoothing * float(np.sum(model['coef'] * (penalty(n) @ model['coef'])))
        history.append(dict(iteration=iteration, mse=mse, objective=objective))
        if iteration >= 5 and abs(history[-2]['objective'] - objective) / max(objective, 1e-15) < 1e-5:
            break
    model = regress(x, uv, n, smoothing, origin)
    model['initialization'] = list(initialization)
    model['history'] = history
    return model, uv


def score(model, x, baseline):
    uv = project(model, x)
    residual = x - predict(model, uv)
    sse = float(np.sum(residual ** 2))
    return dict(n=len(x), mse=sse / x.size,
                r2=1 - sse / float(np.sum((x - baseline) ** 2)),
                boundary_fraction=float(np.mean(np.min(np.c_[uv, 1 - uv], axis=1) < .005)))


def save_surface(path, model, uv):
    metadata = {k: v for k, v in model.items() if k not in ('origin', 'coef')}
    np.savez_compressed(path, coef=model['coef'], origin=model['origin'], uv=uv,
                        metadata=json.dumps(metadata))


def load_surface(path):
    with np.load(path) as data:
        return dict(json.loads(str(data['metadata'])), coef=data['coef'], origin=data['origin']), data['uv']


def support(uv, training_uv):
    tree = cKDTree(training_uv)
    distance, nearest = tree.query(uv)
    local_radius = tree.query(training_uv, k=11)[0][:, -1]
    ratio = distance / np.maximum(local_radius[nearest], 1e-8)
    inside = Delaunay(training_uv).find_simplex(uv) >= 0
    return inside & (ratio <= 1.5), ratio


def tangent(model, uv):
    jacobian = np.stack([predict(model, uv, j) for j in (0, 1)], axis=-1)
    u, singular, _ = np.linalg.svd(jacobian, full_matrices=False)
    return u, singular


def surface_mesh(training_uv):
    """Sample both the populated coordinates and the full rectangular chart."""
    regular = np.stack(np.meshgrid(np.linspace(0, 1, 25), np.linspace(0, 1, 25)), -1).reshape(-1, 2)
    take = np.random.default_rng(42).choice(len(training_uv), min(2000, len(training_uv)), replace=False)
    grid = np.unique(np.vstack((regular, training_uv[take])), axis=0)
    triangles = Delaunay(grid).simplices
    centers = grid[triangles].mean(1)
    midpoints = (grid[triangles] + np.roll(grid[triangles], 1, axis=1)) / 2
    supported, _ = support(np.vstack((centers, midpoints.reshape(-1, 2))), training_uv)
    reliable = supported[:len(triangles)] & supported[len(triangles):].reshape(-1, 3).all(1)
    return grid, triangles, reliable


def isotropic_pcs(model):
    """Exact covariance PCs for W W.T + b I via thin SVD, avoiding D x D matrices."""
    import torch
    with torch.no_grad():
        w = model._W().detach().cpu().double()
        psi = model._psi().detach().cpu().double()
        if not torch.allclose(psi, psi[:, :1].expand_as(psi), rtol=1e-12, atol=0):
            raise ValueError('This experiment requires saved isotropic HDDC models')
        u, s, _ = torch.linalg.svd(w, full_matrices=False)
        return s[:, :3].square() + psi[:, :1], u[:, :, :2]


def compare_gaussians(output, root, model, training_uv):
    import torch
    from dalg.models.adaptive_q.mfa_hddc import load_mfa_hddc
    from dalg.evaluation.subspaces import (subspace_alignment, leading_subspace_is_identifiable,
                                         PC_RELATIVE_EIGENGAP_THRESHOLD)
    rows, summaries, bundles, fingerprints = [], [], {}, {}
    for run in sorted((root / 'sae_manifold_years_neutral_hddc_em').glob('*__l04__*')):
        spec = json.loads((run / 'run_spec.json').read_text())
        saved = run / 'mfa_model.pt'
        fingerprints[str(saved)] = sha256(saved)
        gaussian = load_mfa_hddc(saved, map_location='cpu')
        means = gaussian.mu.detach().cpu().double().numpy()
        uv = project(model, means, steps=50, starts=5)
        nearest = predict(model, uv)
        bases, singular = tangent(model, uv)
        supported, support_ratio = support(uv, training_uv)
        values, pcs = isotropic_pcs(gaussian)
        ranks = gaussian.rank_mask.sum(1).detach().cpu().numpy().astype(int)
        weights = torch.softmax(gaussian.pi_logits.detach().cpu(), dim=0).numpy()
        run_rows = []
        threshold = spec['identity']['training_args']['surgery_threshold']
        for k in range(gaussian.K):
            boundary = bool(np.min(np.r_[uv[k], 1 - uv[k]]) < .005)
            identifiable = leading_subspace_is_identifiable(values[k], 2, gaussian.D,
                                                           PC_RELATIVE_EIGENGAP_THRESHOLD)
            reasons = []
            if weights[k] <= 0:
                reasons.append('inactive')
            if ranks[k] < 2 or not identifiable:
                reasons.append('PC1-PC2 boundary not identifiable')
            if singular[k, 1] <= 1e-4 * singular[k, 0]:
                reasons.append('surface Jacobian nearly rank deficient')
            if boundary:
                reasons.append('surface boundary')
            if not supported[k]:
                reasons.append('outside sampled surface support')
            overlap, worst = (None, None)
            angles = [None, None]
            if not reasons:
                reference = torch.from_numpy(bases[k])
                overlap, worst = subspace_alignment(reference, pcs[k])
                cosines = torch.linalg.svdvals(reference.T @ pcs[k]).clamp(0, 1).numpy()
                angles = np.degrees(np.arccos(cosines)).tolist()
            row = dict(run=run.name, K=gaussian.K, threshold=threshold, component=k,
                       rank=int(ranks[k]), weight=float(weights[k]), u=float(uv[k, 0]), v=float(uv[k, 1]),
                       distance=float(np.linalg.norm(means[k] - nearest[k])),
                       support_ratio=float(support_ratio[k]), jacobian_ratio=float(singular[k, 1] / singular[k, 0]),
                       valid=not reasons, reason='; '.join(reasons), overlap=overlap,
                       worst_direction_cosine=worst, angle_min_deg=angles[0], angle_max_deg=angles[1])
            rows.append(row)
            run_rows.append(row)
        valid = [r for r in run_rows if r['valid']]
        summaries.append(dict(run=run.name, K=gaussian.K, threshold=threshold, valid=len(valid), total=gaussian.K,
                              mean_overlap=float(np.mean([r['overlap'] for r in valid])) if valid else None,
                              median_worst_angle=float(np.median([r['angle_max_deg'] for r in valid])) if valid else None))
        bundles[run.name] = dict(means=means, projected=nearest, tangent=bases,
                                 pcs=pcs.numpy(), uv=uv, rows=run_rows)
        print('Gaussian comparison:', summaries[-1], flush=True)
    if len(summaries) != 8:
        raise ValueError(f'Expected eight current development layer-4 HDDC runs, got {len(summaries)}')
    write_csv(output / 'gaussian_alignment.csv', rows)
    return summaries, bundles, fingerprints


def figures(output, x, years, model, uv, comparisons, bundles):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation
    pca = PCA(n_components=3, svd_solver='randomized', random_state=42).fit(x)
    cloud = pca.transform(x)
    grid, all_triangles, reliable = surface_mesh(uv)
    mesh = pca.transform(predict(model, grid))
    tri = all_triangles[reliable]
    unsupported = all_triangles[~reliable]
    fig = plt.figure(figsize=(15, 4.8), layout='constrained')
    rng = np.random.default_rng(42)
    take = np.sort(rng.choice(len(x), 4500, replace=False))
    for j, axes in enumerate(((0, 1), (0, 2))):
        ax = fig.add_subplot(1, 3, j + 1)
        ax.triplot(Triangulation(mesh[:, axes[0]], mesh[:, axes[1]], triangles=unsupported), color='gray', alpha=.14, lw=.35)
        ax.triplot(Triangulation(mesh[:, axes[0]], mesh[:, axes[1]], triangles=tri), color='orangered', alpha=.35, lw=.45)
        sc = ax.scatter(cloud[take, axes[0]], cloud[take, axes[1]], c=years[take], cmap='viridis', s=2, alpha=.5)
        ax.set(xlabel=f'Activation PC{axes[0]+1}', ylabel=f'Activation PC{axes[1]+1}', aspect='equal', title='Surface mesh + activation points')
    ax = fig.add_subplot(1, 3, 3, projection='3d')
    ax.plot_trisurf(*mesh.T, triangles=unsupported, color='gray', alpha=.08, linewidth=.1)
    ax.plot_trisurf(*mesh.T, triangles=tri, color='orangered', alpha=.3, linewidth=.1)
    ax.scatter(*cloud[take].T, c=years[take], cmap='viridis', s=1, alpha=.3)
    ax.set(xlabel='PC1', ylabel='PC2', zlabel='PC3', title='Same ambient surface, projected to 3D')
    ax.set_box_aspect(np.ptp(cloud, axis=0))
    fig.colorbar(sc, ax=fig.axes, label='Year', shrink=.7)
    fig.suptitle('Orange: sampled support; gray: unsupported continuation of the fitted sheet')
    fig.savefig(output / 'surface_overlay.png', dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4), layout='constrained')
    labels = [f"K={r['K']}, t={r['threshold']:g}\n{r['valid']}/{r['total']} valid" for r in comparisons]
    ax.bar(np.arange(len(labels)), [r['mean_overlap'] or 0 for r in comparisons], color='steelblue')
    ax.set(xticks=np.arange(len(labels)), xticklabels=labels, ylim=(0, 1), ylabel='Mean squared principal-angle cosine', title='Surface tangent vs Gaussian PC1–PC2 in 4096D')
    ax.tick_params(axis='x', labelrotation=40)
    fig.savefig(output / 'gaussian_alignment.png', dpi=180)
    plt.close(fig)
    payload = dict(cloud=np.round(cloud[take], 5).tolist(), years=years[take].tolist(),
                   mesh=np.round(mesh, 5).tolist(), triangles=tri.tolist(), unsupported_triangles=unsupported.tolist(),
                   explained=pca.explained_variance_ratio_.tolist(), runs={})
    scale = float(np.median(np.linalg.norm(cloud - cloud.mean(0), axis=1)) * .18)
    for name, data in bundles.items():
        payload['runs'][name] = dict(means=pca.transform(data['means']).tolist(),
            projected=pca.transform(data['projected']).tolist(),
            tangent=(data['tangent'].transpose(0, 2, 1) @ pca.components_.T * scale).tolist(),
            pcs=(data['pcs'].transpose(0, 2, 1) @ pca.components_.T * scale).tolist(), rows=data['rows'])
    (output / 'visualization.json').write_text(json.dumps(payload, separators=(',', ':')))
    np.savez_compressed(output / 'surface_geometry.npz', uv=uv, grid=grid, triangles=tri, unsupported_triangles=unsupported,
                        surface=predict(model, grid), pca_basis=pca.components_.T, pca_mean=pca.mean_)


def finish(output, root, x, years, final, uv, selected, scores, plane_scores):
    comparisons, bundles, hashes = compare_gaussians(output, root, final, uv)
    figures(output, x, years, final, uv, comparisons, bundles)
    results = dict(selection=selected, evaluation=scores, pca2_evaluation=plane_scores,
                   gaussians=comparisons, gaussian_hashes=hashes,
                   comparison_source_hash=sha256(__file__),
                   note='Ambient reconstruction after inferring two coordinates from each query activation; not year prediction or ground-truth tangents.')
    (output / 'results.json').write_text(json.dumps(results, indent=2))
    (output / 'COMPLETED.json').write_text(json.dumps(dict(slurm_job=os.environ.get('SLURM_JOB_ID'), complete=True)))
    print('Completed:', output, flush=True)


def resume_comparison(output, root):
    manifest = json.loads((output / 'manifest.json').read_text())
    if str(root.resolve()) != manifest['activation_dir']:
        raise ValueError('Resume requires the same activation root')
    assert all(sha256(path) == digest for path, digest in manifest['input_hashes'].items())
    x, years, templates, index = load_activations(root)
    x = x.astype(float)
    masks, membership = split_development(years, templates)
    assert membership == manifest['split']
    assert [r['source_row_id'] for r in index] == manifest['source_row_ids']
    baseline = x[masks['fit']].mean(0)
    evaluated, _ = load_surface(output / 'evaluation_surface.npz')
    final, uv = load_surface(output / 'final_surface_all_development.npz')
    selected = json.loads((output / 'selection.json').read_text())
    scores = {name: score(evaluated, x[masks[name]], baseline) for name in
              ('year_evaluation', 'template_evaluation', 'year_template_evaluation')}
    plane = PCA(n_components=2, svd_solver='randomized', random_state=42).fit(x[masks['refit']])
    plane_scores = {}
    for name in scores:
        data = x[masks[name]]
        error = data - plane.inverse_transform(plane.transform(data))
        plane_scores[name] = dict(mse=float(np.mean(error ** 2)),
                                 r2=1 - float(np.sum(error ** 2)) / float(np.sum((data - baseline) ** 2)))
    shutil.copy2(__file__, output / 'source/fit_years_surface_comparison.py')
    finish(output, root, x, years, final, uv, selected, scores, plane_scores)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--activation-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume-comparison', action='store_true',
                        help='Reuse completed surface fits and rebuild Gaussian comparisons/figures')
    args = parser.parse_args()
    if 'SLURM_STEP_ID' not in os.environ:
        raise RuntimeError('Run numerical work through srun')
    output = args.output
    if args.resume_comparison:
        resume_comparison(output, args.activation_dir)
        return
    output.mkdir(parents=True, exist_ok=False)
    (output / 'source').mkdir()
    for file in (Path(__file__), Path(__file__).with_name('fit_years_spiral.py')):
        shutil.copy2(file, output / 'source' / file.name)
    x, years, templates, index = load_activations(args.activation_dir)
    x = x.astype(np.float64)
    masks, membership = split_development(years, templates)
    baseline = x[masks['fit']].mean(0)
    manifest = dict(activation_dir=str(args.activation_dir.resolve()), layer=4, dimension=x.shape[1],
                    split=membership, source_row_ids=[r['source_row_id'] for r in index],
                    input_hashes={str(p): sha256(p) for p in sorted((args.activation_dir / 'layer04').glob('*.pt'))},
                    source_hash=sha256(__file__))
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    candidates = []
    for n in (6, 10):
        for smoothing in (1e-4, 1e-3):
            for initialization in ((0, 1), (0, 2)):
                fitted, _ = fit_surface(x[masks['fit']], n, smoothing, initialization)
                validation = score(fitted, x[masks['validation']], baseline)
                row = dict(n=n, smoothing=smoothing, initialization=list(initialization), validation=validation)
                candidates.append(row)
                print('Candidate:', row, flush=True)
                write_csv(output / 'candidates.csv', candidates)
    selected = min(candidates, key=lambda r: r['validation']['mse'])
    (output / 'selection.json').write_text(json.dumps(selected, indent=2))
    setting = {k: selected[k] for k in ('n', 'smoothing', 'initialization')}
    evaluated, evaluated_uv = fit_surface(x[masks['refit']], **setting)
    save_surface(output / 'evaluation_surface.npz', evaluated, evaluated_uv)
    scores = {name: score(evaluated, x[masks[name]], baseline) for name in
              ('year_evaluation', 'template_evaluation', 'year_template_evaluation')}
    plane = PCA(n_components=2, svd_solver='randomized', random_state=42).fit(x[masks['refit']])
    plane_scores = {}
    for name in scores:
        data = x[masks[name]]
        error = data - plane.inverse_transform(plane.transform(data))
        plane_scores[name] = dict(mse=float(np.mean(error ** 2)),
                                 r2=1 - float(np.sum(error ** 2)) / float(np.sum((data - baseline) ** 2)))
    print('Evaluation:', scores, 'PCA2 baseline:', plane_scores, flush=True)
    final, uv = fit_surface(x, **setting)
    save_surface(output / 'final_surface_all_development.npz', final, uv)
    finish(output, args.activation_dir, x, years, final, uv, selected, scores, plane_scores)


if __name__ == '__main__':
    main()
