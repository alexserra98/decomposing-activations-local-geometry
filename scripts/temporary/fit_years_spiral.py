"""Layer-4 years curves in ambient coordinates; no pipeline/model changes.

Compare a line, one-period helix, generalized helix and cubic regression spline.
Only development shards are read. See docs/experiments/years-spiral-parameterization.md.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import minimize_scalar

REPO = Path(__file__).resolve().parents[2]
YEAR_GRID = np.arange(1000, 2001)
TEMPLATES = np.arange(20)


def basis(years, spec, derivative=False):
    """Prescribed year domain; spline knots and Fourier phases are serializable."""
    years = np.atleast_1d(years).astype(np.float64)
    if np.any((years < 1000) | (years > 2000)):
        raise ValueError("Years must lie in the fitted 1000–2000 domain")
    t = years - 1500
    columns = [np.zeros(len(t)) if derivative else np.ones(len(t))]
    if spec['kind'] == 'spline':
        knots = np.linspace(1000, 2000, spec['knots'])
        augmented = np.r_[np.repeat(knots[0], 3), knots, np.repeat(knots[-1], 3)]
        n_basis = len(augmented) - 4
        splines = BSpline(augmented, np.eye(n_basis), 3, extrapolate=False)
        # Drop one partition-of-unity column because the intercept is explicit.
        return np.column_stack((columns[0], splines(years, nu=int(derivative))[:, :-1]))
    if spec['kind'] != 'constant':
        columns.append(np.full(len(t), 1 / 500) if derivative else t / 500)
    for period in spec.get('periods', []):
        omega = 2 * np.pi / period
        phase = omega * t
        columns.extend((-omega * np.sin(phase), omega * np.cos(phase)) if derivative
                       else (np.cos(phase), np.sin(phase)))
    return np.column_stack(columns)


def split_development(years, templates, seed=42):
    """Disjoint year groups and withheld templates, entirely inside development."""
    year_order = np.random.default_rng(seed).permutation(np.unique(years))
    n = len(year_order)
    fit_years, validation_years, evaluation_years = np.split(
        year_order, [int(.6 * n), int(.8 * n)])
    template_order = np.random.default_rng(seed + 1).permutation(np.unique(templates))
    heldout_templates = np.sort(template_order[:4])
    seen_templates = np.sort(template_order[4:])
    seen = np.isin(templates, seen_templates)
    fit = np.isin(years, fit_years)
    validation = np.isin(years, validation_years)
    evaluation = np.isin(years, evaluation_years)
    masks = dict(fit=fit & seen, validation=validation & seen,
                 refit=(fit | validation) & seen, year_evaluation=evaluation & seen,
                 template_evaluation=(fit | validation) & ~seen,
                 year_template_evaluation=evaluation & ~seen,
                 all_development=np.ones(len(years), dtype=bool))
    membership = dict(fit_years=sorted(fit_years.tolist()),
                      validation_years=sorted(validation_years.tolist()),
                      evaluation_years=sorted(evaluation_years.tolist()),
                      seen_templates=seen_templates.tolist(),
                      heldout_templates=heldout_templates.tolist())
    return masks, membership


def statistics(x, years, templates, mask, origin):
    """Sufficient statistics retain the exact 4096D squared-error objective."""
    positions = np.flatnonzero(mask)
    if not len(positions):
        raise ValueError("Empty experiment split")
    sums = np.zeros((len(YEAR_GRID), x.shape[1]), dtype=np.float64)
    template_sums = np.zeros((len(TEMPLATES), x.shape[1]), dtype=np.float64)
    counts = np.zeros((len(YEAR_GRID), len(TEMPLATES)), dtype=np.float64)
    energy = 0.
    for start in range(0, len(positions), 512):
        take = positions[start:start + 512]
        values = x[take].astype(np.float64) - origin
        yi, si = years[take] - 1000, templates[take]
        np.add.at(sums, yi, values)
        np.add.at(template_sums, si, values)
        np.add.at(counts, (yi, si), 1)
        energy += float(np.sum(values * values))
    return dict(sums=sums, template_sums=template_sums, counts=counts,
                energy=energy, n=len(positions), dimension=x.shape[1], origin=origin)


def normal_equations(stats, spec):
    f = basis(YEAR_GRID, spec)
    counts = stats['counts']
    year_counts, template_counts = counts.sum(1), counts.sum(0)
    cross = f.T @ counts
    gram = np.block([[f.T @ (year_counts[:, None] * f), cross],
                     [cross.T, np.diag(template_counts)]])
    response = np.vstack((f.T @ stats['sums'], stats['template_sums']))
    return gram, response, f.shape[1]


def fit_curve(stats, spec):
    gram, response, p = normal_equations(stats, spec)
    seen = np.flatnonzero(stats['counts'].sum(0))
    # Reference coding for fitting, then zero-sum offsets over seen templates.
    selected = np.r_[np.arange(p), p + seen[:-1]]
    diagonal = np.sqrt(np.maximum(0, gram.diagonal()[selected]))
    active = diagonal > 1e-10 * diagonal.max()
    selected, scale = selected[active], diagonal[active]
    normalized = gram[np.ix_(selected, selected)] / scale[:, None] / scale[None, :]
    solution, _, rank, _ = np.linalg.lstsq(normalized, response[selected] / scale[:, None],
                                          rcond=1e-11)
    coef = np.zeros_like(response)
    coef[selected] = solution / scale[:, None]
    offset_mean = coef[p + seen].mean(0)
    coef[0] += offset_mean
    coef[p + seen] -= offset_mean
    return dict(spec=spec, coef=coef, year_features=p, origin=stats['origin'],
                rank=int(rank), seen_templates=seen.tolist())


def predict(model, years, templates=None, derivative=False):
    p = model['year_features']
    result = basis(years, model['spec'], derivative) @ model['coef'][:p]
    if not derivative:
        result += model['origin']
        if templates is not None:
            result += model['coef'][p + np.asarray(templates, dtype=int)]
    return result


def nonnegative(value, scale):
    if value < -1e-8 * max(1., scale):
        raise ValueError(f"Negative squared error beyond roundoff: {value}")
    return max(0., float(value))


def evaluate(stats, model):
    gram, response, p = normal_equations(stats, model['spec'])
    coef = model['coef']
    sse = nonnegative(stats['energy'] - 2 * np.sum(coef * response)
                      + np.sum(coef * (gram @ coef)), stats['energy'])
    f = basis(YEAR_GRID, model['spec'])
    year_counts = stats['counts'].sum(1)
    residual_sum = (stats['sums'] - year_counts[:, None] * (f @ coef[:p])
                    - stats['counts'] @ coef[p:])
    present = year_counts > 0
    centroid_sse = float(np.sum(residual_sum[present] ** 2 / year_counts[present, None]))
    denom = stats['n'] * stats['dimension']
    return dict(n=stats['n'], mse=sse / denom,
                r2=1 - sse / stats['energy'],
                year_centroid_mse=centroid_sse / denom,
                within_year_mse=nonnegative(sse - centroid_sse, stats['energy']) / denom)


def select_candidates(fit_stats, validation_stats, periods=(), spline_knots=(6, 10, 18, 34, 66, 130, 258)):
    """Tune periods/complexity on validation years; evaluation groups are absent."""
    rows, best = [], {}

    def consider(family, spec):
        model = fit_curve(fit_stats, spec)
        training, validation = evaluate(fit_stats, model), evaluate(validation_stats, model)
        row = dict(family=family, spec=spec, fit_mse=training['mse'],
                   validation_mse=validation['mse'], effective_rank=model['rank'])
        rows.append(row)
        if family not in best or validation['mse'] < best[family]['validation_mse']:
            best[family] = row
        return training['mse']

    consider('constant', dict(kind='constant'))
    consider('line', dict(kind='line'))
    grid = np.unique(np.r_[2., 5., 10., 20., 50., np.geomspace(100, 10000, 81), periods])
    profile = []
    for period in grid:
        value = consider('helix', dict(kind='fourier', periods=[float(period)]))
        profile.append(value)
    # Refine five training-profile minima, then assess each on validation years.
    for index in np.argsort(profile)[:5]:
        if index == 0 or index == len(grid) - 1:
            continue
        def objective(log_period):
            model = fit_curve(fit_stats, dict(kind='fourier', periods=[float(np.exp(log_period))]))
            return evaluate(fit_stats, model)['mse']
        result = minimize_scalar(objective, bounds=np.log(grid[[index - 1, index + 1]]),
                                 method='bounded', options=dict(xatol=1e-6))
        if not result.success:
            raise RuntimeError(result.message)
        consider('helix', dict(kind='fourier', periods=[float(np.exp(result.x))]))
    single = best['helix']['spec']['periods'][0]
    pool = [2., 5., 10., 100.]
    if all(abs(np.log(single / t)) > .01 for t in pool):
        pool.append(single)
    for k in range(2, min(4, len(pool)) + 1):
        for chosen in itertools.combinations(pool, k):
            consider('generalized_helix', dict(kind='fourier', periods=sorted(chosen)))
    for knots in spline_knots:
        consider('spline', dict(kind='spline', knots=int(knots)))
    return rows, best


def curve_geometry(model):
    """Exact PCA of the sampled fitted year curve, using its small linear span."""
    values = predict(model, YEAR_GRID)
    center = values.mean(0)
    if np.max(np.ptp(values, axis=0)) < 1e-10:
        return dict(center=center, basis=np.zeros((values.shape[1], 3)),
                    coordinates=np.zeros((len(values), 3)), spectrum=np.zeros(1),
                    top3_fraction=None, dimension95=0, dimension99=0)
    q, _ = np.linalg.qr(model['coef'][:model['year_features']].T, mode='reduced')
    small = (values - center) @ q
    _, singular, vt = np.linalg.svd(small, full_matrices=False)
    spectrum = singular ** 2
    cumulative = np.cumsum(spectrum) / spectrum.sum()
    axes = np.zeros((values.shape[1], 3))
    axes[:, :min(3, len(vt))] = q @ vt[:3].T
    coordinates = (values - center) @ axes
    return dict(center=center, basis=axes, coordinates=coordinates,
                spectrum=spectrum, top3_fraction=float(cumulative[min(2, len(cumulative) - 1)]),
                dimension95=int(np.searchsorted(cumulative, .95) + 1),
                dimension99=int(np.searchsorted(cumulative, .99) + 1))


def save_model(path, model):
    np.savez_compressed(path, coef=model['coef'], origin=model['origin'],
                        metadata=json.dumps({key: value for key, value in model.items()
                                             if key not in ('coef', 'origin')}))


def load_model(path):
    with np.load(path, allow_pickle=False) as saved:
        return dict(json.loads(str(saved['metadata'])), coef=saved['coef'], origin=saved['origin'])


def truncate_curve(model, geometry):
    """Project only the shared year trajectory; preserve all template offsets."""
    p, axes = model['year_features'], geometry['basis']
    coef = model['coef'].copy()
    coef[:p] = (coef[:p] @ axes) @ axes.T
    center = geometry['center'] - model['origin']
    coef[0] += center - (center @ axes) @ axes.T
    return dict(model, coef=coef)


def load_activations(root):
    import torch
    from dalg.data.prompt_activations import load_prompt_activation_index
    from dalg.data.shard_activations import ActivationBatchDataset
    config = json.loads((root / 'config.json').read_text())
    if config['num_rows'] != 16016 or config['d_model'] != 4096:
        raise ValueError("Require the 16,016-row development extraction, not the full or test population")
    index = load_prompt_activation_index(root, layer=4)
    if {r['subset'] for r in index} != {'years'}:
        raise ValueError("Require standalone years development activations")
    stream = ActivationBatchDataset(root, layer=4, batch_size=1024, drop_prefix=0,
                                    dtype=torch.float32, return_metadata=True,
                                    shuffle_shards=False, shuffle_within_shard=False)
    values, ids = [], []
    for x, global_rows, token_slots in stream:
        assert (token_slots == 0).all()
        values.append(x.numpy())
        ids.extend(global_rows.tolist())
    if ids != [r['global_row'] for r in index]:
        raise ValueError("Activation/label row alignment mismatch")
    x = np.concatenate(values)
    years = np.array([r['labels']['year'] for r in index], dtype=int)
    templates = np.array([r['labels']['template_id'] for r in index], dtype=int)
    if (not np.isfinite(x).all() or not np.array_equal(np.unique(years), YEAR_GRID)
            or not np.array_equal(np.unique(templates), TEMPLATES)):
        raise ValueError("Unexpected years/template coverage or nonfinite activations")
    return x, years, templates, index


def sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write_csv(path, rows):
    with path.open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({k: json.dumps(v) if isinstance(v, (dict, list)) else v
                         for k, v in row.items()} for row in rows)


def plots(output, x, years, templates, models, scores, final):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    families = list(models)
    geometry = {family: curve_geometry(model) for family, model in models.items()}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')
    for protocol in ('year_evaluation', 'template_evaluation', 'year_template_evaluation'):
        axes[0].plot(families, [scores[f][protocol]['mse'] for f in families],
                     'o-', label=protocol.replace('_evaluation', '').replace('_', ' + '))
    axes[0].set(ylabel='Ambient reconstruction MSE per coordinate', title='Untuned evaluation groups')
    axes[0].tick_params(axis='x', rotation=25)
    axes[0].legend()
    for family, g in geometry.items():
        if not g['spectrum'].sum():
            continue
        axes[1].plot(np.arange(1, len(g['spectrum']) + 1),
                     np.cumsum(g['spectrum']) / g['spectrum'].sum(),
                     label=f'{family.replace("_", " ")} (95%: {g["dimension95"]})')
    axes[1].set(xlabel='Dimensions of fitted year curve', ylabel='Cumulative variance',
                title='Curve sampled uniformly over years',
                xlim=(1, max(len(g['spectrum']) for g in geometry.values())), ylim=(0, 1.02))
    axes[1].axhline(.95, color='gray', lw=.8)
    axes[1].legend()
    fig.savefig(output / 'comparison.png', dpi=180)
    plt.close(fig)
    g = curve_geometry(final)
    p = final['year_features']
    cloud = (x - final['coef'][p + templates] - g['center']) @ g['basis']
    fig = plt.figure(figsize=(12, 5.5), layout='constrained')
    for column in (1, 2):
        ax = fig.add_subplot(1, 2, column, projection='3d')
        if column == 1:
            scatter = ax.scatter(*cloud.T, c=years, cmap='viridis', s=1, alpha=.35, linewidths=0)
        else:
            scatter = ax.scatter(*g['coordinates'].T, c=YEAR_GRID, cmap='viridis', s=5, linewidths=0)
        ax.plot(*g['coordinates'].T, color='black', lw=.7, alpha=.6)
        ax.set(xlabel='Curve PC1', ylabel='Curve PC2', zlabel='Curve PC3',
               title='Template offsets removed' if column == 1 else 'Fitted year trajectory')
        ax.set_box_aspect(np.ptp(g['coordinates'], axis=0).clip(1e-8))
    fig.colorbar(scatter, ax=fig.axes, label='Year', shrink=.7)
    retained = g['top3_fraction']
    retained_text = 'constant' if retained is None else f'{retained:.1%}'
    fig.suptitle(f'Layer 4 · {final["spec"]} · first three curve PCs retain {retained_text}')
    fig.savefig(output / 'fitted_curve.png', dpi=180, bbox_inches='tight')
    plt.close(fig)
    np.savez_compressed(output / 'curve_geometry.npz', years=YEAR_GRID, **g,
                        tangent=predict(final, YEAR_GRID, derivative=True))
    return {family: {key: g[key] for key in ('top3_fraction', 'dimension95', 'dimension99')}
            for family, g in geometry.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--activation-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if not os.environ.get('SLURM_STEP_ID'):
        raise RuntimeError("Run numerical work through srun on a compute node")
    args.output.mkdir(parents=True, exist_ok=False)
    root, output = args.activation_dir.resolve(strict=True), args.output
    (output / 'source').mkdir()
    shutil.copyfile(__file__, output / 'source/fit_years_spiral.py')
    x, years, templates, index = load_activations(root)
    masks, membership = split_development(years, templates, args.seed)
    origin = x[masks['fit']].mean(0, dtype=np.float64)
    stats = {name: statistics(x, years, templates, mask, origin) for name, mask in masks.items()}
    manifest = dict(format='dalg_years_layer04_parameterization_v1', layer=4, dimension=x.shape[1],
                    activation_dir=str(root), population='development_only', seed=args.seed,
                    node=os.uname().nodename, job_id=os.environ.get('SLURM_JOB_ID'),
                    source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    script_sha256=sha256(__file__), membership=membership,
                    splits={name: dict(count=int(mask.sum()), source_row_ids=[index[i]['source_row_id']
                            for i in np.flatnonzero(mask)]) for name, mask in masks.items()},
                    inputs={str(p.relative_to(root)): sha256(p) for p in
                            [root / 'config.json', root / 'progress.json',
                             root / 'source/records.jsonl', *sorted((root / 'layer04').glob('shard_*.pt'))]})
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print('Loaded', x.shape, 'split sizes:', {k: int(v.sum()) for k, v in masks.items()}, flush=True)
    candidates, best = select_candidates(stats['fit'], stats['validation'])
    write_csv(output / 'candidates.csv', candidates)
    # Freeze all family hyperparameters before any evaluation result is read.
    winner = min(best, key=lambda family: best[family]['validation_mse'])
    (output / 'selection.json').write_text(json.dumps(dict(winning_family=winner, families=best), indent=2))
    models, scores = {}, {}
    for family, chosen in best.items():
        model = fit_curve(stats['refit'], chosen['spec'])
        models[family] = model
        save_model(output / f'{family}_evaluation_model.npz', model)
        scores[family] = {name: evaluate(stats[name], model) for name in
                         ('refit', 'year_evaluation', 'template_evaluation', 'year_template_evaluation')}
        reduced = truncate_curve(model, curve_geometry(model))
        scores[family]['three_dimensional_year_evaluation'] = evaluate(stats['year_evaluation'], reduced)
        print(family, chosen['spec'], 'evaluation:', scores[family]['year_evaluation'], flush=True)
    final = fit_curve(stats['all_development'], best[winner]['spec'])
    save_model(output / 'final_curve_all_development.npz', final)
    geometry = plots(output, x, years, templates, models, scores, final)
    # Residuals on untouched evaluation years retain their label structure.
    residual_rows = []
    for family, model in models.items():
        take = np.flatnonzero(masks['year_evaluation'])
        residual = x[take] - predict(model, years[take], templates[take])
        for year in np.unique(years[take]):
            r = residual[years[take] == year]
            residual_rows.append(dict(family=family, year=int(year), count=len(r),
                                      mse=float(np.mean(r * r)),
                                      centroid_mse=float(np.mean(r.mean(0) ** 2))))
        if family == winner:
            residual_means = np.stack([residual[years[take] == y].mean(0)
                                       for y in np.unique(years[take])])
            np.savez_compressed(output / 'evaluation_residual_year_means.npz',
                                years=np.unique(years[take]), means=residual_means)
    write_csv(output / 'residuals_by_year.csv', residual_rows)
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout='constrained')
    for family in ('line', winner):
        selected = [row for row in residual_rows if row['family'] == family]
        axes[0].plot([r['year'] for r in selected], [r['mse'] for r in selected], label=family)
        axes[1].plot([r['year'] for r in selected], [r['centroid_mse'] for r in selected], label=family)
    for ax, title in zip(axes, ('Full residual MSE', 'Residual year-centroid MSE')):
        ax.set(xlabel='Untuned evaluation year', ylabel='Squared error per coordinate', title=title)
        ax.legend()
    fig.savefig(output / 'residual_structure.png', dpi=180)
    plt.close(fig)
    report = dict(selected_family=winner, selection=best, scores=scores, curve_geometry=geometry,
                  final_refit='All development rows after evaluation; no independent score claimed',
                  unseen_templates='Unknown template offsets are zero: mean of training-template offsets',
                  basis_domain=[1000, 2000], model_space='ambient_4096_no_projection_or_whitening')
    (output / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    lines = ['# Layer-4 years parameterization', '',
             f'Selected on validation years: **{winner}**, `{best[winner]["spec"]}`.', '',
             'All fits use 4096D activations and zero-sum prompt-template offsets. External test shards are never read.', '',
             '| Family | Validation MSE | New years MSE | New templates MSE | Both MSE | Curve dimensions at 95% |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for family in best:
        s = scores[family]
        lines.append(f'| {family} | {best[family]["validation_mse"]:.8g} | '
                     f'{s["year_evaluation"]["mse"]:.8g} | {s["template_evaluation"]["mse"]:.8g} | '
                     f'{s["year_template_evaluation"]["mse"]:.8g} | {geometry[family]["dimension95"]} |')
    lines += ['', 'Template checks use zero offsets for unseen templates; this measures raw transfer, including context shifts.',
              'Evaluation uses frozen settings refitted on fit+validation years. The final curve is then refitted on all development data.',
              'The curve is supervised by year, not an unsupervised discovery or proof of continuous topology.', '',
              '![Comparison](comparison.png)', '', '![Fitted curve](fitted_curve.png)', '',
              '![Residual year structure](residual_structure.png)', '']
    (output / 'report.md').write_text('\n'.join(lines))
    (output / 'COMPLETED.json').write_text(json.dumps(dict(selected_family=winner, node=manifest['node'])))
    print('Completed:', output, flush=True)


if __name__ == '__main__':
    main()
