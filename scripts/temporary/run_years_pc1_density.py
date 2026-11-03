"""Evaluate density support of saved years/layer-4 PC1s; never train models."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone
from statistics import NormalDist

import numpy as np
import torch

from dalg.analysis.pc1_density import KNNLogDensity, leading_components, score_component, project_pc1_segments
from dalg.data.shard_activations import ActivationBatchDataset, load_meta_index, stratified_split
from dalg.models.adaptive_q.mfa_hddc import load_mfa_hddc
from dalg.models.kmeans import load_kmeans

REPO = Path(__file__).resolve().parents[2]
DEFAULT_ACTIVATIONS = Path('/orfeo/scratch/dssc/zenocosini/dalg-cache/sae_manifold_years_neutral_llama31_8b_last')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def discover_runs(activation_dir):
    runs = []
    for method, kind, filename in [('hddc_em', 'hddc', 'mfa_model.pt'),
                                   ('kmeans_pca', 'kmeans', 'kmeans_model.pt')]:
        root = activation_dir / f'sae_manifold_years_neutral_{method}'
        for path in sorted(root.glob('*/run_spec.json')):
            spec = json.loads(path.read_text())
            if spec['dataset']['layer'] != 4:
                continue
            cfg = json.loads((path.parent / 'config.json').read_text())
            if (spec['dataset']['id'], spec['training']['model_kind']) != ('years_neutral', kind):
                raise ValueError(f'Unexpected model or dataset in {path}')
            if (cfg['layer'], cfg['d_model'], cfg['window'], cfg['drop_prefix']) != (4, 4096, 1, 0):
                raise ValueError(f'Unexpected activation contract in {path}')
            if Path(cfg['shard_dir']).name != activation_dir.name:
                raise ValueError(f'Unexpected source activations in {path}')
            checkpoint = path.parent / filename
            completion = json.loads((path.parent / 'TRAINING_COMPLETED.json').read_text())
            if not checkpoint.is_file() or completion.get('completed') is not True:
                raise ValueError(f'Incomplete run {path.parent}')
            runs.append(dict(run_id=path.parent.name, kind=kind, K=cfg['K'],
                             threshold=cfg['surgery_threshold'], config=cfg,
                             run_dir=path.parent, checkpoint=checkpoint))
    expected = {(kind, K, threshold) for kind in ['kmeans', 'hddc'] for K in [50, 100]
                for threshold in [0.0001, 0.001, 0.01, 0.1]}
    keys = [(r['kind'], r['K'], r['threshold']) for r in runs]
    if len(keys) != 16 or set(keys) != expected:
        raise ValueError(f'Expected exactly the 16 years/layer-4 runs, got {keys}')
    return sorted(runs, key=lambda r: (r['kind'], r['K'], r['threshold']))


def load_reference(activation_dir, runs):
    extraction = json.loads((activation_dir / 'config.json').read_text())
    if (extraction['num_rows'], extraction['d_model'], extraction['window'],
            extraction['drop_prefix'], extraction['token_selection']) != (20020, 4096, 1, 0, 'last'):
        raise ValueError('Expected the completed 20,020-row years last-token extraction')
    if not json.loads((activation_dir / 'progress.json').read_text())['complete']:
        raise ValueError('Extraction is incomplete')
    meta = load_meta_index(activation_dir, layer=4)
    if len(meta) != 20020 or {m['subset'] for m in meta} != {'years'}:
        raise ValueError('Expected exactly the years population')
    train, val = stratified_split(meta, val_frac=0.1, seed=42)
    ids = [meta[p]['global_row'] for p in val]
    for run in runs:
        cfg = run['config']
        saved = json.loads((run['run_dir'] / 'val_indices.json').read_text())
        if ((cfg['val_frac'], cfg['split_seed']) != (0.1, 42)
                or saved['val_global_rows'] != ids
                or (saved['train_rows'], saved['val_rows']) != (len(train), len(val))):
            raise ValueError(f'Saved split mismatch in {run["run_id"]}')
    dataset = ActivationBatchDataset(activation_dir, layer=4, row_subset=val,
                                    batch_size=512, drop_prefix=0, dtype=torch.float64,
                                    return_metadata=True, shuffle_shards=False,
                                    shuffle_within_shard=False)
    points, observed_ids = [], []
    for x, global_rows, token_positions in dataset:
        if token_positions.any():
            raise ValueError('Expected stored token slot zero')
        points.append(x)
        observed_ids.extend(global_rows.tolist())
    reference = torch.cat(points)
    if observed_ids != ids or reference.shape != (2002, 4096):
        raise ValueError('Reference stream is not aligned with the saved validation split')
    return reference, ids, extraction


def finite_exp(log_value):
    """Leave unrepresentable ordinary-space scores blank; retain exact log score."""
    if not -744 <= log_value <= 709:
        return None
    return math.exp(log_value)


def load_global_pca(path, activation_dir, reference_ids):
    """Validate and reuse a training-only PCA75 artifact, never refitting it."""
    pca = torch.load(path, map_location='cpu', weights_only=True)
    meta = load_meta_index(activation_dir, layer=4)
    train, _ = stratified_split(meta, val_frac=.1, seed=42)
    train_ids = [meta[p]['global_row'] for p in train]
    source = pca['source']
    if (pca['format'] != 'dalg_years_global_pca75_v1'
            or pca['target_variance_fraction'] != .75
            or pca['centering'] is not True or pca['standardization'] is not False
            or pca['whitening'] is not False or pca['training_count'] != 18018
            or (source['dataset'], source['layer'], source['split']) != ('years_neutral', 4, 'train')
            or source['training_global_rows'].tolist() != train_ids
            or source['validation_global_rows'].tolist() != reference_ids
            or source['extraction_config_sha256'] != sha256(activation_dir / 'config.json')):
        raise ValueError('PCA artifact does not match this training population and protocol')
    shards = {p.name: sha256(p) for p in sorted((activation_dir/'layer04').glob('shard_*.pt'))}
    if source['layer_shard_sha256'] != shards:
        raise ValueError('Activation shards differ from those used to fit the PCA')
    d = pca['dimension']
    cumulative = pca['cumulative_explained_variance']
    if (pca['basis'].shape != (d, 4096) or pca['mean'].shape != (4096,)
            or d < 1 or not cumulative[d-1] >= .75
            or (d > 1 and cumulative[d-2] >= .75)
            or not torch.allclose(pca['basis'] @ pca['basis'].T, torch.eye(d, dtype=torch.float64), atol=1e-10, rtol=1e-10)):
        raise ValueError('PCA basis or minimal retained dimension is invalid')
    return pca


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_report(output, rows, summaries):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    manifest = json.loads((output / 'manifest.json').read_text())
    pca_info = manifest.get('global_pca')
    dimension = manifest['dimension']
    valid = [row for row in rows if not row['exclusion_reason']]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for kind, color in [('kmeans', '#dd822e'), ('hddc', '#168a89')]:
        # Threshold runs share data and are dependent; each dot is descriptive.
        selected = [r for r in valid if r['kind'] == kind]
        axes[0].scatter([r['center_log_density'] for r in selected],
                        [r['log_retention'] / math.log(10) for r in selected],
                        s=9, alpha=.35, color=color, label=kind)
        axes[1].scatter([2 * r['half_length'] for r in selected],
                        [r['log_enrichment'] / math.log(10) for r in selected],
                        s=9, alpha=.35, color=color, label=kind)
        axes[2].scatter([r['alpha'] if pca_info else r['relative_eigengap'] for r in selected],
                        [r['log_retention' if pca_info else 'log_enrichment'] / math.log(10) for r in selected],
                        s=9, alpha=.35, color=color)
    axes[0].set(xlabel='Center log density (natural log)', ylabel='log10 normalized retention')
    axes[1].set(xlabel='Projected 80% interval length' if pca_info else '80% marginal interval length',
                ylabel='log10 enrichment over random')
    axes[2].set(xlabel='Retained original PC1 norm squared (alpha)' if pca_info else 'Relative leading eigengap',
                ylabel='log10 normalized retention' if pca_info else 'log10 enrichment over random')
    for ax in axes:
        ax.axhline(0, color='gray', lw=.7)
    axes[0].legend()
    title = (f'global PCA75, d={dimension}, variance={pca_info["retained_variance_fraction"]:.2%}'
             if pca_info else 'ambient D=4096')
    fig.suptitle(f'Years, layer 4: {title} density support (descriptive)')
    fig.savefig(output / 'diagnostics.png', dpi=160)
    plt.close(fig)

    # Fixed selection: the median-retention component in each method at K50/tau0.1.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for ax, kind in zip(axes, ['kmeans', 'hddc']):
        selected = sorted([r for r in valid if r['kind'] == kind and r['K'] == 50
                           and r['threshold'] == .1], key=lambda r: r['log_retention'])
        if not selected:
            continue
        row = selected[len(selected) // 2]
        result = torch.load(output / (row['run_id'] + '.pt'), weights_only=True)
        profile = result['components'][row['component']]
        x = profile['relative_positions'].numpy()
        curves = profile['log_relative_density_profiles'].numpy() / math.log(10)
        ax.plot(x, curves[1:].T, color='gray', alpha=.2, lw=.7)
        ax.plot(x, curves[0], color='#168a89', label='PC1', lw=2)
        ax.axhline(0, color='gray', lw=.7)
        ax.set(title=f'{kind}, K50, threshold 0.1, component {row["component"]}',
               xlabel='Position / interval half-length', ylabel='log10 density / center density')
        ax.legend()
    fig.savefig(output / 'example_profiles.png', dpi=160)
    plt.close(fig)

    lines = [f'# Years/layer-4 PC1 density pilot: {title}', '',
             f'Scores use k=20 and the saved 2,002-row validation reference; density dimension is {dimension}.',
             'Primary score: normalized density retention. Secondary control: enrichment over random directions.',
             'Retention = average density along the central 80% marginal interval / center density.',
             'Enrichment = PC1 retention / arithmetic mean random-direction retention (32 matched directions).',
             'Natural-log metrics are canonical; ordinary-space fields are blank if unrepresentable.', '',
             '| Method | K | Threshold | Eligible / total | Median log10 retention | Median log10 enrichment | Median alpha | Grid flags |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for s in summaries:
        lines.append(f'| {s["kind"]} | {s["K"]} | {s["threshold"]:g} | '
                     f'{s["eligible"]}/{s["total"]} | {s["median_log_retention"] / math.log(10):.5g} | '
                     f'{s["median_log_enrichment"] / math.log(10):.5g} | {s["median_alpha"]:.5g} | {s["grid_flagged"]} |')
    lines += ['', 'The pilot is a density-support diagnostic, not a tangent-angle measurement or a year-specific alignment score.',
              'The reference includes year and prompt-template variation. Validation participated in checkpoint selection.',
              f'kNN density uses the volume and exponent in {dimension} dimensions; center log density and radius are retained.',
              'Random directions are easy high-dimensional controls. Exceedance fractions are descriptive, not p-values.',
              'KMeans threshold variants are repeated PC1 geometry, not independent evidence. Summaries weight eligible components equally.',
              'Summary medians are calculated in log space; their exponentials use geometric interpolation for even component counts.',
              'Scores use adaptive integration. Uniform grids of 101/201 points are diagnostics and locate density peaks.',
              'Grid flags mark >0.05 natural-log change between uniform grids or between adaptive and 201-point retention.',
              'A large center quadrature fraction signals a potentially unresolved peak; inspect saved profiles.', '',
              '![Diagnostics](diagnostics.png)', '', '![Example profiles](example_profiles.png)', '']
    if pca_info:
        pca = torch.load(pca_info['path'], map_location='cpu', weights_only=True)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
        n = min(pca['input_dimension'], max(32, 2 * dimension))
        axes[0].plot(np.arange(1, n + 1), pca['cumulative_explained_variance'][:n].numpy())
        axes[0].axhline(.75, ls='--', color='gray')
        axes[0].axvline(dimension, ls='--', color='#168a89')
        axes[0].set(xlabel='Global PCA dimension', ylabel='Cumulative training variance',
                    title=f'Minimal d={dimension}; retained {pca_info["retained_variance_fraction"]:.4%}')
        for kind, color in [('kmeans', '#dd822e'), ('hddc', '#168a89')]:
            axes[1].hist([r['alpha'] for r in valid if r['kind'] == kind], bins=np.linspace(0, 1, 26),
                         histtype='step', lw=2, color=color, label=kind)
        axes[1].set(xlabel='alpha = ||P v_original|| squared', ylabel='Components (dependent threshold variants)')
        axes[1].legend()
        fig.savefig(output / 'pca_diagnostics.png', dpi=160)
        plt.close(fig)
        lines += [f'The single global PCA was fitted on 18,018 training activations: d={dimension}, '
                  f'variance={pca_info["retained_variance_fraction"]:.8%}; '
                  f'previous dimension={pca_info["previous_variance_fraction"]:.8%}.',
                  'Original covariance PC1 and eigenvalue are preserved. The original 80% interval is projected; no new cluster PC1 is fitted.',
                  'Each random baseline is a unit direction sampled directly in the reduced space at the same projected center and length.',
                  'Interpretation: density support of the projected original PC1. Discarded dimensions are invisible; template variation remains.',
                  '![Global PCA and direction retention](pca_diagnostics.png)', '']
    (output / 'report.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--activation-dir', type=Path, default=DEFAULT_ACTIVATIONS)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--global-pca', type=Path, help='Saved training-only global PCA75; omission retains the ambient experiment')
    parser.add_argument('--smoke', action='store_true', help='Only two components of the first run; separate partial output')
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    activation_dir = args.activation_dir.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    runs = discover_runs(activation_dir)
    reference, reference_ids, extraction = load_reference(activation_dir, runs)
    pca = None
    pca_info = None
    if args.global_pca:
        pca_path = args.global_pca.resolve(strict=True)
        pca = load_global_pca(pca_path, activation_dir, reference_ids)
        reference = (reference - pca['mean']) @ pca['basis'].T
        pca_info = dict(path=str(pca_path), sha256=sha256(pca_path), dimension=pca['dimension'],
                        retained_variance_fraction=pca['retained_variance_fraction'],
                        previous_variance_fraction=pca['previous_variance_fraction'],
                        target_variance_fraction=.75, training_count=18018,
                        null_projection_norm_tolerance=1e-12)
        print(f'Using shared training PCA: d={pca["dimension"]}, retained variance={pca["retained_variance_fraction"]:.6%}', flush=True)
    estimator = KNNLogDensity(reference.to(args.device), k=20)
    output.mkdir(parents=True)
    sources = []
    for run in runs:
        sources.append(dict(run_id=run['run_id'], checkpoint=str(run['checkpoint']),
                            checkpoint_sha256=sha256(run['checkpoint']),
                            config_sha256=sha256(run['run_dir'] / 'config.json'),
                            split_sha256=sha256(run['run_dir'] / 'val_indices.json')))
    manifest = dict(format='dalg_years_pc1_density_pca75_v1' if pca else 'dalg_years_pc1_density_v1',
                    created_at=datetime.now(timezone.utc).isoformat(),
                    activation_dir=str(activation_dir), dataset='years_neutral', layer=4,
                    reference_split='validation', reference_global_rows=reference_ids,
                    reference_source_sha256=extraction['source_records_sha256'],
                    extraction_config_sha256=sha256(activation_dir / 'config.json'),
                    estimator='k/(n V_d r_k^d)', k=20, dimension=estimator.dimension, input_dimension=4096,
                    evaluation_space='global_pca75' if pca else 'ambient', global_pca=pca_info,
                    preprocessing='training-centered orthogonal PCA, no whitening or standardization' if pca else 'none',
                    evaluated_direction='original covariance PC1 projected as a segment' if pca else 'original covariance PC1',
                    primary_metric='normalized_retention', probability=.8, grid_points=201, coarse_grid_points=101,
                    integration='peak-split adaptive QUADPACK, epsrel=1e-4; grid profiles diagnostic only',
                    n_random=32, random_seed_rule='42 + component index, common across configurations',
                    random_direction_space='unit isotropic in density space; same projected length as PC1',
                    score='log average density minus center log density',
                    enrichment='PC1 density average / arithmetic mean random density averages',
                    torch_version=str(torch.__version__), device=args.device, sources=sources,
                    source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
                    source_sha256={str(p.relative_to(REPO)): sha256(p) for p in
                                   [Path(__file__), REPO / 'src/dalg/analysis/pc1_density.py']},
                    smoke=args.smoke, caveats=['validation used for model selection',
                    'projected support surrogate; discarded dimensions invisible' if pca else 'ambient support surrogate',
                    'includes template variation', 'dependent threshold variants'])
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    rows, summaries = [], []
    kmeans_geometry = {}
    for run in runs[:1] if args.smoke else runs:
        print(f'Loading {run["run_id"]}', flush=True)
        loader = load_kmeans if run['kind'] == 'kmeans' else load_mfa_hddc
        model = loader(run['checkpoint'], map_location='cpu')
        if (model.K, model.D) != (run['K'], 4096):
            raise ValueError('Checkpoint dimensions differ from run configuration')
        geometry = leading_components(model, run['kind'])
        del model
        if run['kind'] == 'kmeans':
            key = run['K']
            if key in kmeans_geometry:
                baseline = kmeans_geometry[key]
                for field in ['centers', 'eigenvalues']:
                    if not torch.equal(geometry[field], baseline[field]):
                        raise ValueError(f'KMeans threshold variants differ in {field}')
                overlaps = (geometry['vectors'] * baseline['vectors']).sum(1).abs()
                eligible = torch.tensor([not r for r in geometry['exclusion_reasons']])
                if not torch.allclose(overlaps[eligible], torch.ones_like(overlaps[eligible]), atol=1e-8):
                    raise ValueError('KMeans threshold variants have different PC1 directions')
            else:
                kmeans_geometry[key] = geometry
        projected_geometry = project_pc1_segments(geometry, pca['mean'], pca['basis']) if pca else None
        evaluated = projected_geometry if pca else geometry
        results, run_rows = {}, []
        for component in range(min(2, run['K']) if args.smoke else run['K']):
            row = dict(run_id=run['run_id'], kind=run['kind'], K=run['K'],
                       threshold=run['threshold'], component=component,
                       effective_rank=int(geometry['ranks'][component]),
                       occupancy=float(geometry['occupancy'][component]),
                       occupancy_kind='training_count' if run['kind'] == 'kmeans' else 'mixture_weight',
                       density_dimension=estimator.dimension,
                       alpha=float(projected_geometry['alpha'][component]) if pca else 1.,
                       exclusion_reason=evaluated['exclusion_reasons'][component])
            if not row['exclusion_reason']:
                eigenvalue = float(geometry['eigenvalues'][component])
                projected_half_length = float(projected_geometry['half_lengths'][component]) if pca else None
                result = score_component(estimator, evaluated['centers'][component],
                                         evaluated['vectors'][component], eigenvalue,
                                         half_length=projected_half_length, seed=42 + component, adaptive=True)
                results[component] = result
                row.update({key: value for key, value in result.items() if not isinstance(value, torch.Tensor)})
                row.update(leading_eigenvalue=eigenvalue,
                           original_half_length=NormalDist().inv_cdf(.9) * math.sqrt(eigenvalue),
                           projected_half_length=result['half_length'],
                           relative_eigengap=float(geometry['eigengap'][component]) / eigenvalue,
                           normalized_retention=finite_exp(row['log_retention']),
                           enrichment=finite_exp(row['log_enrichment']),
                           center_density=finite_exp(row['center_log_density']),
                           grid_flagged=(row['max_grid_log_retention_change'] > .05
                                         or row['grid_log_enrichment_change'] > .05
                                         or row['max_adaptive_vs_grid_log_retention_change'] > .05))
            run_rows.append(row)
            if component % 10 == 0:
                print(f'  component {component + 1}/{run["K"]}', flush=True)
        torch.save(dict(run_id=run['run_id'], geometry=geometry, projected_geometry=projected_geometry, components=results),
                   output / (run['run_id'] + '.pt'))
        rows.extend(run_rows)
        eligible = [r for r in run_rows if not r['exclusion_reason']]
        if not eligible:
            raise ValueError(f'No eligible PC1 geometry in {run["run_id"]}')
        med_ret = float(np.median([r['log_retention'] for r in eligible]))
        med_enrich = float(np.median([r['log_enrichment'] for r in eligible]))
        summaries.append(dict(run_id=run['run_id'], kind=run['kind'], K=run['K'], threshold=run['threshold'],
                              total=len(run_rows), eligible=len(eligible),
                              median_log_retention=med_ret, median_retention=finite_exp(med_ret),
                              median_log_enrichment=med_enrich, median_enrichment=finite_exp(med_enrich),
                              density_dimension=estimator.dimension,
                              median_alpha=float(np.median([r['alpha'] for r in eligible])),
                              min_alpha=min(r['alpha'] for r in eligible), max_alpha=max(r['alpha'] for r in eligible),
                              median_center_log_density=float(np.median([r['center_log_density'] for r in eligible])),
                              grid_flagged=sum(r['grid_flagged'] for r in eligible)))
        write_csv(output / 'components.csv', rows)
        write_csv(output / 'summary.csv', summaries)
        print(f'  median log retention={med_ret:.4g}, log enrichment={med_enrich:.4g}', flush=True)
    if not args.smoke:
        make_report(output, rows, summaries)
    (output / 'COMPLETED.json').write_text(json.dumps(dict(completed=True, smoke=args.smoke,
                                                         configurations=len(summaries), components=len(rows)), indent=2))
    print(f'Saved results in {output}', flush=True)


if __name__ == '__main__':
    main()
