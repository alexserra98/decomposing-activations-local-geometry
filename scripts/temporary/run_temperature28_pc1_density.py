"""Repeat the PCA75 PC1-density pilot on saved temperature/layer-28 models."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
import math
import multiprocessing
import os
from pathlib import Path
import socket

import numpy as np
import torch

from dalg.analysis.pc1_density import (
    KNNLogDensity, fit_global_variance_pca, leading_components,
    project_pc1_segments, score_component,
)
from dalg.data.shard_activations import ActivationBatchDataset, load_meta_index, stratified_split
from dalg.models.adaptive_q.mfa_hddc import load_mfa_hddc
from dalg.models.kmeans import load_kmeans
from run_years_pc1_density import REPO, finite_exp, sha256, write_csv

DATASET, LAYER = 'temperature_02d', 28
DEFAULT_ACTIVATIONS = Path('/orfeo/cephfs/scratch/dssc/zenocosini/dalg-cache/sae_manifold_temperature_02d_llama31_8b_last')
ROOT_OUTPUT = REPO / 'outputs/experiments/temperature_layer28_pc1_density_pca75'


def initialize_density_worker(reference):
    global WORKER_DENSITY
    torch.set_num_threads(1)
    WORKER_DENSITY = KNNLogDensity(reference, k=20)


def score_in_worker(center, vector, eigenvalue, length, seed):
    return score_component(WORKER_DENSITY, center, vector, eigenvalue,
                           half_length=length, seed=seed, adaptive=True)


def validate_resume_manifest(previous, current):
    fields = ['dataset', 'layer', 'reference_global_rows', 'dimension', 'k', 'n_random',
              'probability', 'random_seed_rule', 'integration', 'sources']
    metric_path = 'src/dalg/analysis/pc1_density.py'
    if (any(previous[field] != current[field] for field in fields)
            or previous['global_pca']['sha256'] != current['global_pca']['sha256']
            or previous['source_sha256'][metric_path] != current['source_sha256'][metric_path]):
        raise ValueError('Resume inputs or metric protocol differ from this evaluation')


def discover_runs(root):
    runs = []
    for method, kind, filename in [('hddc_em', 'hddc', 'mfa_model.pt'),
                                   ('kmeans_pca', 'kmeans', 'kmeans_model.pt')]:
        for path in sorted((root/f'sae_manifold_{DATASET}_{method}').glob('*/run_spec.json')):
            spec = json.loads(path.read_text())
            if spec['dataset']['layer'] != LAYER:
                continue
            cfg = json.loads((path.parent/'config.json').read_text())
            if ((spec['dataset']['id'], spec['training']['model_kind']) != (DATASET, kind)
                    or (cfg['layer'], cfg['d_model'], cfg['window'], cfg['drop_prefix']) != (LAYER, 4096, 1, 0)
                    or Path(cfg['shard_dir']).name != root.name):
                raise ValueError(f'Unexpected dataset/layer/model in {path}')
            checkpoint = path.parent/filename
            if (not checkpoint.is_file()
                    or json.loads((path.parent/'TRAINING_COMPLETED.json').read_text()).get('completed') is not True):
                raise ValueError(f'Incomplete saved fit: {path.parent}')
            runs.append(dict(run_id=path.parent.name, kind=kind, K=cfg['K'],
                             threshold=cfg['surgery_threshold'], config=cfg,
                             run_dir=path.parent, checkpoint=checkpoint))
    expected = {(kind, K, tau) for kind in ['hddc', 'kmeans'] for K in [50, 100]
                for tau in [.0001, .001, .01, .1]}
    if len(runs) != 16 or {(r['kind'], r['K'], r['threshold']) for r in runs} != expected:
        raise ValueError('Require all 16 saved temperature/layer-28 configurations')
    return sorted(runs, key=lambda r: (r['kind'], r['K'], r['threshold']))


def validate_saved_splits(runs, train_ids, val_ids):
    for run in runs:
        saved = json.loads((run['run_dir']/'val_indices.json').read_text())
        cfg = run['config']
        if ((cfg['val_frac'], cfg['split_seed']) != (.1, 42)
                or saved['val_global_rows'] != val_ids
                or (saved['train_rows'], saved['val_rows']) != (len(train_ids), len(val_ids))):
            raise ValueError(f'Saved split mismatch in {run["run_id"]}')


def population(root, runs):
    extraction = json.loads((root/'config.json').read_text())
    if ((extraction['num_rows'], extraction['d_model'], extraction['window'],
         extraction['drop_prefix'], extraction['token_selection']) != (15000, 4096, 1, 0, 'last')
            or LAYER not in extraction['layers']
            or not json.loads((root/'progress.json').read_text())['complete']):
        raise ValueError('Require the complete 15,000-row temperature last-token extraction')
    meta = load_meta_index(root, layer=LAYER)
    if len(meta) != 15000 or {m['subset'] for m in meta} != {'temperature'}:
        raise ValueError('Unexpected temperature activation population')
    train, val = stratified_split(meta, val_frac=.1, seed=42)
    train_ids = [meta[p]['global_row'] for p in train]
    val_ids = [meta[p]['global_row'] for p in val]
    if (len(train_ids), len(val_ids)) != (13500, 1500):
        raise ValueError('Unexpected temperature split sizes')
    validate_saved_splits(runs, train_ids, val_ids)
    return extraction, train, val, train_ids, val_ids


def stream(root, positions, expected_ids):
    observed = []
    data = ActivationBatchDataset(root, layer=LAYER, row_subset=positions, batch_size=512,
                                  drop_prefix=0, dtype=torch.float64, return_metadata=True,
                                  shuffle_shards=False, shuffle_within_shard=False)
    for x, global_rows, token_positions in data:
        if token_positions.any():
            raise ValueError('Expected last-token slot zero')
        observed.extend(global_rows.tolist())
        yield x
    if observed != expected_ids:
        raise ValueError('Activation stream order differs from the saved global row IDs')


def pca_artifact(path, root, train, train_ids, val_ids, extraction):
    shards = {p.name: sha256(p) for p in sorted((root/'layer28').glob('shard_*.pt'))}
    source = dict(activation_dir=str(root), dataset=DATASET, layer=LAYER, split='train',
                  split_seed=42, val_frac=.1, training_global_rows=torch.tensor(train_ids),
                  validation_global_rows=torch.tensor(val_ids),
                  extraction_config_sha256=sha256(root/'config.json'),
                  source_records_sha256=extraction['source_records_sha256'], layer_shard_sha256=shards)
    if path.exists():
        pca = torch.load(path, weights_only=True, map_location='cpu')
        validate_pca(pca, source)
        print(f'Reusing training PCA {path}', flush=True)
        return pca
    print('Fitting centered, unwhitened PCA on 13,500 training rows; target 75% variance', flush=True)
    pca = fit_global_variance_pca(stream(root, train, train_ids), .75)
    pca.update(format='dalg_temperature28_global_pca75_v1', source=source,
               created_at=datetime.now(timezone.utc).isoformat(),
               source_sha256={str(p.relative_to(REPO)): sha256(p) for p in
                              [Path(__file__), REPO/'src/dalg/analysis/pc1_density.py']})
    validate_pca(pca, source)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as handle:
        torch.save(pca, handle)
    print(f'Saved PCA: d={pca["dimension"]}, retained={pca["retained_variance_fraction"]:.8%}, '
          f'previous={pca["previous_variance_fraction"]:.8%}', flush=True)
    return pca


def validate_pca(pca, source):
    stored = pca['source']
    fields = ['dataset', 'layer', 'split', 'split_seed', 'val_frac', 'extraction_config_sha256',
              'source_records_sha256', 'layer_shard_sha256']
    if (pca['format'] != 'dalg_temperature28_global_pca75_v1'
            or any(stored[key] != source[key] for key in fields)
            or not torch.equal(stored['training_global_rows'], source['training_global_rows'])
            or not torch.equal(stored['validation_global_rows'], source['validation_global_rows'])
            or pca['training_count'] != 13500 or pca['input_dimension'] != 4096
            or pca['target_variance_fraction'] != .75 or pca['centering'] is not True
            or pca['whitening'] is not False or pca['standardization'] is not False):
        raise ValueError('PCA artifact does not match this dataset, layer and training split')
    d, cumulative = pca['dimension'], pca['cumulative_explained_variance']
    if (pca['mean'].shape != (4096,) or pca['basis'].shape != (d, 4096)
            or d < 1 or not cumulative[d-1] >= .75 or (d > 1 and cumulative[d-2] >= .75)
            or not torch.allclose(pca['basis']@pca['basis'].T, torch.eye(d, dtype=torch.float64), atol=1e-10, rtol=1e-10)):
        raise ValueError('Invalid PCA basis or nonminimal retained dimension')


def summarize(run, rows, dimension, reused_from):
    valid = [r for r in rows if not r['exclusion_reason']]
    summary = dict(run_id=run['run_id'], kind=run['kind'], K=run['K'], threshold=run['threshold'],
                   total=len(rows), eligible=len(valid), density_dimension=dimension,
                   numerical_evaluation_reused_from=reused_from,
                   exclusion_counts=json.dumps(dict(Counter(r['exclusion_reason'] for r in rows if r['exclusion_reason']))))
    for field in ['log_retention', 'log_enrichment', 'alpha', 'center_log_density', 'center_knn_radius']:
        summary['median_'+field] = float(np.median([r[field] for r in valid])) if valid else None
    for score in ['retention', 'enrichment']:
        summary['median_'+score] = finite_exp(summary['median_log_'+score]) if valid else None
    summary['grid_flagged'] = sum(r['grid_flagged'] for r in valid)
    return summary


def evaluate(root, runs, reference, pca, pca_path, output, val_ids, smoke, workers, resume_from):
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    estimator = KNNLogDensity((reference-pca['mean'])@pca['basis'].T, k=20)
    manifest = dict(format='dalg_temperature28_pc1_density_pca75_v1', dataset=DATASET, layer=LAYER,
        created_at=datetime.now(timezone.utc).isoformat(), activation_dir=str(root),
        reference_split='validation', reference_global_rows=val_ids, reference_count=len(val_ids),
        training_count=pca['training_count'], extraction_config_sha256=sha256(root/'config.json'),
        global_pca=dict(path=str(pca_path), sha256=sha256(pca_path), dimension=pca['dimension'],
                        retained_variance_fraction=pca['retained_variance_fraction'],
                        previous_variance_fraction=pca['previous_variance_fraction']),
        dimension=estimator.dimension, input_dimension=4096, k=20, n_random=32, probability=.8,
        evaluation_space='global_pca75', primary_metric='normalized_retention',
        score='log average density minus log center density',
        enrichment='PC1 retention / arithmetic mean of 32 random retentions',
        random_seed_rule='42 + component index, shared across configurations',
        random_direction_space='unit isotropic directly in retained density space; projected PC1 length',
        integration='adaptive QUADPACK epsrel=1e-4; splits at all 201 pilot positions and refined peaks',
        original_pc1='KMeans empirical covariance; HDDC full covariance including isotropic noise',
        smoke=smoke, torch_version=str(torch.__version__),
        execution=dict(hostname=socket.gethostname(), job=os.environ['SLURM_JOB_ID'], step=os.environ['SLURM_STEP_ID']),
        component_workers=workers, worker_torch_threads=1,
        source_sha256={str(p.relative_to(REPO)): sha256(p) for p in
                       [Path(__file__), REPO/'src/dalg/analysis/pc1_density.py',
                        REPO/'scripts/temporary/run_years_pc1_density.py']},
        sources=[dict(run_id=r['run_id'], checkpoint=str(r['checkpoint']), checkpoint_sha256=sha256(r['checkpoint']),
                      config_sha256=sha256(r['run_dir']/'config.json'), split_sha256=sha256(r['run_dir']/'val_indices.json')) for r in runs],
        caveats=['validation participated in checkpoint selection', 'includes template/prompt variation',
                 'discarded dimensions invisible', 'dependent threshold variants',
                 'component-specific centers and scales; not a tangent-angle measurement'])
    resumed = {}
    if resume_from is not None:
        resume_from = resume_from.resolve(strict=True)
        previous_manifest = json.loads((resume_from/'manifest.json').read_text())
        validate_resume_manifest(previous_manifest, manifest)
        paths = [resume_from/(run['run_id']+'.pt') for run in runs if (resume_from/(run['run_id']+'.pt')).exists()]
        manifest['resumed_from'] = dict(path=str(resume_from), manifest_sha256=sha256(resume_from/'manifest.json'),
                                       run_bundle_sha256={p.name: sha256(p) for p in paths})
        for path in paths:
            bundle = torch.load(path, map_location='cpu', weights_only=True)
            if bundle['run_id'] != path.stem:
                raise ValueError('Resume bundle identity mismatch')
            resumed[path.stem] = bundle
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    rows, summaries, cache = [], [], {}
    pool = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                               initializer=initialize_density_worker, initargs=(estimator.reference,))
    for run in runs[:1] if smoke else runs:
        print(f'Loading {run["run_id"]}', flush=True)
        saved = resumed.get(run['run_id'])
        if saved is not None:
            geometry = saved['geometry']
            if geometry['centers'].shape != (run['K'], 4096):
                raise ValueError('Resume geometry dimensions differ')
        else:
            loader = load_kmeans if run['kind'] == 'kmeans' else load_mfa_hddc
            model = loader(run['checkpoint'], map_location='cpu')
            if (model.K, model.D) != (run['K'], 4096):
                raise ValueError('Checkpoint dimensions differ from the run configuration')
            geometry = leading_components(model, run['kind'])
            del model
        projected = project_pc1_segments(geometry, pca['mean'], pca['basis'])
        if saved is not None:
            for field in ['centers', 'vectors', 'half_lengths', 'alpha']:
                torch.testing.assert_close(projected[field], saved['projected_geometry'][field], rtol=0, atol=0)
            expected = {i for i, reason in enumerate(projected['exclusion_reasons']) if not reason}
            if set(saved['components']) != expected:
                raise ValueError('Resume bundle has an incomplete set of evaluated components')
        previous = cache.get(run['K']) if run['kind'] == 'kmeans' else None
        if previous:
            old = previous['geometry']
            for field in ['centers', 'eigenvalues']:
                if not torch.equal(geometry[field], old[field]):
                    raise ValueError(f'KMeans threshold variants differ in {field}')
            eligible = torch.tensor([not reason for reason in geometry['exclusion_reasons']])
            if not torch.allclose((geometry['vectors']*old['vectors']).sum(1).abs()[eligible],
                                  torch.ones(int(eligible.sum()), dtype=torch.float64), atol=1e-8, rtol=1e-8):
                raise ValueError('KMeans threshold variants have different original PC1 geometry')
        reuse = saved is not None or bool(previous and geometry['exclusion_reasons'] == previous['geometry']['exclusion_reasons']
                     and projected['exclusion_reasons'] == previous['projected']['exclusion_reasons']
                     and all(torch.equal(projected[field], previous['projected'][field])
                             for field in ['centers', 'vectors', 'half_lengths']))
        reused_from = f'{resume_from.name}/{run["run_id"]}' if saved is not None else previous['run_id'] if reuse else ''
        reused_results = saved['components'] if saved is not None else previous['results'] if reuse else None
        count = min(2, run['K']) if smoke else run['K']
        futures = {} if reuse else {i: pool.submit(score_in_worker, projected['centers'][i], projected['vectors'][i],
                                  float(geometry['eigenvalues'][i]), float(projected['half_lengths'][i]), 42+i)
                                  for i in range(count) if not projected['exclusion_reasons'][i]}
        results, run_rows = {}, []
        for component in range(count):
            row = dict(run_id=run['run_id'], kind=run['kind'], K=run['K'], threshold=run['threshold'],
                       component=component, density_dimension=estimator.dimension,
                       effective_rank=int(geometry['ranks'][component]), occupancy=float(geometry['occupancy'][component]),
                       occupancy_kind='training_count' if run['kind'] == 'kmeans' else 'mixture_weight',
                       alpha=float(projected['alpha'][component]), exclusion_reason=projected['exclusion_reasons'][component],
                       numerical_evaluation_reused_from=reused_from)
            if not row['exclusion_reason']:
                result = reused_results[component] if reuse else futures[component].result()
                results[component] = result
                row.update({key: value for key, value in result.items() if not isinstance(value, torch.Tensor)})
                row.update(leading_eigenvalue=float(geometry['eigenvalues'][component]),
                           relative_eigengap=float(geometry['eigengap'][component]/geometry['eigenvalues'][component]),
                           original_half_length=float(projected['original_half_lengths'][component]),
                           projected_half_length=result['half_length'], normalized_retention=finite_exp(row['log_retention']),
                           enrichment=finite_exp(row['log_enrichment']), center_density=finite_exp(row['center_log_density']),
                           grid_flagged=max(row['max_grid_log_retention_change'], row['grid_log_enrichment_change'],
                                            row['max_adaptive_vs_grid_log_retention_change']) > .05)
            run_rows.append(row)
            if component % 10 == 0:
                print(f'  {component+1}/{run["K"]} components; reused={bool(reuse)}', flush=True)
        torch.save(dict(run_id=run['run_id'], geometry=geometry, projected_geometry=projected, components=results),
                   output/(run['run_id']+'.pt'))
        if run['kind'] == 'kmeans' and not previous:
            cache[run['K']] = dict(run_id=run['run_id'], geometry=geometry, projected=projected, results=results)
        rows.extend(run_rows)
        summaries.append(summarize(run, run_rows, estimator.dimension, reused_from))
        write_csv(output/'components.csv', rows)
        write_csv(output/'summary.csv', summaries)
        print(summaries[-1], flush=True)
    pool.shutdown()
    if not smoke and (len(rows) != 1200 or len(summaries) != 16):
        raise ValueError('Full pilot must cover 16 configurations and 1,200 component records')
    (output/'EVALUATED.json').write_text(json.dumps(dict(smoke=smoke, configurations=len(summaries), components=len(rows)), indent=2))
    if smoke:
        return
    validate_integrals(estimator, output, rows)
    make_report(output, rows, summaries, manifest)
    (output/'COMPLETED.json').write_text(json.dumps(dict(completed=True, smoke=False, configurations=16,
                                                        components=1200, eligible=sum(not r['exclusion_reason'] for r in rows)), indent=2))


def validate_integrals(estimator, output, rows):
    selected = {}
    for row in rows:
        if row['exclusion_reason']:
            continue
        key = row['kind'], row['K']
        if key not in selected or row['max_adaptive_vs_grid_log_retention_change'] > selected[key]['max_adaptive_vs_grid_log_retention_change']:
            selected[key] = row
    checks = []
    for row in selected.values():
        bundle = torch.load(output/(row['run_id']+'.pt'), weights_only=True)
        i, projected = row['component'], bundle['projected_geometry']
        original = bundle['components'][i]
        dense = score_component(estimator, projected['centers'][i], projected['vectors'][i],
                                float(bundle['geometry']['eigenvalues'][i]), half_length=original['half_length'],
                                seed=42+i, grid_points=201, adaptive=True)
        change = float((dense['log_direction_retentions']-original['log_direction_retentions']).abs().max())
        enrichment_change = abs(dense['log_enrichment']-original['log_enrichment'])
        checks.append(dict(run_id=row['run_id'], component=i, max_direction_log_score_change=change,
                           log_enrichment_change=enrichment_change, passed=max(change, enrichment_change) < .001))
        print('Denser-pilot check:', checks[-1], flush=True)
    expected = {(kind, K) for kind in ['hddc', 'kmeans'] for K in [50, 100]}
    no_eligible = sorted(expected-set(selected))
    result = dict(passed=bool(checks) and all(r['passed'] for r in checks), checks=checks,
                  groups_without_eligible_components=no_eligible, pilot_positions=[201, 401],
                  tolerance_natural_log=.001, selection='largest adaptive/grid discrepancy per method/K')
    (output/'integration_validation.json').write_text(json.dumps(result, indent=2))
    if not result['passed']:
        raise RuntimeError('Denser-pilot numerical validation failed')


def make_report(output, rows, summaries, manifest):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {'hddc': '#007F86', 'kmeans': '#D67D24'}
    info = manifest['global_pca']
    title = f'Temperature · layer 28 · PCA d={manifest["dimension"]}, variance={info["retained_variance_fraction"]:.4%}'
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True, layout='constrained')
    profile_rows = []
    for ax, K in zip(axes, [50, 100]):
        for kind in ['hddc', 'kmeans']:
            summary = next(r for r in summaries if r['kind'] == kind and r['K'] == K and r['threshold'] == .1)
            bundle = torch.load(output/(summary['run_id']+'.pt'), weights_only=True)
            if not bundle['components']:
                continue
            results = list(bundle['components'].values())
            positions = results[0]['relative_positions']
            profiles = torch.stack([r['log_relative_density_profiles'] for r in results])
            profiles = .5*(profiles+profiles.flip(-1))
            for name, curve, style in [('PC1', profiles[:, 0].mean(0), '-'),
                                       ('random', profiles[:, 1:].mean((0, 1)), '--')]:
                ax.plot(positions.numpy(), curve.numpy(), ls=style, color=colors[kind], lw=2,
                        label=f'{kind.upper() if kind == "hddc" else "KMeans"} {name}')
                profile_rows.extend(dict(K=K, kind=kind, direction=name, position=float(x),
                                         mean_log_relative_density=float(y)) for x, y in zip(positions, curve))
        ax.set(title=f'K={K}, threshold 0.1', xlabel='Normalized position s/L')
        ax.axhline(0, color='gray', lw=.6)
        ax.legend(frameon=False, fontsize=9)
    axes[0].set_ylabel('Mean log(density / center density)')
    fig.suptitle(title+'\nEqual-component mean, sign-symmetrized profiles', fontsize=12)
    for extension in ['png', 'pdf', 'svg']:
        fig.savefig(output/f'mean_profiles.{extension}', dpi=170)
    plt.close(fig)
    if profile_rows:
        write_csv(output/'mean_profiles.csv', profile_rows)
    valid = [r for r in rows if not r['exclusion_reason'] and r['threshold'] == .1]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout='constrained')
    for kind in ['hddc', 'kmeans']:
        for K, marker in [(50, 'o'), (100, 'x')]:
            group = [r for r in valid if r['kind'] == kind and r['K'] == K]
            for ax, field, label in zip(axes, ['center_log_density', 'projected_half_length', 'alpha'],
                                       ['Center log density', 'Projected half-length', 'Retained PC1 norm squared (alpha)']):
                ax.scatter([r[field] for r in group], [r['log_retention'] for r in group],
                           s=15, alpha=.5, marker=marker, color=colors[kind], label=f'{kind} K={K}')
                ax.set(xlabel=label, ylabel='Log retention')
                ax.axhline(0, color='gray', lw=.5)
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle(title+' · threshold 0.1 diagnostics', fontsize=11)
    fig.savefig(output/'diagnostics.png', dpi=170)
    plt.close(fig)
    fmt = lambda x: 'NA' if x is None else f'{x:.5g}'
    lines = ['# Temperature/layer-28 PC1 density support', '', title, '',
             'All 16 saved fits are reused; no model training, activation extraction or assignment is rerun.',
             'The shared PCA is fitted only on 13,500 training activations, centered without standardization or whitening.',
             f'The minimum dimension reaching 75% variance is {manifest["dimension"]}; '
             f'the preceding dimension retains {info["previous_variance_fraction"]:.8%}.',
             'The common kNN reference is the exact saved 1,500-row validation split; k=20, Euclidean distances.',
             'Retain original covariance PC1/eigenvalue, including HDDC isotropic noise; project the original central-80% marginal segment.',
             'Random baseline: 32 unit isotropic directions in the retained space, same center/projected length; seed=42+component index.', '',
             '```text', 'a = Phi^-1(0.9) sqrt(lambda); alpha = ||P v||^2; L = a sqrt(alpha)',
             'log p(z) = log(20/1500) - log V_d - d log r20(z)',
             'A(u) = (1/(2L)) integral[-L,L] p(z+s*u)/p(z) ds',
             'E = A(PC1) / arithmetic_mean(A(random_1), ..., A(random_32))', '```', '',
             'Retention is primary; center density/radius, alpha, interval lengths and exclusions are diagnostics.',
             'Medians are computed in log space and exponentiated, with equal component weights.', '',
             '| Method | K | Threshold | Eligible/total | Median retention | Median enrichment | Median alpha |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for s in summaries:
        lines.append(f'| {s["kind"]} | {s["K"]} | {s["threshold"]:g} | {s["eligible"]}/{s["total"]} | '
                     f'{fmt(s["median_retention"])} | {fmt(s["median_enrichment"])} | {fmt(s["median_alpha"])} |')
    validation = json.loads((output/'integration_validation.json').read_text())
    maximum = max(c['max_direction_log_score_change'] for c in validation['checks'])
    lines += ['', f'Numerical validation: {len(validation["checks"])} denser-pilot checks passed; maximum log-score change {maximum:.3g}.',
              f'Eligible components: {len([r for r in rows if not r["exclusion_reason"]])}/1200. '
              f'Exclusions: {dict(Counter(r["exclusion_reason"] for r in rows if r["exclusion_reason"]))}.',
              'Estimated adaptive quadrature relative error is required to be <=1e-4. Uniform-grid flags are diagnostic.',
              'Exact repeated KMeans evaluation inputs reuse scores/profiles across thresholds; recorded in the CSV.', '',
              '![Mean profiles](mean_profiles.png)', '', '![Diagnostics](diagnostics.png)', '',
              'Mean profile curves average log densities, whereas scores integrate density. The PC1 sign is arbitrary, so curves are symmetrized.',
              'Scores measure projected density support at model-specific centers and scales, rather than a known manifold tangent angle.',
              'Discarded coordinates are invisible; prompt/template variation remains. Validation participated in checkpoint selection.',
              'KMeans threshold variants and all shared-data models are dependent; no calibrated significance claim is made.',
              'If d=1, every unit direction describes the same symmetric interval and random enrichment cannot distinguish orientation.', '',
              '[Components](components.csv) · [Summaries](summary.csv) · [Manifest](manifest.json) · [Numerical validation](integration_validation.json)', '']
    (output/'report.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--activation-dir', type=Path, default=DEFAULT_ACTIVATIONS)
    parser.add_argument('--pca', type=Path, default=ROOT_OUTPUT/'global_pca_training_v1.pt')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--resume-from', type=Path)
    parser.add_argument('--fit-only', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_STEP_ID') in [None, 'batch', 'extern']:
        raise RuntimeError('Run numerical work on a compute node through srun')
    if not args.fit_only and args.output is None:
        parser.error('--output is required for evaluation')
    if not args.fit_only and args.output.resolve().exists():
        raise FileExistsError(args.output)
    if not 1 <= args.workers <= args.threads:
        parser.error('--workers must be between 1 and --threads')
    torch.set_num_threads(args.threads)
    root, pca_path = args.activation_dir.resolve(strict=True), args.pca.resolve()
    runs = discover_runs(root)
    extraction, train, val, train_ids, val_ids = population(root, runs)
    pca = pca_artifact(pca_path, root, train, train_ids, val_ids, extraction)
    if args.fit_only:
        return
    reference = torch.cat(list(stream(root, val, val_ids)))
    if reference.shape != (1500, 4096):
        raise ValueError('Unexpected validation reference shape')
    evaluate(root, runs, reference, pca, pca_path, args.output.resolve(), val_ids, args.smoke,
             args.workers, args.resume_from)


if __name__ == '__main__':
    main()
