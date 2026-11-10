"""Calibrated kNN support coverage on the four saved PCA75 PC1 geometries."""
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

from dalg.analysis.pc1_density import random_unit_directions
from export_pc1_density_overview import SOURCES, NAMES, panel_tables, read_csv
from run_pc1_pak_controlled import reference_from_manifest
from run_years_pc1_density import sha256, write_csv

K = 20
SEED = 42
GRID_SIZES = (2001, 4001, 8001, 16001)
GRID_TOLERANCE = .002


def split_and_calibrate(points, original_rows, k=K):
    """Split canonical row identities before calibrating independent queries."""
    points = np.asarray(points, dtype=np.float64)
    original_rows = np.asarray(original_rows, dtype=np.int64)
    if points.ndim != 2 or len(points) != len(original_rows) or not np.isfinite(points).all():
        raise ValueError('Invalid evaluation pool')
    if len(set(original_rows.tolist())) != len(points):
        raise ValueError('Evaluation row identities must be unique')
    canonical = np.argsort(original_rows)
    permutation = torch.randperm(len(points), generator=torch.Generator().manual_seed(SEED)).numpy()
    q_count = math.ceil(.2*len(points))
    q_indices, r_indices = canonical[permutation[:q_count]], canonical[permutation[q_count:]]
    if not len(q_indices) or len(r_indices) < k:
        raise ValueError('Insufficient reference or calibration population')
    reference, calibration = points[r_indices], points[q_indices]
    distances = cdist(calibration, reference)
    scores = np.partition(distances, k-1, axis=1)[:, k-1]
    tau = float(np.quantile(scores, .95, method='higher'))
    if not math.isfinite(tau) or tau <= 0:
        raise ValueError('Calibration threshold must be positive and finite')
    metadata = dict(reference_global_rows=original_rows[r_indices].tolist(),
        calibration_global_rows=original_rows[q_indices].tolist(), seed=SEED,
        reference_count=len(reference), calibration_count=len(calibration),
        k=k, quantile=.95, quantile_method='higher',
        threshold_order_statistic_1based=math.ceil(.95*(len(scores)-1))+1, tau=tau,
        calibration_acceptance=float(np.mean(scores <= tau)))
    return reference, scores, metadata


def supported_intervals(reference, center, direction, tau, half_length, k=K):
    """Closed intervals where at least k radius-tau reference balls overlap.

    d_k(x,R)<=tau iff at least k reference points are within tau of x.
    Each ball intersects a unit-direction line in an interval; grouped endpoint
    events give the supported positive-length intervals. Tangencies have zero
    continuous length but remain in starts/ends for endpoint-inclusive masks.
    """
    residual = reference-center
    dots = residual@direction
    norm2 = np.einsum('ij,ij->i', residual, residual)
    perpendicular2 = norm2-dots*dots
    roundoff = 64*np.finfo(np.float64).eps*np.maximum(1., norm2)
    if np.any(perpendicular2 < -roundoff):
        raise ValueError('Perpendicular squared distance is negative beyond roundoff')
    perpendicular2 = np.maximum(0., perpendicular2)
    discriminant = tau*tau-perpendicular2
    hit = discriminant >= 0
    radius = np.sqrt(discriminant[hit])
    starts = np.maximum(-half_length, dots[hit]-radius)
    ends = np.minimum(half_length, dots[hit]+radius)
    active = starts <= ends
    starts, ends = starts[active], ends[active]
    positions, inverse = np.unique(np.r_[starts, ends], return_inverse=True)
    changes = np.zeros(len(positions), dtype=np.int64)
    np.add.at(changes, inverse[:len(starts)], 1)
    np.add.at(changes, inverse[len(starts):], -1)
    counts = np.cumsum(changes)
    accepted = counts[:-1] >= k
    intervals = np.column_stack([positions[:-1][accepted], positions[1:][accepted]])
    exact = float(np.diff(intervals, axis=1).sum()/(2*half_length))
    return np.sort(starts), np.sort(ends), intervals, exact


def grid_mask(starts, ends, offsets, k=K):
    counts = np.searchsorted(starts, offsets, side='right')-np.searchsorted(ends, offsets, side='left')
    return counts >= k


def initialize_worker(reference, tau):
    global REFERENCE, TAU
    torch.set_num_threads(1)
    REFERENCE = np.asarray(reference, dtype=np.float64)
    TAU = tau


def score_component(center, vector, half_length, component):
    center, vector = np.asarray(center), np.asarray(vector)
    if (not math.isfinite(half_length) or half_length <= 0
            or not np.isfinite(center).all() or not np.isfinite(vector).all()
            or not np.isclose(np.linalg.norm(vector), 1., atol=1e-10, rtol=1e-10)):
        raise ValueError('Invalid projected segment geometry')
    directions = np.vstack([vector, random_unit_directions(len(vector), 32, SEED+component).numpy()])
    lines = [supported_intervals(REFERENCE, center, direction, TAU, half_length) for direction in directions]
    exact = np.array([line[3] for line in lines])
    history = []
    previous = None
    coarse_mask = None
    for size in GRID_SIZES:
        offsets = np.linspace(-half_length, half_length, size)
        masks = np.stack([grid_mask(line[0], line[1], offsets) for line in lines])
        coverage = masks.mean(axis=1)
        if coarse_mask is None:
            coarse_mask = np.packbits(masks, axis=1)
        error = float(np.max(np.abs(coverage-exact)))
        change = float(np.max(np.abs(coverage-previous))) if previous is not None else None
        history.append(dict(points=size, max_exact_error=error, max_refinement_change=change))
        if previous is not None and max(error, change) <= GRID_TOLERANCE:
            break
        previous = coverage
    else:
        raise RuntimeError(f'Grid did not resolve component {component}: {history}')
    # Audit the interval optimization against direct distances at five positions
    # in all 33 directions for every component, including both endpoints/center.
    indices = np.linspace(0, size-1, 5, dtype=int)
    queries = (center+directions[:, None, :]*offsets[indices][None, :, None]).reshape(-1, len(center))
    direct = (cdist(queries, REFERENCE) <= TAU).sum(axis=1).reshape(33, 5) >= K
    if not np.array_equal(direct, masks[:, indices]):
        raise ValueError('Interval masks disagree with direct neighbor distances')
    return dict(coverage_pc1=float(coverage[0]), coverage_random=float(coverage[1:].mean()),
        delta_coverage=float(coverage[0]-coverage[1:].mean()),
        center_supported=bool(masks[0, size//2]), exact_coverage=exact,
        direction_coverage=coverage, directions=directions,
        acceptance_mask_packed=np.packbits(masks, axis=1), mask_points=size,
        coarse_acceptance_mask_packed=coarse_mask, coarse_mask_points=GRID_SIZES[0],
        grid_history=history, max_grid_error=error, max_refinement_change=change,
        supported_intervals_pc1=lines[0][2])


def as_tensors(result):
    return {key: torch.from_numpy(value) if isinstance(value, np.ndarray) else value
            for key, value in result.items()}


def run_case(index, root, workers, smoke=False):
    dataset, label, layer, source, _ = SOURCES[index]
    output = root/f'{dataset}_l{layer:02}'
    output.mkdir(parents=True)
    baseline = json.loads((source/'manifest.json').read_text())
    completed = json.loads((source/'COMPLETED.json').read_text())
    if not completed['completed'] or completed['smoke']:
        raise ValueError('Require a complete source geometry experiment')
    pca_path = Path(baseline['global_pca']['path'])
    if sha256(pca_path) != baseline['global_pca']['sha256']:
        raise ValueError('Saved PCA fingerprint mismatch')
    pca = torch.load(pca_path, weights_only=True, map_location='cpu')
    pool, provenance = reference_from_manifest(baseline, pca)
    reference, calibration_scores, calibration = split_and_calibrate(pool.numpy(), baseline['reference_global_rows'])
    protocol = dict(metric='kNN calibrated support coverage', baseline_manifest=baseline,
        baseline_manifest_sha256=sha256(source/'manifest.json'), pca_sha256=sha256(pca_path),
        source_sha256=sha256(Path(__file__)), reference_provenance=provenance,
        calibration=calibration, reference_sha256=hashlib.sha256(reference.tobytes()).hexdigest(),
        grid_sizes=GRID_SIZES, grid_tolerance=GRID_TOLERANCE, random_directions=32,
        random_seed='42 + component_id', interpretation='descriptive validation diagnostic',
        aggregation='ordinary per-component median; delta median computed from component deltas',
        smoke=smoke, job_id=os.environ['SLURM_JOB_ID'], geometry_bundle_sha256={})
    (output/'manifest.json').write_text(json.dumps(protocol, indent=2))
    torch.save(dict(reference=torch.from_numpy(reference), calibration_scores=torch.from_numpy(calibration_scores),
                    calibration=calibration), output/'calibration.pt')
    all_rows, summaries, cache = [], [], {}
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                             initializer=initialize_worker, initargs=(reference, calibration['tau'])) as executor:
        configurations = read_csv(source/'summary.csv')
        for config in configurations[:1] if smoke else configurations:
            run_id = config['run_id']
            bundle_path = source/(run_id+'.pt')
            bundle = torch.load(bundle_path, weights_only=True, map_location='cpu')
            geom = bundle['projected_geometry']
            eligible = sorted(bundle['components'])[:2] if smoke else sorted(bundle['components'])
            signature = hashlib.sha256(b''.join(geom[key].contiguous().numpy().tobytes()
                for key in ['centers', 'vectors', 'half_lengths'])+str(eligible).encode()).hexdigest()
            results = cache.get(signature)
            if results is None:
                futures = {i: executor.submit(score_component, geom['centers'][i].numpy(), geom['vectors'][i].numpy(),
                    float(geom['half_lengths'][i]), i) for i in eligible}
                results = {i: as_tensors(futures[i].result()) for i in eligible}
                cache[signature] = results
            for i in eligible:
                result = results[i]
                all_rows.append(dict(run_id=run_id, component=i, kind=config['kind'], K=int(config['K']),
                    threshold=float(config['threshold']), alpha=float(geom['alpha'][i]),
                    half_length=float(geom['half_lengths'][i]),
                    original_half_length=float(geom['original_half_lengths'][i]),
                    **{key: result[key] for key in ['coverage_pc1', 'coverage_random', 'delta_coverage',
                        'center_supported', 'mask_points', 'max_grid_error', 'max_refinement_change']}))
            protocol['geometry_bundle_sha256'][run_id] = sha256(bundle_path)
            (output/'manifest.json').write_text(json.dumps(protocol, indent=2))
            torch.save(dict(run_id=run_id, components=results), output/(run_id+'.pt'))
            valid = [r for r in all_rows if r['run_id'] == run_id]
            summary = dict(run_id=run_id, kind=config['kind'], K=int(config['K']), threshold=float(config['threshold']),
                eligible=len(valid), total=int(config['K']))
            for field in ['coverage_pc1', 'coverage_random', 'delta_coverage']:
                values = np.array([r[field] for r in valid])
                for statistic, value in [('median', np.median(values)), ('mean', np.mean(values)),
                                          ('q25', np.quantile(values, .25)), ('q75', np.quantile(values, .75))]:
                    summary[f'{statistic}_{field}'] = float(value)
            summary['supported_centers_fraction'] = float(np.mean([r['center_supported'] for r in valid]))
            summaries.append(summary)
            write_csv(output/'components.csv', all_rows)
            write_csv(output/'summary.csv', summaries)
            print(f'{dataset} l{layer} {run_id}: coverage={summary["median_coverage_pc1"]:.4f}, '
                  f'delta={summary["median_delta_coverage"]:.4f}', flush=True)
    if smoke:
        (output/'COMPLETED.json').write_text(json.dumps(dict(completed=False, smoke=True, components=len(all_rows))))
        return
    source_rows = read_csv(source/'components.csv')
    if len(summaries) != 16 or len(all_rows) != sum(not row['exclusion_reason'] for row in source_rows):
        raise ValueError('Incomplete geometry sweep')
    write_csv(output/'exclusions.csv', [r for r in source_rows if r['exclusion_reason']])
    validation = dict(passed=True, components=len(all_rows), unique_geometries=len(cache),
        max_grid_error=max(r['max_grid_error'] for r in all_rows),
        max_refinement_change=max(r['max_refinement_change'] for r in all_rows),
        direct_distance_audit='five positions in all 33 directions for each unique scored component')
    (output/'validation.json').write_text(json.dumps(validation, indent=2))
    (output/'COMPLETED.json').write_text(json.dumps(dict(completed=True, configurations=16, **validation)))


def collect(root):
    entries, mean_entries, selected, combined = [], [], [], []
    shared_splits = {}
    for dataset, label, layer, _, _ in SOURCES:
        source = root/f'{dataset}_l{layer:02}'
        if not json.loads((source/'COMPLETED.json').read_text())['completed']:
            raise ValueError('Incomplete case')
        if not json.loads((source/'validation.json').read_text())['passed']:
            raise ValueError('Unvalidated case')
        protocol = json.loads((source/'manifest.json').read_text())
        baseline, calibration = protocol['baseline_manifest'], protocol['calibration']
        split = (calibration['reference_global_rows'], calibration['calibration_global_rows'])
        if dataset in shared_splits and split != shared_splits[dataset]:
            raise ValueError('Layer reference/calibration splits differ')
        shared_splits[dataset] = split
        components = read_csv(source/'components.csv')
        mean_entry = dict(label=label, layer=layer, dimension=baseline['dimension'],
            variance=baseline['global_pca']['retained_variance_fraction'], selected=[])
        entry = dict(label=label, layer=layer, dimension=baseline['dimension'],
            variance=baseline['global_pca']['retained_variance_fraction'], selected=[])
        summaries = read_csv(source/'summary.csv')
        if len(summaries) != 16:
            raise ValueError('Incomplete sweep')
        for summary in summaries:
            rows = [r for r in components if r['run_id'] == summary['run_id']]
            if len(rows) != int(summary['eligible']):
                raise ValueError('Summary count mismatch')
            for field in ['coverage_pc1', 'coverage_random', 'delta_coverage']:
                values = [float(r[field]) for r in rows]
                for statistic, expected in [('median', np.median(values)), ('mean', np.mean(values))]:
                    if not math.isclose(float(expected), float(summary[statistic+'_'+field]), abs_tol=1e-12):
                        raise ValueError('Table aggregate mismatch')
            combined.append(dict(dataset=dataset, layer=layer, **summary))
            if float(summary['threshold']) != .1:
                continue
            row = dict(dataset=dataset, layer=layer, K=int(summary['K']), kind=summary['kind'], method=NAMES[summary['kind']],
                retention=float(summary['median_coverage_pc1']), enrichment=float(summary['median_delta_coverage']),
                eligible=int(summary['eligible']), total=int(summary['total']))
            entry['selected'].append(row)
            mean_entry['selected'].append(dict(row, retention=float(summary['mean_coverage_pc1']),
                                                enrichment=float(summary['mean_delta_coverage'])))
            selected.append(dict(dataset=dataset, layer=layer, K=row['K'], method=row['method'],
                coverage_pc1=row['retention'], delta_coverage=row['enrichment'], eligible=row['eligible'], total=row['total']))
        entries.append(entry)
        mean_entries.append(mean_entry)
    render_tables(entries, root, '02_pannello_tabelle', 'Mediane')
    render_tables(mean_entries, root, '03_pannello_medie', 'Medie')
    write_csv(root/'tabella_soglia_01.csv', selected)
    write_csv(root/'tutte_le_64_configurazioni.csv', combined)
    (root/'export_validation.json').write_text(json.dumps(dict(passed=True, configurations=64, displayed_rows=16, aggregates=['median', 'mean'])))
    (root/'COMPLETED.json').write_text(json.dumps(dict(completed=True, experiments=4, configurations=64, displayed_rows=16)))
    print(root.resolve(), flush=True)


def render_tables(entries, root, name, aggregation):
    import matplotlib.pyplot as plt
    fig = panel_tables(entries, score_formatter=lambda x: f'{100*x:.2f}'.replace('.', ','))
    fig._suptitle.set_text(f'Supporto kNN calibrato · {aggregation} di copertura PC1 e Δ rispetto al caso casuale')
    for ax, entry in zip(fig.axes, entries):
        table = ax.tables[0]
        ordered = sorted(entry['selected'], key=lambda row: (row['K'], row['kind']))
        for start in [1, 3]:
            for column, field in [(2, 'retention'), (3, 'enrichment')]:
                if ordered[start-1][field] == ordered[start][field]:
                    for row_index in [start, start+1]:
                        table[row_index, column].set_text_props(weight='normal', color='#23364A')
        table[0, 2].get_text().set_text('Copertura PC1 (%) ↑')
        table[0, 3].get_text().set_text('Δ copertura (pp) ↑')
        table.set_fontsize(11)
    fig.texts[-1].set_text('Diagnostica sul validation set ricostruito · PCA75 e segmenti originali · nessuna stima di densità o angolo tangente.')
    fig.texts[-2].set_text('Grassetto: valore maggiore entro esperimento e K; nessun test di significatività. Valutabili: componenti incluse / totali.')
    fig.texts[-3].set_text('Copertura = frazione di posizioni accettate. Δ = differenza per componente rispetto alla media di 32 direzioni casuali.')
    fig.texts[-4].set_text(f'Soglia di rango 0,1 · k=20 · calibrazione al 95° percentile su Q separato da R · {aggregation.lower()} in scala ordinaria')
    for extension in ['png', 'pdf', 'svg']:
        fig.savefig(root/f'{name}.{extension}', dpi=200, facecolor='white')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--experiment-index', type=int, choices=range(4))
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--collect-only', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_STEP_ID') in [None, 'batch', 'extern']:
        raise RuntimeError('Run numerical work via srun on a compute node')
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    if args.collect_only:
        collect(args.output)
    elif args.experiment_index is not None:
        run_case(args.experiment_index, args.output, args.workers, args.smoke)
    else:
        parser.error('Specify --experiment-index or --collect-only')


if __name__ == '__main__':
    main()
