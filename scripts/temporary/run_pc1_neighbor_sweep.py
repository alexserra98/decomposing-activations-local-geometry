"""PC1-only k=10,20,40 sensitivity at fixed 90th-percentile calibration."""
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
import run_pc1_support_coverage as base
from run_years_pc1_density import sha256, write_csv

NEIGHBORS = (10, 20, 40)
QUANTILE = .90
BASELINE = Path('outputs/experiments/pc1_support_coverage/job_1928981')
ANCHOR = Path('outputs/experiments/pc1_percentile_sweep/job_1932231')


def calibrate_neighbors(reference, queries):
    reference, queries = np.asarray(reference, dtype=np.float64), np.asarray(queries, dtype=np.float64)
    if (reference.ndim != 2 or queries.ndim != 2 or reference.shape[1] != queries.shape[1]
            or len(reference) < max(NEIGHBORS) or not len(queries)
            or not np.isfinite(reference).all() or not np.isfinite(queries).all()):
        raise ValueError('Invalid common reference/calibration matrices')
    distances = cdist(queries, reference)
    indices = np.array(NEIGHBORS)-1
    scores = np.partition(distances, indices, axis=1)[:, indices].T
    thresholds = np.quantile(scores, QUANTILE, axis=1, method='higher')
    if (thresholds <= 0).any() or (np.diff(scores, axis=0) < 0).any() or (np.diff(thresholds) < 0).any():
        raise ValueError('Invalid calibrated neighbor thresholds')
    return scores, thresholds


def recover_queries(previous, calibration):
    manifest = previous['baseline_manifest']
    path = Path(manifest['global_pca']['path'])
    if sha256(path) != previous['pca_sha256']:
        raise ValueError('Saved PCA fingerprint differs')
    pca = torch.load(path, weights_only=True, map_location='cpu')
    pool, provenance = base.reference_from_manifest(manifest, pca)
    if provenance['reference_sha256'] != previous['reference_provenance']['reference_sha256']:
        raise ValueError('Restored full evaluation pool differs, including calibration queries')
    positions = {row: index for index, row in enumerate(manifest['reference_global_rows'])}
    metadata = calibration['calibration']
    r_ids, q_ids = metadata['reference_global_rows'], metadata['calibration_global_rows']
    if set(r_ids)&set(q_ids) or set(r_ids)|set(q_ids) != set(positions):
        raise ValueError('Saved R/Q membership does not partition the evaluation pool')
    restored_r = pool[[positions[row] for row in r_ids]].numpy()
    queries = pool[[positions[row] for row in q_ids]].numpy()
    if not np.array_equal(restored_r, calibration['reference'].numpy()):
        raise ValueError('Reference order or coordinates differ')
    return queries, provenance


def initialize_worker(reference, thresholds):
    global REFERENCE, THRESHOLDS
    torch.set_num_threads(1)
    REFERENCE = np.asarray(reference, dtype=np.float64)
    THRESHOLDS = np.asarray(thresholds, dtype=np.float64)


def score_neighbors(center, vector, half_length, component):
    center, vector = np.asarray(center), np.asarray(vector)
    if (half_length <= 0 or not math.isfinite(half_length)
            or not np.isfinite(center).all() or not np.isfinite(vector).all()
            or not np.isclose(np.linalg.norm(vector), 1., atol=1e-10, rtol=1e-10)):
        raise ValueError('Invalid original projected PC1 segment')
    directions = np.vstack([vector, random_unit_directions(len(vector), 32, 42+component).numpy()])
    lines = [[base.supported_intervals(REFERENCE, center, direction, float(tau), half_length, k=k)
              for direction in directions] for k, tau in zip(NEIGHBORS, THRESHOLDS)]
    exact = np.array([[line[3] for line in group] for group in lines])
    history, previous, coarse = [], None, None
    for size in base.GRID_SIZES:
        offsets = np.linspace(-half_length, half_length, size)
        masks = np.stack([np.stack([base.grid_mask(line[0], line[1], offsets, k=k) for line in group]) for k, group in zip(NEIGHBORS, lines)])
        values = masks.mean(axis=2)
        if coarse is None:
            coarse = np.packbits(masks, axis=2)
        error = np.max(np.abs(values-exact), axis=1)
        change = np.max(np.abs(values-previous), axis=1) if previous is not None else None
        history.append(dict(points=size, max_exact_error=error.tolist(),
                            max_refinement_change=change.tolist() if change is not None else None))
        if previous is not None and max(error.max(), change.max()) <= base.GRID_TOLERANCE:
            break
        previous = values
    else:
        raise RuntimeError(f'Common grid failed for component {component}: {history}')
    indices = np.linspace(0, size-1, 5, dtype=int)
    queries = (center+directions[:, None, :]*offsets[indices][None, :, None]).reshape(-1, len(center))
    distances = cdist(queries, REFERENCE)
    indices_k = np.array(NEIGHBORS)-1
    radii = np.partition(distances, indices_k, axis=1)[:, indices_k].T.reshape(3, 33, 5)
    direct = radii <= THRESHOLDS[:, None, None]
    if not np.array_equal(direct, masks[:, :, indices]):
        raise ValueError('Masks differ from direct kNN distances')
    return dict(coverage_pc1=values[:, 0], coverage_random=values[:, 1:].mean(axis=1),
        delta_coverage=values[:, 0]-values[:, 1:].mean(axis=1),
        random_full_fraction=(values[:, 1:] == 1.).mean(axis=1), center_supported=masks[:, 0, size//2],
        direction_coverage=values, exact_coverage=exact, directions=directions,
        acceptance_mask_packed=np.packbits(masks, axis=2), mask_points=size,
        coarse_acceptance_mask_packed=coarse, coarse_mask_points=base.GRID_SIZES[0],
        max_grid_error=error, max_refinement_change=change, grid_history=history)


def validate_k20(result, saved):
    """Reproduce the previous q90/k20 direction scores and masks."""
    old_exact = saved['exact_coverage'][1].numpy()
    if not np.allclose(result['exact_coverage'][1], old_exact, rtol=0, atol=1e-12):
        raise ValueError('Original q90/k20 continuous scores differ')
    difference = float(np.max(np.abs(result['direction_coverage'][1]-saved['direction_coverage'][1].numpy())))
    bound = float(saved['max_grid_error'][1])+float(result['max_grid_error'][1])+1e-12
    if difference > bound:
        raise ValueError('k20 grid scores differ beyond numerical error bounds')
    if result['mask_points'] == saved['mask_points']:
        if not np.array_equal(result['acceptance_mask_packed'][1], saved['acceptance_mask_packed'][1].numpy()):
            raise ValueError('k20 masks differ on the original grid')
    return difference


def run_case(index, root, workers, smoke):
    dataset, _, layer, source, _ = SOURCES[index]
    old = BASELINE/f'{dataset}_l{layer:02}'
    previous = json.loads((old/'manifest.json').read_text())
    if not json.loads((old/'COMPLETED.json').read_text())['completed']:
        raise ValueError('Require completed PC1 source')
    calibration = torch.load(old/'calibration.pt', weights_only=True, map_location='cpu')
    if calibration['calibration'] != previous['calibration']:
        raise ValueError('Saved calibration metadata differs')
    reference = calibration['reference'].numpy()
    if hashlib.sha256(reference.tobytes()).hexdigest() != previous['reference_sha256']:
        raise ValueError('Reference fingerprint differs')
    queries, provenance = recover_queries(previous, calibration)
    scores, thresholds = calibrate_neighbors(reference, queries)
    if not np.array_equal(scores[1], calibration['calibration_scores'].numpy()):
        raise ValueError('Recomputed k20 calibration distances differ')
    anchor = ANCHOR/f'{dataset}_l{layer:02}'
    anchor_protocol = json.loads((anchor/'manifest.json').read_text())
    if (not json.loads((anchor/'COMPLETED.json').read_text())['completed']
            or anchor_protocol['quantiles'] != [.80, .90, .95]
            or thresholds[1] != anchor_protocol['thresholds'][1]):
        raise ValueError('q90/k20 anchor calibration differs')
    output = root/f'{dataset}_l{layer:02}'
    output.mkdir(parents=True)
    protocol = dict(metric='PC1 neighbor sensitivity at fixed q90; no Top5 averaging', neighbors=NEIGHBORS, quantile=QUANTILE,
        thresholds=thresholds.tolist(), calibration_source=str(old), calibration_sha256=sha256(old/'calibration.pt'),
        pc1_manifest_sha256=sha256(old/'manifest.json'), pc1_protocol=previous,
        reference_sha256=previous['reference_sha256'], query_sha256=hashlib.sha256(queries.tobytes()).hexdigest(),
        calibration_acceptance=[float(np.mean(row <= tau)) for row, tau in zip(scores, thresholds)],
        quantile_order_statistic_1based=math.ceil(QUANTILE*(scores.shape[1]-1))+1,
        reconstruction_provenance=provenance, anchor_source=str(anchor), anchor_manifest_sha256=sha256(anchor/'manifest.json'),
        source_sha256=sha256(Path(__file__)), scorer_sha256=sha256(Path(base.__file__)),
        grid_sizes=base.GRID_SIZES, grid_tolerance=base.GRID_TOLERANCE,
        grid_rule='common finer passing grid across all neighbor counts and all 33 directions within a component',
        original_pc1_eligibility=True, random_directions=32, seed='42+component',
        smoke=smoke, job_id=os.environ['SLURM_JOB_ID'], geometry_bundle_sha256={})
    (output/'manifest.json').write_text(json.dumps(protocol, indent=2))
    (output/'runner_at_compute.py').write_bytes(Path(__file__).read_bytes())
    torch.save(dict(reference=torch.from_numpy(reference), queries=torch.from_numpy(queries),
        calibration_scores=torch.from_numpy(scores), thresholds=torch.from_numpy(thresholds),
        neighbors=NEIGHBORS, quantile=QUANTILE, membership=calibration['calibration']), output/'calibration.pt')
    rows, summaries, cache, differences = [], [], {}, []
    configs = read_csv(source/'summary.csv')
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                             initializer=initialize_worker, initargs=(reference, thresholds)) as executor:
        for config in configs[:1] if smoke else configs:
            run_id = config['run_id']
            bundle_path = source/(run_id+'.pt')
            if sha256(bundle_path) != previous['geometry_bundle_sha256'][run_id]:
                raise ValueError('Saved PC1 geometry fingerprint differs')
            geom_bundle = torch.load(bundle_path, weights_only=True, map_location='cpu')
            geom = geom_bundle['projected_geometry']
            old_results = torch.load(anchor/(run_id+'.pt'), weights_only=True, map_location='cpu')['components']
            eligible = sorted(geom_bundle['components'])[:2] if smoke else sorted(geom_bundle['components'])
            if not smoke and set(eligible) != set(old_results):
                raise ValueError('Original PC1 eligibility differs')
            signature = hashlib.sha256(b''.join(geom[key].contiguous().numpy().tobytes()
                for key in ['centers', 'vectors', 'half_lengths'])+str(eligible).encode()).hexdigest()
            results = cache.get(signature)
            if results is None:
                futures = {i: executor.submit(score_neighbors, geom['centers'][i].numpy(), geom['vectors'][i].numpy(),
                    float(geom['half_lengths'][i]), i) for i in eligible}
                results = {i: futures[i].result() for i in eligible}
                cache[signature] = results
            for i, result in results.items():
                differences.append(validate_k20(result, old_results[i]))
                for qi, neighbors in enumerate(NEIGHBORS):
                    row = dict(run_id=run_id, component=i, kind=config['kind'], K=int(config['K']),
                        threshold=float(config['threshold']), neighbors=neighbors, quantile=QUANTILE, tau=float(thresholds[qi]),
                        half_length=float(geom['half_lengths'][i]), alpha=float(geom['alpha'][i]), mask_points=result['mask_points'])
                    for field in ['coverage_pc1', 'coverage_random', 'delta_coverage', 'random_full_fraction',
                                  'center_supported', 'max_grid_error', 'max_refinement_change']:
                        row[field] = result[field][qi].item()
                    rows.append(row)
            torch.save(dict(run_id=run_id, neighbors=NEIGHBORS, quantile=QUANTILE, thresholds=torch.from_numpy(thresholds),
                components={i: base.as_tensors(result) for i, result in results.items()}), output/(run_id+'.pt'))
            protocol['geometry_bundle_sha256'][run_id+'.pt'] = sha256(bundle_path)
            (output/'manifest.json').write_text(json.dumps(protocol, indent=2))
            for neighbors in NEIGHBORS:
                valid = [row for row in rows if row['run_id'] == run_id and row['neighbors'] == neighbors]
                summary = dict(run_id=run_id, kind=config['kind'], K=int(config['K']), threshold=float(config['threshold']),
                    neighbors=neighbors, quantile=QUANTILE, eligible=len(valid), total=int(config['K']))
                for field in ['coverage_pc1', 'coverage_random', 'delta_coverage']:
                    values = np.array([row[field] for row in valid])
                    for name, value in [('mean', values.mean()), ('median', np.median(values)),
                                        ('q25', np.quantile(values, .25)), ('q75', np.quantile(values, .75))]:
                        summary[name+'_'+field] = float(value)
                summary['fraction_full_pc1'] = float(np.mean([row['coverage_pc1'] == 1. for row in valid]))
                summary['fraction_full_random_mean'] = float(np.mean([row['coverage_random'] == 1. for row in valid]))
                summary['fraction_full_random_directions'] = float(np.mean([row['random_full_fraction'] for row in valid]))
                summaries.append(summary)
            write_csv(output/'components.csv', rows)
            write_csv(output/'summary.csv', summaries)
            print(f'{dataset} l{layer} {run_id}: {len(eligible)} PC1 components, all three neighbor counts', flush=True)
    expected = len(read_csv(old/'components.csv'))*len(NEIGHBORS)
    if not smoke and (len(summaries) != 48 or len(rows) != expected):
        raise ValueError('Incomplete neighbor experiment')
    write_csv(output/'exclusions.csv', [r for r in read_csv(source/'components.csv') if r['exclusion_reason']])
    validation = dict(passed=True, neighbors=NEIGHBORS, quantile=QUANTILE, summaries=len(summaries), component_neighbor_rows=len(rows),
        max_k20_direction_difference=max(differences), max_grid_error=max(row['max_grid_error'] for row in rows),
        max_refinement_change=max(row['max_refinement_change'] for row in rows), separately_recalibrated=True,
        original_pc1_population=True, direct_audit='five positions, all 33 directions, every unique component and neighbor count')
    (output/'validation.json').write_text(json.dumps(validation, indent=2))
    (output/'COMPLETED.json').write_text(json.dumps(dict(completed=not smoke, smoke=smoke, **validation)))


def render(entries, output, aggregate, neighbors):
    import matplotlib.pyplot as plt
    fig = panel_tables(entries, score_formatter=lambda x: f'{100*x:.2f}'.replace('.', ','))
    fig._suptitle.set_text(f'PC1 · Calibrazione 90° percentile · k={neighbors} · {"Medie" if aggregate == "mean" else "Mediane"} tra componenti')
    for ax, entry in zip(fig.axes, entries):
        table = ax.tables[0]
        table.set_fontsize(11)
        table[0, 2].get_text().set_text('Copertura PC1 (%) ↑')
        table[0, 3].get_text().set_text('Δ copertura (pp) ↑')
        ordered = sorted(entry['selected'], key=lambda row: (row['K'], row['kind']))
        for start in [1, 3]:
            for column, field in [(2, 'retention'), (3, 'enrichment')]:
                if ordered[start-1][field] == ordered[start][field]:
                    for index in [start, start+1]:
                        table[index, column].set_text_props(weight='normal', color='#23364A')
    fig.texts[-4].set_text(f'Sola PC1 · k={neighbors} · percentile 90% · rango 0,1 · stessi R/Q, geometrie, componenti e controlli casuali')
    fig.texts[-3].set_text('Copertura = frazione di posizioni accettate. Δ = differenza per componente rispetto alla media di 32 direzioni casuali.')
    fig.texts[-2].set_text('Grassetto: valore maggiore entro esperimento e K; nessun test di significatività. Valutabili: componenti incluse / totali.')
    fig.texts[-1].set_text('Griglia comune tra valori di k · stessi segmenti PC1 originali · diagnostica descrittiva sul validation set ricostruito.')
    name = '03_pannello_medie' if aggregate == 'mean' else '02_pannello_tabelle'
    for extension in ['png', 'pdf', 'svg']:
        fig.savefig(output/f'{name}.{extension}', dpi=200, facecolor='white')
    plt.close(fig)


def collect(root):
    combined, selected, protocols, rows_by_case = [], [], {}, {}
    for dataset, label, layer, _, _ in SOURCES:
        case = root/f'{dataset}_l{layer:02}'
        if not json.loads((case/'COMPLETED.json').read_text())['completed']:
            raise ValueError('Incomplete case')
        if not json.loads((case/'validation.json').read_text())['passed']:
            raise ValueError('Unvalidated case')
        protocols[(dataset, layer)] = json.loads((case/'manifest.json').read_text())
        summaries, components = read_csv(case/'summary.csv'), read_csv(case/'components.csv')
        if len(summaries) != 48:
            raise ValueError('Incomplete sweep')
        for summary in summaries:
            matched = [r for r in components if r['run_id'] == summary['run_id'] and r['neighbors'] == summary['neighbors']]
            if len(matched) != int(summary['eligible']):
                raise ValueError('Component count mismatch')
            for field in ['coverage_pc1', 'coverage_random', 'delta_coverage']:
                values = [float(r[field]) for r in matched]
                for name, function in [('mean', np.mean), ('median', np.median)]:
                    if not math.isclose(float(summary[name+'_'+field]), float(function(values)), abs_tol=1e-12):
                        raise ValueError('Table aggregate mismatch')
            row = dict(dataset=dataset, layer=layer, **summary)
            combined.append(row)
            if float(summary['threshold']) == .1:
                selected.append(row)
        rows_by_case[(dataset, layer)] = summaries
    for neighbors in NEIGHBORS:
        output = root/f'k{neighbors}'
        output.mkdir(exist_ok=True)
        for aggregate in ['median', 'mean']:
            entries = []
            for dataset, label, layer, _, _ in SOURCES:
                baseline = protocols[(dataset, layer)]['pc1_protocol']['baseline_manifest']
                subset = [r for r in rows_by_case[(dataset, layer)] if int(r['neighbors']) == neighbors and float(r['threshold']) == .1]
                entries.append(dict(label=label, layer=layer, dimension=baseline['dimension'],
                    variance=baseline['global_pca']['retained_variance_fraction'], selected=[dict(
                        K=int(r['K']), kind=r['kind'], method=NAMES[r['kind']],
                        retention=float(r[aggregate+'_coverage_pc1']), enrichment=float(r[aggregate+'_delta_coverage']),
                        eligible=int(r['eligible']), total=int(r['total'])) for r in subset]))
            render(entries, output, aggregate, neighbors)
        write_csv(output/'tabella_soglia_01.csv', [r for r in selected if int(r['neighbors']) == neighbors])
    write_csv(root/'tutte_le_192_configurazioni.csv', combined)
    write_csv(root/'tabelle_soglia_01.csv', selected)
    plot_sensitivity(root, selected)
    (root/'export_validation.json').write_text(json.dumps(dict(passed=True, configurations=192, selected_rows=48)))
    (root/'COMPLETED.json').write_text(json.dumps(dict(completed=True, experiments=4, neighbors=NEIGHBORS, quantile=QUANTILE, configurations=192)))


def plot_sensitivity(root, selected):
    import matplotlib.pyplot as plt
    colors = {'hddc': '#007F86', 'kmeans': '#D67D24'}
    fig, axes = plt.subplots(2, 4, figsize=(19, 8), sharex=True)
    for column, (dataset, label, layer, _, _) in enumerate(SOURCES):
        for kind in ['hddc', 'kmeans']:
            for k in [50, 100]:
                rows = sorted([r for r in selected if r['dataset'] == dataset and r['layer'] == layer
                    and r['kind'] == kind and int(r['K']) == k], key=lambda r: int(r['neighbors']))
                x = [int(r['neighbors']) for r in rows]
                for axis, field in [(axes[0, column], 'mean_coverage_pc1'), (axes[1, column], 'mean_delta_coverage')]:
                    axis.plot(x, [100*float(r[field]) for r in rows], color=colors[kind], marker='o' if k == 50 else 's',
                        linestyle='-' if k == 50 else '--', label=f'{NAMES[kind]} K={k}')
                axes[0, column].plot(x, [100*float(r['mean_coverage_random']) for r in rows], color=colors[kind],
                    linestyle=':', linewidth=1., alpha=.5, marker='o' if k == 50 else 's', markersize=3)
        axes[0, column].set_title(f'{label} · layer {layer}')
        axes[1, column].axhline(0, color='#777777', linewidth=.8)
        axes[1, column].set_xlabel('Numero di vicini k')
        for row in [0, 1]:
            axes[row, column].set_xticks([10, 20, 40])
            axes[row, column].grid(alpha=.2)
    axes[0, 0].set_ylabel('Copertura media (%)')
    axes[1, 0].set_ylabel('Δ copertura media (pp)')
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Sensibilità ai vicini · percentile 90% · sola PC1 · puntinato: copertura casuale · popolazioni e geometrie fisse')
    fig.tight_layout(rect=(0, 0, 1, .95))
    for extension in ['png', 'pdf', 'svg']:
        fig.savefig(root/f'01_sensibilita_vicini.{extension}', dpi=200, facecolor='white')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--experiment-index', type=int, choices=range(4))
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--collect-only', action='store_true')
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
        parser.error('Specify case or collect-only')


if __name__ == '__main__':
    main()
