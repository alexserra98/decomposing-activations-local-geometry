"""PC1 central-mass 90/95/99% crossed with calibration percentiles 80/90/95%."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
import multiprocessing
import os
from statistics import NormalDist
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
import torch

from dalg.analysis.pc1_density import random_unit_directions
from export_pc1_density_overview import SOURCES, NAMES, panel_tables, read_csv
import run_pc1_support_coverage as base
import run_pc1_percentile_sweep as percentile
from run_years_pc1_density import sha256, write_csv

QUANTILES = (.80, .90, .95)
MASSES = (.90, .95, .99)
BASELINE = Path('outputs/experiments/pc1_support_coverage/job_1928981')
ANCHOR = Path('outputs/experiments/pc1_percentile_sweep/job_1932231')


def length_factor(mass):
    if not 0 < mass < 1:
        raise ValueError('Central Gaussian probability must be in (0,1)')
    return NormalDist().inv_cdf((1+mass)/2)/NormalDist().inv_cdf(.9)


def score_lengths(center, vector, half_length80, component):
    all_masses = (.80,)+MASSES
    lengths = np.array([half_length80*length_factor(mass) for mass in all_masses])
    parts = [percentile.score_percentiles(center, vector, float(length), component) for length in lengths]
    exact = np.stack([part['exact_coverage'] for part in parts])
    accepted = 2*lengths[:, None, None]*exact
    unsupported = 2*lengths[:, None, None]-accepted
    tolerance = 1e-10*max(1., lengths.max())
    if (np.diff(accepted, axis=0) < -tolerance).any() or (np.diff(unsupported, axis=0) < -tolerance).any():
        raise ValueError('Physical accepted/unsupported lengths decreased on nested segments')
    centers = np.stack([part['center_supported'] for part in parts])
    if not np.all(centers == centers[0]):
        raise ValueError('Center support changed when only segment extent changed')
    if any(not np.array_equal(part['directions'], parts[0]['directions']) for part in parts[1:]):
        raise ValueError('Direction controls changed with segment mass')
    return dict(control=parts[0], by_mass=parts[1:])


def validate_control(control, saved):
    if not np.allclose(control['exact_coverage'], saved['exact_coverage'].numpy(), rtol=0, atol=1e-12):
        raise ValueError('80%-mass continuous reproduction control differs')
    differences = np.max(np.abs(control['direction_coverage']-saved['direction_coverage'].numpy()), axis=1)
    bound = control['max_grid_error']+saved['max_grid_error'].numpy()+1e-12
    if np.any(differences > bound):
        raise ValueError('80%-mass sampled reproduction differs beyond discretization bounds')
    if control['mask_points'] == saved['mask_points']:
        if not np.array_equal(control['acceptance_mask_packed'], saved['acceptance_mask_packed'].numpy()):
            raise ValueError('80%-mass control masks differ on the same grid')
    return float(differences.max())


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
    scores = calibration['calibration_scores'].numpy()
    thresholds = percentile.percentile_thresholds(scores)
    if thresholds[-1] != previous['calibration']['tau']:
        raise ValueError('q95 calibration threshold differs')
    if previous['baseline_manifest']['probability'] != .80:
        raise ValueError('Saved base geometry is not the central-80% marginal segment')
    anchor = ANCHOR/f'{dataset}_l{layer:02}'
    anchor_protocol = json.loads((anchor/'manifest.json').read_text())
    if (not json.loads((anchor/'COMPLETED.json').read_text())['completed']
            or not np.array_equal(thresholds, anchor_protocol['thresholds'])):
        raise ValueError('Percentile thresholds differ from the prior PC1 sweep')
    output = root/f'{dataset}_l{layer:02}'
    output.mkdir(parents=True)
    protocol = dict(metric='PC1 segment-mass by calibration-percentile factorial; k20 fixed', masses=MASSES, quantiles=QUANTILES,
        length_factors=[length_factor(mass) for mass in MASSES],
        anchor_source=str(anchor), anchor_manifest_sha256=sha256(anchor/'manifest.json'),
        thresholds=thresholds.tolist(), calibration_source=str(old), calibration_sha256=sha256(old/'calibration.pt'),
        pc1_manifest_sha256=sha256(old/'manifest.json'), pc1_protocol=previous,
        calibration_acceptance=[float(np.mean(scores <= tau)) for tau in thresholds],
        quantile_order_statistics_1based=[math.ceil(q*(len(scores)-1))+1 for q in QUANTILES],
        source_sha256=sha256(Path(__file__)), scorer_sha256=sha256(Path(percentile.__file__)), base_scorer_sha256=sha256(Path(base.__file__)),
        grid_sizes=base.GRID_SIZES, grid_tolerance=base.GRID_TOLERANCE,
        grid_rule='common finer passing grid across all percentiles and all 33 directions within each component and segment mass',
        original_pc1_eligibility=True, random_directions=32, seed='42+component',
        smoke=smoke, job_id=os.environ['SLURM_JOB_ID'], geometry_bundle_sha256={})
    (output/'manifest.json').write_text(json.dumps(protocol, indent=2))
    (output/'runner_at_compute.py').write_bytes(Path(__file__).read_bytes())
    rows, summaries, cache, differences = [], [], {}, []
    configs = read_csv(source/'summary.csv')
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                             initializer=percentile.initialize_worker, initargs=(reference, thresholds)) as executor:
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
                futures = {i: executor.submit(score_lengths, geom['centers'][i].numpy(), geom['vectors'][i].numpy(),
                    float(geom['half_lengths'][i]), i) for i in eligible}
                results = {i: futures[i].result() for i in eligible}
                cache[signature] = results
            for i, result in results.items():
                differences.append(validate_control(result['control'], old_results[i]))
                for mi, mass in enumerate(MASSES):
                    part = result['by_mass'][mi]
                    for qi, quantile in enumerate(QUANTILES):
                        row = dict(run_id=run_id, component=i, kind=config['kind'], K=int(config['K']),
                            threshold=float(config['threshold']), segment_mass=mass, quantile=quantile,
                            tau=float(thresholds[qi]), half_length80=float(geom['half_lengths'][i]),
                            half_length=float(geom['half_lengths'][i])*length_factor(mass),
                            alpha=float(geom['alpha'][i]), mask_points=part['mask_points'])
                        for field in ['coverage_pc1', 'coverage_random', 'delta_coverage', 'random_full_fraction',
                                      'center_supported', 'max_grid_error', 'max_refinement_change']:
                            row[field] = part[field][qi].item()
                        rows.append(row)
            torch.save(dict(run_id=run_id, masses=MASSES, quantiles=QUANTILES, thresholds=torch.from_numpy(thresholds),
                components={i: dict(control=base.as_tensors(result['control']), by_mass=[base.as_tensors(part) for part in result['by_mass']])
                            for i, result in results.items()}), output/(run_id+'.pt'))
            protocol['geometry_bundle_sha256'][run_id+'.pt'] = sha256(bundle_path)
            (output/'manifest.json').write_text(json.dumps(protocol, indent=2))
            for mass in MASSES:
                for quantile in QUANTILES:
                    valid = [row for row in rows if row['run_id'] == run_id and row['quantile'] == quantile and row['segment_mass'] == mass]
                    summary = dict(run_id=run_id, kind=config['kind'], K=int(config['K']), threshold=float(config['threshold']),
                        segment_mass=mass, quantile=quantile, eligible=len(valid), total=int(config['K']))
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
            print(f'{dataset} l{layer} {run_id}: {len(eligible)} PC1 components, nine segment-mass/percentile combinations', flush=True)
    expected = len(read_csv(old/'components.csv'))*len(QUANTILES)*len(MASSES)
    if not smoke and (len(summaries) != 144 or len(rows) != expected):
        raise ValueError('Incomplete length/percentile experiment')
    write_csv(output/'exclusions.csv', [r for r in read_csv(source/'components.csv') if r['exclusion_reason']])
    validation = dict(passed=True, masses=MASSES, quantiles=QUANTILES, summaries=len(summaries), component_variant_rows=len(rows),
        reproduction_control_components=len(differences), max_mass80_direction_difference=max(differences), max_grid_error=max(row['max_grid_error'] for row in rows),
        max_refinement_change=max(row['max_refinement_change'] for row in rows), masks_nested_within_mass=True, physical_lengths_nested=True, center_support_invariant=True,
        original_pc1_population=True, direct_audit='five positions, all 33 directions, every unique component, mass and quantile')
    (output/'validation.json').write_text(json.dumps(validation, indent=2))
    (output/'COMPLETED.json').write_text(json.dumps(dict(completed=not smoke, smoke=smoke, **validation)))


def render(entries, output, aggregate, mass, quantile):
    import matplotlib.pyplot as plt
    fig = panel_tables(entries, score_formatter=lambda x: f'{100*x:.2f}'.replace('.', ','))
    fig._suptitle.set_text(f'PC1 · Segmento {100*mass:.0f}% · Calibrazione {100*quantile:.0f}° percentile · {"Medie" if aggregate == "mean" else "Mediane"} tra componenti')
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
    fig.texts[-4].set_text(f'PC1 · k=20 · massa gaussiana centrale {100*mass:.0f}% · percentile {100*quantile:.0f}% · rango 0,1 · stessi R/Q, geometrie, componenti e controlli casuali')
    fig.texts[-3].set_text('Copertura = frazione di posizioni accettate. Δ = differenza per componente rispetto alla media di 32 direzioni casuali.')
    fig.texts[-2].set_text('Grassetto: valore maggiore entro esperimento e K; nessun test di significatività. Valutabili: componenti incluse / totali.')
    fig.texts[-1].set_text('Griglia comune tra percentili · stessi centri e direzioni PC1; cambia solo la lunghezza · diagnostica descrittiva sul validation set ricostruito.')
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
        if len(summaries) != 144:
            raise ValueError('Incomplete sweep')
        for summary in summaries:
            matched = [r for r in components if r['run_id'] == summary['run_id'] and r['quantile'] == summary['quantile'] and r['segment_mass'] == summary['segment_mass']]
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
    for mass in MASSES:
        for quantile in QUANTILES:
            output = root/f'm{round(100*mass)}_q{round(100*quantile)}'
            output.mkdir(exist_ok=True)
            for aggregate in ['median', 'mean']:
                entries = []
                for dataset, label, layer, _, _ in SOURCES:
                    baseline = protocols[(dataset, layer)]['pc1_protocol']['baseline_manifest']
                    subset = [r for r in rows_by_case[(dataset, layer)] if float(r['quantile']) == quantile
                              and float(r['segment_mass']) == mass and float(r['threshold']) == .1]
                    entries.append(dict(label=label, layer=layer, dimension=baseline['dimension'],
                        variance=baseline['global_pca']['retained_variance_fraction'], selected=[dict(
                            K=int(r['K']), kind=r['kind'], method=NAMES[r['kind']],
                            retention=float(r[aggregate+'_coverage_pc1']), enrichment=float(r[aggregate+'_delta_coverage']),
                            eligible=int(r['eligible']), total=int(r['total'])) for r in subset]))
                render(entries, output, aggregate, mass, quantile)
            write_csv(output/'tabella_soglia_01.csv', [r for r in selected if float(r['quantile']) == quantile and float(r['segment_mass']) == mass])
    write_csv(root/'tutte_le_576_configurazioni.csv', combined)
    write_csv(root/'tabelle_soglia_01.csv', selected)
    plot_sensitivity(root, selected)
    (root/'export_validation.json').write_text(json.dumps(dict(passed=True, configurations=576, selected_rows=144)))
    (root/'COMPLETED.json').write_text(json.dumps(dict(completed=True, experiments=4, masses=MASSES, quantiles=QUANTILES, configurations=576)))


def plot_sensitivity(root, selected):
    import matplotlib.pyplot as plt
    colors = {'hddc': '#007F86', 'kmeans': '#D67D24'}
    for quantile in QUANTILES:
        fig, axes = plt.subplots(2, 4, figsize=(19, 8), sharex=True)
        for column, (dataset, label, layer, _, _) in enumerate(SOURCES):
            for kind in ['hddc', 'kmeans']:
                for k in [50, 100]:
                    rows = sorted([r for r in selected if r['dataset'] == dataset and r['layer'] == layer
                        and r['kind'] == kind and int(r['K']) == k and float(r['quantile']) == quantile],
                        key=lambda r: float(r['segment_mass']))
                    x = [100*float(r['segment_mass']) for r in rows]
                    for axis, field in [(axes[0, column], 'mean_coverage_pc1'), (axes[1, column], 'mean_delta_coverage')]:
                        axis.plot(x, [100*float(r[field]) for r in rows], color=colors[kind], marker='o' if k == 50 else 's',
                            linestyle='-' if k == 50 else '--', label=f'{NAMES[kind]} K={k}')
                    axes[0, column].plot(x, [100*float(r['mean_coverage_random']) for r in rows], color=colors[kind],
                        linestyle=':', linewidth=1., alpha=.5, marker='o' if k == 50 else 's', markersize=3)
            axes[0, column].set_title(f'{label} · layer {layer}')
            axes[1, column].axhline(0, color='#777777', linewidth=.8)
            axes[1, column].set_xlabel('Massa gaussiana centrale del segmento (%)')
            for row in [0, 1]:
                axes[row, column].set_xticks([90, 95, 99])
                axes[row, column].grid(alpha=.2)
        axes[0, 0].set_ylabel('Copertura media (%)')
        axes[1, 0].set_ylabel('Δ copertura media (pp)')
        axes[0, 0].legend(fontsize=8)
        fig.suptitle(f'PC1 · k20 · percentile {100*quantile:.0f}% · puntinato: copertura casuale · centri e direzioni fissi')
        fig.tight_layout(rect=(0, 0, 1, .95))
        for extension in ['png', 'pdf', 'svg']:
            fig.savefig(root/f'01_sensibilita_lunghezza_q{round(100*quantile)}.{extension}', dpi=200, facecolor='white')
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
