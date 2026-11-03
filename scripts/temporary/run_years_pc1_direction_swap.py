"""Compare saved HDDC/KMeans PC1s at identical centers and projected lengths."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch

from dalg.analysis.pc1_density import KNNLogDensity, log_average_density, project_pc1_segments
from run_years_pc1_density import (
    REPO, discover_runs, finite_exp, load_global_pca, load_reference, sha256, write_csv,
)

DEFAULT_SOURCE = REPO / 'outputs/experiments/years_layer04_pc1_density_pca75/job_1857111'


def match_centers(h_centers, k_centers):
    """Minimum-total-Euclidean-distance bijection, independent of density scores."""
    if h_centers.shape != k_centers.shape or h_centers.ndim != 2:
        raise ValueError('Require equal numbers of centers in the same density space')
    distances = torch.cdist(h_centers.double(), k_centers.double(),
                            compute_mode='donot_use_mm_for_euclid_dist')
    h_ids, k_ids = linear_sum_assignment(distances.numpy())
    matches = []
    for hi, ki in zip(h_ids.tolist(), k_ids.tolist()):
        distance = float(distances[hi, ki])
        matches.append(dict(h_component=hi, k_component=ki, center_distance=distance,
                            h_neighbor_rank=1 + int((distances[hi] < distance).sum()),
                            k_neighbor_rank=1 + int((distances[:, ki] < distance).sum()),
                            mutual_nearest=(int(distances[hi].argmin()) == ki
                                            and int(distances[:, ki].argmin()) == hi)))
    return matches


def common_geometry(h_geometry, k_geometry, hi, ki):
    """Two centers and two unit directions, with the shorter projected interval."""
    centers = torch.stack([h_geometry['centers'][hi], k_geometry['centers'][ki]])
    directions = torch.stack([h_geometry['vectors'][hi], k_geometry['vectors'][ki]])
    lengths = torch.stack([h_geometry['half_lengths'][hi], k_geometry['half_lengths'][ki]])
    if not torch.isfinite(lengths).all() or (lengths <= 0).any():
        raise ValueError('Both projected intervals must have positive finite length')
    return centers, directions, float(lengths.min())


def evaluate_pair(estimator, centers, directions, half_length, grid_points=101):
    """Rows are H/K centers, columns H/K directions; both share the same length."""
    if centers.shape != (2, estimator.dimension) or directions.shape != centers.shape:
        raise ValueError('Require exactly two centers and two directions in density space')
    if grid_points < 3 or grid_points % 2 != 1 or not math.isfinite(half_length) or half_length <= 0:
        raise ValueError('Require an odd grid and a positive finite half-length')
    positions = torch.linspace(-1, 1, 2 * grid_points - 1, dtype=torch.float64)
    midpoint = len(positions) // 2
    profiles, retentions, averages, center_logs, center_radii = [], [], [], [], []
    errors, evaluations, grid_changes, coarse_changes = [], [], [], []
    for center in centers:
        profile, radii = estimator.line_profiles(center, directions, positions * half_length)
        average, error, neval = estimator.adaptive_log_averages(
            center, directions, half_length, positions, profile)
        center_log = float(profile[0, midpoint])
        averages.append(average)
        retentions.append(average - center_log)
        profiles.append(profile - center_log)
        center_logs.append(center_log)
        center_radii.append(float(radii[0, midpoint]))
        errors.append(error)
        evaluations.append(neval)
        grid_changes.append(float((average - log_average_density(profile)).abs().max()))
        coarse_changes.append(float((log_average_density(profile)
                                     - log_average_density(profile[:, ::2])).abs().max()))
    retentions = torch.stack(retentions)
    deltas = retentions[:, 0] - retentions[:, 1]
    return dict(log_retentions=retentions, log_average_densities=torch.stack(averages),
                center_log_densities=center_logs, center_knn_radii=center_radii,
                delta_at_h_center=float(deltas[0]), delta_at_k_center=float(deltas[1]),
                paired_delta=float(deltas.mean()), relative_positions=positions,
                log_relative_density_profiles=torch.stack(profiles),
                max_quadrature_relative_error=max(errors), max_quadrature_evaluations=max(evaluations),
                max_adaptive_vs_grid_log_change=max(grid_changes),
                max_coarse_grid_log_change=max(coarse_changes))


def summarize(rows):
    deltas = np.array([r['paired_delta'] for r in rows])
    h = np.array([r['delta_at_h_center'] for r in rows])
    k = np.array([r['delta_at_k_center'] for r in rows])
    mutual = [r for r in rows if r['mutual_nearest']]
    return dict(K=rows[0]['K'], threshold=rows[0]['threshold'], pairs=len(rows),
                median_paired_log_ratio=float(np.median(deltas)),
                median_paired_ratio=finite_exp(float(np.median(deltas))),
                mean_paired_log_ratio=float(deltas.mean()),
                q25_paired_log_ratio=float(np.quantile(deltas, .25)),
                q75_paired_log_ratio=float(np.quantile(deltas, .75)),
                h_win_fraction=float((deltas > 0).mean()),
                numerical_tie_pairs=int((np.abs(deltas) <= .001).sum()),
                h_wins_over_tolerance=int((deltas > .001).sum()),
                k_wins_over_tolerance=int((deltas < -.001).sum()),
                median_delta_at_h_center=float(np.median(h)),
                median_delta_at_k_center=float(np.median(k)),
                h_win_fraction_at_h_center=float((h > 0).mean()),
                h_win_fraction_at_k_center=float((k > 0).mean()),
                h_wins_both=int(((h > 0) & (k > 0)).sum()),
                k_wins_both=int(((h < 0) & (k < 0)).sum()),
                own_center_wins=int(((h > 0) & (k < 0)).sum()),
                foreign_center_wins=int(((h < 0) & (k > 0)).sum()),
                tied_at_either_center=int(((h == 0) | (k == 0)).sum()),
                mutual_nearest_pairs=len(mutual),
                mutual_median_paired_log_ratio=float(np.median([r['paired_delta'] for r in mutual])) if mutual else None,
                mutual_h_win_fraction=float(np.mean([r['paired_delta'] > 0 for r in mutual])) if mutual else None,
                median_center_distance=float(np.median([r['center_distance'] for r in rows])),
                median_distance_over_h_radius=float(np.median([r['distance_over_h_radius'] for r in rows])),
                median_distance_over_k_radius=float(np.median([r['distance_over_k_radius'] for r in rows])),
                median_h_length_fraction=float(np.median([r['h_length_fraction'] for r in rows])),
                median_k_length_fraction=float(np.median([r['k_length_fraction'] for r in rows])))


def make_report(output, rows, summaries, manifest, validation):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    selected = [r for r in rows if r['threshold'] == .1]
    for K, color in [(50, '#168a89'), (100, '#dd822e')]:
        group = [r for r in selected if r['K'] == K]
        axes[0].scatter([r['delta_at_h_center'] for r in group],
                        [r['delta_at_k_center'] for r in group], s=18, alpha=.65, color=color, label=f'K={K}')
        axes[1].scatter([r['distance_over_h_radius'] for r in group],
                        [r['paired_delta'] for r in group], s=18, alpha=.65, color=color, label=f'K={K}')
    for ax in axes:
        ax.axhline(0, color='gray', lw=.8)
        ax.legend()
    axes[0].axvline(0, color='gray', lw=.8)
    axes[0].set(xlabel='log(A_H / A_K) at HDDC center', ylabel='log(A_H / A_K) at KMeans center')
    axes[1].set(xlabel='Matched-center distance / HDDC-center r20', ylabel='Paired mean log(A_H / A_K)')
    fig.suptitle('PC1 direction swap, years/layer 4, PCA75 d=141, threshold 0.1')
    fig.savefig(output / 'direction_swap.png', dpi=160)
    plt.close(fig)

    lines = ['# Years/layer-4 PCA75 PC1 direction-swap control', '',
             f"Shared training PCA: d={manifest['dimension']}, variance={manifest['retained_variance_fraction']:.8%}.",
             'Reuse the original 2,002 validation rows and k=20 Euclidean kNN density. No model/PCA refitting.',
             'Match HDDC/KMeans centers bijectively by minimum total Euclidean distance in PCA space, separately for each K/threshold.',
             'Every pair uses L=min(a_H_projected,a_K_projected). Evaluate both projected original unit PC1s at both centers using L.',
             'This common interval is a truncation of the original intervals; it need not contain 80% of either original marginal.',
             'At each center delta=log(A_Hdirection/A_Kdirection). The center density and shared length cancel exactly.',
             'A shared random baseline would also cancel from the log-enrichment difference; no random directions are resampled.',
             'Primary paired delta is the mean of the deltas at the two centers. exp(delta) is their geometric-mean density ratio.',
             'Positive deltas favor HDDC. All summaries give equal weight to matched pairs; no calibrated significance tests.', '',
             '| K | Threshold | Pairs | Median paired log ratio | exp(median) | HDDC wins | Median at H center | Median at K center | Mutual pairs / median log ratio |',
             '| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
    for s in summaries:
        mutual = s['mutual_median_paired_log_ratio']
        mutual_text = f'{mutual:.4g}' if mutual is not None else 'NA'
        lines.append(f"| {s['K']} | {s['threshold']:g} | {s['pairs']} | {s['median_paired_log_ratio']:.4g} | "
                     f"{s['median_paired_ratio']:.4g} | {s['h_win_fraction']:.1%} | "
                     f"{s['median_delta_at_h_center']:.4g} | {s['median_delta_at_k_center']:.4g} | "
                     f"{s['mutual_nearest_pairs']} / {mutual_text} |")
    lines += ['', '## Center interaction and matching diagnostics', '',
              '| K | Threshold | H wins both | K wins both | Each wins at own center | Each wins at foreign center | Median distance/r20 H / K | Median retained length H / K |',
              '| ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |']
    for s in summaries:
        lines.append(f"| {s['K']} | {s['threshold']:g} | {s['h_wins_both']} | {s['k_wins_both']} | "
                     f"{s['own_center_wins']} | {s['foreign_center_wins']} | "
                     f"{s['median_distance_over_h_radius']:.3g} / {s['median_distance_over_k_radius']:.3g} | "
                     f"{s['median_h_length_fraction']:.3g} / {s['median_k_length_fraction']:.3g} |")
    maximum = max(c['max_log_retention_change'] for c in validation['checks'])
    lines += ['', f"Numerical validation: {len(validation['checks'])} doubled-pilot checks passed; maximum log-retention change {maximum:.3g}.",
              'All four integrals per pair use adaptive peak-split QUADPACK with estimated relative error <=1e-4.',
              'The denser pilot uses 401 rather than 201 discovery positions; the tolerance is 0.001 in natural-log scores and paired deltas.',
              'Checks select the largest adaptive/grid discrepancy and largest absolute paired delta in each K/threshold group.', '',
              'The summary CSV also counts pairs within 0.001 of zero as numerically close, and wins beyond +/-0.001.', '',
              '## Limits', '',
              'Hungarian matching can force distant pairs. Inspect pair distances, both nearest-neighbor ranks and the prespecified mutual-nearest subset.',
              'Transporting a local PC1 to another center tests its support there; this is not an angular comparison with a known tangent.',
              'The minimum-length rule changes the evaluated scale and can truncate the two original intervals by different fractions.',
              'Threshold variants share data and repeated KMeans geometry; they are not independent replications.',
              'The density remains a projected support surrogate in 141 dimensions, with discarded coordinates invisible and prompt/template variation present.',
              'Validation participated in checkpoint selection. Matching does not establish generalization across datasets, layers or seeds.', '',
              '![Direction-swap comparison](direction_swap.png)', '',
              '[Pairs and diagnostics](pairs.csv) · [Summaries](summary.csv) · [Numerical validation](integration_validation.json)', '']
    (output / 'report.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--smoke', action='store_true', help='Only two pairs from the first configuration')
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    source, output = args.source.resolve(strict=True), args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    original = json.loads((source / 'manifest.json').read_text())
    completion = json.loads((source / 'COMPLETED.json').read_text())
    if (original['format'] != 'dalg_years_pc1_density_pca75_v1' or original['smoke']
            or completion != dict(completed=True, smoke=False, configurations=16, components=1200)
            or not json.loads((source / 'integration_validation.json').read_text())['passed']):
        raise ValueError('Require the completed and numerically validated PCA75 pilot')
    activation_dir = Path(original['activation_dir'])
    runs = discover_runs(activation_dir)
    reference, ids, _ = load_reference(activation_dir, runs)
    if ids != original['reference_global_rows'] or sha256(activation_dir / 'config.json') != original['extraction_config_sha256']:
        raise ValueError('Source pilot validation population changed')
    pca_info = original['global_pca']
    if sha256(pca_info['path']) != pca_info['sha256']:
        raise ValueError('Shared PCA artifact changed')
    pca = load_global_pca(pca_info['path'], activation_dir, ids)
    estimator = KNNLogDensity((reference - pca['mean']) @ pca['basis'].T, k=original['k'])
    source_runs = {r['run_id']: r for r in original['sources']}
    bundles, bundle_hashes = {}, {}
    for run in runs:
        saved = source_runs[run['run_id']]
        if (sha256(run['checkpoint']) != saved['checkpoint_sha256']
                or sha256(run['run_dir'] / 'config.json') != saved['config_sha256']
                or sha256(run['run_dir'] / 'val_indices.json') != saved['split_sha256']):
            raise ValueError('Saved model/config/split changed')
        path = source / (run['run_id'] + '.pt')
        bundle_hashes[path.name] = sha256(path)
        bundle = torch.load(path, map_location='cpu', weights_only=True)
        projected = project_pc1_segments(bundle['geometry'], pca['mean'], pca['basis'])
        if any(projected['exclusion_reasons']) or bundle['run_id'] != run['run_id']:
            raise ValueError('This paired pilot requires all saved components to be evaluable')
        for field in ['centers', 'vectors', 'half_lengths', 'alpha']:
            torch.testing.assert_close(projected[field], bundle['projected_geometry'][field], rtol=1e-12, atol=1e-12)
        bundles[run['kind'], run['K'], run['threshold']] = projected
    output.mkdir(parents=True)
    manifest = dict(format='dalg_years_pc1_direction_swap_v1', created_at=datetime.now(timezone.utc).isoformat(),
                    dataset='years_neutral', layer=4, source=str(source),
                    source_manifest_sha256=sha256(source / 'manifest.json'), source_bundle_sha256=bundle_hashes,
                    global_pca=pca_info, dimension=estimator.dimension,
                    retained_variance_fraction=pca['retained_variance_fraction'],
                    reference_global_rows=ids, k=estimator.k, smoke=args.smoke,
                    matching='bijective minimum sum Euclidean center distance in shared PCA space; separately per K/threshold',
                    common_half_length='minimum of the two projected original PC1 half-lengths',
                    paired_metric='mean of log(A_Hdirection/A_Kdirection) at both matched centers',
                    random_baseline='not resampled; any shared baseline cancels exactly in paired log-enrichment',
                    integration='adaptive peak-split QUADPACK epsrel=1e-4; 201 pilot positions',
                    validation='401 pilot positions; worst adaptive/grid discrepancy and largest absolute paired delta per K/threshold; tolerance .001',
                    source_sha256={str(p.relative_to(REPO)): sha256(p) for p in
                                   [Path(__file__), REPO / 'src/dalg/analysis/pc1_density.py',
                                    REPO / 'scripts/temporary/run_years_pc1_density.py']})
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    rows, summaries, checks = [], [], []
    keys = sorted({(r['K'], r['threshold']) for r in runs})
    for K, threshold in keys[:1] if args.smoke else keys:
        h, k = bundles['hddc', K, threshold], bundles['kmeans', K, threshold]
        matches = match_centers(h['centers'], k['centers'])
        group, profiles = [], {}
        for match in matches[:2] if args.smoke else matches:
            hi, ki = match['h_component'], match['k_component']
            centers, directions, length = common_geometry(h, k, hi, ki)
            result = evaluate_pair(estimator, centers, directions, length)
            row = dict(K=K, threshold=threshold, **match, common_half_length=length,
                       h_original_projected_half_length=float(h['half_lengths'][hi]),
                       k_original_projected_half_length=float(k['half_lengths'][ki]),
                       h_length_fraction=length / float(h['half_lengths'][hi]),
                       k_length_fraction=length / float(k['half_lengths'][ki]),
                       h_alpha=float(h['alpha'][hi]), k_alpha=float(k['alpha'][ki]),
                       abs_direction_cosine=float((directions[0] * directions[1]).sum().abs()),
                       **{key: value for key, value in result.items() if isinstance(value, (int, float))})
            for ci, name in enumerate(['h', 'k']):
                row[f'{name}_center_log_density'] = result['center_log_densities'][ci]
                row[f'{name}_center_knn_radius'] = result['center_knn_radii'][ci]
                row[f'distance_over_{name}_radius'] = row['center_distance'] / result['center_knn_radii'][ci]
                for di, direction in enumerate(['h', 'k']):
                    row[f'log_retention_{direction}_dir_at_{name}_center'] = float(result['log_retentions'][ci, di])
            profiles[hi] = dict(match=match, half_length=length, **result)
            group.append(row)
            if len(group) % 10 == 0:
                print(f'K={K}, threshold={threshold:g}: {len(group)}/{len(matches)} pairs', flush=True)
        if not args.smoke:
            selected = {max(range(len(group)), key=lambda i: group[i]['max_adaptive_vs_grid_log_change']),
                        max(range(len(group)), key=lambda i: abs(group[i]['paired_delta']))}
            for index in sorted(selected):
                row = group[index]
                hi, ki = row['h_component'], row['k_component']
                centers, directions, length = common_geometry(h, k, hi, ki)
                dense = evaluate_pair(estimator, centers, directions, length, grid_points=201)
                baseline = profiles[hi]
                change = float((dense['log_retentions'] - baseline['log_retentions']).abs().max())
                delta_change = max(abs(dense[field] - baseline[field]) for field in
                                   ['paired_delta', 'delta_at_h_center', 'delta_at_k_center'])
                check = dict(K=K, threshold=threshold, h_component=hi, k_component=ki,
                             max_log_retention_change=change, max_delta_change=delta_change,
                             passed=max(change, delta_change) < .001)
                checks.append(check)
                if not check['passed']:
                    raise RuntimeError(f'Denser-pilot check failed: {check}')
        torch.save(dict(K=K, threshold=threshold, profiles=profiles), output / f'K{K}_threshold{threshold:g}.pt')
        summaries.append(summarize(group))
        rows.extend(group)
        write_csv(output / 'pairs.csv', rows)
        write_csv(output / 'summary.csv', summaries)
        print(summaries[-1], flush=True)
    validation = dict(passed=all(c['passed'] for c in checks), smoke=args.smoke,
                      pilot_positions=[201, 401], tolerance_natural_log=.001, checks=checks)
    (output / 'integration_validation.json').write_text(json.dumps(validation, indent=2))
    if not args.smoke:
        if len(rows) != 600 or len(summaries) != 8 or not checks:
            raise ValueError('Expected all 600 pairs and eight configuration comparisons')
        make_report(output, rows, summaries, manifest, validation)
    (output / 'COMPLETED.json').write_text(json.dumps(dict(completed=True, smoke=args.smoke,
                                                         configurations=len(summaries), pairs=len(rows)), indent=2))
    print(f'Saved direction-swap results in {output}', flush=True)


if __name__ == '__main__':
    main()
