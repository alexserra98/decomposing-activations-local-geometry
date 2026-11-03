"""Check PC1 density integrals using a doubled peak-discovery grid.

Select the worst uniform-grid discrepancy per method/K from a completed pilot.
Recompute all 33 direction integrals with 401 pilot positions instead of 201.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from dalg.analysis.pc1_density import KNNLogDensity, score_component
from run_years_pc1_density import discover_runs, load_reference, load_global_pca, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    output = args.results.resolve(strict=True)
    path = output / 'integration_validation.json'
    if path.exists():
        raise FileExistsError(path)
    completion = json.loads((output / 'COMPLETED.json').read_text())
    if (completion.get('completed') is not True or completion.get('smoke') is not False
            or completion.get('configurations') != 16 or completion.get('components') != 1200):
        raise ValueError('Integration validation requires a completed full 16-configuration pilot')
    manifest = json.loads((output / 'manifest.json').read_text())
    activation_dir = Path(manifest['activation_dir'])
    reference, reference_ids, _ = load_reference(activation_dir, discover_runs(activation_dir))
    pca_info = manifest.get('global_pca')
    if pca_info:
        if sha256(pca_info['path']) != pca_info['sha256']:
            raise ValueError('PCA artifact changed after evaluation')
        pca = load_global_pca(pca_info['path'], activation_dir, reference_ids)
        reference = (reference - pca['mean']) @ pca['basis'].T
    estimator = KNNLogDensity(reference, k=manifest['k'])
    if estimator.dimension != manifest['dimension']:
        raise ValueError('Density dimension differs from the evaluated experiment')
    with (output / 'components.csv').open() as stream:
        rows = [row for row in csv.DictReader(stream) if not row['exclusion_reason']]
    selected = {}
    for row in rows:
        key = row['kind'], row['K']
        if key not in selected or (float(row['max_adaptive_vs_grid_log_retention_change'])
                                  > float(selected[key]['max_adaptive_vs_grid_log_retention_change'])):
            selected[key] = row
    if set(selected) != {(kind, str(K)) for kind in ['hddc', 'kmeans'] for K in [50, 100]}:
        raise ValueError('Expected eligible geometry in all four method/K groups')
    checks = []
    for row in selected.values():
        bundle = torch.load(output / (row['run_id'] + '.pt'), weights_only=True)
        geometry = bundle['geometry']
        evaluated = bundle['projected_geometry'] if pca_info else geometry
        component = int(row['component'])
        original = bundle['components'][component]
        dense = score_component(estimator, evaluated['centers'][component],
                                evaluated['vectors'][component],
                                float(geometry['eigenvalues'][component]),
                                half_length=original['half_length'],
                                seed=42 + component, grid_points=201, adaptive=True)
        direction_change = float((original['log_direction_retentions']
                                  - dense['log_direction_retentions']).abs().max())
        enrichment_change = abs(original['log_enrichment'] - dense['log_enrichment'])
        check = dict(run_id=row['run_id'], component=component,
                     max_direction_log_score_change=direction_change,
                     log_enrichment_change=enrichment_change,
                     denser_quadrature_relative_error=dense['adaptive_quadrature_relative_error'],
                     passed=max(direction_change, enrichment_change) < .001)
        checks.append(check)
        print(check, flush=True)
    result = dict(passed=all(check['passed'] for check in checks),
                  density_dimension=estimator.dimension, evaluation_space=manifest.get('evaluation_space', 'ambient'),
                  pilot_positions=[201, 401], tolerance_natural_log=.001,
                  validation_source_sha256=sha256(Path(__file__)),
                  selection='largest adaptive-vs-grid discrepancy per method/K', checks=checks)
    path.write_text(json.dumps(result, indent=2))
    if not result['passed']:
        raise RuntimeError('Denser-pilot integration check did not pass')
    report = output / 'report.md'
    maximum = max(c['max_direction_log_score_change'] for c in checks)
    report.write_text(report.read_text() + f'\nNumerical validation: {len(checks)}/{len(checks)} denser-pilot checks passed; '
                      f'maximum natural-log score change {maximum:.3g}. '
                      '[Validation details](integration_validation.json).\n')


if __name__ == '__main__':
    main()
