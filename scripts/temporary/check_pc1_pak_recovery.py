"""Reevaluate the complete real component that failed in the first PAk sweep."""
import argparse
import json
import os
from pathlib import Path

import torch
from run_pc1_pak_controlled import SOURCES, initialize_worker, reference_from_manifest, score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', default='hddc__years_neutral__l04__k50__q16__s42__5539869c')
    parser.add_argument('--component', type=int, default=12)
    args = parser.parse_args()
    torch.set_num_threads(1)
    source = SOURCES[0][3]
    manifest = json.loads((source/'manifest.json').read_text())
    pca = torch.load(manifest['global_pca']['path'], weights_only=True)
    reference, _ = reference_from_manifest(manifest, pca)
    initialize_worker(reference.numpy(), 100)
    run_id = args.run_id
    geom = torch.load(source/(run_id+'.pt'), weights_only=True)['projected_geometry']
    i = args.component
    result = score(geom['centers'][i], geom['vectors'][i], float(geom['half_lengths'][i]), i)
    dense = score(geom['centers'][i], geom['vectors'][i], float(geom['half_lengths'][i]), i, pilot_points=401)
    change = float((result['log_direction_retentions']-dense['log_direction_retentions']).abs().max())
    enrichment_change = abs(result['log_enrichment']-dense['log_enrichment'])
    report = dict(run_id=run_id, component=i, log_retention=result['log_retention'],
        log_enrichment=result['log_enrichment'], max_direction_log_change=change,
        enrichment_log_change=enrichment_change,
        max_relative_error=result['quadrature_max_relative_error'],
        passed=max(change, enrichment_change) < .001)
    output = Path('outputs/experiments/pc1_pak_controlled')/f'recovery_{os.environ["SLURM_JOB_ID"]}.json'
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    if not report['passed']:
        raise ValueError('Recovered component failed denser-pilot check')


if __name__ == '__main__':
    main()
