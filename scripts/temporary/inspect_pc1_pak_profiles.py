"""Inspect quadrature instability of PAk on original projected PC1 segments."""
import json
import math
import os
from pathlib import Path

import torch
from run_pc1_pak_controlled import PAkDensity, reference_from_manifest, SOURCES
from dalg.analysis.pc1_density import random_unit_directions, log_average_density


def main():
    torch.set_num_threads(1)
    source = SOURCES[0][3]
    manifest = json.loads((source/'manifest.json').read_text())
    pca = torch.load(manifest['global_pca']['path'], weights_only=True)
    reference, _ = reference_from_manifest(manifest, pca)
    estimator = PAkDensity(reference.numpy(), minimum_k=3)
    run_id = manifest['sources'][0]['run_id']
    geom = torch.load(source/(run_id+'.pt'), weights_only=True)['projected_geometry']
    center, vector, length = [geom[k][0] for k in ['centers', 'vectors', 'half_lengths']]
    directions = torch.cat([vector[None], random_unit_directions(141, 32, 42)])
    center_density = estimator.query(center[None].numpy())[0][0]
    output = Path('outputs/experiments/pc1_pak_controlled')/f'profiles_{os.environ["SLURM_JOB_ID"]}'
    output.mkdir()
    for count in [201, 401, 801, 1601, 3201]:
        positions = torch.linspace(-1, 1, count, dtype=torch.float64)
        logs, errors = [], []
        for direction in directions:
            values, err = estimator.query((center+length*positions[:, None]*direction).numpy())
            logs.append(values-center_density)
            errors.append(err)
        logs = torch.stack(logs)
        error = torch.stack(errors)
        rets = log_average_density(logs)
        maximum = torch.unravel_index(logs.argmax(), logs.shape)
        row = dict(points=count, log_retention=float(rets[0]),
            log_random_mean=float(torch.logsumexp(rets[1:], 0)-math.log(32)),
            max_relative_log_density=float(logs.max()), min_relative_log_density=float(logs.min()),
            max_direction=int(maximum[0]), max_position=float(positions[maximum[1]]),
            min_error=float(error.min()), max_error=float(error.max()), log_retentions=rets.tolist())
        print(json.dumps(row), flush=True)
        torch.save(dict(positions=positions, logs=logs, errors=error), output/f'grid_{count}.pt')
        (output/f'grid_{count}.json').write_text(json.dumps(row, indent=2))


if __name__ == '__main__':
    main()
