"""Check PAk interpolation near a nearest-neighbor switch with k*=3 or 4."""
import json
import os
from pathlib import Path

import numpy as np
import torch
from dadapy._cython import cython_density as cd
from dadapy._utils.density_estimation import return_not_normalised_density_PAk
from dadapy._utils.utils import compute_cross_nn_distances

from run_pc1_pak_controlled import PAkDensity, reference_from_manifest, SOURCES
from dalg.analysis.pc1_density import random_unit_directions


def main():
    torch.set_num_threads(1)
    source = SOURCES[0][3]
    manifest = json.loads((source/'manifest.json').read_text())
    pca = torch.load(manifest['global_pca']['path'], weights_only=True)
    ref, _ = reference_from_manifest(manifest, pca)
    est = PAkDensity(ref.numpy(), minimum_k=3)
    run_id = manifest['sources'][0]['run_id']
    geom = torch.load(source/(run_id+'.pt'), weights_only=True)['projected_geometry']
    center, length = geom['centers'][0], float(geom['half_lengths'][0])
    # Direction 4 is a diagnostic random direction, not selected for a model comparison.
    direction = random_unit_directions(141, 32, 42)[3]
    positions = np.linspace(-1, 1, 1601)
    residual = (ref-center).numpy()
    norms, dots = (residual**2).sum(1), residual@direction.numpy()
    nearest = (norms[None]-2*length*positions[:, None]*dots).argmin(1)
    changes = np.flatnonzero(nearest[:-1] != nearest[1:])
    reports = []
    for index in changes[:8]:
        i, j = nearest[index], nearest[index+1]
        crossing = (norms[i]-norms[j])/(2*length*(dots[i]-dots[j]))
        offsets = np.r_[-np.logspace(-2, -10, 9), np.logspace(-2, -10, 9)]
        q = center.numpy()+length*(crossing+offsets[:, None])*direction.numpy()
        dist, idx = compute_cross_nn_distances(q, est.data.X, 100)
        k = cd._compute_kstar_interp(141, len(q), 100, 1e-6, idx, dist, est.data.distances, False, False)
        original, _, _ = return_not_normalised_density_PAk(dist, 141, k, interpolation=True)
        bounded, _, _ = return_not_normalised_density_PAk(dist, 141, np.maximum(k, 4), interpolation=True)
        report = dict(crossing=float(crossing), offsets=offsets.tolist(), kstar=k.tolist(),
            default_log_density=original.tolist(), min4_log_density=bounded.tolist())
        reports.append(report)
        print(json.dumps(report), flush=True)
    output = Path('outputs/experiments/pc1_pak_controlled')/f'min_neighbors_check_{os.environ["SLURM_JOB_ID"]}.json'
    output.write_text(json.dumps(reports, indent=2))


if __name__ == '__main__':
    main()
