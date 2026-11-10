"""Save a failing PAk likelihood input and check its floating-point scales."""
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from scipy.special import gammaln
from dadapy._cython import cython_density as cd
from dadapy._utils.density_estimation import return_not_normalised_density_PAk, return_not_normalised_density_PAk_optimized
from dadapy._utils.utils import compute_cross_nn_distances
from run_pc1_pak_controlled import SOURCES, PAkDensity, reference_from_manifest, random_unit_directions


def main():
    torch.set_num_threads(1)
    source = SOURCES[0][3]
    manifest = json.loads((source/'manifest.json').read_text())
    pca = torch.load(manifest['global_pca']['path'], weights_only=True)
    reference, _ = reference_from_manifest(manifest, pca)
    estimator = PAkDensity(reference.numpy())
    run_id = 'hddc__years_neutral__l04__k50__q16__s42__5539869c'
    geom = torch.load(source/(run_id+'.pt'), weights_only=True)['projected_geometry']
    component = 12
    center, vector, length = [geom[key][component] for key in ['centers', 'vectors', 'half_lengths']]
    directions = torch.cat([vector[None], random_unit_directions(141, 32, 42+component)])
    positions = torch.linspace(-1, 1, 201, dtype=torch.float64)
    for i, direction in enumerate(directions):
        queries = (center+length*positions[:, None]*direction).numpy()
        dist, idx = compute_cross_nn_distances(queries, estimator.data.X, 100)
        kstar = np.maximum(cd._compute_kstar_interp(141, len(dist), 100, 1e-6,
            idx, dist, estimator.data.distances, False, False), 4)
        density, _, _ = return_not_normalised_density_PAk(dist, 141, kstar, interpolation=True)
        if np.isfinite(density).all():
            continue
        bad = np.flatnonzero(~np.isfinite(density))
        # A change of length units inside the likelihood preserves its objective.
        scales = dist[np.arange(len(dist)), kstar]
        normalized, _, _ = return_not_normalised_density_PAk(dist/scales[:, None], 141, kstar, interpolation=True)
        normalized -= 141*np.log(scales)
        optimized, _, _ = return_not_normalised_density_PAk_optimized(dist, 141, kstar, interpolation=True)
        print('Optimized finite:', np.isfinite(optimized).all(), optimized[bad].tolist(), flush=True)
        initial = np.log(kstar-1)-(.5*141*math.log(math.pi)-gammaln(71.5))-141*np.log(scales)
        report = dict(run_id=run_id, component=component, direction=i, bad_indices=bad.tolist(),
            initial_log_densities=initial[bad].tolist(), normalized_log_densities=normalized[bad].tolist(),
            all_normalized_finite=bool(np.isfinite(normalized).all()),
            optimized_all_finite=bool(np.isfinite(optimized).all()), optimized_log_densities=optimized[bad].tolist(),
            max_change_on_previously_finite=float(np.max(np.abs(density[np.isfinite(density)]-normalized[np.isfinite(density)]))))
        output = Path('outputs/experiments/pc1_pak_controlled')/f'nonfinite_{os.environ["SLURM_JOB_ID"]}'
        output.mkdir()
        np.savez(output/'inputs.npz', distances=dist, indices=idx, kstar=kstar)
        (output/'report.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report), flush=True)
        return
    raise RuntimeError('Reported component did not reproduce the failure')


if __name__ == '__main__':
    main()
