"""Compare DADApy's two PAk likelihood implementations on saved query geometry."""
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from dadapy._cython import cython_density as cd
from dadapy._utils.density_estimation import return_not_normalised_density_PAk_optimized
from dadapy._utils.utils import compute_cross_nn_distances

from run_pc1_pak_controlled import PAkDensity, reference_from_manifest, SOURCES


def main():
    torch.set_num_threads(1)
    checks = []
    for dataset, _, layer, source, _ in SOURCES:
        manifest = json.loads((source/'manifest.json').read_text())
        pca = torch.load(manifest['global_pca']['path'], weights_only=True)
        reference, _ = reference_from_manifest(manifest, pca)
        estimator = PAkDensity(reference.numpy(), minimum_k=3)
        configs = json.loads((source/'manifest.json').read_text())
        run_id = configs['sources'][0]['run_id']
        geometry = torch.load(source/(run_id+'.pt'), weights_only=True)['projected_geometry']
        c, v, a = [geometry[key][0] for key in ['centers', 'vectors', 'half_lengths']]
        queries = (c+torch.linspace(-1, 1, 101, dtype=torch.float64)[:, None]*a*v).numpy()
        started = time.monotonic()
        direct, error = estimator.query(queries)
        direct_seconds = time.monotonic()-started
        started = time.monotonic()
        dist, idx = compute_cross_nn_distances(queries, estimator.data.X, estimator.data.maxk)
        kstar = cd._compute_kstar_interp(estimator.dimension, len(queries), estimator.data.maxk,
            1e-6, idx, dist, estimator.data.distances, False, False)
        optimized, optimized_error, _ = return_not_normalised_density_PAk_optimized(
            dist, estimator.dimension, kstar, interpolation=True)
        optimized -= np.log(len(reference))
        change = float(np.max(np.abs(direct.numpy()-optimized)))
        item = dict(dataset=dataset, layer=layer, max_log_density_change=change,
            max_error_change=float(np.max(np.abs(error.numpy()-optimized_error))),
            direct_seconds=direct_seconds, optimized_seconds=time.monotonic()-started,
            kstar_min=int(kstar.min()), kstar_max=int(kstar.max()))
        print(json.dumps(item), flush=True)
        checks.append(item)
        if change > 1e-7:
            raise ValueError('Optimized PAk differs from public interpolation')
    out = Path('outputs/experiments/pc1_pak_controlled')/f'backend_check_{os.environ["SLURM_JOB_ID"]}.json'
    out.write_text(json.dumps(checks, indent=2))


if __name__ == '__main__':
    main()
