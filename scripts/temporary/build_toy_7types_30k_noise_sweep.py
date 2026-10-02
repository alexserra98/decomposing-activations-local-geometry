"""Generate seven manifolds with 30,000 points each at four paired noise levels."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import torch

from dalg.data import ToyManifoldConfig, save_toy_manifold_shards

DEST = Path('dalg-cache/assets/toy_manifolds_7types_1each_D128_30Keach_noise_sweep_seed0')
NAMES = ('segment', 'circle', 'swiss_roll', 'helix', 'helix_4d',
         'hypersphere_6d', 'product_torus_6d')
CONDITIONS = {'noiseless': None, 'noise_ratio_1000': 1000.0,
              'noise_ratio_100': 100.0, 'noise_ratio_10': 10.0}


def main():
    torch.set_num_threads(4)
    if DEST.exists() and (not DEST.is_dir() or any(DEST.iterdir())):
        raise FileExistsError(f'Destination must be absent or empty: {DEST}')
    config = ToyManifoldConfig(
        ambient_dim=128, n_samples=30_000 * len(NAMES), manifolds_per_type=1,
        manifold_types=NAMES, seed=0, offset_radius=4.0, noise_ratio=None,
    )
    for name, ratio in CONDITIONS.items():
        print(f'Generating {name}: {config.n_samples:,} points', flush=True)
        save_toy_manifold_shards(
            DEST / name, replace(config, noise_ratio=ratio), shard_size=50_000, layer=0,
        )

    print('Validating saved counts, metadata, and paired noise', flush=True)
    metadata = {}
    for name, ratio in CONDITIONS.items():
        root = DEST / name
        saved_config = json.loads((root / 'config.json').read_text())
        assert saved_config['num_rows'] == config.n_samples
        assert saved_config['num_shards'] == 5
        assert saved_config['generator_config'] == json.loads(json.dumps(
            asdict(replace(config, noise_ratio=ratio))))
        meta = torch.load(root / 'manifold_metadata.pt', weights_only=True)
        metadata[name] = meta
        ids = meta['row_manifold_ids']
        assert ids.shape == (config.n_samples,)
        assert torch.equal(torch.bincount(ids, minlength=7), torch.full((7,), 30_000))
        assert meta['intrinsic_dims'] == (1, 1, 2, 1, 1, 6, 6)
        assert meta['embedding_dims'] == (1, 2, 3, 3, 4, 7, 12)
        base = metadata['noiseless']
        for key in ('row_manifold_ids', 'calibration_scales', 'offsets',
                    'raw_max_abs_curvatures', 'curvature_radii'):
            assert torch.equal(meta[key], base[key]), (name, key)
        for key in ('calibration_means', 'embeddings'):
            assert all(torch.equal(a, b) for a, b in zip(meta[key], base[key], strict=True))
        expected = torch.zeros_like(meta['noise_stds']) if ratio is None else meta['curvature_radii'] / ratio
        assert torch.equal(meta['noise_stds'], expected)

    max_error = 0.0
    for i in range(5):
        start, end = i * 50_000, min((i + 1) * 50_000, config.n_samples)
        points = {}
        for name in CONDITIONS:
            root = DEST / name
            x = torch.load(root / 'layer00' / f'shard_{i:05d}.pt', weights_only=True)
            assert x.shape == (end - start, 1, 128) and x.dtype == torch.float32
            assert torch.isfinite(x).all()
            points[name] = x.double()
            rows = json.loads((root / 'meta' / f'shard_{i:05d}.json').read_text())
            assert rows['start'] == start and rows['end'] == end
            assert rows['row_indices'] == list(range(start, end))
            ids = metadata[name]['row_manifold_ids'][start:end].tolist()
            for row, manifold_id in zip(rows['rows'], ids, strict=True):
                assert row == {'subset': NAMES[manifold_id], 'manifold_id': manifold_id,
                               'manifold_type_id': manifold_id,
                               'intrinsic_dim': metadata[name]['intrinsic_dims'][manifold_id]}
        reference_noise = points['noise_ratio_10'] - points['noiseless']
        for name, ratio in CONDITIONS.items():
            if ratio is not None:
                error = (points[name] - points['noiseless'] - reference_noise * (10.0 / ratio)).abs().max().item()
                assert error < 1e-6, (name, i, error)
                max_error = max(max_error, error)

    report = {
        'validation': 'passed', 'points_per_manifold': 30_000,
        'points_per_dataset': config.n_samples, 'base_config': asdict(config),
        'datasets': {name: {'path': str((DEST / name).resolve()), 'noise_ratio': ratio}
                     for name, ratio in CONDITIONS.items()},
        'max_noise_pairing_error': max_error,
        'generator_sha256': hashlib.sha256(Path('src/dalg/data/manifold_dataset.py').read_bytes()).hexdigest(),
        'script': str(Path(__file__).resolve()),
        'torch_version': torch.__version__,
    }
    (DEST / 'sweep.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
