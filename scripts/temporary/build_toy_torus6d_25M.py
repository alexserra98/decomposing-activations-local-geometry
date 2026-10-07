"""Regenerate the 5M noiseless torus dataset at 25M with identical geometry."""
from dataclasses import asdict
import json
from pathlib import Path

import torch

from dalg.data import ToyManifoldConfig, save_toy_manifold_shards

SOURCE = Path('dalg-cache/assets/toy_product_torus_6d_1each_D128_5M_noiseless_seed0')
DEST = Path('dalg-cache/assets/toy_product_torus_6d_1each_D128_25M_noiseless_seed0')


def main():
    torch.set_num_threads(8)
    saved = json.loads((SOURCE / 'config.json').read_text())
    config = dict(saved['generator_config'])
    assert config['n_samples'] == 5_000_000
    assert config['manifold_types'] == ['product_torus_6d']
    assert config['noise_ratio'] is None
    config['n_samples'] = 25_000_000
    config['manifold_types'] = tuple(config['manifold_types'])
    config = ToyManifoldConfig(**config)
    assert saved['layers'] == [0] and saved['shard_size'] == 50_000
    print(f'Generating {config.n_samples:,} points in {DEST}', flush=True)
    save_toy_manifold_shards(DEST, config, shard_size=saved['shard_size'], layer=0)

    actual = json.loads((DEST / 'config.json').read_text())
    expected = dict(saved, num_rows=25_000_000, num_shards=500,
                    generator_config=json.loads(json.dumps(asdict(config))))
    assert actual == expected
    original = torch.load(SOURCE / 'manifold_metadata.pt', weights_only=False)
    metadata = torch.load(DEST / 'manifold_metadata.pt', weights_only=False)
    for key in ('calibration_means', 'calibration_scales', 'raw_max_abs_curvatures',
                'max_abs_curvatures', 'curvature_radii', 'noise_stds', 'embeddings',
                'offset_directions', 'offsets', 'manifold_type_ids'):
        torch.testing.assert_close(metadata[key], original[key], rtol=0, atol=1e-12, msg=key)
    ids = metadata['row_manifold_ids']
    assert ids.shape == (25_000_000,) and ids.dtype == torch.int64
    assert bool((ids == 0).all())
    assert len(list((DEST / 'layer00').glob('*.pt'))) == 500
    assert len(list((DEST / 'meta').glob('*.json'))) == 500
    print('Checking all 500 shards and row metadata', flush=True)
    for i in range(500):
        points = torch.load(DEST / 'layer00' / f'shard_{i:05d}.pt', weights_only=True)
        assert points.shape == (50_000, 1, 128) and points.dtype == torch.float32
        assert bool(torch.isfinite(points).all())
        rows = json.loads((DEST / 'meta' / f'shard_{i:05d}.json').read_text())
        start, end = i * 50_000, (i + 1) * 50_000
        assert rows['start'] == start and rows['end'] == end
        assert rows['row_indices'] == list(range(start, end))
        assert len(rows['rows']) == 50_000
        assert all(row == {'subset': 'product_torus_6d', 'manifold_id': 0,
                           'manifold_type_id': 0, 'intrinsic_dim': 6}
                   for row in rows['rows'])
        if (i + 1) % 100 == 0:
            print(f'Validated {i + 1}/500 shards', flush=True)
    report = {'status': 'passed', 'source': str(SOURCE.resolve()),
              'changed_parameter': {'n_samples': 25_000_000},
              'num_rows': 25_000_000, 'num_shards': 500,
              'geometry_matches_source': True, 'all_shards_validated': True}
    (DEST / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
