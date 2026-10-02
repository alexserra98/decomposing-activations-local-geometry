"""Regenerate the existing ten-manifold sweep at 300,000 points per manifold."""
from __future__ import annotations

import gc
import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch

from dalg.data import ToyManifoldConfig, save_toy_manifold_shards

SOURCE = Path('dalg-cache/assets/toy_manifolds_10types_1each_D128_30Keach_noise_sweep_seed0')
DEST = Path('dalg-cache/assets/toy_manifolds_10types_1each_D128_300Keach_noise_sweep_seed0')
CONDITIONS = {'noiseless': None, 'noise_ratio_10': 10.0,
              'noise_ratio_100': 100.0, 'noise_ratio_1000': 1000.0}
GEOMETRY_KEYS = ('calibration_means', 'calibration_scales', 'raw_max_abs_curvatures',
                 'max_abs_curvatures', 'curvature_radii', 'embeddings',
                 'offset_directions', 'offsets', 'manifold_type_ids')


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def source_inventory():
    return {str(p.relative_to(SOURCE)): (p.stat().st_size, p.stat().st_mtime_ns)
            for p in SOURCE.rglob('*') if p.is_file()}


def same_geometry(left, right):
    for key in GEOMETRY_KEYS:
        a, b = left[key], right[key]
        if isinstance(a, torch.Tensor):
            a, b = (a,), (b,)
        assert len(a) == len(b), key
        for x, y in zip(a, b, strict=True):
            torch.testing.assert_close(x, y, rtol=0, atol=1e-12, msg=key)


def main():
    torch.set_num_threads(4)
    if DEST.exists() and (not DEST.is_dir() or any(DEST.iterdir())):
        raise FileExistsError(f'Destination must be absent or empty: {DEST}')
    before = source_inventory()
    configs, source_configs, hashes = {}, {}, {}
    for name, ratio in CONDITIONS.items():
        path = SOURCE / name / 'config.json'
        saved = json.loads(path.read_text())
        source_configs[name] = saved
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        cfg = dict(saved['generator_config'])
        assert cfg['n_samples'] == 300_000 and cfg['noise_ratio'] == ratio
        assert cfg['ambient_dim'] == 128 and cfg['manifolds_per_type'] == 1
        assert len(cfg['manifold_types']) == 10 and cfg['seed'] == 0
        assert saved['shard_size'] == 50_000 and saved['layers'] == [0]
        cfg['n_samples'] = 3_000_000
        cfg['manifold_types'] = tuple(cfg['manifold_types'])
        configs[name] = ToyManifoldConfig(**cfg)
    base = asdict(configs['noiseless'])
    for name, cfg in configs.items():
        values = asdict(cfg)
        values['noise_ratio'] = None
        assert values == base, name

    DEST.mkdir(parents=True, exist_ok=True)
    generator = Path('src/dalg/data/manifold_dataset.py')
    write_json(DEST / 'sweep.json', {
        'created_at': datetime.now(timezone.utc).isoformat(),
        'source_root': str(SOURCE.resolve()),
        'changed_parameter': {'n_samples': 3_000_000},
        'points_per_manifold': 300_000,
        'generator_sha256': hashlib.sha256(generator.read_bytes()).hexdigest(),
        'script': str(Path(__file__).resolve()),
        'torch_version': torch.__version__,
        'torch_num_threads': torch.get_num_threads(),
        'datasets': [{'path': str((DEST / name).resolve()), 'config': asdict(cfg),
                      'source_config_sha256': hashes[name]}
                     for name, cfg in configs.items()],
    })
    for name, cfg in configs.items():
        print(f'Generating {name}: 3,000,000 points', flush=True)
        save_toy_manifold_shards(DEST / name, cfg, shard_size=50_000, layer=0)
        gc.collect()
        print(f'Completed {name}', flush=True)

    print('Validating all shards and paired noise conditions', flush=True)
    metadata = {}
    report = {'status': 'passed', 'datasets': 4, 'points_per_dataset': 3_000_000,
              'points_per_manifold': 300_000, 'manifold_types': base['manifold_types'],
              'noise_ratios': list(CONDITIONS.values()),
              'pairing_absolute_tolerance': 1e-6, 'per_dataset': {}}
    for name, ratio in CONDITIONS.items():
        root = DEST / name
        config = json.loads((root / 'config.json').read_text())
        expected = dict(source_configs[name])
        expected.update(num_rows=3_000_000, num_shards=60)
        expected['generator_config'] = json.loads(json.dumps(asdict(configs[name])))
        assert config == expected, name
        assert len(list((root / 'layer00').glob('*.pt'))) == 60
        assert len(list((root / 'meta').glob('*.json'))) == 60
        meta = torch.load(root / 'manifold_metadata.pt', weights_only=False, map_location='cpu')
        metadata[name] = meta
        assert meta['config'] == asdict(configs[name])
        ids = meta['row_manifold_ids']
        assert ids.shape == (3_000_000,) and ids.dtype == torch.int64
        assert torch.equal(torch.bincount(ids, minlength=10), torch.full((10,), 300_000))
        original = torch.load(SOURCE / name / 'manifold_metadata.pt', weights_only=False, map_location='cpu')
        same_geometry(meta, original)
        same_geometry(meta, metadata['noiseless'])
        assert torch.equal(ids, metadata['noiseless']['row_manifold_ids'])
        expected_noise = torch.zeros_like(meta['noise_stds']) if ratio is None else meta['curvature_radii'] / ratio
        torch.testing.assert_close(meta['noise_stds'], expected_noise, rtol=0, atol=0)
        report['per_dataset'][name] = {'points': 3_000_000, 'shards': 60,
                                      'points_per_manifold': 300_000,
                                      'max_noise_pairing_error': 0.0}

    for i in range(60):
        tensors = {}
        for name in CONDITIONS:
            root = DEST / name
            x = torch.load(root / 'layer00' / f'shard_{i:05d}.pt', weights_only=True, map_location='cpu')
            assert x.shape == (50_000, 1, 128) and x.dtype == torch.float32
            assert bool(torch.isfinite(x).all()), (name, i)
            tensors[name] = x
            rows = json.loads((root / 'meta' / f'shard_{i:05d}.json').read_text())
            start, end = i * 50_000, (i + 1) * 50_000
            assert rows['start'] == start and rows['end'] == end
            assert rows['row_indices'] == list(range(start, end))
            assert len(rows['rows']) == 50_000
            meta = metadata[name]
            ids = meta['row_manifold_ids'][start:end].tolist()
            for row, manifold_id in zip(rows['rows'], ids, strict=True):
                type_id = int(meta['manifold_type_ids'][manifold_id])
                assert row == {'subset': meta['manifold_types'][type_id],
                               'manifold_id': manifold_id, 'manifold_type_id': type_id,
                               'intrinsic_dim': meta['intrinsic_dims'][type_id]}
        noiseless = tensors['noiseless'].double()
        reference_noise = tensors['noise_ratio_10'].double() - noiseless
        for name, ratio in CONDITIONS.items():
            if ratio is None:
                continue
            error = (tensors[name].double() - noiseless - reference_noise * (10.0 / ratio)).abs().max().item()
            assert error <= 1e-6, (name, i, error)
            entry = report['per_dataset'][name]
            entry['max_noise_pairing_error'] = max(entry['max_noise_pairing_error'], error)
        if (i + 1) % 10 == 0:
            print(f'Validated {i + 1}/60 shards across all conditions', flush=True)

    assert source_inventory() == before, 'Source files changed during generation'
    report['source_unchanged'] = True
    for name in CONDITIONS:
        report['per_dataset'][name]['file_bytes'] = sum(p.stat().st_size for p in (DEST / name).rglob('*') if p.is_file())
    report['dataset_file_bytes'] = sum(entry['file_bytes'] for entry in report['per_dataset'].values())
    write_json(DEST / 'validation.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
