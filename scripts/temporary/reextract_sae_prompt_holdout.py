"""Re-extract the three SAE prompt datasets with a fixed prompt-level test split."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil

from dalg.data.prompt_activations import load_prompt_activation_index, load_prompt_records
from dalg.data.shard_activations import stratified_split
from dalg.llm.prompt_extraction import (
    DTYPES, _json_text, _sha256, _shard_complete, _write_text, run_prompt_extraction,
)

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / 'dalg-cache/sae-manifold'
NAMES = ('temperature_02d', 'years_neutral', 'temperature_02d_years_neutral')


def prompt_split(records, fraction=0.2, seed=42):
    """Split rows independently within each subset, consistently across datasets."""
    if not 0 < fraction < 1:
        raise ValueError('test fraction must be between zero and one')
    development, test = [], []
    for subset in sorted({record['subset'] for record in records}):
        positions = [i for i, record in enumerate(records) if record['subset'] == subset]
        # Reset the RNG per subset so a combined dataset retains standalone membership.
        dev, heldout = stratified_split(records, val_frac=fraction, seed=seed, positions=positions)
        if not dev or not heldout:
            raise ValueError(f'{subset} needs nonempty development and test partitions')
        development.extend(dev)
        test.extend(heldout)
    return sorted(development), sorted(test)


def activation_path(root, name):
    return root / f'sae_manifold_{name}_llama31_8b_last'


def prepare(root, name, fraction=0.2, seed=42):
    """Persist exact row membership and extraction settings before submitting jobs."""
    dataset = root / 'dataset' / name
    records = load_prompt_records(dataset)
    source_hash = _sha256(dataset / 'prompts.jsonl')
    config_path = activation_path(root, name) / 'config.json'
    config = json.loads(config_path.read_text())
    assert config['dataset_sha256'] == source_hash, 'Expected the original full-population extraction'
    original_metadata = json.loads((dataset / 'metadata.json').read_text())
    development, test = prompt_split(records, fraction, seed)
    plan = {
        'version': 1, 'dataset': name, 'unit': 'prompt_row',
        'stratification': 'subset', 'test_fraction': fraction, 'seed': seed,
        'source_sha256': source_hash, 'source_num_rows': len(records),
        'previous_config_sha256': _sha256(config_path), 'extraction': config,
        'partitions': {},
    }
    files = {}
    for role, positions in [('development', development), ('test', test)]:
        selected = [records[i] for i in positions]
        prompts = ''.join(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n' for row in selected)
        partition = {
            'role': role, 'unit': 'prompt_row', 'stratification': 'subset',
            'test_fraction': fraction, 'seed': seed, 'source_sha256': source_hash,
            'source_num_rows': len(records), 'source_row_ids': [row['row_id'] for row in selected],
        }
        metadata = {
            **original_metadata, 'total_rows': len(selected),
            'counts': dict(sorted(Counter(row['subset'] for row in selected).items())),
            'partition': partition,
        }
        files[dataset / role / 'prompts.jsonl'] = prompts
        files[dataset / role / 'metadata.json'] = _json_text(metadata)
        plan['partitions'][role] = partition
    files[dataset / 'split.json'] = _json_text(plan)
    for path, text in files.items():
        if path.exists() and path.read_text() != text:
            raise FileExistsError(f'Refusing to change an existing split: {path}')
    for path, text in files.items():
        _write_text(path, text)
    print(f'{name}: development={len(development):,}, test={len(test):,}', flush=True)
    return plan


def validate_partition(directory, dataset, layers):
    """Verify every shard plus the original prompt/label/row-ID correspondence."""
    records = load_prompt_records(dataset)
    config = json.loads((directory / 'config.json').read_text())
    progress = json.loads((directory / 'progress.json').read_text())
    assert progress['complete'] and progress['completed'] == list(range(config['num_shards']))
    assert config['num_rows'] == len(records) and config['layers'] == layers
    for path, key in [
        (dataset / 'prompts.jsonl', 'dataset_sha256'),
        (dataset / 'metadata.json', 'source_metadata_sha256'),
        (directory / config['source_records'], 'source_records_sha256'),
        (directory / config['source_metadata'], 'source_metadata_sha256'),
    ]:
        assert _sha256(path) == config[key], path
    for layer in layers:
        assert len(list((directory / f'layer{layer:02d}').glob('shard_*.pt'))) == config['num_shards']
    for shard in range(config['num_shards']):
        meta = json.loads((directory / 'meta' / f'shard_{shard:05d}.json').read_text())
        assert _shard_complete(directory, shard, layers, meta, config['d_model'], DTYPES[config['dtype']])
    index = load_prompt_activation_index(directory, layer=layers[0])
    assert len(index) == len(records)
    for record, actual in zip(records, index):
        assert actual['source_row_id'] == record['row_id']
        assert all(actual[field] == record[field] for field in ('prompt', 'labels', 'subset'))


def run(root, name, *, preview=False, device='cuda'):
    dataset = root / 'dataset' / name
    plan = json.loads((dataset / 'split.json').read_text())
    assert _sha256(dataset / 'prompts.jsonl') == plan['source_sha256']
    original = load_prompt_records(dataset)
    by_id = {row['row_id']: row for row in original}
    membership = []
    for role in ('development', 'test'):
        rows = load_prompt_records(dataset / role)
        ids = [row['row_id'] for row in rows]
        assert ids == plan['partitions'][role]['source_row_ids']
        assert all(row == by_id[row['row_id']] for row in rows)
        membership.append(set(ids))
    assert not membership[0] & membership[1]
    assert membership[0] | membership[1] == set(by_id)
    destination = activation_path(root, name)
    staging = root / f'.reextract_{name}'
    archive = root / 'pre_holdout' / destination.name
    config = plan['extraction']
    marker = destination / 'holdout_split.json'
    already_replaced = marker.exists() and json.loads(marker.read_text()) == plan
    if not already_replaced:
        assert _sha256(destination / 'config.json') == plan['previous_config_sha256']
        assert not archive.exists(), f'Archive already exists: {archive}'
        for role, output in [('development', staging), ('test', staging / 'test')]:
            run_prompt_extraction(
                dataset=dataset / role, out_dir=output,
                model_name=config['model'], revision=config['model_revision'],
                layers=config['layers'], dtype=config['dtype'], device=device,
                batch_size=8, shard_size=config['shard_size'], max_length=config['max_length'],
                cache_dir=root.parent / 'huggingface/hub', preview=preview,
            )
        if preview:
            return
        for role, output in [('development', staging), ('test', staging / 'test')]:
            validate_partition(output, dataset / role, config['layers'])
        _write_text(staging / 'holdout_split.json', _json_text(plan))
        # Keep old models and their provenance away from the new training population.
        archive.parent.mkdir(exist_ok=True)
        destination.rename(archive)
        try:
            staging.rename(destination)
        except Exception:
            archive.rename(destination)
            raise
    else:
        for role, output in [('development', destination), ('test', destination / 'test')]:
            validate_partition(output, dataset / role, config['layers'])
    # Only obsolete activation tensors are deleted; old fitted models are retained.
    for layer in config['layers']:
        path = archive / f'layer{layer:02d}'
        if path.exists():
            shutil.rmtree(path)
    _write_text(archive / 'ARCHIVE.txt',
                'Pre-holdout fitted models and extraction provenance. Original activation tensors\n'
                'were replaced. These models used the full prompt population and must not be\n'
                'used as models trained with the new held-out test split. Saved run paths are historical.\n')
    print(f'REPLACED: {destination}\nTEST: {destination / "test"}\nOLD MODELS: {archive}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run'))
    parser.add_argument('--name', choices=NAMES)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--test-fraction', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--preview', action='store_true')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.action == 'prepare':
        for name in ([args.name] if args.name else NAMES):
            prepare(args.root, name, args.test_fraction, args.seed)
    else:
        if args.name is None:
            parser.error('run requires --name')
        run(args.root, args.name, preview=args.preview, device=args.device)


if __name__ == '__main__':
    main()
