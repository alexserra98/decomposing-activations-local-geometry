"""Rebuild years/layer-4 prompt order and evaluate archived K50, tau=.1 models."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
import os
import random
from pathlib import Path
import shutil

import torch

from dalg.data.shard_activations import load_meta_index, stratified_split
from dalg.evaluation.common import load_evaluation_data, load_model
from dalg.evaluation.pc1_density import evaluate_pc1_density, pc1_density_report_valid
from dalg.pipeline.common import _write_json_atomic

BASE = Path('dalg-cache/sae-manifold').resolve()
NAME = 'sae_manifold_years_neutral_llama31_8b_last'
ARCHIVE = BASE / 'pre_holdout' / NAME
CURRENT = BASE / NAME
REBUILT = BASE / 'reconstructed_pre_holdout_years_layer04'
OUTPUT = Path('outputs/evaluations/years_neutral_pre_holdout_pc1_pca75_20261104').resolve()
KINDS = ('hddc', 'kmeans')
SETTINGS = dict(pc1_density=True, pc1_density_space='pca75', pc1_density_neighbors=20,
                pc1_density_random_directions=32, pc1_density_seed=42)


def read(path):
    return json.loads(path.read_text())


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def records(root):
    with (root / 'source/records.jsonl').open() as f:
        return [json.loads(line) for line in f]


def targets():
    result = {}
    for path in sorted(ARCHIVE.glob('*/*/run_spec.json')):
        spec = read(path)
        a = spec['training']['arguments']
        if a['K'] == 50 and a['layer'] == 4 and a.get('surgery_threshold') == .1:
            kind = spec['training']['model_kind']
            assert kind not in result
            result[kind] = path.parent
    assert set(result) == set(KINDS)
    return result


def prepare():
    original = records(ARCHIVE)
    assert len(original) == 20020
    position = {r['row_id']: i for i, r in enumerate(original)}
    assert len(position) == len(original)
    cfg = read(ARCHIVE / 'config.json')
    points = torch.empty((20020, 1, 4096), dtype=torch.bfloat16)
    seen = set()
    sources = []
    for source in (CURRENT, CURRENT / 'test'):
        source_cfg = read(source / 'config.json')
        for field in ('model', 'model_revision', 'tokenizer_revision', 'tokenizer_sha256',
                      'representation', 'd_model', 'window', 'drop_prefix', 'dtype', 'token_selection'):
            assert source_cfg[field] == cfg[field], field
        rows = records(source)
        for path in sorted((source / 'layer04').glob('shard_*.pt')):
            meta = read(source / 'meta' / (path.stem + '.json'))
            tensor = torch.load(path, map_location='cpu', weights_only=True)
            assert tensor.shape == (len(meta['row_indices']), 1, 4096)
            assert tensor.dtype == torch.bfloat16 and torch.isfinite(tensor).all()
            indices = []
            for row in meta['row_indices']:
                record = rows[row]
                i = position[record['row_id']]
                assert i not in seen and record == original[i]
                seen.add(i)
                indices.append(i)
            points[indices] = tensor
            sources.append({'path': str(path), 'sha256': digest(path)})
    assert len(seen) == 20020
    REBUILT.mkdir(exist_ok=False)
    OUTPUT.mkdir(parents=True, exist_ok=False)
    (REBUILT / 'layer04').mkdir()
    shutil.copytree(ARCHIVE / 'meta', REBUILT / 'meta')
    shutil.copytree(ARCHIVE / 'source', REBUILT / 'source')
    cfg['layers'] = [4]
    cfg['reconstruction'] = 'Original prompt order; tensors from re-extracted development/test partitions'
    _write_json_atomic(REBUILT / 'config.json', cfg)
    outputs = []
    for meta_path in sorted((REBUILT / 'meta').glob('shard_*.json')):
        indices = read(meta_path)['row_indices']
        path = REBUILT / 'layer04' / (meta_path.stem + '.pt')
        torch.save(points[indices].clone(), path)
        assert torch.equal(torch.load(path, weights_only=True), points[indices])
        outputs.append({'path': str(path), 'sha256': digest(path)})
    meta = load_meta_index(REBUILT, layer=4)
    assert [m['global_row'] for m in meta] == list(range(20020))
    plan = {'source_archive': str(ARCHIVE), 'reconstructed_shards': str(REBUILT),
            'original_tensor_identity_verified': False, 'prompt_and_tokenization_identity_verified': True,
            'sources': sources, 'reconstructed_files': outputs, 'settings': SETTINGS, 'runs': {}}
    for kind, directory in targets().items():
        stem = 'mfa_model' if kind == 'hddc' else 'kmeans_model'
        split = read(directory / 'val_indices.json')
        assert (split['train_rows'], split['val_rows']) == (18018, 2002)
        dest = OUTPUT / kind
        dest.mkdir()
        assignment_path = directory / f'{stem}_assignments.pt'
        bundle = torch.load(assignment_path, map_location='cpu', weights_only=True)
        # Archived absolute paths are stale; only relocate paths in this working copy.
        bundle['model_path'] = str(directory / f'{stem}.pt')
        bundle['source']['shard_dir'] = str(REBUILT)
        torch.save(bundle, dest / 'assignments_relocated.pt')
        model = load_model(directory, kind)
        data = load_evaluation_data(directory, shard_dir=REBUILT, layer=4, model=model,
                                    model_kind=kind, assignments_path=dest / 'assignments_relocated.pt')
        assert [meta[p]['global_row'] for p in data.val_positions] == split['val_global_rows']
        plan['runs'][kind] = {'run_dir': str(directory), 'checkpoint_sha256': digest(directory / f'{stem}.pt'),
                              'split_sha256': digest(directory / 'val_indices.json'),
                              'original_assignments_sha256': digest(assignment_path),
                              'dataset': data.dataset_summary(4)}
        print(f'Validated {kind}: 18,018 training / 2,002 validation points', flush=True)
    _write_json_atomic(REBUILT / 'reconstruction.json', plan)
    _write_json_atomic(OUTPUT / 'plan.json', plan)
    print(f'Prepared {REBUILT}\nResults: {OUTPUT}', flush=True)


def evaluate(index, reference_seed=None, reference_size=None):
    assert reference_size is None or reference_seed is not None
    kind = KINDS[index]
    plan = read(OUTPUT / 'plan.json')
    run = plan['runs'][kind]
    directory = Path(run['run_dir'])
    stem = 'mfa_model' if kind == 'hddc' else 'kmeans_model'
    assert digest(directory / f'{stem}.pt') == run['checkpoint_sha256']
    assert digest(directory / 'val_indices.json') == run['split_sha256']
    model = load_model(directory, kind)
    data = load_evaluation_data(directory, shard_dir=REBUILT, layer=4, model=model,
                                model_kind=kind, assignments_path=OUTPUT / kind / 'assignments_relocated.pt')
    destination = OUTPUT / kind
    sampling = None
    if reference_seed is not None:
        baseline = read(OUTPUT / kind / 'metrics.json')
        if reference_size is None:
            _, reference_positions = stratified_split(data.meta_index, val_frac=.1, seed=reference_seed)
            assert len(reference_positions) == len(data.val_positions) == 2002
            population = 'all_20020_reconstructed_prompts'
            suffix = f'_shuffle_seed{reference_seed}'
        else:
            assert 20 <= reference_size < len(data.val_positions)
            reference_positions = sorted(random.Random(reference_seed).sample(data.val_positions, reference_size))
            assert len(reference_positions) == len(set(reference_positions)) == reference_size
            assert set(reference_positions).isdisjoint(data.train_positions)
            population = 'original_2002_validation_prompts'
            suffix = f'_val{reference_size}_seed{reference_seed}'
        assert reference_positions != data.val_positions
        sampling = {
            'seed': reference_seed, 'population': population,
            'reference_points': len(reference_positions),
            'overlap_with_original_validation': len(set(reference_positions) & set(data.val_positions)),
            'overlap_with_original_training': len(set(reference_positions) & set(data.train_positions)),
            'interpretation': 'Reference sensitivity diagnostic with fixed model and original PCA',
            'pca_fit_population': 'unchanged_original_18018_training_rows',
            'baseline_report': str(OUTPUT / kind / 'metrics.json'),
        }
        # Keep the original PCA training rows; only change the density-reference rows.
        data = replace(data, val_positions=reference_positions)
        destination = OUTPUT.with_name(OUTPUT.name + suffix) / kind
        destination.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(destination / 'reference_selection.json', {
            **sampling, 'global_rows': [data.meta_index[p]['global_row'] for p in reference_positions],
        })
        print(json.dumps(sampling), flush=True)
    print(f'Evaluating {kind}, PCA75, kNN=20, 32 random directions, seed=42', flush=True)
    metric = evaluate_pc1_density(model, data, model_kind=kind, layer=4, batch_size=4096,
                                  space='pca75', neighbors=20, random_directions=32, seed=42)
    assert pc1_density_report_valid(metric, components=50, cfg=SETTINGS, dataset=data.dataset_summary(4))
    if sampling is not None:
        original_projection = baseline['pc1_density']['projection']
        assert metric['projection']['dimension'] == original_projection['dimension']
        assert math.isclose(metric['projection']['retained_variance_fraction'],
                            original_projection['retained_variance_fraction'], abs_tol=1e-12, rel_tol=0)
    result = {'evaluation': 'pc1_density_only', 'run_id': directory.name, 'model_kind': kind,
              'K': 50, 'surgery_threshold': .1, 'dataset': data.dataset_summary(4),
              'pc1_density': metric, 'provenance': run,
              'original_tensor_identity_verified': False,
              'reference_sampling': sampling,
              'slurm_job_id': os.environ.get('SLURM_JOB_ID')}
    json.dumps(result, allow_nan=False)
    _write_json_atomic(destination / 'metrics.json', result)
    summary = {'model': kind, 'dimension': metric['density_dimension'],
               'retained_variance': metric['projection']['retained_variance_fraction'],
               'retention': math.exp(metric['log_retention']['median']),
               'enrichment': math.exp(metric['log_enrichment']['median'])}
    if sampling is not None:
        summary['reference_sampling'] = sampling
        summary['baseline'] = read(OUTPUT / kind / 'summary.json')
    _write_json_atomic(destination / 'summary.json', summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'evaluate'))
    parser.add_argument('--index', type=int, choices=(0, 1))
    parser.add_argument('--reference-seed', type=int)
    parser.add_argument('--reference-size', type=int, help='Subsample this many original validation rows')
    args = parser.parse_args()
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', '8')))
    if args.action == 'prepare':
        prepare()
    else:
        assert args.index is not None
        evaluate(args.index, reference_seed=args.reference_seed, reference_size=args.reference_size)
