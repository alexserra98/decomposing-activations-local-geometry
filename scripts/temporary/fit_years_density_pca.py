"""Fit the single training-only global PCA75 artifact for years/layer4."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import torch

from dalg.analysis.pc1_density import fit_global_variance_pca
from dalg.data.shard_activations import ActivationBatchDataset, load_meta_index, stratified_split
from run_years_pc1_density import DEFAULT_ACTIVATIONS, discover_runs, load_reference, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--activation-dir', type=Path, default=DEFAULT_ACTIVATIONS)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--threads', type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    root = args.activation_dir.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    runs = discover_runs(root)
    reference, val_ids, extraction = load_reference(root, runs)
    del reference
    meta = load_meta_index(root, layer=4)
    train, _ = stratified_split(meta, val_frac=.1, seed=42)
    train_ids = [meta[p]['global_row'] for p in train]
    stream = ActivationBatchDataset(root, layer=4, row_subset=train, batch_size=512,
                                    dtype=torch.float64, return_metadata=True,
                                    shuffle_shards=False, shuffle_within_shard=False)
    observed_ids = []

    def batches():
        for x, ids, token_positions in stream:
            if token_positions.any():
                raise ValueError('Expected last-token slot zero')
            observed_ids.extend(ids.tolist())
            yield x

    print('Fitting exact global PCA on 18,018 training activations; target variance 75%', flush=True)
    pca = fit_global_variance_pca(batches(), variance_fraction=.75)
    if observed_ids != train_ids or pca['training_count'] != 18018:
        raise ValueError('PCA stream differs from the saved training split')
    pca.update(format='dalg_years_global_pca75_v1',
               created_at=datetime.now(timezone.utc).isoformat(),
               source=dict(activation_dir=str(root), layer=4, dataset='years_neutral',
                           split='train', split_seed=42, val_frac=.1,
                           training_global_rows=torch.tensor(train_ids, dtype=torch.int64),
                           validation_global_rows=torch.tensor(val_ids, dtype=torch.int64),
                           extraction_config_sha256=sha256(root / 'config.json'),
                           source_records_sha256=extraction['source_records_sha256'],
                           layer_shard_sha256={p.name: sha256(p) for p in sorted((root/'layer04').glob('shard_*.pt'))}),
               source_sha256={str(p): sha256(p) for p in
                              [Path(__file__).resolve(), Path(__file__).resolve().parents[2] / 'src/dalg/analysis/pc1_density.py']})
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as f:
        torch.save(pca, f)
    print(f'Saved {output}: d={pca["dimension"]}, variance={pca["retained_variance_fraction"]:.8%}, '
          f'previous={pca["previous_variance_fraction"]:.8%}', flush=True)


if __name__ == '__main__':
    main()
