"""Materialize a subset-spec activation stream as a NumPy array."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dalg.data.shard_activations import ActivationBatchDataset, load_meta_index
from dalg.data.subset_spec import resolve_spec_positions, split_shard_dir_spec


def main() -> None:
    parser = argparse.ArgumentParser(description="Save streamed activation subset as activations.npy")
    parser.add_argument("--shard-dir", type=str, required=True)
    parser.add_argument("--layer", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--drop-prefix", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-items", type=int, default=None, help="Optional smoke-test cap on emitted activations")
    args = parser.parse_args()

    shard_dir, subset_spec = split_shard_dir_spec(args.shard_dir)
    cfg = json.loads((shard_dir / "config.json").read_text())
    window = int(cfg["window"])
    d_model = int(cfg["d_model"])
    drop_prefix = int(cfg.get("drop_prefix", 32) if args.drop_prefix is None else args.drop_prefix)
    meta_index = load_meta_index(shard_dir, layer=args.layer)
    positions = resolve_spec_positions(meta_index, subset_spec, window=window, drop_prefix=drop_prefix)

    ds = ActivationBatchDataset(
        shard_dir,
        layer=args.layer,
        row_subset=positions,
        batch_size=args.batch_size,
        drop_prefix=drop_prefix,
        dtype=torch.float32,
        shuffle_shards=False,
        shuffle_within_shard=False,
    )
    resolved_items = int(ds.num_items)
    materialized_items = int(resolved_items if args.max_items is None else min(resolved_items, args.max_items))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "activations.npy"
    arr = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32, shape=(materialized_items, d_model))

    loader = DataLoader(ds, batch_size=None, num_workers=args.num_workers)
    offset = 0
    pbar = tqdm(total=materialized_items, desc="materializing activations", unit="rows")
    try:
        for batch in loader:
            if offset >= materialized_items:
                break
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x_np = x.detach().cpu().numpy().astype(np.float32, copy=False)
            take = min(x_np.shape[0], materialized_items - offset)
            arr[offset:offset + take] = x_np[:take]
            offset += take
            pbar.update(take)
    finally:
        pbar.close()
    arr.flush()
    if offset != materialized_items:
        raise RuntimeError(f"wrote {offset:,} rows, expected {materialized_items:,}")

    metadata = {
        "shard_dir": str(shard_dir),
        "subset_spec": subset_spec,
        "layer": int(args.layer),
        "window": window,
        "drop_prefix": drop_prefix,
        "tokens_per_row": window - drop_prefix,
        "resolved_rows": len(positions),
        "resolved_items": resolved_items,
        "materialized_items": materialized_items,
        "selected_rows": len(positions),
        "d_model": d_model,
        "shape": [materialized_items, d_model],
        "dtype": "float32",
        "max_items": args.max_items,
        "activations_path": str(out_path),
    }
    (args.out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved {out_path} with shape {(materialized_items, d_model)}")
    print(f"Saved {args.out_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
