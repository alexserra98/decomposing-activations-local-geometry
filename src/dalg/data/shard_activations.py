"""Shard-aware dataset for the `extract-windows` output.

Usage
-----
    meta = load_meta_index(shard_dir)
    train_pos, val_pos = stratified_split(meta, val_frac=0.05, seed=42)

    train_ds = ShardActivationDataset(
        shard_dir, layer=5, row_subset=train_pos,
        drop_prefix=32, shuffle_shards=True, shuffle_within_shard=True,
    )
    loader = DataLoader(train_ds, batch_size=4096, num_workers=4,
                        pin_memory=True, persistent_workers=True)
    for x, t in loader:          # x: (B, d_model) fp32,  t: (B,) long
        ...

Design
------
- `load_meta_index` reads all meta/shard_*.json once, returning one entry per
  dataset row. That flat index is the input to `stratified_split`, which
  balances the 5% val hold-out across the 17 pile subsets.
- `ShardActivationDataset` is an IterableDataset that memory-maps each shard's
  layer + tokens files (`torch.load(..., mmap=True, weights_only=True)`), slices
  the rows the caller asked for, drops the first `drop_prefix` positions to skip
  under-contextualized early-window tokens, flattens to (N_tok, D) and yields
  token-level samples. 322 GB per layer never enters RAM — only the active
  shard's rows get paged in.
"""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import IterableDataset, get_worker_info


# ── Split helpers ────────────────────────────────────────────────────────────

def load_meta_index(shard_dir) -> List[dict]:
    """Return one entry per dataset row: {shard, row_in_shard, global_row, subset}.

    Reads every meta/shard_*.json in order. Cheap — ~643 tiny JSONs.
    """
    shard_dir = Path(shard_dir)
    out: List[dict] = []
    for meta_path in sorted((shard_dir / "meta").glob("shard_*.json")):
        shard_i = int(meta_path.stem.split("_")[1])
        meta = json.loads(meta_path.read_text())
        for r, row in enumerate(meta["rows"]):
            out.append({
                "shard": shard_i,
                "row_in_shard": r,
                "global_row": meta["row_indices"][r],
                "subset": row["subset"],
            })
    return out


def stratified_split(
    meta_index: Sequence[dict],
    val_frac: float = 0.05,
    seed: int = 42,
) -> Tuple[List[int], List[int]]:
    """Stratified-by-subset train/val split over positions into `meta_index`.

    Each subset independently shuffled; `ceil(n * val_frac)` rows go to val.
    """
    by_subset: Dict[str, List[int]] = defaultdict(list)
    for i, row in enumerate(meta_index):
        by_subset[row["subset"]].append(i)

    rng = random.Random(seed)
    train: List[int] = []
    val: List[int] = []
    for subset in sorted(by_subset):
        positions = by_subset[subset][:]
        rng.shuffle(positions)
        n_val = math.ceil(len(positions) * val_frac)
        val.extend(positions[:n_val])
        train.extend(positions[n_val:])
    return sorted(train), sorted(val)


def per_subset_counts(meta_index: Sequence[dict], positions: Sequence[int]) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    for p in positions:
        counts[meta_index[p]["subset"]] += 1
    return dict(sorted(counts.items()))


# ── Dataset ──────────────────────────────────────────────────────────────────

class ShardActivationDataset(IterableDataset):
    """Streams token-level activations from a sharded extraction directory.

    Parameters
    ----------
    shard_dir : path to the extraction output (NOT a single shard file —
        contains layer{L:02d}/, tokens/, meta/ subdirs).
    layer : which layer's shards to read.
    row_subset : positions into `load_meta_index(shard_dir)`. These are the
        dataset rows to iterate.
    drop_prefix : discard the first N token positions in each window to avoid
        under-contextualized early positions.
    dtype : output dtype for activations (shards are fp16 on disk).
    shuffle_shards : shuffle the order of shards each epoch.
    shuffle_within_shard : shuffle tokens within each shard each epoch.
    seed : base seed; effective seed = seed XOR epoch XOR shard_i.
    """

    def __init__(
        self,
        shard_dir,
        layer: int,
        row_subset: Sequence[int],
        *,
        drop_prefix: int = 32,
        dtype: torch.dtype = torch.float32,
        shuffle_shards: bool = True,
        shuffle_within_shard: bool = True,
        seed: int = 0,
    ):
        self.shard_dir = Path(shard_dir)
        self.layer = layer
        self.drop_prefix = int(drop_prefix)
        self.dtype = dtype
        self.shuffle_shards = shuffle_shards
        self.shuffle_within_shard = shuffle_within_shard
        self.seed = int(seed)
        self.epoch = 0

        cfg = json.loads((self.shard_dir / "config.json").read_text())
        self.window = int(cfg["window"])
        self.d_model = int(cfg["d_model"])
        if self.drop_prefix >= self.window:
            raise ValueError(f"drop_prefix={drop_prefix} ≥ window={self.window}")

        # Build shard → in-shard-row-indices map from meta, restricted to row_subset.
        meta = load_meta_index(self.shard_dir)
        wanted = set(int(p) for p in row_subset)
        bucket: Dict[int, List[int]] = defaultdict(list)
        for pos in wanted:
            m = meta[pos]
            bucket[m["shard"]].append(m["row_in_shard"])
        for k in bucket:
            bucket[k].sort()
        self._shard_rows: Dict[int, List[int]] = dict(bucket)
        self._shards: List[int] = sorted(self._shard_rows)
        self._n_rows = sum(len(v) for v in self._shard_rows.values())

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        # Token-level length hint (pre-shuffle, pre-worker-split).
        return self._n_rows * (self.window - self.drop_prefix)

    # ── iterator ────────────────────────────────────────────────────────────

    def _layer_path(self, shard_i: int) -> Path:
        return self.shard_dir / f"layer{self.layer:02d}" / f"shard_{shard_i:05d}.pt"

    def _tokens_path(self, shard_i: int) -> Path:
        return self.shard_dir / "tokens" / f"shard_{shard_i:05d}.pt"

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        info = get_worker_info()
        wid = 0 if info is None else info.id
        nworkers = 1 if info is None else info.num_workers

        shards = list(self._shards)
        if self.shuffle_shards:
            rng = random.Random((self.seed ^ 0x9E3779B1) + self.epoch)
            rng.shuffle(shards)
        # Worker shard assignment: every Nth shard.
        shards = shards[wid::nworkers]

        sl = self.drop_prefix
        window = self.window

        for shard_i in shards:
            rows = self._shard_rows[shard_i]
            acts = torch.load(self._layer_path(shard_i), mmap=True, weights_only=True)
            toks = torch.load(self._tokens_path(shard_i), mmap=True, weights_only=True)

            # Select rows and drop the under-contextualized prefix.
            X = acts[rows][:, sl:, :].to(self.dtype).reshape(-1, self.d_model)
            T = toks[rows][:, sl:].to(torch.long).reshape(-1)

            n = X.shape[0]
            if self.shuffle_within_shard:
                g = torch.Generator()
                g.manual_seed(self.seed + shard_i * 1009 + self.epoch * 7919)
                perm = torch.randperm(n, generator=g)
                X = X[perm]
                T = T[perm]

            for i in range(n):
                yield X[i], T[i]

            # release the mmap'd tensor references so the OS can reclaim pages
            del acts, toks, X, T
