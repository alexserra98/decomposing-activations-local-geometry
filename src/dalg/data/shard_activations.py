"""Shard-aware dataset for the `extract-windows` output.

Usage
-----
    meta = load_meta_index(shard_dir)
    train_pos, val_pos = stratified_split(meta, val_frac=0.05, seed=42)

    train_ds = ShardActivationBatchDataset(
        shard_dir, layer=5, row_subset=train_pos,
        drop_prefix=32, shuffle_shards=True, shuffle_within_shard=True,
        batch_size=4096,
    )
    loader = DataLoader(train_ds, batch_size=None, num_workers=4,
                        pin_memory=True, persistent_workers=True)
    for x in loader:             # x: (B, d_model) fp32
        ...

Design
------
- `load_meta_index` reads all meta/shard_*.json once, returning one entry per
  dataset row. That flat index is the input to `stratified_split`, which
  balances the 5% val hold-out across the 17 pile subsets.
- `ShardActivationBatchDataset` is the high-throughput path for training and
  analysis. It yields already-batched activation tensors, with optional
  `(global_row, tok_pos)` metadata for interpretation/indexing.
"""

from __future__ import annotations

import json
import math
import random
from bisect import bisect_right
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


def _build_shard_row_pairs(meta_index: Sequence[dict], row_subset: Sequence[int]) -> Dict[int, List[Tuple[int, int]]]:
    bucket: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    for pos in row_subset:
        m = meta_index[int(pos)]
        bucket[int(m["shard"])].append((int(m["row_in_shard"]), int(m["global_row"])))
    for pairs in bucket.values():
        pairs.sort()
    return dict(bucket)


class ShardActivationBatchDataset(IterableDataset):
    """Streams pre-batched activations from sharded extraction outputs.

    This is the fast path for workflows that do not need token ids. Set
    `return_metadata=True` for interpretation-style passes that need to map
    activations back to the original window dataset.

    Because this dataset already yields batches, wrap it in a DataLoader with
    `batch_size=None`.
    """

    def __init__(
        self,
        shard_dir,
        layer: int,
        row_subset: Sequence[int],
        *,
        batch_size: int,
        drop_prefix: int = 32,
        dtype: torch.dtype = torch.float32,
        return_metadata: bool = False,
        shuffle_shards: bool = True,
        shuffle_within_shard: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ):
        self.shard_dir = Path(shard_dir)
        self.layer = int(layer)
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.drop_prefix = int(drop_prefix)
        self.dtype = dtype
        self.return_metadata = bool(return_metadata)
        self.shuffle_shards = shuffle_shards
        self.shuffle_within_shard = shuffle_within_shard
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0

        cfg = json.loads((self.shard_dir / "config.json").read_text())
        self.window = int(cfg["window"])
        self.d_model = int(cfg["d_model"])
        if self.drop_prefix >= self.window:
            raise ValueError(f"drop_prefix={drop_prefix} ≥ window={self.window}")

        meta = load_meta_index(self.shard_dir)
        self._shard_row_pairs = _build_shard_row_pairs(meta, row_subset)
        self._shards = sorted(self._shard_row_pairs)
        self._n_rows = sum(len(v) for v in self._shard_row_pairs.values())
        self.tokens_per_row = self.window - self.drop_prefix
        self._n_items = self._n_rows * self.tokens_per_row
        self._shard_offsets: List[int] = []
        total = 0
        for shard_i in self._shards:
            total += len(self._shard_row_pairs[shard_i]) * self.tokens_per_row
            self._shard_offsets.append(total)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        if self.drop_last:
            return self._n_items // self.batch_size
        return math.ceil(self._n_items / self.batch_size)

    @property
    def num_items(self) -> int:
        """Number of token-level activation vectors in canonical flat order."""
        return self._n_items

    def __getitem__(self, index: int):
        """Return the `index`-th activation in canonical unshuffled flat order.

        This is meant for debugging and interpretation lookups. Iteration may
        shuffle shards/tokens, but random access always follows sorted shard
        order, sorted row order within each shard, then increasing token
        position after `drop_prefix`.
        """
        if isinstance(index, torch.Tensor):
            index = int(index.item())
        index = int(index)
        if index < 0:
            index += self._n_items
        if index < 0 or index >= self._n_items:
            raise IndexError(f"index {index} out of range for {self._n_items} flattened activations")

        shard_pos = bisect_right(self._shard_offsets, index)
        prev_offset = 0 if shard_pos == 0 else self._shard_offsets[shard_pos - 1]
        shard_i = self._shards[shard_pos]
        local = index - prev_offset
        row_offset = local // self.tokens_per_row
        tok_pos = self.drop_prefix + (local % self.tokens_per_row)
        row_in_shard, global_row = self._shard_row_pairs[shard_i][row_offset]

        acts = torch.load(self._layer_path(shard_i), mmap=True, weights_only=True)
        x = acts[row_in_shard, tok_pos].to(self.dtype).clone()

        if self.return_metadata:
            return x, int(global_row), int(tok_pos)
        return x

    def _layer_path(self, shard_i: int) -> Path:
        return self.shard_dir / f"layer{self.layer:02d}" / f"shard_{shard_i:05d}.pt"

    def __iter__(self) -> Iterator[torch.Tensor | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        info = get_worker_info()
        wid = 0 if info is None else info.id
        nworkers = 1 if info is None else info.num_workers

        shards = list(self._shards)
        if self.shuffle_shards:
            rng = random.Random((self.seed ^ 0x9E3779B1) + self.epoch)
            rng.shuffle(shards)
        shards = shards[wid::nworkers]

        sl = self.drop_prefix
        for shard_i in shards:
            pairs = self._shard_row_pairs[shard_i]
            rows = [row for row, _global_row in pairs]
            acts = torch.load(self._layer_path(shard_i), mmap=True, weights_only=True)
            X = acts[rows][:, sl:, :].to(self.dtype).reshape(-1, self.d_model)

            global_rows = None
            tok_pos = None
            if self.return_metadata:
                meta_rows = torch.tensor([global_row for _row, global_row in pairs], dtype=torch.long)
                positions = torch.arange(sl, self.window, dtype=torch.long)
                global_rows = meta_rows[:, None].expand(-1, positions.numel()).reshape(-1)
                tok_pos = positions[None, :].expand(len(rows), -1).reshape(-1)

            n = X.shape[0]
            if self.shuffle_within_shard:
                g = torch.Generator()
                g.manual_seed(self.seed + shard_i * 1009 + self.epoch * 7919)
                perm = torch.randperm(n, generator=g)
                X = X[perm]
                if global_rows is not None and tok_pos is not None:
                    global_rows = global_rows[perm]
                    tok_pos = tok_pos[perm]

            stop = n if not self.drop_last else (n // self.batch_size) * self.batch_size
            for start in range(0, stop, self.batch_size):
                xb = X[start:start + self.batch_size]
                if global_rows is None or tok_pos is None:
                    yield xb
                else:
                    yield xb, global_rows[start:start + self.batch_size], tok_pos[start:start + self.batch_size]

            del acts, X, global_rows, tok_pos
