"""Activation batch dataset for extracted activation shards.

The common large-run layout is the output of ``extract-windows``:

    <root>/config.json
    <root>/layer05/shard_00000.pt
    <root>/layer05/shard_00001.pt
    <root>/meta/shard_00000.json

``ActivationBatchDataset`` also handles a layer directory directly, a single
``shard_*.pt`` tensor, and shard metadata synthesis for small tests. When
metadata is absent, rows are treated as belonging to subset ``"all"`` and
global rows are assigned in tensor order.
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


# Split helpers

def _read_json(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def _parse_shard_id(path: Path) -> int:
    stem = path.stem
    if stem.startswith("shard_"):
        return int(stem.split("_", 1)[1])
    return 0


def _resolve_activation_layout(activation_dir, layer: int) -> dict:
    """Resolve the extract-windows layout under ``<root>``.

    Returns the parsed ``<root>/config.json`` plus the sorted list of
    ``<root>/layerNN/shard_*.pt`` tensors for the requested layer.
    """
    root = Path(activation_dir)
    layer_dir = root / f"layer{int(layer):02d}"
    shard_paths = sorted(layer_dir.glob("shard_*.pt"))
    if not shard_paths:
        raise FileNotFoundError(f"No shard_*.pt files in {layer_dir}")
    return {
        "root": root,
        "shard_paths": shard_paths,
        "config": _read_json(root / "config.json"),
    }


def load_meta_index(activation_dir, layer: int) -> List[dict]:
    """Return one metadata entry per activation row from the extract-windows layout.

    Reads ``<root>/meta/shard_NNNNN.json`` files written by ``extract-windows``.
    Each shard meta JSON has ``row_indices`` (global row IDs) and ``rows``
    (per-row attributes including ``subset``, used by ``stratified_split``).
    Each returned entry has ``shard``, ``row_in_shard``, ``global_row``, and
    ``subset``.
    """
    layout = _resolve_activation_layout(activation_dir, layer=layer)
    meta_dir = layout["root"] / "meta"
    out: List[dict] = []
    for shard_path in layout["shard_paths"]:
        shard_i = _parse_shard_id(shard_path)
        meta = json.loads((meta_dir / f"shard_{shard_i:05d}.json").read_text())
        for r, global_row in enumerate(meta["row_indices"]):
            out.append({
                "shard": shard_i,
                "row_in_shard": r,
                "global_row": int(global_row),
                "subset": meta["rows"][r].get("subset", "all"),
            })
    return out


def stratified_split(
    meta_index: Sequence[dict],
    val_frac: float = 0.05,
    seed: int = 42,
    positions: Optional[Sequence[int]] = None,
) -> Tuple[List[int], List[int]]:
    """Stratified-by-subset train/val split over positions into ``meta_index``.

    When ``positions`` is given, only those positions are split (original
    indices into ``meta_index`` are preserved); otherwise all rows are used.
    """
    idxs = range(len(meta_index)) if positions is None else positions
    by_subset: Dict[str, List[int]] = defaultdict(list)
    for i in idxs:
        by_subset[meta_index[int(i)].get("subset", "all")].append(int(i))

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
        counts[meta_index[int(p)].get("subset", "all")] += 1
    return dict(sorted(counts.items()))


def _build_shard_row_pairs(
    meta_index: Sequence[dict],
    row_subset: Sequence[int],
) -> Dict[int, List[Tuple[int, int]]]:
    """Bucket positions in ``meta_index`` by shard.

    Returns ``{shard: [(row_in_shard, global_row), ...]}`` with each list sorted
    by ``row_in_shard`` so per-shard reads stay in canonical order.
    """
    bucket: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    for pos in row_subset:
        m = meta_index[int(pos)]
        bucket[int(m["shard"])].append((int(m["row_in_shard"]), int(m["global_row"])))
    for pairs in bucket.values():
        pairs.sort()
    return dict(bucket)


class ActivationBatchDataset(IterableDataset):
    """Streams activation batches with optional row/token metadata.

    Iteration yields already-batched activation tensors, so use
    ``DataLoader(dataset, batch_size=None, num_workers=0)``. Random access via
    ``dataset[i]`` returns the ``i``-th token-level activation in canonical
    order, which is useful for tests and interpretation lookups.
    """

    def __init__(
        self,
        activation_dir,
        layer: int,
        row_subset: Optional[Sequence[int]] = None,
        *,
        batch_size: int,
        drop_prefix: Optional[int] = None,
        dtype: torch.dtype = torch.float32,
        return_metadata: bool = False,
        shuffle_shards: bool = True,
        shuffle_within_shard: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ):
        self.activation_dir = Path(activation_dir)
        self.layer = int(layer)
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.dtype = dtype
        self.return_metadata = bool(return_metadata)
        self.shuffle_shards = bool(shuffle_shards)
        self.shuffle_within_shard = bool(shuffle_within_shard)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0

        layout = _resolve_activation_layout(self.activation_dir, layer=self.layer)
        self.root = layout["root"]
        self._shard_paths = {
            _parse_shard_id(path): path for path in layout["shard_paths"]
        }
        cfg = layout["config"]

        sample = torch.load(layout["shard_paths"][0], mmap=True, weights_only=True)
        if sample.ndim != 3:
            raise ValueError("activation shards must have shape (rows, window, d_model)")
        self.window = int(cfg.get("window", sample.shape[1]))
        self.d_model = int(cfg.get("d_model", sample.shape[2]))
        self.drop_prefix = int(cfg.get("drop_prefix", 0) if drop_prefix is None else drop_prefix)
        if self.drop_prefix < 0 or self.drop_prefix >= self.window:
            raise ValueError(f"drop_prefix={self.drop_prefix} must be in [0, {self.window})")

        meta = load_meta_index(self.activation_dir, layer=self.layer)
        if row_subset is None:
            row_subset = range(len(meta))
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
        """Return one activation in canonical unshuffled flat order."""
        if isinstance(index, torch.Tensor):
            index = int(index.item())
        index = int(index)
        if index < 0:
            # i.e. dataset[-1]
            index += self._n_items
        if index < 0 or index >= self._n_items:
            raise IndexError(f"index {index} out of range for {self._n_items} flattened activations")

        shard_i, row_in_shard, global_row, tok_pos = self._locate(index)
        x = self._load_one(shard_i, row_in_shard, tok_pos)
        if self.return_metadata:
            return x, int(global_row), int(tok_pos)
        return x

    def _locate(self, index: int) -> Tuple[int, int, int, int]:
        """
        Given a flat token-level index in [0, num_items), return the corresponding
        shard, row_in_shard, global_row, and tok_pos.
        ------------ Explanation of the indexing logic -------------

        shard:           0    1    2    3    4    
        items/shard:    15   15   15   15   15   
        _shard_offsets [15,  30,  45,  60,  75]
        bisect_right is just a binary search for "In which shard does this index fall?"

        """
        shard_pos = bisect_right(self._shard_offsets, index) # which shard does the index fall into?
        prev_offset = 0 if shard_pos == 0 else self._shard_offsets[shard_pos - 1] # the starting index of the current shard
        shard_i = self._shards[shard_pos]
        local = index - prev_offset # index within the shard
        row_offset = local // self.tokens_per_row # which row within the shard
        tok_pos = self.drop_prefix + (local % self.tokens_per_row) # reconstruct original token position (before drop prefix)
        row_in_shard, global_row = self._shard_row_pairs[shard_i][row_offset]
        return shard_i, row_in_shard, global_row, tok_pos

    def _load_one(self, shard_i: int, row_in_shard: int, tok_pos: int) -> torch.Tensor:
        acts = torch.load(self._layer_path(shard_i), mmap=True, weights_only=True)
        return acts[row_in_shard, tok_pos].to(self.dtype).clone()

    def _layer_path(self, shard_i: int) -> Path:
        return self._shard_paths[int(shard_i)]

    def __iter__(self) -> Iterator[torch.Tensor | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Yield batched activation tensors one shard at a time.

        For each shard the loader: loads the full shard (memory-mapped), drops
        the prefix tokens, flattens rows*tokens into a (N, d_model) matrix,
        optionally permutes rows, and emits contiguous slices of `batch_size`.
        Metadata (global_row, tok_pos) is yielded alongside when
        `return_metadata=True`.

        Multi-worker sharding: as an IterableDataset, each DataLoader worker
        runs `__iter__` on its own copy of the dataset. Without partitioning,
        every worker would iterate the full shard list and yield duplicate
        batches. `shards[wid::nworkers]` splits the shard list disjointly
        across workers (worker w sees shards w, w+nworkers, w+2*nworkers, ...).
        When `num_workers=0`, `get_worker_info()` returns None, so
        `wid=0, nworkers=1` and the slice is a no-op.
        """
        info = get_worker_info()
        wid = 0 if info is None else info.id
        nworkers = 1 if info is None else info.num_workers

        shards = list(self._shards)
        if self.shuffle_shards:
            rng = random.Random((self.seed ^ 0x9E3779B1) + self.epoch)
            rng.shuffle(shards)
        shards = shards[wid::nworkers]

        for shard_i in shards:
            pairs = list(self._shard_row_pairs[shard_i])
            X, global_rows, tok_pos = self._shard_rows(shard_i, pairs)

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

            del X, global_rows, tok_pos

    def _shard_rows(self, shard_i: int, pairs: List[Tuple[int, int]]):
        rows = [row for row, _global_row in pairs]
        acts = torch.load(self._layer_path(shard_i), mmap=True, weights_only=True)
        X = acts[rows][:, self.drop_prefix:, :].to(self.dtype).reshape(-1, self.d_model)
        if not self.return_metadata:
            return X, None, None

        meta_rows = torch.tensor([global_row for _row, global_row in pairs], dtype=torch.long)
        positions = torch.arange(self.drop_prefix, self.window, dtype=torch.long)
        global_rows = meta_rows[:, None].expand(-1, positions.numel()).reshape(-1)
        tok_pos = positions[None, :].expand(len(rows), -1).reshape(-1)
        return X, global_rows, tok_pos


ShardActivationBatchDataset = ActivationBatchDataset
