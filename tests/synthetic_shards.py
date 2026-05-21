"""Synthetic activation-shard datasets for testing shard_activations.py.

Two builders that mirror the on-disk layout produced by `extract-windows`:

    <root>/config.json
    <root>/layer05/shard_NNNNN.pt    (rows, window, d_model) float32 tensors
    <root>/meta/shard_NNNNN.json     {"row_indices": [...], "rows": [{"subset": ...}, ...]}

Values are deterministic and human-readable so tests can verify exact
positions, batch contents, and metadata alignment without random seeds:

    activation[row, tok, dim] = shard * 1000 + row_in_shard * 100 + tok * 10 + dim
"""

from __future__ import annotations

import json
from pathlib import Path

import torch


LAYER = 5
WINDOW = 3
D_MODEL = 2


def _fill(shard_i: int, rows: int) -> torch.Tensor:
    x = torch.empty((rows, WINDOW, D_MODEL), dtype=torch.float32)
    for r in range(rows):
        for t in range(WINDOW):
            for d in range(D_MODEL):
                x[r, t, d] = shard_i * 1000 + r * 100 + t * 10 + d
    return x


def _write_shard(root: Path, shard_i: int, tensor: torch.Tensor,
                 global_rows: list[int], subsets: list[str]) -> None:
    layer_dir = root / f"layer{LAYER:02d}"
    meta_dir = root / "meta"
    layer_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    torch.save(tensor, layer_dir / f"shard_{shard_i:05d}.pt")
    (meta_dir / f"shard_{shard_i:05d}.json").write_text(json.dumps({
        "row_indices": global_rows,
        "rows": [{"subset": s} for s in subsets],
    }))


def _write_config(root: Path) -> None:
    (root / "config.json").write_text(json.dumps({
        "window": WINDOW,
        "d_model": D_MODEL,
        "drop_prefix": 0,
    }))


def build_single_shard(root: Path) -> Path:
    """One shard, 4 rows, two subsets (A, B, A, B)."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    _write_config(root)
    tensor = _fill(shard_i=0, rows=4)
    _write_shard(
        root, shard_i=0, tensor=tensor,
        global_rows=[0, 1, 2, 3],
        subsets=["A", "B", "A", "B"],
    )
    return root


def build_multi_shard(root: Path, n_shards: int = 10, rows_per_shard: int = 5) -> Path:
    """`n_shards` shards, each with `rows_per_shard` rows.

    Global rows are contiguous across shards. Subsets alternate A/B within
    each shard so stratified splits have something to stratify on.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    _write_config(root)
    for s in range(n_shards):
        tensor = _fill(shard_i=s, rows=rows_per_shard)
        base = s * rows_per_shard
        _write_shard(
            root, shard_i=s, tensor=tensor,
            global_rows=list(range(base, base + rows_per_shard)),
            subsets=["A" if r % 2 == 0 else "B" for r in range(rows_per_shard)],
        )
    return root


if __name__ == "__main__":
    here = Path(__file__).parent
    single = build_single_shard(here / "fixtures" / "single_shard")
    multi = build_multi_shard(here / "fixtures" / "multi_shard")
    print(f"single -> {single}")
    print(f"multi  -> {multi}")
