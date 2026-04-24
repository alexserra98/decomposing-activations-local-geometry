#!/usr/bin/env python3
"""Lightweight benchmark for shard I/O.

This is meant to answer questions like:
- Is reading from scratch / Ceph unusually slow today?
- Is mmap/open cheap but page-in expensive?
- Are token shards or activation shards the bottleneck?

The script is intentionally conservative:
- it touches only a small number of shards by default
- it touches only a small number of rows/tokens per shard
- it can sleep between shard reads to avoid hammering the filesystem

Typical usage:
    PYTHONPATH=src .venv/bin/python scripts/benchmark_shard_io.py \
        --shard-dir /orfeo/scratch/dssc/zenocosini/pile_gemma2b_activations \
        --layer 17
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dalg.data.shard_activations import load_meta_index


def _fmt_bytes(n: float) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(n)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{n} B"


def _fmt_rate(nbytes: float, seconds: float) -> str:
    if seconds <= 0:
        return "inf"
    return f"{_fmt_bytes(nbytes / seconds)}/s"


def _group_rows_by_shard(meta_index: list[dict]) -> dict[int, list[int]]:
    grouped: dict[int, list[int]] = defaultdict(list)
    for row in meta_index:
        grouped[int(row["shard"])].append(int(row["row_in_shard"]))
    for shard_i in grouped:
        grouped[shard_i].sort()
    return dict(grouped)


def _pick_evenly_spaced(items: list[int], n: int) -> list[int]:
    if n >= len(items):
        return items[:]
    if n <= 1:
        return [items[len(items) // 2]]
    out = []
    last = len(items) - 1
    for i in range(n):
        idx = round(i * last / (n - 1))
        out.append(items[idx])
    return sorted(set(out))


def benchmark_shards(
    shard_dir: Path,
    *,
    layer: int,
    max_shards: int,
    rows_per_shard: int,
    tokens_per_row: int,
    drop_prefix: int | None,
    sleep_seconds: float,
) -> int:
    cfg = json.loads((shard_dir / "config.json").read_text())
    window = int(cfg["window"])
    d_model = int(cfg["d_model"])
    effective_drop_prefix = int(cfg.get("drop_prefix", 32) if drop_prefix is None else drop_prefix)
    if effective_drop_prefix >= window:
        raise SystemExit(
            f"drop_prefix={effective_drop_prefix} must be smaller than window={window}"
        )

    meta_t0 = time.perf_counter()
    meta_index = load_meta_index(shard_dir)
    meta_dt = time.perf_counter() - meta_t0
    by_shard = _group_rows_by_shard(meta_index)
    shard_ids = sorted(by_shard)
    selected_shards = _pick_evenly_spaced(shard_ids, max_shards)

    print(f"shard_dir      : {shard_dir}")
    print(f"layer          : {layer}")
    print(f"window         : {window}")
    print(f"d_model        : {d_model}")
    print(f"drop_prefix    : {effective_drop_prefix}")
    print(f"meta rows      : {len(meta_index):,}")
    print(f"meta load time : {meta_dt:.3f}s")
    print(f"total shards   : {len(shard_ids)}")
    print(f"bench shards   : {selected_shards}")
    print(f"rows/shard     : {rows_per_shard}")
    print(f"tokens/row     : {tokens_per_row}")
    print(f"sleep/shard    : {sleep_seconds:.2f}s")
    print()

    touch_times = []
    open_times = []
    bytes_touched_total = 0

    for shard_i in selected_shards:
        rows = _pick_evenly_spaced(by_shard[shard_i], rows_per_shard)
        layer_path = shard_dir / f"layer{layer:02d}" / f"shard_{shard_i:05d}.pt"
        token_path = shard_dir / "tokens" / f"shard_{shard_i:05d}.pt"

        open_t0 = time.perf_counter()
        acts = torch.load(layer_path, mmap=True, weights_only=True)
        toks = torch.load(token_path, mmap=True, weights_only=True)
        open_dt = time.perf_counter() - open_t0

        # Touch a limited region to force actual I/O without reading the whole shard.
        touch_t0 = time.perf_counter()
        tokens_slice = slice(
            effective_drop_prefix,
            min(window, effective_drop_prefix + tokens_per_row),
        )
        x = acts[rows][:, tokens_slice, :].float()
        t = toks[rows][:, tokens_slice].long()
        # Tiny reductions ensure the data is actually paged in.
        checksum = float(x.mean().item()) + float(t.float().mean().item())
        touch_dt = time.perf_counter() - touch_t0

        touched_rows = len(rows)
        touched_tokens = max(0, tokens_slice.stop - tokens_slice.start)
        activation_bytes = touched_rows * touched_tokens * d_model * 2  # fp16 on disk
        token_bytes = touched_rows * touched_tokens * 4  # int32 on disk
        bytes_touched = activation_bytes + token_bytes
        bytes_touched_total += bytes_touched

        open_times.append(open_dt)
        touch_times.append(touch_dt)

        print(
            f"shard {shard_i:05d} | rows={touched_rows:3d} "
            f"| open={open_dt:6.3f}s | touch={touch_dt:6.3f}s "
            f"| approx={_fmt_bytes(bytes_touched):>9s} "
            f"| rate={_fmt_rate(bytes_touched, touch_dt):>12s} "
            f"| checksum={checksum:.6f}"
        )

        del acts, toks, x, t
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    print()
    print(f"open time avg  : {statistics.mean(open_times):.3f}s")
    print(f"touch time avg : {statistics.mean(touch_times):.3f}s")
    print(f"touch time p95 : {max(touch_times):.3f}s")
    print(f"bytes touched  : {_fmt_bytes(bytes_touched_total)}")
    print(
        f"avg touch rate : {_fmt_rate(bytes_touched_total, sum(touch_times))}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Lightweight shard I/O benchmark")
    parser.add_argument("--shard-dir", type=Path, required=True,
                        help="Root directory produced by extract-windows")
    parser.add_argument("--layer", type=int, required=True,
                        help="Layer number to benchmark")
    parser.add_argument("--max-shards", type=int, default=4,
                        help="Number of shards to sample across the dataset")
    parser.add_argument("--rows-per-shard", type=int, default=8,
                        help="Number of rows to touch per sampled shard")
    parser.add_argument("--tokens-per-row", type=int, default=16,
                        help="Number of token positions to touch per sampled row")
    parser.add_argument("--drop-prefix", type=int, default=None,
                        help="Override config drop_prefix; default uses config.json")
    parser.add_argument("--sleep-seconds", type=float, default=0.25,
                        help="Sleep between shards to reduce filesystem pressure")
    args = parser.parse_args()

    return benchmark_shards(
        args.shard_dir,
        layer=args.layer,
        max_shards=args.max_shards,
        rows_per_shard=args.rows_per_shard,
        tokens_per_row=args.tokens_per_row,
        drop_prefix=args.drop_prefix,
        sleep_seconds=args.sleep_seconds,
    )


if __name__ == "__main__":
    raise SystemExit(main())
