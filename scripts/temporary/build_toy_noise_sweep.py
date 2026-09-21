"""Build four paired toy-noise conditions, excluding flat disk and Mobius."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import torch

from dalg.data.manifold_dataset import (
    EMBEDDING_DIMS,
    INTRINSIC_DIMS,
    MANIFOLD_NAMES,
    ToyManifoldConfig,
    save_toy_manifold_shards,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--points-per-type", type=int, default=30_000)
    args = parser.parse_args()
    torch.set_num_threads(4)
    assert len(MANIFOLD_NAMES) == len(INTRINSIC_DIMS) == len(EMBEDDING_DIMS)

    manifold_types = tuple(name for name in MANIFOLD_NAMES if name not in ("flat_disk", "mobius"))
    config = ToyManifoldConfig(
        ambient_dim=128,
        n_samples=len(manifold_types) * args.points_per_type,
        manifolds_per_type=1,
        manifold_types=manifold_types,
        offset_radius=4.0,
        seed=0,
    )
    ratios = (10, 100, 1_000, 10_000)
    destinations = [args.output_root / f"noise_ratio_{ratio}" for ratio in ratios]
    for destination in destinations:
        if destination.exists():
            raise FileExistsError(destination)

    records = []
    for ratio, destination in zip(ratios, destinations, strict=True):
        condition = replace(config, noise_ratio=float(ratio))
        print(f"Generating {destination}: {condition.n_samples} points", flush=True)
        save_toy_manifold_shards(destination, condition, shard_size=50_000, layer=0)
        records.append({"path": str(destination.resolve()), "config": asdict(condition)})
        print(f"Completed {destination}", flush=True)

    (args.output_root / "sweep.json").write_text(
        json.dumps({"datasets": records}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
