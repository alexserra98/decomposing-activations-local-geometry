"""Generate one noiseless line, circle, and helix with 20,000 points each."""

import argparse
from pathlib import Path

import torch

from dalg.data import ToyManifoldConfig, save_toy_manifold_shards


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-dir", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    config = ToyManifoldConfig(
        ambient_dim=128,
        n_samples=60_000,
        manifolds_per_type=1,
        manifold_types=("segment", "circle", "helix"),
        offset_radius=4.0,
        noise_ratio=None,
        seed=0,
    )
    save_toy_manifold_shards(args.shard_dir, config, shard_size=50_000, layer=0)
    print(f"Saved {config.n_samples:,} noiseless points to {args.shard_dir}", flush=True)


if __name__ == "__main__":
    main()
