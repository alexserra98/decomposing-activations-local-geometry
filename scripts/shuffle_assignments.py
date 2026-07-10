#!/usr/bin/env python
"""Create a shuffled copy of an assignment bundle for null intrinsic-dim checks."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Source assignments .pt bundle")
    parser.add_argument("--output", type=Path, required=True, help="Destination shuffled .pt bundle")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for the permutation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = torch.load(args.input, map_location="cpu", weights_only=True)
    if "assignments" not in data or "cluster_sizes" not in data:
        raise ValueError(f"{args.input} must contain 'assignments' and 'cluster_sizes'")

    assignments = data["assignments"].long().cpu()
    K = int(data.get("K", data["cluster_sizes"].numel()))

    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    permutation = torch.randperm(assignments.numel(), generator=generator)

    shuffled = dict(data)
    shuffled["assignments"] = assignments[permutation]
    shuffled["cluster_sizes"] = torch.bincount(shuffled["assignments"], minlength=K).long()
    if "max_responsibilities" in shuffled:
        shuffled["max_responsibilities"] = shuffled["max_responsibilities"].cpu()[permutation]
    shuffled["shuffle"] = {
        "source_path": str(args.input),
        "seed": int(args.seed),
        "kind": "token_order_permutation",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(shuffled, args.output)
    print(f"Saved shuffled assignments to {args.output}")
    print(f"N={assignments.numel():,} K={K} seed={args.seed}")
    print(f"cluster_sizes sum={int(shuffled['cluster_sizes'].sum().item()):,}")


if __name__ == "__main__":
    main()
