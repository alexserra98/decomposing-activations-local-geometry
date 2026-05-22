import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dalg.analysis.cluster_assignments import (
    _resolve_device,
    compute_assignments,
)
from dalg.data.shard_activations import (
    ShardActivationBatchDataset,
    load_meta_index,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile MFA cluster assignments on a small shard subset")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--num-rows", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--max-batches", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--drop-prefix", type=int, default=None)
    parser.add_argument("--profile-dir", type=Path, default=Path("outputs/jobs/cluster_assignment_profiles"))
    parser.add_argument(
        "--no-inference-cache", "--slow-responsibilities",
        dest="use_inference_cache",
        action="store_false",
        default=True,
    )
    args = parser.parse_args()

    device = _resolve_device(args.device)
    shard_dir = args.shard_dir
    extract_cfg = json.loads((shard_dir / "config.json").read_text())
    drop_prefix = args.drop_prefix
    if drop_prefix is None:
        drop_prefix = int(extract_cfg.get("drop_prefix", 32))

    meta_index = load_meta_index(shard_dir)
    positions = list(range(min(args.num_rows, len(meta_index))))
    ds = ShardActivationBatchDataset(
        shard_dir,
        layer=args.layer,
        row_subset=positions,
        drop_prefix=drop_prefix,
        batch_size=args.batch_size,
        dtype=torch.float32,
        shuffle_shards=False,
        shuffle_within_shard=False,
        seed=args.seed,
    )
    loader = DataLoader(
        ds,
        batch_size=None,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    args.profile_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    save_path = args.profile_dir / f"{args.model_path.stem}_profile_{stamp}.pt"

    print(
        f"Profiling {args.max_batches} batches from {len(positions)} rows "
        f"with batch_size={args.batch_size} on device={device}."
    )
    t0 = time.perf_counter()
    sizes, assignments, max_resp, peakedness = compute_assignments(
        args.model_path,
        loader,
        device=device,
        max_batches=args.max_batches,
        use_inference_cache=args.use_inference_cache,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    n = int(assignments.numel())
    print(f"Processed {n:,} activations in {elapsed:.2f}s ({n / max(elapsed, 1e-9):,.1f} activations/s).")
    torch.save({
        "cluster_sizes": sizes,
        "assignments": assignments,
        "max_responsibilities": max_resp,
        "peakedness": peakedness,
        "K": int(sizes.numel()),
        "elapsed": elapsed,
    }, save_path)
    print(f"Profile output: {save_path}")


if __name__ == "__main__":
    main()
