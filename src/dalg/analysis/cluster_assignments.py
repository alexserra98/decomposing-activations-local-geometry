import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from pathlib import Path
from typing import Any, Callable

import torch
from tqdm import tqdm

from dalg.models.mfa import load_mfa


def _resolve_device(device: str | torch.device) -> torch.device:
    requested = torch.device(device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        print(f"Requested device={requested}, but CUDA is not available; falling back to CPU.")
        return torch.device("cpu")
    if requested.type == "mps" and not torch.backends.mps.is_available():
        print(f"Requested device={requested}, but MPS is not available; falling back to CPU.")
        return torch.device("cpu")
    return requested


PEAKEDNESS_METRICS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "entropy":         lambda r: -(r * (r + 1e-8).log()).sum(dim=1),
    "one_minus_max":   lambda r: 1.0 - r.max(dim=1).values,
    "top1_minus_top2": lambda r: ( # TODO: fix this metric, the results are non senical
        lambda s: s[:, 0] - s[:, -1]
    )(r.topk(min(2, r.shape[1]), dim=1).values), 
}


def compute_assignments(
    model_path: Path,
    loader: Any,
    *,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """
    Single-pass streaming over `loader`. Per point, takes the argmax of the
    MFA responsibilities and accumulates:
      - cluster sizes (K,)
      - hard assignments (N,)
      - mean per-cluster peakedness for each metric in `PEAKEDNESS_METRICS`
    """
    model_path = Path(model_path)
    cache_path = model_path.parent / f"{model_path.stem}_assignments.pt"
    if cache_path.exists():
        data = torch.load(cache_path, map_location="cpu")
        if "peakedness" in data:
            return data["cluster_sizes"], data["assignments"], data["peakedness"]

    device = _resolve_device(device)
    model = load_mfa(model_path, map_location="cpu").to(device)
    model.eval()
    K = model.K
    print(f"MFA: K={K} components  D={model.D}  rank={model.q}")

    # Accumulate on `device` so the loop never has to .cpu() (which would
    # force a CUDA sync each batch). One sync at the end.
    sizes = torch.zeros(K, dtype=torch.long, device=device)
    all_assignments: list[torch.Tensor] = []
    peakedness_sums = {
        name: torch.zeros(K, dtype=torch.float32, device=device)
        for name in PEAKEDNESS_METRICS
    }

    with torch.no_grad():
        for batch in tqdm(loader, desc="streaming assignments + peakedness"):
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device, non_blocking=(device.type == "cuda"))
            r = model.responsibilities(x)           # (B, K)
            assign = r.argmax(dim=1)                # stays on device
            sizes += torch.bincount(assign, minlength=K)
            all_assignments.append(assign)
            for name, fn in PEAKEDNESS_METRICS.items():
                peakedness_sums[name].scatter_add_(0, assign, fn(r).float())

    assignments = torch.cat(all_assignments).cpu()
    sizes = sizes.cpu()
    peakedness = {
        name: s.cpu() / sizes.float().clamp(min=1)
        for name, s in peakedness_sums.items()
    }

    print(f"\nCluster sizes — min={sizes.min().item()}  "
          f"max={sizes.max().item()}  "
          f"mean={sizes.float().mean():.1f}  "
          f"median={sizes.float().median():.1f}")
    print(f"Empty clusters: {(sizes == 0).sum().item()}")

    torch.save({
        "cluster_sizes": sizes,
        "assignments": assignments,
        "peakedness": peakedness,
        "K": K,
    }, cache_path)

    return sizes, assignments, peakedness

def main() -> None:
    import argparse
    import json

    from torch.utils.data import DataLoader

    from dalg.data.shard_activations import (
        ShardActivationDataset,
        load_meta_index,
    )

    parser = argparse.ArgumentParser(description="Compute MFA cluster assignments for sharded activations")
    parser.add_argument("--model-path", type=Path, required=True, help="Path to mfa_model.pt")
    parser.add_argument("--shard-dir", type=Path, required=True, help="Directory produced by extract-windows")
    parser.add_argument("--layer", type=int, required=True, help="Layer index to stream from shard-dir")
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--drop-prefix", type=int, default=None)
    args = parser.parse_args()
    device = _resolve_device(args.device)

    shard_dir = args.shard_dir
    extract_cfg = json.loads((shard_dir / "config.json").read_text())
    drop_prefix = args.drop_prefix
    if drop_prefix is None:
        drop_prefix = int(extract_cfg.get("drop_prefix", 32))

    meta_index = load_meta_index(shard_dir)
    positions = list(range(len(meta_index)))
    print(f"shard_dir={shard_dir}  layer={args.layer}  rows={len(positions):,}")

    ds = ShardActivationDataset(
        shard_dir,
        layer=args.layer,
        row_subset=positions,
        drop_prefix=drop_prefix,
        shuffle_shards=False, shuffle_within_shard=False,
        seed=args.seed,
        dtype=torch.float32,
    )
    loader = DataLoader(
        ds, batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    compute_assignments(args.model_path, loader, device=device)


if __name__ == "__main__":
    main()
