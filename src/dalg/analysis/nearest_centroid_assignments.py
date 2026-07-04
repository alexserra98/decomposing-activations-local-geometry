"""Nearest-centroid hard assignments for activation datasets.

This mirrors the streaming shape of ``cluster_assignments.py``, but uses
Euclidean nearest-centroid assignment instead of MFA responsibilities.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def _resolve_device(device: str | torch.device) -> torch.device:
    requested = torch.device(device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        print(f"Requested device={requested}, but CUDA is not available; falling back to CPU.")
        return torch.device("cpu")
    if requested.type == "mps" and not torch.backends.mps.is_available():
        print(f"Requested device={requested}, but MPS is not available; falling back to CPU.")
        return torch.device("cpu")
    return requested


def _as_tensor(x: Any) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    return torch.as_tensor(x)


def _iter_batches(loader_or_array: Any, batch_size: int) -> Iterator[torch.Tensor]:
    if isinstance(loader_or_array, np.ndarray):
        for start in range(0, len(loader_or_array), batch_size):
            yield torch.from_numpy(np.array(loader_or_array[start:start + batch_size], dtype=np.float32, copy=True))
        return
    if isinstance(loader_or_array, torch.Tensor):
        for start in range(0, loader_or_array.shape[0], batch_size):
            yield loader_or_array[start:start + batch_size]
        return
    for batch in loader_or_array:
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        yield _as_tensor(x)


@torch.no_grad()
def compute_nearest_centroid_assignments(
    centroids: torch.Tensor | np.ndarray,
    loader_or_array: Any,
    *,
    device: str | torch.device = "cpu",
    batch_size: int = 8192,
    max_batches: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Assign each activation to its nearest Euclidean centroid.

    Returns ``(cluster_sizes, assignments, min_distances)`` on CPU.
    """
    device = _resolve_device(device)
    C = _as_tensor(centroids).to(device=device, dtype=torch.float32)
    if C.ndim != 2:
        raise ValueError(f"centroids must have shape (K, D), got {tuple(C.shape)}")
    K, D = int(C.shape[0]), int(C.shape[1])
    if K <= 0:
        raise ValueError("centroids must contain at least one row")

    sizes = torch.zeros(K, dtype=torch.long, device=device)
    assignment_chunks: list[torch.Tensor] = []
    distance_chunks: list[torch.Tensor] = []
    c2 = (C * C).sum(dim=1).unsqueeze(0)

    for batch_idx, x in enumerate(tqdm(_iter_batches(loader_or_array, batch_size), desc="nearest-centroid assignments")):
        if max_batches is not None and batch_idx >= max_batches:
            break
        x = x.to(device=device, dtype=torch.float32, non_blocking=(device.type == "cuda"))
        if x.ndim != 2 or x.shape[1] != D:
            raise ValueError(f"batch must have shape (B, {D}), got {tuple(x.shape)}")
        x2 = (x * x).sum(dim=1, keepdim=True)
        d2 = (x2 + c2 - 2.0 * (x @ C.T)).clamp_min_(0.0)
        min_d2, assign = d2.min(dim=1)
        sizes += torch.bincount(assign, minlength=K)
        assignment_chunks.append(assign.cpu())
        distance_chunks.append(min_d2.sqrt().cpu())

    assignments = torch.cat(assignment_chunks) if assignment_chunks else torch.empty(0, dtype=torch.long)
    min_distances = torch.cat(distance_chunks) if distance_chunks else torch.empty(0, dtype=torch.float32)
    return sizes.cpu(), assignments, min_distances


def _load_centroids(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path)
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        for key in ("centroids", "medoids", "cluster_centers"):
            if key in obj:
                obj = obj[key]
                break
    return _as_tensor(obj).cpu().numpy()


def _loader_from_shards(args, device: torch.device):
    from dalg.data.shard_activations import ActivationBatchDataset, load_meta_index
    from dalg.data.subset_spec import resolve_spec_positions, split_shard_dir_spec

    shard_dir, subset_spec = split_shard_dir_spec(args.shard_dir)
    extract_cfg = json.loads((shard_dir / "config.json").read_text())
    window = int(extract_cfg["window"])
    drop_prefix = args.drop_prefix
    if drop_prefix is None:
        drop_prefix = int(extract_cfg.get("drop_prefix", 32))
    meta_index = load_meta_index(shard_dir, layer=args.layer)
    positions = resolve_spec_positions(meta_index, subset_spec, window=window, drop_prefix=drop_prefix)
    ds = ActivationBatchDataset(
        shard_dir,
        layer=args.layer,
        row_subset=positions,
        drop_prefix=drop_prefix,
        batch_size=args.batch_size,
        dtype=torch.float32,
        shuffle_shards=False,
        shuffle_within_shard=False,
    )
    loader = DataLoader(ds, batch_size=None, num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    return loader, {
        "shard_dir": str(shard_dir),
        "subset_spec": subset_spec,
        "layer": int(args.layer),
        "drop_prefix": int(drop_prefix),
        "num_items": int(ds.num_items),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Assign activations to nearest saved centroids/medoids")
    parser.add_argument("--centroids-path", type=Path, required=True)
    parser.add_argument("--activations-path", type=Path, default=None)
    parser.add_argument("--shard-dir", type=str, default=None)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--drop-prefix", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--save-path", type=Path, required=True)
    args = parser.parse_args()

    if (args.activations_path is None) == (args.shard_dir is None):
        raise ValueError("Pass exactly one of --activations-path or --shard-dir")
    if args.shard_dir is not None and args.layer is None:
        raise ValueError("--layer is required with --shard-dir")

    device = _resolve_device(args.device)
    centroids = _load_centroids(args.centroids_path)
    source: dict[str, Any]
    if args.activations_path is not None:
        data = np.load(args.activations_path, mmap_mode="r")
        loader_or_array = data
        source = {"activations_path": str(args.activations_path), "shape": list(data.shape)}
    else:
        loader_or_array, source = _loader_from_shards(args, device)

    sizes, assignments, min_distances = compute_nearest_centroid_assignments(
        centroids,
        loader_or_array,
        device=device,
        batch_size=args.batch_size,
        max_batches=args.max_batches,
    )
    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "cluster_sizes": sizes,
        "assignments": assignments,
        "min_distances": min_distances,
        "K": int(sizes.numel()),
        "centroids_path": str(args.centroids_path),
        "source": source,
    }, args.save_path)
    print(f"Assignments saved to {args.save_path}")


if __name__ == "__main__":
    main()
