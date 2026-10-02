"""Fit a KMeans model and optional local PCA on the canonical training split."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dalg.data.shard_activations import (
    ActivationBatchDataset, load_meta_index, per_subset_counts, stratified_split,
)
from dalg.data.subset_spec import resolve_spec_positions
from dalg.init.activation_selection import resolve_initialization_rows
from dalg.models.kmeans import KMEANS_MODEL_FORMAT, KMeans, load_kmeans, save_kmeans


def _subsample_points(points: torch.Tensor, *, fraction: float, seed: int) -> torch.Tensor:
    """Select a reproducible subset while preserving canonical stream order."""
    if fraction == 1.0:
        return points
    size = max(1, int(len(points) * fraction))
    indices = torch.randperm(len(points), generator=torch.Generator().manual_seed(seed))[:size]
    return points[indices.sort().values]


def _load_training_points(args):
    root, config, positions, selection = resolve_initialization_rows(
        args.shard_dir, layer=args.layer, val_frac=args.val_frac,
        split_seed=args.split_seed, drop_prefix=args.drop_prefix,
    )
    if "layers" in config and args.layer not in config["layers"]:
        raise ValueError(f"layer {args.layer} is absent from {config['layers']}")
    dataset = ActivationBatchDataset(
        root, layer=args.layer, drop_prefix=selection["drop_prefix"],
        row_subset=positions, batch_size=args.load_batch_size,
        dtype=torch.float32, shuffle_shards=False, shuffle_within_shard=False,
        seed=0,
    )
    points = torch.cat(list(DataLoader(dataset, batch_size=None, num_workers=0)))
    expected = (selection["train_activations"], int(config["d_model"]))
    if points.shape != expected:
        raise ValueError(f"expected activations shape {expected}, got {tuple(points.shape)}")
    if not torch.isfinite(points).all():
        raise ValueError("activations contain non-finite values")
    meta = load_meta_index(root, layer=args.layer)
    selected = resolve_spec_positions(
        meta, selection["subset_spec"], window=int(config["window"]),
        drop_prefix=selection["drop_prefix"],
    )
    train, val = stratified_split(meta, val_frac=args.val_frac, seed=args.split_seed, positions=selected)
    split_info = {
        "seed": args.split_seed,
        "val_frac": args.val_frac,
        "per_row_tokens": int(config["window"]) - selection["drop_prefix"],
        "train_rows": len(train),
        "val_rows": len(val),
        "train_per_subset": per_subset_counts(meta, train),
        "val_per_subset": per_subset_counts(meta, val),
        "val_global_rows": [meta[p]["global_row"] for p in val],
        "world_size": 1,
        "training_mode": "single_process",
        "component_shard": False,
    }
    return root, config, selection, split_info, points


@torch.no_grad()
def cmd_train(args) -> None:
    validate_args(args)
    if args.out_dir is None:
        raise ValueError("--out-dir is required when fitting KMeans")
    out_dir = Path(args.out_dir)
    model_path = out_dir / "kmeans_model.pt"
    config_path = out_dir / "config.json"
    if not args.pca_only and (model_path.exists() or config_path.exists()):
        raise FileExistsError(f"KMeans output already exists in {out_dir}; use a new output directory")
    if args.pca_only and not model_path.is_file():
        raise FileNotFoundError(f"--pca-only requires an existing KMeans checkpoint: {model_path}")

    root, shard_config, selection, split_info, points = _load_training_points(args)
    training_rows = len(points)
    points = _subsample_points(points, fraction=args.sample_fraction, seed=args.sample_seed)
    if not 1 <= args.K < len(points):
        raise ValueError(f"K must be in [1, {len(points) - 1}], got {args.K}")
    if args.rank is not None and not 0 <= args.rank <= points.shape[1]:
        raise ValueError(f"rank must be in [0, {points.shape[1]}], got {args.rank}")
    if args.rank and args.pca_purpose == "initialization" and args.pca_neighbors > len(points):
        raise ValueError("pca_neighbors exceeds the selected training activations")

    previous_config = None
    if args.pca_only:
        model = load_kmeans(model_path, map_location=args.device)
        previous_config = dict(model.checkpoint_extra)
        if (model.K, model.D) != (args.K, points.shape[1]):
            raise ValueError("--pca-only must preserve the checkpoint's K and D")
        if config_path.exists() and json.loads(config_path.read_text()) != previous_config:
            raise ValueError("config.json disagrees with the existing KMeans checkpoint metadata")
        split_path = out_dir / "val_indices.json"
        if split_path.exists() and json.loads(split_path.read_text()) != split_info:
            raise ValueError("val_indices.json disagrees with the requested training split")
        for name in ("max_iter", "restarts", "tol", "seed", "block_x", "block_c"):
            if getattr(args, name) != getattr(model, name):
                raise ValueError(f"--pca-only must preserve the original fitting settings ({name})")
        for key, value in {
            "selection": selection, "source_shard_dir": str(root.resolve()),
            "layer": args.layer, "sample_fraction_requested": args.sample_fraction,
            "sample_seed": args.sample_seed,
            "rows_used": len(points), "shape": list(points.shape),
            "uses_all_training_rows": len(points) == training_rows,
            "window": shard_config["window"], "d_model": shard_config["d_model"],
        }.items():
            if previous_config.get(key) != value:
                raise ValueError(f"--pca-only must preserve the original training population ({key})")
        fit_seconds = previous_config.get("fit_seconds")
    else:
        model = KMeans(
            args.K, max_iter=args.max_iter, restarts=args.restarts,
            tol=args.tol, seed=args.seed, device=args.device,
            block_x=args.block_x, block_c=args.block_c,
        )
        start = time.monotonic()
        model.fit(points)
        fit_seconds = time.monotonic() - start

    points = points.to(args.device)
    labels = model.predict(points)
    counts = torch.bincount(labels, minlength=args.K)
    inertia = float((points - model.mu[labels]).square().sum())
    pca_seconds = None
    if args.rank != 0:
        start = time.monotonic()
        if args.pca_purpose == "initialization":
            model.compute_init_pcs(
                points, rank=args.rank, neighbors=args.pca_neighbors,
                eig_batch_size=args.pca_eig_batch_size,
            )
        else:
            model.compute_pcs(
                points, threshold=args.surgery_threshold, chunk_elems=args.pca_chunk_elems,
                eig_batch_size=args.pca_eig_batch_size,
            )
        pca_seconds = time.monotonic() - start
    pca_info = None
    if args.rank != 0:
        pca_info = {
            "method": "nearest_neighbor_covariance" if args.pca_purpose == "initialization" else "cluster_covariance",
            "rank": model.q_init if args.pca_purpose == "initialization" else model.q,
            "shape": list(model.W_init.shape if args.pca_purpose == "initialization" else model.W.shape),
            "center": "stored_kmeans_centroid",
            "covariance_accumulator_dtype": "float64",
            "uses_all_rows": len(points) == selection["selected_activations"],
            "uses_all_training_rows": len(points) == training_rows,
            "compute_seconds": pca_seconds,
            "device": str(torch.device(args.device)),
            "chunk_elems": args.pca_chunk_elems,
            "eig_batch_size": args.pca_eig_batch_size,
        }
        if args.pca_purpose == "initialization":
            pca_info.update(
                neighbors_per_centroid=args.pca_neighbors,
                neighbor_selection="nearest_euclidean_points_per_centroid",
                neighborhoods_may_overlap=True, hard_assignment_pca=False,
            )
    config = previous_config or {
        "method": "kmeans",
        "source_shard_dir": str(root.resolve()),
        "shard_dir": str(root.resolve()),
        "subset_spec": selection["subset_spec"],
        "layer": args.layer,
        "rows_used": len(points),
        "source_rows": selection["selected_activations"],
        "uses_all_rows": len(points) == selection["selected_activations"],
        "uses_all_training_rows": len(points) == training_rows,
        "sample_fraction_requested": args.sample_fraction,
        "sample_fraction_actual": len(points) / training_rows,
        "sample_seed": args.sample_seed,
        "shape": list(points.shape),
        "K": args.K,
        "metric": "euclidean",
        "implementation": "dalg.models.kmeans.KMeans",
        "initialization": "kmeans++",
        "max_iter": args.max_iter,
        "restarts": args.restarts,
        "tol": args.tol,
        "seed": args.seed,
        "device": torch.device(args.device).type,
        "fit_seconds": fit_seconds,
        "window": shard_config["window"],
        "d_model": shard_config["d_model"],
        "drop_prefix": selection["drop_prefix"],
        "val_frac": args.val_frac,
        "split_seed": args.split_seed,
        "training_mode": "single_process",
        "world_size": 1,
    }
    config.update(
        model_artifact_format=KMEANS_MODEL_FORMAT, model_path="kmeans_model.pt",
        selection=selection, rank=model.q, init_rank=model.q_init,
        surgery_threshold=model.surgery_threshold, cluster_sizes=counts.cpu().tolist(),
        inertia=inertia, quantization_error=inertia / len(points),
    )
    config.setdefault("principal_components", None)
    config.setdefault("initialization_components", None)
    pca_key = "initialization_components" if args.pca_purpose == "initialization" else "principal_components"
    config[pca_key] = pca_info
    out_dir.mkdir(parents=True, exist_ok=True)
    save_kmeans(model, model_path, extra=config)
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    (out_dir / "val_indices.json").write_text(json.dumps(split_info, indent=2) + "\n")
    print(f"Saved KMeans ({model.K}, {model.D}), {model.q} geometry PCs, {model.q_init} initialization PCs, inertia={inertia:.8g} to {model_path}")


def validate_args(args) -> None:
    if int(os.environ.get("WORLD_SIZE", "1")) != 1 or args.training_mode != "single_process":
        raise ValueError("KMeans requires one process and training_mode=single_process")
    if torch.device(args.device).type not in {"cpu", "cuda"}:
        raise ValueError("KMeans supports CPU or CUDA")
    for name in ("K", "max_iter", "restarts", "load_batch_size", "block_x", "block_c", "pca_chunk_elems", "pca_eig_batch_size"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.rank is not None and args.rank < 0:
        raise ValueError("rank must be non-negative; rank 0 stores means without PCA")
    if not math.isfinite(args.tol) or args.tol < 0:
        raise ValueError("tol must be finite and non-negative")
    if not 0 <= args.val_frac < 1:
        raise ValueError("val_frac must be in [0, 1)")
    if not 0 < args.sample_fraction <= 1:
        raise ValueError("sample_fraction must be in (0, 1]")
    if args.pca_only and args.rank == 0:
        raise ValueError("--pca-only requires PCA, not centroid-only rank 0")
    if args.pca_purpose == "initialization":
        if args.rank is None:
            raise ValueError("initialization PCA requires --rank")
        if args.pca_neighbors is None:
            args.pca_neighbors = 64
        if args.pca_neighbors <= 0:
            raise ValueError("pca_neighbors must be positive")
    if args.rank and args.pca_purpose == "initialization" and args.rank >= args.pca_neighbors:
        raise ValueError("KNN PCA requires rank < pca_neighbors")
    if args.pca_purpose == "initialization" and args.surgery_threshold is not None:
        raise ValueError("initialization PCA does not support Cattell rank selection")
    if args.pca_purpose == "geometry":
        if args.pca_neighbors is not None:
            raise ValueError("--pca-neighbors is only valid for initialization PCA")
        if args.rank not in (None, 0):
            raise ValueError("geometry learns rank with Cattell; omit --rank (use 0 for centroids only)")
        if args.rank is None and args.surgery_threshold is None:
            args.surgery_threshold = 0.1
    if args.surgery_threshold is not None and (
        args.rank == 0 or not math.isfinite(args.surgery_threshold) or not 0 <= args.surgery_threshold <= 1
    ):
        raise ValueError("Cattell selection requires geometry PCA and a finite threshold in [0, 1]")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-dir", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--K", type=int, required=True)
    parser.add_argument("--rank", type=int, help="Initialization PC capacity; 0 stores centroids only. Omit for geometry.")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--restarts", type=int, default=10)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-frac", type=float, default=0.05)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--drop-prefix", type=int)
    parser.add_argument("--sample-fraction", type=float, default=1.0)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--training-mode", choices=["single_process"], default="single_process")
    parser.add_argument("--load-batch-size", type=int, default=20_000)
    parser.add_argument("--block-x", type=int, default=8192)
    parser.add_argument("--block-c", type=int, default=8192)
    parser.add_argument("--pca-purpose", choices=["initialization", "geometry"], default="geometry")
    parser.add_argument("--pca-neighbors", type=int, help="Initialization KNN size (default: 64).")
    parser.add_argument("--pca-chunk-elems", type=int, default=1 << 23)
    parser.add_argument("--pca-eig-batch-size", type=int, default=256)
    parser.add_argument("--surgery-threshold", "--cattell-threshold", type=float, help="Geometry Cattell threshold (default: 0.1).")
    parser.add_argument("--pca-only", action="store_true", help="Compute PCA for the checkpoint already in out-dir.")
    return parser


def main() -> None:
    cmd_train(build_parser().parse_args())


if __name__ == "__main__":
    main()
