"""Fit KMeans centroids and local PCA from toy-manifold activation shards."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dalg.data.shard_activations import ActivationBatchDataset
from dalg.data.subset_spec import split_shard_dir_spec
from dalg.init.activation_selection import resolve_initialization_rows
from dalg.init.centroid_artifact import (
    compute_cluster_pca_directions,
    load_centroid_artifact,
    save_centroid_artifact,
    validate_centroid_artifact,
)
from dalg.init.projected_knn import KMeansTorch
from dalg.init.neighborhood_pca import nearest_neighbor_indices, compute_neighborhood_pca


def _check_output_dir(output_dir: Path) -> None:
    if not output_dir.exists():
        return
    if not output_dir.is_dir():
        raise FileExistsError(
            f"output path exists and is not a directory: {output_dir}"
        )
    if any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")


def _load_activations(
    shard_dir: Path,
    *,
    layer: int,
    batch_size: int,
    val_frac: float = 0.0,
    split_seed: int = 42,
    drop_prefix: int | None = None,
) -> tuple[torch.Tensor, dict]:
    shard_dir, config, positions, selection = resolve_initialization_rows(
        shard_dir, layer=layer, val_frac=val_frac, split_seed=split_seed,
        drop_prefix=drop_prefix,
    )
    if "layers" in config and layer not in config["layers"]:
        raise ValueError(f"layer {layer} is absent from {config['layers']}")
    expected_rows = selection["train_activations"]
    expected_dim = int(config["d_model"])

    dataset = ActivationBatchDataset(
        shard_dir,
        layer=layer,
        drop_prefix=selection["drop_prefix"],
        row_subset=positions,
        batch_size=batch_size,
        dtype=torch.float32,
        shuffle_shards=False,
        shuffle_within_shard=False,
        seed=0,
    )
    loader = DataLoader(dataset, batch_size=None, num_workers=0)
    points = torch.cat(list(loader), dim=0)
    if points.shape != (expected_rows, expected_dim):
        raise ValueError(
            f"expected activations shape {(expected_rows, expected_dim)}, "
            f"got {tuple(points.shape)}"
        )
    if not torch.isfinite(points).all():
        raise ValueError("activations contain non-finite values")
    return points, {**config, "initialization_selection": selection}


def _subsample_points(
    points: torch.Tensor,
    *,
    fraction: float,
    seed: int,
) -> torch.Tensor:
    """Select a deterministic uniform subset while preserving canonical row order."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"sample_fraction must be in (0, 1], got {fraction}")
    if fraction == 1.0:
        return points

    sample_size = max(1, int(len(points) * fraction))
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(points), generator=generator)[:sample_size]
    return points[indices.sort().values]


@torch.no_grad()
def build_centroids(args: argparse.Namespace) -> None:
    pca_method = getattr(args, "pca_method", "cluster")
    pca_neighbors = getattr(args, "pca_neighbors", 64)
    if pca_method not in ("cluster", "knn"):
        raise ValueError("pca_method must be 'cluster' or 'knn'")
    if pca_method == "knn" and not 0 < args.pca_rank < pca_neighbors:
        raise ValueError("KNN PCA requires 0 < pca_rank < pca_neighbors")
    centroids_path = args.out_dir / "centroids.pt"
    config_path = args.out_dir / "config.json"
    clean_shard_dir, _subset_spec = split_shard_dir_spec(args.shard_dir)
    shard_config = json.loads((clean_shard_dir / "config.json").read_text())
    expected_dim = int(shard_config["d_model"])
    if not 0 <= args.pca_rank <= expected_dim:
        raise ValueError(
            f"pca_rank must be in [0, {expected_dim}], got {args.pca_rank}"
        )
    if args.pca_only and args.pca_rank == 0:
        raise ValueError("--pca-only requires --pca-rank to be positive")

    existing_centroids = None
    if args.pca_only:
        if not centroids_path.is_file():
            raise FileNotFoundError(
                f"--pca-only requires an existing centroid artifact: {centroids_path}"
            )
        if not config_path.is_file():
            raise FileNotFoundError(
                f"--pca-only requires existing centroid metadata: {config_path}"
            )
        existing_centroids, existing_pcs = load_centroid_artifact(
            centroids_path,
            map_location="cpu",
            mmap=True,
        )
        validate_centroid_artifact(
            existing_centroids,
            existing_pcs,
            expected_k=args.K,
            expected_d=expected_dim,
        )
        saved_config = json.loads(config_path.read_text())
        saved_selection = saved_config.get("selection")
        if saved_selection is not None:
            _, _, _, requested_selection = resolve_initialization_rows(
                args.shard_dir, layer=args.layer,
                val_frac=getattr(args, "val_frac", 0.0),
                split_seed=getattr(args, "split_seed", 42),
                drop_prefix=getattr(args, "drop_prefix", None),
            )
            if requested_selection != saved_selection:
                raise ValueError("--pca-only must use the original centroid training split")
        elif getattr(args, "val_frac", 0.0) != 0.0 or _subset_spec is not None:
            raise ValueError("--pca-only cannot verify this split for legacy centroid metadata")
        for key, default in (("sample_fraction_requested", 1.0), ("sample_seed", 0)):
            argument = "sample_fraction" if key == "sample_fraction_requested" else key
            if saved_config.get(key, default) != getattr(args, argument, default):
                raise ValueError("--pca-only must use the original centroid subsample")
        if existing_pcs is not None:
            pcs_metadata = saved_config.get("principal_components") or {}
            expected_method = "cluster_covariance" if pca_method == "cluster" else "nearest_neighbor_covariance"
            if pcs_metadata.get("method", "cluster_covariance") != expected_method or (
                pca_method == "knn" and pcs_metadata.get("neighbors_per_centroid") != pca_neighbors
            ):
                raise ValueError("existing PCA uses a different method or neighbor count; use a new output directory")
        if existing_pcs is not None and existing_pcs.shape[-1] >= args.pca_rank:
            print(
                f"Centroid artifact already stores {existing_pcs.shape[-1]} principal "
                f"components per cluster: {centroids_path}"
            )
            return
    else:
        _check_output_dir(args.out_dir)

    points, shard_config = _load_activations(
        args.shard_dir,
        layer=args.layer,
        batch_size=args.load_batch_size,
        val_frac=getattr(args, "val_frac", 0.0),
        split_seed=getattr(args, "split_seed", 42),
        drop_prefix=getattr(args, "drop_prefix", None),
    )
    selection = shard_config.get("initialization_selection")
    training_rows = len(points)
    source_rows = selection["selected_activations"] if selection else training_rows
    sample_fraction = float(getattr(args, "sample_fraction", 1.0))
    sample_seed = int(getattr(args, "sample_seed", 0))
    points = _subsample_points(
        points,
        fraction=sample_fraction,
        seed=sample_seed,
    )
    if not 1 <= args.K < len(points):
        raise ValueError(f"K must be in [1, {len(points) - 1}], got {args.K}")
    if pca_method == "knn" and pca_neighbors > len(points):
        raise ValueError("pca_neighbors exceeds the selected training activations")

    kmeans = KMeansTorch(
        k=args.K,
        metric="euclidean",
        n_iter=args.max_iter,
        restarts=args.restarts,
        tol=args.tol,
        seed=args.seed,
        device=torch.device(args.device),
        block_x=args.block_x,
        block_c=args.block_c,
    )
    if existing_centroids is None:
        start = time.time()
        centroids = kmeans.fit(points)
        fit_seconds = time.time() - start
    else:
        centroids = existing_centroids.to(args.device)
        fit_seconds = None

    points_device = points.to(args.device)
    labels = kmeans._assign_streamed(points_device, centroids)
    cluster_sizes = torch.bincount(labels, minlength=args.K).cpu()
    if int(cluster_sizes.sum()) != len(points):
        raise ValueError("final cluster sizes do not cover every activation")
    if pca_method == "cluster" and torch.any(cluster_sizes == 0):
        empty = torch.nonzero(cluster_sizes == 0).flatten().tolist()
        raise ValueError(f"final KMeans solution has empty clusters: {empty}")

    centroids = centroids.float().cpu()
    if centroids.shape != (args.K, points.shape[1]):
        raise ValueError(f"unexpected centroid shape: {tuple(centroids.shape)}")
    if not torch.isfinite(centroids).all():
        raise ValueError("centroids contain non-finite values")

    principal_components = None
    pca_seconds = None
    if args.pca_rank > 0:
        pca_start = time.time()
        if pca_method == "cluster":
            principal_components = compute_cluster_pca_directions(
                points_device,
                labels,
                centroids.to(args.device),
                rank=args.pca_rank,
                chunk_elems=args.pca_chunk_elems,
                eig_batch_size=args.pca_eig_batch_size,
            ).float().cpu()
        else:
            neighbor_indices, _ = nearest_neighbor_indices(
                points_device, centroids, neighbors=pca_neighbors,
                device=torch.device(args.device), point_block_size=args.block_x,
            )
            principal_components = compute_neighborhood_pca(
                points_device, centroids, neighbor_indices, rank=args.pca_rank,
                device=torch.device(args.device), eig_batch_size=args.pca_eig_batch_size,
            )
        pca_seconds = time.time() - pca_start
        if principal_components.shape != (
            args.K,
            points.shape[1],
            args.pca_rank,
        ):
            raise ValueError(
                "unexpected principal-component shape: "
                f"{tuple(principal_components.shape)}"
            )
        if not torch.isfinite(principal_components).all():
            raise ValueError("principal components contain non-finite values")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    save_centroid_artifact(centroids_path, centroids, principal_components)
    device = torch.device(args.device)
    if args.pca_only:
        config = json.loads(config_path.read_text())
        config["cluster_sizes"] = cluster_sizes.tolist()
    else:
        config = {
            "method": "kmeans",
            "source_shard_dir": str(clean_shard_dir.resolve()),
            "layer": args.layer,
            "rows_used": len(points),
            "uses_all_rows": len(points) == source_rows,
            "source_rows": source_rows,
            "sample_fraction_requested": sample_fraction,
            "sample_fraction_actual": len(points) / training_rows,
            "sample_seed": sample_seed,
            "shape": list(points.shape),
            "K": args.K,
            "metric": "euclidean",
            "implementation": "dalg.init.projected_knn.KMeansTorch",
            "initialization": "kmeans++",
            "max_iter": args.max_iter,
            "restarts": args.restarts,
            "tol": args.tol,
            "seed": args.seed,
            "device": device.type,
            "cuda_device": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
            "fit_seconds": fit_seconds,
            "iterations_last_recorded_restart": kmeans.n_iter_run_,
            "inertia": float(kmeans.inertia_),
            "cluster_sizes": cluster_sizes.tolist(),
            "centroids_path": "centroids.pt",
            "source_config": {
                "source_kind": shard_config.get("source_kind"),
                "num_rows": shard_config.get("num_rows", selection["selected_rows"] if selection else source_rows),
                "d_model": shard_config["d_model"],
                "window": shard_config["window"],
                "drop_prefix": selection["drop_prefix"] if selection else shard_config.get("drop_prefix", 0),
            },
        }
    config["centroid_artifact_format"] = "dalg_centroids_v1"
    config["selection"] = selection
    config["uses_all_training_rows"] = len(points) == training_rows
    config["principal_components"] = (
        {
            "method": "cluster_covariance" if pca_method == "cluster" else "nearest_neighbor_covariance",
            "rank": args.pca_rank,
            "shape": list(principal_components.shape),
            "center": "stored_kmeans_centroid",
            "covariance_accumulator_dtype": "float64",
            "uses_all_rows": len(points) == source_rows,
            "compute_seconds": pca_seconds,
        }
        if principal_components is not None
        else None
    )
    if principal_components is not None and pca_method == "knn":
        config["principal_components"].update(
            neighbors_per_centroid=pca_neighbors,
            neighbor_selection="nearest_euclidean_points_per_centroid",
            neighborhoods_may_overlap=True,
            hard_assignment_pca=False,
        )
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    inertia = config.get("inertia")
    inertia_text = f"{float(inertia):.8g}" if inertia is not None else "unknown"
    pca_text = (
        f" and {tuple(principal_components.shape)} PCA directions"
        if principal_components is not None
        else " without PCA directions"
    )
    print(
        f"Saved {tuple(centroids.shape)} centroids{pca_text} to {args.out_dir}; "
        f"inertia={inertia_text}, cluster sizes="
        f"{int(cluster_sizes.min())}..{int(cluster_sizes.max())}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--K", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--restarts", type=int, default=10)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-frac", type=float, default=0.0)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--drop-prefix", type=int, default=None)
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=1.0,
        help="Uniform fraction of activation rows used for both KMeans and PCA.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=0,
        help="Seed for deterministic row subsampling.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--load-batch-size", type=int, default=20_000)
    parser.add_argument("--block-x", type=int, default=8192)
    parser.add_argument("--block-c", type=int, default=8192)
    parser.add_argument(
        "--pca-rank",
        type=int,
        default=32,
        help="Number of per-cluster PCA directions to save; 0 saves centroids only.",
    )
    parser.add_argument(
        "--pca-only",
        action="store_true",
        help="Load <out-dir>/centroids.pt and add PCA directions without refitting KMeans.",
    )
    parser.add_argument("--pca-chunk-elems", type=int, default=1 << 23)
    parser.add_argument("--pca-method", choices=("cluster", "knn"), default="cluster")
    parser.add_argument("--pca-neighbors", type=int, default=64,
                        help="Nearest training points per centroid for --pca-method knn.")
    parser.add_argument("--pca-eig-batch-size", type=int, default=256)
    return parser


def main() -> None:
    build_centroids(build_parser().parse_args())


if __name__ == "__main__":
    main()
