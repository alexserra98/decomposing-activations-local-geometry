"""Fit full-data KMeans centroids from toy-manifold activation shards."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dalg.data.shard_activations import ActivationBatchDataset
from dalg.init.centroid_artifact import (
    compute_cluster_pca_directions,
    load_centroid_artifact,
    save_centroid_artifact,
    validate_centroid_artifact,
)
from dalg.init.projected_knn import KMeansTorch


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
) -> tuple[torch.Tensor, dict]:
    config = json.loads((shard_dir / "config.json").read_text())
    if layer not in config["layers"]:
        raise ValueError(f"layer {layer} is absent from {config['layers']}")
    drop_prefix = int(config.get("drop_prefix", 0))
    tokens_per_row = int(config["window"]) - drop_prefix
    expected_rows = int(config["num_rows"]) * tokens_per_row
    expected_dim = int(config["d_model"])

    dataset = ActivationBatchDataset(
        shard_dir,
        layer=layer,
        drop_prefix=drop_prefix,
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
    return points, config


@torch.no_grad()
def build_centroids(args: argparse.Namespace) -> None:
    centroids_path = args.out_dir / "centroids.pt"
    config_path = args.out_dir / "config.json"
    shard_config = json.loads((args.shard_dir / "config.json").read_text())
    expected_dim = int(shard_config["d_model"])
    if not 1 <= args.pca_rank <= expected_dim:
        raise ValueError(
            f"pca_rank must be in [1, {expected_dim}], got {args.pca_rank}"
        )

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
    )
    if not 1 <= args.K < len(points):
        raise ValueError(f"K must be in [1, {len(points) - 1}], got {args.K}")

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
    if torch.any(cluster_sizes == 0):
        empty = torch.nonzero(cluster_sizes == 0).flatten().tolist()
        raise ValueError(f"final KMeans solution has empty clusters: {empty}")

    centroids = centroids.float().cpu()
    if centroids.shape != (args.K, points.shape[1]):
        raise ValueError(f"unexpected centroid shape: {tuple(centroids.shape)}")
    if not torch.isfinite(centroids).all():
        raise ValueError("centroids contain non-finite values")

    pca_start = time.time()
    principal_components = compute_cluster_pca_directions(
        points_device,
        labels,
        centroids.to(args.device),
        rank=args.pca_rank,
        chunk_elems=args.pca_chunk_elems,
        eig_batch_size=args.pca_eig_batch_size,
    ).float().cpu()
    pca_seconds = time.time() - pca_start
    if principal_components.shape != (args.K, points.shape[1], args.pca_rank):
        raise ValueError(
            f"unexpected principal-component shape: {tuple(principal_components.shape)}"
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
            "source_shard_dir": str(args.shard_dir.resolve()),
            "layer": args.layer,
            "rows_used": len(points),
            "uses_all_rows": True,
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
                "num_rows": shard_config["num_rows"],
                "d_model": shard_config["d_model"],
                "window": shard_config["window"],
                "drop_prefix": shard_config.get("drop_prefix", 0),
            },
        }
    config["centroid_artifact_format"] = "dalg_centroids_v1"
    config["principal_components"] = {
        "rank": args.pca_rank,
        "shape": list(principal_components.shape),
        "center": "stored_kmeans_centroid",
        "covariance_accumulator_dtype": "float64",
        "uses_all_rows": True,
        "compute_seconds": pca_seconds,
    }
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    inertia = config.get("inertia")
    inertia_text = f"{float(inertia):.8g}" if inertia is not None else "unknown"
    print(
        f"Saved {tuple(centroids.shape)} centroids and "
        f"{tuple(principal_components.shape)} PCA directions to {args.out_dir}; "
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
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--load-batch-size", type=int, default=20_000)
    parser.add_argument("--block-x", type=int, default=8192)
    parser.add_argument("--block-c", type=int, default=8192)
    parser.add_argument("--pca-rank", type=int, default=32)
    parser.add_argument(
        "--pca-only",
        action="store_true",
        help="Load <out-dir>/centroids.pt and add PCA directions without refitting KMeans.",
    )
    parser.add_argument("--pca-chunk-elems", type=int, default=1 << 23)
    parser.add_argument("--pca-eig-batch-size", type=int, default=256)
    return parser


def main() -> None:
    build_centroids(build_parser().parse_args())


if __name__ == "__main__":
    main()
