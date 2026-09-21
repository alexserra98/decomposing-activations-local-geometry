"""Attach experimental fixed-neighborhood PCA directions to toy centroids.

This is not hard-assignment cluster PCA. Every centroid uses its own nearest
``neighbors`` points, so neighborhoods may overlap. The script is deliberately
kept outside ``src/`` and writes a new artifact rather than changing its input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from build_toy_kmeans_centroids import _load_activations, _subsample_points
from dalg.init.neighborhood_pca import nearest_neighbor_indices, compute_neighborhood_pca
from dalg.init.centroid_artifact import (
    load_centroid_artifact,
    save_centroid_artifact,
    validate_centroid_artifact,
)


def _check_empty_output_dir(output_dir: Path) -> None:
    if not output_dir.exists():
        return
    if not output_dir.is_dir():
        raise FileExistsError(
            f"output path exists and is not a directory: {output_dir}"
        )
    if any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@torch.no_grad()
def build_knn_pca_artifact(args: argparse.Namespace) -> None:
    _check_empty_output_dir(args.out_dir)
    if args.neighbors <= args.pca_rank:
        raise ValueError("neighbors must be greater than pca_rank")

    source_config_path = args.centroids_path.parent / "config.json"
    if not source_config_path.is_file():
        raise FileNotFoundError(
            f"centroid provenance config not found: {source_config_path}"
        )
    source_config = json.loads(source_config_path.read_text())
    centroids, existing_pcs = load_centroid_artifact(
        args.centroids_path,
        map_location="cpu",
        mmap=True,
    )
    shard_config = json.loads((args.shard_dir / "config.json").read_text())
    validate_centroid_artifact(
        centroids,
        existing_pcs,
        expected_k=int(source_config["K"]),
        expected_d=int(shard_config["d_model"]),
    )

    recorded_fraction = float(source_config["sample_fraction_actual"])
    recorded_seed = int(source_config["sample_seed"])
    if recorded_fraction != args.sample_fraction:
        raise ValueError(
            "sample_fraction does not match source centroid config: "
            f"{args.sample_fraction} != {recorded_fraction}"
        )
    if recorded_seed != args.sample_seed:
        raise ValueError(
            "sample_seed does not match source centroid config: "
            f"{args.sample_seed} != {recorded_seed}"
        )

    points, _shard_config = _load_activations(
        args.shard_dir,
        layer=args.layer,
        batch_size=args.load_batch_size,
    )
    source_rows = len(points)
    points = _subsample_points(
        points,
        fraction=args.sample_fraction,
        seed=args.sample_seed,
    )
    if len(points) != int(source_config["rows_used"]):
        raise ValueError(
            f"selected {len(points)} PCA rows but source centroids used "
            f"{source_config['rows_used']}"
        )

    device = torch.device(args.device)
    start = time.time()
    neighbor_indices, neighbor_distances = nearest_neighbor_indices(
        points,
        centroids,
        neighbors=args.neighbors,
        device=device,
        point_block_size=args.point_block_size,
    )
    principal_components = compute_neighborhood_pca(
        points,
        centroids,
        neighbor_indices,
        rank=args.pca_rank,
        device=device,
        eig_batch_size=args.eig_batch_size,
    )
    compute_seconds = time.time() - start

    validate_centroid_artifact(
        centroids,
        principal_components,
        expected_k=int(source_config["K"]),
        expected_d=int(shard_config["d_model"]),
        required_pca_rank=args.pca_rank,
    )
    if not torch.isfinite(principal_components).all():
        raise ValueError("principal components contain non-finite values")
    gram = principal_components.transpose(1, 2) @ principal_components
    identity = torch.eye(args.pca_rank).expand_as(gram)
    max_orthonormality_error = float((gram - identity).abs().max())
    if max_orthonormality_error > 1e-4:
        raise ValueError(
            "principal components are not orthonormal: maximum error "
            f"{max_orthonormality_error}"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.out_dir / "centroids.pt"
    save_centroid_artifact(output_path, centroids, principal_components)
    saved_centroids, saved_pcs = load_centroid_artifact(
        output_path,
        map_location="cpu",
        mmap=True,
    )
    validate_centroid_artifact(
        saved_centroids,
        saved_pcs,
        expected_k=int(source_config["K"]),
        expected_d=int(shard_config["d_model"]),
        required_pca_rank=args.pca_rank,
    )
    if not torch.equal(saved_centroids, centroids):
        raise ValueError("saved artifact changed the source centroids")

    config = dict(source_config)
    config["centroids_path"] = "centroids.pt"
    config["derived_from_centroids"] = str(args.centroids_path.resolve())
    config["derived_from_centroids_sha256"] = _sha256(args.centroids_path)
    config["principal_components"] = {
        "method": "nearest_neighbor_covariance",
        "rank": args.pca_rank,
        "shape": list(principal_components.shape),
        "neighbors_per_centroid": args.neighbors,
        "neighbor_selection": "nearest_euclidean_points_per_centroid",
        "neighborhoods_may_overlap": True,
        "hard_assignment_pca": False,
        "center": "stored_kmeans_centroid",
        "covariance_accumulator_dtype": "float64",
        "source_rows": source_rows,
        "rows_available": len(points),
        "sample_fraction": args.sample_fraction,
        "sample_seed": args.sample_seed,
        "temporary_experimental_feature": True,
        "compute_seconds": compute_seconds,
        "max_orthonormality_error": max_orthonormality_error,
        "nearest_neighbor_radius": {
            "minimum": float(neighbor_distances[:, -1].sqrt().min()),
            "mean": float(neighbor_distances[:, -1].sqrt().mean()),
            "maximum": float(neighbor_distances[:, -1].sqrt().max()),
        },
    }
    (args.out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(
        f"Saved unchanged {tuple(centroids.shape)} centroids and experimental "
        f"{tuple(principal_components.shape)} KNN-PCA directions to {args.out_dir}; "
        f"neighbors={args.neighbors}, rows={len(points)}, "
        f"max orthonormality error={max_orthonormality_error:.3g}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--centroids-path", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--sample-fraction", type=float, required=True)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--neighbors", type=int, default=64)
    parser.add_argument("--pca-rank", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--load-batch-size", type=int, default=20_000)
    parser.add_argument("--point-block-size", type=int, default=8192)
    parser.add_argument("--eig-batch-size", type=int, default=128)
    return parser


def main() -> None:
    build_knn_pca_artifact(build_parser().parse_args())


if __name__ == "__main__":
    main()
