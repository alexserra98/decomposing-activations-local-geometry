"""Train KMeans+PCA and MFA on a fixed three-way toy-manifold split."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from dalg.data.shard_activations import (
    ActivationBatchDataset,
    load_meta_index,
    per_subset_counts,
)
from dalg.evaluation.coverage import (
    evaluate_heldout_distribution_coverage,
    split_fingerprint,
    stratified_three_way_split,
)
from dalg.evaluation.toy_manifold_metrics import evaluate_toy_manifold_metrics
from dalg.init.centroid_artifact import (
    compute_cluster_pca_directions,
    load_centroid_artifact,
    save_centroid_artifact,
    validate_centroid_artifact,
)
from dalg.init.mixture_weights import initialize_mixture_weights
from dalg.init.projected_knn import KMeansTorch
from dalg.models.mfa import MFA, load_mfa, save_mfa
from dalg.models.train import train_nll


class _KMeansPCAGeometry:
    """Minimal MFA-shaped view used by the shared toy geometry evaluator."""

    def __init__(
        self,
        centroids: torch.Tensor,
        directions: torch.Tensor,
        variances: torch.Tensor,
    ) -> None:
        self.mu = centroids.detach().cpu().float()
        directions = directions.detach().cpu().double()
        variances = variances.detach().cpu().double()
        self.K, self.D = map(int, self.mu.shape)
        self.q = int(directions.shape[2])
        if tuple(variances.shape) != (self.K, self.q):
            raise ValueError("KMeans PCA variances must have shape (K, q)")
        if not torch.isfinite(variances).all() or torch.any(variances < 0.0):
            raise ValueError("KMeans PCA variances must be finite and non-negative")
        self._loadings = directions * variances.sqrt()[:, None, :]
        self.rank_mask = torch.ones(self.K, self.q, dtype=torch.bool)

    def _W(self) -> torch.Tensor:
        return self._loadings

    def _psi(self) -> torch.Tensor:
        return torch.full((self.K, self.D), 1e-12, dtype=torch.float64)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _mark_complete(directory: Path, artifact: Path) -> None:
    _write_json(
        directory / "COMPLETED.json",
        {"completed": True, "artifact": str(artifact.resolve())},
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _source_digest(shard_dir: Path, layer: int) -> str:
    paths = [shard_dir / "config.json"]
    metadata_path = shard_dir / "manifold_metadata.pt"
    if metadata_path.exists():
        paths.append(metadata_path)
    paths.extend(sorted((shard_dir / f"layer{layer:02d}").glob("shard_*.pt")))
    paths.extend(sorted((shard_dir / "meta").glob("shard_*.json")))
    if len(paths) < 3:
        raise FileNotFoundError("source fingerprint requires config, layer, and meta files")

    digest = hashlib.sha256()
    for path in paths:
        relative = str(path.relative_to(shard_dir)).encode("utf-8")
        digest.update(len(relative).to_bytes(8, byteorder="little"))
        digest.update(relative)
        with path.open("rb") as handle:
            while chunk := handle.read(1 << 20):
                digest.update(chunk)
    return digest.hexdigest()


def _coverage_input_fingerprints(args: argparse.Namespace) -> dict[str, str]:
    paths = {
        "kmeans_centroids": args.output_dir / "kmeans_pca" / "centroids.pt",
        "kmeans_train_assignments": (
            args.output_dir / "kmeans_pca" / "train_assignments.pt"
        ),
        "mfa_model": args.output_dir / "mfa" / "mfa_model.pt",
        "mfa_train_assignments": args.output_dir / "mfa" / "train_assignments.pt",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"coverage input artifacts are missing: {missing}")
    return {name: _file_sha256(path) for name, path in paths.items()}


def _geometry_input_fingerprints(args: argparse.Namespace) -> dict[str, str]:
    fingerprints = _coverage_input_fingerprints(args)
    metadata_path = args.shard_dir / "manifold_metadata.pt"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"toy-manifold metadata is required for tangent geometry: {metadata_path}"
        )
    return {**fingerprints, "manifold_metadata": _file_sha256(metadata_path)}


def _source_ambient_dim(args: argparse.Namespace) -> int:
    return int(json.loads((args.shard_dir / "config.json").read_text())["d_model"])


def _resolved_config(args: argparse.Namespace, source_config: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "dataset": {
            "shard_dir": str(args.shard_dir.resolve()),
            "source_sha256": _source_digest(args.shard_dir, args.layer),
            "fingerprint_scope": "config+manifold_metadata+layer_shards+row_metadata",
            "layer": args.layer,
            "rows": int(source_config["num_rows"]),
            "ambient_dim": int(source_config["d_model"]),
            "condition": "noiseless",
        },
        "split": {
            "train_fraction": args.train_fraction,
            "validation_fraction": args.validation_fraction,
            "test_fraction": 1.0 - args.train_fraction - args.validation_fraction,
            "seed": args.split_seed,
            "stratification": "metadata_subset",
        },
        "model": {"K": args.K, "q": args.rank},
        "kmeans": {
            "metric": "euclidean",
            "initialization": "kmeans++",
            "iterations": args.kmeans_iterations,
            "restarts": args.kmeans_restarts,
            "tolerance": args.kmeans_tolerance,
            "pca": "hard_training_assignment_cluster_covariance",
        },
        "mfa": {
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "early_stop_patience": args.early_stop_patience,
            "early_stop_min_delta": args.early_stop_min_delta,
            "direction_initialization": "kmeans_cluster_pca",
        },
        "coverage": {
            "population": "independent_test_split",
            "distance": "ambient_euclidean_to_nearest_training_assignment_live_centroid",
            "thresholds": list(args.coverage_thresholds),
        },
        "seed": args.seed,
        "output_dir": str(args.output_dir.resolve()),
    }


def _ensure_experiment_config(
    args: argparse.Namespace,
    source_config: dict[str, Any],
) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    expected = _resolved_config(args, source_config)
    path = args.output_dir / "experiment_config.json"
    if path.exists():
        actual = json.loads(path.read_text())
        if actual != expected:
            raise ValueError(f"existing experiment configuration does not match: {path}")
    else:
        _write_json(path, expected)
    return expected


def _validate_split(split: dict[str, Any], n_rows: int) -> None:
    positions = [split[name].detach().cpu().long() for name in ("train", "validation", "test")]
    combined = torch.cat(positions)
    if combined.numel() != n_rows:
        raise ValueError("split does not cover every source row exactly once")
    if combined.unique().numel() != n_rows:
        raise ValueError("split populations overlap")
    if not torch.equal(combined.sort().values, torch.arange(n_rows)):
        raise ValueError("split positions do not match the source row range")
    resolved = {name: split[name] for name in ("train", "validation", "test")}
    if split["fingerprint"] != split_fingerprint(resolved):
        raise ValueError("split fingerprint mismatch")


def _ensure_split(
    args: argparse.Namespace,
    meta_index: list[dict[str, Any]],
) -> dict[str, Any]:
    directory = args.output_dir / "split"
    artifact = directory / "split.pt"
    if (directory / "COMPLETED.json").exists():
        if not artifact.exists():
            raise FileNotFoundError(f"completed split is missing its artifact: {artifact}")
        split = torch.load(artifact, map_location="cpu", weights_only=True)
        _validate_split(split, len(meta_index))
        return split
    if directory.exists():
        raise FileExistsError(f"refusing to reuse incomplete split stage: {directory}")

    subsets = sorted({str(row.get("subset", "all")) for row in meta_index})
    subset_to_id = {name: index for index, name in enumerate(subsets)}
    labels = torch.tensor(
        [subset_to_id[str(row.get("subset", "all"))] for row in meta_index],
        dtype=torch.long,
    )
    positions = stratified_three_way_split(
        labels,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        seed=args.split_seed,
    )
    fingerprint = split_fingerprint(positions)
    split = {
        **positions,
        "train_global_rows": torch.tensor(
            [meta_index[int(i)]["global_row"] for i in positions["train"]],
            dtype=torch.long,
        ),
        "validation_global_rows": torch.tensor(
            [meta_index[int(i)]["global_row"] for i in positions["validation"]],
            dtype=torch.long,
        ),
        "test_global_rows": torch.tensor(
            [meta_index[int(i)]["global_row"] for i in positions["test"]],
            dtype=torch.long,
        ),
        "fingerprint": fingerprint,
        "seed": args.split_seed,
    }
    _validate_split(split, len(meta_index))
    temporary = directory.with_name(
        directory.name + f".tmp.{os.getpid()}.{time.time_ns()}"
    )
    temporary.mkdir(parents=True, exist_ok=False)
    torch.save(split, temporary / "split.pt")
    _write_json(
        temporary / "split.json",
        {
            "fingerprint": fingerprint,
            "seed": args.split_seed,
            "counts": {name: int(positions[name].numel()) for name in positions},
            "per_subset": {
                name: per_subset_counts(meta_index, positions[name].tolist())
                for name in positions
            },
            "source_rows": len(meta_index),
        },
    )
    _mark_complete(temporary, artifact)
    temporary.replace(directory)
    return split


def _activation_dataset(
    args: argparse.Namespace,
    positions: torch.Tensor,
    *,
    batch_size: int,
    dtype: torch.dtype,
    shuffle: bool,
) -> ActivationBatchDataset:
    return ActivationBatchDataset(
        args.shard_dir,
        layer=args.layer,
        row_subset=positions.tolist(),
        batch_size=batch_size,
        drop_prefix=0,
        dtype=dtype,
        shuffle_shards=shuffle,
        shuffle_within_shard=shuffle,
        seed=args.seed,
    )


def _materialize_points(
    args: argparse.Namespace,
    positions: torch.Tensor,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    dataset = _activation_dataset(
        args,
        positions,
        batch_size=args.load_batch_size,
        dtype=dtype,
        shuffle=False,
    )
    points = torch.cat(list(DataLoader(dataset, batch_size=None, num_workers=0)), dim=0)
    if points.shape[0] != positions.numel():
        raise ValueError(
            f"materialized {points.shape[0]} activations for {positions.numel()} rows"
        )
    if not torch.isfinite(points).all():
        raise ValueError("materialized activations contain non-finite values")
    return points.contiguous()


@torch.no_grad()
def _kmeans_pca_variances(
    args: argparse.Namespace,
    split: dict[str, Any],
    centroids: torch.Tensor,
    directions: torch.Tensor,
    assignments: torch.Tensor,
) -> torch.Tensor:
    """Recover the omitted local PCA spectrum from the fixed training partition."""
    points = _materialize_points(args, split["train"])
    assignments = assignments.detach().cpu().long()
    if assignments.numel() != points.shape[0]:
        raise ValueError("KMeans assignments do not align with training points")

    counts = torch.bincount(assignments, minlength=centroids.shape[0])
    if torch.any(counts <= directions.shape[2]):
        raise ValueError("KMeans clusters are too small for the stored PCA rank")
    order = assignments.argsort(stable=True)
    ordered_points = points[order]
    offsets = torch.cat((torch.zeros(1, dtype=torch.long), counts.cumsum(dim=0)))
    variances = torch.empty(
        centroids.shape[0],
        directions.shape[2],
        dtype=torch.float64,
    )
    centers = centroids.detach().cpu().double()
    principal = directions.detach().cpu().double()
    for component_id in range(centroids.shape[0]):
        start = int(offsets[component_id])
        stop = int(offsets[component_id + 1])
        residual = ordered_points[start:stop].double() - centers[component_id]
        projected = residual @ principal[component_id]
        variances[component_id] = projected.square().mean(dim=0)
    return variances


def _training_loader(
    args: argparse.Namespace,
    positions: torch.Tensor,
    *,
    shuffle: bool,
    batch_size: int | None = None,
    num_workers: int | None = None,
) -> DataLoader:
    dataset = _activation_dataset(
        args,
        positions,
        batch_size=batch_size or args.batch_size,
        dtype=torch.float32,
        shuffle=shuffle,
    )
    resolved_num_workers = args.num_workers if num_workers is None else num_workers
    return DataLoader(
        dataset,
        batch_size=None,
        num_workers=resolved_num_workers,
        pin_memory=args.device != "cpu",
        persistent_workers=resolved_num_workers > 0,
    )


def _load_kmeans_stage(
    args: argparse.Namespace,
    split: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    directory = args.output_dir / "kmeans_pca"
    if not (directory / "COMPLETED.json").exists():
        raise FileNotFoundError(f"KMeans stage is not complete: {directory}")
    centroids, directions = load_centroid_artifact(
        directory / "centroids.pt", map_location="cpu"
    )
    validate_centroid_artifact(
        centroids,
        directions,
        expected_k=args.K,
        expected_d=_source_ambient_dim(args),
        required_pca_rank=args.rank,
    )
    assignment_artifact = torch.load(
        directory / "train_assignments.pt", map_location="cpu", weights_only=True
    )
    if assignment_artifact["split_fingerprint"] != split["fingerprint"]:
        raise ValueError("KMeans artifact split fingerprint mismatch")
    assignments = assignment_artifact["assignments"].long()
    cluster_sizes = assignment_artifact["cluster_sizes"].long()
    if assignments.numel() != split["train"].numel():
        raise ValueError("KMeans assignments do not cover the training split")
    if int(cluster_sizes.sum()) != split["train"].numel():
        raise ValueError("KMeans cluster sizes do not sum to the training population")
    return centroids, directions, assignments, cluster_sizes


@torch.no_grad()
def _fit_kmeans_pca(
    args: argparse.Namespace,
    split: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    directory = args.output_dir / "kmeans_pca"
    if (directory / "COMPLETED.json").exists():
        return _load_kmeans_stage(args, split)
    if directory.exists():
        raise FileExistsError(f"refusing to overwrite incomplete KMeans stage: {directory}")

    temporary = directory.with_name(
        directory.name + f".tmp.{os.getpid()}.{time.time_ns()}"
    )
    temporary.mkdir(parents=True, exist_ok=False)
    started = time.time()
    points = _materialize_points(args, split["train"])
    device = torch.device(args.device)
    points_device = points.to(device)
    kmeans = KMeansTorch(
        k=args.K,
        metric="euclidean",
        n_iter=args.kmeans_iterations,
        restarts=args.kmeans_restarts,
        tol=args.kmeans_tolerance,
        seed=args.seed,
        device=device,
        block_x=args.distance_batch_size,
        block_c=args.centroid_batch_size,
    )
    centroids_device = kmeans.fit(points_device)
    assignments_device = kmeans._assign_streamed(points_device, centroids_device)
    cluster_sizes = torch.bincount(assignments_device, minlength=args.K).cpu()
    directions = compute_cluster_pca_directions(
        points_device,
        assignments_device,
        centroids_device,
        rank=args.rank,
        chunk_elems=args.pca_chunk_elements,
        eig_batch_size=args.pca_eig_batch_size,
    ).float().cpu()
    centroids = centroids_device.float().cpu()
    assignments = assignments_device.cpu()
    save_centroid_artifact(temporary / "centroids.pt", centroids, directions)
    torch.save(
        {
            "assignments": assignments,
            "cluster_sizes": cluster_sizes,
            "split_fingerprint": split["fingerprint"],
            "population": "train",
            "assignment_rule": "nearest_euclidean_centroid",
        },
        temporary / "train_assignments.pt",
    )
    _write_json(
        temporary / "config.json",
        {
            "K": args.K,
            "ambient_dim": int(points.shape[1]),
            "rank": args.rank,
            "train_rows": int(points.shape[0]),
            "split_fingerprint": split["fingerprint"],
            "inertia": float(kmeans.inertia_),
            "iterations_last_recorded_restart": int(
                kmeans.n_iter_run_ or args.kmeans_iterations
            ),
            "cluster_size_min": int(cluster_sizes.min()),
            "cluster_size_max": int(cluster_sizes.max()),
            "fit_seconds_including_pca": time.time() - started,
        },
    )
    _mark_complete(temporary, directory / "centroids.pt")
    temporary.replace(directory)
    del points, points_device, centroids_device, assignments_device
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return centroids, directions, assignments, cluster_sizes


@torch.no_grad()
def _mfa_train_assignments(
    args: argparse.Namespace,
    model: MFA,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    model = model.to(args.device).eval()
    loader = _training_loader(
        args,
        positions,
        shuffle=False,
        batch_size=args.assignment_batch_size,
        num_workers=0,
    )
    assignments: list[torch.Tensor] = []
    with model.inference_cache(enabled=True):
        for batch in loader:
            assignments.append(
                model.responsibilities(batch.to(args.device)).argmax(dim=1).cpu()
            )
    resolved = torch.cat(assignments)
    return resolved, torch.bincount(resolved, minlength=model.K)


def _load_mfa_stage(
    args: argparse.Namespace,
    split: dict[str, Any],
) -> tuple[MFA, torch.Tensor, torch.Tensor]:
    directory = args.output_dir / "mfa"
    if not (directory / "COMPLETED.json").exists():
        raise FileNotFoundError(f"MFA stage is not complete: {directory}")
    model = load_mfa(directory / "mfa_model.pt", map_location="cpu").eval()
    expected_d = _source_ambient_dim(args)
    if (model.K, model.D, model.q) != (args.K, expected_d, args.rank):
        raise ValueError(
            "MFA shape mismatch: "
            f"stored (K={model.K}, D={model.D}, q={model.q}), "
            f"expected (K={args.K}, D={expected_d}, q={args.rank})"
        )
    assignment_artifact = torch.load(
        directory / "train_assignments.pt", map_location="cpu", weights_only=True
    )
    if assignment_artifact["split_fingerprint"] != split["fingerprint"]:
        raise ValueError("MFA artifact split fingerprint mismatch")
    assignments = assignment_artifact["assignments"].long()
    cluster_sizes = assignment_artifact["cluster_sizes"].long()
    if assignments.numel() != split["train"].numel():
        raise ValueError("MFA assignments do not cover the training split")
    if int(cluster_sizes.sum()) != split["train"].numel():
        raise ValueError("MFA cluster sizes do not sum to the training population")
    return model, assignments, cluster_sizes


def _fit_mfa(
    args: argparse.Namespace,
    split: dict[str, Any],
    centroids: torch.Tensor,
    directions: torch.Tensor,
) -> tuple[MFA, torch.Tensor, torch.Tensor]:
    directory = args.output_dir / "mfa"
    if (directory / "COMPLETED.json").exists():
        return _load_mfa_stage(args, split)
    directory.mkdir(parents=True, exist_ok=True)
    model_path = directory / "mfa_model.pt"
    training_path = directory / "training.json"

    if training_path.exists() and model_path.exists():
        model = load_mfa(model_path, map_location="cpu").eval()
    else:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
            torch.set_float32_matmul_precision("high")
        model = MFA(
            centroids=centroids,
            rank=args.rank,
            init_directions=directions,
        ).to(args.device)
        train_loader = _training_loader(args, split["train"], shuffle=True)
        if not (directory / "checkpoint.pt").exists():
            initialize_mixture_weights(
                model,
                centroids,
                train_loader,
                n_train_tokens=int(split["train"].numel()),
            )
        validation = _materialize_points(
            args,
            split["validation"],
            dtype=torch.float16,
        )
        if args.device != "cpu":
            validation = validation.pin_memory()
        started = time.time()
        training = train_nll(
            model,
            train_loader,
            val_tensor=validation,
            epochs=args.epochs,
            lr=args.learning_rate,
            save_path=str(model_path),
            save_func=save_mfa,
            ckpt_path=str(directory / "checkpoint.pt"),
            early_stop_delta=0.0,
            early_stop_patience=args.early_stop_patience,
            early_stop_min_delta=args.early_stop_min_delta,
            epoch_snapshot_every=0,
            log_interval=max(1, math.ceil(len(train_loader.dataset) / 4)),
        )
        training.update(
            elapsed_seconds=time.time() - started,
            split_fingerprint=split["fingerprint"],
            train_rows=int(split["train"].numel()),
            validation_rows=int(split["validation"].numel()),
        )
        save_mfa(model.cpu(), model_path, extra={"training": training})
        _write_json(training_path, training)

    assignments, cluster_sizes = _mfa_train_assignments(
        args, model, split["train"]
    )
    torch.save(
        {
            "assignments": assignments,
            "cluster_sizes": cluster_sizes,
            "split_fingerprint": split["fingerprint"],
            "population": "train",
            "assignment_rule": "mfa_responsibility_argmax",
        },
        directory / "train_assignments.pt",
    )
    _mark_complete(directory, model_path)
    return model.cpu().eval(), assignments, cluster_sizes


def _write_coverage_plot(path: Path, metrics: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    for method, result in metrics["methods"].items():
        curve = result["coverage_curve"]
        axis.plot(
            [item["radius"] for item in curve],
            [item["fraction"] for item in curve],
            marker="o",
            label=method,
        )
    axis.set_xlabel("ambient Euclidean radius")
    axis.set_ylabel("fraction of held-out test points covered")
    axis.set_xscale("log")
    axis.set_ylim(0.0, 1.01)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


@torch.no_grad()
def _write_embedding_plot(
    path: Path,
    projection_path: Path,
    *,
    train_points: torch.Tensor,
    test_points: torch.Tensor,
    kmeans_centroids: torch.Tensor,
    kmeans_live: torch.Tensor,
    kmeans_distances: torch.Tensor,
    mfa_centroids: torch.Tensor,
    mfa_live: torch.Tensor,
    mfa_distances: torch.Tensor,
    metrics: dict[str, Any],
    device: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fit = train_points.to(device=device, dtype=torch.float32)
    mean = fit.mean(dim=0)
    centered = fit - mean
    covariance = centered.T @ centered / max(1, fit.shape[0] - 1)
    _eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    basis = eigenvectors[:, -2:].flip(-1)

    test_2d = ((test_points.to(device) - mean) @ basis).cpu()
    kmeans_2d = ((kmeans_centroids.to(device) - mean) @ basis).cpu()
    mfa_2d = ((mfa_centroids.to(device) - mean) @ basis).cpu()
    projection_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "mean": mean.cpu(),
            "basis": basis.cpu(),
            "fit_population": "train",
            "projection": "global_pca",
        },
        projection_path,
    )

    panels = (
        ("KMeans+PCA", kmeans_2d[kmeans_live.cpu()], kmeans_distances),
        ("MFA", mfa_2d[mfa_live.cpu()], mfa_distances),
    )
    color_max = max(metrics["methods"][name]["r99"] for name, _, _ in panels)
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(13.5, 5.6),
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    scatter = None
    for axis, (name, centers_2d, distances) in zip(axes, panels, strict=True):
        summary = metrics["methods"][name]
        scatter = axis.scatter(
            test_2d[:, 0],
            test_2d[:, 1],
            c=distances,
            cmap="viridis",
            vmin=0.0,
            vmax=color_max,
            s=3,
            alpha=0.45,
            linewidths=0,
            rasterized=True,
        )
        axis.scatter(
            centers_2d[:, 0],
            centers_2d[:, 1],
            c="#e63946",
            marker="x",
            s=20,
            linewidths=0.8,
            label=f"live centroids ({summary['components_live']})",
        )
        axis.set_title(
            f"{name}\nmean={summary['mean']:.4f}, "
            f"r95={summary['r95']:.4f}, r99={summary['r99']:.4f}"
        )
        axis.set_xlabel("training PCA component 1")
        axis.grid(alpha=0.15)
        axis.legend(loc="best", fontsize=8)
    axes[0].set_ylabel("training PCA component 2")
    if scatter is not None:
        figure.colorbar(
            scatter,
            ax=axes,
            shrink=0.88,
            label="ambient distance to nearest live centroid (clipped at larger r99)",
        )
    figure.suptitle(
        f"Noiseless D={test_points.shape[1]} toy manifolds: untouched test split"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _write_result_readme(path: Path, metrics: dict[str, Any]) -> None:
    lines = [
        f"# D={metrics['ambient_dim']} noiseless held-out coverage test",
        "",
        f"Split fingerprint: `{metrics['split_fingerprint']}`",
        "",
        "| method | live / K | mean | r95 | r99 | maximum |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method, coverage in metrics["methods"].items():
        lines.append(
            f"| {method} | {coverage['components_live']} / {coverage['components_total']} | "
            f"{coverage['mean']:.6f} | {coverage['r95']:.6f} | "
            f"{coverage['r99']:.6f} | {coverage['maximum']:.6f} |"
        )
    lines.extend(
        [
            "",
            "Coverage is computed only on the untouched test split in the "
            f"original D={metrics['ambient_dim']} space.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def _render_coverage_outputs(
    args: argparse.Namespace,
    split: dict[str, Any],
    metrics: dict[str, Any],
    distances: dict[str, Any],
    kmeans_centroids: torch.Tensor,
    kmeans_sizes: torch.Tensor,
    mfa: MFA,
    mfa_sizes: torch.Tensor,
) -> None:
    _write_coverage_plot(args.output_dir / "figures" / "coverage_curve.png", metrics)
    train_points = _materialize_points(args, split["train"])
    test_points = _materialize_points(args, split["test"])
    _write_embedding_plot(
        args.output_dir / "figures" / "embedding_coverage_2d.png",
        args.output_dir / "figures" / "projection.pt",
        train_points=train_points,
        test_points=test_points,
        kmeans_centroids=kmeans_centroids,
        kmeans_live=kmeans_sizes > 0,
        kmeans_distances=distances["kmeans_pca"],
        mfa_centroids=mfa.mu.detach().cpu(),
        mfa_live=mfa_sizes > 0,
        mfa_distances=distances["mfa"],
        metrics=metrics,
        device=args.device,
    )
    _write_result_readme(args.output_dir / "README.md", metrics)


@torch.no_grad()
def _compute_coverage(
    args: argparse.Namespace,
    split: dict[str, Any],
    kmeans_centroids: torch.Tensor,
    kmeans_sizes: torch.Tensor,
    mfa: MFA,
    mfa_sizes: torch.Tensor,
) -> dict[str, Any]:
    directory = args.output_dir / "coverage"
    metrics_path = directory / "metrics.json"
    input_fingerprints = _coverage_input_fingerprints(args)
    if (directory / "COMPLETED.json").exists():
        if not metrics_path.exists() or not (directory / "distances.pt").exists():
            raise FileNotFoundError("completed coverage stage is missing an artifact")
        metrics = json.loads(metrics_path.read_text())
        distances = torch.load(
            directory / "distances.pt", map_location="cpu", weights_only=True
        )
        if metrics["split_fingerprint"] != split["fingerprint"]:
            raise ValueError("coverage artifact split fingerprint mismatch")
        if metrics.get("input_artifact_sha256") != input_fingerprints:
            raise ValueError("coverage artifact model fingerprint mismatch")
        if distances["split_fingerprint"] != split["fingerprint"]:
            raise ValueError("coverage distance split fingerprint mismatch")
        if distances.get("input_artifact_sha256") != input_fingerprints:
            raise ValueError("coverage distance model fingerprint mismatch")
        if not torch.equal(distances["test_positions"], split["test"]):
            raise ValueError("coverage distance rows do not match the test split")
        if not torch.equal(distances["test_global_rows"], split["test_global_rows"]):
            raise ValueError("coverage global rows do not match the test split")
        if not (args.output_dir / "COMPLETED.json").exists():
            _mark_complete(args.output_dir, directory / "metrics.json")
        return metrics
    if directory.exists():
        if not metrics_path.exists() or not (directory / "distances.pt").exists():
            raise FileExistsError(
                f"refusing to reuse incomplete coverage stage: {directory}"
            )
        metrics = json.loads(metrics_path.read_text())
        distances = torch.load(
            directory / "distances.pt", map_location="cpu", weights_only=True
        )
    else:
        temporary = directory.with_name(
            directory.name + f".tmp.{os.getpid()}.{time.time_ns()}"
        )
        temporary.mkdir(parents=True, exist_ok=False)
        test_points = _materialize_points(args, split["test"])
        kmeans_summary, kmeans_distances = evaluate_heldout_distribution_coverage(
            test_points,
            kmeans_centroids.to(args.device),
            live_components=(kmeans_sizes > 0).to(args.device),
            thresholds=args.coverage_thresholds,
            batch_size=args.coverage_batch_size,
        )
        mfa_summary, mfa_distances = evaluate_heldout_distribution_coverage(
            test_points,
            mfa.mu.detach().to(args.device),
            live_components=(mfa_sizes > 0).to(args.device),
            thresholds=args.coverage_thresholds,
            batch_size=args.coverage_batch_size,
        )
        metrics = {
            "schema_version": 1,
            "metric": "heldout_distribution_coverage",
            "ambient_dim": int(test_points.shape[1]),
            "test_rows": int(test_points.shape[0]),
            "split_fingerprint": split["fingerprint"],
            "input_artifact_sha256": input_fingerprints,
            "methods": {"KMeans+PCA": kmeans_summary, "MFA": mfa_summary},
        }
        distances = {
            "test_positions": split["test"],
            "test_global_rows": split["test_global_rows"],
            "split_fingerprint": split["fingerprint"],
            "input_artifact_sha256": input_fingerprints,
            "kmeans_pca": kmeans_distances,
            "mfa": mfa_distances,
        }
        _write_json(temporary / "metrics.json", metrics)
        torch.save(distances, temporary / "distances.pt")
        temporary.replace(directory)

    if metrics["split_fingerprint"] != split["fingerprint"]:
        raise ValueError("coverage artifact split fingerprint mismatch")
    if metrics.get("input_artifact_sha256") != input_fingerprints:
        raise ValueError("coverage artifact model fingerprint mismatch")
    if distances["split_fingerprint"] != split["fingerprint"]:
        raise ValueError("coverage distance split fingerprint mismatch")
    if distances.get("input_artifact_sha256") != input_fingerprints:
        raise ValueError("coverage distance model fingerprint mismatch")
    if not torch.equal(distances["test_positions"], split["test"]):
        raise ValueError("coverage distance rows do not match the test split")
    if not torch.equal(distances["test_global_rows"], split["test_global_rows"]):
        raise ValueError("coverage global rows do not match the test split")
    if distances["kmeans_pca"].numel() != split["test"].numel():
        raise ValueError("KMeans coverage distances do not cover the test split")
    if distances["mfa"].numel() != split["test"].numel():
        raise ValueError("MFA coverage distances do not cover the test split")

    _render_coverage_outputs(
        args,
        split,
        metrics,
        distances,
        kmeans_centroids,
        kmeans_sizes,
        mfa,
        mfa_sizes,
    )
    _mark_complete(directory, directory / "metrics.json")
    _mark_complete(args.output_dir, directory / "metrics.json")
    return metrics


def _validate_geometry_metrics(
    metrics: dict[str, Any],
    *,
    split: dict[str, Any],
    input_fingerprints: dict[str, str],
) -> None:
    if metrics.get("metric") != "toy_manifold_tangent_geometry":
        raise ValueError("geometry artifact has the wrong metric identifier")
    if metrics.get("split_fingerprint") != split["fingerprint"]:
        raise ValueError("geometry artifact split fingerprint mismatch")
    if metrics.get("input_artifact_sha256") != input_fingerprints:
        raise ValueError("geometry artifact input fingerprint mismatch")
    if metrics.get("kmeans_covariance") != (
        "stored_pc_directions_scaled_by_training_assignment_variance"
    ):
        raise ValueError("geometry artifact has the wrong KMeans covariance definition")
    for method_name in ("KMeans+PCA", "MFA"):
        method = metrics.get("methods", {}).get(method_name)
        if method is None:
            raise ValueError(f"geometry artifact is missing {method_name}")
        association = method["association"]
        associated = int(association["associated_components"])
        outside = int(association["outside_cutoff_components"])
        ambiguous = int(association["ambiguous_components"])
        if associated + outside + ambiguous != int(metrics["K"]):
            raise ValueError(f"{method_name} geometry populations do not sum to K")
        for metric_name in ("tangent_alignment", "tangent_containment"):
            metric = method[metric_name]
            for score_name in ("subspace_overlap", "worst_direction_cosine"):
                score = metric[score_name]
                valid = int(score["valid_components"])
                undefined = int(score["undefined_components"])
                if valid + undefined != associated:
                    raise ValueError(
                        f"{method_name} {metric_name} population mismatch"
                    )
                mean = score["mean"]
                if mean is not None and not (0.0 <= float(mean) <= 1.0):
                    raise ValueError(
                        f"{method_name} {metric_name} {score_name} is outside [0, 1]"
                    )


@torch.no_grad()
def _compute_geometry(
    args: argparse.Namespace,
    split: dict[str, Any],
    kmeans_centroids: torch.Tensor,
    kmeans_directions: torch.Tensor,
    kmeans_assignments: torch.Tensor,
    kmeans_sizes: torch.Tensor,
    mfa: MFA,
    mfa_sizes: torch.Tensor,
) -> dict[str, Any]:
    directory = args.output_dir / "geometry"
    metrics_path = directory / "metrics.json"
    input_fingerprints = _geometry_input_fingerprints(args)
    if (directory / "COMPLETED.json").exists():
        if not metrics_path.exists() or not (directory / "kmeans_pca_variances.pt").exists():
            raise FileNotFoundError("completed geometry stage is missing an artifact")
        metrics = json.loads(metrics_path.read_text())
        _validate_geometry_metrics(
            metrics,
            split=split,
            input_fingerprints=input_fingerprints,
        )
        return metrics
    if directory.exists():
        raise FileExistsError(f"refusing to reuse incomplete geometry stage: {directory}")

    metadata = torch.load(
        args.shard_dir / "manifold_metadata.pt",
        map_location="cpu",
        weights_only=True,
    )
    kmeans_variances = _kmeans_pca_variances(
        args,
        split,
        kmeans_centroids,
        kmeans_directions,
        kmeans_assignments,
    )
    kmeans_geometry = evaluate_toy_manifold_metrics(
        _KMeansPCAGeometry(
            kmeans_centroids,
            kmeans_directions,
            kmeans_variances,
        ),
        metadata,
        kmeans_sizes > 0,
        rank_threshold=args.geometry_rank_threshold,
    )
    mfa_geometry = evaluate_toy_manifold_metrics(
        mfa.cpu().eval(),
        metadata,
        mfa_sizes > 0,
        rank_threshold=args.geometry_rank_threshold,
    )
    metrics = {
        "schema_version": 1,
        "metric": "toy_manifold_tangent_geometry",
        "population": "proximity_associated_components",
        "K": int(mfa.K),
        "ambient_dim": int(mfa.D),
        "kmeans_covariance": (
            "stored_pc_directions_scaled_by_training_assignment_variance"
        ),
        "split_fingerprint": split["fingerprint"],
        "input_artifact_sha256": input_fingerprints,
        "rank_threshold": float(args.geometry_rank_threshold),
        "methods": {
            "KMeans+PCA": kmeans_geometry,
            "MFA": mfa_geometry,
        },
    }
    _validate_geometry_metrics(
        metrics,
        split=split,
        input_fingerprints=input_fingerprints,
    )

    temporary = directory.with_name(
        directory.name + f".tmp.{os.getpid()}.{time.time_ns()}"
    )
    temporary.mkdir(parents=True, exist_ok=False)
    _write_json(temporary / "metrics.json", metrics)
    torch.save(kmeans_variances, temporary / "kmeans_pca_variances.pt")
    _mark_complete(temporary, metrics_path)
    temporary.replace(directory)
    return metrics


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.shard_dir = args.shard_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    args.coverage_thresholds = tuple(float(value) for value in args.coverage_thresholds)
    if args.coverage_thresholds != tuple(sorted(set(args.coverage_thresholds))):
        raise ValueError("coverage thresholds must be sorted and unique")
    if args.geometry_rank_threshold <= 0.0:
        raise ValueError("geometry rank threshold must be positive")

    source_config = json.loads((args.shard_dir / "config.json").read_text())
    if int(source_config["window"]) != 1 or int(source_config.get("drop_prefix", 0)) != 0:
        raise ValueError("this experiment requires one activation per toy-manifold row")
    _ensure_experiment_config(args, source_config)
    meta_index = load_meta_index(args.shard_dir, layer=args.layer)
    split = _ensure_split(args, meta_index)

    if args.coverage_only or args.geometry_only:
        kmeans_centroids, directions, assignments, kmeans_sizes = _load_kmeans_stage(
            args, split
        )
        mfa, _mfa_assignments, mfa_sizes = _load_mfa_stage(args, split)
    else:
        kmeans_centroids, directions, assignments, kmeans_sizes = _fit_kmeans_pca(
            args, split
        )
        mfa, _mfa_assignments, mfa_sizes = _fit_mfa(
            args, split, kmeans_centroids, directions
        )
    if args.geometry_only:
        return _compute_geometry(
            args,
            split,
            kmeans_centroids,
            directions,
            assignments,
            kmeans_sizes,
            mfa,
            mfa_sizes,
        )
    return _compute_coverage(
        args,
        split,
        kmeans_centroids,
        kmeans_sizes,
        mfa,
        mfa_sizes,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--K", type=int, default=1000)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--kmeans-iterations", type=int, default=100)
    parser.add_argument("--kmeans-restarts", type=int, default=10)
    parser.add_argument("--kmeans-tolerance", type=float, default=1e-6)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--load-batch-size", type=int, default=20_000)
    parser.add_argument("--assignment-batch-size", type=int, default=4096)
    parser.add_argument("--coverage-batch-size", type=int, default=4096)
    parser.add_argument("--distance-batch-size", type=int, default=8192)
    parser.add_argument("--centroid-batch-size", type=int, default=8192)
    parser.add_argument("--pca-chunk-elements", type=int, default=1 << 23)
    parser.add_argument("--pca-eig-batch-size", type=int, default=256)
    parser.add_argument(
        "--coverage-thresholds",
        type=float,
        nargs="+",
        default=(
            0.05,
            0.1,
            0.2,
            0.3,
            0.5,
            0.75,
            1.0,
            1.5,
            2.0,
            3.0,
            5.0,
            10.0,
            20.0,
        ),
    )
    parser.add_argument("--coverage-only", action="store_true")
    parser.add_argument("--geometry-only", action="store_true")
    parser.add_argument("--geometry-rank-threshold", type=float, default=1.0)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    metrics = run(args)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
