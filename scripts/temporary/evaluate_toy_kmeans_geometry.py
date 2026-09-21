"""Evaluate a KMeans initialization with a Cattell rank-threshold sweep.

The evaluator reuses saved KMeans centroids and empirical principal directions,
or computes missing directions for eligible clusters with --compute-pca.
It derives one Cattell rank per eligible cluster and threshold from
the empirical covariance spectrum, then evaluates rank recovery, matched-rank
tangent alignment, and learned-rank tangent containment. Metrics requiring a
probabilistic MFA density remain undefined.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import torch
from sklearn.metrics import (
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
    normalized_mutual_info_score,
)
from torch.utils.data import DataLoader

from dalg.data.shard_activations import ActivationBatchDataset, load_meta_index
from dalg.evaluation.toy_manifold_metrics import (
    _alignment_summary,
    _associate_component_means,
    _leading_subspace_is_identifiable,
    _rank_summary,
    _subspace_alignment,
)
from dalg.init.centroid_artifact import (
    load_centroid_artifact,
    validate_centroid_artifact,
)


DEFAULT_CATTELL_THRESHOLDS = (
    0.005,
    0.01,
    0.05,
    0.1,
    0.15,
    0.2,
    0.25,
    0.3,
    0.35,
    0.4,
    0.5,
    1.0,
)


def _resolve_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("KMeans geometry evaluation requested unavailable CUDA")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("KMeans geometry evaluation requested unavailable MPS")
    return device


def _same_path(left: str | Path, right: str | Path) -> bool:
    return Path(left).expanduser().resolve() == Path(right).expanduser().resolve()


def _validate_thresholds(values: Iterable[float]) -> tuple[float, ...]:
    thresholds = tuple(float(value) for value in values)
    if not thresholds:
        raise ValueError("at least one Cattell threshold is required")
    if any(not math.isfinite(value) or value <= 0.0 for value in thresholds):
        raise ValueError("Cattell thresholds must be finite and positive")
    if any(right <= left for left, right in zip(thresholds, thresholds[1:])):
        raise ValueError("Cattell thresholds must be strictly increasing")
    return thresholds


def _load_inputs(
    *,
    centroids_path: Path,
    assignments_path: Path,
    shard_dir: Path,
    layer: int,
    compute_pca: bool = False,
) -> tuple[
    torch.Tensor,
    torch.Tensor | None,
    dict[str, Any],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    required = [
        centroids_path,
        centroids_path.parent / "config.json",
        assignments_path,
        shard_dir / "config.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing KMeans evaluation artifacts: {missing}")

    centroid_config = json.loads(
        (centroids_path.parent / "config.json").read_text()
    )
    shard_config = json.loads((shard_dir / "config.json").read_text())
    if centroid_config.get("method") != "kmeans":
        raise ValueError("centroid metadata does not describe a KMeans fit")
    if centroid_config.get("metric") != "euclidean":
        raise ValueError("centroid metadata does not use Euclidean distance")
    if centroid_config.get("centroid_artifact_format") != "dalg_centroids_v1":
        raise ValueError("unexpected centroid artifact format")
    if shard_config.get("source_kind") != "toy_manifolds":
        raise ValueError("evaluation requires shards from save_toy_manifold_shards")
    if int(shard_config["window"]) != 1 or int(
        shard_config.get("drop_prefix", 0)
    ) != 0:
        raise ValueError("evaluation expects exactly one activation per shard row")
    if layer not in [int(value) for value in shard_config["layers"]]:
        raise ValueError(f"layer {layer} is absent from the activation shards")

    metadata_path = shard_dir / shard_config["manifold_metadata"]
    if not metadata_path.is_file():
        raise FileNotFoundError(f"missing toy-manifold metadata: {metadata_path}")
    manifold_metadata = torch.load(
        metadata_path,
        map_location="cpu",
        weights_only=True,
    )

    centroids, principal_components = load_centroid_artifact(
        centroids_path,
        map_location="cpu",
        mmap=True,
    )
    validate_centroid_artifact(
        centroids,
        principal_components,
        expected_k=int(centroid_config["K"]),
        expected_d=int(shard_config["d_model"]),
    )
    if principal_components is None and not compute_pca:
        raise ValueError("centroid artifact does not contain cluster PCA directions")
    if compute_pca and principal_components is not None:
        raise ValueError("--compute-pca expects an artifact without saved PCA directions")
    if not torch.isfinite(centroids).all() or (
        principal_components is not None
        and not torch.isfinite(principal_components).all()
    ):
        raise ValueError("centroid artifact contains non-finite values")

    K, D = map(int, centroids.shape)
    pca_config = centroid_config.get("principal_components")
    if principal_components is not None:
        pca_capacity = int(principal_components.shape[-1])
        if not isinstance(pca_config, dict):
            raise ValueError("centroid metadata does not describe its PCA directions")
        if int(pca_config["rank"]) != pca_capacity or pca_config["shape"] != list(
            principal_components.shape
        ):
            raise ValueError("PCA artifact shape differs from centroid metadata")
        if not bool(pca_config.get("uses_all_rows")):
            raise ValueError("PCA metadata does not describe a full-data fit")
    elif pca_config is not None:
        raise ValueError("PCA metadata exists but the artifact has no directions")

    assignment_bundle = torch.load(
        assignments_path,
        map_location="cpu",
        mmap=True,
        weights_only=True,
    )
    assignments = assignment_bundle["assignments"].reshape(-1).long()
    cluster_sizes = assignment_bundle["cluster_sizes"].reshape(-1).long()
    min_distances = assignment_bundle["min_distances"].reshape(-1).float()
    expected_rows = int(shard_config["num_rows"])
    if assignment_bundle.get("subset_spec") is not None:
        raise ValueError("this full-data experiment does not accept subset assignments")
    if int(assignment_bundle["K"]) != K or cluster_sizes.numel() != K:
        raise ValueError("assignment K does not match centroid K")
    if assignments.numel() != expected_rows:
        raise ValueError(
            f"expected {expected_rows} assignments, got {assignments.numel()}"
        )
    if min_distances.numel() != expected_rows:
        raise ValueError("min_distances must contain one value per assignment")
    if not torch.isfinite(min_distances).all() or torch.any(min_distances < 0.0):
        raise ValueError("nearest-centroid distances must be finite and non-negative")
    if assignments.numel() and (
        int(assignments.min()) < 0 or int(assignments.max()) >= K
    ):
        raise ValueError(f"assignments must lie in [0, {K - 1}]")
    actual_sizes = torch.bincount(assignments, minlength=K)
    if not torch.equal(actual_sizes, cluster_sizes):
        raise ValueError("cluster_sizes is inconsistent with assignments")
    if int(cluster_sizes.sum()) != expected_rows:
        raise ValueError("cluster_sizes does not cover the full activation stream")
    configured_sizes = torch.as_tensor(
        centroid_config["cluster_sizes"], dtype=torch.long
    )
    if not torch.equal(cluster_sizes, configured_sizes):
        raise ValueError("nearest-centroid sizes differ from centroid-fit metadata")

    source = assignment_bundle.get("source", {})
    if not _same_path(assignment_bundle["centroids_path"], centroids_path):
        raise ValueError("assignment centroid provenance does not match input artifact")
    if not _same_path(source["shard_dir"], shard_dir):
        raise ValueError("assignment shard provenance does not match input shards")
    if int(source["layer"]) != layer or int(source["drop_prefix"]) != 0:
        raise ValueError("assignment layer/drop-prefix provenance is inconsistent")
    if int(source["num_items"]) != expected_rows:
        raise ValueError("assignment provenance has the wrong item count")
    if not bool(centroid_config.get("uses_all_rows")):
        raise ValueError("centroid metadata does not describe a full-data fit")
    if int(centroid_config["rows_used"]) != expected_rows:
        raise ValueError("centroid fit row count differs from evaluation row count")
    if not _same_path(centroid_config["source_shard_dir"], shard_dir):
        raise ValueError("centroid source shards differ from evaluation shards")

    meta_index = load_meta_index(shard_dir, layer=layer)
    row_manifold_ids = manifold_metadata["row_manifold_ids"].reshape(-1).long()
    if len(meta_index) != expected_rows or row_manifold_ids.numel() != expected_rows:
        raise ValueError("toy labels, metadata index, and shard rows are not aligned")

    return (
        centroids.float(),
        principal_components.float() if principal_components is not None else None,
        assignment_bundle,
        assignments,
        cluster_sizes,
        min_distances,
        manifold_metadata,
        shard_config,
        centroid_config,
    )


@torch.no_grad()
def _empirical_eigenvalues_and_pca_validation(
    *,
    shard_dir: Path,
    layer: int,
    centroids: torch.Tensor,
    principal_components: torch.Tensor | None,
    pca_capacity: int,
    assignments: torch.Tensor,
    cluster_sizes: torch.Tensor,
    eligible: torch.Tensor,
    batch_size: int,
    chunk_elems: int,
    eig_batch_size: int,
    device: torch.device,
    pca_validation_tolerance: float,
    centroid_mean_tolerance: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    if batch_size <= 0 or chunk_elems <= 0 or eig_batch_size <= 0:
        raise ValueError("batch and chunk sizes must be positive")
    if pca_validation_tolerance <= 0.0 or centroid_mean_tolerance <= 0.0:
        raise ValueError("validation tolerances must be positive")
    if not bool(eligible.any()):
        raise ValueError("no clusters satisfy the evaluation population cutoff")

    K, D = map(int, centroids.shape)
    compute_pca = principal_components is None
    dataset = ActivationBatchDataset(
        shard_dir,
        layer=layer,
        drop_prefix=0,
        batch_size=batch_size,
        dtype=torch.float32,
        shuffle_shards=False,
        shuffle_within_shard=False,
        seed=0,
    )
    loader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    centers = centroids.to(device=device, dtype=torch.float64)
    scatter = torch.zeros(K, D, D, dtype=torch.float64, device=device)
    point_sums = torch.zeros(K, D, dtype=torch.float64, device=device)
    rows_per_chunk = max(1, chunk_elems // (D * D))
    row_offset = 0
    for batch in loader:
        batch_rows = int(batch.shape[0])
        batch_assignments = assignments[row_offset : row_offset + batch_rows]
        if batch_assignments.numel() != batch_rows:
            raise ValueError("activation stream contains more rows than assignments")
        batch = batch.to(
            device=device,
            dtype=torch.float32,
            non_blocking=(device.type == "cuda"),
        )
        batch_assignments = batch_assignments.to(device=device)
        for start in range(0, batch_rows, rows_per_chunk):
            stop = min(start + rows_per_chunk, batch_rows)
            labels = batch_assignments[start:stop]
            points = batch[start:stop].double()
            residual = points - centers[labels]
            point_sums.index_add_(0, labels, points)
            scatter.index_add_(
                0,
                labels,
                residual[:, :, None] * residual[:, None, :],
            )
        row_offset += batch_rows
    if row_offset != assignments.numel():
        raise ValueError(
            f"activation stream has {row_offset} rows, expected {assignments.numel()}"
        )

    sizes_device = cluster_sizes.to(device=device, dtype=torch.float64)
    covariance = scatter / sizes_device.clamp_min(1.0)[:, None, None]
    covariance = 0.5 * (covariance + covariance.transpose(-1, -2))
    stored_pcs = (
        torch.zeros(K, D, pca_capacity, device=device, dtype=torch.float64)
        if compute_pca
        else principal_components.to(device=device, dtype=torch.float64)
    )
    eligible_device = eligible.to(device=device)

    empirical_means = point_sums / sizes_device.clamp_min(1.0)[:, None]
    centroid_errors = (empirical_means - centers).norm(dim=1)[eligible_device]
    max_centroid_mean_l2_error = float(centroid_errors.max())
    mean_centroid_mean_l2_error = float(centroid_errors.mean())
    if not compute_pca and max_centroid_mean_l2_error > centroid_mean_tolerance:
        raise ValueError(
            "stored centroids differ from assigned empirical means: "
            f"max_l2_error={max_centroid_mean_l2_error:.8g}, "
            f"tolerance={centroid_mean_tolerance:.8g}"
        )
    if compute_pca:
        # A finite-iteration KMeans fit need not equal the final partition means.
        # PCA uses centered covariance; proximity still uses the saved centroids.
        mean_residual = empirical_means - centers
        covariance -= mean_residual[:, :, None] * mean_residual[:, None, :]

    leading_eigenvalues = torch.zeros(
        K,
        pca_capacity + 1,
        dtype=torch.float64,
        device="cpu",
    )
    max_relative_subspace_residual = 0.0
    max_relative_eigenvalue_error = 0.0
    tiny = torch.finfo(torch.float64).tiny
    eligible_ids = eligible.nonzero(as_tuple=True)[0].tolist()
    for offset in range(0, len(eligible_ids), eig_batch_size):
        component_ids = eligible_ids[offset : offset + eig_batch_size]
        component_index = torch.tensor(component_ids, device=device)
        batch_covariance = covariance[component_index]
        if compute_pca:
            eigenvalues, eigenvectors = torch.linalg.eigh(batch_covariance)
            eigenvalues = eigenvalues.flip(1)
            stored_pcs[component_index] = eigenvectors[:, :, -pca_capacity:].flip(2)
        else:
            eigenvalues = torch.linalg.eigvalsh(batch_covariance).flip(1)
        batch_pcs = stored_pcs[component_index]
        leading_eigenvalues[component_ids] = eigenvalues[
            :, : pca_capacity + 1
        ].cpu()

        covariance_pcs = torch.bmm(batch_covariance, batch_pcs)
        projected = torch.bmm(batch_pcs.transpose(1, 2), covariance_pcs)
        residual = covariance_pcs - torch.bmm(batch_pcs, projected)
        relative_residual = residual.norm(dim=(1, 2)) / covariance_pcs.norm(
            dim=(1, 2)
        ).clamp_min(tiny)
        max_relative_subspace_residual = max(
            max_relative_subspace_residual,
            float(relative_residual.max()),
        )

        projected_eigenvalues = torch.linalg.eigvalsh(projected).flip(1)
        relative_eigenvalue_error = (
            (projected_eigenvalues - eigenvalues[:, :pca_capacity]).abs()
            / eigenvalues[:, :pca_capacity].abs().clamp_min(tiny)
        )
        max_relative_eigenvalue_error = max(
            max_relative_eigenvalue_error,
            float(relative_eigenvalue_error.max()),
        )

    gram = torch.bmm(stored_pcs.transpose(1, 2), stored_pcs)
    identity = torch.eye(pca_capacity, dtype=torch.float64, device=device)[None]
    max_orthonormal_error = float(
        (gram[eligible_device] - identity).abs().max()
    )
    validation = {
        "source": "computed_from_assignments" if compute_pca else "centroid_artifact",
        "covariance_center": "empirical_cluster_mean" if compute_pca else "saved_centroid",
        "centroid_mean_agreement_required": not compute_pca,
        "pca_tolerance": float(pca_validation_tolerance),
        "centroid_mean_l2_tolerance": float(centroid_mean_tolerance),
        "max_centroid_mean_l2_error": max_centroid_mean_l2_error,
        "mean_centroid_mean_l2_error": mean_centroid_mean_l2_error,
        "max_orthonormal_error": max_orthonormal_error,
        "max_relative_subspace_residual": max_relative_subspace_residual,
        "max_relative_eigenvalue_error": max_relative_eigenvalue_error,
    }
    pca_errors = (
        max_orthonormal_error,
        max_relative_subspace_residual,
        max_relative_eigenvalue_error,
    )
    if max(pca_errors) > pca_validation_tolerance:
        raise ValueError(
            "PCA directions failed empirical covariance validation: "
            f"{validation}"
        )
    return leading_eigenvalues, stored_pcs.cpu(), validation


def _cattell_rank_sweep(
    leading_eigenvalues: torch.Tensor,
    eligible: torch.Tensor,
    *,
    q_max: int,
    thresholds: tuple[float, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the raw HDDC Cattell rule, without shared-noise active-set pruning."""
    if leading_eigenvalues.ndim != 2:
        raise ValueError("leading_eigenvalues must be a matrix")
    if leading_eigenvalues.shape[1] < q_max + 1:
        raise ValueError(f"Cattell q_max={q_max} requires q_max+1 eigenvalues")

    lam = leading_eigenvalues[:, : q_max + 1].clamp_min(0.0)
    denom = leading_eigenvalues[:, 0].clamp_min(
        torch.finfo(leading_eigenvalues.dtype).tiny
    )
    normalized_gaps = (lam[:, :-1] - lam[:, 1:]) / denom[:, None]
    if not torch.isfinite(normalized_gaps[eligible]).all():
        raise ValueError("eligible Cattell gaps contain non-finite values")

    K = int(leading_eigenvalues.shape[0])
    ranks = torch.full((len(thresholds), K), -1, dtype=torch.long)
    dimensions = torch.arange(1, q_max + 1, dtype=torch.long)[None, :]
    for threshold_index, threshold in enumerate(thresholds):
        above = normalized_gaps[eligible] > threshold
        selected = torch.where(
            above,
            dimensions.expand_as(above),
            torch.zeros_like(dimensions).expand_as(above),
        ).max(dim=1).values
        ranks[threshold_index, eligible] = selected.clamp(min=1, max=q_max)

    eligible_ranks = ranks[:, eligible]
    if torch.any(eligible_ranks[1:] > eligible_ranks[:-1]):
        raise RuntimeError("Cattell ranks increased as the threshold increased")
    if torch.any((eligible_ranks < 1) | (eligible_ranks > q_max)):
        raise RuntimeError("Cattell ranks lie outside [1, q_max]")
    return normalized_gaps, ranks


def _association_summary(
    associations,
    K: int,
    cutoff: float | None,
    eligible: torch.Tensor,
) -> dict[str, Any]:
    result = {
        "rule": (
            "unique_nearest_exact_projection"
            if cutoff is None
            else "unique_nearest_exact_projection_within_cutoff"
        ),
        "max_mean_to_manifold_distance": (
            None if cutoff is None else float(cutoff)
        ),
        "population": "all_saved_kmeans_centroids",
        "associated_components": int(associations.associated.sum()),
        "outside_cutoff_components": int(associations.outside_cutoff.sum()),
        "ambiguous_components": int(associations.ambiguous.sum()),
        "eligible_associated_components": int(
            (associations.associated & eligible).sum()
        ),
    }
    if sum(
        result[key]
        for key in (
            "associated_components",
            "outside_cutoff_components",
            "ambiguous_components",
        )
    ) != K:
        raise RuntimeError("association populations do not sum to K")
    return result


def _finite_value(value: float, *, name: str) -> float:
    if not math.isfinite(value):
        raise RuntimeError(f"{name} is not finite: {value}")
    return value


def _bounded_value(value: float, *, name: str, lower: float, upper: float) -> float:
    value = _finite_value(value, name=name)
    if not lower - 1e-12 <= value <= upper + 1e-12:
        raise RuntimeError(f"{name} lies outside [{lower}, {upper}]: {value}")
    return min(upper, max(lower, value))


def _numeric_summary(
    values: torch.Tensor,
    population: torch.Tensor | None = None,
) -> dict[str, float | int | None]:
    values = values.detach().cpu().double().reshape(-1)
    if population is not None:
        population = population.detach().cpu().bool().reshape(-1)
        if population.numel() != values.numel():
            raise ValueError("summary population does not match values")
        values = values[population]
    count = int(values.numel())
    if count == 0:
        return {
            "count": 0,
            "min": None,
            "q25": None,
            "median": None,
            "mean": None,
            "q75": None,
            "max": None,
        }
    if not torch.isfinite(values).all():
        raise ValueError("summary values contain non-finite entries")
    quantiles = torch.quantile(
        values,
        torch.tensor([0.25, 0.5, 0.75], dtype=torch.float64),
    )
    return {
        "count": count,
        "min": float(values.min()),
        "q25": float(quantiles[0]),
        "median": float(quantiles[1]),
        "mean": float(values.mean()),
        "q75": float(quantiles[2]),
        "max": float(values.max()),
    }


def _rank_distribution(
    ranks: torch.Tensor,
    population: torch.Tensor,
    q_max: int,
) -> dict[str, int]:
    selected = ranks[population]
    return {
        str(rank): int((selected == rank).sum())
        for rank in range(1, q_max + 1)
    }


def _compute_alignment(
    *,
    associations,
    intrinsic_dims: torch.Tensor,
    principal_components: torch.Tensor,
    leading_eigenvalues: torch.Tensor,
    population: torch.Tensor,
    ambient_dim: int,
    eigengap_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    K = int(principal_components.shape[0])
    overlap = torch.zeros(K, dtype=torch.float64)
    worst = torch.zeros(K, dtype=torch.float64)
    defined = torch.zeros(K, dtype=torch.bool)
    for component_id in population.nonzero(as_tuple=True)[0].tolist():
        manifold_index = int(associations.manifold_indices[component_id])
        tangent = associations.tangent_bases[component_id]
        if tangent is None:
            continue
        intrinsic_dim = int(intrinsic_dims[manifold_index])
        if not _leading_subspace_is_identifiable(
            leading_eigenvalues[component_id],
            intrinsic_dim,
            ambient_dim,
            eigengap_threshold,
        ):
            continue
        component_overlap, component_worst = _subspace_alignment(
            tangent,
            principal_components[component_id, :, :intrinsic_dim],
        )
        overlap[component_id] = component_overlap
        worst[component_id] = component_worst
        defined[component_id] = True
    return overlap, worst, defined


def _compute_containment(
    *,
    associations,
    principal_components: torch.Tensor,
    leading_eigenvalues: torch.Tensor,
    ranks: torch.Tensor,
    population: torch.Tensor,
    ambient_dim: int,
    eigengap_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    K = int(principal_components.shape[0])
    overlap = torch.zeros(K, dtype=torch.float64)
    worst = torch.zeros(K, dtype=torch.float64)
    defined = torch.zeros(K, dtype=torch.bool)
    for component_id in population.nonzero(as_tuple=True)[0].tolist():
        tangent = associations.tangent_bases[component_id]
        if tangent is None:
            continue
        rank = int(ranks[component_id])
        if not _leading_subspace_is_identifiable(
            leading_eigenvalues[component_id],
            rank,
            ambient_dim,
            eigengap_threshold,
        ):
            continue
        component_overlap, component_worst = _subspace_alignment(
            tangent,
            principal_components[component_id, :, :rank],
        )
        overlap[component_id] = component_overlap
        worst[component_id] = component_worst
        defined[component_id] = True
    return overlap, worst, defined


def _alignment_report(
    *,
    definition: str,
    overlap: torch.Tensor,
    worst: torch.Tensor,
    defined: torch.Tensor,
    population: torch.Tensor,
    eigengap_threshold: float,
) -> dict[str, Any]:
    return {
        "definition": definition,
        "population": "proximity_associated_and_population_eligible_components",
        "aggregation": "unweighted_component_mean",
        "relative_boundary_eigengap_threshold": float(eigengap_threshold),
        **_alignment_summary(overlap, worst, defined, population),
    }


def _per_manifold_base(
    *,
    manifold_metadata: dict[str, Any],
    associations,
    cluster_sizes: torch.Tensor,
    eligible: torch.Tensor,
    alignment_overlap: torch.Tensor,
    alignment_worst: torch.Tensor,
    alignment_defined: torch.Tensor,
) -> list[dict[str, Any]]:
    assignment_live = cluster_sizes > 0
    rows = []
    for manifold_index, manifold in enumerate(manifold_metadata["manifolds"]):
        associated = associations.manifold_indices == manifold_index
        evaluated = associated & eligible
        rows.append(
            {
                "manifold_id": int(manifold["manifold_id"]),
                "type_id": int(manifold["type_id"]),
                "type_name": str(manifold["type_name"]),
                "intrinsic_dim": int(manifold["intrinsic_dim"]),
                "components": {
                    "associated": int(associated.sum()),
                    "eligible_associated": int(evaluated.sum()),
                    "excluded_below_min_population": int(
                        (associated & ~eligible).sum()
                    ),
                    "assignment_live": int((associated & assignment_live).sum()),
                    "assignment_dead": int((associated & ~assignment_live).sum()),
                },
                "mean_to_manifold_distance": _numeric_summary(
                    associations.nearest_distances,
                    associated,
                ),
                "tangent_alignment": _alignment_summary(
                    alignment_overlap,
                    alignment_worst,
                    alignment_defined,
                    evaluated,
                ),
            }
        )
    return rows


def _validate_score_report(report: dict[str, Any], *, name: str) -> None:
    for score_name in ("subspace_overlap", "worst_direction_cosine"):
        summary = report[score_name]
        if summary["valid_components"] + summary["undefined_components"] < 0:
            raise RuntimeError(f"{name} has invalid component counts")
        if summary["mean"] is not None:
            _bounded_value(
                float(summary["mean"]),
                name=f"{name}.{score_name}",
                lower=0.0,
                upper=1.0,
            )


@torch.no_grad()
def evaluate_kmeans_geometry(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    thresholds = _validate_thresholds(args.cattell_thresholds)
    if args.q_max <= 0:
        raise ValueError("q_max must be positive")
    if args.min_population <= 0:
        raise ValueError("min_population must be positive")
    if args.max_mean_to_manifold_distance is not None and (
        not math.isfinite(args.max_mean_to_manifold_distance)
        or args.max_mean_to_manifold_distance <= 0.0
    ):
        raise ValueError("mean-to-manifold cutoff must be finite and positive")
    if args.relative_boundary_eigengap_threshold <= 0.0:
        raise ValueError("relative boundary eigengap threshold must be positive")

    (
        centroids,
        principal_components,
        assignment_bundle,
        assignments,
        cluster_sizes,
        min_distances,
        manifold_metadata,
        shard_config,
        centroid_config,
    ) = _load_inputs(
        centroids_path=args.centroids_path,
        assignments_path=args.assignments_path,
        shard_dir=args.shard_dir,
        layer=args.layer,
        compute_pca=args.compute_pca,
    )
    K, D = map(int, centroids.shape)
    pca_capacity = (
        args.q_max if principal_components is None
        else int(principal_components.shape[-1])
    )
    if not 1 <= pca_capacity < D:
        raise ValueError("PCA capacity must lie in [1, D - 1]")
    max_intrinsic_dim = max(
        int(manifold["intrinsic_dim"])
        for manifold in manifold_metadata["manifolds"]
    )
    if pca_capacity < max_intrinsic_dim:
        raise ValueError(
            f"PCA capacity {pca_capacity} is below maximum planted "
            f"intrinsic dimension {max_intrinsic_dim}"
        )
    if not 1 <= args.q_max <= pca_capacity:
        raise ValueError(
            f"q_max must lie in [1, stored PCA capacity={pca_capacity}]"
        )

    eligible = cluster_sizes >= args.min_population
    unsupported = eligible & (cluster_sizes <= args.q_max)
    if bool(unsupported.any()):
        ids = unsupported.nonzero(as_tuple=True)[0].tolist()
        raise ValueError(
            f"eligible clusters need at least q_max+1={args.q_max + 1} points; "
            f"unsupported cluster ids: {ids[:10]}"
        )

    device = _resolve_device(args.device)
    leading_eigenvalues, principal_components, pca_validation = (
        _empirical_eigenvalues_and_pca_validation(
            shard_dir=args.shard_dir,
            layer=args.layer,
            centroids=centroids,
            principal_components=principal_components,
            pca_capacity=pca_capacity,
            assignments=assignments,
            cluster_sizes=cluster_sizes,
            eligible=eligible,
            batch_size=args.batch_size,
            chunk_elems=args.chunk_elems,
            eig_batch_size=args.eig_batch_size,
            device=device,
            pca_validation_tolerance=args.pca_validation_tolerance,
            centroid_mean_tolerance=args.centroid_mean_tolerance,
        )
    )
    normalized_gaps, cattell_ranks = _cattell_rank_sweep(
        leading_eigenvalues,
        eligible,
        q_max=args.q_max,
        thresholds=thresholds,
    )

    true_ids_tensor = manifold_metadata["row_manifold_ids"].reshape(-1).long()
    true_ids = true_ids_tensor.numpy()
    predicted_ids = assignments.numpy()
    clustering = {
        "homogeneity": _bounded_value(
            float(homogeneity_score(true_ids, predicted_ids)),
            name="homogeneity",
            lower=0.0,
            upper=1.0,
        ),
        "completeness": _bounded_value(
            float(completeness_score(true_ids, predicted_ids)),
            name="completeness",
            lower=0.0,
            upper=1.0,
        ),
        "adjusted_rand_index": _bounded_value(
            float(adjusted_rand_score(true_ids, predicted_ids)),
            name="adjusted_rand_index",
            lower=-1.0,
            upper=1.0,
        ),
        "normalized_mutual_information": _bounded_value(
            float(normalized_mutual_info_score(true_ids, predicted_ids)),
            name="normalized_mutual_information",
            lower=0.0,
            upper=1.0,
        ),
    }

    inertia = float(min_distances.double().square().sum())
    configured_inertia = float(centroid_config["inertia"])
    relative_inertia_error = abs(inertia - configured_inertia) / max(
        abs(configured_inertia),
        torch.finfo(torch.float64).tiny,
    )
    if relative_inertia_error > args.inertia_validation_tolerance:
        raise ValueError(
            "recomputed assignment inertia differs from the KMeans artifact: "
            f"relative_error={relative_inertia_error:.8g}, "
            f"tolerance={args.inertia_validation_tolerance:.8g}"
        )
    mean_squared_distance = inertia / assignments.numel()
    quantization = {
        "inertia": inertia,
        "configured_kmeans_inertia": configured_inertia,
        "relative_inertia_error": relative_inertia_error,
        "mean_squared_distance": mean_squared_distance,
        "root_mean_squared_distance": math.sqrt(mean_squared_distance),
        "point_to_centroid_distance": _numeric_summary(min_distances),
    }

    associations = _associate_component_means(
        centroids,
        manifold_metadata,
        max_mean_to_manifold_distance=args.max_mean_to_manifold_distance,
    )
    associated = associations.associated
    evaluated = associated & eligible
    intrinsic_dims = torch.tensor(
        [
            int(manifold["intrinsic_dim"])
            for manifold in manifold_metadata["manifolds"]
        ],
        dtype=torch.long,
    )
    target_ranks = torch.full((K,), -1, dtype=torch.long)
    target_ranks[associated] = intrinsic_dims[
        associations.manifold_indices[associated]
    ]

    alignment_overlap, alignment_worst, alignment_defined = _compute_alignment(
        associations=associations,
        intrinsic_dims=intrinsic_dims,
        principal_components=principal_components,
        leading_eigenvalues=leading_eigenvalues,
        population=evaluated,
        ambient_dim=D,
        eigengap_threshold=args.relative_boundary_eigengap_threshold,
    )
    tangent_alignment = _alignment_report(
        definition="leading_intrinsic_dim_empirical_cluster_pca_principal_angles",
        overlap=alignment_overlap,
        worst=alignment_worst,
        defined=alignment_defined,
        population=evaluated,
        eigengap_threshold=args.relative_boundary_eigengap_threshold,
    )
    per_manifold = _per_manifold_base(
        manifold_metadata=manifold_metadata,
        associations=associations,
        cluster_sizes=cluster_sizes,
        eligible=eligible,
        alignment_overlap=alignment_overlap,
        alignment_worst=alignment_worst,
        alignment_defined=alignment_defined,
    )

    containment_overlap = torch.zeros(
        len(thresholds), K, dtype=torch.float64
    )
    containment_worst = torch.zeros(len(thresholds), K, dtype=torch.float64)
    containment_defined = torch.zeros(len(thresholds), K, dtype=torch.bool)
    threshold_sweep = []
    for threshold_index, threshold in enumerate(thresholds):
        ranks = cattell_ranks[threshold_index]
        overlap, worst, defined = _compute_containment(
            associations=associations,
            principal_components=principal_components,
            leading_eigenvalues=leading_eigenvalues,
            ranks=ranks,
            population=evaluated,
            ambient_dim=D,
            eigengap_threshold=args.relative_boundary_eigengap_threshold,
        )
        containment_overlap[threshold_index] = overlap
        containment_worst[threshold_index] = worst
        containment_defined[threshold_index] = defined
        rank_report = {
            "definition": "raw_cattell_scree_rank",
            "population": "proximity_associated_and_population_eligible_components",
            **_rank_summary(ranks, target_ranks, evaluated),
        }
        containment_report = _alignment_report(
            definition="leading_cattell_rank_empirical_cluster_pca_principal_angles",
            overlap=overlap,
            worst=worst,
            defined=defined,
            population=evaluated,
            eigengap_threshold=args.relative_boundary_eigengap_threshold,
        )
        threshold_per_manifold = []
        for manifold_index, manifold in enumerate(manifold_metadata["manifolds"]):
            population = (
                (associations.manifold_indices == manifold_index) & eligible
            )
            threshold_per_manifold.append(
                {
                    "manifold_id": int(manifold["manifold_id"]),
                    "type_id": int(manifold["type_id"]),
                    "type_name": str(manifold["type_name"]),
                    "intrinsic_dim": int(manifold["intrinsic_dim"]),
                    "rank": _rank_summary(ranks, target_ranks, population),
                    "tangent_containment": _alignment_summary(
                        overlap,
                        worst,
                        defined,
                        population,
                    ),
                }
            )
        threshold_sweep.append(
            {
                "cattell_threshold": float(threshold),
                "eligible_cluster_rank_distribution": _rank_distribution(
                    ranks,
                    eligible,
                    args.q_max,
                ),
                "evaluated_cluster_rank_distribution": _rank_distribution(
                    ranks,
                    evaluated,
                    args.q_max,
                ),
                "rank": rank_report,
                "tangent_containment": containment_report,
                "per_manifold": threshold_per_manifold,
            }
        )

    assignment_live = cluster_sizes > 0
    details_path = (
        args.details_path
        if args.details_path is not None
        else args.output_path.parent / "component_metrics.pt"
    )
    metrics = {
        "schema_version": 2,
        "evaluation": "toy_kmeans_initialization_cattell_sweep",
        "partition_kind": "nearest_euclidean_centroid",
        "K": K,
        "q_max": int(args.q_max),
        "pca_capacity": pca_capacity,
        "dataset": {
            "shard_dir": str(args.shard_dir.resolve()),
            "subset_spec": assignment_bundle.get("subset_spec"),
            "layer": int(args.layer),
            "selected_rows": int(assignments.numel()),
            "in_sample": True,
            "in_sample_reason": "KMeans centroids and PCA directions used all rows",
        },
        "artifacts": {
            "centroids_path": str(args.centroids_path.resolve()),
            "assignments_path": str(args.assignments_path.resolve()),
            "component_details_path": str(details_path.resolve()),
            "centroid_artifact_format": "dalg_centroids_v1",
        },
        "eligibility": {
            "rule": "cluster_size_greater_than_or_equal_to_min_population",
            "min_population": int(args.min_population),
            "eligible_components": int(eligible.sum()),
            "excluded_components": int((~eligible).sum()),
            "eligible_points": int(cluster_sizes[eligible].sum()),
            "excluded_points": int(cluster_sizes[~eligible].sum()),
        },
        "clustering": clustering,
        "quantization": quantization,
        "components": {
            "live": int(assignment_live.sum()),
            "dead": int((~assignment_live).sum()),
            "cluster_size": _numeric_summary(cluster_sizes),
        },
        "association": _association_summary(
            associations,
            K,
            args.max_mean_to_manifold_distance,
            eligible,
        ),
        "mean_to_manifold_distance": _numeric_summary(
            associations.nearest_distances
        ),
        "pca_validation": pca_validation,
        "cattell": {
            "definition": "max_j_with_normalized_consecutive_eigengap_above_threshold",
            "normalized_gap": "(lambda_j-lambda_j_plus_1)/lambda_1",
            "comparison": "strict_greater_than",
            "empty_selection_rank": 1,
            "q_max": int(args.q_max),
            "thresholds": list(thresholds),
            "shared_noise_active_set_applied": False,
        },
        "tangent_alignment": tangent_alignment,
        "threshold_sweep": threshold_sweep,
        "per_manifold": per_manifold,
        "omitted_metrics": {
            "nll": "requires a probabilistic density",
            "bic": "requires a likelihood and probabilistic parameter count",
            "gaussian_overlap": "requires component Gaussian covariance geometry",
        },
    }

    evaluated_count = int(evaluated.sum())
    _validate_score_report(tangent_alignment, name="tangent_alignment")
    if (
        tangent_alignment["subspace_overlap"]["valid_components"]
        + tangent_alignment["subspace_overlap"]["undefined_components"]
        != evaluated_count
    ):
        raise RuntimeError("alignment validity counts do not cover evaluated components")
    if sum(item["components"]["associated"] for item in per_manifold) != int(
        associated.sum()
    ):
        raise RuntimeError("per-manifold association counts do not match global count")
    if sum(
        item["components"]["eligible_associated"] for item in per_manifold
    ) != evaluated_count:
        raise RuntimeError("per-manifold eligible counts do not match global count")
    for threshold_item in threshold_sweep:
        _validate_score_report(
            threshold_item["tangent_containment"],
            name=(
                "tangent_containment."
                f"threshold_{threshold_item['cattell_threshold']:g}"
            ),
        )
        score = threshold_item["tangent_containment"]["subspace_overlap"]
        if score["valid_components"] + score["undefined_components"] != evaluated_count:
            raise RuntimeError(
                "containment validity counts do not cover evaluated components"
            )
        if threshold_item["rank"]["components"] != evaluated_count:
            raise RuntimeError("rank summary does not cover evaluated components")
    if int(manifold_metadata["num_manifolds"]) != len(per_manifold):
        raise RuntimeError("per-manifold output does not cover every planted manifold")
    if int(shard_config["num_rows"]) != metrics["dataset"]["selected_rows"]:
        raise RuntimeError("evaluation did not cover every shard row")

    details = {
        "schema_version": 1,
        "evaluation": metrics["evaluation"],
        "K": K,
        "q_max": int(args.q_max),
        "min_population": int(args.min_population),
        "cattell_thresholds": torch.tensor(thresholds, dtype=torch.float64),
        "cluster_sizes": cluster_sizes,
        "eligible": eligible,
        "leading_eigenvalues": leading_eigenvalues[:, : args.q_max + 1],
        "normalized_cattell_gaps": normalized_gaps,
        "cattell_ranks": cattell_ranks,
        "associated": associated,
        "associated_manifold_indices": associations.manifold_indices,
        "nearest_manifold_distances": associations.nearest_distances,
        "target_intrinsic_dims": target_ranks,
        "alignment_overlap": alignment_overlap,
        "alignment_worst_direction_cosine": alignment_worst,
        "alignment_defined": alignment_defined,
        "containment_overlap": containment_overlap,
        "containment_worst_direction_cosine": containment_worst,
        "containment_defined": containment_defined,
    }
    if args.compute_pca:
        details["principal_components"] = principal_components
        details["principal_components_defined"] = eligible
    return metrics, details


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--centroids-path", type=Path, required=True)
    parser.add_argument("--assignments-path", type=Path, required=True)
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--chunk-elems", type=int, default=1 << 23)
    parser.add_argument("--eig-batch-size", type=int, default=64)
    parser.add_argument("--q-max", type=int, default=16)
    parser.add_argument("--min-population", type=int, default=26)
    parser.add_argument(
        "--compute-pca", action="store_true",
        help="Compute missing PCs for eligible clusters and save them in the sidecar",
    )
    parser.add_argument(
        "--cattell-thresholds",
        type=float,
        nargs="+",
        default=list(DEFAULT_CATTELL_THRESHOLDS),
    )
    parser.add_argument("--max-mean-to-manifold-distance", type=float, default=None)
    parser.add_argument(
        "--relative-boundary-eigengap-threshold",
        type=float,
        default=1e-6,
    )
    parser.add_argument("--pca-validation-tolerance", type=float, default=1e-4)
    parser.add_argument("--centroid-mean-tolerance", type=float, default=1e-5)
    parser.add_argument("--inertia-validation-tolerance", type=float, default=1e-5)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--details-path", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _atomic_torch_save(path: Path, value: dict[str, Any]) -> None:
    temporary_path = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(value, temporary_path)
    temporary_path.replace(path)


def _atomic_json_save(path: Path, value: dict[str, Any]) -> None:
    temporary_path = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary_path.write_text(json.dumps(value, indent=2) + "\n")
    temporary_path.replace(path)


def main() -> None:
    args = build_parser().parse_args()
    details_path = (
        args.details_path
        if args.details_path is not None
        else args.output_path.parent / "component_metrics.pt"
    )
    existing = [path for path in (args.output_path, details_path) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite evaluation artifacts: {existing}")

    metrics, details = evaluate_kmeans_geometry(args)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    details_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_torch_save(details_path, details)
    _atomic_json_save(args.output_path, metrics)
    print(f"KMeans Cattell-sweep metrics saved to {args.output_path}")
    print(f"Per-component tensors saved to {details_path}")


if __name__ == "__main__":
    main()
