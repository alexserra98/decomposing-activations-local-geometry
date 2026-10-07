"""Held-out point-to-centroid coverage metrics."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from typing import Any

import torch


MAX_EMPIRICAL_COVERAGE_POINTS = 100_000


def check_empirical_coverage_size(count: int) -> None:
    """Keep the exact curve bounded until a larger-scale implementation exists."""
    if count > MAX_EMPIRICAL_COVERAGE_POINTS:
        raise ValueError(
            f"heldout_distribution_coverage has {count:,} test points; the full "
            f"empirical curve limit is {MAX_EMPIRICAL_COVERAGE_POINTS:,}. "
            "Implement scalable coverage before evaluating a larger test population."
        )


def stratified_three_way_split(
    labels: torch.Tensor,
    *,
    train_fraction: float,
    validation_fraction: float,
    seed: int,
) -> dict[str, torch.Tensor]:
    """Return deterministic, disjoint train/validation/test row positions."""
    labels = labels.detach().cpu().long().reshape(-1)
    if labels.numel() == 0:
        raise ValueError("cannot split an empty dataset")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must lie in (0, 1)")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must lie in (0, 1)")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("train and validation fractions must leave a test split")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    splits: dict[str, list[torch.Tensor]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    for label in labels.unique(sorted=True):
        members = (labels == label).nonzero(as_tuple=True)[0]
        members = members[torch.randperm(members.numel(), generator=generator)]
        n_train = math.floor(members.numel() * train_fraction)
        n_validation = math.floor(members.numel() * validation_fraction)
        if min(n_train, n_validation, members.numel() - n_train - n_validation) <= 0:
            raise ValueError("every label needs at least one train, validation, and test row")
        splits["train"].append(members[:n_train])
        splits["validation"].append(members[n_train : n_train + n_validation])
        splits["test"].append(members[n_train + n_validation :])

    return {name: torch.cat(parts).sort().values for name, parts in splits.items()}


def split_fingerprint(split: dict[str, torch.Tensor]) -> str:
    """Hash the ordered row positions of a three-way split."""
    digest = hashlib.sha256()
    for name in ("train", "validation", "test"):
        digest.update(name.encode("utf-8"))
        digest.update(split[name].detach().cpu().long().numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def nearest_live_centroid_distances(
    reference_points: torch.Tensor,
    centroids: torch.Tensor,
    *,
    live_components: torch.Tensor | None = None,
    batch_size: int = 8192,
) -> torch.Tensor:
    """Return each reference point's Euclidean distance to its nearest live mean."""
    if reference_points.ndim != 2 or centroids.ndim != 2:
        raise ValueError("reference_points and centroids must both be rank-2 tensors")
    if reference_points.shape[1] != centroids.shape[1]:
        raise ValueError("reference points and centroids must share an ambient dimension")
    if reference_points.shape[0] == 0:
        raise ValueError("coverage requires at least one reference point")
    if centroids.shape[0] == 0:
        raise ValueError("coverage requires at least one centroid")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not torch.isfinite(reference_points).all() or not torch.isfinite(centroids).all():
        raise ValueError("coverage inputs must be finite")

    if live_components is None:
        live = torch.ones(centroids.shape[0], dtype=torch.bool, device=centroids.device)
    else:
        live = live_components.detach().to(device=centroids.device, dtype=torch.bool).reshape(-1)
        if live.numel() != centroids.shape[0]:
            raise ValueError("live_components must have one entry per centroid")
    if not bool(live.any()):
        raise ValueError("coverage requires at least one live centroid")

    centers = centroids[live].detach().float()
    points = reference_points.detach()
    device = centers.device
    chunks: list[torch.Tensor] = []
    for start in range(0, points.shape[0], batch_size):
        batch = points[start : start + batch_size].to(device=device, dtype=torch.float32)
        # The matrix-multiplication identity for Euclidean distance subtracts
        # similarly sized squared norms. It can lose the small residual for
        # nearby points with a shared ambient offset, especially under TF32.
        distances = torch.cdist(
            batch,
            centers,
            compute_mode="donot_use_mm_for_euclid_dist",
        )
        chunks.append(distances.min(dim=1).values.cpu())
    return torch.cat(chunks)


def summarize_coverage_distances(
    distances: torch.Tensor,
    *,
    thresholds: Sequence[float] | None = (),
) -> dict[str, Any]:
    """Summarize distances; ``thresholds=None`` returns the full empirical CDF."""
    if thresholds is None:
        check_empirical_coverage_size(distances.numel())
    values = distances.detach().cpu().double().reshape(-1)
    if values.numel() == 0:
        raise ValueError("coverage requires at least one distance")
    if not torch.isfinite(values).all() or bool((values < 0).any()):
        raise ValueError("coverage distances must be finite and non-negative")

    resolved_thresholds = [] if thresholds is None else [float(value) for value in thresholds]
    if any(not math.isfinite(value) or value < 0.0 for value in resolved_thresholds):
        raise ValueError("coverage thresholds must be finite and non-negative")
    if resolved_thresholds != sorted(set(resolved_thresholds)):
        raise ValueError("coverage thresholds must be sorted and unique")

    median = torch.quantile(values, 0.5)
    tail_levels = torch.tensor((0.9, 0.95, 0.99), dtype=torch.float64)
    ordered = values.sort().values
    tail_indices = (tail_levels * values.numel()).ceil().long().sub(1).clamp_min(0)
    tail_quantiles = ordered[tail_indices]
    if thresholds is None:
        radii, counts = torch.unique_consecutive(ordered, return_counts=True)
        fractions = counts.cumsum(0).double() / values.numel()
        curve = [
            {"radius": radius, "fraction": fraction}
            for radius, fraction in zip(radii.tolist(), fractions.tolist())
        ]
    else:
        curve = [
            {"radius": radius, "fraction": float((values <= radius).double().mean())}
            for radius in resolved_thresholds
        ]
    return {
        "definition": "heldout_point_to_nearest_live_centroid_euclidean_distance",
        "reference_points": int(values.numel()),
        "mean": float(values.mean()),
        "root_mean_square": float(values.square().mean().sqrt()),
        "median": float(median),
        "r90": float(tail_quantiles[0]),
        "r95": float(tail_quantiles[1]),
        "r99": float(tail_quantiles[2]),
        "maximum": float(values.max()),
        "coverage_curve": curve,
        "convention": "lower_distances_and_higher_coverage_fractions_are_better",
    }


@torch.no_grad()
def evaluate_heldout_distribution_coverage(
    reference_points: torch.Tensor,
    centroids: torch.Tensor,
    *,
    live_components: torch.Tensor | None = None,
    thresholds: Sequence[float] | None = (),
    batch_size: int = 8192,
) -> tuple[dict[str, Any], torch.Tensor]:
    """Evaluate held-out distribution coverage and return its pointwise distances."""
    if thresholds is None:
        check_empirical_coverage_size(reference_points.shape[0])
    distances = nearest_live_centroid_distances(
        reference_points,
        centroids,
        live_components=live_components,
        batch_size=batch_size,
    )
    summary = summarize_coverage_distances(distances, thresholds=thresholds)
    live_count = (
        centroids.shape[0]
        if live_components is None
        else int(live_components.detach().bool().sum())
    )
    summary.update(
        {
            "components_total": int(centroids.shape[0]),
            "components_live": int(live_count),
            "population": "independent_test_split",
        }
    )
    return summary, distances


__all__ = [
    "evaluate_heldout_distribution_coverage",
    "nearest_live_centroid_distances",
    "split_fingerprint",
    "stratified_three_way_split",
    "summarize_coverage_distances",
]
