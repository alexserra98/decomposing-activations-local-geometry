"""Per-component and per-manifold metrics for toy-manifold tilings."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from dalg.evaluation.toy_manifold_geometry import (
    _distance_tied,
    _project_mean_to_manifold,
)


_PC_RELATIVE_EIGENGAP_THRESHOLD = 1e-6
_COVARIANCE_EIGH_BATCH_SIZE = 64


@dataclass(frozen=True)
class _ComponentAssociations:
    manifold_indices: torch.Tensor
    nearest_distances: torch.Tensor
    ambiguous: torch.Tensor
    tangent_bases: tuple[torch.Tensor | None, ...]

    @property
    def associated(self) -> torch.Tensor:
        return self.manifold_indices >= 0

    @property
    def outside_cutoff(self) -> torch.Tensor:
        return (self.manifold_indices < 0) & ~self.ambiguous


@dataclass(frozen=True)
class _ComponentMetrics:
    associations: _ComponentAssociations
    effective_ranks: torch.Tensor
    subspace_overlap: torch.Tensor
    worst_direction_cosine: torch.Tensor
    alignment_defined: torch.Tensor
    containment_overlap: torch.Tensor
    containment_worst_direction_cosine: torch.Tensor
    containment_defined: torch.Tensor


@torch.no_grad()
def _effective_component_ranks(model, threshold: float) -> torch.Tensor:
    """Count loading columns whose variance exceeds the component noise floor."""
    if threshold <= 0.0:
        raise ValueError("rank threshold must be positive")
    scales = model._scale()
    rank_mask = getattr(model, "rank_mask", None)
    if rank_mask is not None:
        scales = scales * rank_mask
    noise_floor = model._psi().mean(dim=1, keepdim=True)
    return (scales.square() > threshold * noise_floor).sum(dim=1).cpu().long()


@torch.no_grad()
def _associate_component_means(
    means: torch.Tensor,
    metadata: dict[str, Any],
    *,
    max_mean_to_manifold_distance: float,
) -> _ComponentAssociations:
    """Associate each mean with its unique nearest manifold inside the cutoff."""
    if not math.isfinite(max_mean_to_manifold_distance) or (
        max_mean_to_manifold_distance <= 0.0
    ):
        raise ValueError("max mean-to-manifold distance must be finite and positive")
    means = means.detach().cpu().double()
    if means.ndim != 2:
        raise ValueError("component means must have shape (K, D)")
    manifolds = metadata["manifolds"]
    if not manifolds:
        raise ValueError("toy-manifold metadata contains no manifold instances")

    manifold_indices = torch.full((len(means),), -1, dtype=torch.long)
    nearest_distances = torch.empty(len(means), dtype=torch.float64)
    ambiguous = torch.zeros(len(means), dtype=torch.bool)
    tangent_bases: list[torch.Tensor | None] = [None] * len(means)

    for component_id, mean in enumerate(means):
        projections = [
            _project_mean_to_manifold(mean, manifold, metadata)
            for manifold in manifolds
        ]
        order = sorted(
            range(len(projections)),
            key=lambda index: projections[index].distance_squared,
        )
        nearest_index = order[0]
        nearest = projections[nearest_index]
        nearest_distance = math.sqrt(max(nearest.distance_squared, 0.0))
        nearest_distances[component_id] = nearest_distance
        tied = [
            index
            for index in order
            if _distance_tied(
                projections[index].distance_squared,
                nearest.distance_squared,
            )
        ]
        if len(tied) != 1:
            ambiguous[component_id] = True
            continue
        if nearest_distance > max_mean_to_manifold_distance:
            continue

        manifold_indices[component_id] = nearest_index
        if nearest.unique:
            tangent_bases[component_id] = nearest.tangent

    return _ComponentAssociations(
        manifold_indices=manifold_indices,
        nearest_distances=nearest_distances,
        ambiguous=ambiguous,
        tangent_bases=tuple(tangent_bases),
    )


@torch.no_grad()
def _leading_covariance_eigenspaces(
    model,
    max_subspace_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return descending leading eigenvalues and eigenvectors of every covariance."""
    if not (1 <= max_subspace_dim <= model.D):
        raise ValueError("maximum subspace dimension must be in [1, D]")

    loadings = model._W().detach().cpu().double()
    psi = model._psi().detach().cpu().double()
    num_eigenvalues = min(max_subspace_dim + 1, model.D)
    eigenvalues = torch.empty(
        (model.K, num_eigenvalues), dtype=torch.float64
    )
    eigenvectors = torch.empty(
        (model.K, model.D, max_subspace_dim), dtype=torch.float64
    )
    for start in range(0, model.K, _COVARIANCE_EIGH_BATCH_SIZE):
        end = min(start + _COVARIANCE_EIGH_BATCH_SIZE, model.K)
        batch_loadings = loadings[start:end]
        covariances = torch.bmm(
            batch_loadings,
            batch_loadings.transpose(1, 2),
        )
        covariances += torch.diag_embed(psi[start:end])
        batch_values, batch_vectors = torch.linalg.eigh(covariances)
        eigenvalues[start:end] = batch_values[:, -num_eigenvalues:].flip(1)
        eigenvectors[start:end] = batch_vectors[:, :, -max_subspace_dim:].flip(2)
    return eigenvalues, eigenvectors


def _subspace_alignment(
    tangent_basis: torch.Tensor,
    principal_basis: torch.Tensor,
) -> tuple[float, float]:
    """Measure coverage of a tangent by an arbitrary-dimensional PC subspace."""
    tangent_basis = tangent_basis.detach().cpu().double()
    principal_basis = principal_basis.detach().cpu().double()
    if tangent_basis.ndim != 2 or principal_basis.ndim != 2:
        raise ValueError("subspace bases must be matrices")
    if tangent_basis.shape[0] != principal_basis.shape[0]:
        raise ValueError("subspace bases must have the same ambient dimension")
    if tangent_basis.shape[1] == 0:
        raise ValueError("tangent basis must contain at least one direction")
    if principal_basis.shape[1] == 0:
        return 0.0, 0.0

    cosines = torch.linalg.svdvals(tangent_basis.T @ principal_basis).clamp(0.0, 1.0)
    if not torch.isfinite(cosines).all():
        raise ValueError("subspace alignment produced non-finite principal angles")
    overlap = float(cosines.square().sum() / tangent_basis.shape[1])
    worst = (
        0.0
        if principal_basis.shape[1] < tangent_basis.shape[1]
        else float(cosines.min())
    )
    return overlap, worst


def _leading_subspace_is_identifiable(
    eigenvalues: torch.Tensor,
    subspace_dim: int,
    ambient_dim: int,
    relative_boundary_eigengap_threshold: float,
) -> bool:
    """Return whether the leading subspace has a unique covariance boundary."""
    if not (1 <= subspace_dim <= ambient_dim):
        raise ValueError("subspace dimension must be in [1, D]")
    if subspace_dim == ambient_dim:
        return True
    retained = eigenvalues[subspace_dim - 1]
    excluded = eigenvalues[subspace_dim]
    relative_gap = (retained - excluded) / retained.abs().clamp_min(
        torch.finfo(torch.float64).tiny
    )
    return float(relative_gap) > relative_boundary_eigengap_threshold


@torch.no_grad()
def _component_metrics(
    model,
    metadata: dict[str, Any],
    *,
    rank_threshold: float,
    max_mean_to_manifold_distance: float,
    relative_boundary_eigengap_threshold: float,
) -> _ComponentMetrics:
    if relative_boundary_eigengap_threshold <= 0.0:
        raise ValueError("relative boundary eigengap threshold must be positive")
    associations = _associate_component_means(
        model.mu,
        metadata,
        max_mean_to_manifold_distance=max_mean_to_manifold_distance,
    )
    intrinsic_dims = torch.tensor(
        [int(manifold["intrinsic_dim"]) for manifold in metadata["manifolds"]],
        dtype=torch.long,
    )
    if torch.any(intrinsic_dims <= 0):
        raise ValueError("manifold intrinsic dimensions must be positive")
    max_intrinsic_dim = int(intrinsic_dims.max())
    effective_ranks = _effective_component_ranks(model, rank_threshold)
    max_subspace_dim = max(max_intrinsic_dim, int(effective_ranks.max()))
    eigenvalues, eigenvectors = _leading_covariance_eigenspaces(
        model,
        max_subspace_dim,
    )

    overlap = torch.zeros(model.K, dtype=torch.float64)
    worst = torch.zeros(model.K, dtype=torch.float64)
    alignment_defined = torch.zeros(model.K, dtype=torch.bool)
    containment_overlap = torch.zeros(model.K, dtype=torch.float64)
    containment_worst = torch.zeros(model.K, dtype=torch.float64)
    containment_defined = torch.zeros(model.K, dtype=torch.bool)
    for component_id in range(model.K):
        manifold_index = int(associations.manifold_indices[component_id])
        if manifold_index < 0:
            continue
        tangent = associations.tangent_bases[component_id]
        if tangent is None:
            continue
        intrinsic_dim = int(intrinsic_dims[manifold_index])
        effective_rank = int(effective_ranks[component_id])
        if effective_rank == 0:
            alignment_defined[component_id] = True
            containment_defined[component_id] = True
            continue

        component_eigenvalues = eigenvalues[component_id]
        if _leading_subspace_is_identifiable(
            component_eigenvalues,
            intrinsic_dim,
            model.D,
            relative_boundary_eigengap_threshold,
        ):
            principal = eigenvectors[component_id, :, :intrinsic_dim]
            component_overlap, component_worst = _subspace_alignment(
                tangent, principal
            )
            overlap[component_id] = component_overlap
            worst[component_id] = component_worst
            alignment_defined[component_id] = True

        if _leading_subspace_is_identifiable(
            component_eigenvalues,
            effective_rank,
            model.D,
            relative_boundary_eigengap_threshold,
        ):
            principal = eigenvectors[component_id, :, :effective_rank]
            component_overlap, component_worst = _subspace_alignment(
                tangent, principal
            )
            containment_overlap[component_id] = component_overlap
            containment_worst[component_id] = component_worst
            containment_defined[component_id] = True

    return _ComponentMetrics(
        associations=associations,
        effective_ranks=effective_ranks,
        subspace_overlap=overlap,
        worst_direction_cosine=worst,
        alignment_defined=alignment_defined,
        containment_overlap=containment_overlap,
        containment_worst_direction_cosine=containment_worst,
        containment_defined=containment_defined,
    )


def _score_summary(
    scores: torch.Tensor,
    valid: torch.Tensor,
    population: torch.Tensor,
) -> dict[str, float | int | None]:
    selected = valid & population
    valid_components = int(selected.sum())
    population_components = int(population.sum())
    return {
        "mean": float(scores[selected].mean()) if valid_components else None,
        "valid_components": valid_components,
        "undefined_components": population_components - valid_components,
    }


def _rank_summary(
    effective_ranks: torch.Tensor,
    target_ranks: torch.Tensor,
    population: torch.Tensor,
) -> dict[str, float | int | None]:
    count = int(population.sum())
    if count == 0:
        return {
            "components": 0,
            "mean_learned": None,
            "exact_match": None,
            "within_one_match": None,
            "mean_absolute_error": None,
        }
    learned = effective_ranks[population]
    targets = target_ranks[population]
    error = learned - targets
    return {
        "components": count,
        "mean_learned": float(learned.float().mean()),
        "exact_match": float((error == 0).float().mean()),
        "within_one_match": float((error.abs() <= 1).float().mean()),
        "mean_absolute_error": float(error.abs().float().mean()),
    }


def _alignment_summary(
    overlap: torch.Tensor,
    worst_direction_cosine: torch.Tensor,
    defined: torch.Tensor,
    population: torch.Tensor,
) -> dict[str, dict[str, float | int | None]]:
    return {
        "subspace_overlap": _score_summary(
            overlap,
            defined,
            population,
        ),
        "worst_direction_cosine": _score_summary(
            worst_direction_cosine,
            defined,
            population,
        ),
    }


@torch.no_grad()
def evaluate_toy_manifold_metrics(
    model,
    metadata: dict[str, Any],
    assignment_live: torch.Tensor,
    *,
    rank_threshold: float = 1.0,
    max_mean_to_manifold_distance: float = 0.1,
    relative_boundary_eigengap_threshold: float = _PC_RELATIVE_EIGENGAP_THRESHOLD,
) -> dict[str, Any]:
    """Evaluate proximity association, rank, alignment, and containment."""
    assignment_live = assignment_live.detach().cpu().bool().reshape(-1)
    if assignment_live.numel() != model.K:
        raise ValueError("assignment-live mask does not match model K")
    manifolds = metadata["manifolds"]
    if int(metadata["num_manifolds"]) != len(manifolds):
        raise ValueError("num_manifolds does not match manifold metadata")

    metrics = _component_metrics(
        model,
        metadata,
        rank_threshold=rank_threshold,
        max_mean_to_manifold_distance=max_mean_to_manifold_distance,
        relative_boundary_eigengap_threshold=relative_boundary_eigengap_threshold,
    )
    associated = metrics.associations.associated
    intrinsic_dims = torch.tensor(
        [int(manifold["intrinsic_dim"]) for manifold in manifolds],
        dtype=torch.long,
    )
    target_ranks = torch.full((model.K,), -1, dtype=torch.long)
    target_ranks[associated] = intrinsic_dims[
        metrics.associations.manifold_indices[associated]
    ]

    association = {
        "rule": "unique_nearest_exact_projection_within_cutoff",
        "max_mean_to_manifold_distance": float(max_mean_to_manifold_distance),
        "associated_components": int(associated.sum()),
        "outside_cutoff_components": int(metrics.associations.outside_cutoff.sum()),
        "ambiguous_components": int(metrics.associations.ambiguous.sum()),
    }
    if sum(
        association[key]
        for key in (
            "associated_components",
            "outside_cutoff_components",
            "ambiguous_components",
        )
    ) != model.K:
        raise RuntimeError("component association populations do not sum to K")

    global_rank = {
        "threshold": float(rank_threshold),
        "population": "proximity_associated_components",
        **_rank_summary(metrics.effective_ranks, target_ranks, associated),
    }
    tangent_alignment = {
        "definition": "leading_intrinsic_dim_covariance_subspace_principal_angles",
        "aggregation": "unweighted_component_mean",
        "relative_boundary_eigengap_threshold": float(
            relative_boundary_eigengap_threshold
        ),
        **_alignment_summary(
            metrics.subspace_overlap,
            metrics.worst_direction_cosine,
            metrics.alignment_defined,
            associated,
        ),
    }
    tangent_containment = {
        "definition": "leading_effective_rank_covariance_subspace_principal_angles",
        "aggregation": "unweighted_component_mean",
        "relative_boundary_eigengap_threshold": float(
            relative_boundary_eigengap_threshold
        ),
        **_alignment_summary(
            metrics.containment_overlap,
            metrics.containment_worst_direction_cosine,
            metrics.containment_defined,
            associated,
        ),
    }

    per_manifold = []
    for manifold_index, manifold in enumerate(manifolds):
        population = metrics.associations.manifold_indices == manifold_index
        associated_components = int(population.sum())
        manifold_rank = {
            "target_intrinsic_dim": int(manifold["intrinsic_dim"]),
            **_rank_summary(metrics.effective_ranks, target_ranks, population),
        }
        per_manifold.append(
            {
                "manifold_id": int(manifold["manifold_id"]),
                "type_id": int(manifold["type_id"]),
                "type_name": str(manifold["type_name"]),
                "intrinsic_dim": int(manifold["intrinsic_dim"]),
                "components": {
                    "associated": associated_components,
                    "assignment_live": int((population & assignment_live).sum()),
                    "assignment_dead": int((population & ~assignment_live).sum()),
                },
                "rank": manifold_rank,
                "tangent_alignment": _alignment_summary(
                    metrics.subspace_overlap,
                    metrics.worst_direction_cosine,
                    metrics.alignment_defined,
                    population,
                ),
                "tangent_containment": _alignment_summary(
                    metrics.containment_overlap,
                    metrics.containment_worst_direction_cosine,
                    metrics.containment_defined,
                    population,
                ),
            }
        )

    return {
        "association": association,
        "rank": global_rank,
        "tangent_alignment": tangent_alignment,
        "tangent_containment": tangent_containment,
        "per_manifold": per_manifold,
    }


__all__ = ["evaluate_toy_manifold_metrics"]
