"""Centroid artifacts with optional per-cluster PCA directions."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch


CENTROID_ARTIFACT_FORMAT = "dalg_centroids_v1"


def unpack_centroid_artifact(value: Any) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Return ``(centroids, principal_components)`` from old or new artifacts."""
    if isinstance(value, torch.Tensor):
        return value, None
    if not isinstance(value, dict) or "centroids" not in value:
        raise ValueError(
            "centroid artifact must be a tensor or a mapping containing 'centroids'"
        )
    centroids = value["centroids"]
    principal_components = value.get("principal_components")
    if not isinstance(centroids, torch.Tensor):
        raise ValueError("centroid artifact field 'centroids' must be a tensor")
    if principal_components is not None and not isinstance(
        principal_components, torch.Tensor
    ):
        raise ValueError(
            "centroid artifact field 'principal_components' must be a tensor"
        )
    return centroids, principal_components


def load_centroid_artifact(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    mmap: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    value = torch.load(
        path,
        map_location=map_location,
        mmap=mmap,
        weights_only=True,
    )
    return unpack_centroid_artifact(value)


def validate_centroid_artifact(
    centroids: torch.Tensor,
    principal_components: torch.Tensor | None,
    *,
    expected_k: int | None = None,
    expected_d: int | None = None,
    required_pca_rank: int | None = None,
) -> None:
    """Validate shapes needed by centroid and optional cluster-PCA initialization."""
    if centroids.ndim != 2:
        raise ValueError(
            f"centroids must have shape (K, D), got {tuple(centroids.shape)}"
        )
    K, D = map(int, centroids.shape)
    if expected_k is not None and K != expected_k:
        raise ValueError(f"centroids K={K} does not match expected K={expected_k}")
    if expected_d is not None and D != expected_d:
        raise ValueError(
            f"centroid dimension D={D} does not match expected D={expected_d}"
        )

    if principal_components is not None:
        if principal_components.ndim != 3:
            raise ValueError(
                "principal_components must have shape (K, D, Q), got "
                f"{tuple(principal_components.shape)}"
            )
        if tuple(principal_components.shape[:2]) != (K, D):
            raise ValueError(
                "principal_components leading dimensions "
                f"{tuple(principal_components.shape[:2])} do not match "
                f"centroids {(K, D)}"
            )

    if required_pca_rank is not None:
        if principal_components is None:
            raise ValueError(
                "direction_init=cluster_pca requires a centroid artifact containing "
                "'principal_components'"
            )
        stored_rank = int(principal_components.shape[-1])
        if stored_rank < required_pca_rank:
            raise ValueError(
                f"centroid artifact stores {stored_rank} principal components per "
                f"cluster, but rank/q_max={required_pca_rank} was requested"
            )


def save_centroid_artifact(
    path: str | Path,
    centroids: torch.Tensor,
    principal_components: torch.Tensor,
) -> None:
    """Atomically save an enriched centroid bundle on CPU."""
    path = Path(path)
    payload = {
        "format": CENTROID_ARTIFACT_FORMAT,
        "centroids": centroids.detach().cpu(),
        "principal_components": principal_components.detach().cpu(),
    }
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(payload, tmp)
    tmp.replace(path)


@torch.no_grad()
def compute_cluster_pca_directions(
    points: torch.Tensor,
    assignments: torch.Tensor,
    centroids: torch.Tensor,
    *,
    rank: int,
    chunk_elems: int = 1 << 23,
    eig_batch_size: int = 256,
) -> torch.Tensor:
    """Compute exact top PCA directions around fixed hard-cluster centroids.

    The explicit ``(K, D, D)`` scatter is intended for the D=128 toy workflow.
    Only the returned ``(K, D, rank)`` eigenvectors are retained by callers.
    """
    if points.ndim != 2 or centroids.ndim != 2:
        raise ValueError("points and centroids must both be rank-2 tensors")
    if points.shape[1] != centroids.shape[1]:
        raise ValueError(
            f"point dimension {points.shape[1]} does not match centroid dimension "
            f"{centroids.shape[1]}"
        )
    if assignments.ndim != 1 or assignments.numel() != points.shape[0]:
        raise ValueError("assignments must have one entry per point")

    K, D = map(int, centroids.shape)
    if not 1 <= rank <= D:
        raise ValueError(f"rank must be in [1, {D}], got {rank}")
    if eig_batch_size <= 0:
        raise ValueError("eig_batch_size must be positive")

    device = points.device
    labels = assignments.to(device=device, dtype=torch.long)
    if labels.numel() and (int(labels.min()) < 0 or int(labels.max()) >= K):
        raise ValueError(f"assignments must lie in [0, {K - 1}]")
    counts = torch.bincount(labels, minlength=K)
    undersized = (counts <= rank).nonzero(as_tuple=True)[0]
    if undersized.numel():
        preview = undersized[:10].tolist()
        raise ValueError(
            f"every cluster needs at least rank+1={rank + 1} points; "
            f"undersized cluster ids (first 10): {preview}"
        )

    centers = centroids.to(device=device, dtype=torch.float64)
    scatter = torch.zeros(K, D, D, dtype=torch.float64, device=device)
    rows_per_chunk = max(1, int(chunk_elems) // max(1, D * D))
    for start in range(0, points.shape[0], rows_per_chunk):
        stop = min(start + rows_per_chunk, points.shape[0])
        cluster_ids = labels[start:stop]
        residual = points[start:stop].to(torch.float64) - centers[cluster_ids]
        outer = residual[:, :, None] * residual[:, None, :]
        scatter.index_add_(0, cluster_ids, outer)

    directions = torch.empty(
        K,
        D,
        rank,
        dtype=centroids.dtype,
        device=device,
    )
    for start in range(0, K, eig_batch_size):
        stop = min(start + eig_batch_size, K)
        covariance = scatter[start:stop] / counts[start:stop, None, None]
        covariance = 0.5 * (covariance + covariance.transpose(-1, -2))
        _eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        directions[start:stop] = eigenvectors[:, :, -rank:].flip(-1).to(
            centroids.dtype
        )
    return directions


__all__ = [
    "CENTROID_ARTIFACT_FORMAT",
    "compute_cluster_pca_directions",
    "load_centroid_artifact",
    "save_centroid_artifact",
    "unpack_centroid_artifact",
    "validate_centroid_artifact",
]
