"""Component ranks, covariance PCs, and comparisons between subspaces."""

from __future__ import annotations

import warnings

import torch

from dalg.models.kmeans import KMeans

_COVARIANCE_EIGH_BATCH_SIZE = 64
PC_RELATIVE_EIGENGAP_THRESHOLD = 1e-6


@torch.no_grad()
def effective_component_ranks(model, threshold: float) -> torch.Tensor:
    """Use saved KMeans/HDDC ranks, or threshold loadings for MFA and ARD."""
    if isinstance(model, KMeans):
        return model.component_ranks.detach().cpu().long()
    rank_mask = getattr(model, "rank_mask", None)
    if rank_mask is not None:
        return rank_mask.sum(dim=1).detach().cpu().long()
    if threshold <= 0.0:
        raise ValueError("rank threshold must be positive")
    scales = model._scale()
    noise_floor = model._psi().mean(dim=1, keepdim=True)
    return (scales.square() > threshold * noise_floor).sum(dim=1).cpu().long()


@torch.no_grad()
def leading_covariance_eigenspaces(
    model,
    max_subspace_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return descending eigenvalues and PCs on CPU in float64.

    Return one extra eigenvalue, when available, for the subspace boundary.
    KMeans retains its saved zero padding and PCA validity conventions.
    """
    if not (1 <= max_subspace_dim <= model.D):
        raise ValueError("maximum subspace dimension must be in [1, D]")
    if isinstance(model, KMeans):
        if max_subspace_dim > model.q:
            raise ValueError(
                f"KMeans stored PC capacity {model.q} cannot support the requested "
                f"subspace; at least {max_subspace_dim} PCs are required"
            )
        return (
            model.eigenvalues[:, : min(max_subspace_dim + 1, model.D)].detach().cpu().double(),
            model.W[:, :, :max_subspace_dim].detach().cpu().double(),
        )

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
        # W W^T
        covariances = torch.bmm(
            batch_loadings,
            batch_loadings.transpose(1, 2),
        )
        # diag(psi)
        covariances += torch.diag_embed(psi[start:end])
        # compute the leading eigenvalues and eigenvectors of each covariance matrix
        batch_values, batch_vectors = torch.linalg.eigh(covariances)
        eigenvalues[start:end] = batch_values[:, -num_eigenvalues:].flip(1)
        eigenvectors[start:end] = batch_vectors[:, :, -max_subspace_dim:].flip(2)
    return eigenvalues, eigenvectors


def warn_tied_eigenvalues(
    eigenvalues: torch.Tensor,
    ranks: torch.Tensor,
    eligible: torch.Tensor,
    *,
    relative_tolerance: float = PC_RELATIVE_EIGENGAP_THRESHOLD,
) -> None:
    """Warn about adjacent ties touching retained PCs, including their boundary."""
    affected = []
    for component in torch.where(eligible)[0].tolist():
        rank = int(ranks[component])
        values = eigenvalues[component, : rank + 1].detach().cpu().double()
        if rank <= 0 or values.numel() < 2:
            continue
        gaps = (values[:-1] - values[1:]) / values[:-1].abs().clamp_min(
            torch.finfo(torch.float64).tiny
        )
        if bool((gaps <= relative_tolerance).any()):
            affected.append(component)
    if affected:
        warnings.warn(
            f"Tied or nearly tied eigenvalues (relative eigengap <= {relative_tolerance:g}) "
            f"touch retained PCs in components {affected}. Individual PCs and PC-subset "
            "alignment scores may be unstable; scores are computed without correction.",
            RuntimeWarning,
            stacklevel=2,
        )


def subspace_alignment(
    reference_basis: torch.Tensor,
    principal_basis: torch.Tensor,
) -> tuple[float, float]:
    """Compare orthonormal bases, normalizing overlap by the first one's dimension."""
    reference_basis = reference_basis.detach().cpu().double()
    principal_basis = principal_basis.detach().cpu().double()
    if reference_basis.ndim != 2 or principal_basis.ndim != 2:
        raise ValueError("subspace bases must be matrices")
    if reference_basis.shape[0] != principal_basis.shape[0]:
        raise ValueError("subspace bases must have the same ambient dimension")
    if reference_basis.shape[1] == 0:
        raise ValueError("reference basis must contain at least one direction")
    if principal_basis.shape[1] == 0:
        return 0.0, 0.0

    cosines = torch.linalg.svdvals(reference_basis.T @ principal_basis).clamp(0.0, 1.0)
    if not torch.isfinite(cosines).all():
        raise ValueError("subspace alignment produced non-finite principal angles")
    overlap = float(cosines.square().sum() / reference_basis.shape[1])
    worst = (
        0.0
        if principal_basis.shape[1] < reference_basis.shape[1]
        else float(cosines.min())
    )
    return overlap, worst


def best_subset_alignment(
    reference_basis: torch.Tensor,
    principal_basis: torch.Tensor,
) -> tuple[float, float]:
    """Choose reference_dim PCs maximizing overlap with the reference subspace."""
    reference_basis = reference_basis.detach().cpu().double()
    principal_basis = principal_basis.detach().cpu().double()
    reference_dim = reference_basis.shape[1]
    if not 1 <= reference_dim <= principal_basis.shape[1]:
        raise ValueError("best-subset alignment requires at least reference_dim PCs")
    contributions = (reference_basis.T @ principal_basis).square().sum(dim=0)
    if not torch.isfinite(contributions).all():
        raise ValueError("PC reference contributions must be finite")
    # Stable sorting favors earlier PCs when reference contributions are equal.
    selected = torch.argsort(contributions, descending=True, stable=True)[:reference_dim]
    return subspace_alignment(reference_basis, principal_basis[:, selected])


def leading_subspace_is_identifiable(
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


__all__ = [
    "PC_RELATIVE_EIGENGAP_THRESHOLD",
    "warn_tied_eigenvalues",
    "effective_component_ranks",
    "leading_covariance_eigenspaces",
    "subspace_alignment",
    "best_subset_alignment",
    "leading_subspace_is_identifiable",
]
