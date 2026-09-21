"""Nearest-neighbor selection and PCA around fixed centroids."""

import torch


@torch.no_grad()
def nearest_neighbor_indices(
    points: torch.Tensor,
    centroids: torch.Tensor,
    *,
    neighbors: int,
    device: torch.device,
    point_block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return nearest point indexes and squared distances for every centroid."""
    if points.ndim != 2 or centroids.ndim != 2:
        raise ValueError("points and centroids must both be rank-2 tensors")
    if points.shape[1] != centroids.shape[1]:
        raise ValueError("points and centroids must have the same dimension")
    if not 1 <= neighbors <= points.shape[0]:
        raise ValueError(
            f"neighbors must be in [1, {points.shape[0]}], got {neighbors}"
        )
    if point_block_size <= 0:
        raise ValueError("point_block_size must be positive")

    points_device = points.to(device=device, dtype=torch.float32)
    centers = centroids.to(device=device, dtype=torch.float32)
    K = centers.shape[0]
    center_norms = (centers * centers).sum(dim=1)[None, :]
    best_distances = torch.full(
        (K, neighbors),
        float("inf"),
        dtype=torch.float32,
        device=device,
    )
    best_indices = torch.full(
        (K, neighbors),
        -1,
        dtype=torch.long,
        device=device,
    )

    for start in range(0, points_device.shape[0], point_block_size):
        stop = min(start + point_block_size, points_device.shape[0])
        block = points_device[start:stop]
        point_norms = (block * block).sum(dim=1)[:, None]
        distances = point_norms + center_norms - 2.0 * (block @ centers.T)
        distances.clamp_min_(0.0)

        candidates = torch.cat((best_distances, distances.T), dim=1)
        best_distances, positions = torch.topk(
            candidates,
            k=neighbors,
            dim=1,
            largest=False,
            sorted=True,
        )
        old_positions = positions.clamp_max(neighbors - 1)
        old_indices = best_indices.gather(1, old_positions)
        new_indices = start + positions - neighbors
        best_indices = torch.where(
            positions < neighbors,
            old_indices,
            new_indices,
        )

    if torch.any(best_indices < 0):
        raise RuntimeError("nearest-neighbor search left unfilled indices")
    sorted_indices = best_indices.sort(dim=1).values
    if torch.any(sorted_indices[:, 1:] == sorted_indices[:, :-1]):
        raise RuntimeError("nearest-neighbor search produced duplicate points")
    return best_indices.cpu(), best_distances.cpu()


@torch.no_grad()
def compute_neighborhood_pca(
    points: torch.Tensor,
    centroids: torch.Tensor,
    neighbor_indices: torch.Tensor,
    *,
    rank: int,
    device: torch.device,
    eig_batch_size: int,
) -> torch.Tensor:
    """Compute PCA around each fixed centroid using its selected neighborhood."""
    if neighbor_indices.ndim != 2:
        raise ValueError("neighbor_indices must have shape (K, neighbors)")
    K, D = map(int, centroids.shape)
    if neighbor_indices.shape[0] != K:
        raise ValueError("neighbor_indices must have one row per centroid")
    neighbors = int(neighbor_indices.shape[1])
    if not 1 <= rank < neighbors or rank > D:
        raise ValueError(
            f"rank must be in [1, min(D={D}, neighbors-1={neighbors - 1})], got {rank}"
        )
    if eig_batch_size <= 0:
        raise ValueError("eig_batch_size must be positive")

    points_device = points.to(device=device, dtype=torch.float32)
    centers = centroids.to(device=device, dtype=torch.float64)
    indices = neighbor_indices.to(device=device, dtype=torch.long)
    directions = torch.empty(K, D, rank, dtype=torch.float32)

    for start in range(0, K, eig_batch_size):
        stop = min(start + eig_batch_size, K)
        local_points = points_device[indices[start:stop]].to(torch.float64)
        residuals = local_points - centers[start:stop, None, :]
        covariance = residuals.transpose(1, 2) @ residuals
        covariance /= neighbors
        covariance = 0.5 * (covariance + covariance.transpose(-1, -2))
        _eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        directions[start:stop] = (
            eigenvectors[:, :, -rank:].flip(-1).to(torch.float32).cpu()
        )

    return directions

