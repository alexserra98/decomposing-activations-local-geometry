"""Exact neighborhood selection and covariance about stored centroids."""

import pytest
import torch

from dalg.init.neighborhood_pca import nearest_neighbor_indices, compute_neighborhood_pca


@pytest.mark.parametrize("block_size", [3, 17, 100])
def test_neighbors_and_pca_match_dense_reference(block_size):
    generator = torch.Generator().manual_seed(17)
    points = torch.randn(80, 5, generator=generator)
    centers = torch.randn(4, 5, generator=generator) + 2
    indices, distances = nearest_neighbor_indices(
        points, centers, neighbors=64, device=torch.device("cpu"),
        point_block_size=block_size,
    )
    expected_distances = (centers[:, None, :] - points[None, :, :]).square().sum(-1)
    expected_distances, expected_indices = expected_distances.topk(64, largest=False)
    assert torch.equal(indices, expected_indices)
    torch.testing.assert_close(distances, expected_distances)
    pcs = compute_neighborhood_pca(
        points, centers, indices, rank=3, device=torch.device("cpu"), eig_batch_size=2,
    )
    for k in range(len(centers)):
        residual = points[expected_indices[k]].double() - centers[k].double()
        _, vectors = torch.linalg.eigh(residual.T @ residual / 64)
        expected = vectors[:, -3:].flip(-1).float()
        torch.testing.assert_close(pcs[k] @ pcs[k].T, expected @ expected.T)


def test_neighborhood_pca_rejects_unsupported_rank():
    with pytest.raises(ValueError, match="rank must be"):
        compute_neighborhood_pca(
            torch.zeros(10, 2), torch.zeros(1, 2), torch.arange(8)[None],
            rank=3, device=torch.device("cpu"), eig_batch_size=1,
        )
