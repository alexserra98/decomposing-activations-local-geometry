"""Check geometry PCA against known subspaces and a sparse excluded cluster."""

from unittest.mock import patch

import pytest
import torch

from scripts.temporary.evaluate_toy_kmeans_geometry import (
    _empirical_eigenvalues_and_pca_validation,
)


@pytest.mark.parametrize("use_saved_pcs", [False, True])
@pytest.mark.parametrize("shift_centroid", [False, True])
def test_empirical_pca_excludes_sparse_cluster(tmp_path, use_saved_pcs, shift_centroid):
    centers = torch.tensor([[0., 0., 0., 0.], [10., 0., 0., 0.]])
    # Symmetric observations have exactly diagonal covariance (2, .5, 0, 0).
    residuals = torch.tensor([[2., 0., 0., 0.], [-2., 0., 0., 0.],
                              [0., 1., 0., 0.], [0., -1., 0., 0.]]).repeat(8, 1)
    points = torch.cat((residuals, centers[1:]))
    labels = torch.tensor([0] * 32 + [1])
    if shift_centroid:
        centers[0, 3] = 3.0
    saved = torch.eye(4)[:, :2].expand(2, 4, 2).clone() if use_saved_pcs else None
    with patch(
        "scripts.temporary.evaluate_toy_kmeans_geometry.ActivationBatchDataset",
        return_value=[points[:19], points[19:]],
    ):
        kwargs = dict(
            shard_dir=tmp_path, layer=0, centroids=centers,
            principal_components=saved, pca_capacity=2,
            assignments=labels, cluster_sizes=torch.tensor([32, 1]),
            eligible=torch.tensor([True, False]), batch_size=19, chunk_elems=128,
            eig_batch_size=1, device=torch.device("cpu"),
            pca_validation_tolerance=1e-4, centroid_mean_tolerance=1e-5,
        )
        if use_saved_pcs and shift_centroid:
            with pytest.raises(ValueError, match="stored centroids differ"):
                _empirical_eigenvalues_and_pca_validation(**kwargs)
            return
        eigenvalues, pcs, validation = _empirical_eigenvalues_and_pca_validation(**kwargs)
    torch.testing.assert_close(eigenvalues[0], torch.tensor([2., .5, 0.]).double())
    torch.testing.assert_close(pcs[0] @ pcs[0].T, torch.diag(torch.tensor([1., 1., 0., 0.])).double())
    assert validation["max_centroid_mean_l2_error"] == (3.0 if shift_centroid else 0.0)
    assert validation["source"] == ("centroid_artifact" if use_saved_pcs else "computed_from_assignments")
    assert torch.count_nonzero(eigenvalues[1]) == 0
    if not use_saved_pcs:
        assert torch.count_nonzero(pcs[1]) == 0
