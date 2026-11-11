"""Shared PCA coordinates for activation secants and saved model directions."""

import pytest
import torch

from dalg.analysis.activation_projection import (
    fit_activation_pca, project_alignment_inputs, project_gaussian_components,
)
from dalg.analysis.label_tangent_alignment import estimate_label_tangents, score_pc1_alignment


def axis_points():
    axes = torch.diag(torch.tensor([3., 2., 1., .1], dtype=torch.float64))
    return torch.cat((axes, -axes)) + torch.tensor([10., 20., 30., 40.])


def test_pca_keeps_leading_three_axes_without_whitening_or_mutating_inputs():
    values = axis_points()
    before = values.clone()
    projection = fit_activation_pca(values)
    projected, _ = project_alignment_inputs(values, {}, projection)
    torch.testing.assert_close(values, before)
    torch.testing.assert_close(projection["mean"], values.mean(0))
    torch.testing.assert_close(projection["basis"].T @ projection["basis"],
                               torch.diag(torch.tensor([1., 1., 1., 0.], dtype=torch.float64)))
    torch.testing.assert_close(projected.mean(0), torch.zeros(3, dtype=torch.float64))
    torch.testing.assert_close(projected.var(0), values.var(0)[:3])
    assert projection["retained_variance_fraction"] == pytest.approx(14 / 14.01)
    shifted, _ = project_alignment_inputs(values + 7, {}, projection)
    torch.testing.assert_close(shifted - projected, (7 * projection["basis"].sum(1)).expand_as(projected))


def test_projection_renormalizes_directions_and_preserves_native_assignments():
    values = axis_points()
    projection = fit_activation_pca(values)
    pc1 = torch.tensor([[1., 0., 0., 1.], [0., 0., 0., 1.],
                        [float("nan")] * 4], dtype=torch.float64)
    pc1[0] /= pc1[0].norm()
    geometry = dict(pc1=pc1, identifiable=torch.tensor([True, True, False]),
                    exclusion_reasons=["", "", "zero_rank"],
                    ranks=torch.tensor([1, 1, 0]), assignments=torch.arange(len(values)) % 3)
    _, projected = project_alignment_inputs(values, {"hddc": geometry}, projection)
    result = projected["hddc"]
    assert result["identifiable"].tolist() == [True, False, False]
    assert result["exclusion_reasons"] == ["", "null_pc1_projection", "zero_rank"]
    torch.testing.assert_close(result["pc1"].norm(dim=1), torch.tensor([1., 0., 0.], dtype=torch.float64))
    torch.testing.assert_close(result["pc1_projection_squared_norm"],
                               torch.tensor([.5, 0., 0.], dtype=torch.float64))
    torch.testing.assert_close(result["assignments"], geometry["assignments"])
    torch.testing.assert_close(result["ranks"], geometry["ranks"])
    assert geometry["exclusion_reasons"] == ["", "", "zero_rank"]
    assert geometry["identifiable"].tolist() == [True, True, False]
    assert geometry["pc1"].shape == (3, 4)


def test_alignment_is_computed_in_the_shared_projected_space():
    projection = fit_activation_pca(axis_points())
    values = torch.tensor([[0., 0., 0., 0.], [1., 0., 0., -1.]], dtype=torch.float64)
    geometry = dict(pc1=torch.tensor([[1., 0., 0., 1.]], dtype=torch.float64) / 2**.5,
                    identifiable=torch.tensor([True]), exclusion_reasons=[""],
                    ranks=torch.tensor([1]), assignments=torch.tensor([0, 0]))
    projected, geometries = project_alignment_inputs(values, {"hddc": geometry}, projection)
    tangents = estimate_label_tangents(projected, torch.tensor([2000, 2001]),
                                       ["a", "a"], template_id="a", block_size=1)["tangents"]
    result = score_pc1_alignment(tangents, geometries["hddc"]["pc1"], geometry["identifiable"],
                                 torch.ones((1, 1), dtype=torch.float64))
    # The raw secant and PC1 are orthogonal, but their 3D projections coincide.
    assert float((values[1] @ geometry["pc1"][0]).abs()) == 0
    assert result["score"] == pytest.approx(1.)


@pytest.mark.parametrize("values,n_components", [
    (torch.zeros((5, 4)), 3),
    (torch.ones((3, 4)), 3),
    (torch.eye(4), 5),
    (torch.tensor([[float("nan")]]), 1),
])
def test_pca_rejects_undefined_fit(values, n_components):
    with pytest.raises(ValueError):
        fit_activation_pca(values, n_components)


@pytest.mark.parametrize("kind", ["hddc", "kmeans"])
def test_projected_gaussians_match_full_covariance_projection(kind):
    from dalg.models.adaptive_q.mfa_hddc import MFA_HDDC
    from dalg.models.kmeans import KMeans
    from dalg.analysis.kmeans_gaussian import gaussian_from_cluster_pca

    values = axis_points().float()
    centers = values.mean(0, keepdim=True)
    if kind == "kmeans":
        model = KMeans.from_centroids(centers)
        model.compute_pcs(values)
        gaussian = gaussian_from_cluster_pca(model, train_cluster_sizes=model.cluster_counts)
    else:
        model = MFA_HDDC(centers, rank=2, isotropic_psi=True,
                         init_directions=torch.eye(4)[:, :2].unsqueeze(0))
        gaussian = model
    projection = fit_activation_pca(values)
    result = project_gaussian_components(model, kind, projection)
    basis = projection["basis"]
    w = gaussian._W().detach().double()
    covariance = w @ w.transpose(1, 2) + torch.diag_embed(gaussian._psi().detach().double())
    torch.testing.assert_close(result["covariances"], basis @ covariance @ basis.T)
    torch.testing.assert_close(result["centers"], (centers.double() - projection["mean"]) @ basis.T)
    assert result["valid"].tolist() == [True]
