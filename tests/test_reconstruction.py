from __future__ import annotations

import pytest
import torch

from dalg.analysis.reconstruction import ReconstructionMoments, reconstruction_errors
from dalg.evaluation.reconstruction import reconstruction_metadata, reconstruction_report_valid
from dalg.models.adaptive_q.mfa_hddc import MFA_HDDC
from dalg.models.kmeans import KMeans


@pytest.mark.parametrize("shared_b", [False, True])
@pytest.mark.parametrize("zero_rank", [False, True])
def test_hddc_reconstruction_matches_dense_posterior(shared_b, zero_rank):
    model = MFA_HDDC(
        torch.tensor([[-0.5, 0., 0.], [0.5, 0., 0.]]),
        rank=2, init_directions=torch.eye(3)[:, :2].repeat(2, 1, 1),
        shared_b=shared_b, isotropic_psi=not shared_b,
    )
    with torch.no_grad():
        model.rank_mask[0, 1] = 0
        if zero_rank:
            model.rank_mask[1] = 0
        model.pi_logits.copy_(torch.tensor([0.2, -0.1]))
        if not shared_b:
            model.psi_rho[1].add_(0.5)
    x = torch.tensor([[0., 1., 2.], [0.2, -0.5, 1.], [-1., 0., 0.]])
    W = model.W.detach().double()
    covariance = W @ W.transpose(-1, -2) + torch.diag_embed(model._psi().detach().double())
    gaussian = torch.distributions.MultivariateNormal(model.mu.detach().double(), covariance_matrix=covariance)
    weights = (gaussian.log_prob(x.double()[:, None]) + model.pi_logits.detach().double().log_softmax(0)).softmax(1)
    centered = x.double()[:, None] - model.mu.detach().double()
    # E[Wz | x,k] = WW^T C_k^{-1}(x-mu_k), independently of the low-rank solver.
    offset = torch.linalg.solve(covariance, centered[..., None])
    components = model.mu.detach().double() + ((W @ W.transpose(-1, -2)) @ offset).squeeze(-1)
    reconstructed = (weights[..., None] * components).sum(1)
    expected = (x - reconstructed).square().mean(1)
    with torch.no_grad(), model.inference_cache():
        actual, excluded = reconstruction_errors(model, x, model_kind="hddc")
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-8)
    assert excluded == 0
    hard = components[torch.arange(len(x)), weights.argmax(1)]
    assert not torch.allclose(expected, (x - hard).square().mean(1))


def test_kmeans_uses_retained_cluster_pcs_and_excludes_invalid_assignments():
    model = KMeans.from_centroids(torch.tensor([[0., 0., 0.], [10., 0., 0.], [20., 0., 0.]]))
    model._pcs = torch.zeros(3, 3, 2)
    model._pcs[0, 0, 0] = 1
    model._pcs[1, 1, 0] = model._pcs[1, 2, 1] = 1
    model._component_ranks = torch.tensor([1, 2, 0])
    model._pca_valid = torch.tensor([True, True, False])
    model._init_pcs = torch.ones(3, 3, 2)  # Must not influence reconstruction.
    x = torch.tensor([[1., 2., 0.], [11., 2., 3.], [21., 0., 0.]])
    actual, excluded = reconstruction_errors(model, x, model_kind="kmeans")
    torch.testing.assert_close(actual, torch.tensor([4 / 3, 1 / 3], dtype=torch.float64))
    assert excluded == 1
    model._pca_valid.zero_()
    actual, excluded = reconstruction_errors(model, x, model_kind="kmeans")
    assert actual.numel() == 0 and excluded == 3


@pytest.mark.parametrize("batch_size", [1, 3, 7])
def test_streaming_moments_equal_sample_mean_and_standard_error(batch_size):
    errors = torch.tensor([0., 0.5, 2., 4.5, 1., 10., 2.], dtype=torch.float64)
    moments = ReconstructionMoments()
    for batch in errors.split(batch_size):
        moments.update(batch)
    moments.update(torch.empty(0), excluded_n=2)
    result = moments.summary()
    assert result["mse"] == pytest.approx(float(errors.mean()))
    assert result["standard_error"] == pytest.approx(float(errors.std() / len(errors) ** 0.5))
    assert result["n"] == len(errors) and result["excluded_n"] == 2


def test_empty_and_single_sample_statistics():
    moments = ReconstructionMoments()
    moments.update(torch.empty(0), excluded_n=3)
    assert moments.summary() == {"mse": None, "standard_error": None, "n": 0, "excluded_n": 3}
    moments.update(torch.tensor([0.5]))
    assert moments.summary() == {"mse": 0.5, "standard_error": None, "n": 1, "excluded_n": 3}
    with pytest.raises(ValueError, match="finite nonnegative"):
        moments.update(torch.tensor([float("nan")]))


@pytest.mark.parametrize("included", [0, 1, 3])
def test_report_validation_for_sparse_and_empty_populations(included):
    moments = ReconstructionMoments()
    moments.update(torch.arange(included, dtype=torch.float64), excluded_n=3 - included)
    report = {
        **reconstruction_metadata("kmeans", 2),
        "train": moments.summary(), "validation": ReconstructionMoments().summary(),
    }
    dataset = {"d_model": 2, "window": 1, "drop_prefix": 0,
               "train_rows": 3, "train_points": 3, "validation_rows": 0, "validation_points": 0}
    assert reconstruction_report_valid(report, model_kind="kmeans", dataset=dataset)
    report["validation"]["mse"] = 0.0
    assert not reconstruction_report_valid(report, model_kind="kmeans", dataset=dataset)
