"""Gaussian conversion estimates, parameter packing, and metric integration."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from dalg.analysis import kmeans_gaussian as exercise
from dalg.evaluation import common
from dalg.models.adaptive_q.mfa_hddc import MFA_HDDC
from dalg.models.kmeans import KMeans


def test_exercise_mixture_weights():
    counts = torch.tensor([6, 2, 0])
    logits = exercise.fit_mixture_logits(counts)
    torch.testing.assert_close(logits.softmax(0).double(), torch.tensor([0.75, 0.25, 0.], dtype=torch.float64))
    assert torch.isneginf(logits[2])
    assert counts.tolist() == [6, 2, 0]


@pytest.mark.parametrize(
    "spectrum, rank, expected_noise, expected_scales, expected_rank",
    [
        ([9., 4., 1.], 1, 2.5, [6.5 ** 0.5, 0.], 1),
        ([9., 4., 1.], 2, 1., [8. ** 0.5, 3. ** 0.5], 2),
        ([0., 0., 0.], 1, 1e-4, [0., 0.], 0),
        ([9., 0., 0.], 2, 1e-4, [(9. - 1e-4) ** 0.5, 0.], 1),
        ([4.], 1, 4., [0., 0.], 0),
    ],
)
def test_exercise_covariance(spectrum, rank, expected_noise, expected_scales, expected_rank):
    eigenvalues = torch.tensor([spectrum], dtype=torch.float64)
    ranks = torch.tensor([rank])
    noise, scales, effective_ranks = exercise.fit_cluster_covariances(
        eigenvalues, ranks, q=2, noise_floor=1e-4,
    )
    torch.testing.assert_close(noise.double(), torch.tensor([expected_noise], dtype=torch.float64))
    torch.testing.assert_close(scales.double(), torch.tensor([expected_scales], dtype=torch.float64))
    assert effective_ranks.dtype == torch.long
    assert effective_ranks.tolist() == [expected_rank]
    assert eigenvalues.tolist() == [spectrum] and ranks.tolist() == [rank]


def test_parameter_packing_preserves_source_and_matches_dense_gaussian(monkeypatch):
    source = KMeans.from_centroids(torch.zeros(1, 3), device="cpu")
    source.compute_pcs(torch.tensor([[-3., 0., 0.], [3., 0., 0.]]))
    before = {key: value.clone() for key, value in source.state_dict().items()}
    # Supply known parameters, without implementing either exercise algorithm.
    monkeypatch.setattr(exercise, "fit_mixture_logits", lambda counts: torch.zeros(1))
    monkeypatch.setattr(exercise, "fit_cluster_covariances", lambda *args, **kwargs: (
        torch.tensor([2.5]), torch.tensor([[6.5 ** 0.5]]), torch.tensor([1]),
    ))
    gaussian = exercise.gaussian_from_cluster_pca(source, train_cluster_sizes=torch.tensor([2]))
    covariance = gaussian.W @ gaussian.W.transpose(-1, -2) + torch.diag_embed(gaussian._psi())
    torch.testing.assert_close(covariance[0], torch.diag(torch.tensor([9., 2.5, 2.5])))
    points = torch.tensor([[1., 2., 3.], [-1., 0., 0.]])
    expected = torch.distributions.MultivariateNormal(torch.zeros(3), covariance[0]).log_prob(points)
    torch.testing.assert_close(gaussian.log_prob(points), expected)
    for key, value in source.state_dict().items():
        torch.testing.assert_close(value, before[key])
    assert all(not parameter.requires_grad for parameter in gaussian.parameters())


def test_fit_uses_gaussian_metrics_and_gaussian_map_activity(monkeypatch):
    gaussian = MFA_HDDC(
        torch.zeros(2, 3), rank=1,
        init_directions=torch.tensor([[[1.], [0.], [0.]]]).repeat(2, 1, 1),
        isotropic_psi=True,
    ).eval()
    # Identical Gaussians with unequal weights: all MAP assignments go to 0,
    # even though the source KMeans training counts say both clusters are live.
    with torch.no_grad():
        gaussian.pi_logits.copy_(torch.tensor([2., 0.]))
    data = SimpleNamespace(
        shard_dir=Path("unused"), train_positions=[0], val_positions=[1],
        config={"window": 4}, drop_prefix=0, train_cluster_sizes=torch.tensor([2, 2]),
    )
    batches = {
        0: torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [1., 1., 1.]]),
        1: torch.full((4, 3), 5.),
    }
    monkeypatch.setattr(common, "ActivationBatchDataset", lambda *args, row_subset, **kwargs: [batches[row_subset[0]]])
    source = object()

    def convert(model, *, train_cluster_sizes):
        assert model is source
        assert train_cluster_sizes.tolist() == [2, 2]
        return gaussian

    monkeypatch.setattr(common, "gaussian_from_cluster_pca", convert)
    report = common.evaluate_fit(
        source, data, model_kind="kmeans", layer=0, batch_size=4,
        device=torch.device("cpu"),
    )
    assert "quantization" not in report
    assert report["bic"]["active_components"] == 1
    assert data.train_cluster_sizes.tolist() == [2, 2]
    assert report["reconstruction"]["method"] == "mfa_posterior_mean"
    assert report["gaussian_surrogate"]["em_refit"] is False
    with torch.no_grad():
        for split, index in (("train", 0), ("validation", 1)):
            x = batches[index]
            assert report["nll"][split] == pytest.approx(float(gaussian.nll(x)))
            expected_mse = (x.double() - gaussian.reconstruct(x).double()).square().mean()
            assert report["reconstruction"][split]["mse"] == pytest.approx(float(expected_mse))


@pytest.mark.parametrize("evaluation_kind", ["unsupervised", "toy_manifold_tiling"])
def test_default_gaussian_report_is_accepted_by_pipeline(tmp_path, evaluation_kind):
    import copy
    import json

    from dalg.pipeline import execute_run, map_run_to_manifest
    from dalg.pipeline.evaluation import _evaluation_metrics_valid
    from tests.test_kmeans_pipeline import _config as toy_config
    from tests.test_pipeline_phases import _config as activation_config

    config = (activation_config if evaluation_kind == "unsupervised" else toy_config)(tmp_path)

    run = map_run_to_manifest(config)
    directory = execute_run(run)
    report = json.loads((directory / "metrics.json").read_text())
    assert _evaluation_metrics_valid(run, report)
    assert config["model"]["kind"] == "kmeans"
    assert run["training"]["model_kind"] == "kmeans"
    assert report["model_kind"] == "kmeans"
    assert report["gaussian_surrogate"]["source_model"] == "kmeans"
    assert "quantization" not in report
    assert "nll" in report and "bic" in report
    assert report["reconstruction"]["method"] == "mfa_posterior_mean"
    for field in ("nll", "bic", "gaussian_surrogate"):
        broken = copy.deepcopy(report)
        del broken[field]
        assert not _evaluation_metrics_valid(run, broken)
    broken = copy.deepcopy(report)
    broken["reconstruction"]["method"] = "nearest_centroid_local_pca"
    assert not _evaluation_metrics_valid(run, broken)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_covariance_handles_different_ranks_without_mutating_inputs(dtype):
    spectra = torch.tensor([
        [9., 4., 1.], [9., 0., 0.], [0., 0., 0.], [2., 2., 2.],
    ], dtype=dtype)
    proposed = torch.tensor([2, 2, 2, 2])
    original = spectra.clone()
    noise, scales, ranks = exercise.fit_cluster_covariances(
        spectra, proposed, q=2, noise_floor=1e-4,
    )
    torch.testing.assert_close(noise, torch.tensor([1., 1e-4, 1e-4, 2.], dtype=dtype))
    assert ranks.tolist() == [2, 1, 0, 0]
    active = torch.arange(2)[None, :] < ranks[:, None]
    recovered = scales.square() + noise[:, None]
    torch.testing.assert_close(recovered[active], spectra[:, :2][active])
    assert torch.all(scales[~active] == 0)
    assert torch.all(scales[active] > 0)
    assert proposed.tolist() == [2, 2, 2, 2]
    torch.testing.assert_close(spectra, original)


def test_real_conversion_with_empty_component_matches_dense_gaussian():
    # The six symmetric points have covariance diag(9, 4, 1).
    points = torch.diag(torch.tensor([27., 12., 3.]).sqrt())
    points = torch.cat([points, -points])
    source = KMeans.from_centroids(torch.tensor([[0., 0., 0.], [100., 100., 100.]]), device="cpu")
    source.compute_pcs(points, threshold=0.4)
    gaussian = exercise.gaussian_from_cluster_pca(source, train_cluster_sizes=torch.tensor([6, 0]))
    assert source.component_ranks.tolist() == [1, 0]
    assert gaussian.component_ranks.tolist() == [1, 0]
    torch.testing.assert_close(gaussian.pi_logits.softmax(0), torch.tensor([1., 0.]))
    assert torch.isneginf(gaussian.pi_logits[1])
    dense = torch.distributions.MultivariateNormal(
        torch.zeros(3), covariance_matrix=torch.diag(torch.tensor([9., 2.5, 2.5])),
    )
    with torch.no_grad(), gaussian.inference_cache():
        torch.testing.assert_close(gaussian.log_prob(points), dense.log_prob(points))
        expected = points * torch.tensor([6.5 / 9., 0., 0.])
        torch.testing.assert_close(gaussian.reconstruct(points), expected)
