from __future__ import annotations

import copy
import json

import pytest
import torch

from dalg.cli.adaptive_q.run_training_hddc import build_parser, validate_args
from dalg.models.adaptive_q.mfa_hddc import MFA_HDDC, load_mfa_hddc
from dalg.models.adaptive_q.train_em_hddc import (
    EMConfig, EMStatistics, expectation_step, maximization_step, train_em_hddc,
)


@pytest.fixture(autouse=True)
def small_torch_thread_pool():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def fixture():
    torch.manual_seed(23)
    centers = torch.tensor([[-5., 0., 0., 0.], [5., 0., 0., 0.]])
    x = torch.cat([
        centers[k] + torch.randn(n, 4) * torch.tensor([1.2, .6, .2, .2])
        for k, n in enumerate([183, 117])
    ])
    model = MFA_HDDC(centers + .25, rank=2, shared_b=True)
    return model, x


def dense_joint(model, x):
    W = model._W().detach().double()
    covariance = W @ W.transpose(-1, -2) + torch.diag_embed(model._psi().detach().double())
    return torch.distributions.MultivariateNormal(
        model.mu.detach().double(), covariance_matrix=covariance,
    ).log_prob(x.double()[:, None]) + model.pi_logits.detach().double().log_softmax(0)


@pytest.mark.parametrize("chunk", [1, 3])
@pytest.mark.parametrize("small_noise", [False, True])
def test_centered_cached_likelihood_matches_dense_gaussians(chunk, small_noise):
    model, x = fixture()
    with torch.no_grad():
        model.mu.add_(1e5)
        x.add_(1e5)
        if small_noise:
            model.psi_rho.fill_(-15)
    reference = dense_joint(model, x)
    with torch.no_grad(), model.inference_cache(dtype=torch.float64, component_chunk_size=chunk):
        actual = model.log_prob_components(x) + model.pi_logits.double().log_softmax(0)
    torch.testing.assert_close(actual, reference, rtol=2e-8, atol=1e-6)
    torch.testing.assert_close(actual.softmax(1), reference.softmax(1), rtol=1e-7, atol=1e-8)


def test_streamed_moments_and_full_m_step_match_dense_reference():
    model, x = fixture()
    before = copy.deepcopy(model)
    r = dense_joint(model, x).softmax(1)
    n = r.sum(0)
    means = r.T @ x.double() / n[:, None]
    centered = x.double()[:, None] - means[None]
    covariance = torch.einsum("bk,bkd,bke->kde", r, centered, centered) / n[:, None, None]
    stats = expectation_step(model, x.split(37), component_chunk_size=1, expected_rows=len(x))
    torch.testing.assert_close(stats.counts, n)
    shift = stats.residual_sum / n[:, None]
    torch.testing.assert_close(model.mu.double() + shift, means)
    torch.testing.assert_close(stats.scatter / n[:, None, None] - shift[:, :, None] * shift[:, None], covariance)
    summary = maximization_step(model, stats, EMConfig(threshold=.03, eig_batch_size=1))
    ranks = model.component_ranks
    values, vectors = torch.linalg.eigh(covariance)
    values, vectors = values.flip(-1), vectors.flip(-1)
    residual = torch.stack([values[k, ranks[k]:].sum() for k in range(model.K)])
    b = (n * residual).sum() / (n * (model.D - ranks)).sum()
    torch.testing.assert_close(model.mu.double(), means, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(model.pi_logits.double().softmax(0), n / len(x))
    torch.testing.assert_close(model._psi()[0, 0].double(), b, rtol=1e-6, atol=1e-7)
    assert summary["n_updated"] == model.K
    for k, rank in enumerate(ranks.tolist()):
        target = (vectors[k, :, :rank] * (values[k, :rank] - b)) @ vectors[k, :, :rank].T + b * torch.eye(model.D)
        W = model._W()[k].double()
        torch.testing.assert_close(W @ W.T + model._psi()[k, 0].double() * torch.eye(model.D), target, rtol=2e-6, atol=2e-6)
    assert not torch.equal(before.mu, model.mu)
    assert not torch.equal(before.pi_logits, model.pi_logits)


def test_batch_and_component_chunking_are_invariant():
    model, x = fixture()
    a = expectation_step(model, [x], component_chunk_size=2)
    b = expectation_step(model, x.split(29), component_chunk_size=1)
    for field in ("counts", "residual_sum", "scatter"):
        torch.testing.assert_close(getattr(a, field), getattr(b, field), rtol=1e-12, atol=1e-10)
    assert a.nll == pytest.approx(b.nll, abs=1e-12)
    assert b.n_rows == len(x)


def test_hard_initialization_uses_full_partition_and_weighted_means():
    model, x = fixture()
    labels = torch.cdist(x.double(), model.mu.double()).argmin(1)
    stats = expectation_step(model, x.split(31), hard=True)
    maximization_step(model, stats, EMConfig())
    for k in range(model.K):
        torch.testing.assert_close(model.mu[k], x[labels == k].mean(0))
    torch.testing.assert_close(model.pi_logits.softmax(0), torch.bincount(labels).float() / len(x))
    assert model.component_ranks.tolist() == [2, 2]


@pytest.mark.parametrize("failure", ["negative_mass", "all_zero_mass", "nonfinite"])
def test_failed_m_step_does_not_mutate_any_parameters(failure):
    model, x = fixture()
    stats = expectation_step(model, [x])
    if failure == "negative_mass":
        stats.counts[0] = -1
    elif failure == "all_zero_mass":
        stats.counts.zero_()
    else:
        stats.scatter[0, 0, 0] = float("nan")
    before = copy.deepcopy(model.state_dict())
    with pytest.raises((ValueError, RuntimeError)):
        maximization_step(model, stats, EMConfig())
    for key, value in before.items():
        torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)


def test_zero_scatter_m_step_keeps_spherical_components_and_can_run_next_e_step():
    model, x = fixture()
    stats = expectation_step(model, [x])
    stats.residual_sum.zero_()
    stats.scatter.zero_()
    summary = maximization_step(model, stats, EMConfig())
    assert model.component_ranks.tolist() == [0, 0]
    assert summary["n_updated"] == 2
    assert torch.count_nonzero(model._W()) == 0
    torch.testing.assert_close(model.pi_logits.softmax(0), (stats.counts / stats.counts.sum()).float())
    subsequent = expectation_step(model, [model.mu.detach().clone()], component_chunk_size=1)
    assert subsequent.nll is not None and torch.isfinite(torch.tensor(subsequent.nll))
    assert (subsequent.counts > 0).all()


def test_resume_equals_uninterrupted_and_model_file_is_best_state(tmp_path):
    model, x = fixture()
    uninterrupted = copy.deepcopy(model)
    cfg = EMConfig(tol=0)
    full = train_em_hddc(uninterrupted, x.split(41), cfg=cfg, epochs=4, val_tensor=x[::7], log=lambda _: None)
    train_em_hddc(model, x.split(41), cfg=cfg, epochs=2, val_tensor=x[::7], out_dir=tmp_path, log=lambda _: None)
    history = train_em_hddc(model, x.split(41), cfg=cfg, epochs=4, val_tensor=x[::7], out_dir=tmp_path, log=lambda _: None)
    assert [row["iteration"] for row in history] == list(range(5))
    for actual, expected in zip(history, full):
        assert actual["train_nll"] == pytest.approx(expected["train_nll"], abs=1e-10)
    for key, value in uninterrupted.state_dict().items():
        torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
    checkpoint = torch.load(tmp_path / "checkpoint.pt", weights_only=False)
    assert checkpoint["fit_method"] == "em" and "optimizer" not in checkpoint
    loaded = load_mfa_hddc(tmp_path / "mfa_model.pt")
    for key, value in loaded.state_dict().items():
        torch.testing.assert_close(value, checkpoint["best_state"][key], rtol=0, atol=0)
    measured = expectation_step(loaded, x[::7].split(23), collect_statistics=False).nll
    assert measured == pytest.approx(min(row["val_nll"] for row in history), abs=1e-10)


def test_fixed_rank_iterations_improve_likelihood_and_no_optimizer_is_used(monkeypatch):
    model, x = fixture()
    monkeypatch.setattr(torch.optim, "Adam", lambda *a, **kw: pytest.fail("EM constructed Adam"))
    history = train_em_hddc(model, x.split(51), cfg=EMConfig(tol=0), epochs=4, log=lambda _: None)
    for previous, current in zip(history, history[1:]):
        if current["rank_changes"] == 0:
            assert current["train_nll"] <= previous["train_nll"] + 1e-6


def test_em_stops_after_three_stable_iterations():
    model, x = fixture()
    history = train_em_hddc(model, x.split(51), cfg=EMConfig(tol=1), epochs=30, log=lambda _: None)
    assert history[-1]["iteration"] == 3


def test_adam_checkpoint_cannot_be_resumed_as_em(tmp_path):
    model, x = fixture()
    torch.save({"model": model.state_dict(), "optimizer": {}}, tmp_path / "checkpoint.pt")
    with pytest.raises(ValueError, match="Adam checkpoint"):
        train_em_hddc(model, [x], out_dir=tmp_path)


def test_resume_rejects_changed_numerical_floor(tmp_path):
    model, x = fixture()
    train_em_hddc(model, [x], epochs=1, out_dir=tmp_path, log=lambda _: None)
    model._eps = 0.1
    with pytest.raises(ValueError, match="numerical floor"):
        train_em_hddc(model, [x], epochs=2, out_dir=tmp_path)


@pytest.mark.parametrize("flags", [
    ["--surgery-min-count", "1"], ["--surgery-every-epochs", "1"],
    ["--steps-per-epoch", "2"], ["--max-steps", "2"],
    ["--epochs", "0"], ["--lr", "0.1"], ["--em-component-chunk-size", "0"],
])
def test_em_cli_rejects_incompatible_controls(flags):
    args = build_parser().parse_args(["--shard-dir", "unused", "--layer", "0", "--K", "2", "--fit-method", "em", "--shared-b", *flags])
    with pytest.raises(SystemExit):
        validate_args(args)


def test_stream_contract_rejects_missing_rows():
    model, x = fixture()
    with pytest.raises(ValueError, match="expected"):
        expectation_step(model, [x], expected_rows=len(x) + 1)
    with pytest.raises(ValueError, match="non-empty"):
        expectation_step(model, [])


def test_rank_can_decrease_and_increase():
    model, _ = fixture()
    for spectrum, expected in [
        ([4., .1, .1, .1], 1),
        ([0., 0., 0., 0.], 0),
        ([4., 2., .1, .1], 2),
    ]:
        stats = EMStatistics(
            torch.ones(2, dtype=torch.float64) * 100,
            torch.zeros(2, 4, dtype=torch.float64),
            torch.diag(torch.tensor(spectrum, dtype=torch.float64)).repeat(2, 1, 1) * 100,
            200, None,
        )
        maximization_step(model, stats, EMConfig())
        assert model.component_ranks.tolist() == [expected, expected]


def test_warm_start_preserves_initial_state_before_first_update():
    model, x = fixture()
    nll = expectation_step(model, [x], collect_statistics=False).nll
    history = train_em_hddc(model, [x], initialize=False, epochs=1, log=lambda _: None)
    assert history[0]["train_nll"] == pytest.approx(nll)


@pytest.mark.parametrize("workers", [0, 2])
def test_em_pipeline_with_shards_and_assignments(tmp_path, monkeypatch, workers):
    from dalg.pipeline import execute_run, resolve_experiment
    from tests.synthetic_shards import build_multi_shard
    from tests.test_training_pipeline import _config, _write_yaml

    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    shard_dir = build_multi_shard(tmp_path / "shards", n_shards=2, rows_per_shard=6)
    config = _config(tmp_path, shard_dir)
    from dalg.models.kmeans import KMeans, save_kmeans

    initializer = tmp_path / "kmeans_model.pt"
    save_kmeans(KMeans.from_centroids(torch.tensor([[260., 261.], [1260., 1261.]])), initializer)
    config["model"] = {"kind": "hddc", "K": 2, "q_max": 1, "shared_b": True}
    config["training"].update({
        "fit_method": "em", "epochs": 2, "num_workers": workers,
        "kmeans_model_path": str(initializer), "em_component_chunk_size": 1,
        "em_eig_batch_size": 1, "epoch_snapshot_every": 1,
    })
    run = resolve_experiment(_write_yaml(tmp_path / "em.yaml", config))[0]
    assert run["training"]["arguments"]["fit_method"] == "em"
    run_dir = execute_run(run)
    checkpoint = torch.load(run_dir / "checkpoint.pt", weights_only=False)
    assert checkpoint["fit_method"] == "em" and checkpoint["epoch"] == 2
    assert checkpoint["history"][-1]["n_rows"] == 24
    saved_config = json.loads((run_dir / "config.json").read_text())
    assert saved_config["fit_method"] == "em"
    assert (run_dir / "epoch_0001" / "mfa_model.pt").exists()
    assert load_mfa_hddc(run_dir / "mfa_model.pt").shared_b
    assert list(run_dir.rglob("*assignment*.pt"))
    assert not list(run_dir.rglob("centroids.pt"))


@pytest.mark.parametrize("hard", [False, True])
def test_zero_membership_components_stay_dead_through_resume(tmp_path, hard):
    model, x = fixture()
    with torch.no_grad():
        model.mu[1].fill_(1e4)
    before = copy.deepcopy(model.state_dict())
    stats = expectation_step(model, x.split(41), hard=hard)
    assert stats.counts[1] == 0
    summary = maximization_step(model, stats, EMConfig())
    assert summary["n_updated"] == 1
    torch.testing.assert_close(model.pi_logits.softmax(0), torch.tensor([1., 0.]))
    for key in ("mu", "dir_raw", "scale_rho", "rank_mask"):
        torch.testing.assert_close(model.state_dict()[key][1], before[key][1], rtol=0, atol=0)
    # The shared floor and live covariance must match a fit with only the live component.
    single = MFA_HDDC(before["mu"][:1].clone(), rank=2, shared_b=True)
    single_stats = EMStatistics(stats.counts[:1], stats.residual_sum[:1], stats.scatter[:1], len(x), None)
    maximization_step(single, single_stats, EMConfig())
    torch.testing.assert_close(model._psi()[0], single._psi()[0])
    torch.testing.assert_close(model._W()[0] @ model._W()[0].T, single._W()[0] @ single._W()[0].T)
    cfg = EMConfig(tol=0)
    train_em_hddc(model, x.split(41), cfg=cfg, initialize=False, epochs=1, out_dir=tmp_path, log=lambda _: None)
    loaded = load_mfa_hddc(tmp_path / "mfa_model.pt")
    history = train_em_hddc(loaded, x.split(41), cfg=cfg, initialize=False, epochs=3, out_dir=tmp_path, log=lambda _: None)
    assert history[-1]["iteration"] == 3
    assert all(row["dead_components"] == 1 for row in history)
    subsequent = expectation_step(loaded, [x])
    assert subsequent.counts[1] == 0
    assert torch.isfinite(torch.tensor(subsequent.nll))
    assert torch.count_nonzero(loaded.responsibilities(x)[:, 1]) == 0
