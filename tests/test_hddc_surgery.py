"""Tests for the rank mask, isotropic Psi, and HDDC covariance surgery.

Coverage:
- an all-ones mask with isotropic Psi reproduces the plain-MFA likelihood, and
  masked columns are exactly zero in W with exactly zero gradient
- the mask and the isotropic Psi shape survive save_mfa/load_mfa and the
  component-sharded save/load path, and pre-mask checkpoints still load
- surgery on a planted low-rank Gaussian recovers Q, lambda, b and d_k, with
  the b_k > 0 and lam_j >= b_k guarantees holding
- rank can go back up at a later surgery (all q_max columns are rewritten)
- the train_nll hook runs surgery on schedule without blowing up the NLL
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from dalg.analysis.cluster_assignments import compute_assignments  # noqa: E402
from dalg.models.adaptive_q.hddc_surgery import (  # noqa: E402
    SurgeryConfig,
    accumulate_statistics,
    hddc_surgery,
    parameter_count,
    reset_optimizer_state,
    surgery_params,
)
from dalg.models.adaptive_q.mfa_hddc import (  # noqa: E402
    MFA_HDDC,
    ComponentShardedMFA_HDDC,
    load_component_shards_hddc,
    load_mfa_hddc,
    save_component_shard_hddc,
    save_mfa_hddc,
)
from dalg.models.adaptive_q.train_hddc import train_nll_hddc  # noqa: E402


def _planted_gaussian(
    *, D: int = 32, d_true: int = 3, b_true: float = 0.02, n: int = 120_000, seed: int = 0
):
    """One Gaussian with covariance U diag(lam) U^T + b I and a known mean."""
    g = torch.Generator().manual_seed(seed)
    lam = torch.tensor([4.0, 2.0, 1.0])[:d_true]
    U = torch.linalg.qr(torch.randn(D, D, generator=g)).Q[:, :d_true]
    W = U * (lam - b_true).sqrt()
    mu = torch.randn(D, generator=g) * 3.0
    z = torch.randn(n, d_true, generator=g)
    x = z @ W.T + mu + (b_true ** 0.5) * torch.randn(n, D, generator=g)
    return x, mu, U, lam


def _batches(x: torch.Tensor, size: int = 8192):
    return [x[i:i + size] for i in range(0, x.shape[0], size)]


# --------------------------------------------------------------------------
# Model additions: isotropic Psi and the rank mask
# --------------------------------------------------------------------------


def test_isotropic_psi_matches_equivalent_per_component_psi():
    torch.manual_seed(0)
    K, D, q = 5, 12, 4
    centroids = torch.randn(K, D)
    x = torch.randn(32, D)

    iso = MFA_HDDC(centroids, rank=q, isotropic_psi=True, psi_init=0.7)
    ref = MFA_HDDC(centroids, rank=q, psi_per_component=True, psi_init=0.7)
    ref.dir_raw.data.copy_(iso.dir_raw.data)
    ref.scale_rho.data.copy_(iso.scale_rho.data)

    assert tuple(iso.psi_rho.shape) == (K, 1)
    assert torch.allclose(iso._psi(), ref._psi())
    assert torch.allclose(iso.nll(x), ref.nll(x))


def test_all_ones_mask_is_a_no_op():
    torch.manual_seed(1)
    K, D, q = 4, 10, 3
    model = MFA_HDDC(torch.randn(K, D), rank=q, isotropic_psi=True)
    x = torch.randn(16, D)

    assert torch.equal(model.rank_mask, torch.ones(K, q))
    with torch.no_grad():
        baseline = float(model.nll(x))
        model.rank_mask.fill_(1.0)
        assert float(model.nll(x)) == pytest.approx(baseline, abs=0.0)


def test_inference_cache_agrees_with_uncached_path_for_isotropic_psi():
    torch.manual_seed(2)
    model = MFA_HDDC(torch.randn(6, 16), rank=4, isotropic_psi=True)
    model.rank_mask[2, 1] = 0.0
    x = torch.randn(24, 16)
    with torch.no_grad():
        plain = model.log_prob_components(x)
        with model.inference_cache():
            cached = model.log_prob_components(x)
    assert torch.allclose(plain, cached, atol=1e-4)


def test_masked_columns_are_zero_and_receive_zero_gradient():
    torch.manual_seed(3)
    K, D, q = 4, 10, 3
    model = MFA_HDDC(torch.randn(K, D), rank=q, isotropic_psi=True)
    model.rank_mask[1, 2] = 0.0

    assert torch.equal(model._W()[1, :, 2], torch.zeros(D))
    assert model.component_ranks.tolist() == [q, q - 1, q, q]

    model.nll(torch.randn(20, D)).backward()
    assert torch.equal(model.dir_raw.grad[1, :, 2], torch.zeros(D))
    assert float(model.scale_rho.grad[1, 2]) == 0.0
    assert float(model.dir_raw.grad[1, :, 0].abs().max()) > 0.0


def test_masking_a_column_equals_a_model_without_it():
    """A masked rank-q model must equal the rank-(q-1) model on the shared columns."""
    torch.manual_seed(4)
    K, D, q = 3, 12, 4
    x = torch.randn(40, D)
    full = MFA_HDDC(torch.randn(K, D), rank=q, isotropic_psi=True)
    full.rank_mask[:, q - 1] = 0.0

    small = MFA_HDDC(full.mu.data.clone(), rank=q - 1, isotropic_psi=True)
    small.dir_raw.data.copy_(full.dir_raw.data[:, :, : q - 1])
    small.scale_rho.data.copy_(full.scale_rho.data[:, : q - 1])
    small.psi_rho.data.copy_(full.psi_rho.data)

    assert torch.allclose(full.nll(x), small.nll(x), atol=1e-5)


# --------------------------------------------------------------------------
# Checkpoint round-trip
# --------------------------------------------------------------------------


def test_mask_and_isotropic_psi_survive_save_load():
    torch.manual_seed(5)
    model = MFA_HDDC(torch.randn(6, 14), rank=4, isotropic_psi=True)
    model.rank_mask[0, 3] = 0.0
    model.rank_mask[4, 1:] = 0.0
    x = torch.randn(16, 14)

    with tempfile.TemporaryDirectory() as d:
        path = str(Path(d) / "mfa.pt")
        save_mfa_hddc(model, path)
        loaded = load_mfa_hddc(path)

    assert loaded.isotropic_psi is True
    assert tuple(loaded.psi_rho.shape) == (6, 1)
    assert torch.equal(loaded.rank_mask, model.rank_mask)
    assert loaded.component_ranks.tolist() == model.component_ranks.tolist()
    assert torch.allclose(loaded.nll(x), model.nll(x))


def test_hddc_checkpoint_supports_assignment_analysis():
    torch.manual_seed(51)
    model = MFA_HDDC(torch.randn(6, 14), rank=4, isotropic_psi=True)
    model.rank_mask[0, 2:] = 0.0
    x = torch.randn(23, 14)

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "mfa_model.pt"
        save_mfa_hddc(model, str(path))
        sizes, assignments, max_resp, peakedness = compute_assignments(
            path,
            [x],
            device="cpu",
            model_type="hddc",
        )

    assert int(sizes.sum()) == len(x)
    assert assignments.shape == (len(x),)
    assert max_resp.shape == (len(x),)
    assert set(peakedness) == {"entropy", "one_minus_max", "top1_minus_top2"}


def test_checkpoints_without_a_rank_mask_load_as_full_rank():
    torch.manual_seed(6)
    model = MFA_HDDC(torch.randn(5, 9), rank=3)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "legacy.pt"
        save_mfa_hddc(model, str(path))
        blob = torch.load(path, weights_only=False)
        del blob["state_dict"]["rank_mask"]
        blob["meta"].pop("isotropic_psi", None)
        torch.save(blob, path)
        loaded = load_mfa_hddc(str(path))

    assert torch.equal(loaded.rank_mask, torch.ones(5, 3))
    assert loaded.isotropic_psi is False


def test_component_shard_round_trip_preserves_mask_and_isotropic_psi():
    torch.manual_seed(7)
    K, D, q, world = 6, 10, 3, 2
    centroids = torch.randn(K, D)
    x = torch.randn(12, D)

    reference = MFA_HDDC(centroids.clone(), rank=q, isotropic_psi=True)
    reference.rank_mask[1, 2] = 0.0
    reference.rank_mask[5, 1:] = 0.0

    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        for r in range(world):
            start = r * (K // world)
            end = start + (K // world)
            shard = ComponentShardedMFA_HDDC(
                centroids[start:end].clone(),
                rank=q,
                global_K=K,
                component_start=start,
                isotropic_psi=True,
            )
            for name in ("mu", "dir_raw", "scale_rho", "psi_rho", "pi_logits"):
                getattr(shard, name).data.copy_(
                    getattr(reference, name).data[start:end]
                )
            shard.rank_mask.data.copy_(reference.rank_mask.data[start:end])
            save_component_shard_hddc(shard, out / f"mfa_model_rank{r:04d}.pt")
        (out / "mfa_model_shards.json").write_text(json.dumps({
            "format": "component_sharded_mfa",
            "global_K": K,
            "rank": q,
            "world_size": world,
            "shards": [f"mfa_model_rank{r:04d}.pt" for r in range(world)],
        }))
        merged = load_component_shards_hddc(out)

    assert merged.isotropic_psi is True
    assert tuple(merged.psi_rho.shape) == (K, 1)
    assert torch.equal(merged.rank_mask, reference.rank_mask)
    assert torch.allclose(merged.nll(x), reference.nll(x), atol=1e-5)


# --------------------------------------------------------------------------
# Surgery: phases A and B
# --------------------------------------------------------------------------


def test_surgery_recovers_a_planted_low_rank_covariance():
    x, mu, U, lam = _planted_gaussian(D=32, d_true=3, b_true=0.02)
    q = 8
    model = MFA_HDDC(mu[None, :].clone(), rank=q, isotropic_psi=True, psi_init=0.5)

    summary = hddc_surgery(
        model,
        _batches(x),
        SurgeryConfig(enabled=True, every=1, threshold=0.01, min_count=10.0),
    )

    assert summary["d_k_per_component"] == [3]
    assert summary["n_updated"] == 1 and summary["n_skipped"] == 0
    assert model.rank_mask[0].tolist() == [1, 1, 1, 0, 0, 0, 0, 0]

    b_hat = summary["b_k_mean"]
    assert b_hat > 0.0
    assert b_hat == pytest.approx(0.02, rel=0.05)

    with torch.no_grad():
        lam_hat = model._scale()[0] ** 2 + model._psi()[0, 0]
    # The retained eigenvalues dominate the noise floor by construction.
    assert bool((lam_hat[:3] >= b_hat).all())
    assert torch.allclose(lam_hat[:3], lam, rtol=0.05)
    # Recovered subspace matches the planted one: sum of squared cosines == d.
    U_hat = model._dir_hat()[0][:, :3].detach()
    assert float((U.T @ U_hat).pow(2).sum()) == pytest.approx(3.0, abs=1e-2)


def test_higher_threshold_selects_a_smaller_rank():
    x, mu, _U, _lam = _planted_gaussian(D=32, d_true=3, b_true=0.02)
    ranks = {}
    for t in (0.01, 0.5):
        model = MFA_HDDC(mu[None, :].clone(), rank=8, isotropic_psi=True, psi_init=0.5)
        summary = hddc_surgery(
            model,
            _batches(x),
            SurgeryConfig(enabled=True, every=1, threshold=t, min_count=10.0),
        )
        ranks[t] = summary["d_k_per_component"][0]
    # lam = [4, 2, 1, b, ...]: only the leading gap clears t = 0.5 * lam_1.
    assert ranks[0.5] < ranks[0.01]
    assert ranks[0.5] >= 1


def test_low_count_components_are_skipped_untouched():
    x, mu, _U, _lam = _planted_gaussian(D=16, d_true=2, b_true=0.05, n=4_000)
    # A second component parked far away owns essentially no responsibility.
    centroids = torch.stack([mu, mu + 500.0])
    model = MFA_HDDC(centroids, rank=4, isotropic_psi=True, psi_init=0.5)
    before = model.dir_raw.data.clone()

    summary = hddc_surgery(
        model,
        _batches(x),
        SurgeryConfig(enabled=True, every=1, threshold=0.01, min_count=50.0),
    )

    assert summary["n_skipped"] == 1
    assert summary["n_updated"] == 1
    assert torch.equal(model.dir_raw.data[1], before[1])
    assert model.rank_mask[1].tolist() == [1, 1, 1, 1]


def test_rank_can_increase_at_a_later_surgery():
    """All q_max columns are rewritten, so a narrowed component can widen again."""
    x, mu, _U, _lam = _planted_gaussian(D=32, d_true=3, b_true=0.02)
    batches = _batches(x)
    model = MFA_HDDC(mu[None, :].clone(), rank=8, isotropic_psi=True, psi_init=0.5)

    tight = hddc_surgery(
        model, batches,
        SurgeryConfig(enabled=True, every=1, threshold=0.5, min_count=10.0),
    )
    loose = hddc_surgery(
        model, batches,
        SurgeryConfig(enabled=True, every=1, threshold=0.01, min_count=10.0),
    )
    assert loose["d_k_per_component"][0] > tight["d_k_per_component"][0]
    assert loose["d_k_per_component"][0] == 3


def test_statistics_center_on_the_model_mean_not_the_empirical_mean():
    """S_k is the ML covariance *given* mu_k, so a displaced mu inflates it."""
    x, mu, _U, _lam = _planted_gaussian(D=16, d_true=2, b_true=0.05, n=20_000)
    shift = torch.zeros(16)
    shift[0] = 1.0

    on_mean = MFA_HDDC(mu[None, :].clone(), rank=4, isotropic_psi=True, psi_init=0.5)
    off_mean = MFA_HDDC((mu + shift)[None, :].clone(), rank=4, isotropic_psi=True,
                   psi_init=0.5)

    N_a, S_a, rows = accumulate_statistics(on_mean, _batches(x), device=x.device)
    N_b, S_b, _ = accumulate_statistics(off_mean, _batches(x), device=x.device)

    assert rows == x.shape[0]
    assert float(N_a.sum()) == pytest.approx(x.shape[0], rel=1e-6)
    trace_a = float(torch.diagonal(S_a[0] / N_a[0]).sum())
    trace_b = float(torch.diagonal(S_b[0] / N_b[0]).sum())
    # The displacement adds ||shift||^2 = 1 to the trace.
    assert trace_b - trace_a == pytest.approx(1.0, rel=0.05)


def test_surgery_requires_isotropic_psi():
    x, mu, _U, _lam = _planted_gaussian(D=16, d_true=2, n=2_000)
    model = MFA_HDDC(mu[None, :].clone(), rank=4)
    with pytest.raises(ValueError, match="isotropic_psi"):
        hddc_surgery(model, _batches(x), SurgeryConfig(enabled=True, every=1))


def test_parameter_count_tracks_the_rank_mask():
    model = MFA_HDDC(torch.zeros(4, 20), rank=5, isotropic_psi=True)
    full = parameter_count(model)
    model.rank_mask[:, 3:] = 0.0
    assert parameter_count(model) < full


# --------------------------------------------------------------------------
# Phase C and the training-loop hook
# --------------------------------------------------------------------------


def test_reset_optimizer_state_only_clears_surgery_params():
    model = MFA_HDDC(torch.randn(3, 8), rank=2, isotropic_psi=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.nll(torch.randn(10, 8)).backward()
    opt.step()
    assert len(opt.state) == len(list(model.parameters()))

    dropped = reset_optimizer_state(opt, surgery_params(model))
    assert dropped == 3
    assert model.mu in opt.state and model.pi_logits in opt.state
    assert model.dir_raw not in opt.state


def test_train_nll_runs_surgery_on_schedule_without_blowing_up():
    torch.manual_seed(11)
    D, q = 24, 6
    basis = torch.linalg.qr(torch.randn(D, D)).Q
    blobs = []
    for c in range(3):
        centre = torch.randn(D) * 4.0
        loadings = basis[:, 2 * c:2 * c + 2] * torch.tensor([2.0, 1.0])
        blobs.append(
            torch.randn(4_000, 2) @ loadings.T + centre + 0.05 * torch.randn(4_000, D)
        )
    x = torch.cat(blobs)[torch.randperm(12_000)]
    x_train, x_val = x[:10_000], x[10_000:]

    model = MFA_HDDC(x_train[:6].clone(), rank=q, isotropic_psi=True, psi_init=0.1)
    info = train_nll_hddc(
        model,
        [x_train[i:i + 500] for i in range(0, 10_000, 500)],
        val_tensor=x_val,
        epochs=4,
        lr=1e-2,
        log_interval=10_000,
        early_stop_delta=0.0,
        surgery=SurgeryConfig(enabled=True, every=2, threshold=0.01,
                              min_count=50.0, warmup_steps=5),
    )

    assert "surgery" in info
    assert info["surgery"]["nll_after"] < info["surgery"]["nll_before"]
    # Every component sits on a planted rank-2 blob.
    assert model.component_ranks.tolist() == [2] * 6
    assert all(torch.isfinite(p).all() for p in model.parameters())


def test_surgery_schedule_respects_enabled_and_every():
    cfg = SurgeryConfig(enabled=True, every=3)
    assert [ep for ep in range(1, 10) if cfg.active_at(ep)] == [3, 6, 9]
    assert not SurgeryConfig(enabled=False, every=1).active_at(1)
    assert not SurgeryConfig(enabled=True, every=0).active_at(1)


def test_train_nll_without_surgery_is_unchanged():
    torch.manual_seed(12)
    batches = [torch.randn(16, 8) for _ in range(4)]
    a = MFA_HDDC(torch.randn(4, 8), rank=2)
    b = MFA_HDDC(torch.randn(4, 8), rank=2)
    b.load_state_dict(a.state_dict())

    train_nll_hddc(a, batches, epochs=2, lr=1e-3, log_interval=1_000)
    train_nll_hddc(b, batches, epochs=2, lr=1e-3, log_interval=1_000, surgery=None)
    for key in a.state_dict():
        assert torch.allclose(a.state_dict()[key], b.state_dict()[key])
