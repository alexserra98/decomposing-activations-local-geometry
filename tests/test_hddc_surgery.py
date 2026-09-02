"""Tests for the rank mask, isotropic Psi, and HDDC covariance surgery.

Coverage:
- an all-ones mask with isotropic Psi reproduces the plain-MFA likelihood, and
  masked columns are exactly zero in W with exactly zero gradient
- the mask and the isotropic Psi shape survive save_mfa/load_mfa and the
  component-sharded save/load path, and pre-mask checkpoints still load
- surgery on a planted low-rank Gaussian recovers Q, lambda, b and d_k, with
  the b_k > 0 and lam_j >= b_k guarantees holding
- shared-b surgery reconciles Cattell rank caps with the common floor without
  the over-pruning caused by dropping all initial violations at once
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
    reconstruct_components,
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
from dalg.models.adaptive_q.train_hddc import (  # noqa: E402
    seed_training_checkpoint,
    train_nll_hddc,
)


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


def test_shared_b_matches_identical_component_noise_and_has_one_gradient():
    torch.manual_seed(101)
    K, D, q = 5, 12, 4
    centroids = torch.randn(K, D)
    x = torch.randn(32, D)

    shared = MFA_HDDC(centroids, rank=q, shared_b=True, psi_init=0.7)
    per_component = MFA_HDDC(
        centroids, rank=q, isotropic_psi=True, psi_init=0.7
    )
    per_component.dir_raw.data.copy_(shared.dir_raw.data)
    per_component.scale_rho.data.copy_(shared.scale_rho.data)

    assert shared.shared_b is True
    assert shared.isotropic_psi is False
    assert tuple(shared.psi_rho.shape) == (1,)
    assert torch.allclose(shared._psi(), per_component._psi())
    assert torch.allclose(shared.nll(x), per_component.nll(x))

    shared.nll(x).backward()
    assert tuple(shared.psi_rho.grad.shape) == (1,)
    assert torch.isfinite(shared.psi_rho.grad).all()


def test_shared_b_is_a_distinct_noise_mode():
    centroids = torch.randn(3, 8)
    with pytest.raises(ValueError, match="distinct noise mode"):
        MFA_HDDC(centroids, rank=2, isotropic_psi=True, shared_b=True)
    with pytest.raises(ValueError, match="distinct noise mode"):
        MFA_HDDC(centroids, rank=2, psi_per_component=True, shared_b=True)


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


def test_inference_cache_agrees_with_uncached_path_for_shared_b():
    torch.manual_seed(102)
    model = MFA_HDDC(torch.randn(6, 16), rank=4, shared_b=True)
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


def test_epoch_zero_checkpoint_restores_initial_model():
    torch.manual_seed(42)
    model = MFA_HDDC(torch.randn(3, 8), rank=2, isotropic_psi=True)
    model.rank_mask[:, 1] = 0.0
    x = torch.randn(20, 8)

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "checkpoint.pt"
        initial_nll = seed_training_checkpoint(
            model,
            str(path),
            lr=1e-3,
            val_tensor=x,
        )
        saved = torch.load(path, map_location="cpu", weights_only=False)
        restored = MFA_HDDC(torch.zeros(3, 8), rank=2, isotropic_psi=True)
        train_nll_hddc(
            restored,
            [],
            val_tensor=x,
            epochs=1,
            ckpt_path=str(path),
        )

    assert saved["epoch"] == 0
    assert saved["best_epoch"] == 0
    assert saved["optimizer"]["state"] == {}
    assert saved["best_metric"] == pytest.approx(initial_nll)
    assert torch.equal(restored.rank_mask, model.rank_mask)
    assert torch.allclose(restored.nll(x), model.nll(x), atol=0.0, rtol=0.0)


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


def test_shared_b_survives_single_file_save_load():
    torch.manual_seed(103)
    model = MFA_HDDC(torch.randn(6, 14), rank=4, shared_b=True)
    model.rank_mask[0, 3] = 0.0
    x = torch.randn(16, 14)

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "mfa.pt"
        save_mfa_hddc(model, str(path))
        blob = torch.load(path, map_location="cpu", weights_only=False)
        loaded = load_mfa_hddc(path)

    assert blob["meta"]["shared_b"] is True
    assert loaded.shared_b is True
    assert loaded.isotropic_psi is False
    assert tuple(loaded.psi_rho.shape) == (1,)
    assert torch.equal(loaded.rank_mask, model.rank_mask)
    assert torch.allclose(loaded.nll(x), model.nll(x))


def test_metadata_free_shared_b_is_inferred_from_shape():
    model = MFA_HDDC(torch.randn(3, 8), rank=2, shared_b=True)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "mfa.pt"
        save_mfa_hddc(model, str(path))
        blob = torch.load(path, map_location="cpu", weights_only=False)
        blob["meta"].pop("shared_b")
        torch.save(blob, path)
        loaded = load_mfa_hddc(path)

    assert loaded.shared_b is True
    assert tuple(loaded.psi_rho.shape) == (1,)


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


def test_zero_min_count_disables_the_membership_cutoff():
    model = MFA_HDDC(torch.zeros(2, 4), rank=2, isotropic_psi=True)
    N = torch.tensor([0.25, 2.0], dtype=torch.float64)
    cfg = SurgeryConfig(enabled=True, every=1, threshold=0.1, min_count=0.0)
    covariances = torch.stack(
        [
            torch.diag(torch.tensor([5.0, 2.0, 1.0, 1.0], dtype=torch.float64)),
            torch.diag(torch.tensor([7.0, 3.0, 1.0, 1.0], dtype=torch.float64)),
        ]
    )

    stats = reconstruct_components(
        model,
        N,
        covariances * N[:, None, None],
        cfg,
    )

    assert cfg.n_min() == 0.0
    assert stats["eligible"].tolist() == [True, True]
    assert stats["n_updated"] == 2
    assert stats["n_skipped"] == 0


def test_zero_min_count_rejects_exactly_zero_soft_membership():
    model = MFA_HDDC(torch.zeros(1, 4), rank=2, isotropic_psi=True)

    with pytest.raises(RuntimeError, match="non-positive effective membership"):
        reconstruct_components(
            model,
            torch.zeros(1, dtype=torch.float64),
            torch.zeros(1, 4, 4, dtype=torch.float64),
            SurgeryConfig(enabled=True, every=1, min_count=0.0),
        )


def test_negative_or_nonfinite_min_count_is_rejected():
    for value in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite and non-negative"):
            SurgeryConfig(min_count=value).n_min()


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


def test_shared_b_surgery_uses_membership_weighted_pooled_residual():
    model = MFA_HDDC(torch.zeros(2, 4), rank=2, shared_b=True, psi_init=0.5)
    N = torch.tensor([100.0, 25.0], dtype=torch.float64)
    covariances = torch.stack(
        [
            torch.diag(torch.tensor([9.0, 4.0, 3.0, 3.0], dtype=torch.float64)),
            torch.diag(torch.tensor([16.0, 9.0, 2.0, 2.0], dtype=torch.float64)),
        ]
    )
    S_acc = covariances * N[:, None, None]

    stats = reconstruct_components(
        model,
        N,
        S_acc,
        SurgeryConfig(enabled=True, every=1, threshold=0.2, min_count=1.0),
    )

    # Cattell selects d=[1, 2]. The pooled floor is
    # (100*(19-9) + 25*(29-25)) / (100*3 + 25*2) = 22/7.
    expected_b = 22.0 / 7.0
    assert stats["d_k"].tolist() == [1, 2]
    assert float(stats["b_shared"]) == pytest.approx(expected_b)
    assert stats["b_k"].tolist() == pytest.approx([expected_b, expected_b])
    assert float(model._psi()[0, 0].detach()) == pytest.approx(expected_b, rel=1e-6)
    assert torch.equal(model._psi()[0], model._psi()[1])


def test_shared_b_active_set_prunes_infeasible_cattell_directions():
    model = MFA_HDDC(torch.zeros(2, 4), rank=3, shared_b=True, psi_init=0.5)
    N = torch.tensor([100.0, 100.0], dtype=torch.float64)
    covariances = torch.stack(
        [
            torch.diag(torch.tensor([20.0, 4.0, 3.0, 2.0], dtype=torch.float64)),
            torch.diag(torch.tensor([100.0, 10.0, 10.0, 10.0], dtype=torch.float64)),
        ]
    )

    stats = reconstruct_components(
        model,
        N,
        covariances * N[:, None, None],
        SurgeryConfig(enabled=True, every=1, threshold=0.04, min_count=1.0),
    )

    # Cattell proposes [3, 1], whose mandatory tails give b=8. Directions with
    # eigenvalues 3 and 4 enter the noise pool in that order, giving final b=6.5.
    assert stats["d_k"].tolist() == [1, 1]
    assert float(stats["b_shared_at_cattell"]) == pytest.approx(8.0)
    assert float(stats["b_shared"]) == pytest.approx(6.5)
    assert stats["n_shared_b_pruned_components"] == 1
    assert stats["n_shared_b_pruned_directions"] == 2
    assert model.rank_mask.tolist() == [[1, 0, 0], [1, 0, 0]]


def test_shared_b_active_set_does_not_batch_prune_a_later_valid_direction():
    model = MFA_HDDC(torch.zeros(2, 4), rank=3, shared_b=True, psi_init=0.5)
    N = torch.tensor([100.0, 100.0], dtype=torch.float64)
    covariances = torch.stack(
        [
            torch.diag(torch.tensor([20.0, 9.0, 1.0, 0.0], dtype=torch.float64)),
            torch.diag(torch.tensor([100.0, 14.0, 13.0, 13.0], dtype=torch.float64)),
        ]
    )

    stats = reconstruct_components(
        model,
        N,
        covariances * N[:, None, None],
        SurgeryConfig(enabled=True, every=1, threshold=0.04, min_count=1.0),
    )

    # Cattell proposes [3, 1] and its pooled floor is 10. Moving lambda=1 into
    # the noise pool lowers b to 8.2, so lambda=9 is valid and must stay active.
    # A simultaneous prune against the initial b would incorrectly return [1, 1].
    assert stats["d_k"].tolist() == [2, 1]
    assert float(stats["b_shared_at_cattell"]) == pytest.approx(10.0)
    assert float(stats["b_shared"]) == pytest.approx(41.0 / 5.0)
    assert stats["n_shared_b_pruned_components"] == 1
    assert stats["n_shared_b_pruned_directions"] == 1
    assert model.rank_mask.tolist() == [[1, 1, 0], [1, 0, 0]]


def test_shared_b_active_set_treats_equality_with_floor_as_noise():
    model = MFA_HDDC(torch.zeros(1, 5), rank=4, shared_b=True, psi_init=0.5)
    N = torch.tensor([100.0], dtype=torch.float64)
    covariance = torch.diag(
        torch.tensor([10.0, 3.0, 2.0, 1.0, 0.5], dtype=torch.float64)
    )[None, :, :]

    stats = reconstruct_components(
        model,
        N,
        covariance * N[:, None, None],
        SurgeryConfig(
            enabled=True,
            every=1,
            threshold=0.04,
            min_count=1.0,
            psi_floor=2.0,
        ),
    )

    # The configured floor binds. lambda=1 and then lambda=2 enter the noise
    # pool; equality is not reported as a zero-variance signal direction.
    assert stats["d_k"].tolist() == [2]
    assert float(stats["b_shared_at_cattell"]) == pytest.approx(2.0)
    assert float(stats["b_shared"]) == pytest.approx(2.0)
    assert stats["n_shared_b_pruned_directions"] == 2


def test_surgery_floor_respects_model_psi_parameterization_floor():
    model = MFA_HDDC(
        torch.zeros(1, 2),
        rank=1,
        shared_b=True,
        psi_init=0.5,
        eps_floor=0.1,
    )
    N = torch.tensor([100.0], dtype=torch.float64)
    covariance = torch.diag(torch.tensor([1.0, 0.0], dtype=torch.float64))[None, :, :]

    stats = reconstruct_components(
        model,
        N,
        covariance * N[:, None, None],
        SurgeryConfig(
            enabled=True,
            every=1,
            threshold=0.01,
            min_count=1.0,
            psi_floor=1e-6,
        ),
    )

    written_b = float(model._psi()[0, 0].detach())
    assert float(stats["b_shared"]) > model._eps
    assert written_b == pytest.approx(float(stats["b_shared"]), abs=1e-6)


def test_component_specific_surgery_rejects_floor_above_retained_eigenvalue():
    model = MFA_HDDC(
        torch.zeros(1, 2),
        rank=1,
        isotropic_psi=True,
        psi_init=0.5,
        eps_floor=0.1,
    )
    before = {key: value.clone() for key, value in model.state_dict().items()}
    N = torch.tensor([100.0], dtype=torch.float64)
    covariance = torch.diag(torch.tensor([0.05, 0.0], dtype=torch.float64))[None, :, :]

    with pytest.raises(
        RuntimeError,
        match=r"component-b surgery.*component=0, direction=1, lambda=.* <= b=",
    ):
        reconstruct_components(
            model,
            N,
            covariance * N[:, None, None],
            SurgeryConfig(
                enabled=True,
                every=1,
                threshold=0.01,
                min_count=1.0,
                psi_floor=1e-6,
            ),
        )
    for key, value in model.state_dict().items():
        assert torch.equal(value, before[key])


def test_hddc_surgery_reports_shared_b_without_dropping_b_k_mean():
    x, mu, _U, _lam = _planted_gaussian(
        D=16, d_true=2, b_true=0.05, n=20_000, seed=104
    )
    model = MFA_HDDC(mu[None, :], rank=4, shared_b=True)
    summary = hddc_surgery(
        model,
        _batches(x),
        SurgeryConfig(enabled=True, every=1, threshold=0.01, min_count=10.0),
    )

    assert summary["b_shared"] == pytest.approx(0.05, rel=0.1)
    assert summary["b_k_mean"] == pytest.approx(summary["b_shared"])
    assert summary["b_shared_at_cattell"] == pytest.approx(summary["b_shared"])
    assert summary["n_shared_b_pruned_components"] == 0
    assert summary["n_shared_b_pruned_directions"] == 0


def test_shared_b_surgery_still_rejects_mandatory_first_direction_below_floor():
    model = MFA_HDDC(torch.zeros(2, 4), rank=3, shared_b=True)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    N = torch.tensor([100.0, 100.0], dtype=torch.float64)
    covariances = torch.stack(
        [
            torch.diag(torch.tensor([5.0, 4.0, 3.0, 2.0], dtype=torch.float64)),
            torch.diag(torch.tensor([100.0, 50.0, 50.0, 50.0], dtype=torch.float64)),
        ]
    )
    S_acc = covariances * N[:, None, None]

    with pytest.raises(
        RuntimeError,
        match=r"component=0, direction=1, lambda=.* <= b=",
    ):
        reconstruct_components(
            model,
            N,
            S_acc,
            SurgeryConfig(enabled=True, every=1, threshold=0.1, min_count=1.0),
        )
    for key, value in model.state_dict().items():
        assert torch.equal(value, before[key])


def test_shared_b_surgery_with_no_eligible_components_is_a_no_op():
    model = MFA_HDDC(torch.zeros(2, 4), rank=2, shared_b=True)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    stats = reconstruct_components(
        model,
        torch.tensor([1.0, 2.0], dtype=torch.float64),
        torch.zeros(2, 4, 4, dtype=torch.float64),
        SurgeryConfig(enabled=True, every=1, min_count=10.0),
    )

    assert stats["b_shared"] is None
    assert stats["b_shared_at_cattell"] is None
    assert stats["n_shared_b_pruned_components"] == 0
    assert stats["n_shared_b_pruned_directions"] == 0
    assert stats["n_updated"] == 0
    for key, value in model.state_dict().items():
        assert torch.equal(value, before[key])


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


def test_parameter_count_counts_one_shared_b_parameter():
    K, D, q = 4, 20, 5
    shared = MFA_HDDC(torch.zeros(K, D), rank=q, shared_b=True)
    per_component = MFA_HDDC(torch.zeros(K, D), rank=q, isotropic_psi=True)
    assert parameter_count(shared) == parameter_count(per_component) - (K - 1)


# --------------------------------------------------------------------------
# Phase C and the training-loop hook
# --------------------------------------------------------------------------


def test_fractional_epoch_schedule_tracks_global_progress():
    half = SurgeryConfig(enabled=True, every=0.5)
    thirds = SurgeryConfig(enabled=True, every=0.3)
    integer = SurgeryConfig(enabled=True, every=1)

    assert [
        step for step in range(1, 10) if half.active_after_batch(step, 9)
    ] == [5]
    assert half.active_at(1)
    assert [
        [
            step
            for step in range(1, 10)
            if thirds.active_after_batch(step, 9, epoch=epoch)
        ]
        for epoch in range(1, 4)
    ] == [[3, 6, 9], [2, 5, 8], [1, 4, 7]]
    assert not thirds.active_at(1)
    assert not thirds.active_at(2)
    assert thirds.active_at(3)
    assert not integer.active_after_batch(5, 9)
    assert integer.active_at(1)


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


def test_train_nll_runs_half_epoch_surgery_twice(monkeypatch):
    import dalg.models.adaptive_q.hddc_surgery as surgery_module

    torch.manual_seed(12)
    batches = [torch.randn(8, 4) for _ in range(4)]
    model = MFA_HDDC(torch.randn(2, 4), rank=1, isotropic_psi=True)
    calls = []

    def fake_surgery(model, loader, cfg, *, device=None, log=None):
        calls.append(sum(batch.shape[0] for batch in loader))
        return {
            "d_k_hist": [0, model.K],
            "d_k_per_component": [1] * model.K,
        }

    monkeypatch.setattr(surgery_module, "hddc_surgery", fake_surgery)
    train_nll_hddc(
        model,
        batches,
        surgery_loader=batches,
        val_tensor=batches[0],
        epochs=1,
        steps_per_epoch=4,
        early_stop_delta=0.0,
        log_interval=10_000,
        surgery=SurgeryConfig(enabled=True, every=0.5),
    )

    assert calls == [32, 32]


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
