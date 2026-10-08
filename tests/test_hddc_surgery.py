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


@pytest.mark.parametrize("all_spherical", [False, True])
def test_spherical_components_match_dense_likelihoods_and_have_zero_loading_gradients(all_spherical):
    torch.manual_seed(105)
    model = MFA_HDDC(torch.randn(2, 4), rank=2, shared_b=True, psi_init=0.7)
    model.rank_mask[0].zero_()
    if all_spherical:
        model.rank_mask.zero_()
    x = torch.randn(12, 4)
    W = model._W().detach().double()
    covariance = W @ W.transpose(-1, -2) + torch.diag_embed(model._psi().detach().double())
    reference = torch.distributions.MultivariateNormal(
        model.mu.detach().double(), covariance_matrix=covariance,
    ).log_prob(x.double()[:, None])

    torch.testing.assert_close(model.log_prob_components(x).double(), reference, atol=2e-5, rtol=2e-5)
    for dtype, chunk in [(None, None), (torch.float64, 1)]:
        with torch.no_grad(), model.inference_cache(dtype=dtype, component_chunk_size=chunk):
            torch.testing.assert_close(model.log_prob_components(x).double(), reference, atol=2e-5, rtol=2e-5)
            assert torch.isfinite(model.responsibilities(x)).all()

    spherical = model.component_ranks == 0
    assert torch.count_nonzero(model._W()[spherical]) == 0
    model.nll(x).backward()
    assert torch.count_nonzero(model.dir_raw.grad[spherical]) == 0
    assert torch.count_nonzero(model.scale_rho.grad[spherical]) == 0
    assert torch.isfinite(model.mu.grad).all()
    assert torch.isfinite(model.pi_logits.grad).all()
    assert torch.isfinite(model.psi_rho.grad).all()


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
    model.rank_mask[1].zero_()
    x = torch.randn(16, 14)

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "mfa.pt"
        save_mfa_hddc(model, str(path))
        blob = torch.load(path, map_location="cpu", weights_only=False)
        loaded = load_mfa_hddc(path)

    assert blob["meta"]["shared_b"] is True
    assert loaded.shared_b is True
    assert loaded.component_ranks.tolist() == model.component_ranks.tolist()
    assert torch.count_nonzero(loaded._W()[1]) == 0
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
        torch.zeros_like(model.mu, dtype=torch.float64),
        covariances * N[:, None, None],
        cfg,
    )

    assert cfg.n_min() == 0.0
    assert stats["eligible"].tolist() == [True, True]
    assert stats["n_updated"] == 2
    assert stats["n_skipped"] == 0


@pytest.mark.parametrize("min_count", [0.0, 10.0])
def test_zero_soft_membership_is_always_skipped(min_count):
    model = MFA_HDDC(torch.zeros(1, 4), rank=2, isotropic_psi=True)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    stats = reconstruct_components(
        model,
        torch.zeros(1, dtype=torch.float64),
        torch.zeros_like(model.mu, dtype=torch.float64),
        torch.zeros(1, 4, 4, dtype=torch.float64),
        SurgeryConfig(enabled=True, every=1, min_count=min_count),
    )
    assert stats["n_updated"] == 0
    for key, value in model.state_dict().items():
        assert torch.equal(value, before[key])


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
        torch.zeros_like(model.mu, dtype=torch.float64),
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
        torch.zeros_like(model.mu, dtype=torch.float64),
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
        torch.zeros_like(model.mu, dtype=torch.float64),
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
        torch.zeros_like(model.mu, dtype=torch.float64),
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
        torch.zeros_like(model.mu, dtype=torch.float64),
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


@pytest.mark.parametrize("spectrum, expected_rank", [
    ([0.05, 0., 0., 0.], 0),
    ([2., 0.05, 0.02, 0.], 1),
    ([2., 0.5, 0.2, 0.15], 3),
    ([0., 0., 0., 0.], 0),
    ([0.5, 0.5, 0.5, 0.5], 0),
])
def test_component_specific_surgery_corrects_rank_against_stored_noise(spectrum, expected_rank):
    model = MFA_HDDC(torch.zeros(1, 4), rank=3, isotropic_psi=True, eps_floor=0.1)
    N = torch.tensor([100.], dtype=torch.float64)
    covariance = torch.diag(torch.tensor(spectrum, dtype=torch.float64))[None]
    stats = reconstruct_components(
        model, N, torch.zeros_like(model.mu, dtype=torch.float64),
        covariance * N[:, None, None], SurgeryConfig(threshold=1.0),
    )
    assert model.component_ranks.tolist() == [expected_rank]
    assert stats["n_pruned_directions"] == 3 - expected_rank
    expected_b = max(sum(spectrum[expected_rank:]) / (4 - expected_rank), 0.1)
    stored_b = model._psi()[0, 0].detach().double()
    assert float(stored_b) == pytest.approx(expected_b)
    torch.testing.assert_close(stats["b_k"][0], stored_b, rtol=0, atol=0)
    expected_spectrum = torch.tensor(spectrum, dtype=torch.float64)
    expected_spectrum[expected_rank:] = stored_b
    W = model._W()[0].detach().double()
    actual = W @ W.T + stored_b * torch.eye(4)
    torch.testing.assert_close(actual, torch.diag(expected_spectrum), rtol=1e-6, atol=1e-7)
    assert torch.isfinite(model.log_prob(torch.zeros(1, 4))).all()


def test_component_specific_noise_rounding_and_components_are_independent():
    model = MFA_HDDC(torch.zeros(2, 2), rank=1, isotropic_psi=True)
    N = torch.tensor([100., 50.], dtype=torch.float64)
    spectra = torch.tensor([[1.0 - 2**-27, 1.0 - 2**-26], [4., 0.2]], dtype=torch.float64)
    stats = reconstruct_components(
        model, N, torch.zeros(2, 2, dtype=torch.float64),
        torch.diag_embed(spectra) * N[:, None, None], SurgeryConfig(),
    )
    assert model.component_ranks.tolist() == [0, 1]
    torch.testing.assert_close(model._psi()[:, 0].detach(), torch.tensor([1., 0.2]))
    torch.testing.assert_close(stats["b_k"], model._psi()[:, 0].detach().double(), rtol=0, atol=0)
    assert stats["n_pruned_directions"] == 1



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


def test_shared_b_surgery_keeps_component_with_first_direction_below_floor():
    model = MFA_HDDC(torch.zeros(2, 4), rank=3, shared_b=True)
    N = torch.tensor([100.0, 50.0], dtype=torch.float64)
    covariances = torch.stack(
        [
            torch.diag(torch.tensor([5.0, 4.0, 3.0, 2.0], dtype=torch.float64)),
            torch.diag(torch.tensor([100.0, 50.0, 50.0, 50.0], dtype=torch.float64)),
        ]
    )
    # Both means shift by one; covariance updates must still update the weights.
    A = N[:, None] * torch.ones(2, 4, dtype=torch.float64)
    S_acc = (covariances + 1.0) * N[:, None, None]
    stats = reconstruct_components(
        model, N, A, S_acc,
        SurgeryConfig(enabled=True, every=1, threshold=0.1, min_count=1.0),
    )

    assert model.K == 2 and model.q == 3
    assert model.component_ranks.tolist() == [0, 1]
    assert stats["n_updated"] == 2
    assert stats["n_shared_b_pruned_components"] == 1
    assert stats["n_shared_b_pruned_directions"] == 3
    # The first component contributes its full trace and four noise dimensions.
    assert float(stats["b_shared"]) == pytest.approx(178.0 / 11.0)
    assert float(stats["b_shared"]) == float(model._psi()[0, 0].detach())
    torch.testing.assert_close(model.mu, torch.ones(2, 4))
    torch.testing.assert_close(model.pi_logits.softmax(0), (N / N.sum()).float())
    assert torch.count_nonzero(model._W()[0]) == 0
    W = model._W().detach().double()
    covariance = W @ W.transpose(-1, -2) + torch.diag_embed(model._psi().detach().double())
    b = float(stats["b_shared"])
    torch.testing.assert_close(covariance[0], b * torch.eye(4, dtype=torch.float64))
    torch.testing.assert_close(
        covariance[1], torch.diag(torch.tensor([100., b, b, b], dtype=torch.float64)),
        rtol=2e-6, atol=2e-6,
    )


def test_pooling_first_direction_preserves_another_components_signal():
    model = MFA_HDDC(torch.zeros(3, 2), rank=1, shared_b=True)
    N = torch.ones(3, dtype=torch.float64)
    spectra = torch.tensor([[2., 1.], [9., 8.], [100., 21.]], dtype=torch.float64)
    stats = reconstruct_components(
        model, N, torch.zeros(3, 2, dtype=torch.float64),
        torch.diag_embed(spectra), SurgeryConfig(),
    )
    # The pooled tail starts at 10. Adding lambda=2 lowers it to 8, saving 9.
    assert float(stats["b_shared_at_cattell"]) == pytest.approx(10.0)
    assert float(stats["b_shared"]) == pytest.approx(8.0)
    assert model.component_ranks.tolist() == [0, 1, 1]
    assert stats["n_shared_b_pruned_directions"] == 1


@pytest.mark.parametrize("q", [1, 3])
@pytest.mark.parametrize("variance, floor", [(2.0, 1e-6), (0.0, 0.1)])
def test_shared_b_can_make_every_component_spherical(q, variance, floor):
    model = MFA_HDDC(torch.zeros(2, 4), rank=q, shared_b=True)
    N = torch.tensor([100., 50.], dtype=torch.float64)
    stats = reconstruct_components(
        model, N, torch.zeros(2, 4, dtype=torch.float64),
        variance * torch.eye(4, dtype=torch.float64)[None] * N[:, None, None],
        SurgeryConfig(psi_floor=floor),
    )
    assert model.component_ranks.tolist() == [0, 0]
    assert torch.count_nonzero(model._W()) == 0
    assert float(stats["b_shared"]) == pytest.approx(max(variance, floor))
    assert float(stats["b_shared"]) == float(model._psi()[0, 0].detach())
    assert stats["n_shared_b_pruned_directions"] == 2 * q  # Both no-gap proposals start at q_max.
    assert torch.isfinite(model.log_prob(torch.zeros(1, 4))).all()
    assert all(torch.isfinite(p).all() for p in model.parameters())


def test_shared_b_pools_directions_invalidated_by_float32_rounding():
    model = MFA_HDDC(torch.zeros(2, 2), rank=1, shared_b=True)
    N = torch.tensor([100., 50.], dtype=torch.float64)
    tail = 1.0 - 2**-26
    leading = 1.0 - 2**-27
    covariance = torch.diag(torch.tensor([leading, tail], dtype=torch.float64))
    stats = reconstruct_components(
        model, N, torch.zeros(2, 2, dtype=torch.float64),
        covariance[None] * N[:, None, None], SurgeryConfig(),
    )
    # Both directions clear the float64 pooled floor, but its encoding is 1.0.
    assert float(stats["b_shared_at_cattell"]) < leading < float(model._psi()[0, 0].detach())
    assert model.component_ranks.tolist() == [0, 0]
    assert float(stats["b_shared"]) == 1.0
    assert stats["n_shared_b_pruned_components"] == 2
    assert stats["n_shared_b_pruned_directions"] == 2
    assert torch.count_nonzero(model._W()) == 0


def test_spherical_surgery_reports_zero_rank_and_parameter_count():
    model = MFA_HDDC(torch.zeros(2, 4), rank=2, shared_b=True)
    summary = hddc_surgery(model, [torch.zeros(10, 4)], SurgeryConfig())
    assert summary["n_components"] == 2
    assert summary["n_updated"] == 2
    assert summary["d_k_hist"] == [2, 0, 0]
    assert summary["d_k_min"] == summary["d_k_max"] == 0
    assert summary["n_shared_b_pruned_directions"] == 4
    assert parameter_count(model) == 2 * 4 + 1 + 1  # Means, weights, shared noise.


def test_shared_b_surgery_with_no_eligible_components_is_a_no_op():
    model = MFA_HDDC(torch.zeros(2, 4), rank=2, shared_b=True)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    stats = reconstruct_components(
        model,
        torch.tensor([1.0, 2.0], dtype=torch.float64),
        torch.zeros_like(model.mu, dtype=torch.float64),
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


@pytest.mark.parametrize("shared_b", [False, True])
@pytest.mark.parametrize("eig_batch_size", [None, 1])
@pytest.mark.parametrize(
    "eigenvalues, threshold, expected_rank",
    [
        ([8., 4., 2., 1.], 0.6, 3),  # No gap passes: retain q_max when feasible.
        ([8., 4., 2., 1.], 0.5, 3),  # Equality does not pass either.
        ([8., 4., 2., 1.], 0.25, 1),
        ([8., 4., 2., 1.], 0.125, 2),
        ([8., 4., 3., 1.], 0.2, 3),  # Last passing gap, despite an earlier small gap.
    ],
)
def test_cattell_rank_proposal_and_no_gap_fallback(
    shared_b, eig_batch_size, eigenvalues, threshold, expected_rank,
):
    model = MFA_HDDC(
        torch.zeros(1, 4), rank=3, shared_b=shared_b, isotropic_psi=not shared_b,
    )
    N = torch.tensor([100.], dtype=torch.float64)
    covariance = torch.diag(torch.tensor(eigenvalues, dtype=torch.float64))
    stats = reconstruct_components(
        model, N, torch.zeros(1, 4, dtype=torch.float64), covariance[None] * N[:, None, None],
        SurgeryConfig(threshold=threshold, eig_batch_size=eig_batch_size),
    )
    assert model.component_ranks.tolist() == [expected_rank]
    assert stats["n_shared_b_pruned_directions"] == 0
    expected_b = sum(eigenvalues[expected_rank:]) / (4 - expected_rank)
    assert float(model._psi()[0, 0].detach()) == pytest.approx(expected_b)


def test_no_gap_fallback_is_per_component():
    model = MFA_HDDC(torch.zeros(2, 4), rank=3, isotropic_psi=True)
    N = torch.tensor([100., 50.], dtype=torch.float64)
    covariance = torch.diag_embed(torch.tensor(
        [[8., 7., 6., 5.], [8., 6., 2., 1.]], dtype=torch.float64,
    ))
    reconstruct_components(
        model, N, torch.zeros(2, 4, dtype=torch.float64), covariance * N[:, None, None],
        SurgeryConfig(threshold=0.25),
    )
    assert model.component_ranks.tolist() == [3, 2]


def test_shared_b_prunes_no_gap_q_max_proposals():
    model = MFA_HDDC(torch.zeros(2, 4), rank=3, shared_b=True)
    N = torch.tensor([100., 100.], dtype=torch.float64)
    covariance = torch.diag_embed(torch.tensor(
        [[20., 4., 3., 2.], [100., 10., 10., 10.]], dtype=torch.float64,
    ))
    stats = reconstruct_components(
        model, N, torch.zeros(2, 4, dtype=torch.float64), covariance * N[:, None, None],
        SurgeryConfig(threshold=1.0),
    )
    # Both caps are 3. Pooling lambda=3 then lambda=4 lowers b from 6 to 4.75.
    assert float(stats["b_shared_at_cattell"]) == pytest.approx(6.)
    assert float(stats["b_shared"]) == pytest.approx(4.75)
    assert model.component_ranks.tolist() == [1, 3]
    assert stats["n_shared_b_pruned_directions"] == 2
    assert stats["n_shared_b_pruned_components"] == 1


def test_higher_threshold_selects_a_smaller_rank_when_gaps_qualify():
    x, mu, _U, _lam = _planted_gaussian(D=32, d_true=3, b_true=0.02)
    ranks = {}
    for t in (0.01, 0.4):
        model = MFA_HDDC(mu[None, :].clone(), rank=8, isotropic_psi=True, psi_init=0.5)
        summary = hddc_surgery(
            model,
            _batches(x),
            SurgeryConfig(enabled=True, every=1, threshold=t, min_count=10.0),
        )
        ranks[t] = summary["d_k_per_component"][0]
    # lam = [4, 2, 1, b, ...]: the leading gap comfortably clears t = 0.4.
    assert ranks[0.4] == 1
    assert ranks[0.01] == 3


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
        SurgeryConfig(enabled=True, every=1, threshold=0.4, min_count=10.0),
    )
    loose = hddc_surgery(
        model, batches,
        SurgeryConfig(enabled=True, every=1, threshold=0.01, min_count=10.0),
    )
    assert loose["d_k_per_component"][0] > tight["d_k_per_component"][0]
    assert loose["d_k_per_component"][0] == 3


def test_residual_statistics_recover_covariance_about_the_updated_mean():
    """The first-moment correction removes displacement of the initial mean."""
    x, mu, _U, _lam = _planted_gaussian(D=16, d_true=2, b_true=0.05, n=20_000)
    shift = torch.zeros(16)
    shift[0] = 1.0

    on_mean = MFA_HDDC(mu[None, :].clone(), rank=4, isotropic_psi=True, psi_init=0.5)
    off_mean = MFA_HDDC((mu + shift)[None, :].clone(), rank=4, isotropic_psi=True,
                   psi_init=0.5)

    N_a, A_a, B_a, rows = accumulate_statistics(on_mean, _batches(x), device=x.device)
    N_b, A_b, B_b, _ = accumulate_statistics(off_mean, _batches(x), device=x.device)

    assert rows == x.shape[0]
    assert float(N_a.sum()) == pytest.approx(x.shape[0], rel=1e-6)
    S_a = B_a[0] / N_a[0] - torch.outer(A_a[0], A_a[0]) / N_a[0] ** 2
    S_b = B_b[0] / N_b[0] - torch.outer(A_b[0], A_b[0]) / N_b[0] ** 2
    torch.testing.assert_close(S_a, S_b, atol=1e-10, rtol=1e-10)
    centered = x.double() - x.double().mean(0)
    torch.testing.assert_close(S_b, centered.T @ centered / len(x))
    cfg = SurgeryConfig(threshold=0.01)
    reconstruct_components(on_mean, N_a, A_a, B_a, cfg)
    reconstruct_components(off_mean, N_b, A_b, B_b, cfg)
    torch.testing.assert_close(off_mean.mu, x.mean(0)[None])
    torch.testing.assert_close(off_mean.W, on_mean.W)


def test_surgery_requires_isotropic_psi():
    x, mu, _U, _lam = _planted_gaussian(D=16, d_true=2, n=2_000)
    model = MFA_HDDC(mu[None, :].clone(), rank=4)
    with pytest.raises(ValueError, match="isotropic_psi"):
        hddc_surgery(model, _batches(x), SurgeryConfig(enabled=True, every=1))


@pytest.mark.parametrize("shared_b", [False, True])
@pytest.mark.parametrize("hard", [False, True])
def test_m_step_matches_frozen_responsibilities_and_dense_covariance(shared_b, hard, tmp_path):
    torch.manual_seed(93)
    x = torch.randn(150, 4) * torch.tensor([3.0, 1.5, 0.3, 0.2]) + 2.0
    model = MFA_HDDC(
        torch.randn(3, 4), rank=1, shared_b=shared_b,
        isotropic_psi=not shared_b, psi_init=4.0,
    )
    if hard:
        # Give every component hard support; empty clusters have a separate test.
        with torch.no_grad():
            model.mu.fill_(2.0)
            model.mu[:, 0].copy_(torch.tensor([-3.0, 2.0, 7.0]))
    batches = _batches(x, size=17)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    with torch.no_grad():
        r = torch.cat([model.responsibilities(batch) for batch in batches]).double()
    if hard:
        r = torch.nn.functional.one_hot(r.argmax(1), model.K).double()
    counts = r.sum(0)
    assert (counts > 0).all()
    means = r.T @ x.double() / counts[:, None]
    residual = x.double()[:, None, :] - means[None]
    covariance = torch.einsum("nk,nkd,nke->kde", r, residual, residual) / counts[:, None, None]

    N, A, B, rows = accumulate_statistics(
        model, batches, device=x.device, chunk_elems=3 * model.K * model.D,
        hard_assignment_covariance=hard,
    )
    assert rows == len(x) and N.dtype == A.dtype == B.dtype == torch.float64
    for key, value in model.state_dict().items():
        assert torch.equal(value, before[key])
    torch.testing.assert_close(N, counts)
    shifts = A / N[:, None]
    torch.testing.assert_close(
        B / N[:, None, None] - shifts[:, :, None] * shifts[:, None, :], covariance,
    )
    hddc_surgery(model, batches, SurgeryConfig(threshold=0.01, hard_assignment_covariance=hard))
    torch.testing.assert_close(model.mu.double(), means, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(model.pi_logits.softmax(0).double(), counts / counts.sum())

    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    noise = eigenvalues[:, :-1].mean(-1)
    if shared_b:
        noise = ((counts * noise).sum() / counts.sum()).expand_as(noise)
    principal = eigenvectors[:, :, -1]
    expected = (
        (eigenvalues[:, -1] - noise)[:, None, None]
        * principal[:, :, None] * principal[:, None, :]
        + noise[:, None, None] * torch.eye(model.D)
    )
    actual = model.W @ model.W.transpose(-1, -2) + torch.diag_embed(model._psi())
    torch.testing.assert_close(actual.double(), expected, atol=3e-6, rtol=3e-6)
    path = tmp_path / "m_step.pt"
    save_mfa_hddc(model, str(path))
    restored = load_mfa_hddc(str(path))
    for key, value in model.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key])


@pytest.mark.parametrize("shared_b", [False, True])
def test_cutoff_preserves_skipped_means_logits_and_probabilities(shared_b):
    model = MFA_HDDC(
        torch.zeros(4, 4), rank=1, shared_b=shared_b, isotropic_psi=not shared_b,
    )
    with torch.no_grad():
        model.pi_logits.copy_(torch.tensor([0.4, 0.2, 0.1, 0.3]).log() + 30.0)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    old_pi = model.pi_logits.softmax(0).detach()
    N = torch.tensor([60.0, 20.0, 2.0, 0.0], dtype=torch.float64)
    means = torch.arange(16, dtype=torch.float64).reshape(4, 4) / 4
    covariance = torch.diag(torch.tensor([9.0, 1.0, 1.0, 1.0], dtype=torch.float64))
    B = N[:, None, None] * (covariance + means[:, :, None] * means[:, None, :])
    stats = reconstruct_components(
        model, N, N[:, None] * means, B, SurgeryConfig(min_count=20.0),
    )
    assert stats["eligible"].tolist() == [True, True, False, False]
    torch.testing.assert_close(model.mu[:2].double(), means[:2])
    for name in ("mu", "pi_logits", "dir_raw", "scale_rho", "rank_mask"):
        assert torch.equal(getattr(model, name)[2:], before[name][2:])
    pi = model.pi_logits.softmax(0).detach()
    torch.testing.assert_close(pi[2:], old_pi[2:])
    torch.testing.assert_close(pi[:2], old_pi[:2].sum() * torch.tensor([0.75, 0.25]))
    if not shared_b:
        assert torch.equal(model.psi_rho[2:], before["psi_rho"][2:])


def test_component_without_hard_assignments_is_updated_from_positive_soft_mass():
    torch.manual_seed(43)
    model = MFA_HDDC(torch.zeros(2, 4), rank=1, isotropic_psi=True)
    with torch.no_grad():
        model.dir_raw[1].copy_(model.dir_raw[0])
        model.pi_logits.copy_(torch.tensor([0.9, 0.1]).log())
    x = torch.randn(100, 4) * torch.tensor([3.0, 0.4, 0.3, 0.2]) + 1
    assert (model.responsibilities(x).argmax(-1) == 0).all()
    summary = hddc_surgery(model, _batches(x, 11), SurgeryConfig(min_count=0.0))
    assert summary["n_updated"] == 2
    torch.testing.assert_close(model.mu, x.mean(0).expand(2, -1))


@pytest.mark.parametrize("bad_stat", ["negative_count", "count", "residual", "scatter"])
def test_invalid_statistics_fail_before_mutation(bad_stat):
    model = MFA_HDDC(torch.zeros(2, 4), rank=1, isotropic_psi=True)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    N = torch.ones(2, dtype=torch.float64)
    A = torch.zeros(2, 4, dtype=torch.float64)
    B = torch.eye(4, dtype=torch.float64).expand(2, -1, -1).clone()
    if bad_stat == "negative_count":
        N[0] = -1
    elif bad_stat == "count":
        N[0] = torch.nan
    elif bad_stat == "residual":
        A[0, 0] = torch.inf
    else:
        B[0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        reconstruct_components(model, N, A, B, SurgeryConfig())
    for key, value in model.state_dict().items():
        assert torch.equal(value, before[key])


def test_empty_e_pass_is_rejected():
    model = MFA_HDDC(torch.zeros(1, 4), rank=1, isotropic_psi=True)
    with pytest.raises(ValueError, match="non-empty E-pass"):
        hddc_surgery(model, [], SurgeryConfig())
    assert model.training


def test_surgery_invalidates_active_inference_cache():
    torch.manual_seed(41)
    model = MFA_HDDC(torch.zeros(1, 4), rank=1, isotropic_psi=True)
    x = torch.randn(100, 4) * torch.tensor([3.0, 0.4, 0.3, 0.2]) + 2
    with torch.no_grad(), model.inference_cache():
        old_ll = model.log_prob(x)
        hddc_surgery(model, _batches(x, 15), SurgeryConfig())
        assert model._inference_cache is None
        new_ll = model.log_prob(x)
        assert not torch.allclose(old_ll, new_ll)
    with torch.no_grad(), model.inference_cache():
        torch.testing.assert_close(model.log_prob(x), new_ll, atol=2e-5, rtol=2e-5)


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
    assert dropped == 5
    assert not opt.state


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
            "n_updated": model.K,
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


def test_all_skipped_surgery_preserves_optimizer_and_training_trajectory(tmp_path):
    torch.manual_seed(27)
    batches = [torch.randn(12, 4) for _ in range(4)]
    a = MFA_HDDC(torch.randn(2, 4), rank=1, isotropic_psi=True)
    b = MFA_HDDC(torch.zeros(2, 4), rank=1, isotropic_psi=True)
    b.load_state_dict(a.state_dict())
    for model, name, surgery in (
        (a, "baseline", None),
        (b, "skipped", SurgeryConfig(
            enabled=True, every=0.5, min_count=1e9, warmup_steps=5,
        )),
    ):
        train_nll_hddc(
            model, batches, epochs=2, steps_per_epoch=4, track_best=False,
            early_stop_delta=0.0, log_interval=10_000, surgery=surgery,
            ckpt_path=str(tmp_path / f"{name}.pt"),
        )
    for key, value in a.state_dict().items():
        assert torch.equal(value, b.state_dict()[key])
    baseline = torch.load(tmp_path / "baseline.pt", weights_only=False)["optimizer"]
    skipped = torch.load(tmp_path / "skipped.pt", weights_only=False)["optimizer"]
    assert baseline["param_groups"] == skipped["param_groups"]
    for param_id, state in baseline["state"].items():
        for key, value in state.items():
            assert torch.equal(value, skipped["state"][param_id][key])


@pytest.mark.parametrize("shared_b", [False, True])
@pytest.mark.parametrize("min_count", [0.0, 20.0])
def test_hard_surgery_skips_empty_and_small_clusters(shared_b, min_count):
    torch.manual_seed(321)
    centers = torch.zeros(3, 4)
    centers[:, 0] = torch.tensor([0., 20., 40.])
    model = MFA_HDDC(centers, rank=1, shared_b=shared_b, isotropic_psi=not shared_b)
    x = torch.cat([
        centers[k] + torch.randn(n, 4) * torch.tensor([1.5, .3, .2, .1])
        for k, n in enumerate([80, 12])
    ])
    winners = model.responsibilities(x).argmax(1)
    assert torch.bincount(winners, minlength=3).tolist() == [80, 12, 0]
    before = {key: value.clone() for key, value in model.state_dict().items()}
    old_pi = model.pi_logits.softmax(0).detach()
    updated = 2 if min_count == 0 else 1
    summary = hddc_surgery(
        model, _batches(x, 13),
        SurgeryConfig(min_count=min_count, hard_assignment_covariance=True),
    )
    assert summary["n_updated"] == updated
    assert summary["n_skipped"] == 3 - updated
    for name in ("mu", "pi_logits", "dir_raw", "scale_rho", "rank_mask"):
        assert torch.equal(getattr(model, name)[updated:], before[name][updated:])
    if not shared_b:
        assert torch.equal(model.psi_rho[updated:], before["psi_rho"][updated:])
    torch.testing.assert_close(model.pi_logits.softmax(0)[updated:], old_pi[updated:])


def _hard_surgery_shard_worker(rank, rendezvous):
    from datetime import timedelta
    import torch.distributed as dist
    from dalg.models.adaptive_q.mfa_hddc import component_shard_bounds

    torch.set_num_threads(1)
    torch.manual_seed(732)
    centers = torch.zeros(5, 4)
    centers[:, 0] = torch.tensor([-12., -4., 4., -12., 12.])
    full = MFA_HDDC(centers, rank=1, isotropic_psi=True)
    with torch.no_grad():
        full.dir_raw[3].copy_(full.dir_raw[0])
    x = torch.cat([centers[k] + torch.randn(17, 4) * .3 for k in [0, 1, 2, 4]])
    winners = full.responsibilities(x).argmax(1)
    assert torch.bincount(winners, minlength=5).tolist() == [17, 17, 17, 0, 17]
    start, end = component_shard_bounds(full.K, rank, 2)
    shard = ComponentShardedMFA_HDDC(
        centers[start:end], rank=1, isotropic_psi=True,
        global_K=full.K, component_start=start,
    )
    shard.load_state_dict({key: value[start:end] for key, value in full.state_dict().items()})
    dist.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=2,
        timeout=timedelta(seconds=60),
    )
    try:
        for hard in [False, True]:
            expected = accumulate_statistics(
                full, _batches(x, 11), device="cpu", hard_assignment_covariance=hard,
            )
            actual = accumulate_statistics(
                shard, _batches(x, 11), device="cpu", hard_assignment_covariance=hard,
            )
            for reference, local in zip(expected[:3], actual[:3]):
                torch.testing.assert_close(local, reference[start:end], atol=2e-6, rtol=2e-6)
            assert actual[3] == expected[3] == len(x)
            if hard:
                assert actual[0].tolist() == [17, 17, 17, 0, 17][start:end]
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_gloo_available(), reason="requires Gloo")
def test_hard_surgery_global_argmax_matches_full_model_with_cross_shard_ties(tmp_path):
    torch.multiprocessing.spawn(
        _hard_surgery_shard_worker,
        args=((tmp_path / "gloo_init").as_uri(),), nprocs=2, join=True,
    )


def test_explicit_soft_statistics_match_default():
    torch.manual_seed(730)
    model = MFA_HDDC(torch.randn(3, 4), rank=1, isotropic_psi=True)
    batches = _batches(torch.randn(21, 4), 8)
    default = accumulate_statistics(model, batches, device="cpu")
    explicit = accumulate_statistics(model, batches, device="cpu", hard_assignment_covariance=False)
    for left, right in zip(default[:3], explicit[:3]):
        assert torch.equal(left, right)
    assert default[3] == explicit[3]
