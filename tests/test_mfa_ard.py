"""Unit tests for the ARD-regularized MFA in src/dalg/models/mfa_ard.py.

Coverage:
- the closed-form nu makes `ard_penalty` equal the profiled penalty
  c*log(1/2 s^2 + b0) + c*(1 - log c)
- detaching nu is exact: its gradient equals that of the profiled penalty
- hyperparameter validation (b0 > 0, D/2 + alpha0 - 1 > 0)
- ARD shrinks unused loading columns, so effective rank drops below --rank
- save/load round-trip, and plain `load_mfa` compatibility (downstream analysis
  reads ARD checkpoints unchanged)
- `train_nll_ard` checkpoint contents and resume
"""

from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from dalg.models.mfa import MFA, load_mfa  # noqa: E402
from dalg.models.mfa_ard import MFA_ARD, load_mfa_ard, save_mfa_ard  # noqa: E402
from dalg.models.train_ard import ard_beta_schedule, train_nll_ard  # noqa: E402


# Most tests exercise the penalty itself, so they disable the beta schedule and
# apply full ARD pressure from epoch 1.
NO_SCHEDULE = dict(ard_warmup_frac=0.0, ard_ramp_frac=0.0)


def _build_tiny_ard(
    seed: int = 0,
    K: int = 4,
    D: int = 8,
    q: int = 4,
    **kwargs,
) -> MFA_ARD:
    torch.manual_seed(seed)
    centroids = torch.randn(K, D)
    return MFA_ARD(centroids, rank=q, **kwargs)


def _fixed_batches(n: int = 4, B: int = 16, D: int = 8, seed: int = 1):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(B, D, generator=g) for _ in range(n)]


def _rank_one_batches(n: int = 6, B: int = 64, D: int = 8, seed: int = 3):
    """Batches whose support is a 1-D subspace plus small isotropic noise."""
    g = torch.Generator().manual_seed(seed)
    direction = torch.randn(D, generator=g)
    direction = direction / direction.norm()
    out = []
    for _ in range(n):
        t = torch.randn(B, 1, generator=g) * 3.0
        noise = torch.randn(B, D, generator=g) * 0.01
        out.append(t * direction[None, :] + noise)
    return out


# --------------------------------------------------------------------------
# Penalty algebra
# --------------------------------------------------------------------------


class PenaltyTests(unittest.TestCase):
    def test_penalty_equals_profiled_form(self):
        # Substituting nu* = c / (1/2 s^2 + b0) into the penalty gives
        #   c * log(1/2 s^2 + b0) + c * (1 - log c)
        model = _build_tiny_ard(alpha0=2.0, b0=1e-3)
        with torch.no_grad():
            model.scale_rho.normal_(mean=0.0, std=1.0)

        c = model.log_coeff
        s = model.column_scales()
        expected = (c * torch.log(0.5 * s ** 2 + model.b0)).sum() + s.numel() * c * (
            1.0 - math.log(c)
        )
        self.assertAlmostEqual(
            float(model.ard_penalty().item()), float(expected.item()), places=3
        )

    def test_detached_nu_gradient_matches_profiled_penalty(self):
        # The load-bearing claim: with nu detached, d(pen)/d(scale_rho) equals
        # the gradient of the nu-eliminated penalty c*log(1/2 s^2 + b0).
        model = _build_tiny_ard(alpha0=1.5, b0=1e-3)
        with torch.no_grad():
            model.scale_rho.normal_(mean=0.0, std=1.0)

        grad_ard = torch.autograd.grad(model.ard_penalty(), model.scale_rho)[0]

        s = torch.nn.functional.softplus(model.scale_rho)
        profiled = (model.log_coeff * torch.log(0.5 * s ** 2 + model.b0)).sum()
        grad_profiled = torch.autograd.grad(profiled, model.scale_rho)[0]

        self.assertTrue(torch.allclose(grad_ard, grad_profiled, atol=1e-6, rtol=0.0))

    def test_loss_terms_sum_matches_total(self):
        model = _build_tiny_ard(ard_weight=1e-2)
        x = _fixed_batches(n=1)[0]
        total, nll, penalty = model.loss_terms(x)
        self.assertAlmostEqual(
            float(total.item()),
            float(nll.item()) + model.ard_weight * float(penalty.item()),
            places=5,
        )
        # forward() returns the same total, so generic loops still work.
        self.assertAlmostEqual(float(model(x).item()), float(total.item()), places=5)

    def test_penalty_uses_column_norms_of_W(self):
        # dir_raw columns are unit-normalized, so s == ||w_j||. This identity is
        # what lets the penalty skip materializing W.
        model = _build_tiny_ard()
        with torch.no_grad():
            model.scale_rho.normal_()
        self.assertTrue(
            torch.allclose(
                model.column_scales(), model._W().norm(dim=1), atol=1e-5, rtol=0.0
            )
        )


class ValidationTests(unittest.TestCase):
    def test_non_positive_b0_rejected(self):
        with self.assertRaises(ValueError):
            _build_tiny_ard(b0=0.0)

    def test_non_positive_log_coefficient_rejected(self):
        # D=8 -> alpha0 must exceed 1 - D/2 = -3
        with self.assertRaises(ValueError):
            _build_tiny_ard(D=8, alpha0=-3.0)

    def test_rank_threshold_range_enforced(self):
        with self.assertRaises(ValueError):
            _build_tiny_ard(rank_threshold=0.0)


# --------------------------------------------------------------------------
# Shrinkage behaviour
# --------------------------------------------------------------------------


def _inv_softplus(y: float) -> float:
    return math.log(math.exp(y) - 1.0)


class EffectiveRankTests(unittest.TestCase):
    def test_effective_rank_counts_columns_above_the_noise_floor(self):
        model = _build_tiny_ard(K=2, D=8, q=4, rank_threshold=1.0)
        with torch.no_grad():
            model.psi_rho.fill_(_inv_softplus(1.0))          # Psi ~= 1.0
            big, small = _inv_softplus(2.0), _inv_softplus(0.5)   # s^2 = 4.0 / 0.25
            model.scale_rho.copy_(
                torch.tensor([[big, big, small, small], [big, small, small, small]])
            )
        self.assertEqual(model.effective_ranks().tolist(), [2, 1])

    def test_ard_shrinks_unused_columns(self):
        # Data lives on a 1-D subspace, so at most one column per component is
        # justified. Without ARD the fit keeps all 4; with ARD they are pruned.
        batches = _rank_one_batches()
        D = batches[0].shape[1]

        def _run(ard_weight: float) -> float:
            model = _build_tiny_ard(
                seed=5, K=2, D=D, q=4,
                alpha0=1.0, b0=1e-4,
                ard_weight=ard_weight,
                rank_threshold=1.0,
            )
            train_nll_ard(
                model, batches,
                epochs=40, lr=5e-2,
                track_best=False, log_interval=10_000,
                **NO_SCHEDULE,
            )
            return float(model.effective_ranks().float().mean().item())

        with_ard = _run(1e-2)
        without_ard = _run(0.0)
        self.assertLess(with_ard, without_ard)
        self.assertLessEqual(with_ard, 1.0)
        self.assertGreaterEqual(without_ard, 2.0)

    def test_effective_rank_detects_total_collapse(self):
        # The over-pruning failure mode STATE.md flags: full ARD pressure from a
        # cold start collapses every column and Psi absorbs the variance. The
        # metric must report q_k = 0 here, not full rank — that is why it
        # references Psi rather than each component's largest column.
        batches = _rank_one_batches()
        model = _build_tiny_ard(
            seed=5, K=2, D=batches[0].shape[1], q=4,
            ard_weight=1e-1, rank_threshold=1.0,
        )
        train_nll_ard(
            model, batches,
            epochs=40, lr=5e-2,
            track_best=False, log_interval=10_000,
            **NO_SCHEDULE,
        )
        self.assertEqual(int(model.effective_ranks().sum().item()), 0)

    def test_beta_warmup_prevents_the_collapse(self):
        # Same weight and data as the collapse test above; the only difference
        # is that beta ramps in. The columns get to align with the data before
        # the penalty's s->0 well can trap them, so real structure survives.
        batches = _rank_one_batches()
        model = _build_tiny_ard(
            seed=5, K=2, D=batches[0].shape[1], q=4,
            ard_weight=1e-1, rank_threshold=1.0,
        )
        train_nll_ard(
            model, batches,
            epochs=40, lr=5e-2,
            track_best=False, log_interval=10_000,
        )
        self.assertGreater(int(model.effective_ranks().sum().item()), 0)


# --------------------------------------------------------------------------
# Beta schedule
# --------------------------------------------------------------------------


class BetaScheduleTests(unittest.TestCase):
    def test_default_shape_over_20_epochs(self):
        # 15% of 20 = 3 warmup epochs, then a 20% = 4-epoch ramp, then flat.
        betas = [round(ard_beta_schedule(ep, 20), 3) for ep in range(1, 21)]
        self.assertEqual(betas[:3], [0.0, 0.0, 0.0])            # epochs 1-3
        self.assertEqual(betas[3:7], [0.25, 0.5, 0.75, 1.0])    # epochs 4-7 ramp
        self.assertTrue(all(b == 1.0 for b in betas[7:]))       # epochs 8-20

    def test_bounds_and_monotonicity(self):
        betas = [ard_beta_schedule(ep, 37) for ep in range(1, 38)]
        self.assertTrue(all(0.0 <= b <= 1.0 for b in betas))
        self.assertTrue(all(b <= n for b, n in zip(betas, betas[1:])))
        self.assertEqual(betas[0], 0.0)
        self.assertEqual(betas[-1], 1.0)

    def test_fraction_boundaries_are_respected(self):
        # Exactly 15% of epochs at zero pressure, and full pressure by the end
        # of the warmup+ramp window.
        for total in (10, 20, 37, 100):
            betas = [ard_beta_schedule(ep, total) for ep in range(1, total + 1)]
            zeros = sum(1 for b in betas if b == 0.0)
            self.assertEqual(zeros, math.ceil(0.15 * total), f"total={total}")
            first_full = betas.index(1.0) + 1
            self.assertLessEqual(first_full, math.ceil(0.35 * total), f"total={total}")

    def test_disabled_without_a_horizon(self):
        self.assertEqual(ard_beta_schedule(1, None), 1.0)
        self.assertEqual(ard_beta_schedule(1, 0), 1.0)

    def test_zero_fractions_apply_full_pressure_immediately(self):
        self.assertEqual(ard_beta_schedule(1, 20, warmup_frac=0.0, ramp_frac=0.0), 1.0)

    def test_fractions_must_be_a_valid_split(self):
        with self.assertRaises(ValueError):
            ard_beta_schedule(1, 20, warmup_frac=0.8, ramp_frac=0.4)

    def test_schedule_drives_the_model_weight(self):
        # ard_weight must follow beta during training and be restored to the
        # configured target at the end.
        model = _build_tiny_ard(ard_weight=0.5)
        seen = []
        batches = _fixed_batches(n=1)

        class _Spy(list):
            def __iter__(self_inner):
                seen.append(model.ard_weight)
                return iter(batches)

        train_nll_ard(model, _Spy(), epochs=20, lr=1e-3,
                      track_best=False, log_interval=10_000)
        self.assertEqual(seen[:3], [0.0, 0.0, 0.0])
        self.assertAlmostEqual(seen[3], 0.5 * 0.25)   # first ramp epoch
        self.assertAlmostEqual(seen[-1], 0.5)
        self.assertAlmostEqual(model.ard_weight, 0.5)


# --------------------------------------------------------------------------
# Pruning (post-training only)
# --------------------------------------------------------------------------


class PruningTests(unittest.TestCase):
    def _model_with_known_columns(self) -> MFA_ARD:
        model = _build_tiny_ard(K=2, D=8, q=4, rank_threshold=1.0)
        with torch.no_grad():
            model.psi_rho.fill_(_inv_softplus(1.0))              # Psi ~= 1.0
            big, small = _inv_softplus(2.0), _inv_softplus(0.5)  # s^2 = 4.0 / 0.25
            model.scale_rho.copy_(
                torch.tensor([[big, big, small, small], [big, small, small, small]])
            )
        return model

    def test_prune_zeroes_sub_threshold_columns_exactly(self):
        model = self._model_with_known_columns()
        kept = model.prune_columns()

        self.assertEqual(kept.tolist(), [2, 1])
        W = model._W()
        self.assertTrue(torch.all(W[0, :, 2:] == 0))
        self.assertTrue(torch.all(W[1, :, 1:] == 0))
        # Survivors are untouched.
        self.assertTrue(torch.allclose(W[0, :, :2].norm(dim=0), torch.full((2,), 2.0), atol=1e-5))

    def test_prune_preserves_shape_and_downstream_loading(self):
        model = self._model_with_known_columns()
        model.prune_columns()
        self.assertEqual(tuple(model._W().shape), (2, 8, 4))
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "mfa_model.pt"
            save_mfa_ard(model, str(path), pruned=True)
            plain = load_mfa(path)                       # downstream still works
            self.assertEqual(plain.q, 4)
            meta = torch.load(path, weights_only=False)["meta"]
            self.assertTrue(meta["ard"]["pruned"])
            self.assertEqual(meta["ard"]["effective_ranks"], [2, 1])

    def test_prune_is_idempotent_and_stable(self):
        model = self._model_with_known_columns()
        first = model.prune_columns()
        second = model.prune_columns()
        self.assertEqual(first.tolist(), second.tolist())

    def test_prune_leaves_the_likelihood_essentially_unchanged(self):
        # Pruned columns are below the noise floor, so removing them should
        # barely move the likelihood. A large jump means the threshold is wrong.
        torch.manual_seed(0)
        model = self._model_with_known_columns()
        x = torch.randn(128, 8)
        with torch.no_grad():
            before = float(model.nll(x).item())
            model.prune_columns()
            after = float(model.nll(x).item())
        self.assertLess(abs(after - before), 0.5)

    def test_training_never_prunes_on_its_own(self):
        # Pruning is strictly a post-training step: after train_nll_ard the
        # sub-threshold columns must still be present (non-zero), so the caller
        # decides when the zeroing happens.
        model = _build_tiny_ard(K=2, D=8, q=4, ard_weight=1e-3)
        train_nll_ard(model, _fixed_batches(), epochs=6, lr=1e-3,
                      track_best=False, log_interval=10_000)
        self.assertTrue(torch.all(model.column_scales() > 0))
        self.assertTrue(torch.all(model._W().norm(dim=1) > 0))


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------


class PersistenceTests(unittest.TestCase):
    def test_round_trip_preserves_hyperparameters_and_weights(self):
        model = _build_tiny_ard(alpha0=1.7, b0=5e-3, ard_weight=1e-6, rank_threshold=5e-2)
        with torch.no_grad():
            model.scale_rho.normal_()

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "mfa_model.pt"
            save_mfa_ard(model, str(path))
            loaded = load_mfa_ard(path)

        self.assertAlmostEqual(loaded.alpha0, 1.7)
        self.assertAlmostEqual(loaded.b0, 5e-3)
        self.assertAlmostEqual(loaded.ard_weight, 1e-6)
        self.assertAlmostEqual(loaded.rank_threshold, 5e-2)
        self.assertAlmostEqual(loaded.log_coeff, model.log_coeff)
        for k, v in model.state_dict().items():
            self.assertTrue(torch.allclose(v, loaded.state_dict()[k]))

    def test_plain_load_mfa_reads_ard_checkpoints(self):
        # Closed-form nu adds no parameters, so downstream analysis code that
        # calls load_mfa works on ARD runs unchanged.
        model = _build_tiny_ard()
        x = _fixed_batches(n=1)[0]
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "mfa_model.pt"
            save_mfa_ard(model, str(path))
            plain = load_mfa(path)

        self.assertIsInstance(plain, MFA)
        with torch.no_grad():
            self.assertAlmostEqual(
                float(plain.nll(x).item()), float(model.nll(x).item()), places=4
            )


# --------------------------------------------------------------------------
# Training loop
# --------------------------------------------------------------------------


class TrainLoopTests(unittest.TestCase):
    EXPECTED_KEYS = {
        "epoch", "model", "optimizer",
        "best_metric", "best_state", "best_epoch",
        "last_val_metric", "epochs_without_improvement", "rng_state",
        "ard_schedule_epochs",
    }

    def test_checkpoint_file_written_with_expected_keys(self):
        with tempfile.TemporaryDirectory() as d:
            ckpt = Path(d) / "ckpt.pt"
            info = train_nll_ard(
                _build_tiny_ard(ard_weight=1e-3), _fixed_batches(),
                epochs=2, lr=1e-3,
                ckpt_path=str(ckpt), log_interval=1000,
            )
            obj = torch.load(ckpt, weights_only=False)
            self.assertEqual(set(obj.keys()), self.EXPECTED_KEYS)
            self.assertEqual(obj["epoch"], 2)
            self.assertIsNotNone(info["q_eff_mean"])

    def test_resume_runs_only_the_remaining_epochs(self):
        # Stopping partway through a 4-epoch schedule and finishing it: the
        # horizon is pinned, so the guard passes and the ramp continues.
        with tempfile.TemporaryDirectory() as d:
            ckpt = Path(d) / "ckpt.pt"
            train_nll_ard(_build_tiny_ard(), _fixed_batches(),
                          epochs=2, lr=1e-3, ard_schedule_epochs=4,
                          ckpt_path=str(ckpt), log_interval=1000)
            self.assertEqual(torch.load(ckpt, weights_only=False)["epoch"], 2)

            train_nll_ard(_build_tiny_ard(), _fixed_batches(),
                          epochs=4, lr=1e-3, ard_schedule_epochs=4,
                          ckpt_path=str(ckpt), log_interval=1000)
            self.assertEqual(torch.load(ckpt, weights_only=False)["epoch"], 4)

    def test_resume_matches_single_shot_training(self):
        # beta is a function of (epoch, horizon), so a resume reproduces the
        # single-shot trajectory as long as the horizon is the same. Here the
        # first leg stops at epoch 2 of a 4-epoch schedule, rather than running
        # a complete 2-epoch schedule.
        batches = _fixed_batches()
        sched = dict(ard_schedule_epochs=4)

        single = _build_tiny_ard(ard_weight=1e-2)
        train_nll_ard(single, batches, epochs=4, lr=1e-3, log_interval=1000, **sched)

        with tempfile.TemporaryDirectory() as d:
            ckpt = Path(d) / "ckpt.pt"
            train_nll_ard(_build_tiny_ard(ard_weight=1e-2), batches,
                          epochs=2, lr=1e-3,
                          ckpt_path=str(ckpt), log_interval=1000, **sched)
            resumed = _build_tiny_ard(ard_weight=1e-2)
            train_nll_ard(resumed, batches,
                          epochs=4, lr=1e-3,
                          ckpt_path=str(ckpt), log_interval=1000, **sched)

        for k, v in single.state_dict().items():
            self.assertTrue(
                torch.allclose(v, resumed.state_dict()[k], atol=1e-5, rtol=0.0),
                f"resumed state diverged on {k}",
            )

    def test_checkpoint_records_the_schedule_horizon(self):
        with tempfile.TemporaryDirectory() as d:
            ckpt = Path(d) / "ckpt.pt"
            train_nll_ard(_build_tiny_ard(ard_weight=1e-3), _fixed_batches(),
                          epochs=6, lr=1e-3, ckpt_path=str(ckpt), log_interval=1000)
            self.assertEqual(
                torch.load(ckpt, weights_only=False)["ard_schedule_epochs"], 6
            )
            # An explicit horizon overrides the epoch count.
            ckpt2 = Path(d) / "ckpt2.pt"
            train_nll_ard(_build_tiny_ard(ard_weight=1e-3), _fixed_batches(),
                          epochs=6, lr=1e-3, ard_schedule_epochs=50,
                          ckpt_path=str(ckpt2), log_interval=1000)
            self.assertEqual(
                torch.load(ckpt2, weights_only=False)["ard_schedule_epochs"], 50
            )

    def test_resume_with_a_changed_horizon_raises(self):
        # The footgun this guard exists for: resuming a 6-epoch run as a
        # 60-epoch one would drop ard_beta back to 0 for epochs already trained
        # at full pressure.
        with tempfile.TemporaryDirectory() as d:
            ckpt = Path(d) / "ckpt.pt"
            train_nll_ard(_build_tiny_ard(ard_weight=1e-3), _fixed_batches(),
                          epochs=6, lr=1e-3, ckpt_path=str(ckpt), log_interval=1000)

            with self.assertRaises(RuntimeError) as cm:
                train_nll_ard(_build_tiny_ard(ard_weight=1e-3), _fixed_batches(),
                              epochs=60, lr=1e-3, ckpt_path=str(ckpt), log_interval=1000)
            msg = str(cm.exception)
            self.assertIn("ard_schedule_epochs=6", msg)
            self.assertIn("60", msg)

    def test_pinning_the_horizon_allows_extending_the_epoch_cap(self):
        # The documented escape hatch: keep the original ramp, raise the cap.
        with tempfile.TemporaryDirectory() as d:
            ckpt = Path(d) / "ckpt.pt"
            train_nll_ard(_build_tiny_ard(ard_weight=1e-3), _fixed_batches(),
                          epochs=6, lr=1e-3, ckpt_path=str(ckpt), log_interval=1000)
            train_nll_ard(_build_tiny_ard(ard_weight=1e-3), _fixed_batches(),
                          epochs=10, lr=1e-3, ard_schedule_epochs=6,
                          ckpt_path=str(ckpt), log_interval=1000)
            obj = torch.load(ckpt, weights_only=False)
            self.assertEqual(obj["epoch"], 10)
            self.assertEqual(obj["ard_schedule_epochs"], 6)

    def test_horizon_check_skipped_without_ard_pressure(self):
        # With ard_weight=0 the schedule cannot affect the run, so changing the
        # epoch count on resume must not be blocked (lambda=0 control runs).
        with tempfile.TemporaryDirectory() as d:
            ckpt = Path(d) / "ckpt.pt"
            train_nll_ard(_build_tiny_ard(ard_weight=0.0), _fixed_batches(),
                          epochs=4, lr=1e-3, ckpt_path=str(ckpt), log_interval=1000)
            train_nll_ard(_build_tiny_ard(ard_weight=0.0), _fixed_batches(),
                          epochs=8, lr=1e-3, ckpt_path=str(ckpt), log_interval=1000)
            self.assertEqual(torch.load(ckpt, weights_only=False)["epoch"], 8)

    def test_legacy_checkpoint_without_horizon_warns_but_resumes(self):
        with tempfile.TemporaryDirectory() as d:
            ckpt = Path(d) / "ckpt.pt"
            train_nll_ard(_build_tiny_ard(ard_weight=1e-3), _fixed_batches(),
                          epochs=4, lr=1e-3, ckpt_path=str(ckpt), log_interval=1000)
            obj = torch.load(ckpt, weights_only=False)
            del obj["ard_schedule_epochs"]          # simulate a pre-guard file
            torch.save(obj, ckpt)

            train_nll_ard(_build_tiny_ard(ard_weight=1e-3), _fixed_batches(),
                          epochs=8, lr=1e-3, ckpt_path=str(ckpt), log_interval=1000)
            self.assertEqual(torch.load(ckpt, weights_only=False)["epoch"], 8)

    def test_changing_the_horizon_changes_the_remaining_schedule(self):
        # Documents the flip side: resuming with a different --epochs rescales
        # the schedule, so the trajectory is not the single-shot one. Pin
        # --ard-schedule-epochs when identical trajectories matter.
        self.assertNotEqual(
            ard_beta_schedule(2, 2),   # epoch 2 of a 2-epoch run: past the ramp
            ard_beta_schedule(2, 20),  # epoch 2 of a 20-epoch run: still warming up
        )

    def test_best_tracking_uses_validation_nll_only(self):
        # Selection must ignore the ARD penalty so ARD runs stay comparable to
        # baseline MFA runs: the reported best_metric is a pure NLL.
        torch.manual_seed(0)
        D = 8
        val_tensor = torch.randn(64, D)
        model = _build_tiny_ard(D=D, ard_weight=1e-2)
        info = train_nll_ard(
            model, _fixed_batches(n=4, B=16, D=D),
            val_tensor=val_tensor,
            epochs=8, lr=5e-1,
            log_interval=1000,
        )
        with torch.no_grad():
            cur = float(model.nll(val_tensor.float()).item())
        self.assertAlmostEqual(cur, info["best_metric"], places=4)


if __name__ == "__main__":
    unittest.main()
