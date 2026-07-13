"""Sanity checks for MFA Gaussian overlap and responsibility peakiness."""

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

from dalg.analysis.cluster_assignments import compute_assignments  # noqa: E402
from dalg.analysis.gaussian_overlap import compute_gaussian_overlap  # noqa: E402
from dalg.models.mfa import MFA, save_mfa  # noqa: E402


def _inv_softplus(x: float) -> float:
    return math.log(math.exp(float(x)) - 1.0)


def _set_simple_mfa(
    model: MFA,
    *,
    mu: torch.Tensor,
    directions: torch.Tensor,
    scales: torch.Tensor,
    psi: float = 1.0,
) -> MFA:
    """Set MFA parameters directly through their public parameterization."""
    model.mu.data.copy_(mu)
    model.dir_raw.data.copy_(directions)
    model.scale_rho.data.copy_(torch.log(torch.exp(scales) - 1.0))
    model.psi_rho.data.fill_(_inv_softplus(psi))
    model.pi_logits.data.zero_()
    return model


def _one_axis_directions(K: int, D: int, q: int, axis: int = 0) -> torch.Tensor:
    directions = torch.zeros(K, D, q)
    for k in range(K):
        directions[k, axis, :] = 1.0
    return directions


def _save_temp_model(model: MFA) -> tuple[tempfile.TemporaryDirectory, Path]:
    tmp = tempfile.TemporaryDirectory()
    model_path = Path(tmp.name) / "mfa_model.pt"
    save_mfa(model, str(model_path))
    return tmp, model_path


class OverlapAndAssignmentSanityTests(unittest.TestCase):
    def test_identical_components_have_full_overlap_and_flat_responsibilities(self):
        D = 100
        K = 2
        q = 1
        model = MFA(torch.zeros(K, D), rank=q, psi_init=1.0, scale_init=0.5)
        model = _set_simple_mfa(
            model,
            mu=torch.zeros(K, D),
            directions=_one_axis_directions(K, D, q),
            scales=torch.full((K, q), 0.5),
            psi=1.0,
        )

        tmp, model_path = _save_temp_model(model)
        self.addCleanup(tmp.cleanup)

        overlap = compute_gaussian_overlap(model_path, batch_pairs=16)
        self.assertTrue(torch.allclose(overlap["kl_sym"], torch.zeros(K, K), atol=1e-5))
        self.assertTrue(torch.allclose(overlap["db"], torch.zeros(K, K), atol=1e-5))
        self.assertTrue(torch.allclose(overlap["db_mean"], torch.zeros(K, K), atol=1e-5))
        self.assertTrue(torch.allclose(overlap["db_cov"], torch.zeros(K, K), atol=1e-5))
        self.assertTrue(torch.allclose(overlap["bc"], torch.ones(K, K), atol=1e-5))

        x = torch.randn(32, D)
        r = model.responsibilities(x)
        self.assertTrue(torch.allclose(r, torch.full((32, K), 0.5), atol=1e-6))

        sizes, assignments, max_resp, peakedness = compute_assignments(
            model_path,
            [x],
            device="cpu",
            use_inference_cache=False,
        )
        self.assertEqual(int(sizes.sum().item()), 32)
        self.assertTrue(torch.all(assignments == 0))
        self.assertTrue(torch.allclose(max_resp, torch.full((32,), 0.5), atol=1e-6))
        self.assertAlmostEqual(float(peakedness["entropy"][0]), math.log(2.0), places=5)
        self.assertAlmostEqual(float(peakedness["one_minus_max"][0]), 0.5, places=5)
        self.assertAlmostEqual(float(peakedness["top1_minus_top2"][0]), 0.0, places=5)

    def test_equal_covariance_overlap_matches_known_mahalanobis_values(self):
        D = 100
        K = 4
        q = 1
        mu = torch.zeros(K, D)
        mu[1, 1] = 1.0
        mu[2, 1] = 2.0
        mu[3, 2] = 3.0

        model = MFA(mu, rank=q, psi_init=1.0, scale_init=0.7)
        model = _set_simple_mfa(
            model,
            mu=mu,
            directions=_one_axis_directions(K, D, q, axis=0),
            scales=torch.full((K, q), 0.7),
            psi=1.0,
        )

        tmp, model_path = _save_temp_model(model)
        self.addCleanup(tmp.cleanup)

        overlap = compute_gaussian_overlap(model_path, batch_pairs=16)
        maha = torch.cdist(mu, mu) ** 2
        expected_db = maha / 8.0
        expected_kl_sym = maha / 2.0
        expected_bc = torch.exp(-expected_db)
        expected_bc.fill_diagonal_(1.0)

        self.assertTrue(torch.allclose(overlap["db"], expected_db, atol=1e-5))
        self.assertTrue(torch.allclose(overlap["db_mean"], expected_db, atol=1e-5))
        self.assertTrue(torch.allclose(overlap["db_cov"], torch.zeros(K, K), atol=1e-5))
        self.assertTrue(torch.allclose(overlap["kl_sym"], expected_kl_sym, atol=1e-5))
        self.assertTrue(torch.allclose(overlap["bc"], expected_bc, atol=1e-5))

    def test_assignment_cache_matches_uncached_responsibilities(self):
        D = 100
        K = 2
        q = 5
        mu = torch.zeros(K, D)
        mu[1, 0] = 2.0
        directions = torch.zeros(K, D, q)
        for k in range(K):
            directions[k, :q, :] = torch.eye(q)

        model = MFA(mu, rank=q, psi_init=1.0, scale_init=0.5)
        model = _set_simple_mfa(
            model,
            mu=mu,
            directions=directions,
            scales=torch.full((K, q), 0.5),
            psi=1.0,
        )

        tmp, model_path = _save_temp_model(model)
        self.addCleanup(tmp.cleanup)

        x = torch.randn(64, D)
        cached = compute_assignments(model_path, [x], device="cpu", use_inference_cache=True)
        uncached = compute_assignments(model_path, [x], device="cpu", use_inference_cache=False)

        self.assertTrue(torch.equal(cached[0], uncached[0]))
        self.assertTrue(torch.equal(cached[1], uncached[1]))
        self.assertTrue(torch.allclose(cached[2], uncached[2], atol=1e-6))
        for key in cached[3]:
            self.assertTrue(torch.allclose(cached[3][key], uncached[3][key], atol=1e-6))


if __name__ == "__main__":
    unittest.main()
