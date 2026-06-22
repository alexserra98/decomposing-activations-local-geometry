"""Smoke tests for the synthetic MFA K/q sweep script."""

from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from synthetic_mfa_qk_sweep import (  # noqa: E402
    SweepConfig,
    collect_results,
    default_dataset_path,
    fit_one_from_dataset,
    generate_dataset,
)


class SyntheticMFAQKSweepTests(unittest.TestCase):
    def test_tiny_cpu_sweep_writes_outputs_and_metrics(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = SweepConfig(
                dataset_path=Path(d) / "synthetic_dataset.pt",
                model_root=Path(d) / "models",
                run_name="tiny",
                D=12,
                K_true=3,
                q_true=2,
                K_fit=(2, 3),
                q_fit=(1, 2),
                n_train=180,
                n_test=90,
                n_seeds=1,
                seed=123,
                batch_size=32,
                epochs=1,
                lr=1e-3,
                grad_clip=5.0,
                mean_scale=3.0,
                factor_scale=0.5,
                psi=0.1,
                kmeans_max_iter=20,
                kmeans_n_init=1,
                device="cpu",
                no_plots=False,
            )

            generate_dataset(cfg)
            rows = [
                fit_one_from_dataset(cfg, K_fit=K_fit, q_fit=q_fit)
                for K_fit in cfg.K_fit
                for q_fit in cfg.q_fit
            ]
            collected = collect_results(cfg)

            self.assertEqual(len(rows), len(cfg.K_fit) * len(cfg.q_fit) * cfg.n_seeds)
            self.assertEqual(len(collected), len(rows))

            run_dir = cfg.model_root / cfg.run_name
            self.assertTrue(cfg.dataset_path.exists())
            self.assertTrue((run_dir / "results.csv").exists())
            self.assertTrue((run_dir / "results.pt").exists())
            self.assertTrue((run_dir / "config.json").exists())
            self.assertTrue((run_dir / "mean_max_resp_vs_q.png").exists())
            self.assertTrue((run_dir / "mean_max_resp_heatmap.png").exists())

            loaded = torch.load(run_dir / "results.pt", weights_only=False)
            self.assertEqual(len(loaded), len(rows))

            scalar_metric_names = [
                "train_nll",
                "test_nll",
                "mean_entropy",
                "norm_mean_entropy",
                "mean_max_resp",
                "mean_top1_minus_top2",
                "hungarian_accuracy",
                "adjusted_rand",
                "normalized_mutual_info",
            ]
            for row in rows:
                for name in scalar_metric_names:
                    self.assertTrue(math.isfinite(float(row[name])), name)
                self.assertEqual(len(row["max_resp_hist"]), 10)

            true_setting = next(
                row for row in rows
                if row["K_fit"] == cfg.K_true and row["q_fit"] == cfg.q_true
            )
            self.assertGreater(true_setting["hungarian_accuracy"], 0.5)

    def test_default_dataset_path_includes_truth_geometry(self):
        path = default_dataset_path(1000, 20, D=500, seed=0)

        self.assertIn("Ktrue1000", path.name)
        self.assertIn("qtrue20", path.name)

    def test_generation_supports_more_components_than_dimensions(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = SweepConfig(
                dataset_path=Path(d) / "synthetic_dataset.pt",
                D=4,
                K_true=7,
                q_true=2,
                n_train=20,
                n_test=10,
                seed=123,
                device="cpu",
            )

            dataset = generate_dataset(cfg)

            self.assertEqual(dataset["truth"]["mu"].shape, (7, 4))
            self.assertEqual(dataset["truth"]["dir_raw"].shape, (7, 4, 2))
            self.assertGreaterEqual(int(dataset["y_train"].min().item()), 0)
            self.assertLess(int(dataset["y_train"].max().item()), 7)

    def test_fit_rejects_dataset_with_wrong_truth_geometry(self):
        with tempfile.TemporaryDirectory() as d:
            dataset_path = Path(d) / "synthetic_dataset.pt"
            original = SweepConfig(
                dataset_path=dataset_path,
                model_root=Path(d) / "models",
                run_name="original",
                D=8,
                K_true=3,
                q_true=2,
                n_train=30,
                n_test=12,
                batch_size=8,
                epochs=1,
                device="cpu",
            )
            requested = SweepConfig(
                dataset_path=dataset_path,
                model_root=Path(d) / "models",
                run_name="requested",
                D=8,
                K_true=4,
                q_true=2,
                n_train=30,
                n_test=12,
                batch_size=8,
                epochs=1,
                device="cpu",
            )

            generate_dataset(original)

            with self.assertRaisesRegex(ValueError, "does not match this sweep config"):
                fit_one_from_dataset(requested, K_fit=2, q_fit=1)


if __name__ == "__main__":
    unittest.main()
