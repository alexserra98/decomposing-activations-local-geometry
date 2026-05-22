"""Tests for VAE mixture priors and shard-loader training."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from dalg.analysis.cluster_assignments import compute_assignments  # noqa: E402
from dalg.cli.run_metrics import build_parser as build_metrics_parser  # noqa: E402
from dalg.cli.run_metrics import cmd_assignments  # noqa: E402
from dalg.cli.run_training import build_parser, cmd_train_vae, validate_args  # noqa: E402
from dalg.models.train import train_vae  # noqa: E402
from dalg.models.vae import (  # noqa: E402
    MoGPrior,
    VAE,
    VampPrior,
    adapt_activation_batch,
    load_vae,
    save_vae,
)
from tests.synthetic_shards import LAYER, build_multi_shard  # noqa: E402


class MoGPriorTests(unittest.TestCase):
    def test_responsibilities_match_bayes_rule(self):
        prior = MoGPrior(latent_dim=2, n_components=3)
        with torch.no_grad():
            prior.logits.copy_(torch.tensor([0.2, -0.4, 1.1]))
            prior.means.copy_(torch.tensor([
                [0.0, 0.0],
                [2.0, 0.0],
                [0.0, -1.0],
            ]))
            prior.logvars.copy_(torch.tensor([
                [0.0, 0.0],
                [0.3, -0.2],
                [-0.5, 0.4],
            ]))

        z = torch.tensor([
            [0.1, -0.2],
            [1.5, 0.2],
            [-0.4, -1.2],
        ])
        log_joint = prior.log_joint(z)
        expected = torch.exp(log_joint - torch.logsumexp(log_joint, dim=1, keepdim=True))

        resp = prior.responsibilities(z)
        self.assertTrue(torch.allclose(resp, expected, atol=1e-6))
        self.assertTrue(torch.allclose(resp.sum(dim=1), torch.ones(z.shape[0]), atol=1e-6))
        self.assertTrue(torch.allclose(prior.log_prob(z), torch.logsumexp(log_joint, dim=1), atol=1e-6))

    def test_vamp_prior_responsibilities_match_its_log_joint(self):
        torch.manual_seed(0)
        prior = VampPrior(encoder=None, latent_dim=2, input_dim=2, n_components=4)
        VAE(
            input_dim=2,
            latent_dim=2,
            enc_hidden_dims=(4,),
            dec_hidden_dims=(4,),
            prior=prior,
        )
        z = torch.randn(3, 2)
        log_joint = prior.log_joint(z)
        expected = torch.exp(log_joint - torch.logsumexp(log_joint, dim=1, keepdim=True))

        resp = prior.responsibilities(z)
        self.assertTrue(torch.allclose(resp, expected, atol=1e-6))
        self.assertTrue(torch.allclose(resp.sum(dim=1), torch.ones(z.shape[0]), atol=1e-6))


class VAETrainingTests(unittest.TestCase):
    def test_adapt_activation_batch_accepts_metadata_contract(self):
        x = torch.randn(2, 3, 4)
        rows = torch.tensor([10, 11, 12, 13, 14, 15])
        tok_pos = torch.tensor([0, 1, 2, 0, 1, 2])
        adapted = adapt_activation_batch((x, rows, tok_pos), input_dim=4)
        self.assertEqual(adapted.shape, (6, 4))

    def test_train_vae_smoke_with_metadata_batches(self):
        torch.manual_seed(0)
        batches = [
            (torch.randn(4, 2), torch.arange(4), torch.arange(4)),
            (torch.randn(4, 2), torch.arange(4, 8), torch.arange(4)),
        ]
        model = VAE(
            input_dim=2,
            latent_dim=2,
            enc_hidden_dims=(4,),
            dec_hidden_dims=(4,),
            prior=MoGPrior(latent_dim=2, n_components=2),
            beta=0.1,
        )

        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            info = train_vae(
                model,
                batches,
                epochs=1,
                lr=1e-3,
                weight_decay=0.0,
                save_path=str(out / "vae_model.pt"),
                ckpt_path=str(out / "checkpoint.pt"),
                steps_per_epoch=2,
                log_interval=1000,
            )

            self.assertTrue((out / "vae_model.pt").exists())
            self.assertTrue((out / "checkpoint.pt").exists())
            self.assertGreaterEqual(info["best_epoch"], 1)
            saved = torch.load(out / "vae_model.pt", map_location="cpu", weights_only=False)
            self.assertEqual(saved["input_dim"], 2)
            self.assertEqual(saved["prior"]["name"], "mog")

    def test_load_vae_round_trip(self):
        torch.manual_seed(0)
        model = VAE(
            input_dim=2,
            latent_dim=2,
            enc_hidden_dims=(4,),
            dec_hidden_dims=(4,),
            prior=MoGPrior(latent_dim=2, n_components=3),
            beta=0.2,
        )

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "vae_model.pt"
            save_vae(model, path)
            loaded = load_vae(path, map_location="cpu")

            self.assertEqual(loaded.input_dim, 2)
            self.assertEqual(loaded.latent_dim, 2)
            self.assertEqual(loaded.enc_hidden_dims, (4,))
            self.assertEqual(loaded.dec_hidden_dims, (4,))
            self.assertEqual(loaded.prior.n_components, 3)
            for key, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, loaded.state_dict()[key]), key)

    def test_compute_assignments_supports_vae_model_type(self):
        torch.manual_seed(0)
        model = VAE(
            input_dim=2,
            latent_dim=2,
            enc_hidden_dims=(4,),
            dec_hidden_dims=(4,),
            prior=MoGPrior(latent_dim=2, n_components=3),
        )
        loader = [torch.randn(5, 2), torch.randn(5, 2)]

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "vae_model.pt"
            save_vae(model, path)
            sizes, assignments, max_resp, peakedness = compute_assignments(
                path,
                loader,
                model_type="vae",
                device="cpu",
            )

            self.assertEqual(sizes.numel(), 3)
            self.assertEqual(assignments.numel(), 10)
            self.assertEqual(max_resp.numel(), 10)
            self.assertEqual(int(sizes.sum().item()), 10)
            self.assertIn("entropy", peakedness)

    def test_cmd_train_vae_uses_shard_loader_contract(self):
        with tempfile.TemporaryDirectory() as d:
            root = build_multi_shard(Path(d) / "shards", n_shards=2, rows_per_shard=4)
            out_dir = Path(d) / "vae_run"
            args = build_parser().parse_args([
                "--training-mode", "vae",
                "--device", "cpu",
                "--shard-dir", str(root),
                "--layer", str(LAYER),
                "--out-dir", str(out_dir),
                "--batch-size", "8",
                "--num-workers", "0",
                "--epochs", "1",
                "--max-steps", "2",
                "--lr", "1e-4",
                "--grad-clip", "10.0",
                "--val-frac", "0.25",
                "--seed", "0",
                "--vae-latent-dim", "2",
                "--vae-enc-hidden-dims", "4",
                "--vae-dec-hidden-dims", "4",
                "--vae-prior", "standard",
                "--vae-beta", "0.01",
            ])
            validate_args(args)
            cmd_train_vae(args)

            self.assertTrue((out_dir / "vae_model.pt").exists())
            self.assertTrue((out_dir / "checkpoint.pt").exists())
            self.assertTrue((out_dir / "val_indices.json").exists())
            cfg = json.loads((out_dir / "config.json").read_text())
            self.assertEqual(cfg["training_mode"], "vae")
            self.assertEqual(cfg["d_model"], 2)

    def test_cmd_assignments_supports_vae_model_type(self):
        torch.manual_seed(0)
        model = VAE(
            input_dim=2,
            latent_dim=2,
            enc_hidden_dims=(4,),
            dec_hidden_dims=(4,),
            prior=MoGPrior(latent_dim=2, n_components=3),
        )

        with tempfile.TemporaryDirectory() as d:
            root = build_multi_shard(Path(d) / "shards", n_shards=1, rows_per_shard=2)
            model_path = Path(d) / "vae_model.pt"
            save_path = Path(d) / "vae_assignments.pt"
            save_vae(model, model_path)

            args = build_metrics_parser().parse_args([
                "assignments",
                "--model-type", "vae",
                "--data-dir", str(model_path),
                "--shard-dir", str(root),
                "--layer", str(LAYER),
                "--batch-size", "4",
                "--device", "cpu",
                "--save-path", str(save_path),
            ])
            cmd_assignments(args)

            saved = torch.load(save_path, map_location="cpu", weights_only=False)
            self.assertEqual(saved["model_type"], "vae")
            self.assertEqual(saved["K"], 3)
            self.assertEqual(saved["assignments"].numel(), 6)


if __name__ == "__main__":
    unittest.main()
