"""Unit tests for ``train_nll`` in src/dalg/models/train.py.

Coverage:
- checkpoint files contain the expected payload after each epoch
- resume from a checkpoint continues from the next epoch and is bit-equivalent
  (within fp noise) to a single-shot run thanks to the saved RNG state
- best-epoch tracking with ``val_tensor`` rolls the model back to that epoch
- with no validation, best tracking falls back to training NLL (current,
  documented behavior — locked in here)
- the same checkpoint + resume + best-broadcast paths work under
  ``torch.distributed`` (2 ranks, CPU/gloo via ``mp.spawn``)
"""

from __future__ import annotations

import os
import socket
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.multiprocessing as mp

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from dalg.models.mfa import ComponentShardedMFA, MFA, component_shard_bounds  # noqa: E402
from dalg.models.train import train_nll  # noqa: E402


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _build_tiny_mfa(seed: int = 0, K: int = 4, D: int = 8, q: int = 2) -> MFA:
    torch.manual_seed(seed)
    centroids = torch.randn(K, D)
    return MFA(centroids, rank=q)


def _fixed_batches(n: int = 4, B: int = 16, D: int = 8, seed: int = 1):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(B, D, generator=g) for _ in range(n)]


def _state_dicts_close(sd_a, sd_b, atol: float = 1e-5) -> bool:
    if set(sd_a.keys()) != set(sd_b.keys()):
        return False
    for k in sd_a:
        if not torch.allclose(sd_a[k], sd_b[k], atol=atol, rtol=0.0):
            return False
    return True


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


# --------------------------------------------------------------------------
# Checkpoint contents
# --------------------------------------------------------------------------
#
# train_nll writes a checkpoint after every epoch via _atomic_torch_save. The
# saved object is a single dict; here we just verify its keys and the epoch
# counter, since the other fields are exercised by the resume tests below.


class CheckpointContentsTests(unittest.TestCase):
    EXPECTED_KEYS = {
        "epoch", "model", "optimizer",
        "best_metric", "best_state", "best_epoch",
        "last_val_metric", "rng_state",
    }

    def test_checkpoint_file_written_with_expected_keys(self):
        with tempfile.TemporaryDirectory() as d:
            ckpt = Path(d) / "ckpt.pt"
            train_nll(
                _build_tiny_mfa(), _fixed_batches(),
                epochs=2, lr=1e-3,
                ckpt_path=str(ckpt), log_interval=1000,
            )
            self.assertTrue(ckpt.exists())
            obj = torch.load(ckpt, weights_only=False)
            self.assertEqual(set(obj.keys()), self.EXPECTED_KEYS)
            self.assertEqual(obj["epoch"], 2)

    def test_checkpoint_epoch_matches_final_epoch(self):
        with tempfile.TemporaryDirectory() as d:
            ckpt = Path(d) / "ckpt.pt"
            train_nll(
                _build_tiny_mfa(), _fixed_batches(),
                epochs=5, lr=1e-3,
                ckpt_path=str(ckpt), log_interval=1000,
            )
            self.assertEqual(torch.load(ckpt, weights_only=False)["epoch"], 5)

    def test_checkpoint_model_state_matches_in_memory_model_at_save_time(self):
        # The ckpt is written before any "restore best" rollback, so its
        # 'model' field must equal the live model's state after the last
        # epoch (not the best state, when those differ).
        with tempfile.TemporaryDirectory() as d:
            ckpt = Path(d) / "ckpt.pt"
            model = _build_tiny_mfa()
            train_nll(
                model, _fixed_batches(),
                epochs=3, lr=1e-3, track_best=False,  # so no end-of-run rollback
                ckpt_path=str(ckpt), log_interval=1000,
            )
            obj = torch.load(ckpt, weights_only=False)
            self.assertTrue(_state_dicts_close(obj["model"], model.state_dict()))


# --------------------------------------------------------------------------
# Resume
# --------------------------------------------------------------------------
#
# Resume works because train_nll persists model + optimizer + RNG state per
# epoch and reloads them on startup. With deterministic ops (MFA forward has
# no random ops) and a deterministic loader, 2+2 resumed training must match
# a single 4-epoch run bit-for-bit (within fp noise).


class ResumeTests(unittest.TestCase):
    def test_resume_runs_only_the_remaining_epochs(self):
        with tempfile.TemporaryDirectory() as d:
            ckpt = Path(d) / "ckpt.pt"
            train_nll(_build_tiny_mfa(), _fixed_batches(),
                      epochs=2, lr=1e-3,
                      ckpt_path=str(ckpt), log_interval=1000)
            self.assertEqual(torch.load(ckpt, weights_only=False)["epoch"], 2)

            train_nll(_build_tiny_mfa(), _fixed_batches(),
                      epochs=4, lr=1e-3,
                      ckpt_path=str(ckpt), log_interval=1000)
            self.assertEqual(torch.load(ckpt, weights_only=False)["epoch"], 4)

    def test_resume_matches_single_shot_training(self):
        batches = _fixed_batches()

        single = _build_tiny_mfa()
        train_nll(single, batches, epochs=4, lr=1e-3, log_interval=1000)

        with tempfile.TemporaryDirectory() as d:
            ckpt = Path(d) / "ckpt.pt"
            train_nll(_build_tiny_mfa(), batches,
                      epochs=2, lr=1e-3,
                      ckpt_path=str(ckpt), log_interval=1000)
            resumed = _build_tiny_mfa()
            train_nll(resumed, batches,
                      epochs=4, lr=1e-3,
                      ckpt_path=str(ckpt), log_interval=1000)

        self.assertTrue(
            _state_dicts_close(single.state_dict(), resumed.state_dict()),
            "resumed state_dict diverged from single-shot training",
        )

    def test_resume_no_op_when_already_at_target_epoch(self):
        # If the ckpt is at epoch == epochs, resume sets start_epoch = epochs+1
        # and the training loop body is skipped entirely. Model state must not
        # change between two consecutive calls with the same target.
        with tempfile.TemporaryDirectory() as d:
            ckpt = Path(d) / "ckpt.pt"
            model = _build_tiny_mfa()
            train_nll(model, _fixed_batches(),
                      epochs=3, lr=1e-3, track_best=False,
                      ckpt_path=str(ckpt), log_interval=1000)
            after_first = {k: v.clone() for k, v in model.state_dict().items()}

            train_nll(model, _fixed_batches(),
                      epochs=3, lr=1e-3, track_best=False,
                      ckpt_path=str(ckpt), log_interval=1000)
            self.assertTrue(_state_dicts_close(after_first, model.state_dict()))


# --------------------------------------------------------------------------
# Best-epoch tracking
# --------------------------------------------------------------------------


class BestEpochTests(unittest.TestCase):
    def test_best_epoch_with_val_tensor_rolls_model_back(self):
        # Train long enough with an aggressive lr that val NLL eventually
        # increases. Then assert the returned model is the best-by-val one,
        # i.e. evaluating its NLL on val_tensor matches the reported best_metric.
        torch.manual_seed(0)
        D = 8
        val_tensor = torch.randn(64, D)
        batches = _fixed_batches(n=4, B=16, D=D)

        model = _build_tiny_mfa(D=D)
        info = train_nll(
            model, batches,
            val_tensor=val_tensor,
            epochs=8, lr=5e-1,           # large lr -> overshoot, non-monotone
            log_interval=1000,
        )

        self.assertGreaterEqual(info["best_epoch"], 1)
        self.assertLessEqual(info["best_epoch"], 8)
        with torch.no_grad():
            cur = float(model.nll(val_tensor.float()).item())
        self.assertAlmostEqual(cur, info["best_metric"], places=4)

    def test_best_tracking_disabled_keeps_final_state(self):
        # track_best=False -> no rollback, best_epoch stays 0.
        model = _build_tiny_mfa()
        before = {k: v.clone() for k, v in model.state_dict().items()}
        info = train_nll(model, _fixed_batches(),
                         epochs=3, lr=1e-3, track_best=False,
                         log_interval=1000)
        self.assertEqual(info["best_epoch"], 0)
        after = model.state_dict()
        self.assertTrue(any(not torch.equal(before[k], after[k]) for k in before))

    def test_no_validation_falls_back_to_train_nll(self):
        # No val_tensor and no val_loader: the current behavior is that
        # select_metric = avg_train_nll, so best tracking still runs (just
        # against training loss). This test locks that in.
        model = _build_tiny_mfa()
        info = train_nll(model, _fixed_batches(),
                         epochs=3, lr=1e-3, log_interval=1000)
        self.assertGreaterEqual(info["best_epoch"], 1)
        self.assertLess(info["best_metric"], float("inf"))

    def test_unbounded_epochs_stop_on_validation_delta(self):
        model = _build_tiny_mfa()
        with tempfile.TemporaryDirectory() as d:
            ckpt = Path(d) / "ckpt.pt"
            train_nll(
                model,
                _fixed_batches(),
                val_tensor=torch.randn(32, 8),
                epochs=0,
                lr=0.0,
                ckpt_path=str(ckpt),
                log_interval=1000,
                early_stop_delta=1e-3,
            )
            self.assertEqual(torch.load(ckpt, weights_only=False)["epoch"], 2)


# --------------------------------------------------------------------------
# Distributed (2 ranks, CPU/gloo via mp.spawn)
# --------------------------------------------------------------------------
#
# Mirrors the component-sharded training pattern: every rank owns its own
# per-rank checkpoint file and `checkpoint_all_ranks=True`. We verify:
#   * each rank writes its own ckpt
#   * each rank reloads its own ckpt and resume advances start_epoch
#     identically across ranks (the all_reduce min/max guard otherwise raises)
#   * select_metric is broadcast from rank 0, so both ranks observe the same
#     best_metric and improvement decisions


def _dist_worker(rank, world_size, tmpdir, port, epochs, out_path):
    """One distributed worker: train ``epochs`` with a per-rank ckpt path.

    The checkpoint file lives at ``{tmpdir}/ckpt_rank{rank:04d}.pt`` so each
    rank both writes and (on subsequent calls) reloads its own shard, which
    is the production layout for component-sharded training.
    """
    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(backend="gloo")

    try:
        model = _build_tiny_mfa()
        batches = _fixed_batches()
        # Plain MFA validation can run only on rank 0; the metric is broadcast
        # inside train_nll.
        val_tensor = torch.randn(32, 8, generator=torch.Generator().manual_seed(7)) \
            if rank == 0 else None
        ckpt = str(Path(tmpdir) / f"ckpt_rank{rank:04d}.pt")

        info = train_nll(
            model, batches,
            val_tensor=val_tensor,
            epochs=epochs, lr=1e-3,
            ckpt_path=ckpt, log_interval=1000,
            checkpoint_all_ranks=True,
        )

        torch.save({
            "rank": rank,
            "best_epoch": info["best_epoch"],
            "best_metric": info["best_metric"],
            "ckpt_path": ckpt,
            "ckpt_exists": Path(ckpt).exists(),
            "ckpt_epoch": torch.load(ckpt, weights_only=False)["epoch"],
        }, out_path.replace("RANK", str(rank)))
    finally:
        dist.destroy_process_group()


def _component_sharded_val_worker(rank, world_size, port, out_path):
    """Train a tiny component-sharded MFA with validation on every rank."""
    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(backend="gloo")

    try:
        K, D, q = 4, 8, 2
        centroids = torch.randn(K, D, generator=torch.Generator().manual_seed(11))
        start, end = component_shard_bounds(K, rank, world_size)
        model = ComponentShardedMFA(
            centroids[start:end].clone(),
            rank=q,
            global_K=K,
            component_start=start,
        )
        batches = _fixed_batches(n=2, B=6, D=D, seed=13)
        val_tensor = torch.randn(18, D, generator=torch.Generator().manual_seed(17))

        info = train_nll(
            model,
            batches,
            val_tensor=val_tensor,
            epochs=2,
            lr=1e-3,
            log_interval=1000,
            checkpoint_all_ranks=True,
        )

        torch.save({
            "rank": rank,
            "best_epoch": info["best_epoch"],
            "best_metric": info["best_metric"],
        }, out_path.replace("RANK", str(rank)))
    finally:
        dist.destroy_process_group()


class DistributedTests(unittest.TestCase):
    def test_per_rank_checkpoint_and_resume(self):
        # Stage 1: train 2 epochs. Each rank writes its own ckpt file.
        # Stage 2: train with target epochs=4. Each rank loads its own ckpt,
        # the per-rank start_epoch all-reduce min/max guard passes, training
        # advances epochs 3..4, both ranks see the same broadcast best_metric.
        with tempfile.TemporaryDirectory() as d:
            out_template = str(Path(d) / "out_RANK.pt")

            mp.spawn(
                _dist_worker,
                args=(2, d, _free_port(), 2, out_template),
                nprocs=2, join=True,
            )
            for r in (0, 1):
                self.assertTrue((Path(d) / f"ckpt_rank{r:04d}.pt").exists())
                rec = torch.load(Path(d) / f"out_{r}.pt", weights_only=False)
                self.assertEqual(rec["ckpt_epoch"], 2)

            mp.spawn(
                _dist_worker,
                args=(2, d, _free_port(), 4, out_template),
                nprocs=2, join=True,
            )
            r0 = torch.load(Path(d) / "out_0.pt", weights_only=False)
            r1 = torch.load(Path(d) / "out_1.pt", weights_only=False)
            self.assertEqual(r0["ckpt_epoch"], 4)
            self.assertEqual(r1["ckpt_epoch"], 4)
            # select_metric is broadcast from rank 0 -> both ranks agree.
            self.assertAlmostEqual(r0["best_metric"], r1["best_metric"], places=6)
            # With checkpoint_all_ranks=True every rank tracks best_epoch.
            self.assertGreater(r0["best_epoch"], 0)
            self.assertGreater(r1["best_epoch"], 0)

    def test_component_sharded_validation_runs_on_all_ranks(self):
        with tempfile.TemporaryDirectory() as d:
            out_template = str(Path(d) / "val_RANK.pt")
            mp.spawn(
                _component_sharded_val_worker,
                args=(2, _free_port(), out_template),
                nprocs=2,
                join=True,
            )
            r0 = torch.load(Path(d) / "val_0.pt", weights_only=False)
            r1 = torch.load(Path(d) / "val_1.pt", weights_only=False)
            self.assertGreater(r0["best_epoch"], 0)
            self.assertGreater(r1["best_epoch"], 0)
            self.assertAlmostEqual(r0["best_metric"], r1["best_metric"], places=6)
            self.assertLess(r0["best_metric"], float("inf"))


if __name__ == "__main__":
    unittest.main()
