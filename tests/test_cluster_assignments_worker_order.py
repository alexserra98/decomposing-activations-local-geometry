"""Assignment-order checks for ``cluster_assignments.compute_assignments``."""

from __future__ import annotations

import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

import dalg.analysis.cluster_assignments as cluster_assignments  # noqa: E402
from dalg.data.shard_activations import ActivationBatchDataset  # noqa: E402
from tests.synthetic_shards import LAYER, build_multi_shard  # noqa: E402


class _EncodedAssignmentModel:
    """Fake MFA that assigns each activation to the integer in x[:, 0]."""

    def __init__(self, K: int, D: int):
        self.K = int(K)
        self.D = int(D)
        self.q = 0

    def to(self, device: str | torch.device):
        return self

    def eval(self) -> None:
        return None

    @contextmanager
    def inference_cache(self, enabled: bool = True):
        yield

    def responsibilities(self, x: torch.Tensor) -> torch.Tensor:
        labels = x[:, 0].round().long()
        r = torch.zeros((labels.numel(), self.K), dtype=torch.float32, device=x.device)
        r[torch.arange(labels.numel(), device=x.device), labels] = 1.0
        return r


def _canonical_encoded_assignments(root: Path, batch_size: int) -> torch.Tensor:
    ds = ActivationBatchDataset(
        root,
        layer=LAYER,
        batch_size=batch_size,
        shuffle_shards=False,
        shuffle_within_shard=False,
    )
    return torch.stack([ds[i] for i in range(ds.num_items)], dim=0)[:, 0].round().long()


def _compute_encoded_assignments(
    root: Path,
    *,
    batch_size: int,
    num_workers: int,
) -> torch.Tensor:
    canonical = _canonical_encoded_assignments(root, batch_size=batch_size)
    fake_model = _EncodedAssignmentModel(
        K=int(canonical.max().item()) + 1,
        D=ActivationBatchDataset(root, layer=LAYER, batch_size=batch_size).d_model,
    )

    ds = ActivationBatchDataset(
        root,
        layer=LAYER,
        batch_size=batch_size,
        shuffle_shards=False,
        shuffle_within_shard=False,
    )
    loader = DataLoader(ds, batch_size=None, num_workers=num_workers)
    with mock.patch.object(cluster_assignments, "load_mfa", return_value=fake_model):
        _sizes, assignments, _max_resp, _peakedness = cluster_assignments.compute_assignments(
            Path("fake_mfa_model.pt"),
            loader,
            device="cpu",
            use_inference_cache=False,
        )
    return assignments


class AssignmentWorkerOrderTests(unittest.TestCase):
    def test_compute_assignments_follows_canonical_order_with_num_workers_0(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = build_multi_shard(Path(tmp) / "multi_shard")
            batch_size = 5

            expected = _canonical_encoded_assignments(root, batch_size=batch_size)
            actual = _compute_encoded_assignments(
                root,
                batch_size=batch_size,
                num_workers=0,
            )

            self.assertTrue(torch.equal(actual, expected))

    @unittest.expectedFailure
    def test_compute_assignments_follows_canonical_order_with_num_workers_gt_0(self):
        """Documents that multi-worker assignment streaming is not canonical."""
        with tempfile.TemporaryDirectory() as tmp:
            root = build_multi_shard(Path(tmp) / "multi_shard")
            batch_size = 5

            expected = _canonical_encoded_assignments(root, batch_size=batch_size)
            actual = _compute_encoded_assignments(
                root,
                batch_size=batch_size,
                num_workers=2,
            )

            self.assertTrue(torch.equal(actual, expected))


if __name__ == "__main__":
    unittest.main()
