"""Tests for nearest-centroid activation assignments."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from dalg.analysis.nearest_centroid_assignments import compute_nearest_centroid_assignments  # noqa: E402


class NearestCentroidAssignmentTests(unittest.TestCase):
    def test_known_assignments_from_tensor(self):
        centroids = torch.tensor([
            [0.0, 0.0],
            [10.0, 0.0],
            [0.0, 10.0],
        ])
        x = torch.tensor([
            [0.1, 0.0],
            [9.0, 0.0],
            [0.0, 9.0],
            [8.0, 1.0],
            [1.0, 8.0],
        ])
        sizes, assignments, distances = compute_nearest_centroid_assignments(
            centroids,
            x,
            device="cpu",
            batch_size=2,
        )
        self.assertTrue(torch.equal(assignments, torch.tensor([0, 1, 2, 1, 2])))
        self.assertTrue(torch.equal(sizes, torch.tensor([1, 2, 2])))
        self.assertEqual(distances.shape, (5,))

    def test_accepts_loader_batches(self):
        centroids = torch.tensor([[0.0], [5.0]])
        loader = [torch.tensor([[0.0], [1.0]]), torch.tensor([[4.0], [6.0]])]
        sizes, assignments, _distances = compute_nearest_centroid_assignments(
            centroids,
            loader,
            device="cpu",
            batch_size=10,
        )
        self.assertTrue(torch.equal(assignments, torch.tensor([0, 0, 1, 1])))
        self.assertTrue(torch.equal(sizes, torch.tensor([2, 2])))


if __name__ == "__main__":
    unittest.main()
