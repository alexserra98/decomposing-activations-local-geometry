"""Unit tests for src/dalg/data/shard_activations.py."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from dalg.data.shard_activations import _resolve_activation_layout  # noqa: E402
from tests.synthetic_shards import (  # noqa: E402
    LAYER,
    build_multi_shard,
    build_single_shard,
)


FIXTURES = REPO_ROOT / "tests" / "fixtures"


class ResolveActivationLayoutTests(unittest.TestCase):
    def test_single_shard(self):
        root = build_single_shard(FIXTURES / "single_shard")
        layout = _resolve_activation_layout(root, layer=LAYER)

        self.assertEqual(layout["root"], root)
        self.assertEqual(layout["shard_paths"], [root / "layer05" / "shard_00000.pt"])
        self.assertEqual(layout["config"], {"window": 3, "d_model": 2, "drop_prefix": 0})

    def test_multi_shard_returns_sorted_paths(self):
        root = build_multi_shard(FIXTURES / "multi_shard")
        layout = _resolve_activation_layout(root, layer=LAYER)

        self.assertEqual(layout["root"], root)
        expected = [root / "layer05" / f"shard_{i:05d}.pt" for i in range(10)]
        self.assertEqual(layout["shard_paths"], expected)
        self.assertEqual(layout["config"], {"window": 3, "d_model": 2, "drop_prefix": 0})

    def test_missing_layer_dir_raises(self):
        root = build_single_shard(FIXTURES / "single_shard")
        with self.assertRaises(FileNotFoundError):
            _resolve_activation_layout(root, layer=99)


if __name__ == "__main__":
    unittest.main()
