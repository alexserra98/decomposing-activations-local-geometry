"""Unit tests for `load_meta_index` in src/dalg/data/shard_activations.py."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from dalg.data.shard_activations import load_meta_index  # noqa: E402
from tests.synthetic_shards import (  # noqa: E402
    LAYER,
    build_multi_shard,
    build_single_shard,
)


FIXTURES = REPO_ROOT / "tests" / "fixtures"


class LoadMetaIndexTests(unittest.TestCase):
    def test_single_shard(self):
        root = build_single_shard(FIXTURES / "single_shard")
        meta = load_meta_index(root, layer=LAYER)

        expected = [
            {"shard": 0, "row_in_shard": 0, "global_row": 0, "subset": "A"},
            {"shard": 0, "row_in_shard": 1, "global_row": 1, "subset": "B"},
            {"shard": 0, "row_in_shard": 2, "global_row": 2, "subset": "A"},
            {"shard": 0, "row_in_shard": 3, "global_row": 3, "subset": "B"},
        ]
        self.assertEqual(meta, expected)

    def test_multi_shard_length_and_order(self):
        root = build_multi_shard(FIXTURES / "multi_shard")
        meta = load_meta_index(root, layer=LAYER)

        self.assertEqual(len(meta), 10 * 5)

        # rows in canonical shard-then-row order
        for s in range(10):
            for r in range(5):
                entry = meta[s * 5 + r]
                self.assertEqual(entry["shard"], s)
                self.assertEqual(entry["row_in_shard"], r)
                self.assertEqual(entry["global_row"], s * 5 + r)
                self.assertEqual(entry["subset"], "A" if r % 2 == 0 else "B")

    def test_multi_shard_subset_counts(self):
        root = build_multi_shard(FIXTURES / "multi_shard")
        meta = load_meta_index(root, layer=LAYER)

        counts = {"A": 0, "B": 0}
        for row in meta:
            counts[row["subset"]] += 1
        # 3 A's and 2 B's per shard, 10 shards
        self.assertEqual(counts, {"A": 30, "B": 20})


if __name__ == "__main__":
    unittest.main()
