"""Unit tests for `stratified_split` in src/dalg/data/shard_activations.py."""

from __future__ import annotations

import math
import sys
import unittest
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from dalg.data.shard_activations import load_meta_index, stratified_split  # noqa: E402
from tests.synthetic_shards import LAYER, build_multi_shard  # noqa: E402


FIXTURES = REPO_ROOT / "tests" / "fixtures"
VAL_FRAC = 0.05


class StratifiedSplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = build_multi_shard(FIXTURES / "multi_shard")
        cls.meta = load_meta_index(root, layer=LAYER)
        cls.train, cls.val = stratified_split(cls.meta, val_frac=VAL_FRAC, seed=42)

    def test_train_and_val_cover_all_positions_without_overlap(self):
        n = len(self.meta)
        self.assertEqual(len(self.train) + len(self.val), n)
        self.assertEqual(set(self.train) | set(self.val), set(range(n)))
        self.assertEqual(set(self.train) & set(self.val), set())

    def test_per_subset_proportion_matches_val_frac(self):
        # 30 A's and 2 B's → ceil(30*0.05)=2, ceil(20*0.05)=1
        subset_of = [row["subset"] for row in self.meta]
        total = Counter(subset_of)
        val_counts = Counter(subset_of[p] for p in self.val)

        for subset, n_total in total.items():
            expected_val = math.ceil(n_total * VAL_FRAC)
            self.assertEqual(val_counts[subset], expected_val,
                             msg=f"subset {subset!r}: expected {expected_val} val rows, got {val_counts[subset]}")

    def test_no_subset_starved(self):
        # Every subset appears in val (stratification, not random sampling).
        subset_of = [row["subset"] for row in self.meta]
        val_subsets = {subset_of[p] for p in self.val}
        self.assertEqual(val_subsets, set(subset_of))

    def test_outputs_are_sorted(self):
        self.assertEqual(self.train, sorted(self.train))
        self.assertEqual(self.val, sorted(self.val))


if __name__ == "__main__":
    unittest.main()
