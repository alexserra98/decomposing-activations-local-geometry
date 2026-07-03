"""Unit tests for the throwaway subset-spec slice selector.

Covers `src/dalg/data/subset_spec.py` (suffix split, token->row math,
determinism) and the optional `positions=` path of `stratified_split`.
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from dalg.data.shard_activations import stratified_split  # noqa: E402
from dalg.data.subset_spec import (  # noqa: E402
    resolve_spec_positions,
    split_shard_dir_spec,
)


WINDOW = 256
DROP_PREFIX = 32
TOKENS_PER_ROW = WINDOW - DROP_PREFIX  # 224


def _meta(n_wiki: int, n_other: int) -> list[dict]:
    """Synthetic meta_index: wiki rows interleaved with another subset."""
    rows: list[dict] = []
    gr = 0
    for i in range(max(n_wiki, n_other)):
        if i < n_wiki:
            rows.append({"shard": 0, "row_in_shard": gr, "global_row": gr,
                         "subset": "pile-wikipedia_en"})
            gr += 1
        if i < n_other:
            rows.append({"shard": 0, "row_in_shard": gr, "global_row": gr,
                         "subset": "pile-arxiv"})
            gr += 1
    return rows


class SplitShardDirSpecTests(unittest.TestCase):
    def test_splits_suffix(self):
        path, spec = split_shard_dir_spec("a/b#pile_wikipedia_100K")
        self.assertEqual(path, Path("a/b"))
        self.assertEqual(spec, "pile_wikipedia_100K")

    def test_no_suffix_returns_none(self):
        path, spec = split_shard_dir_spec("a/b")
        self.assertEqual(path, Path("a/b"))
        self.assertIsNone(spec)


class ResolveSpecPositionsTests(unittest.TestCase):
    def test_none_spec_returns_all_positions(self):
        meta = _meta(10, 10)
        pos = resolve_spec_positions(meta, None, window=WINDOW, drop_prefix=DROP_PREFIX)
        self.assertEqual(pos, list(range(len(meta))))

    def test_filters_to_subset_and_token_budget(self):
        meta = _meta(1000, 1000)
        n_tokens = 100_000
        pos = resolve_spec_positions(
            meta, "pile_wikipedia_100K", window=WINDOW, drop_prefix=DROP_PREFIX
        )
        expected_rows = math.ceil(n_tokens / TOKENS_PER_ROW)  # 447
        self.assertEqual(len(pos), expected_rows)
        # every selected row is a wikipedia row
        self.assertTrue(all(meta[i]["subset"] == "pile-wikipedia_en" for i in pos))
        # sorted (canonical streaming order)
        self.assertEqual(pos, sorted(pos))

    def test_deterministic_across_calls(self):
        meta = _meta(1000, 1000)
        a = resolve_spec_positions(meta, "pile_wikipedia_1M", window=WINDOW, drop_prefix=DROP_PREFIX)
        b = resolve_spec_positions(meta, "pile_wikipedia_1M", window=WINDOW, drop_prefix=DROP_PREFIX)
        self.assertEqual(a, b)

    def test_budget_exceeding_pool_returns_whole_pool(self):
        meta = _meta(50, 50)  # 50 wiki rows = 11_200 tokens < 1M
        pos = resolve_spec_positions(meta, "pile_wikipedia_1M", window=WINDOW, drop_prefix=DROP_PREFIX)
        wiki = [i for i, m in enumerate(meta) if m["subset"] == "pile-wikipedia_en"]
        self.assertEqual(pos, wiki)

    def test_unknown_spec_raises(self):
        meta = _meta(10, 10)
        with self.assertRaises(ValueError):
            resolve_spec_positions(meta, "not_a_spec", window=WINDOW, drop_prefix=DROP_PREFIX)

    def test_missing_subset_raises(self):
        meta = _meta(0, 10)  # only arxiv rows
        with self.assertRaises(ValueError):
            resolve_spec_positions(meta, "pile_wikipedia_100K", window=WINDOW, drop_prefix=DROP_PREFIX)


class StratifiedSplitPositionsTests(unittest.TestCase):
    def test_split_restricted_to_given_positions(self):
        meta = _meta(1000, 1000)
        keep = resolve_spec_positions(
            meta, "pile_wikipedia_100K", window=WINDOW, drop_prefix=DROP_PREFIX
        )
        train, val = stratified_split(meta, val_frac=0.1, seed=42, positions=keep)
        # original indices preserved and confined to keep
        self.assertEqual(sorted(train + val), keep)
        # all selected rows belong to the wikipedia subset
        self.assertTrue(all(meta[i]["subset"] == "pile-wikipedia_en" for i in train + val))
        self.assertGreater(len(val), 0)


if __name__ == "__main__":
    unittest.main()
