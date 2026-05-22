"""Unit tests for ``ActivationBatchDataset`` in src/dalg/data/shard_activations.py."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from dalg.data.shard_activations import ActivationBatchDataset  # noqa: E402
from tests.synthetic_shards import (  # noqa: E402
    LAYER,
    build_multi_shard,
    build_single_shard,
)


FIXTURES = REPO_ROOT / "tests" / "fixtures"


# --------------------------------------------------------------------------
# __len__
# --------------------------------------------------------------------------


class LenTests(unittest.TestCase):
    def test_single_shard_drop_last_false(self):
        # 4 rows * 3 tokens = 12 items, batch_size=4 -> ceil(12/4) = 3
        root = build_single_shard(FIXTURES / "single_shard")
        ds = ActivationBatchDataset(root, layer=LAYER, batch_size=4, drop_last=False)
        self.assertEqual(ds.num_items, 12)
        self.assertEqual(len(ds), 3)

    def test_single_shard_drop_last_true_uneven_batch(self):
        # 12 items, batch_size=5 -> drop_last=True -> 12 // 5 = 2
        root = build_single_shard(FIXTURES / "single_shard")
        ds = ActivationBatchDataset(root, layer=LAYER, batch_size=5, drop_last=True)
        self.assertEqual(ds.num_items, 12)
        self.assertEqual(len(ds), 2)

    def test_multi_shard_drop_last_false(self):
        # 50 rows * 3 tokens = 150 items, batch_size=16 -> ceil(150/16) = 10
        root = build_multi_shard(FIXTURES / "multi_shard")
        ds = ActivationBatchDataset(root, layer=LAYER, batch_size=16, drop_last=False)
        self.assertEqual(ds.num_items, 150)
        self.assertEqual(len(ds), 10)

    def test_multi_shard_drop_last_true(self):
        # 150 items, batch_size=16, drop_last=True -> 150 // 16 = 9
        root = build_multi_shard(FIXTURES / "multi_shard")
        ds = ActivationBatchDataset(root, layer=LAYER, batch_size=16, drop_last=True)
        self.assertEqual(ds.num_items, 150)
        self.assertEqual(len(ds), 9)


# --------------------------------------------------------------------------
# _locate
# --------------------------------------------------------------------------
#
# Ground truth for canonical flat order:
#   iterate shards in sorted order; within a shard, iterate rows in sorted
#   row_in_shard order; within a row, emit tokens drop_prefix..window-1.
# batch_size has no effect on _locate (only __iter__/__len__ use it).


class LocateTests(unittest.TestCase):
    def test_single_shard_drop_prefix_0(self):
        # 4 rows * 3 tokens = 12 items
        root = build_single_shard(FIXTURES / "single_shard")
        ds = ActivationBatchDataset(root, layer=LAYER, batch_size=4, drop_prefix=0)
        self.assertEqual(ds.num_items, 12)

        self.assertEqual(ds._locate(0),  (0, 0, 0, 0))
        self.assertEqual(ds._locate(5),  (0, 1, 1, 2))
        self.assertEqual(ds._locate(6),  (0, 2, 2, 0))
        self.assertEqual(ds._locate(11), (0, 3, 3, 2))

    def test_single_shard_drop_prefix_1(self):
        # 4 rows * 2 tokens = 8 items, token positions are 1 and 2
        root = build_single_shard(FIXTURES / "single_shard")
        ds = ActivationBatchDataset(root, layer=LAYER, batch_size=4, drop_prefix=1)
        self.assertEqual(ds.num_items, 8)

        self.assertEqual(ds._locate(0), (0, 0, 0, 1))
        self.assertEqual(ds._locate(1), (0, 0, 0, 2))
        self.assertEqual(ds._locate(7), (0, 3, 3, 2))

    def test_single_shard_drop_prefix_2(self):
        # 4 rows * 1 token = 4 items, only token position 2 survives
        root = build_single_shard(FIXTURES / "single_shard")
        ds = ActivationBatchDataset(root, layer=LAYER, batch_size=4, drop_prefix=2)
        self.assertEqual(ds.num_items, 4)

        self.assertEqual(ds._locate(0), (0, 0, 0, 2))
        self.assertEqual(ds._locate(3), (0, 3, 3, 2))

    def test_multi_shard_drop_prefix_0(self):
        # 10 shards * 5 rows * 3 tokens = 150 items, 15 per shard
        root = build_multi_shard(FIXTURES / "multi_shard")
        ds = ActivationBatchDataset(root, layer=LAYER, batch_size=8, drop_prefix=0)
        self.assertEqual(ds.num_items, 150)

        self.assertEqual(ds._locate(0),   (0, 0, 0, 0))
        self.assertEqual(ds._locate(14),  (0, 4, 4, 2))   # last item of shard 0
        self.assertEqual(ds._locate(15),  (1, 0, 5, 0))   # first item of shard 1
        self.assertEqual(ds._locate(44),  (2, 4, 14, 2))
        self.assertEqual(ds._locate(149), (9, 4, 49, 2))

    def test_multi_shard_drop_prefix_1(self):
        # 10 shards * 5 rows * 2 tokens = 100 items, 10 per shard
        root = build_multi_shard(FIXTURES / "multi_shard")
        ds = ActivationBatchDataset(root, layer=LAYER, batch_size=8, drop_prefix=1)
        self.assertEqual(ds.num_items, 100)

        self.assertEqual(ds._locate(9),  (0, 4, 4, 2))   # last item of shard 0
        self.assertEqual(ds._locate(10), (1, 0, 5, 1))   # first item of shard 1

    def test_multi_shard_drop_prefix_2(self):
        # 10 shards * 5 rows * 1 token = 50 items, 5 per shard
        root = build_multi_shard(FIXTURES / "multi_shard")
        ds = ActivationBatchDataset(root, layer=LAYER, batch_size=8, drop_prefix=2)
        self.assertEqual(ds.num_items, 50)

        self.assertEqual(ds._locate(5),  (1, 0, 5, 2))
        self.assertEqual(ds._locate(49), (9, 4, 49, 2))

    def test_locate_invariant_to_batch_size(self):
        # _locate only depends on shard layout + drop_prefix; varying batch_size
        # must not change any of the (shard, row, global_row, tok_pos) tuples.
        root = build_multi_shard(FIXTURES / "multi_shard")
        ds_small = ActivationBatchDataset(root, layer=LAYER, batch_size=1,   drop_prefix=0)
        ds_large = ActivationBatchDataset(root, layer=LAYER, batch_size=64,  drop_prefix=0)

        self.assertEqual(ds_small.num_items, ds_large.num_items)
        for i in range(ds_small.num_items):
            self.assertEqual(ds_small._locate(i), ds_large._locate(i))


# --------------------------------------------------------------------------
# __getitem__
# --------------------------------------------------------------------------
#
# __getitem__ is the canonical (unshuffled) random-access view: dataset[i]
# loads the same activation that _locate(i) points to. Ground truth is built
# by independently torch.load-ing the shard tensor and indexing it directly.


class GetItemTests(unittest.TestCase):
    def test_matches_independent_disk_load(self):
        # For every i in [0, num_items): use _locate to find (shard, row, tok),
        # load that shard from disk, and confirm shard[row, tok] == ds[i].
        root = build_multi_shard(FIXTURES / "multi_shard")
        ds = ActivationBatchDataset(root, layer=LAYER, batch_size=8, drop_prefix=0)

        shard_cache: dict[int, torch.Tensor] = {}
        for i in range(ds.num_items):
            shard_i, row_in_shard, _gr, tok_pos = ds._locate(i)
            if shard_i not in shard_cache:
                shard_cache[shard_i] = torch.load(
                    root / f"layer{LAYER:02d}" / f"shard_{shard_i:05d}.pt",
                    weights_only=True,
                )
            expected = shard_cache[shard_i][row_in_shard, tok_pos].to(ds.dtype)
            self.assertTrue(torch.equal(ds[i], expected), msg=f"mismatch at i={i}")

    def test_returns_shape_and_dtype(self):
        root = build_multi_shard(FIXTURES / "multi_shard")
        ds = ActivationBatchDataset(root, layer=LAYER, batch_size=8)
        x = ds[0]
        self.assertEqual(x.shape, (ds.d_model,))
        self.assertEqual(x.dtype, ds.dtype)

    def test_negative_index_wraps(self):
        # ds[-1] == ds[num_items - 1]; ds[-num_items] == ds[0]
        root = build_multi_shard(FIXTURES / "multi_shard")
        ds = ActivationBatchDataset(root, layer=LAYER, batch_size=8)
        self.assertTrue(torch.equal(ds[-1], ds[ds.num_items - 1]))
        self.assertTrue(torch.equal(ds[-ds.num_items], ds[0]))

    def test_out_of_range_raises(self):
        root = build_multi_shard(FIXTURES / "multi_shard")
        ds = ActivationBatchDataset(root, layer=LAYER, batch_size=8)
        with self.assertRaises(IndexError):
            ds[ds.num_items]
        with self.assertRaises(IndexError):
            ds[-ds.num_items - 1]

    def test_accepts_tensor_index(self):
        # __getitem__ unwraps a 0-d torch.Tensor index via .item().
        root = build_multi_shard(FIXTURES / "multi_shard")
        ds = ActivationBatchDataset(root, layer=LAYER, batch_size=8)
        idx = torch.tensor(7)
        self.assertTrue(torch.equal(ds[idx], ds[7]))

    def test_return_metadata_triple(self):
        # With return_metadata=True, ds[i] -> (x, global_row, tok_pos) matching _locate.
        root = build_multi_shard(FIXTURES / "multi_shard")
        ds = ActivationBatchDataset(
            root, layer=LAYER, batch_size=8, return_metadata=True,
        )
        for i in [0, 14, 15, 73, ds.num_items - 1]:
            x, gr, tp = ds[i]
            _shard_i, _row, exp_gr, exp_tp = ds._locate(i)
            self.assertEqual(x.shape, (ds.d_model,))
            self.assertEqual(gr, exp_gr)
            self.assertEqual(tp, exp_tp)

    def test_drop_prefix_shifts_tok_pos(self):
        # With drop_prefix=1, ds[0] is the (shard 0, row 0, tok 1) activation,
        # i.e. NOT the same tensor as the drop_prefix=0 ds[0].
        root = build_multi_shard(FIXTURES / "multi_shard")
        ds0 = ActivationBatchDataset(root, layer=LAYER, batch_size=8, drop_prefix=0, return_metadata=True)
        ds1 = ActivationBatchDataset(root, layer=LAYER, batch_size=8, drop_prefix=1, return_metadata=True)

        _, gr0, tp0 = ds0[0]
        _, gr1, tp1 = ds1[0]
        self.assertEqual((gr0, tp0), (0, 0))
        self.assertEqual((gr1, tp1), (0, 1))


# --------------------------------------------------------------------------
# __iter__
# --------------------------------------------------------------------------
#
# Multi-shard fixture encoding (recap): activation[s, r, t, d] = s*1000 + r*100
# + t*10 + d. So `xb[:, 0]` uniquely identifies (shard, row_in_shard, tok) and
# we use that as the key for coverage/multiset checks.


def _no_shuffle_ds(root, **kwargs):
    return ActivationBatchDataset(
        root, layer=LAYER,
        shuffle_shards=False, shuffle_within_shard=False,
        **kwargs,
    )


class IterTests(unittest.TestCase):
    def test_unshuffled_yields_canonical_order(self):
        # With both shuffles off, concatenated batches must equal the canonical
        # stacking dataset[i] for i in 0..num_items-1 (which goes through _locate).
        root = build_multi_shard(FIXTURES / "multi_shard")
        ds = _no_shuffle_ds(root, batch_size=8)

        from_iter = torch.cat(list(ds), dim=0)
        from_getitem = torch.stack([ds[i] for i in range(ds.num_items)], dim=0)
        self.assertTrue(torch.equal(from_iter, from_getitem))

    def test_shuffled_covers_every_item_exactly_once(self):
        # Multiset of activations under shuffling matches the canonical multiset.
        root = build_multi_shard(FIXTURES / "multi_shard")
        ds_shuf = ActivationBatchDataset(
            root, layer=LAYER, batch_size=8,
            shuffle_shards=True, shuffle_within_shard=True, seed=123,
        )
        ds_canon = _no_shuffle_ds(root, batch_size=8)

        shuf_keys = sorted(torch.cat(list(ds_shuf))[:, 0].tolist())
        canon_keys = sorted(torch.cat(list(ds_canon))[:, 0].tolist())
        self.assertEqual(shuf_keys, canon_keys)

    def test_batch_shapes_drop_last_false(self):
        # 15 items per shard, batch_size=8 -> shards emit (8) + (7); 10 shards
        # -> 20 batches, 150 total items, last batch of each shard size 7.
        root = build_multi_shard(FIXTURES / "multi_shard")
        ds = _no_shuffle_ds(root, batch_size=8, drop_last=False)

        batches = list(ds)
        self.assertEqual(len(batches), 20)
        self.assertEqual(sum(b.shape[0] for b in batches), 150)
        sizes = [b.shape[0] for b in batches]
        self.assertEqual(sizes, [8, 7] * 10)
        for b in batches:
            self.assertEqual(b.shape[1], ds.d_model)

    def test_drop_last_trims_each_shard_independently(self):
        # 15 items per shard, batch_size=4 -> 15 // 4 = 3 full batches per shard,
        # 3 items dropped per shard. 10 shards -> 30 batches, 120 items kept.
        root = build_multi_shard(FIXTURES / "multi_shard")
        ds = _no_shuffle_ds(root, batch_size=4, drop_last=True)

        batches = list(ds)
        self.assertEqual(len(batches), 30)
        self.assertTrue(all(b.shape[0] == 4 for b in batches))
        self.assertEqual(sum(b.shape[0] for b in batches), 120)

    def test_return_metadata_aligns_with_activation_values(self):
        # For every (xb[i], global_row[i], tok_pos[i]) yielded by the shuffled
        # iterator: look up the on-disk shard+row from meta_index, load the raw
        # activation independently, and confirm it matches xb[i] exactly.
        from dalg.data.shard_activations import load_meta_index

        root = build_multi_shard(FIXTURES / "multi_shard")
        meta = load_meta_index(root, layer=LAYER)
        gr_to_shard_row = {m["global_row"]: (m["shard"], m["row_in_shard"]) for m in meta}

        ds = ActivationBatchDataset(
            root, layer=LAYER, batch_size=8,
            shuffle_shards=True, shuffle_within_shard=True, seed=7,
            return_metadata=True,
        )

        seen = 0
        for xb, global_rows, tok_pos in ds:
            self.assertEqual(xb.shape[0], global_rows.shape[0])
            self.assertEqual(xb.shape[0], tok_pos.shape[0])

            for i in range(xb.shape[0]):
                gr = int(global_rows[i])
                tp = int(tok_pos[i])
                shard_i, row_in_shard = gr_to_shard_row[gr]

                shard_tensor = torch.load(
                    root / f"layer{LAYER:02d}" / f"shard_{shard_i:05d}.pt",
                    weights_only=True,
                )
                expected = shard_tensor[row_in_shard, tp].to(xb.dtype)
                self.assertTrue(torch.equal(xb[i], expected),
                                msg=f"mismatch at batch item {i}: gr={gr}, tok={tp}")
            seen += xb.shape[0]
        self.assertEqual(seen, ds.num_items)

    def test_iteration_is_deterministic_with_same_seed_and_epoch(self):
        root = build_multi_shard(FIXTURES / "multi_shard")
        kwargs = dict(layer=LAYER, batch_size=8, seed=42)

        a = list(ActivationBatchDataset(root, **kwargs))
        b = list(ActivationBatchDataset(root, **kwargs))
        self.assertEqual(len(a), len(b))
        for xa, xb in zip(a, b):
            self.assertTrue(torch.equal(xa, xb))

    def test_set_epoch_changes_order(self):
        # Different epoch -> different shard/within-shard shuffles, so at least
        # one batch must differ. (Same multiset though — that's covered above.)
        root = build_multi_shard(FIXTURES / "multi_shard")
        ds = ActivationBatchDataset(root, layer=LAYER, batch_size=8, seed=42)

        ds.set_epoch(0)
        ep0 = list(ds)
        ds.set_epoch(1)
        ep1 = list(ds)

        self.assertEqual(len(ep0), len(ep1))
        differs = any(not torch.equal(a, b) for a, b in zip(ep0, ep1))
        self.assertTrue(differs)


if __name__ == "__main__":
    unittest.main()
