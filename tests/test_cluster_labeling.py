from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from dalg.analysis.cluster_labeling import (
    build_context_examples,
    label_mfa_clusters,
    label_one_cluster,
    make_orfeo_client,
    map_positions_to_token_coordinates,
    parse_json_response,
    resolve_labeling_config,
    select_top_activations,
)


class FakeTokenizer:
    def decode(self, token_ids, skip_special_tokens=False):
        del skip_special_tokens
        return "".join(chr(ord("A") + int(t) - 1) for t in token_ids)


class FakeDataset:
    def __init__(self, rows):
        self.rows = rows

    def __getitem__(self, idx):
        return self.rows[idx]


class FakeCompletions:
    def __init__(self, content):
        self.content = content
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        message = SimpleNamespace(content=self.content)
        choice = SimpleNamespace(message=message)
        return SimpleNamespace(choices=[choice])


class FakeClient:
    def __init__(self, content):
        self.completions = FakeCompletions(content)
        self.chat = SimpleNamespace(completions=self.completions)


class ClusterLabelingTests(unittest.TestCase):
    def test_select_top_activations_per_cluster(self):
        assignments = torch.tensor([0, 1, 0, 1, 0, 2, 2])
        scores = torch.tensor([0.2, 0.9, 0.8, 0.3, 0.7, 0.5, 0.6])

        top = select_top_activations(
            assignments,
            scores,
            K=4,
            top_n=2,
            chunk_size=3,
            show_progress=False,
        )

        self.assertEqual(top["cluster_ids"].tolist(), [0, 1, 2, 3])
        self.assertEqual(top["positions"][0].tolist(), [2, 4])
        self.assertEqual(top["positions"][1].tolist(), [1, 3])
        self.assertEqual(top["positions"][2].tolist(), [6, 5])
        self.assertEqual(top["positions"][3].tolist(), [-1, -1])
        self.assertTrue(torch.allclose(top["responsibilities"][0], torch.tensor([0.8, 0.7])))

    def test_map_positions_and_recover_context_examples(self):
        top_index = {
            "cluster_ids": torch.tensor([0]),
            "positions": torch.tensor([[1, 3]]),
            "responsibilities": torch.tensor([[0.95, 0.90]]),
        }
        meta_index = [
            {"shard": 0, "row_in_shard": 0, "global_row": 10},
            {"shard": 0, "row_in_shard": 1, "global_row": 11},
        ]

        coords = map_positions_to_token_coordinates(
            top_index["positions"],
            meta_index,
            window=5,
            drop_prefix=2,
            assignment_count=6,
        )

        self.assertEqual(coords["global_row"].tolist(), [[10, 11]])
        self.assertEqual(coords["tok_pos"].tolist(), [[3, 2]])

        rows = [{"token_ids": [0, 0, 0, 0, 0]} for _ in range(12)]
        rows[10] = {"token_ids": [1, 2, 3, 4, 5]}
        rows[11] = {"token_ids": [6, 7, 8, 9, 10]}
        clusters = build_context_examples(
            top_index,
            coords,
            FakeDataset(rows),
            FakeTokenizer(),
            cluster_sizes=torch.tensor([2]),
            pad=1,
        )

        examples = clusters[0]["examples"]
        self.assertEqual(clusters[0]["cluster_size"], 2)
        self.assertEqual(examples[0]["token"], "D")
        self.assertEqual(examples[0]["snippet"], "C<target>D</target>E")
        self.assertEqual(examples[1]["token"], "H")
        self.assertEqual(examples[1]["snippet"], "G<target>H</target>I")
        self.assertEqual(clusters[0]["top_tokens"], [{"token": "D", "count": 1}, {"token": "H", "count": 1}])

    def test_resolve_labeling_config_from_neighboring_configs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shard_dir = root / "shards"
            mfa_dir = root / "layer05_1000_mfa"
            windows_dir = root / "windows"
            shard_dir.mkdir()
            mfa_dir.mkdir()
            windows_dir.mkdir()
            (mfa_dir / "config.json").write_text(json.dumps({
                "shard_dir": str(shard_dir),
                "layer": 5,
                "drop_prefix": 4,
            }))
            (shard_dir / "config.json").write_text(json.dumps({
                "window": 16,
                "drop_prefix": 3,
                "dataset": str(windows_dir),
            }))

            cfg = resolve_labeling_config(mfa_dir / "mfa_model_assignments.pt")

        self.assertEqual(cfg["shard_dir"], str(shard_dir))
        self.assertEqual(cfg["layer"], 5)
        self.assertEqual(cfg["window"], 16)
        self.assertEqual(cfg["drop_prefix"], 3)
        self.assertEqual(cfg["windows_dataset"], str(windows_dir))

    def test_resolve_labeling_config_falls_back_from_stale_scratch_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shard_dir = root / "pile_gemma2b_activations"
            mfa_dir = shard_dir / "layer05_2_mfa"
            windows_dir = root / "pile_gemma2b_100M_windows" / "merged"
            shard_dir.mkdir()
            mfa_dir.mkdir()
            windows_dir.mkdir(parents=True)

            stale_shard = root / "missing" / "pile_gemma2b_activations"
            stale_windows = root / "missing" / "pile_gemma2b_100M_windows" / "merged"
            (mfa_dir / "config.json").write_text(json.dumps({
                "shard_dir": str(stale_shard),
                "layer": 5,
            }))
            (shard_dir / "config.json").write_text(json.dumps({
                "window": 16,
                "drop_prefix": 3,
                "dataset": str(stale_windows),
            }))

            cfg = resolve_labeling_config(mfa_dir / "mfa_model_assignments.pt")

        self.assertEqual(cfg["shard_dir"], str(shard_dir))
        self.assertEqual(cfg["windows_dataset"], str(windows_dir))

    def test_llm_call_and_json_parsing(self):
        client = FakeClient(
            '```json\n{"label": "acronym expansion punctuation", '
            '"description": "Targets mark acronym expansions.", '
            '"evidence": "Examples contain colon-separated acronyms."}\n```'
        )
        cluster = {
            "cluster_id": 7,
            "cluster_size": 12,
            "top_tokens": [{"token": ":", "count": 2}],
            "examples": [
                {
                    "rank": 1,
                    "responsibility": 0.99,
                    "token": ":",
                    "snippet": "as soon as possible<target>:</target> ASAP",
                }
            ],
        }

        label = label_one_cluster(client, cluster, model="fake-model", max_tokens=64)

        self.assertEqual(label["label"], "acronym expansion punctuation")
        self.assertEqual(client.completions.calls[0]["model"], "fake-model")
        user_message = client.completions.calls[0]["messages"][1]["content"]
        self.assertIn("<target>:</target>", user_message)
        self.assertIn("Top target tokens", user_message)

    def test_parse_json_response_fallback(self):
        parsed = parse_json_response('Here is the label: {"label": "x", "description": "y", "evidence": "z"}')
        self.assertEqual(parsed["label"], "x")

    def test_label_mfa_clusters_synthetic_pipeline_writes_expected_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shard_dir = root / "shards"
            meta_dir = shard_dir / "meta"
            mfa_dir = root / "layer05_2_mfa"
            out_dir = root / "labels"
            windows_dir = root / "windows"
            meta_dir.mkdir(parents=True)
            mfa_dir.mkdir()
            windows_dir.mkdir()

            (shard_dir / "config.json").write_text(json.dumps({
                "window": 4,
                "drop_prefix": 1,
                "dataset": str(windows_dir),
            }))
            (meta_dir / "shard_00000.json").write_text(json.dumps({
                "row_indices": [0, 1],
                "rows": [{"subset": "a"}, {"subset": "a"}],
            }))
            (mfa_dir / "config.json").write_text(json.dumps({
                "shard_dir": str(shard_dir),
                "layer": 5,
                "K": 2,
            }))
            assignments_path = mfa_dir / "mfa_model_assignments.pt"
            torch.save({
                "cluster_sizes": torch.tensor([3, 3]),
                "assignments": torch.tensor([0, 1, 0, 1, 1, 0]),
                "max_responsibilities": torch.tensor([0.1, 0.4, 0.8, 0.3, 0.95, 0.9]),
                "K": 2,
            }, assignments_path)

            output = label_mfa_clusters(
                assignments_path,
                out_dir=out_dir,
                top_n=1,
                pad=1,
                skip_llm=True,
                show_progress=False,
                windows_dataset_obj=FakeDataset([
                    {"token_ids": [1, 2, 3, 4]},
                    {"token_ids": [5, 6, 7, 8]},
                ]),
                tokenizer=FakeTokenizer(),
                meta_index=[
                    {"shard": 0, "row_in_shard": 0, "global_row": 0},
                    {"shard": 0, "row_in_shard": 1, "global_row": 1},
                ],
            )

            cluster0 = output["clusters"]["0"]
            cluster1 = output["clusters"]["1"]
            self.assertIsNone(cluster0["label"])
            self.assertEqual(cluster0["examples"][0]["stream_position"], 5)
            self.assertEqual(cluster0["examples"][0]["snippet"], "G<target>H</target>")
            self.assertEqual(cluster1["examples"][0]["stream_position"], 4)
            self.assertEqual(cluster1["examples"][0]["snippet"], "F<target>G</target>H")
            self.assertTrue((out_dir / "top_activations.pt").exists())
            self.assertTrue((out_dir / "cluster_examples.json").exists())
            self.assertTrue((out_dir / "cluster_labels.json").exists())

    @unittest.skipUnless(
        os.getenv("RUN_ORFEO_LIVE_TEST") == "1" and os.getenv("ORFEO_API_KEY"),
        "set RUN_ORFEO_LIVE_TEST=1 and ORFEO_API_KEY to run the live Orfeo smoke test",
    )
    def test_live_orfeo_smoke(self):
        client = make_orfeo_client()
        cluster = {
            "cluster_id": 0,
            "cluster_size": 3,
            "top_tokens": [{"token": ":", "count": 3}],
            "examples": [
                {
                    "rank": 1,
                    "responsibility": 0.99,
                    "token": ":",
                    "snippet": "Personal Identification Number<target>:</target> PIN",
                }
            ],
        }
        label = label_one_cluster(client, cluster, max_tokens=128)
        self.assertIn("label", label)
        self.assertIn("description", label)


if __name__ == "__main__":
    unittest.main()
