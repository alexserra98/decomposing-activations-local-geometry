from __future__ import annotations

import json
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from dalg.analysis.description_metrics import (
    compute_description_semantics,
    compute_detection_scores,
    compute_gaussian_group_label_coherence,
    compute_token_embedding_scores,
    load_labeled_clusters,
    select_detection_examples,
)
from dalg.cli.run_metrics import build_parser, validate_args


def _write_labels(path: Path) -> None:
    data = {
        "metadata": {"source": "test"},
        "clusters": {
            "0": {
                "label": "colon punctuation",
                "description": "Targets are colon punctuation used as separators.",
                "top_tokens": [{"token": ":", "count": 2}],
                "examples": [
                    {
                        "rank": 1,
                        "responsibility": 0.9,
                        "token": ":",
                        "snippet": "Name<target>:</target> value",
                    },
                    {
                        "rank": 2,
                        "responsibility": 0.8,
                        "token": ":",
                        "snippet": "Time<target>:</target> 12pm",
                    },
                ],
            },
            "1": {
                "label": "capitalized place names",
                "description": "Targets are capitalized geographic names.",
                "top_tokens": [{"token": "France", "count": 1}],
                "examples": [
                    {
                        "rank": 1,
                        "responsibility": 0.95,
                        "token": "France",
                        "snippet": "in <target>France</target> today",
                    }
                ],
            },
            "2": {
                "label": "semicolon punctuation",
                "description": "Targets are punctuation separators.",
                "top_tokens": [{"token": ";", "count": 1}],
                "examples": [
                    {
                        "rank": 1,
                        "responsibility": 0.7,
                        "token": ";",
                        "snippet": "first<target>;</target> second",
                    }
                ],
            },
        },
    }
    path.write_text(json.dumps(data))


class FakeCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        prompt = kwargs["messages"][1]["content"]
        example_ids = re.findall(r"example_id: (\S+)", prompt)
        examples = []
        for example_id in example_ids:
            examples.append({
                "example_id": example_id,
                "activates": "_pos" in example_id,
                "confidence": 0.9,
                "reason": "synthetic response",
            })
        content = json.dumps({"examples": examples, "summary": "ok"})
        message = SimpleNamespace(content=content)
        choice = SimpleNamespace(message=message)
        return SimpleNamespace(choices=[choice])


class FakeClient:
    def __init__(self):
        self.completions = FakeCompletions()
        self.chat = SimpleNamespace(completions=self.completions)


class FakeEmbedder:
    model_name = "fake-embedder"

    def encode_texts(self, texts, *, batch_size=32):
        del batch_size
        vectors = []
        for text in texts:
            text = text.lower()
            if "colon" in text or "semicolon" in text or "punctuation" in text:
                vectors.append(torch.tensor([1.0, 0.0]))
            else:
                vectors.append(torch.tensor([0.0, 1.0]))
        return torch.nn.functional.normalize(torch.stack(vectors), p=2, dim=1)

    def encode_target_tokens(self, snippets, *, batch_size=16):
        del batch_size
        vectors = []
        infos = []
        for snippet in snippets:
            if "<target>:</target>" in snippet or "<target>;</target>" in snippet:
                vectors.append(torch.tensor([1.0, 0.0]))
            else:
                vectors.append(torch.tensor([0.0, 1.0]))
            infos.append({"target_text": "x", "matched_token_positions": [1], "matched_target_span": True})
        return torch.nn.functional.normalize(torch.stack(vectors), p=2, dim=1), infos


class DescriptionMetricsTests(unittest.TestCase):
    def test_load_and_select_detection_examples(self):
        with tempfile.TemporaryDirectory() as tmp:
            labels_path = Path(tmp) / "cluster_labels.json"
            _write_labels(labels_path)
            loaded = load_labeled_clusters(labels_path)

            examples = select_detection_examples(
                loaded["clusters"],
                0,
                positive_examples=1,
                negative_examples=2,
                seed=123,
            )

        self.assertEqual(sum(ex["expected_activates"] for ex in examples), 1)
        self.assertEqual(sum(not ex["expected_activates"] for ex in examples), 2)
        self.assertTrue(any(ex["snippet"] == "Name<target>:</target> value" for ex in examples))

    def test_compute_detection_scores_with_fake_judge(self):
        with tempfile.TemporaryDirectory() as tmp:
            labels_path = Path(tmp) / "cluster_labels.json"
            _write_labels(labels_path)

            client = FakeClient()
            out = compute_detection_scores(
                labels_path,
                client=client,
                model="fake",
                positive_examples=1,
                negative_examples=1,
                cluster_ids=[0],
                show_progress=False,
            )

        cluster = out["clusters"]["0"]
        self.assertEqual(cluster["metrics"]["true_positive"], 1)
        self.assertEqual(cluster["metrics"]["true_negative"], 1)
        self.assertEqual(cluster["metrics"]["detection_score"], 1.0)
        prompt = client.completions.calls[0]["messages"][1]["content"]
        self.assertIn("marked target token", prompt)

    def test_token_embedding_scores_compare_positive_and_negative_examples(self):
        with tempfile.TemporaryDirectory() as tmp:
            labels_path = Path(tmp) / "cluster_labels.json"
            _write_labels(labels_path)

            out = compute_token_embedding_scores(
                labels_path,
                embedder=FakeEmbedder(),
                positive_examples=1,
                negative_examples=1,
                cluster_ids=[1],
            )

        metrics = out["clusters"]["1"]["metrics"]
        self.assertGreater(metrics["positive_mean_cosine"], metrics["negative_mean_cosine"])
        self.assertGreater(metrics["positive_minus_negative"], 0.0)
        self.assertIn("rough diagnostic", out["metadata"]["note"])

    def test_description_semantics_groups_similar_descriptions(self):
        with tempfile.TemporaryDirectory() as tmp:
            labels_path = Path(tmp) / "cluster_labels.json"
            _write_labels(labels_path)

            tensors, groups = compute_description_semantics(
                labels_path,
                embedder=FakeEmbedder(),
                top_k=2,
                similarity_threshold=0.8,
                save_full_matrix=True,
            )

        self.assertIn("similarity_matrix", tensors)
        grouped_ids = [set(group["cluster_ids"]) for group in groups["groups"]]
        self.assertIn({0, 2}, grouped_ids)

    def test_gaussian_group_label_coherence_uses_distance_clusters(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            labels_path = root / "cluster_labels.json"
            gaussian_overlap_path = root / "gaussian_overlap.pt"
            _write_labels(labels_path)
            torch.save({
                "db": torch.tensor([
                    [0.0, 4.0, 0.1],
                    [4.0, 0.0, 5.0],
                    [0.1, 5.0, 0.0],
                ])
            }, gaussian_overlap_path)

            out = compute_gaussian_group_label_coherence(
                labels_path,
                gaussian_overlap_path,
                embedder=FakeEmbedder(),
                distance_threshold=0.2,
                linkage="average",
                top_groups=10,
            )

        self.assertEqual(out["summary"]["n_gaussian_groups"], 2)
        self.assertEqual(out["summary"]["n_non_singleton_gaussian_groups"], 1)
        self.assertEqual(out["groups"][0]["component_ids"], [0, 2])
        self.assertEqual(out["groups"][0]["label_cosine"]["count"], 1)
        self.assertGreater(out["groups"][0]["label_cosine"]["mean"], 0.99)
        self.assertAlmostEqual(out["groups"][0]["distance"]["mean"], 0.1, places=6)
        self.assertEqual(
            out["metadata"]["gaussian_overlap_path"],
            str(gaussian_overlap_path),
        )

    def test_run_metrics_parser_accepts_gaussian_overlap_and_description_commands(self):
        parser = build_parser()

        args = parser.parse_args([
            "gaussian-overlap",
            "--data-dir",
            "mfa_run",
        ])
        validate_args(args)
        self.assertEqual(args.command, "gaussian-overlap")

        args = parser.parse_args([
            "description-fit",
            "--labels-path",
            "cluster_labels.json",
            "--skip-detection",
        ])
        validate_args(args)
        self.assertEqual(args.command, "description-fit")

        bad = parser.parse_args([
            "description-fit",
            "--labels-path",
            "cluster_labels.json",
            "--skip-detection",
            "--skip-token-embedding",
        ])
        with self.assertRaises(SystemExit):
            validate_args(bad)

        args = parser.parse_args([
            "description-semantics",
            "--labels-path",
            "cluster_labels.json",
            "--full-matrix",
            "never",
        ])
        validate_args(args)
        self.assertEqual(args.command, "description-semantics")

        args = parser.parse_args([
            "gaussian-group-semantics",
            "--labels-path",
            "cluster_labels.json",
            "--gaussian-overlap-path",
            "gaussian_overlap.pt",
            "--distance-threshold",
            "0.5",
        ])
        validate_args(args)
        self.assertEqual(args.command, "gaussian-group-semantics")


if __name__ == "__main__":
    unittest.main()
