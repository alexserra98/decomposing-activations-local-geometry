"""Standalone plots preserve training-only PCA/liveness and full test coverage."""

import importlib.util
import json
from pathlib import Path

import pytest
import torch

from dalg.data.shard_activations import ActivationBatchDataset
from dalg.evaluation.coverage import evaluate_heldout_distribution_coverage
from tests.test_kmeans_evaluation import _build_kmeans_run
from tests.test_toy_manifold_tiling import _build_evaluation_artifacts

spec = importlib.util.spec_from_file_location(
    "plot_toy_coverage", Path(__file__).parents[1] / "scripts/temporary/plot_toy_coverage.py",
)
plotting = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plotting)


@pytest.mark.parametrize("kind", ["kmeans", "mfa", "ard", "hddc"])
def test_plot_uses_training_activity_and_full_test_metrics(tmp_path, kind):
    if kind == "kmeans":
        run, root, *_ = _build_kmeans_run(tmp_path, "cluster", False)
        (run / "config.json").write_text(json.dumps({
            "model_kind": kind, "shard_dir": str(root), "layer": 0,
        }))
    else:
        run, root = _build_evaluation_artifacts(tmp_path, kind)
    stem = "kmeans_model" if kind == "kmeans" else "mfa_model"
    path = run / f"{stem}_assignments.pt"
    bundle = torch.load(path, weights_only=True)
    validation = json.loads((run / "val_indices.json").read_text())["val_global_rows"]
    bundle["assignments"].zero_()
    bundle["assignments"][validation] = 1
    bundle["cluster_sizes"] = torch.bincount(bundle["assignments"], minlength=bundle["K"])
    torch.save(bundle, path)
    before = {p: p.read_bytes() for p in run.iterdir() if p.is_file()}

    output = tmp_path / "figures" / "coverage.png"
    # Exercise both the one-panel and shared multi-panel renderer.
    runs = [(kind, run)] * (2 if kind == "kmeans" else 1)
    report = plotting.plot_coverage(runs, output, batch_size=7, max_points=3)
    data = torch.load(output.with_suffix(".pt"), weights_only=True)
    loaded = plotting._load_run(kind, run)
    test = torch.cat(list(ActivationBatchDataset(
        root / "test", layer=0, batch_size=7,
        shuffle_shards=False, shuffle_within_shard=False,
    )))
    expected, distances = evaluate_heldout_distribution_coverage(
        test, loaded["centroids"][:1], batch_size=7,
    )
    assert report["plotted_points"] == 3 < len(test)
    assert report["panels"][0]["coverage"]["reference_points"] == len(test)
    assert report["panels"][0]["coverage"]["components_live"] == 1
    assert report["panels"][0]["coverage"]["mean"] == expected["mean"]
    torch.testing.assert_close(data["panels"][0]["distances"], distances)
    train = torch.cat(list(plotting._stream(root, 0, 7, loaded["train"]))).double()
    torch.testing.assert_close(data["mean"], train.mean(0))
    _, vectors = torch.linalg.eigh(torch.cov(train.T))
    expected_basis = vectors[:, -2:]
    basis = data["basis"]
    torch.testing.assert_close(basis @ basis.T, expected_basis @ expected_basis.T)
    assert output.read_bytes().startswith(b"\x89PNG")
    assert before == {p: p.read_bytes() for p in run.iterdir() if p.is_file()}

    with pytest.raises(FileExistsError):
        plotting.plot_coverage(runs, output)
    split_path = run / "val_indices.json"
    split = json.loads(split_path.read_text())
    split["train_rows"] += 1
    split_path.write_text(json.dumps(split))
    with pytest.raises(ValueError, match="split does not match"):
        plotting._load_run(kind, run)
