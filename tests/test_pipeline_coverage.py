"""Coverage uses an independent test stream and training-only component activity."""

import copy
import json
import shutil
from pathlib import Path

import pytest
import torch

from dalg.data.manifold_dataset import ToyManifoldConfig, save_toy_manifold_shards
from dalg.data.shard_activations import ActivationBatchDataset
from dalg.evaluation.coverage import evaluate_heldout_distribution_coverage
from dalg.evaluation.toy_manifold_coverage import (
    coverage_report_valid, evaluate_toy_test_coverage, validate_toy_test_split,
)
from dalg.evaluation.toy_manifold_tiling import _load_model, evaluate_toy_manifold_tiling
from dalg.pipeline import _evaluation_metrics_valid, execute_run, resolve_run
from tests.test_kmeans_evaluation import _build_kmeans_run
from tests.test_kmeans_pipeline import _config
from tests.test_toy_manifold_test_split import add_test_split
from tests.test_toy_manifold_tiling import _build_evaluation_artifacts


def _points(root):
    return torch.cat(list(ActivationBatchDataset(
        root, layer=0, batch_size=7, shuffle_shards=False, shuffle_within_shard=False,
    )))


@pytest.mark.parametrize("kind", ["kmeans", "mfa", "ard", "hddc"])
def test_pipeline_coverage_matches_direct_calculation_and_excludes_nontraining_activity(tmp_path, kind):
    if kind == "kmeans":
        run_dir, root, *_ = _build_kmeans_run(tmp_path, "cluster", False)
    else:
        run_dir, root = _build_evaluation_artifacts(tmp_path, kind)
    model = _load_model(run_dir, kind)
    stem = "kmeans_model" if kind == "kmeans" else "mfa_model"
    path = run_dir / f"{stem}_assignments.pt"
    bundle = torch.load(path, weights_only=True)
    val = json.loads((run_dir / "val_indices.json").read_text())["val_global_rows"]
    labels = torch.zeros_like(bundle["assignments"])
    labels[val] = 1  # Component 1 is occupied only by validation rows.
    bundle.update(assignments=labels, cluster_sizes=torch.bincount(labels, minlength=model.K))
    torch.save(bundle, path)
    metrics = evaluate_toy_manifold_tiling(
        run_dir, shard_dir=root, layer=0, model_kind=kind, device="cpu", batch_size=7,
    )
    test_points = _points(root / "test")
    live = torch.arange(model.K) == 0
    expected, distances = evaluate_heldout_distribution_coverage(
        test_points, model.mu, live_components=live, thresholds=None, batch_size=7,
    )
    result = metrics["heldout_distribution_coverage"]
    for key, value in expected.items():
        assert result[key] == value
    assert result["components_live"] == 1
    assert metrics["components"]["live"] == 2
    # Test points lie closer to excluded means too, but cannot make them live.
    _, unrestricted = evaluate_heldout_distribution_coverage(test_points, model.mu)
    assert bool((unrestricted < distances).any())
    assert result["source"]["shard_dir"] == str(root / "test")
    assert coverage_report_valid(result, components=model.K)
    run = {"training": {"model_kind": kind, "arguments": {"K": model.K}},
           "dataset": {"shard_dir": str(root), "layer": 0},
           "evaluation": {"kind": "toy_manifold_tiling"}, "identity_hash": "test"}
    metrics["identity_hash"] = "test"
    assert _evaluation_metrics_valid(run, metrics)
    foreign = copy.deepcopy(metrics)
    foreign["heldout_distribution_coverage"]["source"]["shard_dir"] = str(tmp_path / "foreign/test")
    assert not _evaluation_metrics_valid(run, foreign)
    without_coverage = {k: v for k, v in metrics.items() if k != "heldout_distribution_coverage"}
    assert not _evaluation_metrics_valid(run, without_coverage)
    run["evaluation"]["heldout_distribution_coverage"] = False
    assert _evaluation_metrics_valid(run, without_coverage)
    for field, value in (("coverage_curve", []), ("components_live", 0),
                         ("liveness_split", "validation"), ("mean", float("nan"))):
        assert not coverage_report_valid({**result, field: value}, components=model.K)
    broken = copy.deepcopy(result)
    broken["coverage_curve"][-1]["fraction"] = .99
    assert not coverage_report_valid(broken, components=model.K)


@pytest.fixture
def toy_root(tmp_path):
    return save_toy_manifold_shards(tmp_path / "shards", ToyManifoldConfig(
        ambient_dim=3, n_samples=30, calibration_size=20,
        manifold_types=("segment",), manifolds_per_type=1,
    ), layer=0, shard_size=7)


@pytest.mark.parametrize("change,message", [
    ({"num_rows": 0}, "positive integer"),
    ({"num_rows": 100_001}, "100,001.*100,000.*scalable"),
    ({"num_rows": 7}, "row count"),
    ({"d_model": 4}, "ambient dimension"),
    ({"layers": [1]}, "dataset/layer"),
    ({"partition": {}}, "partition"),
])
def test_invalid_test_config_rejected(toy_root, change, message):
    path = toy_root / "test/config.json"
    config = json.loads(path.read_text())
    config.update(change)
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match=message):
        validate_toy_test_split(toy_root, layer=0)


@pytest.mark.parametrize("filename", ["config.json", "manifold_metadata.pt"])
def test_test_split_rejects_changed_parent(toy_root, filename):
    path = toy_root / filename
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        validate_toy_test_split(toy_root, layer=0)


def test_supplemental_test_split_and_disabled_coverage(tmp_path):
    config = _config(tmp_path)  # Supplemental population; development rows unchanged.
    root = Path(config["dataset"]["shard_dir"])
    source = validate_toy_test_split(root, layer=0)
    assert source["partition"]["kind"] == "supplemental"
    run = resolve_run(config)
    shutil.rmtree(root / "test")
    message = "add_toy_manifold_test_split.py"
    with pytest.raises(FileNotFoundError, match=message):
        resolve_run(config)
    with pytest.raises(FileNotFoundError, match=message):
        execute_run(run)
    assert not Path(run["run_dir"]).exists()
    with pytest.raises(FileNotFoundError, match=message):
        evaluate_toy_manifold_tiling(
            tmp_path / "missing_run", shard_dir=root, layer=0, model_kind="kmeans", device="cpu",
        )
    config["evaluation"]["heldout_distribution_coverage"] = False
    disabled = resolve_run(config)
    directory = execute_run(disabled)
    assert "heldout_distribution_coverage" not in json.loads((directory / "metrics.json").read_text())
    add_test_split(root, n_samples=8)
    result = evaluate_toy_manifold_tiling(
        directory, shard_dir=root, layer=0, model_kind="kmeans", device="cpu",
    )["heldout_distribution_coverage"]
    assert result["reference_points"] == 8
    assert result["source"]["partition"]["kind"] == "supplemental"


def test_declared_size_checked_before_loading_test_data(toy_root, monkeypatch):
    path = toy_root / "test/config.json"
    config = json.loads(path.read_text())
    config["num_rows"] = 100_001
    path.write_text(json.dumps(config))
    monkeypatch.setattr(torch, "load", lambda *a, **kw: pytest.fail("must reject before torch.load"))
    with pytest.raises(ValueError, match="scalable"):
        validate_toy_test_split(toy_root, layer=0)


@pytest.mark.parametrize("extra_rows", [-1, 1, 100_001])
def test_streamed_counts_are_checked(toy_root, monkeypatch, extra_rows):
    from dalg.evaluation import toy_manifold_coverage as coverage

    source = validate_toy_test_split(toy_root, layer=0)
    monkeypatch.setattr(coverage, "DataLoader", lambda *a, **kw: [
        torch.zeros(source["num_rows"] + extra_rows, 3),
    ])
    with pytest.raises(ValueError, match="row count|scalable"):
        evaluate_toy_test_coverage(source, torch.zeros(1, 3), torch.tensor([True]), batch_size=7)
