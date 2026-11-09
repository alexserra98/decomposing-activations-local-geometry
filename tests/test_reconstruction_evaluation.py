from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd
import pytest
import torch

from dalg.analysis.aggregate_metrics import aggregate_metrics
from dalg.evaluation import evaluate_pipeline_run, saved_runs
from dalg.evaluation.common import load_evaluation_data, load_model
from dalg.pipeline import evaluate_run, execute_run, map_run_to_manifest, write_manifest
from dalg.pipeline.evaluation import _evaluation_metrics_valid
from tests.synthetic_shards import LAYER
from tests.test_pipeline_phases import _config


@pytest.mark.parametrize("kind", ["kmeans", "hddc"])
def test_saved_splits_subsets_prefix_aggregation_and_reevaluation(tmp_path, monkeypatch, kind):
    config = _config(tmp_path, kind)
    root = Path(config["dataset"]["shard_dir"])
    shard_config = json.loads((root / "config.json").read_text())
    shard_config["drop_prefix"] = 1
    (root / "config.json").write_text(json.dumps(shard_config))
    config["dataset"]["shard_dir"] += "#pile_A_10"
    if kind == "kmeans":
        config["training"]["drop_prefix"] = 1
    generator = torch.Generator().manual_seed(51)
    all_points = []
    for index in range(2):
        points = torch.randn(6, 3, 2, generator=generator)
        points[:, 0] = 10000  # Dropped prefixes must not affect the score.
        all_points.append(points)
        torch.save(points, root / f"layer{LAYER:02d}/shard_{index:05d}.pt")
        meta_path = root / f"meta/shard_{index:05d}.json"
        meta = json.loads(meta_path.read_text())
        for row in meta["rows"]:
            row["subset"] = "pile-" + row["subset"]
        meta_path.write_text(json.dumps(meta))
    all_points = torch.cat(all_points)
    run = map_run_to_manifest(config)
    directory = execute_run(run)
    report_path = directory / "metrics.json"
    report = json.loads(report_path.read_text())
    assert _evaluation_metrics_valid(run, report)
    model = load_model(directory, kind)
    data = load_evaluation_data(directory, shard_dir=run["dataset"]["shard_dir"],
                                layer=LAYER, model=model, model_kind=kind)
    assert len(data.positions) == 5
    for split, positions in (("train", data.train_positions), ("validation", data.val_positions)):
        x = all_points[positions, 1:].reshape(-1, 2)
        total = len(x)
        with torch.no_grad():
            if kind == "hddc":
                reconstructed = model.reconstruct(x)
            else:
                labels = model.predict(x)
                eligible = model.pca_valid[labels]
                x, labels = x[eligible], labels[eligible]
                bases = model.W[labels]
                centered = x - model.mu[labels]
                scores = torch.einsum("bd,bdq->bq", centered, bases)
                reconstructed = model.mu[labels] + torch.einsum("bdq,bq->bd", bases, scores)
        errors = (x.double() - reconstructed.double()).square().mean(1)
        result = report["reconstruction"][split]
        assert result["n"] == len(errors)
        assert result["excluded_n"] == total - len(errors)
        assert result["mse"] == pytest.approx(float(errors.mean()), rel=1e-5, abs=1e-8)
        assert result["standard_error"] == pytest.approx(float(errors.std() / len(errors) ** 0.5), rel=1e-5)
    second = evaluate_pipeline_run(run, evaluation={**run["evaluation"], "batch_size": 3})
    for split in ("train", "validation"):
        assert second["reconstruction"][split] == pytest.approx(report["reconstruction"][split], rel=1e-5, abs=1e-8)

    for mutation in ("missing", "negative", "nan", "count", "dimension", "method", "null", "rows"):
        broken = copy.deepcopy(report)
        block = broken["reconstruction"]
        if mutation == "missing":
            del broken["reconstruction"]
        elif mutation == "negative":
            block["train"]["standard_error"] = -1
        elif mutation == "nan":
            block["validation"]["mse"] = float("nan")
        elif mutation == "count":
            block["train"]["excluded_n"] += 1
        elif mutation == "dimension":
            block["dimension"] += 1
        elif mutation == "method":
            block["method"] = "hard_assignment"
        elif mutation == "rows":
            broken["dataset"]["train_rows"] += 1
        else:
            block["train"]["mse"] = None
        assert not _evaluation_metrics_valid(run, broken), mutation

    general, _ = aggregate_metrics(directory.parent)
    for split in ("train", "validation"):
        for key, value in report["reconstruction"][split].items():
            assert general.iloc[0][f"reconstruction.{split}.{key}"] == pytest.approx(value)

    # Obsolete reports remain readable, but ordinary resume must preserve them.
    stale = copy.deepcopy(report)
    del stale["reconstruction"]
    report_path.write_text(json.dumps(stale))
    assert len(aggregate_metrics(directory.parent)[0]) == 1
    with pytest.raises(RuntimeError, match="refusing to overwrite invalid evaluation"):
        evaluate_run(run)
    assert json.loads(report_path.read_text()) == stale

    monkeypatch.setattr(saved_runs, "REPO_ROOT", tmp_path / "workspace")
    manifest = write_manifest([run], tmp_path / "manifest.jsonl")
    plan = saved_runs.plan_evaluation([manifest])
    saved_runs.save_plan(plan)
    saved_runs.evaluate_saved_run(plan, 0)
    saved_runs.collect_evaluations(plan)
    updated = json.loads(report_path.read_text())
    assert _evaluation_metrics_valid(run, updated)
    csv = pd.read_csv(directory.parent / "general_metrics.csv")
    assert csv.iloc[0]["reconstruction.validation.mse"] == pytest.approx(report["reconstruction"]["validation"]["mse"])
