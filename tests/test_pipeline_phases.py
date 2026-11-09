"""Phase boundaries preserve planning, artifact reuse, and failure behavior."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import torch
import yaml

from dalg.pipeline import (
    REPO_ROOT, PipelineConfigError, assign_run, complete_run, evaluate_run,
    execute_run, initialize_run, pipeline_status, save_experiment_manifest_to_disk, prepare_run,
    read_manifest, map_run_to_manifest, train_run, write_manifest,
)
from dalg.pipeline import common
from dalg.pipeline.evaluation import _evaluation_artifact_valid
from tests.synthetic_shards import LAYER, build_multi_shard


def _config(tmp_path: Path, kind: str = "kmeans") -> dict:
    shards = build_multi_shard(tmp_path / "shards", n_shards=2, rows_per_shard=6)
    config = {
        "experiment": {"name": "phase-test", "output_root": str(tmp_path / "runs")},
        "dataset": {"shard_dir": str(shards), "layer": LAYER},
        "model": {"kind": kind, "K": 2},
        "training": {"device": "cpu", "seed": 7, "val_frac": 0.25},
        "assignments": {"enabled": True, "device": "cpu", "batch_size": 8},
        "evaluation": {
            "enabled": True, "kind": "unsupervised", "device": "cpu",
            "batch_size": 8, "heldout_distribution_coverage": False,
        },
        "resources": {"gpus": 0, "gpu_type": ""},
    }
    if kind == "kmeans":
        config["training"].update(max_iter=5, restarts=1)
    else:
        config["model"].update(q_max=1, shared_b=True)
        config["training"].update(
            fit_method="em", epochs=1, batch_size=8, pool_size=64,
            refine_epochs=0, epoch_snapshot_every=0,
        )
    return config


def test_plan_writes_only_an_immutable_manifest(tmp_path):
    config = _config(tmp_path)
    config["sweep"] = {"model.K": [2, 3], "training.seed": [7, 8]}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    manifest, runs = save_experiment_manifest_to_disk(path, tmp_path / "plan.jsonl")

    assert REPO_ROOT == Path(__file__).resolve().parents[1]
    assert read_manifest(manifest) == runs
    assert [(r["training"]["arguments"]["K"], r["training"]["arguments"]["seed"])
            for r in runs] == [(2, 7), (2, 8), (3, 7), (3, 8)]
    assert not (tmp_path / "runs").exists()
    original = manifest.read_bytes()
    assert save_experiment_manifest_to_disk(path, manifest) == (manifest, runs)
    with pytest.raises(PipelineConfigError, match="refusing to overwrite"):
        write_manifest(list(reversed(runs)), manifest)
    assert manifest.read_bytes() == original


@pytest.mark.parametrize("kind", ["kmeans", "hddc"])
@pytest.mark.parametrize("coverage", [False, True])
def test_real_unsupervised_phases_return_artifacts_and_resume(tmp_path, monkeypatch, kind, coverage):
    config = _config(tmp_path, kind)
    if coverage:
        test_root = build_multi_shard(tmp_path / "independent", n_shards=1, rows_per_shard=4)
        source = json.loads((test_root / "config.json").read_text())
        source["layers"] = [LAYER]
        (test_root / "config.json").write_text(json.dumps(source))
        generator = torch.Generator().manual_seed(19)
        torch.save(torch.randn(4, 3, 2, generator=generator) * 100 + 500,
                   test_root / f"layer{LAYER:02d}" / "shard_00000.pt")
        config["evaluation"].update(heldout_distribution_coverage=True, test_shard_dir=str(test_root))
    run = map_run_to_manifest(config)
    directory = prepare_run(run)
    assert initialize_run(run) is None
    stem = "kmeans_model" if kind == "kmeans" else "mfa_model"
    assert train_run(run) == directory / f"{stem}.pt"
    assert assign_run(run) == directory / f"{stem}_assignments.pt"
    assert evaluate_run(run) == directory / "metrics.json"
    assert not pipeline_status([run])[0]["pipeline"]
    assert complete_run(run) == directory

    report_path = directory / "metrics.json"
    report_bytes = report_path.read_bytes()
    report = json.loads(report_bytes)
    assert report["evaluation"] == "unsupervised"
    assert report["tile_alignment"]["version"] == 1
    assert report["tile_alignment"]["neighbors"] == 5
    assert ("heldout_distribution_coverage" in report) == coverage
    assert _evaluation_artifact_valid(run)
    assert pipeline_status([run])[0]["pipeline"]

    monkeypatch.setattr(common, "_run_command", lambda _: pytest.fail("valid artifacts must be reused"))
    monkeypatch.setattr("dalg.evaluation.evaluate_pipeline_run",
                        lambda _: pytest.fail("valid metrics must be reused"))
    assert execute_run(run) == directory
    assert report_path.read_bytes() == report_bytes

    report_path.write_text("invalid report")
    with pytest.raises(RuntimeError, match="refusing to overwrite invalid evaluation"):
        evaluate_run(run)
    assert report_path.read_text() == "invalid report"

    assignment_path = directory / f"{stem}_assignments.pt"
    assignment_path.write_bytes(b"")
    with pytest.raises(RuntimeError, match="refusing to overwrite invalid assignment"):
        assign_run(run)
    assert assignment_path.read_bytes() == b""


@pytest.mark.parametrize("assignments_enabled", [False, True])
def test_training_failure_does_not_run_later_phases_or_mark_completion(tmp_path, monkeypatch, assignments_enabled):
    config = _config(tmp_path)
    config["evaluation"]["enabled"] = False
    config["assignments"]["enabled"] = assignments_enabled
    run = map_run_to_manifest(config)
    commands = []

    def fail(command):
        commands.append(command)
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(common, "_run_command", fail)
    with pytest.raises(subprocess.CalledProcessError):
        execute_run(run)
    assert len(commands) == 1
    directory = Path(run["run_dir"])
    assert (directory / "run_spec.json").is_file()
    assert not list(directory.glob("*_COMPLETED.json"))
    assert not list(directory.glob("*assignments.pt"))
    assert not (directory / "metrics.json").exists()


def test_disabled_phases_leave_no_artifacts_and_count_as_satisfied(tmp_path):
    config = _config(tmp_path)
    config["assignments"]["enabled"] = config["evaluation"]["enabled"] = False
    run = map_run_to_manifest(config)
    directory = execute_run(run)
    assert initialize_run(run) is assign_run(run) is evaluate_run(run) is None
    assert not list(directory.glob("*assignments.pt"))
    assert not (directory / "metrics.json").exists()
    status = pipeline_status([run])[0]
    assert all(status[k] for k in ("initialization", "training", "assignments", "evaluation", "pipeline"))
