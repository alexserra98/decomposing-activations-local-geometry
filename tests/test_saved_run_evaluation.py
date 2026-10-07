from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from dalg.cli.run_pipeline import build_parser
from dalg.evaluation import saved_runs as saved
from dalg.pipeline import _RESOURCE_DEFAULTS, execute_run, resolve_run, write_manifest


@pytest.fixture(autouse=True)
def job_workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(saved, "REPO_ROOT", tmp_path / "workspace")


def _run(root, name="same-name", *, kind="mfa", K=2):
    root.mkdir(parents=True, exist_ok=True)
    row = {
        "run_dir": str(root), "run_id": name, "identity_hash": str(root),
        "identity": {"experiment": "saved-test", "source": str(root)},
        "training": {"model_kind": kind, "arguments": {"K": K}},
        "dataset": {"shard_dir": str(root / "shards"), "layer": 0},
        "evaluation": {"enabled": False, "kind": None, "device": "cpu"},
        "resources": dict(_RESOURCE_DEFAULTS),
    }
    (root / "run_spec.json").write_text(json.dumps(row))
    return row


def _manifest(path, runs):
    return write_manifest(runs, path)


@pytest.fixture
def planning(monkeypatch, tmp_path):
    a, b = _run(tmp_path / "a"), _run(tmp_path / "b")
    ma = _manifest(tmp_path / "a.jsonl", [a, b])
    mb = _manifest(tmp_path / "b.jsonl", [b, a])
    monkeypatch.setattr(saved, "_prerequisites", lambda run, evaluation: ([], copy.deepcopy(run)))
    return a, b, ma, mb


def test_multiple_manifests_deduplicate_and_select(planning):
    a, b, ma, mb = planning
    plan = saved.plan_evaluation([ma, mb])
    assert len(plan["runs"]) == 2
    assert all(len(row["sources"]) == 2 for row in plan["runs"])
    assert [row["run"]["run_dir"] for row in plan["runs"]] == [a["run_dir"], b["run_dir"]]
    selected = saved.plan_evaluation([ma, mb], indices=["0:1", "1:0"])
    assert len(selected["runs"]) == 1
    assert selected["runs"][0]["run"] == b
    single = saved.plan_evaluation([ma], indices=["1"])
    assert single["runs"][0]["run"] == b


@pytest.mark.parametrize("indices", [["0"], ["2:0"], ["0:9"], ["-1:0"]])
def test_bad_indices(planning, indices):
    _, _, ma, mb = planning
    with pytest.raises(ValueError):
        saved.plan_evaluation([ma, mb], indices=indices)


@pytest.mark.parametrize("field", ["identity_hash", "evaluation"])
def test_duplicate_conflicts(planning, tmp_path, field):
    a, _, ma, _ = planning
    changed = copy.deepcopy(a)
    changed[field] = "different" if field == "identity_hash" else {**a[field], "batch_size": 17}
    mc = _manifest(tmp_path / "c.jsonl", [changed])
    with pytest.raises(ValueError, match="conflicting references"):
        saved.plan_evaluation([ma, mc])


def test_overrides_and_new_invocation(planning):
    _, _, ma, _ = planning
    plan = saved.plan_evaluation([ma], overrides={
        "device": "cuda", "batch_size": 7, "rank_threshold": 2.0,
        "max_mean_to_manifold_distance": None,
    }, resources={"memory": "12G", "max_parallel": 2})
    row = plan["runs"][0]
    assert row["evaluation"]["enabled"]
    assert row["evaluation"]["kind"] == "toy_manifold_tiling"
    assert row["evaluation"]["batch_size"] == 7
    assert row["resources"]["gpus"] == 1
    assert row["resources"]["memory"] == "12G"
    path = saved.save_plan(plan)
    assert saved.save_plan(plan) == path
    other = copy.deepcopy(plan)
    other["runs"][0]["evaluation"]["batch_size"] = 8
    with pytest.raises(ValueError, match="conflicting evaluation plan"):
        saved.save_plan(other)
    assert saved.plan_evaluation([ma])["job_dir"] != plan["job_dir"]


def test_missing_runs_reported_and_no_eligible_fails(planning, monkeypatch):
    a, b, ma, _ = planning
    monkeypatch.setattr(saved, "_prerequisites", lambda run, evaluation: (["missing.pt"], None) if run == a else ([], run))
    plan = saved.plan_evaluation([ma])
    assert len(plan["runs"]) == len(plan["skipped"]) == 1
    assert plan["runs"][0]["run"] == b
    monkeypatch.setattr(saved, "_prerequisites", lambda run, evaluation: (["missing.pt"], None))
    args = build_parser().parse_args(["evaluate", "--manifest", str(ma)])
    with pytest.raises(ValueError, match="no eligible"):
        args.func(args)
    reports = list(saved.REPO_ROOT.glob("outputs/evaluations/*/skipped_runs.json"))
    assert len(reports) == 1
    assert len(json.loads(reports[0].read_text())["skipped"]) == 2


def test_slurm_groups_map_indices_and_collect_dependencies(planning, monkeypatch):
    _, _, ma, _ = planning
    plan = saved.plan_evaluation([ma])
    plan["runs"][1]["resources"]["memory"] = "12G"
    third = copy.deepcopy(plan["runs"][0])
    third["index"] = 2
    plan["runs"].append(third)
    saved.save_plan(plan)
    commands = []

    def submit(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(stdout=f"{100 + len(commands)};cluster\n")

    monkeypatch.setattr(saved.subprocess, "run", submit)
    saved.submit_evaluations(plan)
    assert len(commands) == 3
    assert "--array=0,2%4" in commands[0]
    assert "--array=1%4" in commands[1]
    assert "--dependency=afterok:101:102" in commands[2]
    assert not any(arg.startswith(("--gres=", "--array=")) for arg in commands[2])
    assert commands[0][-1] == "run" and commands[2][-1] == "collect"
    assert all("sbatch_evaluation_pipeline.sh" in " ".join(cmd) for cmd in commands)


def test_dry_run_is_read_only(planning, tmp_path, monkeypatch, capsys):
    _, _, ma, mb = planning
    monkeypatch.setattr(saved.subprocess, "run", lambda *a, **k: pytest.fail("must not submit"))
    before = _snapshot(tmp_path)
    args = build_parser().parse_args([
        "evaluate", "--manifest", str(ma), str(mb), "--submit", "--dry-run",
        "--max-mean-to-manifold-distance", "none",
    ])
    args.func(args)
    assert _snapshot(tmp_path) == before
    assert "--dependency=afterok:" in capsys.readouterr().out


@pytest.fixture(params=["mfa", "hddc", "kmeans"])
def real_run(request, tmp_path):
    if request.param == "kmeans":
        from tests.test_kmeans_pipeline import _config
        run = resolve_run(_config(tmp_path))
        execute_run(run)
        return run
    from dalg.data.manifold_dataset import ToyManifoldConfig, save_toy_manifold_shards
    from dalg.models.mfa import MFA, save_mfa
    from dalg.models.adaptive_q.mfa_hddc import MFA_HDDC, save_mfa_hddc
    shards = save_toy_manifold_shards(tmp_path / "shards", ToyManifoldConfig(
        ambient_dim=3, n_samples=40, calibration_size=20,
        manifold_types=("segment",), manifolds_per_type=2, seed=3,
    ), layer=0, shard_size=20, test_fraction=0)
    from tests.test_toy_manifold_test_split import add_test_split
    add_test_split(shards, n_samples=8)
    run = _run(tmp_path / "run", kind=request.param)
    run["dataset"]["shard_dir"] = str(shards)
    directory = Path(run["run_dir"])
    (directory / "run_spec.json").write_text(json.dumps(run))
    (directory / "config.json").write_text(json.dumps({"model_kind": request.param}))
    (directory / "val_indices.json").write_text(json.dumps({
        "train_rows": 32, "val_rows": 8, "val_global_rows": list(range(0, 40, 5)),
    }))
    points = torch.cat([torch.load(path, weights_only=True).reshape(-1, 3)
                        for path in sorted((shards / "layer00").glob("*.pt"))])
    if request.param == "mfa":
        model = MFA(points[[0, 20]], rank=1)
        save_mfa(model, str(directory / "mfa_model.pt"))
    else:
        model = MFA_HDDC(points[[0, 20]], rank=1, isotropic_psi=True)
        save_mfa_hddc(model, str(directory / "mfa_model.pt"))
    labels = torch.cdist(points, model.mu.detach()).argmin(dim=1)
    torch.save({"K": 2, "assignments": labels, "cluster_sizes": torch.bincount(labels, minlength=2),
                "subset_spec": None}, directory / "mfa_model_assignments.pt")
    (directory / "metrics.json").write_text('{"old_report": true}')
    return run



def _snapshot(directory):
    return {str(path.relative_to(directory)): path.read_bytes()
            for path in directory.rglob("*") if path.is_file()}


def _source_artifacts(directory):
    return {key: value for key, value in _snapshot(directory).items()
            if key not in {"metrics.json", "EVALUATION_COMPLETED.json"}}


def _plans():
    return [saved.read_evaluation_plan(path)
            for path in saved.REPO_ROOT.glob("outputs/evaluations/*/evaluation_plan.json")]


def test_real_saved_evaluation_overwrites_every_invocation(real_run, tmp_path, monkeypatch):
    from dalg.evaluation.toy_manifold_tiling import evaluate_pipeline_run
    manifest = _manifest(tmp_path / "manifest.jsonl", [real_run])
    source = Path(real_run["run_dir"])
    before = _source_artifacts(source)
    # Obsolete or broken reports and markers can be repaired by reevaluation.
    (source / "metrics.json").write_text('broken report')
    (source / "EVALUATION_COMPLETED.json").write_text('broken marker')
    monkeypatch.setattr("dalg.pipeline._run_command", lambda *a: pytest.fail("must not train or assign"))
    monkeypatch.setattr("dalg.pipeline.execute_run", lambda *a: pytest.fail("must not execute pipeline"))
    calls = []

    def evaluate(*args, **kwargs):
        calls.append(kwargs["evaluation"])
        return evaluate_pipeline_run(*args, **kwargs)

    monkeypatch.setattr("dalg.evaluation.toy_manifold_tiling.evaluate_pipeline_run", evaluate)
    args = build_parser().parse_args([
        "evaluate", "--manifest", str(manifest), "--batch-size", "13",
        "--max-mean-to-manifold-distance", "none",
    ])
    args.func(args)
    first = json.loads((source / "metrics.json").read_text())
    assert first["evaluation_config"]["batch_size"] == 13
    assert first["identity_hash"] == real_run["identity_hash"]
    assert first["evaluation_provenance"]["source_run_dir"] == str(source)
    assert _source_artifacts(source) == before
    table = pd.read_csv(source.parent / "general_metrics.csv")
    assert len(table) == 1 and table["evaluation_config.batch_size"].iloc[0] == 13
    assert table["config.training.model_kind"].iloc[0] == real_run["training"]["model_kind"]
    assert table["heldout_distribution_coverage.reference_points"].iloc[0] > 0
    curve = json.loads(table["heldout_distribution_coverage.coverage_curve"].iloc[0])
    assert curve[-1]["fraction"] == 1.0
    args.func(args)
    second = json.loads((source / "metrics.json").read_text())
    assert len(calls) == 2
    assert second["evaluation_provenance"]["plan_hash"] != first["evaluation_provenance"]["plan_hash"]
    assert _source_artifacts(source) == before
    assert len(_plans()) == 2
    assert not list(saved.REPO_ROOT.rglob("metrics.json"))
    marker = json.loads((source / "EVALUATION_COMPLETED.json").read_text())
    assert marker["completed"] and marker["artifact"] == str(source / "metrics.json")


@pytest.mark.parametrize("failure", ["exception", "invalid_metrics"])
def test_failed_evaluation_preserves_previous_report(real_run, tmp_path, monkeypatch, failure):
    manifest = _manifest(tmp_path / "manifest.jsonl", [real_run])
    source = Path(real_run["run_dir"])
    (source / "EVALUATION_COMPLETED.json").write_text('{"old_marker": true}')
    before = _snapshot(source)
    plan = saved.plan_evaluation([manifest])
    saved.save_plan(plan)

    def fail(*args, **kwargs):
        if failure == "exception":
            raise RuntimeError("evaluation failed")
        return {}

    monkeypatch.setattr("dalg.evaluation.toy_manifold_tiling.evaluate_pipeline_run", fail)
    with pytest.raises((RuntimeError, ValueError)):
        saved.evaluate_saved_run(plan, 0)
    assert _snapshot(source) == before
    with pytest.raises(ValueError):
        saved.collect_evaluations(plan)
    assert not (source.parent / "general_metrics.csv").exists()


def test_prerequisite_failures(real_run, tmp_path):
    manifest = _manifest(tmp_path / "manifest.jsonl", [real_run])
    source = Path(real_run["run_dir"])
    assignment = saved._assignments_path(real_run)
    original = assignment.read_bytes()
    assignment.unlink()
    plan = saved.plan_evaluation([manifest])
    assert not plan["runs"] and str(assignment) in plan["skipped"][0]["reasons"]
    assignment.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="invalid saved assignments"):
        saved.plan_evaluation([manifest])
    assignment.write_bytes(original)
    spec = copy.deepcopy(real_run)
    spec["identity_hash"] = "wrong"
    (source / "run_spec.json").write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="identity mismatch"):
        saved.plan_evaluation([manifest])


def test_combined_csvs_preserve_unselected_rows(real_run, tmp_path, monkeypatch):
    original = Path(real_run["run_dir"])
    second = original.parent / "second-run"
    shutil.copytree(original, second)
    other = copy.deepcopy(real_run)
    other["run_dir"] = str(second)
    (second / "run_spec.json").write_text(json.dumps(other))
    if other["training"]["model_kind"] == "kmeans":
        assignment = saved._assignments_path(other)
        bundle = torch.load(assignment, weights_only=True)
        bundle["model_path"] = str(second / "kmeans_model.pt")
        torch.save(bundle, assignment)
    ma = _manifest(tmp_path / "ma.jsonl", [real_run, other])
    mb = _manifest(tmp_path / "mb.jsonl", [real_run])
    before = _source_artifacts(original), _source_artifacts(second)
    monkeypatch.setattr("dalg.pipeline._run_command", lambda *a: pytest.fail("must not train or assign"))
    args = build_parser().parse_args(["evaluate", "--manifest", str(ma), str(mb)])
    args.func(args)
    plan = _plans()[0]
    assert len(plan["runs"]) == 2
    assert len(plan["runs"][0]["sources"]) == 2
    general = pd.read_csv(original.parent / "general_metrics.csv")
    manifolds = pd.read_csv(original.parent / "manifold_metrics.csv")
    assert len(general) == 2
    assert general["run_id"].nunique() == 1
    assert general["evaluation_id"].nunique() == 2
    assert set(manifolds["evaluation_id"]) == set(general["evaluation_id"])
    assert (_source_artifacts(original), _source_artifacts(second)) == before
    unselected = _snapshot(second)
    selected_args = build_parser().parse_args(["evaluate", "--manifest", str(ma), "--indices", "0"])
    selected_args.func(selected_args)
    assert _snapshot(second) == unselected
    assert len(pd.read_csv(original.parent / "general_metrics.csv")) == 2
    with pytest.raises(ValueError, match="provenance mismatch"):
        saved.collect_evaluations(plan)


def test_summary_roots_refresh_existing_ancestors(planning, tmp_path):
    _, _, ma, _ = planning
    (tmp_path / "general_metrics.csv").write_text('old summary')
    assert saved.plan_evaluation([ma])["summary_dirs"] == [str(tmp_path)]
    nested = _run(tmp_path / "experiment" / "runs" / "one")
    (tmp_path / "experiment" / "manifold_metrics.csv").write_text('old summary')
    manifest = _manifest(tmp_path / "nested.jsonl", [nested])
    assert saved.plan_evaluation([manifest])["summary_dirs"] == [str(tmp_path), str(tmp_path / "experiment")]


def test_summary_roots_for_multiple_experiments(planning, tmp_path):
    a = _run(tmp_path / "experiment_a" / "run")
    b = _run(tmp_path / "experiment_b" / "run")
    ma = _manifest(tmp_path / "ma.jsonl", [a])
    mb = _manifest(tmp_path / "mb.jsonl", [b])
    assert saved.plan_evaluation([ma, mb])["summary_dirs"] == [
        str(tmp_path / "experiment_a"), str(tmp_path / "experiment_b"),
    ]


def test_invalid_model_and_dataset_are_errors(real_run, tmp_path):
    manifest = _manifest(tmp_path / "manifest.jsonl", [real_run])
    source = Path(real_run["run_dir"])
    checkpoint = source / f"{saved._model_stem(real_run)}.pt"
    contents = checkpoint.read_bytes()
    checkpoint.write_bytes(b'')
    with pytest.raises(ValueError, match="invalid saved model"):
        saved.plan_evaluation([manifest])
    checkpoint.write_bytes(contents)
    config = Path(real_run["dataset"]["shard_dir"]) / "config.json"
    dataset = json.loads(config.read_text())
    dataset["source_kind"] = "not-toy"
    config.write_text(json.dumps(dataset))
    with pytest.raises(ValueError, match="incompatible toy-manifold dataset"):
        saved.plan_evaluation([manifest])


def test_shard_only_model_is_reported(tmp_path):
    run = _run(tmp_path / "run")
    (Path(run["run_dir"]) / "mfa_model_shards.json").write_text('{}')
    manifest = _manifest(tmp_path / "manifest.jsonl", [run])
    plan = saved.plan_evaluation([manifest])
    assert not plan["runs"]
    assert any("unsupported shard-only" in reason for reason in plan["skipped"][0]["reasons"])


@pytest.mark.parametrize("overrides", [
    {"batch_size": 0}, {"rank_threshold": -1}, {"max_mean_to_manifold_distance": 0},
    {"max_mean_to_manifold_distance": float("inf")}, {"kind": "unknown"},
    {"heldout_distribution_coverage": "false"},
])
def test_invalid_evaluation_settings(planning, overrides):
    _, _, ma, _ = planning
    with pytest.raises(ValueError):
        saved.plan_evaluation([ma], overrides=overrides)


def test_resource_yaml(planning, tmp_path):
    _, _, ma, _ = planning
    resource_file = tmp_path / "resources.yaml"
    resource_file.write_text('memory: 4G\npartition: cpu\nmax_parallel: 2\n')
    args = build_parser().parse_args([
        "evaluate", "--manifest", str(ma), "--resources", str(resource_file), "--dry-run", "--submit",
    ])
    args.func(args)
    resource_file.write_text('unknown_resource: 1\n')
    with pytest.raises(ValueError, match="unknown resource"):
        args.func(args)


def test_modified_or_old_plan_rejected(planning):
    _, _, ma, _ = planning
    plan = saved.plan_evaluation([ma])
    path = saved.save_plan(plan)
    assert saved.read_evaluation_plan(path) == plan
    plan["runs"][0]["evaluation"]["batch_size"] = 1
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="modified evaluation plan"):
        saved.read_evaluation_plan(path)
    plan["version"] = 1
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="modified evaluation plan"):
        saved.read_evaluation_plan(path)


@pytest.mark.parametrize("flag,expected", [
    (None, True), ("--heldout-distribution-coverage", True),
    ("--no-heldout-distribution-coverage", False),
])
def test_coverage_overrides_do_not_rewrite_old_manifests(planning, flag, expected):
    _, _, manifest, _ = planning
    original = manifest.read_bytes()
    args = build_parser().parse_args([
        "evaluate", "--manifest", str(manifest), "--dry-run", *([flag] if flag else []),
    ])
    overrides = ({"heldout_distribution_coverage": args.heldout_distribution_coverage}
                 if hasattr(args, "heldout_distribution_coverage") else {})
    plan = saved.plan_evaluation([manifest], overrides=overrides)
    assert all(row["evaluation"]["heldout_distribution_coverage"] is expected for row in plan["runs"])
    assert all("heldout_distribution_coverage" not in row["run"]["evaluation"] for row in plan["runs"])
    args.func(args)
    assert manifest.read_bytes() == original


def test_missing_test_split_is_error_and_override_preserves_saved_artifacts(real_run, tmp_path):
    manifest = _manifest(tmp_path / "manifest.jsonl", [real_run])
    original_manifest = manifest.read_bytes()
    source = Path(real_run["run_dir"])
    enabled_plan = saved.plan_evaluation([manifest])
    shutil.rmtree(Path(real_run["dataset"]["shard_dir"]) / "test")
    before = _snapshot(source)
    for action in (
        lambda: saved.plan_evaluation([manifest]),
        lambda: saved.evaluate_saved_run(enabled_plan, 0),
    ):
        with pytest.raises(FileNotFoundError, match="add_toy_manifold_test_split.py"):
            action()
        assert _snapshot(source) == before
    args = build_parser().parse_args([
        "evaluate", "--manifest", str(manifest), "--no-heldout-distribution-coverage",
    ])
    args.func(args)
    metrics = json.loads((source / "metrics.json").read_text())
    assert "heldout_distribution_coverage" not in metrics
    assert metrics["evaluation_config"]["heldout_distribution_coverage"] is False
    assert manifest.read_bytes() == original_manifest
