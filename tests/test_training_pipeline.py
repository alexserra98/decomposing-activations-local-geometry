from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest
import torch
import yaml

from dalg.data.manifold_dataset import ToyManifoldConfig, save_toy_manifold_shards
from dalg.pipeline import (
    PipelineConfigError,
    _training_command,
    execute_run,
    pipeline_status,
    resolve_experiment,
)
from tests.synthetic_shards import LAYER, build_multi_shard


def _config(tmp_path: Path, shard_dir: Path) -> dict:
    return {
        "experiment": {
            "name": "pipeline-test",
            "output_root": str(tmp_path / "models"),
        },
        "dataset": {
            "id": "tiny-shards",
            "shard_dir": str(shard_dir),
            "layer": LAYER,
        },
        "model": {"kind": "mfa", "K": 2, "rank": 1},
        "training": {
            "device": "cpu",
            "epochs": 1,
            "batch_size": 4,
            "num_workers": 0,
            "pool_size": 8,
            "refine_epochs": 0,
            "val_frac": 0.25,
            "seed": 3,
            "epoch_snapshot_every": 0,
            "early_stop_delta": 0.0,
        },
        "assignments": {"enabled": True, "device": "cpu", "batch_size": 4},
        "evaluation": {"enabled": False},
        "resources": {"gpus": 0, "gpu_type": "", "max_parallel": 2},
    }


def _write_yaml(path: Path, payload: dict) -> Path:
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return path


def test_sweep_expands_to_stable_distinct_runs(tmp_path: Path) -> None:
    shard_dir = build_multi_shard(tmp_path / "shards", n_shards=2, rows_per_shard=4)
    config = _config(tmp_path, shard_dir)
    config["sweep"] = {"model.K": [2, 3], "training.seed": [7, 8]}
    path = _write_yaml(tmp_path / "experiment.yaml", config)

    first = resolve_experiment(path)
    second = resolve_experiment(path)

    assert len(first) == 4
    assert [run["run_id"] for run in first] == [run["run_id"] for run in second]
    assert len({run["run_id"] for run in first}) == 4
    assert {run["training"]["arguments"]["K"] for run in first} == {2, 3}
    assert {run["training"]["arguments"]["seed"] for run in first} == {7, 8}
    assert all(Path(run["run_dir"]).is_absolute() for run in first)


def test_unknown_training_parameter_is_rejected(tmp_path: Path) -> None:
    shard_dir = build_multi_shard(tmp_path / "shards", n_shards=1, rows_per_shard=4)
    config = _config(tmp_path, shard_dir)
    config["training"]["batch_szie"] = 8
    path = _write_yaml(tmp_path / "experiment.yaml", config)

    with pytest.raises(PipelineConfigError, match="batch_szie"):
        resolve_experiment(path)


def test_shared_centroids_are_validated_resolved_and_forwarded(tmp_path: Path) -> None:
    shard_dir = build_multi_shard(tmp_path / "shards", n_shards=1, rows_per_shard=4)
    centroids_dir = tmp_path / "shared_centroids"
    centroids_dir.mkdir()
    centroids = torch.tensor([[-2.0, -1.0], [2.0, 1.0]])
    centroids_path = centroids_dir / "centroids.pt"
    torch.save(centroids, centroids_path)
    config = _config(tmp_path, shard_dir)
    config["training"]["centroids_path"] = str(centroids_path)
    path = _write_yaml(tmp_path / "experiment.yaml", config)

    run = resolve_experiment(path)[0]
    resolved = str(centroids_path.resolve())

    assert run["training"]["arguments"]["centroids_path"] == resolved
    assert run["identity"]["training_args"]["centroids_path"] == resolved
    command = _training_command(run)
    assert command[command.index("--centroids-path") + 1] == resolved


@pytest.mark.parametrize(
    ("centroids", "message"),
    [
        (torch.zeros(3, 2), "centroids K=3 does not match model.K=2"),
        (torch.zeros(2, 3), "centroid dimension D=3 does not match activation d_model=2"),
    ],
)
def test_incompatible_shared_centroids_are_rejected(
    tmp_path: Path,
    centroids: torch.Tensor,
    message: str,
) -> None:
    shard_dir = build_multi_shard(tmp_path / "shards", n_shards=1, rows_per_shard=4)
    centroids_path = tmp_path / "centroids.pt"
    torch.save(centroids, centroids_path)
    config = _config(tmp_path, shard_dir)
    config["training"]["centroids_path"] = str(centroids_path)
    path = _write_yaml(tmp_path / "experiment.yaml", config)

    with pytest.raises(PipelineConfigError, match=message):
        resolve_experiment(path)


def test_missing_shared_centroids_are_rejected(tmp_path: Path) -> None:
    shard_dir = build_multi_shard(tmp_path / "shards", n_shards=1, rows_per_shard=4)
    config = _config(tmp_path, shard_dir)
    config["training"]["centroids_path"] = str(tmp_path / "missing.pt")
    path = _write_yaml(tmp_path / "experiment.yaml", config)

    with pytest.raises(PipelineConfigError, match="centroids file not found"):
        resolve_experiment(path)


@pytest.mark.parametrize("invalid_name", ["centroids.pth", "centroids", "centroids.PT"])
def test_shared_centroids_require_pt_extension(
    tmp_path: Path,
    invalid_name: str,
) -> None:
    shard_dir = build_multi_shard(tmp_path / "shards", n_shards=1, rows_per_shard=4)
    invalid_path = tmp_path / invalid_name
    torch.save(torch.zeros(2, 2), invalid_path)
    config = _config(tmp_path, shard_dir)
    config["training"]["centroids_path"] = str(invalid_path)
    path = _write_yaml(tmp_path / "experiment.yaml", config)

    with pytest.raises(PipelineConfigError, match="must point directly to a \\.pt file"):
        resolve_experiment(path)


def test_shared_centroids_reject_directory(tmp_path: Path) -> None:
    shard_dir = build_multi_shard(tmp_path / "shards", n_shards=1, rows_per_shard=4)
    centroids_dir = tmp_path / "centroids"
    centroids_dir.mkdir()
    torch.save(torch.zeros(2, 2), centroids_dir / "centroids.pt")
    config = _config(tmp_path, shard_dir)
    config["training"]["centroids_path"] = str(centroids_dir)
    path = _write_yaml(tmp_path / "experiment.yaml", config)

    with pytest.raises(PipelineConfigError, match="must point directly to a \\.pt file"):
        resolve_experiment(path)


def test_component_sharded_command_uses_torchrun(tmp_path: Path) -> None:
    shard_dir = build_multi_shard(tmp_path / "shards", n_shards=1, rows_per_shard=4)
    config = _config(tmp_path, shard_dir)
    config["training"].update({"device": "cuda", "training_mode": "component_shard"})
    config["resources"].update({"gpus": 2, "gpu_type": "H100"})
    path = _write_yaml(tmp_path / "experiment.yaml", config)

    run = resolve_experiment(path)[0]
    command = _training_command(run)

    assert command[:3] == [sys.executable, "-m", "torch.distributed.run"]
    assert "--nproc_per_node=2" in command
    assert command[-2:] != ["--training-mode", "vanilla"]
    assert "component_shard" in command


def test_execute_run_skips_valid_completed_stages(tmp_path: Path, monkeypatch) -> None:
    shard_dir = build_multi_shard(tmp_path / "shards", n_shards=2, rows_per_shard=4)
    config = _config(tmp_path, shard_dir)
    path = _write_yaml(tmp_path / "experiment.yaml", config)
    run = resolve_experiment(path)[0]
    commands: list[list[str]] = []

    def fake_run(command: list[str]) -> None:
        commands.append(command)
        run_dir = Path(run["run_dir"])
        if "dalg.cli.run_metrics" in command:
            assignments = torch.tensor([0, 1, 0, 1], dtype=torch.long)
            torch.save(
                {
                    "K": 2,
                    "assignments": assignments,
                    "cluster_sizes": torch.bincount(assignments, minlength=2),
                },
                run_dir / "mfa_model_assignments.pt",
            )
        else:
            (run_dir / "config.json").write_text("{}")
            (run_dir / "val_indices.json").write_text("{}")
            (run_dir / "mfa_model.pt").write_bytes(b"model")

    monkeypatch.setattr("dalg.pipeline._run_command", fake_run)
    execute_run(run)
    execute_run(run)

    assert len(commands) == 2
    run_dir = Path(run["run_dir"])
    assert (run_dir / "run_spec.json").is_file()
    assert (run_dir / "TRAINING_COMPLETED.json").is_file()
    assert (run_dir / "ASSIGNMENTS_COMPLETED.json").is_file()
    assert (run_dir / "PIPELINE_COMPLETED.json").is_file()
    assert pipeline_status([run])[0]["pipeline"] is True


def test_existing_run_spec_mismatch_is_rejected(tmp_path: Path, monkeypatch) -> None:
    shard_dir = build_multi_shard(tmp_path / "shards", n_shards=1, rows_per_shard=4)
    config = _config(tmp_path, shard_dir)
    run = resolve_experiment(_write_yaml(tmp_path / "experiment.yaml", config))[0]
    run_dir = Path(run["run_dir"])
    run_dir.mkdir(parents=True)
    wrong = copy.deepcopy(run)
    wrong["identity_hash"] = "wrong"
    (run_dir / "run_spec.json").write_text(json.dumps(wrong))

    monkeypatch.setattr("dalg.pipeline._run_command", lambda _command: None)
    with pytest.raises(PipelineConfigError, match="different configuration"):
        execute_run(run)


def test_real_cpu_training_and_assignment_pipeline_smoke(tmp_path: Path) -> None:
    shard_dir = build_multi_shard(tmp_path / "shards", n_shards=2, rows_per_shard=4)
    config = _config(tmp_path, shard_dir)
    shared_centroids = torch.tensor([[10.0, 11.0], [1010.0, 1011.0]])
    centroids_path = tmp_path / "shared_centroids.pt"
    torch.save(shared_centroids, centroids_path)
    config["training"]["centroids_path"] = str(centroids_path)
    run = resolve_experiment(_write_yaml(tmp_path / "experiment.yaml", config))[0]

    run_dir = execute_run(run)

    assert (run_dir / "mfa_model.pt").is_file()
    copied_centroids = torch.load(
        run_dir / "centroids.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert torch.equal(copied_centroids, shared_centroids)
    assert (run_dir / "mfa_model_assignments.pt").is_file()
    assert (run_dir / "PIPELINE_COMPLETED.json").is_file()
    assignments = torch.load(
        run_dir / "mfa_model_assignments.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert assignments["assignments"].numel() == 2 * 4 * 3
    assert int(assignments["cluster_sizes"].sum()) == 2 * 4 * 3


def test_real_adaptive_q_pipeline_runs_end_to_end(tmp_path: Path) -> None:
    shard_dir = save_toy_manifold_shards(
        tmp_path / "toy_shards",
        ToyManifoldConfig(
            ambient_dim=8,
            n_train=64,
            n_val=32,
            calibration_size=32,
            manifolds_per_type=1,
            offset_radius=2.0,
            seed=0,
        ),
        shard_size=24,
        layer=0,
    )
    config = _config(tmp_path, shard_dir)
    config["dataset"].update({"id": "toy-manifolds", "layer": 0})
    config["model"] = {
        "kind": "hddc",
        "K": 8,
        "q_max": 2,
        "isotropic_psi": True,
        "surgery_every_epochs": 0,
    }
    config["training"].update(
        {
            "batch_size": 16,
            "max_steps": 2,
            "pool_size": 32,
        }
    )
    config["assignments"].update({"batch_size": 32})
    config["evaluation"] = {
        "enabled": True,
        "kind": "adaptive_q_toy",
        "batch_size": 32,
        "device": "cpu",
    }
    run = resolve_experiment(_write_yaml(tmp_path / "experiment.yaml", config))[0]

    run_dir = execute_run(run)

    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["evaluation"] == "adaptive_q_toy"
    assert metrics["identity_hash"] == run["identity_hash"]
    assert metrics["dataset"]["selected_rows"] == 96
    assert (run_dir / "EVALUATION_COMPLETED.json").is_file()
    assert pipeline_status([run])[0]["pipeline"] is True
