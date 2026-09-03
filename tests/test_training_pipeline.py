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
from dalg.models.adaptive_q.mfa_hddc import (
    MFA_HDDC,
    load_mfa_hddc,
    save_mfa_hddc,
)
from dalg.models.mfa import load_mfa
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


@pytest.mark.parametrize(
    "model",
    [
        {"kind": "mfa", "K": 2, "rank": 1},
        {"kind": "ard", "K": 2, "rank": 1, "ard_lambda": 0.0},
        {"kind": "hddc", "K": 2, "q_max": 1},
    ],
)
def test_cluster_pca_direction_init_is_forwarded_for_every_trainer(
    tmp_path: Path,
    model: dict,
) -> None:
    shard_dir = build_multi_shard(tmp_path / "shards", n_shards=1, rows_per_shard=4)
    centroids_path = tmp_path / "centroids.pt"
    torch.save(
        {
            "centroids": torch.zeros(2, 2),
            "principal_components": torch.eye(2).reshape(2, 2, 1),
        },
        centroids_path,
    )
    config = _config(tmp_path, shard_dir)
    config["model"] = model
    config["training"].update(
        {
            "centroids_path": str(centroids_path),
            "direction_init": "cluster_pca",
        }
    )

    run = resolve_experiment(_write_yaml(tmp_path / "experiment.yaml", config))[0]
    command = _training_command(run)

    assert run["training"]["arguments"]["direction_init"] == "cluster_pca"
    option = command.index("--direction-init")
    assert command[option + 1] == "cluster_pca"


def test_cluster_pca_requires_principal_components(tmp_path: Path) -> None:
    shard_dir = build_multi_shard(tmp_path / "shards", n_shards=1, rows_per_shard=4)
    centroids_path = tmp_path / "centroids.pt"
    torch.save(torch.zeros(2, 2), centroids_path)
    config = _config(tmp_path, shard_dir)
    config["training"].update(
        {
            "centroids_path": str(centroids_path),
            "direction_init": "cluster_pca",
        }
    )

    with pytest.raises(PipelineConfigError, match="containing principal_components"):
        resolve_experiment(_write_yaml(tmp_path / "experiment.yaml", config))


def test_cluster_pca_requires_centroids_path(tmp_path: Path) -> None:
    shard_dir = build_multi_shard(tmp_path / "shards", n_shards=1, rows_per_shard=4)
    config = _config(tmp_path, shard_dir)
    config["training"]["direction_init"] = "cluster_pca"

    with pytest.raises(PipelineConfigError, match="requires --centroids-path"):
        resolve_experiment(_write_yaml(tmp_path / "experiment.yaml", config))


def test_cluster_pca_rejects_insufficient_stored_rank(tmp_path: Path) -> None:
    shard_dir = build_multi_shard(tmp_path / "shards", n_shards=1, rows_per_shard=4)
    centroids_path = tmp_path / "centroids.pt"
    torch.save(
        {
            "centroids": torch.zeros(2, 2),
            "principal_components": torch.ones(2, 2, 1),
        },
        centroids_path,
    )
    config = _config(tmp_path, shard_dir)
    config["model"]["rank"] = 2
    config["training"].update(
        {
            "centroids_path": str(centroids_path),
            "direction_init": "cluster_pca",
        }
    )

    with pytest.raises(PipelineConfigError, match="stores 1 principal components"):
        resolve_experiment(_write_yaml(tmp_path / "experiment.yaml", config))


def test_cluster_pca_rejects_malformed_direction_shape(tmp_path: Path) -> None:
    shard_dir = build_multi_shard(tmp_path / "shards", n_shards=1, rows_per_shard=4)
    centroids_path = tmp_path / "centroids.pt"
    torch.save(
        {
            "centroids": torch.zeros(2, 2),
            "principal_components": torch.ones(2, 1, 2),
        },
        centroids_path,
    )
    config = _config(tmp_path, shard_dir)
    config["training"].update(
        {
            "centroids_path": str(centroids_path),
            "direction_init": "cluster_pca",
        }
    )

    with pytest.raises(PipelineConfigError, match="leading dimensions"):
        resolve_experiment(_write_yaml(tmp_path / "experiment.yaml", config))


def test_hddc_initial_model_is_validated_resolved_and_forwarded(tmp_path: Path) -> None:
    shard_dir = build_multi_shard(tmp_path / "shards", n_shards=1, rows_per_shard=4)
    initial_path = tmp_path / "mfa_model.pt"
    save_mfa_hddc(
        MFA_HDDC(torch.zeros(2, 2), rank=2, isotropic_psi=True),
        str(initial_path),
    )
    config = _config(tmp_path, shard_dir)
    config["model"] = {
        "kind": "hddc",
        "K": 2,
        "q_max": 2,
        "isotropic_psi": True,
    }
    config["training"]["init_model_path"] = str(initial_path)
    path = _write_yaml(tmp_path / "experiment.yaml", config)

    run = resolve_experiment(path)[0]
    resolved = str(initial_path.resolve())

    assert run["training"]["arguments"]["init_model_path"] == resolved
    assert run["identity"]["training_args"]["init_model_path"] == resolved
    command = _training_command(run)
    assert command[command.index("--init-model-path") + 1] == resolved


def test_hddc_shared_b_is_forwarded_as_a_single_process_model(tmp_path: Path) -> None:
    shard_dir = build_multi_shard(tmp_path / "shards", n_shards=1, rows_per_shard=4)
    config = _config(tmp_path, shard_dir)
    config["model"] = {
        "kind": "hddc",
        "K": 2,
        "q_max": 1,
        "shared_b": True,
    }
    path = _write_yaml(tmp_path / "experiment.yaml", config)

    run = resolve_experiment(path)[0]
    command = _training_command(run)

    assert run["training"]["arguments"]["shared_b"] is True
    assert run["training"]["arguments"]["isotropic_psi"] is False
    assert "--shared-b" in command
    assert "--isotropic-psi" not in command


def test_hddc_fractional_epoch_surgery_is_forwarded(tmp_path: Path) -> None:
    shard_dir = build_multi_shard(tmp_path / "shards", n_shards=1, rows_per_shard=4)
    config = _config(tmp_path, shard_dir)
    config["model"] = {
        "kind": "hddc",
        "K": 2,
        "q_max": 1,
        "shared_b": True,
        "surgery_every_epochs": 0.3,
    }
    path = _write_yaml(tmp_path / "experiment.yaml", config)

    run = resolve_experiment(path)[0]
    command = _training_command(run)

    assert run["training"]["arguments"]["surgery_every_epochs"] == 0.3
    option = command.index("--surgery-every-epochs")
    assert command[option + 1] == "0.3"


def test_hddc_zero_surgery_min_count_disables_the_cutoff(tmp_path: Path) -> None:
    shard_dir = build_multi_shard(tmp_path / "shards", n_shards=1, rows_per_shard=4)
    config = _config(tmp_path, shard_dir)
    config["model"] = {
        "kind": "hddc",
        "K": 2,
        "q_max": 1,
        "shared_b": True,
        "surgery_every_epochs": 1,
        "surgery_min_count": 0,
    }

    run = resolve_experiment(_write_yaml(tmp_path / "experiment.yaml", config))[0]
    command = _training_command(run)

    assert run["training"]["arguments"]["surgery_min_count"] == 0.0
    option = command.index("--surgery-min-count")
    assert command[option + 1] == "0.0"


def test_hddc_negative_surgery_min_count_is_rejected(tmp_path: Path) -> None:
    shard_dir = build_multi_shard(tmp_path / "shards", n_shards=1, rows_per_shard=4)
    config = _config(tmp_path, shard_dir)
    config["model"] = {
        "kind": "hddc",
        "K": 2,
        "q_max": 1,
        "shared_b": True,
        "surgery_every_epochs": 1,
        "surgery_min_count": -1,
    }

    with pytest.raises(PipelineConfigError, match="finite and non-negative"):
        resolve_experiment(_write_yaml(tmp_path / "experiment.yaml", config))


def test_hddc_shared_b_warm_start_requires_the_same_noise_mode(tmp_path: Path) -> None:
    shard_dir = build_multi_shard(tmp_path / "shards", n_shards=1, rows_per_shard=4)
    initial_path = tmp_path / "mfa_model.pt"
    save_mfa_hddc(
        MFA_HDDC(torch.zeros(2, 2), rank=1, isotropic_psi=True),
        str(initial_path),
    )
    config = _config(tmp_path, shard_dir)
    config["model"] = {
        "kind": "hddc",
        "K": 2,
        "q_max": 1,
        "shared_b": True,
    }
    config["training"]["init_model_path"] = str(initial_path)

    with pytest.raises(PipelineConfigError, match="same Psi noise mode"):
        resolve_experiment(_write_yaml(tmp_path / "experiment.yaml", config))


def test_hddc_shared_b_rejects_component_sharding(tmp_path: Path) -> None:
    shard_dir = build_multi_shard(tmp_path / "shards", n_shards=1, rows_per_shard=4)
    config = _config(tmp_path, shard_dir)
    config["model"] = {
        "kind": "hddc",
        "K": 2,
        "q_max": 1,
        "shared_b": True,
    }
    config["training"].update(
        {"device": "cuda", "training_mode": "component_shard"}
    )
    config["resources"].update({"gpus": 2, "gpu_type": "H100"})

    with pytest.raises(PipelineConfigError, match="shared-b supports.*single_process only"):
        resolve_experiment(_write_yaml(tmp_path / "experiment.yaml", config))


def test_hddc_defaults_to_single_process_training_mode(tmp_path: Path) -> None:
    shard_dir = build_multi_shard(tmp_path / "shards", n_shards=1, rows_per_shard=4)
    config = _config(tmp_path, shard_dir)
    config["model"] = {"kind": "hddc", "K": 2, "q_max": 1}

    run = resolve_experiment(_write_yaml(tmp_path / "experiment.yaml", config))[0]
    command = _training_command(run)

    assert run["training"]["arguments"]["training_mode"] == "single_process"
    option = command.index("--training-mode")
    assert command[option + 1] == "single_process"


def test_hddc_rejects_vanilla_as_a_training_mode(tmp_path: Path) -> None:
    shard_dir = build_multi_shard(tmp_path / "shards", n_shards=1, rows_per_shard=4)
    config = _config(tmp_path, shard_dir)
    config["model"] = {"kind": "hddc", "K": 2, "q_max": 1}
    config["training"]["training_mode"] = "vanilla"

    with pytest.raises(PipelineConfigError, match="invalid hddc training parameters"):
        resolve_experiment(_write_yaml(tmp_path / "experiment.yaml", config))


def test_hddc_shared_b_and_isotropic_psi_are_mutually_exclusive(tmp_path: Path) -> None:
    shard_dir = build_multi_shard(tmp_path / "shards", n_shards=1, rows_per_shard=4)
    config = _config(tmp_path, shard_dir)
    config["model"] = {
        "kind": "hddc",
        "K": 2,
        "q_max": 1,
        "isotropic_psi": True,
        "shared_b": True,
    }

    with pytest.raises(PipelineConfigError, match="select different noise modes"):
        resolve_experiment(_write_yaml(tmp_path / "experiment.yaml", config))


def test_hddc_initial_model_rejects_rank_mismatch(tmp_path: Path) -> None:
    shard_dir = build_multi_shard(tmp_path / "shards", n_shards=1, rows_per_shard=4)
    initial_path = tmp_path / "mfa_model.pt"
    save_mfa_hddc(
        MFA_HDDC(torch.zeros(2, 2), rank=1, isotropic_psi=True),
        str(initial_path),
    )
    config = _config(tmp_path, shard_dir)
    config["model"] = {
        "kind": "hddc",
        "K": 2,
        "q_max": 2,
        "isotropic_psi": True,
    }
    config["training"]["init_model_path"] = str(initial_path)
    path = _write_yaml(tmp_path / "experiment.yaml", config)

    with pytest.raises(PipelineConfigError, match="rank q=1 does not match model.q_max=2"):
        resolve_experiment(path)


def test_hddc_initial_model_rejects_k_mismatch(tmp_path: Path) -> None:
    shard_dir = build_multi_shard(tmp_path / "shards", n_shards=1, rows_per_shard=4)
    initial_path = tmp_path / "mfa_model.pt"
    save_mfa_hddc(
        MFA_HDDC(torch.zeros(2, 2), rank=1, isotropic_psi=True),
        str(initial_path),
    )
    config = _config(tmp_path, shard_dir)
    config["model"] = {
        "kind": "hddc",
        "K": 3,
        "q_max": 1,
        "isotropic_psi": True,
    }
    config["training"]["init_model_path"] = str(initial_path)
    path = _write_yaml(tmp_path / "experiment.yaml", config)

    with pytest.raises(
        PipelineConfigError,
        match=r"initial model has \(K=2, D=2\), expected \(K=3, D=2\)",
    ):
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


def test_real_cpu_pipeline_preserves_cluster_pca_directions_at_zero_lr(
    tmp_path: Path,
) -> None:
    shard_dir = build_multi_shard(tmp_path / "shards", n_shards=2, rows_per_shard=4)
    centroids = torch.tensor([[10.0, 11.0], [1010.0, 1011.0]])
    directions = torch.tensor([[[1.0], [0.0]], [[0.0], [1.0]]])
    centroids_path = tmp_path / "shared_centroids.pt"
    torch.save(
        {
            "centroids": centroids,
            "principal_components": directions,
        },
        centroids_path,
    )
    config = _config(tmp_path, shard_dir)
    config["training"].update(
        {
            "centroids_path": str(centroids_path),
            "direction_init": "cluster_pca",
            "lr": 0.0,
            "max_steps": 1,
        }
    )
    config["assignments"]["enabled"] = False
    run = resolve_experiment(_write_yaml(tmp_path / "experiment.yaml", config))[0]

    run_dir = execute_run(run)
    trained = load_mfa(run_dir / "mfa_model.pt", map_location="cpu")

    assert torch.equal(trained.mu.detach(), centroids)
    assert torch.equal(trained.dir_raw.detach(), directions)
    saved_config = json.loads((run_dir / "config.json").read_text())
    assert saved_config["direction_init"] == "cluster_pca"


def test_real_hddc_initial_model_pipeline_smoke(tmp_path: Path) -> None:
    shard_dir = build_multi_shard(tmp_path / "shards", n_shards=2, rows_per_shard=4)
    initial = MFA_HDDC(
        torch.tensor([[10.0, 11.0], [1010.0, 1011.0]]),
        rank=2,
        isotropic_psi=True,
    )
    initial.rank_mask[:, 1] = 0.0
    initial_path = tmp_path / "initial_mfa_model.pt"
    save_mfa_hddc(initial, str(initial_path))

    config = _config(tmp_path, shard_dir)
    config["model"] = {
        "kind": "hddc",
        "K": 2,
        "q_max": 2,
        "isotropic_psi": True,
        "surgery_every_epochs": 0,
    }
    config["training"].update(
        {
            "init_model_path": str(initial_path),
            "max_steps": 1,
        }
    )
    config["assignments"]["enabled"] = False
    run = resolve_experiment(_write_yaml(tmp_path / "experiment.yaml", config))[0]

    run_dir = execute_run(run)

    checkpoint = torch.load(
        run_dir / "checkpoint.pt",
        map_location="cpu",
        weights_only=False,
    )
    trained = load_mfa_hddc(run_dir / "mfa_model.pt", map_location="cpu")
    assert checkpoint["epoch"] == 1
    assert trained.q == 2
    assert torch.equal(trained.rank_mask[:, 1], torch.zeros(2))
    assert torch.equal(
        torch.load(run_dir / "centroids.pt", weights_only=True),
        initial.mu.detach(),
    )


def test_real_shared_b_pipeline_smoke(tmp_path: Path) -> None:
    shard_dir = build_multi_shard(tmp_path / "shards", n_shards=2, rows_per_shard=4)
    config = _config(tmp_path, shard_dir)
    config["model"] = {
        "kind": "hddc",
        "K": 2,
        "q_max": 1,
        "shared_b": True,
        "surgery_every_epochs": 0,
    }
    config["assignments"]["enabled"] = False
    run = resolve_experiment(_write_yaml(tmp_path / "experiment.yaml", config))[0]

    run_dir = execute_run(run)
    trained = load_mfa_hddc(run_dir / "mfa_model.pt", map_location="cpu")
    saved_config = json.loads((run_dir / "config.json").read_text())

    assert trained.shared_b is True
    assert tuple(trained.psi_rho.shape) == (1,)
    assert saved_config["shared_b"] is True


def test_toy_manifold_tiling_evaluation_accepts_vanilla_mfa_config(
    tmp_path: Path,
) -> None:
    shard_dir = build_multi_shard(tmp_path / "shards", n_shards=1, rows_per_shard=4)
    config = _config(tmp_path, shard_dir)
    config["evaluation"] = {
        "enabled": True,
        "kind": "toy_manifold_tiling",
        "device": "cpu",
    }

    run = resolve_experiment(_write_yaml(tmp_path / "experiment.yaml", config))[0]

    assert run["training"]["model_kind"] == "mfa"
    assert run["evaluation"]["kind"] == "toy_manifold_tiling"
    assert run["evaluation"]["rank_threshold"] == 1.0
    assert run["evaluation"]["max_mean_to_manifold_distance"] == 0.1


@pytest.mark.parametrize("distance", [0.0, -0.1, float("inf"), float("nan")])
def test_toy_manifold_tiling_rejects_invalid_mean_distance(
    tmp_path: Path,
    distance: float,
) -> None:
    shard_dir = build_multi_shard(tmp_path / "shards", n_shards=1, rows_per_shard=4)
    config = _config(tmp_path, shard_dir)
    config["evaluation"] = {
        "enabled": True,
        "kind": "toy_manifold_tiling",
        "max_mean_to_manifold_distance": distance,
    }

    with pytest.raises(
        PipelineConfigError,
        match="max_mean_to_manifold_distance must be finite and positive",
    ):
        resolve_experiment(_write_yaml(tmp_path / "experiment.yaml", config))


def test_real_toy_manifold_tiling_pipeline_runs_end_to_end(tmp_path: Path) -> None:
    shard_dir = save_toy_manifold_shards(
        tmp_path / "toy_shards",
        ToyManifoldConfig(
            ambient_dim=32,
            n_samples=96,
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
        "kind": "toy_manifold_tiling",
        "batch_size": 32,
        "device": "cpu",
    }
    run = resolve_experiment(_write_yaml(tmp_path / "experiment.yaml", config))[0]

    run_dir = execute_run(run)

    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["schema_version"] == 1
    assert metrics["evaluation"] == "toy_manifold_tiling"
    assert metrics["rank"]["threshold"] == 1.0
    assert metrics["association"]["max_mean_to_manifold_distance"] == 0.1
    assert metrics["identity_hash"] == run["identity_hash"]
    assert metrics["dataset"]["selected_rows"] == 96
    assert metrics["bic"]["n"] == metrics["dataset"]["train_rows"]
    assert metrics["bic"]["parameters"] > 0
    assert metrics["bic"]["convention"] == "lower_is_better"
    assert torch.isfinite(torch.tensor(metrics["bic"]["value"]))
    association_counts = sum(
        metrics["association"][key]
        for key in (
            "associated_components",
            "outside_cutoff_components",
            "ambiguous_components",
        )
    )
    assert association_counts == metrics["K"]
    assert len(metrics["per_manifold"]) == 10
    assert sum(
        manifold["components"]["associated"]
        for manifold in metrics["per_manifold"]
    ) == metrics["association"]["associated_components"]
    alignment = metrics["tangent_alignment"]
    assert alignment["definition"] == (
        "leading_intrinsic_dim_covariance_subspace_principal_angles"
    )
    containment = metrics["tangent_containment"]
    assert containment["definition"] == (
        "leading_effective_rank_covariance_subspace_principal_angles"
    )
    for metric in (alignment, containment):
        for score_name in ("subspace_overlap", "worst_direction_cosine"):
            summary = metric[score_name]
            assert summary["valid_components"] + summary["undefined_components"] == (
                metrics["association"]["associated_components"]
            )
            if summary["mean"] is not None:
                assert 0.0 <= summary["mean"] <= 1.0
    assert (run_dir / "EVALUATION_COMPLETED.json").is_file()
    assert pipeline_status([run])[0]["pipeline"] is True
