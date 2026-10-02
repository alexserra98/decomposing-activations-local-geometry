from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest
import torch
import yaml

from dalg.data.manifold_dataset import (
    MANIFOLD_NAMES,
    ToyManifoldConfig,
    save_toy_manifold_shards,
)
from dalg.pipeline import (
    PipelineConfigError,
    _evaluation_artifact_valid,
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
from dalg.models.kmeans import KMeans, save_kmeans
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
    payload = copy.deepcopy(payload)
    training = payload["training"]
    if (not training.get("kmeans_model_path") and not training.get("init_model_path")
            and not (payload["model"]["kind"] == "hddc" and training.get("fit_method") == "em")):
        # Tiny shard fixtures need an explicit neighborhood smaller than 64.
        payload.setdefault("initialization", {"pca_neighbors": 4})
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


def _save_kmeans(path: Path, means: torch.Tensor, rank: int = 0) -> KMeans:
    model = KMeans.from_centroids(means)
    if rank:
        generator = torch.Generator().manual_seed(11)
        points = torch.randn(32, means.shape[1], generator=generator)
        model.compute_init_pcs(points, rank=rank, neighbors=16)
    save_kmeans(model, path)
    return model


def test_shared_kmeans_is_validated_resolved_and_forwarded(tmp_path: Path) -> None:
    shards = build_multi_shard(tmp_path / "shards", n_shards=1, rows_per_shard=4)
    checkpoint = tmp_path / "kmeans_model.pt"
    _save_kmeans(checkpoint, torch.zeros(2, 2))
    config = _config(tmp_path, shards)
    config["training"]["kmeans_model_path"] = str(checkpoint)
    run = resolve_experiment(_write_yaml(tmp_path / "experiment.yaml", config))[0]
    assert "initialization" not in run
    assert run["identity"]["training_args"]["kmeans_model_path"] == str(checkpoint.resolve())
    command = _training_command(run)
    assert command[command.index("--kmeans-model-path") + 1] == str(checkpoint.resolve())


@pytest.mark.parametrize("means", [torch.zeros(3, 2), torch.zeros(2, 3)])
def test_incompatible_kmeans_dimensions_are_rejected(tmp_path: Path, means: torch.Tensor) -> None:
    shards = build_multi_shard(tmp_path / "shards", n_shards=1, rows_per_shard=4)
    checkpoint = tmp_path / "kmeans_model.pt"
    _save_kmeans(checkpoint, means)
    config = _config(tmp_path, shards)
    config["training"]["kmeans_model_path"] = str(checkpoint)
    with pytest.raises(PipelineConfigError, match="does not match"):
        resolve_experiment(_write_yaml(tmp_path / "experiment.yaml", config))


def test_missing_kmeans_model_is_rejected(tmp_path: Path) -> None:
    shards = build_multi_shard(tmp_path / "shards", n_shards=1, rows_per_shard=4)
    config = _config(tmp_path, shards)
    config["training"]["kmeans_model_path"] = str(tmp_path / "missing.pt")
    with pytest.raises(PipelineConfigError, match="model file not found"):
        resolve_experiment(_write_yaml(tmp_path / "experiment.yaml", config))


@pytest.mark.parametrize("name", ["model.pth", "model", "model.PT", "directory.pt"])
def test_kmeans_model_requires_direct_pt_file(tmp_path: Path, name: str) -> None:
    shards = build_multi_shard(tmp_path / "shards", n_shards=1, rows_per_shard=4)
    checkpoint = tmp_path / name
    if name == "directory.pt":
        checkpoint.mkdir()
    else:
        _save_kmeans(checkpoint, torch.zeros(2, 2))
    config = _config(tmp_path, shards)
    config["training"]["kmeans_model_path"] = str(checkpoint)
    with pytest.raises(PipelineConfigError, match="must point directly"):
        resolve_experiment(_write_yaml(tmp_path / "experiment.yaml", config))


@pytest.mark.parametrize("legacy_key", ["centroids_path", "kmeans_model_path"])
def test_legacy_centroid_artifact_is_not_a_pipeline_model(tmp_path: Path, legacy_key: str) -> None:
    shards = build_multi_shard(tmp_path / "shards", n_shards=1, rows_per_shard=4)
    checkpoint = tmp_path / "centroids.pt"
    torch.save(torch.zeros(2, 2), checkpoint)
    config = _config(tmp_path, shards)
    config["training"][legacy_key] = str(checkpoint)
    with pytest.raises(PipelineConfigError, match="deprecated|invalid KMeans model"):
        resolve_experiment(_write_yaml(tmp_path / "experiment.yaml", config))


@pytest.mark.parametrize("model", [
    {"kind": "mfa", "K": 2, "rank": 1},
    {"kind": "ard", "K": 2, "rank": 1, "ard_lambda": 0.0},
    {"kind": "hddc", "K": 2, "q_max": 1},
])
def test_cluster_pca_direction_init_is_forwarded_for_every_trainer(tmp_path: Path, model: dict) -> None:
    shards = build_multi_shard(tmp_path / "shards", n_shards=1, rows_per_shard=4)
    checkpoint = tmp_path / "kmeans_model.pt"
    _save_kmeans(checkpoint, torch.zeros(2, 2), rank=1)
    config = _config(tmp_path, shards)
    config["model"] = model
    config["training"].update(kmeans_model_path=str(checkpoint), direction_init="cluster_pca")
    run = resolve_experiment(_write_yaml(tmp_path / "experiment.yaml", config))[0]
    command = _training_command(run)
    assert command[command.index("--direction-init") + 1] == "cluster_pca"


@pytest.mark.parametrize("stored_rank,requested_rank", [(0, 1), (1, 2)])
def test_cluster_pca_requires_enough_stored_directions(tmp_path: Path, stored_rank: int, requested_rank: int) -> None:
    shards = build_multi_shard(tmp_path / "shards", n_shards=1, rows_per_shard=4)
    checkpoint = tmp_path / "kmeans_model.pt"
    _save_kmeans(checkpoint, torch.zeros(2, 2), rank=stored_rank)
    config = _config(tmp_path, shards)
    config["model"]["rank"] = requested_rank
    config["training"].update(kmeans_model_path=str(checkpoint), direction_init="cluster_pca")
    with pytest.raises(PipelineConfigError, match="principal components"):
        resolve_experiment(_write_yaml(tmp_path / "experiment.yaml", config))


def test_cluster_pca_plans_generated_model_without_fitting(tmp_path: Path) -> None:
    shards = build_multi_shard(tmp_path / "shards", n_shards=1, rows_per_shard=4)
    config = _config(tmp_path, shards)
    run = resolve_experiment(_write_yaml(tmp_path / "experiment.yaml", config))[0]
    assert run["initialization"]["rank"] == 1
    assert run["initialization"]["version"] == 4
    assert run["initialization"]["method"] == "kmeans_model"
    assert not Path(run["training"]["arguments"]["kmeans_model_path"]).exists()


def test_cluster_pca_rejects_malformed_checkpoint(tmp_path: Path) -> None:
    shards = build_multi_shard(tmp_path / "shards", n_shards=1, rows_per_shard=4)
    checkpoint = tmp_path / "kmeans_model.pt"
    _save_kmeans(checkpoint, torch.zeros(2, 2), rank=1)
    payload = torch.load(checkpoint, weights_only=True)
    payload["state_dict"]["_pcs"] = torch.ones(2, 1, 2)
    torch.save(payload, checkpoint)
    config = _config(tmp_path, shards)
    config["training"].update(kmeans_model_path=str(checkpoint), direction_init="cluster_pca")
    with pytest.raises(PipelineConfigError, match="invalid KMeans model"):
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
    assert "initialization" not in run
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
    centroids_path = tmp_path / "centroids.pt"
    _save_kmeans(centroids_path, torch.zeros(2, 2))
    config["training"]["kmeans_model_path"] = str(centroids_path)
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
    _save_kmeans(centroids_path, shared_centroids)
    config["training"]["kmeans_model_path"] = str(centroids_path)
    run = resolve_experiment(_write_yaml(tmp_path / "experiment.yaml", config))[0]

    run_dir = execute_run(run)

    assert (run_dir / "mfa_model.pt").is_file()
    assert not (run_dir / "centroids.pt").exists()
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
    initial = _save_kmeans(centroids_path, centroids, rank=1)
    directions = initial.W_init.clone()
    config = _config(tmp_path, shard_dir)
    config["training"].update(
        {
            "kmeans_model_path": str(centroids_path),
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
    assert not (run_dir / "centroids.pt").exists()


@pytest.mark.parametrize("hard", [False, True])
def test_real_shared_b_pipeline_smoke(tmp_path: Path, hard: bool) -> None:
    shard_dir = build_multi_shard(tmp_path / "shards", n_shards=2, rows_per_shard=4)
    config = _config(tmp_path, shard_dir)
    config["model"] = {
        "kind": "hddc",
        "K": 2,
        "q_max": 1,
        "shared_b": True,
        "surgery_every_epochs": 1 if hard else 0,
    }
    config["experimental"] = {"hard_assignment_covariance": hard}
    config["assignments"]["enabled"] = False
    run = resolve_experiment(_write_yaml(tmp_path / "experiment.yaml", config))[0]

    run_dir = execute_run(run)
    trained = load_mfa_hddc(run_dir / "mfa_model.pt", map_location="cpu")
    saved_config = json.loads((run_dir / "config.json").read_text())

    assert trained.shared_b is True
    assert tuple(trained.psi_rho.shape) == (1,)
    assert saved_config["shared_b"] is True
    assert saved_config["hard_assignment_covariance"] is hard


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
    assert run["evaluation"]["max_mean_to_manifold_distance"] is None


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
    assert metrics["schema_version"] == 2
    assert metrics["evaluation"] == "toy_manifold_tiling"
    for rank_name in ("rank", "ambient_rank"):
        assert metrics[rank_name]["definition"] == "hddc_rank_mask_count"
        assert "threshold" not in metrics[rank_name]
        assert metrics[rank_name]["mean_learned"] == 2.0
    assert metrics["association"] == {
        "rule": "unique_nearest_exact_projection",
        "max_mean_to_manifold_distance": None,
        "associated_components": metrics["K"],
        "outside_cutoff_components": 0,
        "ambiguous_components": 0,
    }
    assert metrics["identity_hash"] == run["identity_hash"]
    assert metrics["dataset"]["selected_rows"] == 96
    assert metrics["bic"]["n"] == metrics["dataset"]["train_rows"]
    assert metrics["bic"]["parameters"] > 0
    assert metrics["bic"]["convention"] == "higher_is_better"
    assert metrics["bic"]["value"] == pytest.approx(
        -metrics["bic"]["standard_bic"] / metrics["bic"]["n"]
        + metrics["bic"]["active_components"]
    )
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
    assert len(metrics["per_manifold"]) == len(MANIFOLD_NAMES)
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
        "best_intrinsic_dim_subset_of_leading_rank_covariance_principal_angles"
    )
    for metric in (alignment, containment, metrics["tangent_partial_containment"]):
        for score_name in ("subspace_overlap", "worst_direction_cosine"):
            summary = metric[score_name]
            assert summary["valid_components"] + summary["undefined_components"] == (
                metrics["association"]["associated_components"]
            )
            if summary["mean"] is not None:
                assert 0.0 <= summary["mean"] <= 1.0
    assert (run_dir / "EVALUATION_COMPLETED.json").is_file()
    assert pipeline_status([run])[0]["pipeline"] is True
    assert _evaluation_artifact_valid(run)
    containment_definition = metrics["tangent_containment"].pop("definition")
    (run_dir / "metrics.json").write_text(json.dumps(metrics))
    assert not _evaluation_artifact_valid(run)
    for old_definition in ['leading_learned_rank_covariance_subspace_principal_angles', 'leading_effective_rank_covariance_subspace_principal_angles']:
        metrics["tangent_containment"]["definition"] = old_definition
        (run_dir / "metrics.json").write_text(json.dumps(metrics))
        assert not _evaluation_artifact_valid(run)
    metrics["tangent_containment"]["definition"] = containment_definition
    (run_dir / "metrics.json").write_text(json.dumps(metrics))
    assert _evaluation_artifact_valid(run)
    for metric_name in ("tangent_alignment", "tangent_containment", "tangent_partial_containment"):
        rank_requirement = metrics[metric_name].pop("rank_requirement")
        (run_dir / "metrics.json").write_text(json.dumps(metrics))
        assert not _evaluation_artifact_valid(run)
        metrics[metric_name]["rank_requirement"] = rank_requirement
    adjusted = metrics.pop("tangent_adjusted_alignment")
    (run_dir / "metrics.json").write_text(json.dumps(metrics))
    assert not _evaluation_artifact_valid(run)
    metrics["tangent_adjusted_alignment"] = adjusted
    for field in ("definition", "rank_requirement", "normalization", "zero_rank", "aggregation"):
        original = adjusted[field]
        adjusted[field] = "obsolete_contract"
        (run_dir / "metrics.json").write_text(json.dumps(metrics))
        assert not _evaluation_artifact_valid(run)
        adjusted[field] = original
    (run_dir / "metrics.json").write_text(json.dumps(metrics))
    assert _evaluation_artifact_valid(run)
    partial = metrics.pop("tangent_partial_containment")
    (run_dir / "metrics.json").write_text(json.dumps(metrics))
    assert not _evaluation_artifact_valid(run)
    metrics["tangent_partial_containment"] = partial
    partial["normalization"] = "intrinsic_dim"
    (run_dir / "metrics.json").write_text(json.dumps(metrics))
    assert not _evaluation_artifact_valid(run)
    partial["normalization"] = "effective_rank"
    current_definition = metrics["tangent_alignment"]["definition"]
    metrics["tangent_alignment"]["definition"] = "leading_min_intrinsic_effective_rank_covariance_subspace_principal_angles"
    (run_dir / "metrics.json").write_text(json.dumps(metrics))
    assert not _evaluation_artifact_valid(run)
    metrics["tangent_alignment"]["definition"] = current_definition
    for rank_name in ("rank", "ambient_rank"):
        metrics[rank_name].pop("definition")
        metrics[rank_name]["threshold"] = 1.0
    (run_dir / "metrics.json").write_text(json.dumps(metrics))
    assert not _evaluation_artifact_valid(run)


@pytest.mark.parametrize(
    ("schema_version", "convention", "formula", "value", "expected"),
    [
        (2, "higher_is_better", "-standard_bic / n + active_components", 12.0, True),
        (1, "lower_is_better", None, 12.0, False),
        (1, "higher_is_better", "-standard_bic / n + active_components", 12.0, False),
        (2, "lower_is_better", "-standard_bic / n + active_components", 12.0, False),
        (2, "higher_is_better", None, 12.0, False),
        (2, "higher_is_better", "-standard_bic / n + active_components", float("nan"), False),
    ],
)
def test_evaluation_artifact_requires_augmented_bic(
    tmp_path: Path,
    schema_version: int,
    convention: str,
    formula: str | None,
    value: float,
    expected: bool,
) -> None:
    run = {
        "run_dir": str(tmp_path),
        "training": {"model_kind": "mfa"},
        "evaluation": {"kind": "toy_manifold_tiling"},
        "identity_hash": "test-run",
    }
    (tmp_path / "metrics.json").write_text(
        json.dumps(
            {
                "schema_version": schema_version,
                "evaluation": "toy_manifold_tiling",
                "identity_hash": "test-run",
                "tangent_alignment": {
                    "definition": "leading_intrinsic_dim_covariance_subspace_principal_angles",
                    "rank_requirement": "effective_rank_gte_intrinsic_dim",
                },
                "tangent_containment": {
                    "definition": "best_intrinsic_dim_subset_of_leading_rank_covariance_principal_angles",
                    "rank_requirement": "effective_rank_gte_intrinsic_dim",
                },
                "tangent_partial_containment": {
                    "definition": "leading_covariance_subspace_within_tangent_principal_angles",
                    "rank_requirement": "effective_rank_gt_zero_lt_intrinsic_dim",
                    "normalization": "effective_rank",
                },
                "tangent_adjusted_alignment": {
                    "definition": "leading_min_intrinsic_effective_rank_covariance_subspace_overlap",
                    "rank_requirement": "effective_rank_gte_zero",
                    "normalization": "intrinsic_dim",
                    "zero_rank": "zero_if_tangent_defined",
                    "aggregation": "unweighted_component_mean",
                },
                "bic": {"value": value, "formula": formula, "convention": convention},
            }
        )
    )

    assert _evaluation_artifact_valid(run) is expected


def test_hard_covariance_sweep_reaches_cli_and_surgery(tmp_path):
    from dalg.cli.adaptive_q.run_training_hddc import _surgery_config, build_parser

    shard_dir = build_multi_shard(tmp_path / "shards", n_shards=1, rows_per_shard=4)
    config = _config(tmp_path, shard_dir)
    config["model"] = {
        "kind": "hddc", "K": 2, "q_max": 1, "shared_b": True,
        "surgery_every_epochs": 1,
    }
    config["experimental"] = {"hard_assignment_covariance": False}
    config["sweep"] = {"experimental.hard_assignment_covariance": [False, True]}
    soft, hard = resolve_experiment(_write_yaml(tmp_path / "sweep.yaml", config))
    assert soft["run_id"] != hard["run_id"]
    for run, enabled in [(soft, False), (hard, True)]:
        assert run["identity"]["training_args"]["hard_assignment_covariance"] is enabled
        command = _training_command(run)
        assert ("--hard-assignment-covariance" in command) is enabled
        cli_args = build_parser().parse_args(command[command.index("--shard-dir"):])
        assert _surgery_config(cli_args).hard_assignment_covariance is enabled
    del config["experimental"], config["sweep"]
    default = resolve_experiment(_write_yaml(tmp_path / "default.yaml", config))[0]
    assert default["run_id"] == soft["run_id"]


@pytest.mark.parametrize("case, message", [
    ("unknown", "unknown experimental parameters"),
    ("string", "must be true or false"),
    ("integer", "must be true or false"),
    ("non_mapping", "experimental must be a mapping"),
    ("mfa", "requires model.kind: hddc"),
    ("ard", "requires model.kind: hddc"),
    ("kmeans", "requires model.kind: hddc"),
    ("em", "requires Adam with enabled periodic surgery"),
    ("disabled", "requires Adam with enabled periodic surgery"),
    ("misplaced", "belongs in experimental"),
])
def test_invalid_hard_covariance_configuration(tmp_path, case, message):
    shard_dir = build_multi_shard(tmp_path / "shards", n_shards=1, rows_per_shard=4)
    config = _config(tmp_path, shard_dir)
    config["model"] = {
        "kind": "hddc", "K": 2, "rank": 1, "shared_b": True,
        "surgery_every_epochs": 1,
    }
    config["experimental"] = {"hard_assignment_covariance": True}
    if case == "unknown":
        config["experimental"]["typo"] = True
    elif case in {"string", "integer"}:
        config["experimental"]["hard_assignment_covariance"] = "true" if case == "string" else 1
    elif case == "non_mapping":
        config["experimental"] = True
    elif case in {"mfa", "ard", "kmeans"}:
        config["model"] = {"kind": case, "K": 2, "rank": 1}
    elif case == "em":
        config["training"]["fit_method"] = "em"
        config["model"]["surgery_every_epochs"] = 0
    elif case == "disabled":
        config["model"]["surgery_every_epochs"] = 0
    elif case == "misplaced":
        config["training"]["hard_assignment_covariance"] = True
    with pytest.raises(PipelineConfigError, match=message):
        resolve_experiment(_write_yaml(tmp_path / "invalid.yaml", config))
