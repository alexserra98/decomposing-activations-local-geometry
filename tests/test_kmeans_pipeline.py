from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from dalg.data.manifold_dataset import ToyManifoldConfig, save_toy_manifold_shards
from dalg.data.shard_activations import ActivationBatchDataset
from dalg.init.activation_selection import resolve_initialization_rows
from dalg.models.kmeans import KMeans, load_kmeans
from dalg.pipeline import (
    PipelineConfigError, _assignment_artifact_valid, _evaluation_artifact_valid,
    _training_artifacts_valid, execute_run, pipeline_status, resolve_run,
)


def _config(tmp_path: Path, *, test_fraction: float = 0) -> dict:
    shards = save_toy_manifold_shards(
        tmp_path / "shards",
        ToyManifoldConfig(
            ambient_dim=3, n_samples=120, calibration_size=32,
            manifold_types=("segment", "circle"), manifolds_per_type=1,
            offset_radius=4.0, seed=5,
        ),
        layer=0, shard_size=40, test_fraction=test_fraction,
    )
    if test_fraction == 0:
        from tests.test_toy_manifold_test_split import add_test_split
        add_test_split(shards, n_samples=24)
    return {
        "experiment": {"name": "kmeans-test", "output_root": str(tmp_path / "runs")},
        "dataset": {"shard_dir": str(shards), "layer": 0},
        "model": {"kind": "kmeans", "K": 3},
        "training": {
            "device": "cpu", "max_iter": 10, "restarts": 1,
            "seed": 7, "val_frac": 0.2, "split_seed": 9,
        },
        "assignments": {"enabled": True, "device": "cpu", "batch_size": 19},
        "evaluation": {"enabled": True, "kind": "toy_manifold_tiling", "device": "cpu"},
        "resources": {"gpus": 0, "gpu_type": ""},
    }


@pytest.mark.parametrize("threshold", [None, 0.3])
def test_kmeans_pipeline_end_to_end(tmp_path: Path, monkeypatch, threshold: float | None) -> None:
    config = _config(tmp_path)
    config["model"]["surgery_threshold"] = threshold
    run = resolve_run(config)
    assert "initialization" not in run
    run_dir = execute_run(run)
    model = load_kmeans(run_dir / "kmeans_model.pt", map_location="cpu")
    assignments = torch.load(run_dir / "kmeans_model_assignments.pt", weights_only=True)
    assert not list(run_dir.rglob("centroids.pt"))
    assert not (run_dir / "mfa_model.pt").exists()
    assert not (run_dir / "mfa_model_assignments.pt").exists()
    assert model.q_init == 0
    assert model.pca_valid.all()
    assert model.surgery_threshold == (0.1 if threshold is None else threshold)
    assert model.q == int(model.component_ranks.max())
    assert torch.all(assignments["max_responsibilities"] == 1)
    assert int(assignments["cluster_sizes"].sum()) == 120
    points = torch.cat(list(ActivationBatchDataset(
        config["dataset"]["shard_dir"], layer=0, drop_prefix=0,
        batch_size=17, shuffle_shards=False, shuffle_within_shard=False,
    )))
    assert torch.equal(assignments["assignments"], model.predict(points))
    root, _, train_positions, selection = resolve_initialization_rows(
        config["dataset"]["shard_dir"], layer=0, val_frac=0.2, split_seed=9,
    )
    train_points = torch.cat(list(ActivationBatchDataset(
        root, layer=0, drop_prefix=0, row_subset=train_positions,
        batch_size=17, shuffle_shards=False, shuffle_within_shard=False,
    )))
    expected = KMeans(3, max_iter=10, restarts=1, seed=7, device="cpu").fit(train_points)
    torch.testing.assert_close(model.mu, expected.mu)
    expected.compute_pcs(train_points, threshold=model.surgery_threshold)
    torch.testing.assert_close(model.W @ model.W.transpose(1, 2), expected.W @ expected.W.transpose(1, 2))
    assert model.checkpoint_extra["selection"] == selection
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert "nll" not in metrics and "bic" not in metrics
    assert metrics["rank"]["definition"] == "kmeans_component_ranks"
    assert metrics["rank"]["mean_learned"] == pytest.approx(float(model.component_ranks.float().mean()))
    assert metrics["quantization"]["train"]["n"] == len(train_points)
    assert metrics["quantization"]["validation"]["n"] == 120 - len(train_points)
    expected_sse = (train_points - model.mu[model.predict(train_points)]).square().sum()
    assert metrics["quantization"]["train"]["sum_squared_distance"] == pytest.approx(float(expected_sse), rel=1e-5)
    assert _evaluation_artifact_valid(run)
    assert pipeline_status([run])[0]["pipeline"]
    monkeypatch.setattr("dalg.pipeline._run_command", lambda _: pytest.fail("completed stages must be reused"))
    execute_run(run)
    if threshold is None:
        assignment_path = run_dir / "kmeans_model_assignments.pt"
        original = dict(assignments)
        assignments["assignments"] = assignments["assignments"][:-1]
        assignments["cluster_sizes"] = torch.bincount(assignments["assignments"], minlength=model.K)
        torch.save(assignments, assignment_path)
        assert not _assignment_artifact_valid(run)
        torch.save(original, assignment_path)
        checkpoint_path = run_dir / "kmeans_model.pt"
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        checkpoint["meta"]["extra"]["seed"] = 1234
        torch.save(checkpoint, checkpoint_path)
        assert not _training_artifacts_valid(run)
    containment_definition = metrics["tangent_containment"].pop("definition")
    (run_dir / "metrics.json").write_text(json.dumps(metrics))
    assert not _evaluation_artifact_valid(run)
    for old_definition in ['leading_selected_rank_pca_subspace_principal_angles']:
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
    metrics["tangent_alignment"]["definition"] = "leading_min_intrinsic_effective_rank_pca_subspace_principal_angles"
    (run_dir / "metrics.json").write_text(json.dumps(metrics))
    assert not _evaluation_artifact_valid(run)
    metrics["tangent_alignment"]["definition"] = current_definition
    metrics["quantization"]["train"]["mean_squared_distance"] = float("nan")
    (run_dir / "metrics.json").write_text(json.dumps(metrics))
    assert not _evaluation_artifact_valid(run)


@pytest.mark.parametrize("section,changes,message", [
    ("model", {"rank": 0}, "omit model.rank"),
    ("model", {"rank": 4}, "omit --rank"),
    ("training", {"training_mode": "component_shard"}, "component_shard"),
    ("training", {"lr": 0.01}, "unknown training"),
    ("training", {"pca_method": "knn"}, "unknown training"),
    ("training", {"pca_purpose": "initialization", "rank": 2}, "require cluster geometry"),
    ("training", {"pca_neighbors": 16}, "only valid for initialization"),
    ("resources", {"gpus": 2}, "one CPU or CUDA process"),
])
def test_invalid_kmeans_pipeline_options(tmp_path: Path, section: str, changes: dict, message: str) -> None:
    config = _config(tmp_path)
    config[section].update(changes)
    with pytest.raises(PipelineConfigError, match=message):
        resolve_run(config)


def test_cattell_options_change_run_identity(tmp_path: Path) -> None:
    config = _config(tmp_path)
    first = resolve_run(config)
    config["model"]["surgery_threshold"] = 0.3
    second = resolve_run(config)
    config["model"]["surgery_threshold"] = 0.5
    third = resolve_run(config)
    assert len({r["identity_hash"] for r in (first, second, third)}) == 3
    assert all(not Path(r["run_dir"]).exists() for r in (first, second, third))


def test_kmeans_pipeline_without_validation(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["training"]["val_frac"] = 0.0
    run = resolve_run(config)
    directory = execute_run(run)
    metrics = json.loads((directory / "metrics.json").read_text())
    assert metrics["quantization"]["validation"] == {
        "n": 0, "sum_squared_distance": 0.0, "mean_squared_distance": None,
    }
    assert pipeline_status([run])[0]["pipeline"]


def test_kmeans_pipeline_excludes_reserved_test_points(tmp_path: Path) -> None:
    config = _config(tmp_path, test_fraction=0.2)
    run = resolve_run(config)
    directory = execute_run(run)
    metrics = json.loads((directory / "metrics.json").read_text())
    assert metrics["quantization"]["train"]["n"] == 76
    assert metrics["quantization"]["validation"]["n"] == 20
    assignments = torch.load(directory / "kmeans_model_assignments.pt", weights_only=True)
    assert assignments["assignments"].numel() == 96
    test_config = json.loads((Path(config["dataset"]["shard_dir"]) / "test/config.json").read_text())
    assert test_config["num_rows"] == 24
    assert metrics["heldout_distribution_coverage"]["reference_points"] == 24
    assert metrics["heldout_distribution_coverage"]["source"]["partition"]["kind"] == "reserved"
    assert pipeline_status([run])[0]["pipeline"]


def test_legacy_initialization_manifest_rejected_without_writes(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["model"] = {"kind": "mfa", "K": 2, "rank": 1}
    config["training"] = {"device": "cpu", "val_frac": 0.2}
    config["evaluation"]["enabled"] = False
    run = resolve_run(config)
    run["initialization"].update(method="kmeans_pca", version=1)
    with pytest.raises(PipelineConfigError, match="legacy centroid initialization"):
        execute_run(run)
    assert not Path(run["run_dir"]).exists()


@pytest.mark.parametrize('K,threshold', [(40, None), (40, .1), (95, None), (95, .1)])
def test_sparse_pipeline_keeps_partition_and_resumes(tmp_path, monkeypatch, K, threshold):
    config = _config(tmp_path)
    config['model'].update(K=K, surgery_threshold=threshold)
    run = resolve_run(config)
    directory = execute_run(run)
    model = load_kmeans(directory / 'kmeans_model.pt')
    assert (~model.pca_valid).any()
    metrics = json.loads((directory / 'metrics.json').read_text())
    assert metrics['pca_geometry']['minimum_cluster_points'] == 2
    eligible = int(model.pca_valid.sum())
    assert metrics['pca_geometry']['eligible_components'] == eligible
    assert metrics['pca_geometry']['excluded_components'] == K - eligible
    assert metrics['association']['associated_components'] == K
    assert metrics['rank']['components'] == eligible
    assert metrics['quantization']['train']['n'] == 96
    assert metrics['quantization']['validation']['n'] == 24
    assert metrics['heldout_distribution_coverage']['components_live'] == K
    assignments = torch.load(directory / 'kmeans_model_assignments.pt', weights_only=True)
    assert assignments['assignments'].numel() == 120
    assert assignments['cluster_sizes'].numel() == K
    assert sum(m['components']['pca_excluded'] for m in metrics['per_manifold']) == K - eligible
    if K == 95:
        assert eligible == 1  # 96 training points in 95 distinct clusters.
        assert metrics['rank']['mean_learned'] == 1
    assert pipeline_status([run])[0]['pipeline']
    monkeypatch.setattr('dalg.pipeline._run_command', lambda _: pytest.fail('resume reran a stage'))
    execute_run(run)
    metrics.pop('pca_geometry')
    (directory / 'metrics.json').write_text(json.dumps(metrics))
    assert not _evaluation_artifact_valid(run)


@pytest.mark.parametrize('kind', ['mfa', 'ard', 'hddc'])
def test_sparse_automatic_initialization_uses_knn_only(tmp_path, kind):
    config = _config(tmp_path)
    config['model'] = {'kind': kind, 'K': 40, 'rank': 3}
    if kind == 'hddc':
        config['model']['surgery_every_epochs'] = 0
    config['training'] = {'device': 'cpu', 'val_frac': .2, 'split_seed': 9,
                          'seed': 7, 'max_steps': 1, 'epochs': 1, 'lr': 0., 'num_workers': 0}
    config['initialization'] = {'pca_neighbors': 8}
    config['assignments']['enabled'] = False
    config['evaluation']['enabled'] = False
    run = resolve_run(config)
    directory = execute_run(run)
    initial = load_kmeans(directory / 'initialization' / 'kmeans_model.pt')
    assert initial.q == 0 and initial.q_init == 3
    assert initial.init_pca_neighbors == 8
    assert initial.init_pca_n_samples == 96
    assert initial.surgery_threshold is None
    assert min(initial.checkpoint_extra['cluster_sizes']) <= 3
    assert not list(directory.rglob('centroids.pt'))
    assert pipeline_status([run])[0]['pipeline']


@pytest.mark.parametrize('version', [1, 2, 3])
def test_previous_model_initialization_manifest_rejected(tmp_path, version):
    config = _config(tmp_path)
    config['model'] = {'kind': 'mfa', 'K': 2, 'rank': 1}
    config['training'] = {'device': 'cpu', 'val_frac': .2}
    config['evaluation']['enabled'] = False
    run = resolve_run(config)
    run['initialization']['version'] = version
    with pytest.raises(PipelineConfigError, match='initialization version 4'):
        execute_run(run)
    assert not Path(run['run_dir']).exists()


def test_surgery_threshold_sweep_and_legacy_checkpoint(tmp_path):
    from dalg.cli.run_training_kmeans import build_parser
    from dalg.models.kmeans import save_kmeans
    from dalg.pipeline import expand_sweep

    config = _config(tmp_path)
    config['model']['surgery_threshold'] = 0.05
    config['sweep'] = {'model.surgery_threshold': [0.005, 0.5, 1.0]}
    runs = [resolve_run(row) for row in expand_sweep(config)]
    assert [r['training']['arguments']['surgery_threshold'] for r in runs] == [0.005, 0.5, 1.0]
    assert len({r['run_id'] for r in runs}) == 3
    for flag in ('--surgery-threshold', '--cattell-threshold'):
        args = build_parser().parse_args(['--shard-dir', 'unused', '--layer', '0', '--K', '1', flag, '0.5'])
        assert args.surgery_threshold == 0.5

    points = torch.tensor([[-2., 0.], [-1., 0.], [1., 0.], [2., 0.]])
    model = KMeans.from_centroids(torch.zeros(1, 2)).compute_pcs(points, threshold=0.5)
    path = tmp_path / 'threshold_model.pt'
    save_kmeans(model, path)
    payload = torch.load(path, weights_only=True)
    assert payload['meta']['surgery_threshold'] == 0.5
    assert load_kmeans(path).surgery_threshold == 0.5
    payload['meta']['cattell_threshold'] = payload['meta'].pop('surgery_threshold')
    torch.save(payload, path)
    restored = load_kmeans(path)
    assert restored.surgery_threshold == 0.5
    torch.testing.assert_close(restored.W, model.W)
