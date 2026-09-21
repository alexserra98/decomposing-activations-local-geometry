"""Training-only KMeans/PCA, pipeline defaults, and initialization resume."""

import copy
import hashlib
import importlib
import json
from argparse import Namespace
from pathlib import Path

import pytest
import torch

from dalg.data.shard_activations import load_meta_index, stratified_split
from dalg.init.activation_selection import resolve_initialization_rows
from dalg.init.centroid_artifact import load_centroid_artifact
from dalg.pipeline import (
    _ensure_initialization,
    _initialization_command,
    _training_command,
    execute_run,
    pipeline_status,
    PipelineConfigError,
    resolve_experiment,
    resolve_run,
)
from scripts.temporary.build_toy_kmeans_centroids import (
    _load_activations,
    build_centroids,
    build_parser,
)
from tests.synthetic_shards import LAYER, build_multi_shard
from tests.test_training_pipeline import _config


@pytest.mark.parametrize("kind", ["mfa", "ard", "hddc"])
@pytest.mark.parametrize("direction", [None, "random", "cluster_pca"])
def test_generated_initialization_defaults_and_direct_cli_contract(tmp_path, kind, direction):
    root = build_multi_shard(tmp_path / "shards", n_shards=2, rows_per_shard=8)
    config = _config(tmp_path, root)
    config["model"] = {"kind": kind, "K": 2, "rank": 1}
    if direction is not None:
        config["training"]["direction_init"] = direction
    run = resolve_run(config)
    args = run["training"]["arguments"]
    assert args["direction_init"] == (direction or "cluster_pca")
    assert run["initialization"] == run["identity"]["initialization"]
    assert run["identity"]["training_args"]["centroids_path"] is None
    assert Path(args["centroids_path"]) == Path(run["run_dir"]) / "initialization/centroids.pt"
    assert not Path(run["run_dir"]).exists()
    assert resolve_run(config, check_inputs=False)["identity_hash"] == run["identity_hash"]
    command = _training_command(run)
    assert command[command.index("--centroids-path") + 1] == args["centroids_path"]
    trainer = importlib.import_module(run["training"]["module"])
    direct = trainer.build_parser().parse_args([
        "--shard-dir", str(root), "--layer", str(LAYER), "--K", "2",
        "--direction-init", "cluster_pca",
    ])
    with pytest.raises(SystemExit, match="requires --centroids-path"):
        trainer.validate_args(direct)


@pytest.mark.parametrize("val_frac", [0.0, 0.25])
@pytest.mark.parametrize("subset", [False, True])
def test_builder_rows_match_every_trainer(tmp_path, val_frac, subset):
    root = build_multi_shard(tmp_path / "shards", n_shards=2, rows_per_shard=8)
    for path in (root / "meta").glob("*.json"):
        metadata = json.loads(path.read_text())
        for row in metadata["rows"]:
            if row["subset"] == "A":
                row["subset"] = "pile-wikipedia_en"
        path.write_text(json.dumps(metadata))
    source = json.loads((root / "config.json").read_text())
    source["drop_prefix"] = 1
    (root / "config.json").write_text(json.dumps(source))
    shard_dir = str(root) + ("#pile_wikipedia_12" if subset else "")
    clean, _, positions, selection = resolve_initialization_rows(
        shard_dir, layer=LAYER, val_frac=val_frac, split_seed=19,
    )
    assert clean == root
    assert selection["drop_prefix"] == 1
    expected = torch.cat([
        torch.load(root / f"layer{LAYER:02d}/shard_{p // 8:05d}.pt", weights_only=True)[p % 8, 1:]
        for p in positions
    ])
    actual, _ = _load_activations(
        Path(shard_dir), layer=LAYER, batch_size=7, val_frac=val_frac, split_seed=19,
    )
    assert torch.equal(actual, expected)
    assert selection["train_activations"] == len(expected)
    assert selection["train_rows_sha256"] == hashlib.sha256(
        json.dumps(positions, separators=(",", ":")).encode()
    ).hexdigest()
    for module in (
        "dalg.cli.run_training", "dalg.cli.adaptive_q.run_training_ard",
        "dalg.cli.adaptive_q.run_training_hddc",
    ):
        data = importlib.import_module(module)._resolve_activation_data(
            Namespace(shard_dir=shard_dir, layer=LAYER, K=2, val_frac=val_frac,
                      split_seed=19, out_dir=str(tmp_path / "unused")),
            log=lambda _message: None,
        )
        assert positions == data["train_pos_full"]
        assert selection["train_activations"] == data["n_train_tokens"]


@pytest.mark.parametrize("pca_method", ["cluster", "knn"])
def test_held_out_values_cannot_change_kmeans_or_pca(tmp_path, pca_method):
    root = build_multi_shard(tmp_path / "shards", n_shards=2, rows_per_shard=8)
    config = _config(tmp_path, root)
    if pca_method == "knn":
        config["initialization"] = {"pca_method": "knn", "pca_neighbors": 8}
    run = resolve_run(config)
    command = _initialization_command(run)
    args = build_parser().parse_args(command[2:])
    build_centroids(args)
    before = load_centroid_artifact(args.out_dir / "centroids.pt")
    meta = load_meta_index(root, LAYER)
    _, val = stratified_split(meta, val_frac=0.25, seed=42)
    for shard in range(2):
        path = root / f"layer{LAYER:02d}/shard_{shard:05d}.pt"
        values = torch.load(path, weights_only=True)
        for p in val:
            if meta[p]["shard"] == shard:
                values[meta[p]["row_in_shard"]] = float("nan")
        torch.save(values, path)
    args.out_dir = tmp_path / "after"
    build_centroids(args)
    after = load_centroid_artifact(args.out_dir / "centroids.pt")
    assert all(torch.equal(a, b) for a, b in zip(before, after))
    metadata = json.loads((args.out_dir / "config.json").read_text())
    assert metadata["rows_used"] == 36
    assert metadata["source_rows"] == 48
    assert metadata["uses_all_training_rows"]
    assert not metadata["uses_all_rows"]


@pytest.mark.parametrize("kind", ["mfa", "ard", "hddc"])
@pytest.mark.parametrize("pca_method", ["cluster", "knn"])
def test_cpu_pipeline_initializes_trains_and_resumes(tmp_path, monkeypatch, kind, pca_method):
    root = build_multi_shard(tmp_path / "shards", n_shards=2, rows_per_shard=8)
    config = _config(tmp_path, root)
    config["model"] = {"kind": kind, "K": 2, "rank": 1}
    if pca_method == "knn":
        config["initialization"] = {"pca_method": "knn", "pca_neighbors": 8}
    if kind == "ard":
        config["model"]["ard_lambda"] = 0.0
    if kind == "hddc":
        config["model"].update(shared_b=True, surgery_every_epochs=1)
    config["training"].update(lr=0.0, early_stop_delta=0.0)
    run = resolve_run(config)
    directory = execute_run(run)
    centroids, pcs = load_centroid_artifact(directory / "initialization/centroids.pt")
    assert centroids.shape == (2, 2)
    assert pcs.shape == (2, 2, 1)
    assert torch.allclose(pcs.transpose(1, 2) @ pcs, torch.ones(2, 1, 1))
    assert (directory / "INITIALIZATION_COMPLETED.json").exists()
    split = json.loads((directory / "val_indices.json").read_text())
    metadata = json.loads((directory / "initialization/config.json").read_text())
    assert metadata["selection"]["train_rows"] == split["train_rows"]
    assert metadata["rows_used"] == split["train_rows"] * split["per_row_tokens"]
    for name in ("centroids.pt", "config.json"):
        assert (directory / "initialization" / name).exists()
    monkeypatch.setattr("dalg.pipeline._run_command", lambda _cmd: pytest.fail("completed stage reran"))
    execute_run(run)
    assert pipeline_status([run])[0]["initialization"]
    assert pipeline_status([run])[0]["pipeline"]


def test_resume_after_initialization_and_reject_changed_provenance(tmp_path, monkeypatch):
    root = build_multi_shard(tmp_path / "shards", n_shards=2, rows_per_shard=8)
    run = resolve_run(_config(tmp_path, root))
    _ensure_initialization(run)
    monkeypatch.setattr("dalg.pipeline._run_command", lambda _cmd: pytest.fail("initialization reran"))
    _ensure_initialization(run)
    directory = Path(run["run_dir"])
    path = directory / "initialization/config.json"
    metadata = json.loads(path.read_text())
    metadata["selection"]["split_seed"] += 1
    path.write_text(json.dumps(metadata))
    original = path.read_bytes()
    with pytest.raises(RuntimeError, match="refusing to overwrite invalid initialization"):
        _ensure_initialization(run)
    assert path.read_bytes() == original
    assert not pipeline_status([run])[0]["initialization"]


def test_partial_initialization_is_not_overwritten(tmp_path):
    root = build_multi_shard(tmp_path / "shards", n_shards=2, rows_per_shard=8)
    run = resolve_run(_config(tmp_path, root))
    path = Path(run["run_dir"]) / "initialization/centroids.pt"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"incomplete")
    with pytest.raises(RuntimeError, match="refusing to overwrite invalid initialization"):
        _ensure_initialization(run)
    assert path.read_bytes() == b"incomplete"


def test_legacy_manifest_runs_without_generated_initialization(tmp_path):
    root = build_multi_shard(tmp_path / "shards", n_shards=2, rows_per_shard=8)
    run = copy.deepcopy(resolve_run(_config(tmp_path, root)))
    run.pop("initialization")
    run["identity"].pop("initialization")
    for args in (run["training"]["arguments"], run["identity"]["training_args"]):
        args["centroids_path"] = None
        args["direction_init"] = "random"
    assert "--centroids-path" not in _training_command(run)
    directory = execute_run(run)
    assert isinstance(torch.load(directory / "centroids.pt", weights_only=True), torch.Tensor)
    assert not (directory / "initialization").exists()
    assert pipeline_status([run])[0]["pipeline"]


def test_builder_rejects_undersized_clusters_without_saving(tmp_path):
    root = build_multi_shard(tmp_path / "shards", n_shards=1, rows_per_shard=4)
    args = build_parser().parse_args([
        "--shard-dir", str(root), "--layer", str(LAYER), "--K", "10",
        "--pca-rank", "2", "--out-dir", str(tmp_path / "bad"), "--device", "cpu",
    ])
    with pytest.raises(ValueError, match=r"rank\+1|empty clusters"):
        build_centroids(args)
    assert not args.out_dir.exists()


def test_pca_only_reuses_training_split_and_rejects_split_change(tmp_path):
    root = build_multi_shard(tmp_path / "shards", n_shards=2, rows_per_shard=8)
    run = resolve_run(_config(tmp_path, root))
    args = build_parser().parse_args(_initialization_command(run)[2:])
    args.pca_rank = 0
    build_centroids(args)
    centroids, _ = load_centroid_artifact(args.out_dir / "centroids.pt")
    args.pca_only = True
    args.pca_rank = 1
    build_centroids(args)
    updated, pcs = load_centroid_artifact(args.out_dir / "centroids.pt")
    assert torch.equal(centroids, updated)
    assert pcs.shape == (2, 2, 1)
    args.split_seed += 1
    with pytest.raises(ValueError, match="original centroid training split"):
        build_centroids(args)


def test_builder_default_and_prefix_override(tmp_path):
    root = build_multi_shard(tmp_path / "shards", n_shards=2, rows_per_shard=8)
    points, source = _load_activations(root, layer=LAYER, batch_size=7)
    assert len(points) == 48
    assert source["initialization_selection"]["train_rows"] == 16
    truncated, source = _load_activations(root, layer=LAYER, batch_size=7, drop_prefix=2)
    assert len(truncated) == 16
    assert source["initialization_selection"]["drop_prefix"] == 2


def test_pipeline_passes_trainer_prefix_default(tmp_path):
    root = build_multi_shard(tmp_path / "shards", n_shards=2, rows_per_shard=8)
    config_path = root / "config.json"
    source = json.loads(config_path.read_text())
    source.pop("drop_prefix")
    source["window"] = 33
    config_path.write_text(json.dumps(source))
    for path in (root / f"layer{LAYER:02d}").glob("*.pt"):
        x = torch.load(path, weights_only=True)[:, :1].expand(-1, 33, -1).clone()
        torch.save(x, path)
    run = resolve_run(_config(tmp_path, root))
    command = _initialization_command(run)
    assert command[command.index("--drop-prefix") + 1] == "32"


def test_pca_choice_sweep_and_default_identity(tmp_path):
    import yaml

    root = build_multi_shard(tmp_path / "shards", n_shards=2, rows_per_shard=16)
    config = _config(tmp_path, root)
    original = resolve_run(config)
    config["initialization"] = {"pca_method": "cluster"}
    assert resolve_run(config) == original
    config["sweep"] = {"initialization.pca_method": ["cluster", "knn"]}
    path = tmp_path / "sweep.yaml"
    path.write_text(yaml.safe_dump(config))
    cluster, knn = resolve_experiment(path)
    assert cluster == original
    assert cluster["run_dir"] != knn["run_dir"]
    assert knn["initialization"]["pca_neighbors"] == 64
    command = _initialization_command(knn)
    assert command[command.index("--pca-method") + 1] == "knn"
    assert command[command.index("--pca-neighbors") + 1] == "64"
    assert "--pca-method" not in _training_command(knn)
    config.pop("sweep")
    config["initialization"] = {"pca_method": "knn", "pca_neighbors": 8}
    assert resolve_run(config)["run_dir"] != knn["run_dir"]


@pytest.mark.parametrize("settings, message", [
    ({"pca_method": "invalid"}, "pca_method"),
    ({"pca_method": "knn", "pca_neighbors": 1}, "must exceed"),
    ({"pca_method": "knn", "pca_neighbors": 37}, "exceeds training"),
    ({"pca_method": "knn", "pca_neighbors": True}, "positive integer"),
    ({"pca_method": "knn", "pca_neighbors": 8.5}, "positive integer"),
    ({"pca_neighbors": 64}, "requires pca_method"),
    ({"unknown": True}, "unknown initialization"),
])
def test_invalid_pca_settings(tmp_path, settings, message):
    root = build_multi_shard(tmp_path / "shards", n_shards=2, rows_per_shard=8)
    config = _config(tmp_path, root)
    config["initialization"] = settings
    with pytest.raises(PipelineConfigError, match=message):
        resolve_run(config)


@pytest.mark.parametrize("bypass", ["centroids_path", "init_model_path", "em"])
def test_pca_settings_reject_bypassed_initialization(tmp_path, bypass):
    config = _config(tmp_path, tmp_path / "unused")
    config["initialization"] = {"pca_method": "knn"}
    config["model"]["kind"] = "hddc"
    if bypass == "em":
        config["training"]["fit_method"] = "em"
    else:
        config["training"][bypass] = str(tmp_path / "supplied.pt")
    with pytest.raises(PipelineConfigError, match="require automatic"):
        resolve_run(config, check_inputs=False)


def test_knn_initialization_accepts_small_clusters_and_checks_resume_method(tmp_path):
    root = build_multi_shard(tmp_path / "shards", n_shards=1, rows_per_shard=8)
    config = _config(tmp_path, root)
    config["model"].update(K=10, rank=2)
    config["initialization"] = {"pca_method": "knn", "pca_neighbors": 8}
    run = resolve_run(config)
    _ensure_initialization(run)
    directory = Path(run["run_dir"]) / "initialization"
    path = directory / "config.json"
    metadata = json.loads(path.read_text())
    assert min(metadata["cluster_sizes"]) <= 2
    assert pipeline_status([run])[0]["initialization"]
    centroids, pcs = load_centroid_artifact(directory / "centroids.pt")
    assert pcs.shape == (10, 2, 2)
    # Verify that only the PCA population changes, using identical KMeans settings.
    args = build_parser().parse_args(_initialization_command(run)[2:])
    args.pca_method, args.pca_rank = "cluster", 0
    args.out_dir = tmp_path / "means_only"
    build_centroids(args)
    assert torch.equal(centroids, load_centroid_artifact(args.out_dir / "centroids.pt")[0])
    for key, value in (("method", "cluster_covariance"), ("neighbors_per_centroid", 9)):
        changed = copy.deepcopy(metadata)
        changed["principal_components"][key] = value
        path.write_text(json.dumps(changed))
        original_bytes = (directory / "centroids.pt").read_bytes()
        with pytest.raises(RuntimeError, match="refusing to overwrite invalid initialization"):
            _ensure_initialization(run)
        assert (directory / "centroids.pt").read_bytes() == original_bytes


def test_pca_only_rejects_switching_method(tmp_path):
    root = build_multi_shard(tmp_path / "shards", n_shards=2, rows_per_shard=8)
    run = resolve_run(_config(tmp_path, root))
    args = build_parser().parse_args(_initialization_command(run)[2:])
    build_centroids(args)
    original = (args.out_dir / "centroids.pt").read_bytes()
    args.pca_only, args.pca_method, args.pca_neighbors = True, "knn", 8
    with pytest.raises(ValueError, match="different method"):
        build_centroids(args)
    assert (args.out_dir / "centroids.pt").read_bytes() == original
