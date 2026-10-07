from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from dalg.cli import run_training, run_training_kmeans
from dalg.cli.adaptive_q import run_training_ard, run_training_hddc
from dalg.data.shard_activations import load_meta_index, stratified_split
from dalg.init.kmeans_model import load_kmeans_initialization
from dalg.models.kmeans import KMeans, load_kmeans, save_kmeans


@pytest.fixture(autouse=True)
def small_thread_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def _shards(path: Path):
    path.mkdir()
    (path / "layer00").mkdir()
    (path / "meta").mkdir()
    gen = torch.Generator().manual_seed(12)
    points = torch.randn(40, 2, 3, generator=gen) * torch.tensor([3.0, 1.0, .1])
    points[:, 0] = 1e5  # Dropped prefix tokens must never enter fitting or PCA.
    config = {"window": 2, "d_model": 3, "drop_prefix": 1, "layers": [0]}
    (path / "config.json").write_text(json.dumps(config))
    (path / "meta" / "shard_00000.json").write_text(json.dumps({
        "row_indices": list(range(40)), "rows": [{"subset": "test"}] * 40,
    }))
    torch.save(points, path / "layer00" / "shard_00000.pt")
    meta = load_meta_index(path, layer=0)
    train, val = stratified_split(meta, val_frac=.2, seed=42)
    points[val, 1] += 10000  # A leaked validation row would dominate the mean.
    torch.save(points, path / "layer00" / "shard_00000.pt")
    return points[train, 1], val


def _args(shards, output, *extra):
    return run_training_kmeans.build_parser().parse_args([
        "--shard-dir", str(shards), "--layer", "0", "--K", "1",
        "--out-dir", str(output), "--device", "cpu",
        "--max-iter", "3", "--restarts", "1", "--val-frac", ".2", *extra,
    ])


@pytest.mark.parametrize("method", ["cluster", "knn"])
def test_cli_uses_training_points_for_means_and_pca(tmp_path, method):
    shards, output = tmp_path / "shards", tmp_path / "model"
    training, val = _shards(shards)
    args = _args(shards, output, *(["--pca-purpose", "initialization", "--rank", "2", "--pca-neighbors", "12"] if method == "knn" else []))
    run_training_kmeans.cmd_train(args)
    model = load_kmeans(output / "kmeans_model.pt")
    torch.testing.assert_close(model.mu[0], training.mean(0))
    population = training
    if method == "knn":
        order = (training - model.mu[0]).square().sum(1).argsort()[:12]
        population = training[order]
    centered = population.double() - model.mu[0].double()
    values, directions = torch.linalg.eigh(centered.T @ centered / len(population))
    if method == "cluster":
        torch.testing.assert_close(model.eigenvalues[0], values.flip(0))
    basis = model.W if method == "cluster" else model.W_init
    rank = int(model.component_ranks[0]) if method == "cluster" else 2
    target = directions[:, -rank:]
    torch.testing.assert_close(basis[0].double() @ basis[0].double().T, target @ target.T, atol=1e-6, rtol=1e-6)
    assert model.checkpoint_extra["selection"]["train_activations"] == len(training)
    assert model.checkpoint_extra["cluster_sizes"] == [len(training)]
    split = json.loads((output / "val_indices.json").read_text())
    assert split["val_global_rows"] == val
    assert set(p.name for p in output.iterdir()) == {"kmeans_model.pt", "config.json", "val_indices.json"}


def test_pca_only_preserves_means_and_rejects_a_different_split(tmp_path):
    shards, output = tmp_path / "shards", tmp_path / "model"
    _shards(shards)
    args = _args(shards, output, "--rank", "0")
    run_training_kmeans.cmd_train(args)
    before = load_kmeans(output / "kmeans_model.pt")
    assert before.q == 0
    args = _args(shards, output, "--pca-only", "--surgery-threshold", ".2")
    run_training_kmeans.cmd_train(args)
    after = load_kmeans(output / "kmeans_model.pt")
    torch.testing.assert_close(after.mu, before.mu, rtol=0, atol=0)
    assert after.q == 1
    assert after.W.shape == (1, 3, 1)
    assert after.eigenvalues.shape == (1, 3)
    assert after.component_ranks.tolist() == [1]
    args.split_seed = 43
    with pytest.raises(ValueError, match="training split|original training population"):
        run_training_kmeans.cmd_train(args)
    unchanged = load_kmeans(output / "kmeans_model.pt")
    torch.testing.assert_close(unchanged.W, after.W, rtol=0, atol=0)


@pytest.mark.parametrize("trainer", [run_training, run_training_ard, run_training_hddc])
def test_trainers_read_model_means_and_pcs_without_export(tmp_path, trainer):
    points = torch.randn(20, 3, generator=torch.Generator().manual_seed(3))
    model = KMeans.from_centroids(points.mean(0, keepdim=True))
    model.compute_init_pcs(points, rank=2, neighbors=12)
    model.compute_pcs(points)
    path = tmp_path / "initializer.pt"
    save_kmeans(model, path)
    args = trainer.build_parser().parse_args([
        "--shard-dir", "unused", "--layer", "0", "--K", "1", "--rank", "2",
        "--direction-init", "cluster_pca", "--kmeans-model-path", str(path),
        "--device", "cpu",
    ])
    trainer.validate_args(args)
    output = tmp_path / "training"
    output.mkdir()
    kwargs = {"out_dir": output, "device": "cpu"}
    if trainer is not run_training_ard:
        kwargs.update(is_main=True, barrier=False)
    means, directions = trainer._ensure_centroids({"d_model": 3}, args, **kwargs)
    torch.testing.assert_close(means, model.mu, atol=0, rtol=0)
    torch.testing.assert_close(directions, model.W_init, atol=0, rtol=0)
    assert not list(output.iterdir())
    args.centroids_path = "legacy.pt"
    with pytest.raises(SystemExit, match="set only one"):
        trainer.validate_args(args)


def test_loader_rejects_missing_pcs_wrong_shape_and_legacy_bundle(tmp_path):
    path = tmp_path / "kmeans_model.pt"
    save_kmeans(KMeans.from_centroids(torch.zeros(2, 3)), path)
    with pytest.raises(ValueError, match="stores 0 initialization PCs"):
        load_kmeans_initialization(path, expected_k=2, expected_d=3, rank=1)
    with pytest.raises(ValueError, match="expected"):
        load_kmeans_initialization(path, expected_k=3, expected_d=3)
    torch.save(torch.zeros(2, 3), path)
    with pytest.raises(ValueError, match="legacy centroid"):
        load_kmeans_initialization(path, expected_k=2, expected_d=3)


def test_reservoir_fallback_saves_a_model_instead_of_centroids(tmp_path, monkeypatch):
    from dalg.init import projected_knn

    shards, output = tmp_path / "shards", tmp_path / "training"
    training, _ = _shards(shards)
    args = run_training.build_parser().parse_args([
        "--shard-dir", str(shards), "--layer", "0", "--K", "1", "--rank", "2",
        "--out-dir", str(output), "--device", "cpu", "--val-frac", ".2",
    ])
    data = run_training._resolve_activation_data(args, log=lambda _: None)
    output.mkdir()
    seen = {}

    class Reservoir:
        def __init__(self, **options):
            seen.update(options)

        def fit(self, loader, **options):
            points = torch.cat(list(loader))
            torch.testing.assert_close(points.sort(0).values, training.sort(0).values)
            return points.mean(0, keepdim=True)

    monkeypatch.setattr(projected_knn, "ReservoirKMeans", Reservoir)
    means, directions = run_training._ensure_centroids(
        data, args, out_dir=output, is_main=True, device="cpu", barrier=False,
    )
    checkpoint = output / "initialization" / "kmeans_model.pt"
    assert checkpoint.exists()
    assert not (output / "centroids.pt").exists()
    assert directions is None
    assert seen["proj_dim"] == args.proj_dim
    restored = load_kmeans(checkpoint)
    torch.testing.assert_close(means, restored.mu, rtol=0, atol=0)
    assert restored.checkpoint_extra["method"] == "reservoir_kmeans"


@pytest.mark.parametrize("flags", [
    ["--sample-fraction", "0"], ["--rank", "-1"], ["--device", "mps"],
    ["--rank", "0", "--surgery-threshold", ".1"],
    ["--pca-purpose", "initialization", "--rank", "2", "--pca-neighbors", "2"],
    ["--pca-purpose", "initialization", "--surgery-threshold", ".1"],
    ["--pca-neighbors", "12"], ["--rank", "2"], ["--pca-purpose", "initialization"],
])
def test_cli_rejects_incompatible_settings(tmp_path, flags):
    with pytest.raises(ValueError):
        run_training_kmeans.validate_args(_args(tmp_path, tmp_path / "out", *flags))


@pytest.mark.parametrize("field,value", [("seed", 4), ("block_x", 32), ("sample_seed", 5)])
def test_pca_only_rejects_changed_original_settings(tmp_path, field, value):
    shards, output = tmp_path / "shards", tmp_path / "model"
    _shards(shards)
    args = _args(shards, output, "--rank", "0")
    run_training_kmeans.cmd_train(args)
    before = (output / "kmeans_model.pt").read_bytes()
    args = _args(shards, output, "--pca-only")
    setattr(args, field, value)
    with pytest.raises(ValueError, match="original"):
        run_training_kmeans.cmd_train(args)
    assert (output / "kmeans_model.pt").read_bytes() == before


def test_pca_only_rejects_conflicting_metadata(tmp_path):
    shards, output = tmp_path / "shards", tmp_path / "model"
    _shards(shards)
    run_training_kmeans.cmd_train(_args(shards, output, "--rank", "0"))
    config_path = output / "config.json"
    metadata = json.loads(config_path.read_text())
    metadata["rows_used"] += 1
    config_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="disagrees"):
        run_training_kmeans.cmd_train(_args(shards, output, "--pca-only"))
    assert load_kmeans(output / "kmeans_model.pt").q == 0


def test_hddc_em_reservoir_and_warm_start_do_not_export_centroids(tmp_path, monkeypatch):
    from dalg.init.projected_knn import ReservoirKMeans

    shards, output = tmp_path / "shards", tmp_path / "em"
    _shards(shards)
    flags = [
        "--shard-dir", str(shards), "--layer", "0", "--K", "1", "--rank", "2",
        "--device", "cpu", "--val-frac", ".2", "--fit-method", "em", "--shared-b",
        "--epochs", "1", "--batch-size", "8", "--proj-dim", "2", "--pool-size", "24",
        "--refine-epochs", "1", "--seed", "4",
    ]
    args = run_training_hddc.build_parser().parse_args([*flags, "--out-dir", str(output)])
    run_training_hddc.validate_args(args)
    run_training_hddc.cmd_train_single_process(args)
    initializer = load_kmeans(output / "initialization" / "kmeans_model.pt")
    assert initializer.q == 0
    assert initializer.checkpoint_extra["method"] == "reservoir_kmeans"
    assert initializer.checkpoint_extra["selection"]["train_activations"] == 32
    assert not list(output.rglob("centroids.pt"))
    initial_checkpoint = (output / "initialization" / "kmeans_model.pt").read_bytes()
    monkeypatch.setattr(ReservoirKMeans, "fit", lambda *a, **kw: pytest.fail("refitted cached initializer"))
    run_training_hddc.cmd_train_single_process(args)
    assert (output / "initialization" / "kmeans_model.pt").read_bytes() == initial_checkpoint
    warm = tmp_path / "warm"
    args = run_training_hddc.build_parser().parse_args([
        *flags, "--out-dir", str(warm), "--init-model-path", str(output / "mfa_model.pt"),
    ])
    run_training_hddc.validate_args(args)
    run_training_hddc.cmd_train_single_process(args)
    assert (warm / "mfa_model.pt").exists()
    assert not list(warm.rglob("centroids.pt"))
    assert not list(warm.rglob("kmeans_model.pt"))


def test_pca_only_adds_each_state_without_changing_the_other(tmp_path):
    shards, output = tmp_path / 'shards', tmp_path / 'model'
    _shards(shards)
    run_training_kmeans.cmd_train(_args(shards, output, '--pca-purpose', 'initialization', '--rank', '2', '--pca-neighbors', '12'))
    first = load_kmeans(output / 'kmeans_model.pt')
    assert first.q == 0 and first.q_init == 2
    run_training_kmeans.cmd_train(_args(shards, output, '--pca-only', '--surgery-threshold', '.2'))
    second = load_kmeans(output / 'kmeans_model.pt')
    torch.testing.assert_close(second.W_init, first.W_init, rtol=0, atol=0)
    assert second.checkpoint_extra['initialization_components'] == first.checkpoint_extra['initialization_components']
    run_training_kmeans.cmd_train(_args(shards, output, '--pca-only', '--pca-purpose', 'initialization', '--rank', '2', '--pca-neighbors', '16'))
    third = load_kmeans(output / 'kmeans_model.pt')
    torch.testing.assert_close(third.W, second.W, rtol=0, atol=0)
    torch.testing.assert_close(third.component_ranks, second.component_ranks)
    assert third.surgery_threshold == second.surgery_threshold
    assert third.checkpoint_extra['principal_components'] == second.checkpoint_extra['principal_components']


def test_geometry_checkpoint_cannot_supply_initialization_directions(tmp_path):
    points = torch.randn(20, 3, generator=torch.Generator().manual_seed(3))
    model = KMeans.from_centroids(points.mean(0, keepdim=True)).compute_pcs(points)
    path = tmp_path / 'geometry.pt'
    save_kmeans(model, path)
    with pytest.raises(ValueError, match='initialization PCs'):
        load_kmeans_initialization(path, expected_k=1, expected_d=3, rank=1)
    means, directions = load_kmeans_initialization(path, expected_k=1, expected_d=3)
    torch.testing.assert_close(means, model.mu)
    assert directions is None
