from __future__ import annotations

import importlib
import json
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from dalg.init.mixture_weights import initialize_mixture_weights
from dalg.models.adaptive_q.mfa_ard import MFA_ARD
from dalg.models.adaptive_q.mfa_hddc import (
    ComponentShardedMFA_HDDC,
    MFA_HDDC,
    load_mfa_hddc,
    save_mfa_hddc,
)
from dalg.models.mfa import ComponentShardedMFA, MFA, load_mfa
from tests.synthetic_shards import LAYER, build_multi_shard


@pytest.mark.parametrize("model_cls", [MFA, MFA_ARD, MFA_HDDC])
def test_weights_match_unequal_cluster_proportions(model_cls):
    centers = torch.tensor([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
    points = centers.repeat_interleave(torch.tensor([1, 3, 6]), dim=0)
    model = model_cls(centers, rank=1)
    before = {name: value.clone() for name, value in model.state_dict().items()}

    counts = initialize_mixture_weights(
        model, centers, points.split(4), n_train_tokens=10,
    )

    torch.testing.assert_close(counts, torch.tensor([1, 3, 6]))
    torch.testing.assert_close(model.pi_logits.softmax(0), torch.tensor([0.1, 0.3, 0.6]))
    for name, value in model.state_dict().items():
        if name != "pi_logits":
            assert torch.equal(value, before[name])
    model.nll(points[:1]).backward()
    assert torch.isfinite(model.pi_logits.grad).all()
    assert model.pi_logits.grad.abs().sum() > 0


@pytest.mark.parametrize("model_cls", [MFA, MFA_ARD, MFA_HDDC])
def test_empty_cluster_gets_one_pseudocount_and_can_receive_gradients(model_cls, capsys):
    centers = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    points = centers.repeat_interleave(torch.tensor([0, 3, 6]), dim=0)
    model = model_cls(centers, rank=1)
    before = {name: value.clone() for name, value in model.state_dict().items()}

    counts = initialize_mixture_weights(
        model, centers, points.split(4), n_train_tokens=9,
    )

    torch.testing.assert_close(counts, torch.tensor([0, 3, 6]))
    torch.testing.assert_close(model.pi_logits.softmax(0), torch.tensor([0.1, 0.3, 0.6]))
    assert torch.isfinite(model.pi_logits).all()
    for name, value in model.state_dict().items():
        if name != "pi_logits":
            assert torch.equal(value, before[name])
    output = capsys.readouterr().out
    assert "giving 1 empty training clusters one pseudocount each" in output
    assert "empty cluster ids (first 10): [0]" in output

    model.nll(points).backward()
    assert torch.isfinite(model.pi_logits.grad).all()
    assert model.pi_logits.grad[0].abs() > 0
    assert torch.isfinite(model.mu.grad).all()
    assert model.mu.grad[0].abs().sum() > 0


def test_empty_training_stream_is_rejected_without_changing_logits():
    centers = torch.tensor([[0.0, 0.0], [10.0, 0.0]])
    model = MFA(centers, rank=1)
    with pytest.raises(ValueError, match="positive training-token count"):
        initialize_mixture_weights(model, centers, [], n_train_tokens=0)
    assert torch.equal(model.pi_logits, torch.zeros(2))


def test_truncated_training_stream_is_rejected():
    centers = torch.tensor([[0.0, 0.0], [10.0, 0.0]])
    model = MFA(centers, rank=1)
    with pytest.raises(ValueError, match="counted 2 training tokens, expected 4"):
        initialize_mixture_weights(model, centers, [centers], n_train_tokens=4)


def _cli_fixture(tmp_path, kind, direction_init="random"):
    modules = {
        "mfa": ("dalg.cli.run_training", "cmd_train"),
        "ard": ("dalg.cli.adaptive_q.run_training_ard", "cmd_train_ard"),
        "hddc": ("dalg.cli.adaptive_q.run_training_hddc", "cmd_train_single_process"),
    }
    module_name, command_name = modules[kind]
    cli = importlib.import_module(module_name)
    shard_dir = build_multi_shard(tmp_path / "shards", n_shards=2, rows_per_shard=4)
    config_path = shard_dir / "config.json"
    config = json.loads(config_path.read_text())
    config["drop_prefix"] = 1
    config_path.write_text(json.dumps(config))
    centers = torch.tensor([[10.0, 11.0], [210.0, 211.0]])
    centroid_path = tmp_path / "centroids.pt"
    torch.save(
        {"centroids": centers, "principal_components": torch.eye(2)[:, :1].repeat(2, 1, 1)},
        centroid_path,
    )
    args = cli.build_parser().parse_args([
        "--device", "cpu", "--shard-dir", str(shard_dir), "--layer", str(LAYER),
        "--out-dir", str(tmp_path / "run"), "--centroids-path", str(centroid_path),
        "--direction-init", direction_init, "--K", "2", "--rank", "1",
        "--epochs", "1", "--batch-size", "3", "--val-frac", "0.25",
        "--split-seed", "42", "--seed", "3", "--lr", "0",
        "--max-steps", "1", "--steps-per-epoch", "1", "--epoch-snapshot-every", "0",
    ])
    if kind == "ard":
        args.ard_lambda = 0.0
        args.prune_at_end = False
    return cli, getattr(cli, command_name), args, centers


def _forbid_initialization(*_args, **_kwargs):
    raise AssertionError("must preserve saved mixture weights")


@pytest.mark.parametrize("kind", ["mfa", "ard", "hddc"])
@pytest.mark.parametrize("direction_init", ["random", "cluster_pca"])
def test_cli_counts_full_training_split_and_preserves_weights_on_resume(
    tmp_path, monkeypatch, kind, direction_init,
):
    cli, command, args, centers = _cli_fixture(tmp_path, kind, direction_init)
    data = cli._resolve_activation_data(args, log=lambda *_: None)
    # Independently select training rows and retained tokens from the raw tensors.
    activations = torch.cat([
        torch.load(path, weights_only=True)
        for path in sorted((Path(args.shard_dir) / f"layer{LAYER:02d}").glob("*.pt"))
    ])
    points = activations[data["train_pos_full"], 1:].reshape(-1, 2)
    expected = torch.bincount(torch.cdist(points, centers).argmin(1), minlength=2).float()
    expected /= expected.sum()
    assert expected.min() > 0
    assert not torch.allclose(expected, torch.full((2,), 0.5))

    command(args)
    load = load_mfa_hddc if kind == "hddc" else load_mfa
    saved = Path(args.out_dir) / "mfa_model.pt"
    model = load(saved, map_location="cpu")
    torch.testing.assert_close(model.pi_logits.softmax(0), expected)

    monkeypatch.setattr(cli, "initialize_mixture_weights", _forbid_initialization)
    args.epochs = 2
    args.max_steps = 2
    command(args)
    torch.testing.assert_close(load(saved, map_location="cpu").pi_logits.softmax(0), expected)


def test_hddc_warm_start_preserves_supplied_weights(tmp_path, monkeypatch):
    cli, command, args, centers = _cli_fixture(tmp_path, "hddc")
    initial = MFA_HDDC(centers, rank=1)
    with torch.no_grad():
        initial.pi_logits.copy_(torch.tensor([0.8, 0.2]).log())
    initial_path = tmp_path / "initial.pt"
    save_mfa_hddc(initial, str(initial_path))
    args.centroids_path = None
    args.init_model_path = str(initial_path)
    monkeypatch.setattr(cli, "initialize_mixture_weights", _forbid_initialization)

    command(args)
    trained = load_mfa_hddc(Path(args.out_dir) / "mfa_model.pt", map_location="cpu")
    torch.testing.assert_close(trained.pi_logits.softmax(0), torch.tensor([0.8, 0.2]))


def _distributed_weights_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=2,
        timeout=timedelta(seconds=45),
    )
    try:
        centers = torch.stack([torch.arange(5) * 10.0, torch.zeros(5)], dim=1)
        for expected_counts in (torch.tensor([1, 2, 3, 4, 5]), torch.tensor([1, 2, 3, 0, 0])):
            points = centers.repeat_interleave(expected_counts, dim=0)
            for model_cls in (ComponentShardedMFA, ComponentShardedMFA_HDDC):
                model = model_cls.from_global_centroids(
                    centers, rank=1, dist_rank=rank, world_size=2,
                )
                # Only rank zero has data. The other rank must receive global counts.
                batches = points.split(4) if rank == 0 else []
                counts = initialize_mixture_weights(
                    model, centers, batches, n_train_tokens=len(points),
                )
                torch.testing.assert_close(counts, expected_counts)
                expected_weights = expected_counts.clamp_min(1).float()
                expected_weights /= expected_weights.sum()
                expected = expected_weights[model.component_start:model.component_end]
                torch.testing.assert_close(model.local_log_pi().exp(), expected)
    finally:
        dist.destroy_process_group()


def test_component_shards_use_global_proportions(tmp_path):
    mp.spawn(
        _distributed_weights_worker,
        args=((tmp_path / "rendezvous").as_uri(),), nprocs=2, join=True,
    )
