"""Tests for the synthetic manifold-instance dataset generator."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from dalg.data import ToyManifoldConfig, make_toy_manifold_datasets
from dalg.data.manifold_dataset import (
    EMBEDDING_DIMS,
    INTRINSIC_DIMS,
    MANIFOLD_NAMES,
)
from dalg.models.mfa import MFA
from dalg.models.train import train_nll


def _tiny_config(**overrides) -> ToyManifoldConfig:
    config = ToyManifoldConfig(
        ambient_dim=16,
        n_train=80,
        n_val=40,
        calibration_size=256,
        seed=17,
    )
    return replace(config, **overrides)


def test_shapes_dtypes_labels_and_metadata() -> None:
    train, val, metadata = make_toy_manifold_datasets(
        _tiny_config(n_train=83, n_val=41)
    )
    x_train, y_train = train.tensors
    x_val, y_val = val.tensors

    assert isinstance(train, TensorDataset)
    assert isinstance(val, TensorDataset)
    assert x_train.shape == (83, 16)
    assert x_val.shape == (41, 16)
    assert x_train.dtype == torch.float32
    assert x_val.dtype == torch.float32
    assert y_train.dtype == torch.long
    assert y_val.dtype == torch.long
    assert metadata["num_manifolds"] == 64
    assert tuple(metadata["manifold_types"]) == MANIFOLD_NAMES
    assert tuple(metadata["intrinsic_dims"]) == INTRINSIC_DIMS
    assert tuple(metadata["embedding_dims"]) == EMBEDDING_DIMS
    assert metadata["type_id_to_name"] == dict(enumerate(MANIFOLD_NAMES))

    train_counts = torch.bincount(y_train, minlength=64)
    val_counts = torch.bincount(y_val, minlength=64)
    assert int(train_counts.max() - train_counts.min()) <= 1
    assert int(val_counts.max() - val_counts.min()) <= 1

    manifolds = metadata["manifolds"]
    assert len(manifolds) == 64
    assert [item["manifold_id"] for item in manifolds] == list(range(64))
    type_counts = torch.bincount(metadata["manifold_type_ids"], minlength=8)
    assert torch.equal(type_counts, torch.full((8,), 8))
    for item in manifolds:
        type_id = item["type_id"]
        assert item["type_name"] == MANIFOLD_NAMES[type_id]
        assert item["intrinsic_dim"] == INTRINSIC_DIMS[type_id]
        assert item["embedding_dim"] == EMBEDDING_DIMS[type_id]
        assert torch.equal(
            item["position"], metadata["offsets"][item["manifold_id"]]
        )


def test_generation_is_deterministic_and_seeded() -> None:
    config = _tiny_config()
    train_a, val_a, metadata_a = make_toy_manifold_datasets(config)
    train_b, val_b, metadata_b = make_toy_manifold_datasets(config)
    train_c, _, _ = make_toy_manifold_datasets(replace(config, seed=config.seed + 1))

    for tensors_a, tensors_b in (
        (train_a.tensors, train_b.tensors),
        (val_a.tensors, val_b.tensors),
    ):
        assert all(torch.equal(a, b) for a, b in zip(tensors_a, tensors_b))
    assert all(
        torch.equal(a, b)
        for a, b in zip(metadata_a["embeddings"], metadata_b["embeddings"])
    )
    assert not torch.equal(train_a.tensors[0], train_c.tensors[0])


def test_instances_have_independent_embeddings_and_offset_directions() -> None:
    _, _, metadata = make_toy_manifold_datasets(
        _tiny_config(offset_radius=2.0)
    )

    for manifold in metadata["manifolds"]:
        local_dim = manifold["embedding_dim"]
        embedding = manifold["embedding"]
        assert embedding.shape == (local_dim, 16)
        assert torch.allclose(
            embedding @ embedding.T,
            torch.eye(local_dim, dtype=embedding.dtype),
            atol=1e-10,
            rtol=0.0,
        )

    directions = metadata["offset_directions"]
    offsets = metadata["offsets"]
    assert directions.shape == (64, 16)
    assert torch.allclose(
        directions.norm(dim=1), torch.ones(64, dtype=directions.dtype)
    )
    assert torch.allclose(
        offsets.norm(dim=1),
        torch.full((64,), 2.0, dtype=offsets.dtype),
        atol=1e-10,
        rtol=0.0,
    )
    assert not torch.equal(metadata["embeddings"][0], metadata["embeddings"][1])
    assert not torch.equal(offsets[0], offsets[1])


def test_centered_manifolds_have_zero_mean_and_unit_rms() -> None:
    train, _, _ = make_toy_manifold_datasets(
        _tiny_config(
            n_train=128_000,
            n_val=80,
            calibration_size=30_000,
            offset_radius=0.0,
        )
    )
    x, manifold_ids = train.tensors

    for manifold_id in range(64):
        points = x[manifold_ids == manifold_id]
        assert points.mean(dim=0).norm() < 0.12
        rms = points.square().sum(dim=1).mean().sqrt()
        assert abs(float(rms) - 1.0) < 0.08


def test_offset_condition_differs_only_by_manifold_offset() -> None:
    centered_config = _tiny_config(
        n_train=160,
        n_val=80,
        calibration_size=1_000,
        offset_radius=0.0,
    )
    separated_config = replace(centered_config, offset_radius=2.0)
    centered_train, centered_val, centered_metadata = make_toy_manifold_datasets(
        centered_config
    )
    separated_train, separated_val, separated_metadata = make_toy_manifold_datasets(
        separated_config
    )

    assert torch.count_nonzero(centered_metadata["offsets"]) == 0
    for centered, separated in (
        (centered_train, separated_train),
        (centered_val, separated_val),
    ):
        x_centered, ids_centered = centered.tensors
        x_separated, ids_separated = separated.tensors
        assert torch.equal(ids_centered, ids_separated)
        expected = separated_metadata["offsets"].float()[ids_centered]
        assert torch.allclose(
            x_separated - x_centered, expected, atol=5e-7, rtol=1e-6
        )


def test_tensor_dataset_runs_through_train_nll() -> None:
    train, val, _ = make_toy_manifold_datasets(_tiny_config())
    x_train, _ = train.tensors
    x_val, _ = val.tensors
    loader = DataLoader(train, batch_size=16, shuffle=False)
    model = MFA(x_train[:4].clone(), rank=1, psi_init=0.5)

    train_nll(
        model,
        loader,
        val_tensor=x_val,
        epochs=1,
        steps_per_epoch=1,
        lr=1e-3,
        log_interval=1_000,
        track_best=False,
        early_stop_delta=None,
    )
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"ambient_dim": 2}, "ambient_dim"),
        ({"n_train": 0}, "n_train"),
        ({"n_val": 0}, "n_val"),
        ({"calibration_size": 1}, "calibration_size"),
        ({"manifolds_per_type": 0}, "manifolds_per_type"),
        ({"offset_radius": -1.0}, "offset_radius"),
        ({"segment_min": 1.0, "segment_max": 1.0}, "segment"),
        ({"torus_major_radius": 1.0, "torus_minor_radius": 1.0}, "torus"),
        ({"mobius_half_width": 1.0}, "mobius"),
        ({"swiss_height_min": 3.0, "swiss_height_max": 2.0}, "swiss"),
        ({"helix_alpha": 0.0}, "helix_alpha"),
    ],
)
def test_invalid_configs_raise_clear_errors(overrides, message) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        make_toy_manifold_datasets(_tiny_config(**overrides))


def test_non_finite_geometry_is_rejected() -> None:
    with pytest.raises(ValueError, match="finite"):
        make_toy_manifold_datasets(_tiny_config(swiss_theta_max=float("inf")))
