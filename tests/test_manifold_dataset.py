"""Tests for the synthetic manifold-instance dataset generator."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from dalg.data import (
    ToyManifoldConfig,
    make_toy_manifold_dataset,
    save_toy_manifold_shards,
)
from dalg.data.shard_activations import ActivationBatchDataset, load_meta_index
from dalg.data.manifold_dataset import (
    EMBEDDING_DIMS,
    INTRINSIC_DIMS,
    MANIFOLD_NAMES,
    _generator,
    _sample_hypersphere_10d,
    _sample_product_torus_12d,
)
from dalg.models.mfa import MFA
from dalg.models.train import train_nll


def _tiny_config(**overrides) -> ToyManifoldConfig:
    config = ToyManifoldConfig(
        ambient_dim=32,
        n_samples=120,
        calibration_size=256,
        seed=17,
    )
    return replace(config, **overrides)


def test_shapes_dtypes_labels_and_metadata() -> None:
    dataset, metadata = make_toy_manifold_dataset(_tiny_config(n_samples=124))
    points, manifold_ids = dataset.tensors

    assert isinstance(dataset, TensorDataset)
    assert points.shape == (124, 32)
    assert points.dtype == torch.float32
    assert manifold_ids.dtype == torch.long
    assert metadata["num_manifolds"] == 80
    assert tuple(metadata["manifold_types"]) == MANIFOLD_NAMES
    assert tuple(metadata["intrinsic_dims"]) == INTRINSIC_DIMS
    assert tuple(metadata["embedding_dims"]) == EMBEDDING_DIMS
    assert metadata["type_id_to_name"] == dict(enumerate(MANIFOLD_NAMES))
    assert metadata["curvature_definition"] == (
        "maximum absolute extrinsic principal curvature"
    )
    assert metadata["flat_radius_convention"] == "unit RMS radius"
    assert metadata["max_abs_curvatures"].shape == (10,)
    assert metadata["curvature_radii"].shape == (10,)
    assert metadata["noise_stds"].shape == (10,)
    assert torch.equal(
        metadata["max_abs_curvatures"][[0, 2]],
        torch.zeros(2, dtype=torch.float64),
    )
    assert torch.all(metadata["curvature_radii"] > 0)
    assert torch.all(metadata["noise_stds"] > 0)
    assert torch.allclose(
        metadata["curvature_radii"] / metadata["noise_stds"],
        torch.full((10,), 10_000.0, dtype=torch.float64),
    )

    counts = torch.bincount(manifold_ids, minlength=80)
    assert int(counts.max() - counts.min()) <= 1

    manifolds = metadata["manifolds"]
    assert len(manifolds) == 80
    assert [item["manifold_id"] for item in manifolds] == list(range(80))
    type_counts = torch.bincount(metadata["manifold_type_ids"], minlength=10)
    assert torch.equal(type_counts, torch.full((10,), 8))
    for item in manifolds:
        type_id = item["type_id"]
        assert item["type_name"] == MANIFOLD_NAMES[type_id]
        assert item["intrinsic_dim"] == INTRINSIC_DIMS[type_id]
        assert item["embedding_dim"] == EMBEDDING_DIMS[type_id]
        assert item["max_abs_curvature"] == metadata["max_abs_curvatures"][type_id]
        assert item["curvature_radius"] == metadata["curvature_radii"][type_id]
        assert item["noise_std"] == metadata["noise_stds"][type_id]
        assert torch.equal(
            item["position"], metadata["offsets"][item["manifold_id"]]
        )


def test_selected_manifold_types() -> None:
    config = _tiny_config(
        n_samples=30,
        manifolds_per_type=1,
        manifold_types=("circle", "helix"),
    )
    dataset, metadata = make_toy_manifold_dataset(config)

    assert metadata["manifold_types"] == ("circle", "helix")
    assert metadata["intrinsic_dims"] == (1, 1)
    assert metadata["embedding_dims"] == (2, 3)
    assert metadata["num_manifolds"] == 2
    assert torch.equal(metadata["manifold_type_ids"], torch.tensor([0, 1]))
    assert torch.bincount(dataset.tensors[1], minlength=2).tolist() == [15, 15]
    assert [item["type_name"] for item in metadata["manifolds"]] == [
        "circle",
        "helix",
    ]


def test_high_dimensional_raw_samples_satisfy_manifold_constraints() -> None:
    config = _tiny_config()
    hypersphere = _sample_hypersphere_10d(128, _generator(config.seed, 1), config)
    product_torus = _sample_product_torus_12d(
        128,
        _generator(config.seed, 2),
        config,
    )

    assert hypersphere.shape == (128, 11)
    assert torch.allclose(
        hypersphere.norm(dim=1),
        torch.ones(128, dtype=torch.float64),
        atol=1e-12,
        rtol=0.0,
    )
    assert product_torus.shape == (128, 24)
    assert torch.allclose(
        product_torus.reshape(128, 12, 2).norm(dim=2),
        torch.ones((128, 12), dtype=torch.float64),
        atol=1e-12,
        rtol=0.0,
    )


def test_generation_is_deterministic_and_seeded() -> None:
    config = _tiny_config()
    dataset_a, metadata_a = make_toy_manifold_dataset(config)
    dataset_b, metadata_b = make_toy_manifold_dataset(config)
    dataset_c, _ = make_toy_manifold_dataset(replace(config, seed=config.seed + 1))

    assert all(
        torch.equal(a, b) for a, b in zip(dataset_a.tensors, dataset_b.tensors)
    )
    assert all(
        torch.equal(a, b)
        for a, b in zip(metadata_a["embeddings"], metadata_b["embeddings"])
    )
    assert not torch.equal(dataset_a.tensors[0], dataset_c.tensors[0])


def test_raw_curvatures_match_manifold_geometry() -> None:
    config = _tiny_config(
        torus_major_radius=3.0,
        torus_minor_radius=1.0,
        swiss_theta_min=2.0,
        swiss_theta_max=5.0,
        helix_alpha=0.5,
    )
    _, metadata = make_toy_manifold_dataset(config)
    curvatures = metadata["raw_max_abs_curvatures"]

    assert torch.equal(curvatures[[0, 2]], torch.zeros(2, dtype=torch.float64))
    assert curvatures[1] == 1.0
    assert curvatures[3] == 1.0
    assert curvatures[4] == 1.0
    assert curvatures[5] > 0.0
    assert float(curvatures[6]) == pytest.approx(6.0 / 5.0**1.5)
    assert float(curvatures[7]) == pytest.approx(0.8)
    assert torch.equal(curvatures[8:], torch.ones(2, dtype=torch.float64))
    assert torch.allclose(
        metadata["max_abs_curvatures"],
        curvatures * metadata["calibration_scales"],
    )


def test_instances_have_independent_embeddings_and_offset_directions() -> None:
    _, metadata = make_toy_manifold_dataset(_tiny_config(offset_radius=2.0))

    for manifold in metadata["manifolds"]:
        local_dim = manifold["embedding_dim"]
        embedding = manifold["embedding"]
        assert embedding.shape == (local_dim, 32)
        assert torch.allclose(
            embedding @ embedding.T,
            torch.eye(local_dim, dtype=embedding.dtype),
            atol=1e-10,
            rtol=0.0,
        )

    directions = metadata["offset_directions"]
    offsets = metadata["offsets"]
    assert directions.shape == (80, 32)
    assert torch.allclose(
        directions.norm(dim=1), torch.ones(80, dtype=directions.dtype)
    )
    assert torch.allclose(
        offsets.norm(dim=1),
        torch.full((80,), 2.0, dtype=offsets.dtype),
        atol=1e-10,
        rtol=0.0,
    )
    assert not torch.equal(metadata["embeddings"][0], metadata["embeddings"][1])
    assert not torch.equal(offsets[0], offsets[1])


def test_centered_manifolds_have_zero_mean_and_unit_rms() -> None:
    dataset, _ = make_toy_manifold_dataset(
        _tiny_config(
            n_samples=128_000,
            calibration_size=30_000,
            offset_radius=0.0,
        )
    )
    x, manifold_ids = dataset.tensors

    for manifold_id in range(80):
        points = x[manifold_ids == manifold_id]
        assert points.mean(dim=0).norm() < 0.12
        rms = points.square().sum(dim=1).mean().sqrt()
        assert abs(float(rms) - 1.0) < 0.08


def test_ambient_noise_matches_curvature_scaled_standard_deviation() -> None:
    dataset, metadata = make_toy_manifold_dataset(
        _tiny_config(
            n_samples=6_400,
            calibration_size=2_000,
            offset_radius=2.0,
            noise_ratio=1_000.0,
        )
    )
    points, manifold_ids = dataset.tensors
    normalized_normal_energy = []

    for manifold in metadata["manifolds"]:
        manifold_id = manifold["manifold_id"]
        local_points = points[manifold_ids == manifold_id].double()
        centered = local_points - manifold["position"]
        embedding = manifold["embedding"]
        tangent_projection = (centered @ embedding.T) @ embedding
        normal_noise = centered - tangent_projection
        normal_dim = points.shape[1] - manifold["embedding_dim"]
        normalized_normal_energy.append(
            normal_noise.square().sum()
            / (len(local_points) * normal_dim * manifold["noise_std"].square())
        )

    observed = torch.stack(normalized_normal_energy)
    assert torch.allclose(
        observed.mean(), torch.tensor(1.0, dtype=torch.float64), atol=0.04, rtol=0.0
    )
    assert torch.allclose(
        metadata["curvature_radii"] / metadata["noise_stds"],
        torch.full((10,), 1_000.0, dtype=torch.float64),
    )


def test_offset_condition_differs_only_by_manifold_offset() -> None:
    centered_config = _tiny_config(
        n_samples=240,
        calibration_size=1_000,
        offset_radius=0.0,
    )
    separated_config = replace(centered_config, offset_radius=2.0)
    centered, centered_metadata = make_toy_manifold_dataset(centered_config)
    separated, separated_metadata = make_toy_manifold_dataset(separated_config)

    assert torch.count_nonzero(centered_metadata["offsets"]) == 0
    x_centered, ids_centered = centered.tensors
    x_separated, ids_separated = separated.tensors
    assert torch.equal(ids_centered, ids_separated)
    expected = separated_metadata["offsets"].float()[ids_centered]
    assert torch.allclose(
        x_separated - x_centered, expected, atol=5e-7, rtol=1e-6
    )


def test_tensor_dataset_runs_through_train_nll() -> None:
    dataset, _ = make_toy_manifold_dataset(_tiny_config())
    x, y = dataset.tensors
    x_train, x_val = x[:-16], x[-16:]
    loader = DataLoader(
        TensorDataset(x_train, y[:-16]), batch_size=16, shuffle=False
    )
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


def test_shard_writer_matches_activation_training_protocol(tmp_path) -> None:
    config = _tiny_config(n_samples=124)
    expected_dataset, expected_metadata = make_toy_manifold_dataset(config)
    root = save_toy_manifold_shards(
        tmp_path / "toy_manifold_shards",
        config,
        shard_size=25,
        layer=0,
    )

    shard_config = json.loads((root / "config.json").read_text())
    assert shard_config["source_kind"] == "toy_manifolds"
    assert shard_config["layers"] == [0]
    assert shard_config["window"] == 1
    assert shard_config["drop_prefix"] == 0
    assert shard_config["d_model"] == 32
    assert shard_config["dtype"] == "float32"
    assert shard_config["num_rows"] == 124
    assert shard_config["num_shards"] == 5
    assert not (root / "tokens").exists()

    shard_paths = sorted((root / "layer00").glob("shard_*.pt"))
    assert len(shard_paths) == shard_config["num_shards"]
    assert all(
        torch.load(path, mmap=True, weights_only=True).ndim == 3
        for path in shard_paths
    )
    assert all(
        torch.load(path, mmap=True, weights_only=True).shape[1:] == (1, 32)
        for path in shard_paths
    )
    for path in shard_paths:
        shard = torch.load(path, mmap=True, weights_only=True)
        assert shard.untyped_storage().nbytes() == shard.numel() * shard.element_size()

    meta_index = load_meta_index(root, layer=0)
    assert len(meta_index) == 124
    assert [row["global_row"] for row in meta_index] == list(range(124))
    assert {row["subset"] for row in meta_index} == set(MANIFOLD_NAMES)

    first_meta = json.loads((root / "meta" / "shard_00000.json").read_text())
    assert set(first_meta["rows"][0]) == {
        "subset",
        "manifold_id",
        "manifold_type_id",
        "intrinsic_dim",
    }

    dataset = ActivationBatchDataset(
        root,
        layer=0,
        batch_size=17,
        drop_prefix=None,
        shuffle_shards=False,
        shuffle_within_shard=False,
    )
    streamed = torch.cat(list(dataset))
    assert dataset.num_items == 124
    assert torch.equal(streamed, expected_dataset.tensors[0])

    saved_metadata = torch.load(
        root / "manifold_metadata.pt", map_location="cpu", weights_only=True
    )
    assert torch.equal(
        saved_metadata["row_manifold_ids"], expected_dataset.tensors[1]
    )
    assert saved_metadata["canonical_order"] == "generated"
    assert torch.equal(
        saved_metadata["manifold_type_ids"],
        expected_metadata["manifold_type_ids"],
    )


def test_shard_writer_rejects_nonempty_output(tmp_path) -> None:
    output_dir = tmp_path / "existing"
    output_dir.mkdir()
    (output_dir / "keep.txt").write_text("user data")

    with pytest.raises(FileExistsError, match="not empty"):
        save_toy_manifold_shards(output_dir, _tiny_config())
    assert (output_dir / "keep.txt").read_text() == "user data"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"ambient_dim": 2}, "ambient_dim"),
        (
            {
                "ambient_dim": 23,
                "manifold_types": ("product_torus_12d",),
            },
            "largest selected native embedding dimension",
        ),
        ({"n_samples": 0}, "n_samples"),
        ({"calibration_size": 1}, "calibration_size"),
        ({"manifolds_per_type": 0}, "manifolds_per_type"),
        ({"manifold_types": ()}, "manifold_types"),
        ({"manifold_types": ("circle", "circle")}, "unique"),
        ({"manifold_types": ("circle", "unknown")}, "unknown manifold"),
        ({"offset_radius": -1.0}, "offset_radius"),
        ({"noise_ratio": 0.0}, "noise_ratio"),
        ({"segment_min": 1.0, "segment_max": 1.0}, "segment"),
        ({"torus_major_radius": 1.0, "torus_minor_radius": 1.0}, "torus"),
        ({"mobius_half_width": 1.0}, "mobius"),
        ({"swiss_height_min": 3.0, "swiss_height_max": 2.0}, "swiss"),
        ({"helix_alpha": 0.0}, "helix_alpha"),
    ],
)
def test_invalid_configs_raise_clear_errors(overrides, message) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        make_toy_manifold_dataset(_tiny_config(**overrides))


def test_non_finite_geometry_is_rejected() -> None:
    with pytest.raises(ValueError, match="finite"):
        make_toy_manifold_dataset(_tiny_config(swiss_theta_max=float("inf")))
