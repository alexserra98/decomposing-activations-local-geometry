"""Tests for the synthetic manifold-instance dataset generator."""

from __future__ import annotations

import json
import math
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
    _sample_cylinder,
    _sample_cylinder_10d,
    _sample_swiss_roll_10d,
    _sample_helix_4d,
    _sample_helix_12d,
    _sample_swiss_roll_12d,
    _sample_hypersphere_10d,
    _sample_product_torus_12d,
    _sample_product_torus_6d,
    _sample_product_torus_4d,
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
    num_types = len(MANIFOLD_NAMES)
    num_manifolds = num_types * 8

    assert isinstance(dataset, TensorDataset)
    assert points.shape == (124, 32)
    assert points.dtype == torch.float32
    assert manifold_ids.dtype == torch.long
    assert metadata["num_manifolds"] == num_manifolds
    assert tuple(metadata["manifold_types"]) == MANIFOLD_NAMES
    assert tuple(metadata["intrinsic_dims"]) == INTRINSIC_DIMS
    assert tuple(metadata["embedding_dims"]) == EMBEDDING_DIMS
    assert metadata["type_id_to_name"] == dict(enumerate(MANIFOLD_NAMES))
    assert metadata["curvature_definition"] == (
        "maximum absolute extrinsic principal curvature"
    )
    assert metadata["flat_radius_convention"] == "unit RMS radius"
    assert metadata["max_abs_curvatures"].shape == (num_types,)
    assert metadata["curvature_radii"].shape == (num_types,)
    assert metadata["noise_stds"].shape == (num_types,)
    assert torch.equal(
        metadata["max_abs_curvatures"][[0, 2]],
        torch.zeros(2, dtype=torch.float64),
    )
    assert torch.all(metadata["curvature_radii"] > 0)
    assert torch.all(metadata["noise_stds"] > 0)
    assert torch.allclose(
        metadata["curvature_radii"] / metadata["noise_stds"],
        torch.full((num_types,), 10_000.0, dtype=torch.float64),
    )

    counts = torch.bincount(manifold_ids, minlength=num_manifolds)
    assert int(counts.max() - counts.min()) <= 1

    manifolds = metadata["manifolds"]
    assert len(manifolds) == num_manifolds
    assert [item["manifold_id"] for item in manifolds] == list(
        range(num_manifolds)
    )
    type_counts = torch.bincount(
        metadata["manifold_type_ids"], minlength=num_types
    )
    assert torch.equal(type_counts, torch.full((num_types,), 8))
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


def test_special_raw_samples_satisfy_manifold_constraints() -> None:
    config = _tiny_config()
    helix_4d = _sample_helix_4d(128, _generator(config.seed, 0), config)
    hypersphere = _sample_hypersphere_10d(128, _generator(config.seed, 1), config)
    product_torus = _sample_product_torus_12d(
        128,
        _generator(config.seed, 2),
        config,
    )

    assert helix_4d.shape == (128, 4)
    helix_pairs = helix_4d.reshape(128, 2, 2)
    assert torch.allclose(
        helix_pairs[:, 0].norm(dim=1),
        torch.full((128,), config.helix_4d_radius_xy, dtype=torch.float64),
        atol=1e-12,
        rtol=0.0,
    )
    assert torch.allclose(
        helix_pairs[:, 1].norm(dim=1),
        torch.full((128,), config.helix_4d_radius_zw, dtype=torch.float64),
        atol=1e-12,
        rtol=0.0,
    )
    assert torch.allclose(
        helix_4d[:, 2],
        helix_4d[:, 0].square() - helix_4d[:, 1].square(),
        atol=1e-12,
        rtol=0.0,
    )
    assert torch.allclose(
        helix_4d[:, 3],
        2.0 * helix_4d[:, 0] * helix_4d[:, 1],
        atol=1e-12,
        rtol=0.0,
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


@pytest.mark.parametrize(
    ("dim", "sampler"), [(4, _sample_product_torus_4d), (6, _sample_product_torus_6d)]
)
def test_product_torus_samples_and_saved_geometry(dim, sampler) -> None:
    config = _tiny_config(
        ambient_dim=2 * dim,
        manifold_types=(f"product_torus_{dim}d",),
        manifolds_per_type=1,
        noise_ratio=None,
    )
    raw = sampler(128, _generator(config.seed, 0), config)
    assert raw.shape == (128, 2 * dim)
    assert torch.allclose(
        raw.reshape(128, dim, 2).norm(dim=2),
        torch.ones((128, dim), dtype=torch.float64),
        atol=1e-12,
        rtol=0.0,
    )
    dataset, metadata = make_toy_manifold_dataset(config)
    raw_points = (
        (dataset.tensors[0].double() - metadata["offsets"][0])
        @ metadata["embeddings"][0].T
    ) * metadata["calibration_scales"][0] + metadata["calibration_means"][0]
    assert torch.allclose(
        raw_points.reshape(-1, dim, 2).norm(dim=2),
        torch.ones((config.n_samples, dim), dtype=torch.float64),
        atol=1e-6,
        rtol=0.0,
    )
    assert metadata["intrinsic_dims"] == (dim,)
    assert metadata["embedding_dims"] == (2 * dim,)
    assert metadata["raw_max_abs_curvatures"].tolist() == [1.0]


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


def test_cylinder_surface_and_normalized_aspect_ratio() -> None:
    config = _tiny_config(
        manifold_types=("cylinder",),
        manifolds_per_type=1,
        n_samples=2_000,
        offset_radius=0.0,
        noise_ratio=1e12,
    )
    raw = _sample_cylinder(2_000, _generator(0, 0), config)
    assert torch.allclose(raw[:, [0, 2]].norm(dim=1), torch.ones(2_000).double())
    assert raw[:, 1].min() >= 0.0
    assert raw[:, 1].max() <= 5.0

    dataset, metadata = make_toy_manifold_dataset(config)
    local = dataset.tensors[0].double() @ metadata["embeddings"][0].T
    scale = metadata["calibration_scales"][0]
    local += metadata["calibration_means"][0] / scale
    radii = local[:, [0, 2]].norm(dim=1)
    assert torch.allclose(radii, torch.ones_like(radii) / scale, atol=1e-6)
    height = local[:, 1].max() - local[:, 1].min()
    assert float(height / radii.mean()) == pytest.approx(5.0, rel=0.01)
    assert tuple(metadata["intrinsic_dims"]) == (2,)
    assert tuple(metadata["embedding_dims"]) == (3,)


def test_raw_curvatures_match_manifold_geometry() -> None:
    config = _tiny_config(
        torus_major_radius=3.0,
        torus_minor_radius=1.0,
        swiss_theta_min=2.0,
        swiss_theta_max=5.0,
        helix_alpha=0.5,
        helix_4d_radius_xy=2.0,
        helix_4d_radius_zw=0.5,
        helix_4d_frequency_xy=1.5,
        helix_4d_frequency_zw=3.0,
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
    speed_squared = (2.0 * 1.5) ** 2 + (0.5 * 3.0) ** 2
    acceleration_norm = math.sqrt((2.0 * 1.5**2) ** 2 + (0.5 * 3.0**2) ** 2)
    assert float(curvatures[8]) == pytest.approx(
        acceleration_norm / speed_squared
    )
    assert torch.equal(curvatures[9:12], torch.ones(3, dtype=torch.float64))
    assert curvatures[12] == curvatures[6]
    assert curvatures[13] == 1.0
    assert torch.allclose(
        metadata["max_abs_curvatures"],
        curvatures * metadata["calibration_scales"],
    )


def test_instances_have_independent_embeddings_and_offset_directions() -> None:
    _, metadata = make_toy_manifold_dataset(_tiny_config(offset_radius=2.0))
    num_manifolds = len(MANIFOLD_NAMES) * 8

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
    assert directions.shape == (num_manifolds, 32)
    assert torch.allclose(
        directions.norm(dim=1),
        torch.ones(num_manifolds, dtype=directions.dtype),
    )
    assert torch.allclose(
        offsets.norm(dim=1),
        torch.full((num_manifolds,), 2.0, dtype=offsets.dtype),
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

    for manifold_id in range(len(MANIFOLD_NAMES) * 8):
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
        torch.full((len(MANIFOLD_NAMES),), 1_000.0, dtype=torch.float64),
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
    n_samples = max(124, len(MANIFOLD_NAMES) * 8 + 1)
    config = _tiny_config(n_samples=n_samples)
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
    assert shard_config["num_rows"] == n_samples
    assert shard_config["num_shards"] == math.ceil(n_samples / 25)
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
    assert len(meta_index) == n_samples
    assert [row["global_row"] for row in meta_index] == list(range(n_samples))
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
    assert dataset.num_items == n_samples
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
        (
            {"ambient_dim": 11, "manifold_types": ("product_torus_6d",)},
            "largest selected native embedding dimension",
        ),
        (
            {"ambient_dim": 7, "manifold_types": ("product_torus_4d",)},
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
        (
            {"helix_4d_theta_min": 2.0, "helix_4d_theta_max": 1.0},
            "helix_4d_theta",
        ),
        ({"helix_4d_radius_xy": 0.0}, "helix_4d_radius_xy"),
        ({"helix_4d_radius_zw": 0.0}, "helix_4d_radius_zw"),
        ({"helix_4d_frequency_xy": 0.0}, "helix_4d_frequency_xy"),
        ({"helix_4d_frequency_zw": 0.0}, "helix_4d_frequency_zw"),
    ],
)
def test_invalid_configs_raise_clear_errors(overrides, message) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        make_toy_manifold_dataset(_tiny_config(**overrides))


def test_non_finite_geometry_is_rejected() -> None:
    with pytest.raises(ValueError, match="finite"):
        make_toy_manifold_dataset(_tiny_config(swiss_theta_max=float("inf")))


@pytest.mark.parametrize("custom_bounds", [False, True])
def test_swiss_roll_10d_raw_geometry(custom_bounds):
    config = _tiny_config()
    if custom_bounds:
        config = replace(config, swiss_theta_min=0.3, swiss_theta_max=1.2,
                         swiss_height_min=-3.0, swiss_height_max=7.0)
    raw = _sample_swiss_roll_10d(256, _generator(17, 0), config)
    assert raw.shape == (256, 11) and raw.dtype == torch.float64
    theta = raw[:, :2].norm(dim=1)
    assert torch.all((theta >= config.swiss_theta_min) & (theta <= config.swiss_theta_max))
    expected = torch.stack((theta * theta.cos(), theta * theta.sin()), dim=1)
    assert torch.allclose(raw[:, :2], expected, atol=1e-12, rtol=0)
    assert torch.all((raw[:, 2:] >= config.swiss_height_min) &
                     (raw[:, 2:] <= config.swiss_height_max))
    assert not torch.equal(raw[:, 2], raw[:, 3])


def test_cylinder_10d_raw_geometry():
    raw = _sample_cylinder_10d(256, _generator(17, 0), _tiny_config())
    assert raw.shape == (256, 11) and raw.dtype == torch.float64
    assert torch.allclose(raw[:, :10].norm(dim=1), torch.ones(256).double(), atol=1e-12)
    assert torch.all((raw[:, 10] >= -2.5) & (raw[:, 10] <= 2.5))
    assert raw[:, 10].min() < 0 < raw[:, 10].max()


@pytest.mark.parametrize("type_name", ["swiss_roll_10d", "cylinder_10d"])
def test_ten_dimensional_native_embedding_requirement(type_name):
    config = _tiny_config(manifold_types=(type_name,), manifolds_per_type=1)
    with pytest.raises(ValueError, match="dimension \\(11\\)"):
        make_toy_manifold_dataset(replace(config, ambient_dim=10))
    dataset, metadata = make_toy_manifold_dataset(replace(config, ambient_dim=11))
    assert dataset.tensors[0].shape == (120, 11)
    assert metadata["intrinsic_dims"] == (10,)
    assert metadata["embedding_dims"] == (11,)


def test_mixed_ten_dimensional_shards_and_evaluation_geometry(tmp_path):
    from dalg.evaluation.toy_manifold_geometry import _project_mean_to_manifold

    names = ("hypersphere_10d", "swiss_roll_10d", "cylinder_10d")
    config = _tiny_config(n_samples=30, manifold_types=names,
                         manifolds_per_type=1, noise_ratio=None)
    dataset, metadata = make_toy_manifold_dataset(config)
    assert metadata["intrinsic_dims"] == (10, 10, 10)
    assert metadata["embedding_dims"] == (11, 11, 11)
    assert torch.equal(metadata["noise_stds"], torch.zeros(3).double())
    for point, manifold_id in zip(*dataset.tensors):
        projection = _project_mean_to_manifold(
            point, metadata["manifolds"][int(manifold_id)], metadata,
        )
        assert projection.unique
        assert projection.distance_squared < 1e-11
        assert projection.tangent.shape == (32, 10)
        assert torch.allclose(projection.tangent.T @ projection.tangent,
                              torch.eye(10).double(), atol=1e-10)
    root = save_toy_manifold_shards(tmp_path / "mixed_10d", config, shard_size=11)
    saved = torch.load(root / "manifold_metadata.pt", weights_only=True)
    shard_config = json.loads((root / "config.json").read_text())
    assert shard_config["generator_config"]["manifold_types"] == list(names)
    assert torch.equal(saved["row_manifold_ids"], dataset.tensors[1])
    index = load_meta_index(root, layer=0)
    assert [row["subset"] for row in index] == [names[int(i)] for i in dataset.tensors[1]]
    stream = ActivationBatchDataset(root, layer=0, batch_size=7, drop_prefix=0,
                                    shuffle_shards=False, shuffle_within_shard=False)
    assert torch.equal(torch.cat(list(stream)), dataset.tensors[0])


def test_ten_dimensional_noisy_and_noiseless_conditions_are_paired():
    config = _tiny_config(n_samples=60, manifolds_per_type=1,
                         manifold_types=("swiss_roll_10d", "cylinder_10d"), noise_ratio=None)
    clean, clean_metadata = make_toy_manifold_dataset(config)
    noisy, metadata = make_toy_manifold_dataset(replace(config, noise_ratio=10.0))
    quieter, quiet_metadata = make_toy_manifold_dataset(replace(config, noise_ratio=20.0))
    assert torch.equal(clean.tensors[1], noisy.tensors[1])
    assert torch.equal(clean.tensors[1], quieter.tensors[1])
    assert torch.equal(clean_metadata["calibration_scales"], metadata["calibration_scales"])
    assert torch.equal(metadata["noise_stds"], 2 * quiet_metadata["noise_stds"])
    assert torch.allclose(noisy.tensors[0] - clean.tensors[0],
                          2 * (quieter.tensors[0] - clean.tensors[0]), atol=1e-6)


@pytest.mark.parametrize("bounds", [(1.5 * math.pi, 4.5 * math.pi), (-3.0, 4.0)])
def test_swiss_roll_12d_raw_geometry_and_affine_span(bounds):
    config = _tiny_config(swiss_theta_min=bounds[0], swiss_theta_max=bounds[1],
                         swiss_height_min=-2.0, swiss_height_max=3.0)
    raw = _sample_swiss_roll_12d(512, _generator(17, 0), config)
    assert raw.shape == (512, 12) and raw.dtype == torch.float64
    theta, height = raw[:, 0], raw[:, 1]
    assert torch.all((theta >= bounds[0]) & (theta <= bounds[1]))
    assert torch.all((height >= -2.0) & (height <= 3.0))
    for k in range(1, 6):
        assert torch.allclose(raw[:, 2 * k], theta / k * (k * theta).cos())
        assert torch.allclose(raw[:, 2 * k + 1], theta / k * (k * theta).sin())
    assert torch.linalg.matrix_rank(raw - raw.mean(dim=0)) == 12


def test_helix_12d_raw_geometry_and_affine_span():
    raw = _sample_helix_12d(512, _generator(17, 0), _tiny_config())
    assert raw.shape == (512, 12) and raw.dtype == torch.float64
    theta = torch.atan2(raw[:, 1], raw[:, 0])
    for k in range(1, 7):
        assert torch.allclose(raw[:, 2 * k - 2], (k * theta).cos(), atol=1e-12)
        assert torch.allclose(raw[:, 2 * k - 1], (k * theta).sin(), atol=1e-12)
    assert torch.allclose(raw.square().sum(dim=1), torch.full((512,), 6.0).double())
    assert torch.linalg.matrix_rank(raw - raw.mean(dim=0)) == 12


@pytest.mark.parametrize("type_name,intrinsic_dim", [("swiss_roll_12d", 2), ("helix_12d", 1)])
def test_twelve_coordinate_native_embedding_requirement(type_name, intrinsic_dim):
    config = _tiny_config(manifold_types=(type_name,), manifolds_per_type=1)
    with pytest.raises(ValueError, match=r"dimension \(12\)"):
        make_toy_manifold_dataset(replace(config, ambient_dim=11))
    data, metadata = make_toy_manifold_dataset(replace(config, ambient_dim=12))
    assert data.tensors[0].shape == (120, 12)
    assert metadata["intrinsic_dims"] == (intrinsic_dim,)
    assert metadata["embedding_dims"] == (12,)


@pytest.mark.parametrize("noise_ratio", [None, 1_000.0])
def test_twelve_coordinate_mixture_shards_and_geometry_in_128d(tmp_path, noise_ratio):
    from dalg.evaluation.toy_manifold_geometry import _project_mean_to_manifold

    names = ("swiss_roll_12d", "helix_12d", "product_torus_12d")
    config = _tiny_config(ambient_dim=128, n_samples=18, manifold_types=names,
                         manifolds_per_type=1, noise_ratio=noise_ratio)
    data, metadata = make_toy_manifold_dataset(config)
    assert metadata["intrinsic_dims"] == (2, 1, 12)
    assert metadata["embedding_dims"] == (12, 12, 24)
    if noise_ratio is None:
        assert torch.count_nonzero(metadata["noise_stds"]) == 0
        for point, label in zip(*data.tensors):
            manifold = metadata["manifolds"][int(label)]
            projection = _project_mean_to_manifold(point, manifold, metadata)
            assert projection.unique
            assert projection.distance_squared < 1e-11
            rank = manifold["intrinsic_dim"]
            assert projection.tangent.shape == (128, rank)
            assert torch.allclose(projection.tangent.T @ projection.tangent,
                                  torch.eye(rank).double(), atol=1e-10)
    else:
        assert torch.allclose(metadata["noise_stds"],
                              metadata["curvature_radii"] / noise_ratio)
    root = save_toy_manifold_shards(tmp_path / "mixture", config, shard_size=7)
    saved = torch.load(root / "manifold_metadata.pt", weights_only=True)
    shard_config = json.loads((root / "config.json").read_text())
    assert shard_config["d_model"] == 128
    assert shard_config["generator_config"]["manifold_types"] == list(names)
    assert saved["intrinsic_dims"] == (2, 1, 12)
    assert saved["embedding_dims"] == (12, 12, 24)
    assert torch.equal(saved["row_manifold_ids"], data.tensors[1])
    index = load_meta_index(root, layer=0)
    assert [row["subset"] for row in index] == [names[int(i)] for i in data.tensors[1]]
    stream = ActivationBatchDataset(root, layer=0, batch_size=5, drop_prefix=0,
                                    shuffle_shards=False, shuffle_within_shard=False)
    assert torch.equal(torch.cat(list(stream)), data.tensors[0])


def test_hypersphere_6d_samples_and_geometry() -> None:
    from dalg.evaluation.toy_manifold_geometry import (
        _project_mean_to_manifold,
        _project_raw_point,
    )

    config = _tiny_config(
        ambient_dim=7, n_samples=20, manifold_types=("hypersphere_6d",),
        manifolds_per_type=1, noise_ratio=None,
    )
    with pytest.raises(ValueError, match=r"dimension \(7\)"):
        make_toy_manifold_dataset(replace(config, ambient_dim=6))
    dataset, metadata = make_toy_manifold_dataset(config)
    assert dataset.tensors[0].shape == (20, 7)
    assert metadata["intrinsic_dims"] == (6,)
    assert metadata["embedding_dims"] == (7,)
    assert metadata["raw_max_abs_curvatures"].tolist() == [1.0]
    raw = (
        (dataset.tensors[0].double() - metadata["offsets"][0])
        @ metadata["embeddings"][0].T
    ) * metadata["calibration_scales"][0] + metadata["calibration_means"][0]
    assert torch.allclose(raw.norm(dim=1), torch.ones(20).double(), atol=1e-6)
    for point in dataset.tensors[0]:
        projection = _project_mean_to_manifold(point, metadata["manifolds"][0], metadata)
        assert projection.unique
        assert projection.distance_squared < 1e-11
        assert projection.tangent.shape == (7, 6)
    origin = _project_raw_point(torch.zeros(7).double(), "hypersphere_6d", {})
    assert not origin.unique
    assert origin.point.norm() == 1.0
    assert torch.isfinite(origin.tangent).all()
