from __future__ import annotations

import math
from dataclasses import asdict

import pytest
import torch

from dalg.data.manifold_dataset import ToyManifoldConfig, make_toy_manifold_dataset
from dalg.evaluation.toy_manifold_geometry import (
    _nearest_manifold_projection,
    _orthonormal_basis,
    _project_mean_to_manifold,
    _project_raw_point,
)


def _hypersphere_point() -> torch.Tensor:
    point = torch.arange(1, 12, dtype=torch.float64)
    return point / point.norm()


def _product_torus_point() -> torch.Tensor:
    angles = torch.linspace(0.2, 2.4, 12, dtype=torch.float64)
    return torch.stack((torch.cos(angles), torch.sin(angles)), dim=1).reshape(-1)


def _ambient_point(
    metadata: dict[str, object],
    manifold: dict[str, object],
    raw_point: torch.Tensor,
) -> torch.Tensor:
    type_id = int(manifold["type_id"])
    return (
        (raw_point - metadata["calibration_means"][type_id])
        / metadata["calibration_scales"][type_id]
    ) @ manifold["embedding"] + manifold["position"]


@pytest.mark.parametrize(
    ("type_name", "target", "expected", "tangent_dim", "atol"),
    [
        ("segment", [2.0], [1.0], 1, 1e-10),
        ("circle", [math.cos(0.7), math.sin(0.7)], None, 1, 1e-10),
        ("flat_disk", [2.0, 0.0], [1.0, 0.0], 2, 1e-10),
        (
            "sphere",
            [
                math.sin(0.9) * math.cos(0.4),
                math.sin(0.9) * math.sin(0.4),
                math.cos(0.9),
            ],
            None,
            2,
            1e-10,
        ),
        (
            "torus",
            [
                (2.0 + math.cos(0.8)) * math.cos(0.4),
                (2.0 + math.cos(0.8)) * math.sin(0.4),
                math.sin(0.8),
            ],
            None,
            2,
            1e-10,
        ),
        (
            "mobius",
            [
                (1.0 + 0.5 * math.cos(0.35)) * math.cos(0.7),
                (1.0 + 0.5 * math.cos(0.35)) * math.sin(0.7),
                0.5 * math.sin(0.35),
            ],
            None,
            2,
            1e-7,
        ),
        ("mobius", [1.3, 0.0, 0.0], None, 2, 1e-7),
        (
            "swiss_roll",
            [1.5 * math.pi * math.cos(1.5 * math.pi), 0.0, -1.5 * math.pi],
            None,
            2,
            1e-8,
        ),
        ("helix", [1.0, 0.0, 0.0], None, 1, 1e-8),
    ],
)
def test_raw_projection_and_tangent_geometry_for_every_manifold(
    type_name: str,
    target: list[float],
    expected: list[float] | None,
    tangent_dim: int,
    atol: float,
) -> None:
    target_tensor = torch.tensor(target, dtype=torch.float64)
    projection = _project_raw_point(
        target_tensor,
        type_name,
        asdict(ToyManifoldConfig()),
    )
    expected_tensor = (
        target_tensor if expected is None else torch.tensor(expected, dtype=torch.float64)
    )

    assert projection.unique
    assert torch.allclose(projection.point, expected_tensor, atol=atol, rtol=0.0)
    basis, full_rank = _orthonormal_basis(projection.tangent)
    assert full_rank
    assert basis.shape == (target_tensor.numel(), tangent_dim)
    assert torch.allclose(
        basis.T @ basis,
        torch.eye(tangent_dim, dtype=torch.float64),
        atol=1e-10,
        rtol=0.0,
    )


@pytest.mark.parametrize(
    ("type_name", "target"),
    [
        ("circle", [0.0, 0.0]),
        ("sphere", [0.0, 0.0, 0.0]),
        ("torus", [0.0, 0.0, 0.0]),
        ("torus", [2.0, 0.0, 0.0]),
    ],
)
def test_raw_projection_marks_non_unique_geometry(
    type_name: str,
    target: list[float],
) -> None:
    projection = _project_raw_point(
        torch.tensor(target, dtype=torch.float64),
        type_name,
        asdict(ToyManifoldConfig()),
    )

    assert not projection.unique
    assert torch.isfinite(projection.point).all()
    assert torch.isfinite(projection.tangent).all()


def test_hypersphere_projection_and_tangent_in_ten_dimensions() -> None:
    expected = _hypersphere_point()
    target = 2.5 * expected

    projection = _project_raw_point(
        target,
        "hypersphere_10d",
        asdict(ToyManifoldConfig()),
    )

    assert projection.unique
    assert torch.allclose(projection.point, expected, atol=1e-12, rtol=0.0)
    basis, full_rank = _orthonormal_basis(projection.tangent)
    assert full_rank
    assert basis.shape == (11, 10)
    assert torch.allclose(
        basis.T @ basis,
        torch.eye(10, dtype=torch.float64),
        atol=1e-12,
        rtol=0.0,
    )
    assert torch.allclose(
        expected @ basis,
        torch.zeros(10, dtype=torch.float64),
        atol=1e-12,
        rtol=0.0,
    )


def test_product_torus_projection_and_tangent_in_twelve_dimensions() -> None:
    expected = _product_torus_point().reshape(12, 2)
    radii = torch.linspace(0.5, 1.6, 12, dtype=torch.float64)
    target = (radii[:, None] * expected).reshape(-1)

    projection = _project_raw_point(
        target,
        "product_torus_12d",
        asdict(ToyManifoldConfig()),
    )

    assert projection.unique
    assert torch.allclose(
        projection.point,
        expected.reshape(-1),
        atol=1e-12,
        rtol=0.0,
    )
    basis, full_rank = _orthonormal_basis(projection.tangent)
    assert full_rank
    assert basis.shape == (24, 12)
    assert torch.allclose(
        basis.T @ basis,
        torch.eye(12, dtype=torch.float64),
        atol=1e-12,
        rtol=0.0,
    )
    assert torch.allclose(
        (expected.reshape(-1)[:, None] * basis).reshape(12, 2, 12).sum(dim=1),
        torch.zeros((12, 12), dtype=torch.float64),
        atol=1e-12,
        rtol=0.0,
    )


def test_high_dimensional_projection_degeneracies_are_non_unique() -> None:
    hypersphere = _project_raw_point(
        torch.zeros(11, dtype=torch.float64),
        "hypersphere_10d",
        asdict(ToyManifoldConfig()),
    )
    torus_target = _product_torus_point().reshape(12, 2)
    torus_target[4] = 0.0
    product_torus = _project_raw_point(
        torus_target.reshape(-1),
        "product_torus_12d",
        asdict(ToyManifoldConfig()),
    )

    assert not hypersphere.unique
    assert torch.equal(
        hypersphere.point,
        torch.nn.functional.one_hot(torch.tensor(0), num_classes=11).double(),
    )
    assert float(hypersphere.point.square().sum()) == pytest.approx(1.0)
    assert torch.isfinite(hypersphere.tangent).all()

    assert not product_torus.unique
    assert torch.equal(
        product_torus.point.reshape(12, 2)[4],
        torch.tensor((1.0, 0.0), dtype=torch.float64),
    )
    assert float(
        (product_torus.point - torus_target.reshape(-1)).square().sum()
    ) == pytest.approx(1.0)
    assert torch.isfinite(product_torus.tangent).all()


def test_high_dimensional_ambient_projection_reverses_saved_transforms() -> None:
    _, metadata = make_toy_manifold_dataset(
        ToyManifoldConfig(
            ambient_dim=32,
            n_samples=20,
            calibration_size=128,
            manifolds_per_type=1,
            manifold_types=("hypersphere_10d", "product_torus_12d"),
            offset_radius=2.0,
            seed=6,
        )
    )
    raw_points = (_hypersphere_point(), _product_torus_point())
    raw_targets = (
        1.4 * raw_points[0],
        (
            torch.linspace(0.6, 1.7, 12, dtype=torch.float64)[:, None]
            * raw_points[1].reshape(12, 2)
        ).reshape(-1),
    )

    for manifold, raw_point, raw_target in zip(
        metadata["manifolds"],
        raw_points,
        raw_targets,
        strict=True,
    ):
        ambient_target = _ambient_point(metadata, manifold, raw_target)
        expected_point = _ambient_point(metadata, manifold, raw_point)
        projection = _project_mean_to_manifold(
            ambient_target,
            manifold,
            metadata,
        )
        intrinsic_dim = int(manifold["intrinsic_dim"])

        assert projection.unique
        assert torch.allclose(
            projection.point,
            expected_point,
            atol=1e-10,
            rtol=0.0,
        )
        assert projection.distance_squared == pytest.approx(
            float((ambient_target - expected_point).square().sum()),
            abs=1e-12,
        )
        assert projection.tangent.shape == (32, intrinsic_dim)
        assert torch.allclose(
            projection.tangent.T @ projection.tangent,
            torch.eye(intrinsic_dim, dtype=torch.float64),
            atol=1e-10,
            rtol=0.0,
        )


def test_ambient_projection_uses_type_calibration_for_multiple_instances() -> None:
    _, metadata = make_toy_manifold_dataset(
        ToyManifoldConfig(
            ambient_dim=8,
            n_samples=16,
            calibration_size=64,
            manifolds_per_type=2,
            manifold_types=("circle", "helix"),
            offset_radius=2.0,
            seed=5,
        )
    )
    alpha = float(metadata["config"]["helix_alpha"])
    raw_points = {
        "circle": torch.tensor((1.0, 0.0), dtype=torch.float64),
        "helix": torch.tensor((math.cos(1.0), math.sin(1.0), alpha), dtype=torch.float64),
    }

    for manifold in metadata["manifolds"]:
        type_id = int(manifold["type_id"])
        raw_point = raw_points[manifold["type_name"]]
        ambient_point = (
            (raw_point - metadata["calibration_means"][type_id])
            / metadata["calibration_scales"][type_id]
        ) @ manifold["embedding"] + manifold["position"]
        projection = _project_mean_to_manifold(ambient_point, manifold, metadata)

        assert projection.unique
        assert projection.distance_squared == pytest.approx(0.0, abs=1e-12)
        assert torch.allclose(projection.point, ambient_point, atol=1e-7, rtol=0.0)

    first = metadata["manifolds"][0]
    raw_point = raw_points[first["type_name"]]
    ambient_point = (
        (raw_point - metadata["calibration_means"][0])
        / metadata["calibration_scales"][0]
    ) @ first["embedding"] + first["position"]
    tied_metadata = dict(metadata)
    duplicate = dict(first)
    duplicate["manifold_id"] = 999
    tied_metadata["manifolds"] = [first, duplicate]

    assert not _nearest_manifold_projection(ambient_point, tied_metadata).unique
