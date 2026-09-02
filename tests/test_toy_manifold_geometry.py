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

