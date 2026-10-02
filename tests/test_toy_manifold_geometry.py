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


def _product_torus_point(dim: int = 12) -> torch.Tensor:
    angles = torch.linspace(0.2, 2.4, dim, dtype=torch.float64)
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
        ("cylinder", [3.0, 2.0, 4.0], [0.6, 2.0, 0.8], 2, 1e-10),
        ("cylinder", [0.5, -2.0, 0.0], [1.0, 0.0, 0.0], 2, 1e-10),
        ("cylinder", [0.0, 7.0, 2.0], [0.0, 5.0, 1.0], 2, 1e-10),
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
        (
            "helix_4d",
            [
                math.cos(0.37),
                math.sin(0.37),
                math.cos(0.74),
                math.sin(0.74),
            ],
            None,
            1,
            1e-8,
        ),
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
        ("cylinder", [0.0, 2.0, 0.0]),
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


@pytest.mark.parametrize("dim", [6, 10])
def test_hypersphere_projection_and_tangent(dim: int) -> None:
    expected = torch.arange(1, dim + 2, dtype=torch.float64)
    expected /= expected.norm()
    target = 2.5 * expected

    projection = _project_raw_point(
        target,
        f"hypersphere_{dim}d",
        asdict(ToyManifoldConfig()),
    )

    assert projection.unique
    assert torch.allclose(projection.point, expected, atol=1e-12, rtol=0.0)
    basis, full_rank = _orthonormal_basis(projection.tangent)
    assert full_rank
    assert basis.shape == (dim + 1, dim)
    assert torch.allclose(
        basis.T @ basis,
        torch.eye(dim, dtype=torch.float64),
        atol=1e-12,
        rtol=0.0,
    )
    assert torch.allclose(
        expected @ basis,
        torch.zeros(dim, dtype=torch.float64),
        atol=1e-12,
        rtol=0.0,
    )


@pytest.mark.parametrize("dim", [4, 6, 12])
def test_product_torus_projection_and_tangent(dim: int) -> None:
    expected = _product_torus_point(dim).reshape(dim, 2)
    radii = torch.linspace(0.5, 1.6, dim, dtype=torch.float64)
    target = (radii[:, None] * expected).reshape(-1)

    projection = _project_raw_point(
        target,
        f"product_torus_{dim}d",
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
    assert basis.shape == (2 * dim, dim)
    assert torch.allclose(
        basis.T @ basis,
        torch.eye(dim, dtype=torch.float64),
        atol=1e-12,
        rtol=0.0,
    )
    assert torch.allclose(
        (expected.reshape(-1)[:, None] * basis).reshape(dim, 2, dim).sum(dim=1),
        torch.zeros((dim, dim), dtype=torch.float64),
        atol=1e-12,
        rtol=0.0,
    )


@pytest.mark.parametrize("dim", [4, 6, 12])
def test_high_dimensional_projection_degeneracies_are_non_unique(dim: int) -> None:
    hypersphere = _project_raw_point(
        torch.zeros(11, dtype=torch.float64),
        "hypersphere_10d",
        asdict(ToyManifoldConfig()),
    )
    torus_target = _product_torus_point(dim).reshape(dim, 2)
    torus_target[-1] = 0.0
    product_torus = _project_raw_point(
        torus_target.reshape(-1),
        f"product_torus_{dim}d",
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
        product_torus.point.reshape(dim, 2)[-1],
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
            manifold_types=("circle", "helix", "cylinder"),
            offset_radius=2.0,
            seed=5,
        )
    )
    alpha = float(metadata["config"]["helix_alpha"])
    raw_points = {
        "circle": torch.tensor((1.0, 0.0), dtype=torch.float64),
        "cylinder": torch.tensor((1.0, 2.5, 0.0), dtype=torch.float64),
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


def _ten_dimensional_point_and_tangent(type_name):
    if type_name == "swiss_roll_10d":
        theta = 7.0
        point = torch.tensor(
            [theta * math.cos(theta), theta * math.sin(theta)] + [20.0] * 9,
            dtype=torch.float64,
        )
        tangent = torch.zeros(11, 10, dtype=torch.float64)
        tangent[:2, 0] = torch.tensor(
            [math.cos(theta) - theta * math.sin(theta),
             math.sin(theta) + theta * math.cos(theta)], dtype=torch.float64,
        )
        tangent[2:, 1:] = torch.eye(9)
    else:
        radial = torch.arange(1, 11, dtype=torch.float64)
        radial /= radial.norm()
        point = torch.cat((radial, torch.tensor([0.7], dtype=torch.float64)))
        _, _, vh = torch.linalg.svd(radial[None], full_matrices=True)
        tangent = torch.zeros(11, 10, dtype=torch.float64)
        tangent[:10, :9] = vh[1:].T
        tangent[10, 9] = 1.0
    return point, torch.linalg.qr(tangent, mode="reduced")[0]


@pytest.mark.parametrize("type_name", ["swiss_roll_10d", "cylinder_10d"])
def test_ten_dimensional_projection_and_analytic_tangent(type_name):
    point, expected_basis = _ten_dimensional_point_and_tangent(type_name)
    config = asdict(ToyManifoldConfig())
    for boundary in (None, -500.0, 500.0):
        target = point.clone()
        expected = point.clone()
        if boundary is not None:
            if type_name == "swiss_roll_10d":
                target[2:] = boundary
                expected[2:] = min(max(boundary, config["swiss_height_min"]),
                                   config["swiss_height_max"])
            else:
                target[:10] *= 2.5
                target[10] = boundary
                expected[10] = min(max(boundary, -2.5), 2.5)
        projection = _project_raw_point(target, type_name, config)
        assert projection.unique
        assert torch.allclose(projection.point, expected, atol=1e-6, rtol=0)
        basis, full_rank = _orthonormal_basis(projection.tangent)
        assert full_rank and basis.shape == (11, 10)
        assert torch.allclose(basis.T @ basis, torch.eye(10).double(), atol=1e-12)
        assert torch.allclose(basis @ basis.T, expected_basis @ expected_basis.T, atol=2e-7)


@pytest.mark.parametrize("type_name", ["swiss_roll_10d", "cylinder_10d"])
@pytest.mark.parametrize("dimension", [3, 10, 12])
def test_ten_dimensional_projection_rejects_wrong_coordinates(type_name, dimension):
    with pytest.raises(ValueError, match="eleven local coordinates"):
        _project_raw_point(torch.zeros(dimension), type_name, asdict(ToyManifoldConfig()))


def test_ten_dimensional_cylinder_axis_is_non_unique():
    target = torch.zeros(11, dtype=torch.float64)
    target[10] = 8.0
    projection = _project_raw_point(target, "cylinder_10d", {})
    expected = torch.zeros_like(target)
    expected[0], expected[10] = 1.0, 2.5
    assert not projection.unique
    assert torch.equal(projection.point, expected)
    assert torch.isfinite(projection.tangent).all()
    assert _orthonormal_basis(projection.tangent)[1]


def test_ten_dimensional_swiss_roll_off_manifold_and_tied_projections():
    config = asdict(ToyManifoldConfig())
    target = torch.tensor([4.0, -7.0] + [-5.0, 200.0, 30.0] * 3).double()
    projection = _project_raw_point(target, "swiss_roll_10d", config)
    theta = torch.linspace(config["swiss_theta_min"], config["swiss_theta_max"], 100001).double()
    sampled = torch.stack((theta * theta.cos(), theta * theta.sin()), dim=1)
    sampled_min = (sampled - target[:2]).square().sum(1).min()
    assert float((projection.point[:2] - target[:2]).square().sum()) <= float(sampled_min) + 1e-8
    assert torch.equal(projection.point[2:], target[2:].clamp(config["swiss_height_min"], config["swiss_height_max"]))
    config.update(swiss_theta_min=-math.pi, swiss_theta_max=math.pi)
    target[:2] = torch.tensor([0.0, 4.0])
    assert not _project_raw_point(target, "swiss_roll_10d", config).unique


@pytest.fixture
def mixed_ten_dimensional_geometry():
    names = ("hypersphere_10d", "swiss_roll_10d", "cylinder_10d")
    generator = torch.Generator().manual_seed(37)
    embedding = torch.linalg.qr(torch.randn(16, 11, generator=generator, dtype=torch.float64))[0].T
    metadata = {"config": asdict(ToyManifoldConfig()), "num_manifolds": 3,
                "calibration_means": [], "calibration_scales": [], "manifolds": []}
    points, bases, raw_points = [], [], []
    for i, name in enumerate(names):
        if name == "hypersphere_10d":
            raw = _hypersphere_point()
            _, _, vh = torch.linalg.svd(raw[None], full_matrices=True)
            basis = vh[1:].T
        else:
            raw, basis = _ten_dimensional_point_and_tangent(name)
        calibration_mean = torch.linspace(-0.5, 0.5, 11).double()
        scale = 2.0 + i
        offset = torch.zeros(16, dtype=torch.float64)
        offset[0] = i * 1000.0
        manifold = {"manifold_id": i, "type_id": i, "type_name": name,
                    "intrinsic_dim": 10, "embedding_dim": 11,
                    "embedding": embedding, "position": offset}
        metadata["calibration_means"].append(calibration_mean)
        metadata["calibration_scales"].append(scale)
        metadata["manifolds"].append(manifold)
        points.append(_ambient_point(metadata, manifold, raw))
        bases.append(embedding.T @ basis)
        raw_points.append(raw)
    return metadata, torch.stack(points), torch.stack(bases), raw_points


def test_ten_dimensional_ambient_projection_with_saved_transforms(mixed_ten_dimensional_geometry):
    metadata, points, bases, raw_points = mixed_ten_dimensional_geometry
    for i in (1, 2):
        manifold = metadata["manifolds"][i]
        target = raw_points[i].clone()
        if i == 1:
            target[2] = -3.0
            expected_raw = raw_points[i].clone()
            expected_raw[2] = 0.0
        else:
            target[:10] *= 2.0
            expected_raw = raw_points[i]
        ambient = _ambient_point(metadata, manifold, target)
        expected = _ambient_point(metadata, manifold, expected_raw)
        projection = _project_mean_to_manifold(ambient, manifold, metadata)
        assert projection.unique
        assert torch.allclose(projection.point, expected, atol=1e-6, rtol=0)
        assert projection.distance_squared == pytest.approx(float((ambient - expected).square().sum()), abs=1e-10)
        assert torch.allclose(projection.tangent @ projection.tangent.T, bases[i] @ bases[i].T, atol=2e-7)


@pytest.mark.parametrize("model_kind", ["hddc", "kmeans"])
def test_mixed_ten_dimensional_metrics(model_kind, mixed_ten_dimensional_geometry):
    from dalg.evaluation.toy_manifold_metrics import evaluate_toy_manifold_metrics
    from dalg.models.adaptive_q.mfa_hddc import MFA_HDDC
    from dalg.models.kmeans import KMeans

    metadata, points, bases, _ = mixed_ten_dimensional_geometry
    if model_kind == "hddc":
        model = MFA_HDDC(points, rank=10, init_directions=bases, scale_init=2.0, psi_init=0.1)
    else:
        model = KMeans.from_centroids(points)
        # Symmetric tangent samples give exactly the specified ten-dimensional PCA.
        samples = torch.cat([torch.cat((point + basis.T, point - basis.T))
                             for point, basis in zip(points, bases)])
        model.compute_pcs(samples, threshold=0.1)
    result = evaluate_toy_manifold_metrics(model, metadata, torch.ones(3, dtype=torch.bool))
    assert result["association"]["associated_components"] == 3
    assert result["rank"]["mean_learned"] == pytest.approx(10)
    for summary in (result, *result["per_manifold"]):
        for name in ("tangent_alignment", "tangent_containment"):
            for score in ("subspace_overlap", "worst_direction_cosine"):
                assert summary[name][score]["mean"] == pytest.approx(1.0, abs=1e-6)


# Independent parameterizations for numerical geometry checks.
def _twelve_coordinate_points(type_name, theta, height=0.7):
    if type_name == "swiss_roll_12d":
        coordinates = [theta, torch.full_like(theta, height)]
        for k in range(1, 6):
            coordinates.extend([theta * torch.cos(k * theta) / k,
                                theta * torch.sin(k * theta) / k])
    else:
        coordinates = []
        for k in range(1, 7):
            coordinates.extend([torch.cos(k * theta), torch.sin(k * theta)])
    return torch.stack(coordinates, dim=-1)


@pytest.mark.parametrize("type_name", ["swiss_roll_12d", "helix_12d"])
@pytest.mark.parametrize("theta", [0.0, 1e-7, 0.8, 2.0 * math.pi - 1e-7])
def test_twelve_coordinate_projection_and_numerical_tangents(type_name, theta):
    config = asdict(ToyManifoldConfig(swiss_theta_min=-1.0, swiss_theta_max=7.0))
    theta = torch.tensor(theta, dtype=torch.float64)
    target = _twelve_coordinate_points(type_name, theta)
    projection = _project_raw_point(target, type_name, config)
    assert projection.unique
    assert torch.allclose(projection.point, target, atol=2e-7, rtol=0)
    step = 1e-6
    derivative = (_twelve_coordinate_points(type_name, theta + step)
                  - _twelve_coordinate_points(type_name, theta - step)) / (2 * step)
    assert torch.allclose(projection.tangent[:, 0], derivative, atol=2e-6, rtol=0)
    basis, full_rank = _orthonormal_basis(projection.tangent)
    assert full_rank
    intrinsic_dim = 2 if type_name == "swiss_roll_12d" else 1
    assert basis.shape == (12, intrinsic_dim)
    if intrinsic_dim == 2:
        height_derivative = (_twelve_coordinate_points(type_name, theta, 0.7 + step)
                             - _twelve_coordinate_points(type_name, theta, 0.7 - step)) / (2 * step)
        assert torch.allclose(projection.tangent[:, 1], height_derivative, atol=1e-9)


@pytest.mark.parametrize("theta", [-2.0, 0.6, 3.0])
@pytest.mark.parametrize("height,expected_height", [(-4.0, -1.0), (5.0, 2.0)])
def test_swiss_roll_12d_boundaries(theta, height, expected_height):
    config = asdict(ToyManifoldConfig(swiss_theta_min=-2.0, swiss_theta_max=3.0,
                                     swiss_height_min=-1.0, swiss_height_max=2.0))
    theta = torch.tensor(theta, dtype=torch.float64)
    target = _twelve_coordinate_points("swiss_roll_12d", theta, height)
    expected = _twelve_coordinate_points("swiss_roll_12d", theta, expected_height)
    projection = _project_raw_point(target, "swiss_roll_12d", config)
    assert projection.unique
    assert torch.allclose(projection.point, expected, atol=3e-7, rtol=0)
    assert torch.linalg.matrix_rank(projection.tangent) == 2


@pytest.mark.parametrize("type_name", ["swiss_roll_12d", "helix_12d"])
@pytest.mark.parametrize("seed", [8, 17, 29])
def test_twelve_coordinate_off_manifold_projection_matches_dense_search(type_name, seed):
    config = asdict(ToyManifoldConfig())
    generator = torch.Generator().manual_seed(seed)
    target = 4 * torch.randn(12, generator=generator, dtype=torch.float64)
    if type_name == "swiss_roll_12d":
        grid = torch.linspace(config["swiss_theta_min"], config["swiss_theta_max"],
                              65_537, dtype=torch.float64)
        height = float(target[1].clamp(config["swiss_height_min"], config["swiss_height_max"]))
    else:
        grid = torch.linspace(0, 2 * math.pi, 65_537, dtype=torch.float64)
        height = 0.0
    points = _twelve_coordinate_points(type_name, grid, height)
    dense_distance = float((points - target).square().sum(dim=1).min())
    projection = _project_raw_point(target, type_name, config)
    distance = float((projection.point - target).square().sum())
    assert distance <= dense_distance + 1e-9
    assert distance == pytest.approx(dense_distance, abs=2e-4)


@pytest.mark.parametrize("type_name", ["swiss_roll_12d", "helix_12d"])
@pytest.mark.parametrize("bounds", [(-4.0, -1.0), (-2.0, 3.0), (0.0, 2.0), (4.0, 9.0)])
def test_twelve_coordinate_max_curvature_matches_numerical_derivatives(type_name, bounds):
    from dalg.data.manifold_dataset import MANIFOLD_NAMES, _raw_max_abs_curvatures

    config = ToyManifoldConfig(swiss_theta_min=bounds[0], swiss_theta_max=bounds[1])
    theta = torch.linspace(*bounds, 2049, dtype=torch.float64)
    if bounds[0] <= 0 <= bounds[1]:
        theta = torch.cat((theta, theta.new_zeros(1)))
    step = 1e-4
    center = _twelve_coordinate_points(type_name, theta)
    plus = _twelve_coordinate_points(type_name, theta + step)
    minus = _twelve_coordinate_points(type_name, theta - step)
    velocity = (plus - minus) / (2 * step)
    acceleration = (plus - 2 * center + minus) / step**2
    speed_squared = velocity.square().sum(dim=1)
    normal_acceleration = acceleration - (
        (acceleration * velocity).sum(dim=1) / speed_squared
    )[:, None] * velocity
    numerical_curvature = normal_acceleration.norm(dim=1) / speed_squared
    expected = float(_raw_max_abs_curvatures(config)[MANIFOLD_NAMES.index(type_name)])
    assert float(numerical_curvature.max()) == pytest.approx(expected, rel=2e-5)


def test_helix_12d_origin_and_discrete_projection_ties():
    config = asdict(ToyManifoldConfig())
    origin = torch.zeros(12, dtype=torch.float64)
    projection = _project_raw_point(origin, "helix_12d", config)
    assert not projection.unique
    assert torch.equal(projection.point, torch.tensor([1.0, 0.0] * 6).double())
    assert float(projection.point.square().sum()) == pytest.approx(6.0)
    assert torch.isfinite(projection.tangent).all()
    # A target in only the second harmonic plane has distinct minima at 0 and pi.
    target = origin.clone()
    target[2] = 1.0
    tied = _project_raw_point(target, "helix_12d", config)
    assert not tied.unique
    assert float((tied.point - target).square().sum()) == pytest.approx(5.0)


def test_swiss_roll_12d_distinct_projection_ties():
    config = asdict(ToyManifoldConfig(swiss_theta_min=-1.0, swiss_theta_max=1.0))
    target = torch.zeros(12, dtype=torch.float64)
    target[3] = 4.0
    projection = _project_raw_point(target, "swiss_roll_12d", config)
    assert not projection.unique
    assert abs(float(projection.point[0])) == pytest.approx(1.0)


@pytest.mark.parametrize("type_name", ["swiss_roll_12d", "helix_12d"])
def test_twelve_coordinate_projection_rejects_wrong_dimension(type_name):
    with pytest.raises(ValueError, match="twelve local coordinates"):
        _project_raw_point(torch.zeros(11), type_name, asdict(ToyManifoldConfig()))


def test_twelve_coordinate_mixture_metrics_in_128d():
    from dalg.evaluation.toy_manifold_metrics import evaluate_toy_manifold_metrics
    from dalg.models.kmeans import KMeans

    names = ("swiss_roll_12d", "helix_12d", "product_torus_12d")
    data, metadata = make_toy_manifold_dataset(ToyManifoldConfig(
        ambient_dim=128, n_samples=18, calibration_size=256, manifolds_per_type=1,
        manifold_types=names, noise_ratio=None, seed=19,
    ))
    points, labels = data.tensors
    projections = [_project_mean_to_manifold(points[labels == i][0], manifold, metadata)
                   for i, manifold in enumerate(metadata["manifolds"])]
    means = torch.stack([p.point for p in projections])
    model = KMeans.from_centroids(means)
    samples = torch.cat([torch.cat((p.point + 0.05 * p.tangent.T,
                                   p.point - 0.05 * p.tangent.T)) for p in projections])
    model.compute_pcs(samples, threshold=0.1)
    result = evaluate_toy_manifold_metrics(model, metadata, torch.ones(3, dtype=torch.bool))
    assert result["association"]["associated_components"] == 3
    assert result["rank"]["exact_match"] == pytest.approx(1.0)
    for manifold, rank in zip(result["per_manifold"], (2, 1, 12)):
        assert manifold["intrinsic_dim"] == rank
        assert manifold["tangent_alignment"]["subspace_overlap"]["mean"] == pytest.approx(1.0, abs=1e-6)
