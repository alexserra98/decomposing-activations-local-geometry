"""Exact projections and tangent spaces for the planted toy manifolds."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import torch
from scipy.optimize import minimize_scalar

from dalg.data.manifold_dataset import _helix_12d_points, _swiss_roll_12d_points


_PROJECTION_GRID_POINTS = 4_097
_PROJECTION_XATOL = 1e-10
_GEOMETRY_EPS = 1e-12
_DISTANCE_RTOL = 1e-9
_DISTANCE_ATOL = 1e-12


@dataclass(frozen=True)
class _RawProjection:
    point: torch.Tensor
    tangent: torch.Tensor
    unique: bool


@dataclass(frozen=True)
class _AmbientProjection:
    point: torch.Tensor
    tangent: torch.Tensor
    distance_squared: float
    unique: bool


def _distance_tied(first: float, second: float) -> bool:
    return math.isclose(
        first,
        second,
        rel_tol=_DISTANCE_RTOL,
        abs_tol=_DISTANCE_ATOL,
    )


def _refined_grid_candidates(
    grid: torch.Tensor,
    values: torch.Tensor,
    objective: Callable[[float], float],
    *,
    periodic: bool,
) -> list[float]:
    """Refine every coarse local minimum of a bounded one-dimensional objective."""
    if grid.ndim != 1 or values.shape != grid.shape or grid.numel() < 3:
        raise ValueError("projection grid and values must be aligned one-dimensional arrays")
    if not torch.isfinite(values).all():
        raise ValueError("manifold projection objective contains non-finite values")

    candidates: list[float] = []
    if periodic:
        step = 2.0 * math.pi / int(grid.numel())
        previous = values.roll(1)
        following = values.roll(-1)
        local = (values <= previous) & (values <= following)
        local &= (values < previous) | (values < following)
        indices = torch.nonzero(local).flatten().tolist()
        if not indices:
            indices = [int(values.argmin())]
        for index in indices:
            center = float(grid[index])
            result = minimize_scalar(
                objective,
                bounds=(center - step, center + step),
                method="bounded",
                options={"xatol": _PROJECTION_XATOL},
            )
            if not result.success or not math.isfinite(float(result.fun)):
                raise RuntimeError("periodic manifold projection refinement failed")
            candidates.append(float(result.x) % (2.0 * math.pi))
    else:
        candidates.extend((float(grid[0]), float(grid[-1])))
        local = (values[1:-1] <= values[:-2]) & (values[1:-1] <= values[2:])
        local &= (values[1:-1] < values[:-2]) | (values[1:-1] < values[2:])
        indices = (torch.nonzero(local).flatten() + 1).tolist()
        if not indices:
            best = int(values.argmin())
            if 0 < best < grid.numel() - 1:
                indices = [best]
        for index in indices:
            result = minimize_scalar(
                objective,
                bounds=(float(grid[index - 1]), float(grid[index + 1])),
                method="bounded",
                options={"xatol": _PROJECTION_XATOL},
            )
            if not result.success or not math.isfinite(float(result.fun)):
                raise RuntimeError("bounded manifold projection refinement failed")
            candidates.append(float(result.x))

    return candidates


def _orthonormal_basis(tangent: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if tangent.ndim != 2 or tangent.shape[1] == 0:
        raise ValueError("tangent must contain at least one direction")
    q, r = torch.linalg.qr(tangent.double(), mode="reduced")
    full_rank = bool(torch.all(torch.diagonal(r).abs() > _GEOMETRY_EPS))
    return q, full_rank


def _same_projection_geometry(
    first: _RawProjection,
    second: _RawProjection,
) -> bool:
    if not torch.allclose(
        first.point,
        second.point,
        rtol=_DISTANCE_RTOL,
        atol=_DISTANCE_ATOL,
    ):
        return False
    first_basis, first_full_rank = _orthonormal_basis(first.tangent)
    second_basis, second_full_rank = _orthonormal_basis(second.tangent)
    if not first_full_rank or not second_full_rank:
        return False
    return bool(
        torch.allclose(
            first_basis @ first_basis.T,
            second_basis @ second_basis.T,
            rtol=_DISTANCE_RTOL,
            atol=_DISTANCE_ATOL,
        )
    )


def _select_raw_projection(
    candidates: list[tuple[float, _RawProjection]],
) -> _RawProjection:
    if not candidates:
        raise ValueError("manifold projection produced no candidates")
    candidates.sort(key=lambda item: item[0])
    minimum, best = candidates[0]
    tied = [
        projection
        for distance, projection in candidates
        if _distance_tied(distance, minimum)
    ]
    unique = best.unique and all(
        projection.unique and _same_projection_geometry(best, projection)
        for projection in tied[1:]
    )
    return _RawProjection(best.point, best.tangent, unique)


def _mobius_at(phi: float, target: torch.Tensor, half_width: float) -> _RawProjection:
    phi %= 2.0 * math.pi
    cos_phi = math.cos(phi)
    sin_phi = math.sin(phi)
    cos_half = math.cos(0.5 * phi)
    sin_half = math.sin(0.5 * phi)
    center = target.new_tensor((cos_phi, sin_phi, 0.0))
    width_direction = target.new_tensor(
        (cos_half * cos_phi, cos_half * sin_phi, sin_half)
    )
    width = float(torch.dot(target - center, width_direction).clamp(
        -half_width,
        half_width,
    ))
    radius = 1.0 + width * cos_half
    radius_derivative = -0.5 * width * sin_half
    point = center + width * width_direction
    tangent_phi = target.new_tensor(
        (
            radius_derivative * cos_phi - radius * sin_phi,
            radius_derivative * sin_phi + radius * cos_phi,
            0.5 * width * cos_half,
        )
    )
    tangent_width = width_direction
    tangent = torch.stack((tangent_phi, tangent_width), dim=1)
    return _RawProjection(point, tangent, True)


def _swiss_roll_at(
    theta: float,
    target: torch.Tensor,
    height_min: float,
    height_max: float,
) -> _RawProjection:
    height = float(target[1].clamp(height_min, height_max))
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    point = target.new_tensor(
        (theta * cos_theta, height, theta * sin_theta)
    )
    tangent_theta = target.new_tensor(
        (cos_theta - theta * sin_theta, 0.0, sin_theta + theta * cos_theta)
    )
    tangent_height = target.new_tensor((0.0, 1.0, 0.0))
    tangent = torch.stack((tangent_theta, tangent_height), dim=1)
    return _RawProjection(point, tangent, True)


def _helix_at(theta: float, target: torch.Tensor, alpha: float) -> _RawProjection:
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    point = target.new_tensor((cos_theta, sin_theta, alpha * theta))
    tangent = target.new_tensor((-sin_theta, cos_theta, alpha))[:, None]
    return _RawProjection(point, tangent, True)


def _helix_4d_at(
    theta: float,
    target: torch.Tensor,
    radius_xy: float,
    radius_zw: float,
    frequency_xy: float,
    frequency_zw: float,
) -> _RawProjection:
    angle_xy = frequency_xy * theta
    angle_zw = frequency_zw * theta
    cos_xy = math.cos(angle_xy)
    sin_xy = math.sin(angle_xy)
    cos_zw = math.cos(angle_zw)
    sin_zw = math.sin(angle_zw)
    point = target.new_tensor(
        (
            radius_xy * cos_xy,
            radius_xy * sin_xy,
            radius_zw * cos_zw,
            radius_zw * sin_zw,
        )
    )
    tangent = target.new_tensor(
        (
            -radius_xy * frequency_xy * sin_xy,
            radius_xy * frequency_xy * cos_xy,
            -radius_zw * frequency_zw * sin_zw,
            radius_zw * frequency_zw * cos_zw,
        )
    )[:, None]
    return _RawProjection(point, tangent, True)


def _swiss_roll_12d_at(
    theta: float,
    target: torch.Tensor,
    height_min: float,
    height_max: float,
) -> _RawProjection:
    height = target[1].clamp(height_min, height_max)
    point = _swiss_roll_12d_points(target.new_tensor(theta), height)
    frequencies = torch.arange(1, 6, dtype=target.dtype)
    angles = theta * frequencies
    spiral_derivative = torch.stack(
        (angles.cos() / frequencies - theta * angles.sin(),
         angles.sin() / frequencies + theta * angles.cos()),
        dim=1,
    )
    tangent = target.new_zeros((12, 2))
    tangent[0, 0] = 1.0
    tangent[1, 1] = 1.0
    tangent[2:, 0] = spiral_derivative.flatten()
    return _RawProjection(point, tangent, True)


def _helix_12d_at(theta: float, target: torch.Tensor) -> _RawProjection:
    theta %= 2.0 * math.pi
    point = _helix_12d_points(target.new_tensor(theta))
    frequencies = torch.arange(1, 7, dtype=target.dtype)
    angles = theta * frequencies
    tangent = torch.stack(
        (-frequencies * angles.sin(), frequencies * angles.cos()), dim=1
    ).reshape(12, 1)
    return _RawProjection(point, tangent, True)


def _project_raw_point(
    target: torch.Tensor,
    type_name: str,
    config: dict[str, Any],
) -> _RawProjection:
    """Project one raw local point onto an exact toy-manifold parameterization."""
    target = target.detach().cpu().double().reshape(-1)

    if type_name == "segment":
        if target.numel() != 1:
            raise ValueError("segment projection expects one local coordinate")
        coordinate = target[0].clamp(
            float(config["segment_min"]),
            float(config["segment_max"]),
        )
        return _RawProjection(coordinate.reshape(1), target.new_ones((1, 1)), True)

    if type_name == "circle":
        if target.numel() != 2:
            raise ValueError("circle projection expects two local coordinates")
        radius = float(target.norm())
        unique = radius > _GEOMETRY_EPS
        theta = math.atan2(float(target[1]), float(target[0])) if radius > 0.0 else 0.0
        point = target.new_tensor((math.cos(theta), math.sin(theta)))
        tangent = target.new_tensor((-math.sin(theta), math.cos(theta)))[:, None]
        return _RawProjection(point, tangent, unique)

    if type_name == "cylinder":
        if target.numel() != 3:
            raise ValueError("cylinder projection expects three local coordinates")
        circle = _project_raw_point(target[[0, 2]], "circle", config)
        point = target.new_tensor(
            (circle.point[0], target[1].clamp(0.0, 5.0), circle.point[1])
        )
        tangent = target.new_zeros((3, 2))
        tangent[[0, 2], 0] = circle.tangent[:, 0]
        tangent[1, 1] = 1.0
        return _RawProjection(point, tangent, circle.unique)

    if type_name == "cylinder_10d":
        if target.numel() != 11:
            raise ValueError("10D-cylinder projection expects eleven local coordinates")
        radius = float(target[:10].norm())
        radial_point = target[:10] / radius if radius > 0.0 else target.new_zeros(10)
        if radius == 0.0:
            radial_point[0] = 1.0
        _, _, vh = torch.linalg.svd(radial_point[None, :], full_matrices=True)
        point = torch.cat((radial_point, target[10:].clamp(-2.5, 2.5)))
        tangent = target.new_zeros((11, 10))
        tangent[:10, :9] = vh[1:].T
        tangent[10, 9] = 1.0
        return _RawProjection(point, tangent, radius > _GEOMETRY_EPS)

    if type_name == "flat_disk":
        if target.numel() != 2:
            raise ValueError("flat-disk projection expects two local coordinates")
        radius = float(target.norm())
        point = target.clone() if radius <= 1.0 else target / radius
        return _RawProjection(point, torch.eye(2, dtype=torch.float64), True)

    if type_name == "sphere":
        if target.numel() != 3:
            raise ValueError("sphere projection expects three local coordinates")
        radius = float(target.norm())
        unique = radius > _GEOMETRY_EPS
        point = target / radius if radius > 0.0 else target.new_tensor((1.0, 0.0, 0.0))
        _, _, vh = torch.linalg.svd(point[None, :], full_matrices=True)
        return _RawProjection(point, vh[1:].T.contiguous(), unique)

    if type_name in ("hypersphere_6d", "hypersphere_10d"):
        intrinsic_dim = 6 if type_name == "hypersphere_6d" else 10
        if target.numel() != intrinsic_dim + 1:
            raise ValueError(
                f"{intrinsic_dim}D-hypersphere projection expects "
                f"{intrinsic_dim + 1} local coordinates"
            )
        radius = float(target.norm())
        unique = radius > _GEOMETRY_EPS
        point = target / radius if radius > 0.0 else target.new_zeros(intrinsic_dim + 1)
        if radius == 0.0:
            point[0] = 1.0
        _, _, vh = torch.linalg.svd(point[None, :], full_matrices=True)
        return _RawProjection(point, vh[1:].T.contiguous(), unique)

    if type_name == "torus":
        if target.numel() != 3:
            raise ValueError("torus projection expects three local coordinates")
        major = float(config["torus_major_radius"])
        minor = float(config["torus_minor_radius"])
        radial = math.hypot(float(target[0]), float(target[1]))
        theta_unique = radial > _GEOMETRY_EPS
        theta = math.atan2(float(target[1]), float(target[0])) if radial > 0.0 else 0.0
        cross_x = radial - major
        cross_z = float(target[2])
        cross_radius = math.hypot(cross_x, cross_z)
        phi_unique = cross_radius > _GEOMETRY_EPS
        phi = math.atan2(cross_z, cross_x) if cross_radius > 0.0 else 0.0
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        cos_phi = math.cos(phi)
        sin_phi = math.sin(phi)
        tube = major + minor * cos_phi
        point = target.new_tensor(
            (tube * cos_theta, tube * sin_theta, minor * sin_phi)
        )
        tangent_theta = target.new_tensor(
            (-tube * sin_theta, tube * cos_theta, 0.0)
        )
        tangent_phi = target.new_tensor(
            (-minor * sin_phi * cos_theta, -minor * sin_phi * sin_theta, minor * cos_phi)
        )
        tangent = torch.stack((tangent_theta, tangent_phi), dim=1)
        return _RawProjection(point, tangent, theta_unique and phi_unique)

    if type_name in ("product_torus_4d", "product_torus_6d", "product_torus_12d"):
        intrinsic_dim = {
            "product_torus_4d": 4,
            "product_torus_6d": 6,
            "product_torus_12d": 12,
        }[type_name]
        if target.numel() != 2 * intrinsic_dim:
            raise ValueError(
                f"{type_name} projection expects {2 * intrinsic_dim} local coordinates"
            )
        target_pairs = target.reshape(intrinsic_dim, 2)
        pair_radii = target_pairs.norm(dim=1)
        nonzero_pairs = pair_radii > _GEOMETRY_EPS
        point_pairs = target_pairs / pair_radii.clamp_min(_GEOMETRY_EPS)[:, None]
        point_pairs[~nonzero_pairs] = target.new_tensor((1.0, 0.0))

        tangent = target.new_zeros((2 * intrinsic_dim, intrinsic_dim))
        for circle_id, pair in enumerate(point_pairs):
            tangent[2 * circle_id : 2 * circle_id + 2, circle_id] = torch.stack(
                (-pair[1], pair[0])
            )
        return _RawProjection(
            point_pairs.reshape(-1),
            tangent,
            bool(nonzero_pairs.all()),
        )

    if type_name == "mobius":
        if target.numel() != 3:
            raise ValueError("Mobius projection expects three local coordinates")
        half_width = float(config["mobius_half_width"])
        grid = torch.arange(_PROJECTION_GRID_POINTS, dtype=torch.float64)
        grid *= 2.0 * math.pi / _PROJECTION_GRID_POINTS
        cos_phi = torch.cos(grid)
        sin_phi = torch.sin(grid)
        cos_half = torch.cos(0.5 * grid)
        sin_half = torch.sin(0.5 * grid)
        centers = torch.stack((cos_phi, sin_phi, torch.zeros_like(grid)), dim=1)
        directions = torch.stack(
            (cos_half * cos_phi, cos_half * sin_phi, sin_half), dim=1
        )
        widths = ((target[None, :] - centers) * directions).sum(dim=1)
        widths.clamp_(-half_width, half_width)
        points = centers + widths[:, None] * directions
        values = (points - target[None, :]).square().sum(dim=1)

        def objective(phi: float) -> float:
            projection = _mobius_at(phi, target, half_width)
            return float((projection.point - target).square().sum())

        parameters = _refined_grid_candidates(
            grid, values, objective, periodic=True
        )
        candidates = []
        for phi in parameters:
            projection = _mobius_at(phi, target, half_width)
            candidates.append((objective(phi), projection))
        return _select_raw_projection(candidates)

    if type_name == "swiss_roll_12d":
        if target.numel() != 12:
            raise ValueError("12D-Swiss-roll projection expects twelve local coordinates")
        theta_min = float(config["swiss_theta_min"])
        theta_max = float(config["swiss_theta_max"])
        height_min = float(config["swiss_height_min"])
        height_max = float(config["swiss_height_max"])
        grid = torch.linspace(
            theta_min, theta_max, _PROJECTION_GRID_POINTS, dtype=torch.float64
        )
        height = target[1].clamp(height_min, height_max)
        points = _swiss_roll_12d_points(grid, height.expand_as(grid))
        values = (points - target).square().sum(dim=1)

        def objective(theta: float) -> float:
            point = _swiss_roll_12d_points(target.new_tensor(theta), height)
            return float((point - target).square().sum())

        parameters = _refined_grid_candidates(grid, values, objective, periodic=False)
        return _select_raw_projection([
            (objective(theta), _swiss_roll_12d_at(theta, target, height_min, height_max))
            for theta in parameters
        ])

    if type_name == "helix_12d":
        if target.numel() != 12:
            raise ValueError("12D-helix projection expects twelve local coordinates")
        if float(target.norm()) <= _GEOMETRY_EPS:
            representative = _helix_12d_at(0.0, target)
            return _RawProjection(representative.point, representative.tangent, False)
        grid = torch.arange(_PROJECTION_GRID_POINTS, dtype=torch.float64)
        grid *= 2.0 * math.pi / _PROJECTION_GRID_POINTS
        points = _helix_12d_points(grid)
        values = (points - target).square().sum(dim=1)

        def objective(theta: float) -> float:
            point = _helix_12d_points(target.new_tensor(theta))
            return float((point - target).square().sum())

        parameters = _refined_grid_candidates(grid, values, objective, periodic=True)
        return _select_raw_projection([
            (objective(theta), _helix_12d_at(theta, target)) for theta in parameters
        ])

    if type_name == "swiss_roll_10d":
        if target.numel() != 11:
            raise ValueError("10D-Swiss-roll projection expects eleven local coordinates")
        height_min = float(config["swiss_height_min"])
        height_max = float(config["swiss_height_max"])
        # The flat factors separate from the spiral's one-dimensional search.
        spiral_target = target.new_tensor((target[0], height_min, target[1]))
        spiral = _project_raw_point(spiral_target, "swiss_roll", config)
        point = torch.cat((spiral.point[[0, 2]], target[2:].clamp(height_min, height_max)))
        tangent = target.new_zeros((11, 10))
        tangent[:2, 0] = spiral.tangent[[0, 2], 0]
        tangent[2:, 1:] = torch.eye(9, dtype=target.dtype)
        return _RawProjection(point, tangent, spiral.unique)

    if type_name == "swiss_roll":
        if target.numel() != 3:
            raise ValueError("Swiss-roll projection expects three local coordinates")
        theta_min = float(config["swiss_theta_min"])
        theta_max = float(config["swiss_theta_max"])
        height_min = float(config["swiss_height_min"])
        height_max = float(config["swiss_height_max"])
        grid = torch.linspace(
            theta_min,
            theta_max,
            _PROJECTION_GRID_POINTS,
            dtype=torch.float64,
        )
        height = target[1].clamp(height_min, height_max)
        points = torch.stack(
            (grid * torch.cos(grid), height.expand_as(grid), grid * torch.sin(grid)),
            dim=1,
        )
        values = (points - target[None, :]).square().sum(dim=1)

        def objective(theta: float) -> float:
            projection = _swiss_roll_at(
                theta, target, height_min, height_max
            )
            return float((projection.point - target).square().sum())

        parameters = _refined_grid_candidates(
            grid, values, objective, periodic=False
        )
        candidates = []
        for theta in parameters:
            projection = _swiss_roll_at(theta, target, height_min, height_max)
            candidates.append((objective(theta), projection))
        return _select_raw_projection(candidates)

    if type_name == "helix":
        if target.numel() != 3:
            raise ValueError("helix projection expects three local coordinates")
        theta_min = float(config["helix_theta_min"])
        theta_max = float(config["helix_theta_max"])
        alpha = float(config["helix_alpha"])
        grid = torch.linspace(
            theta_min,
            theta_max,
            _PROJECTION_GRID_POINTS,
            dtype=torch.float64,
        )
        points = torch.stack(
            (torch.cos(grid), torch.sin(grid), alpha * grid), dim=1
        )
        values = (points - target[None, :]).square().sum(dim=1)

        def objective(theta: float) -> float:
            projection = _helix_at(theta, target, alpha)
            return float((projection.point - target).square().sum())

        parameters = _refined_grid_candidates(
            grid, values, objective, periodic=False
        )
        candidates = []
        for theta in parameters:
            projection = _helix_at(theta, target, alpha)
            candidates.append((objective(theta), projection))
        return _select_raw_projection(candidates)

    if type_name == "helix_4d":
        if target.numel() != 4:
            raise ValueError("4D-helix projection expects four local coordinates")
        theta_min = float(config["helix_4d_theta_min"])
        theta_max = float(config["helix_4d_theta_max"])
        radius_xy = float(config["helix_4d_radius_xy"])
        radius_zw = float(config["helix_4d_radius_zw"])
        frequency_xy = float(config["helix_4d_frequency_xy"])
        frequency_zw = float(config["helix_4d_frequency_zw"])
        grid = torch.linspace(
            theta_min,
            theta_max,
            _PROJECTION_GRID_POINTS,
            dtype=torch.float64,
        )
        angle_xy = frequency_xy * grid
        angle_zw = frequency_zw * grid
        points = torch.stack(
            (
                radius_xy * torch.cos(angle_xy),
                radius_xy * torch.sin(angle_xy),
                radius_zw * torch.cos(angle_zw),
                radius_zw * torch.sin(angle_zw),
            ),
            dim=1,
        )
        values = (points - target[None, :]).square().sum(dim=1)

        def objective(theta: float) -> float:
            projection = _helix_4d_at(
                theta,
                target,
                radius_xy,
                radius_zw,
                frequency_xy,
                frequency_zw,
            )
            return float((projection.point - target).square().sum())

        parameters = _refined_grid_candidates(
            grid, values, objective, periodic=False
        )
        candidates = []
        for theta in parameters:
            projection = _helix_4d_at(
                theta,
                target,
                radius_xy,
                radius_zw,
                frequency_xy,
                frequency_zw,
            )
            candidates.append((objective(theta), projection))
        return _select_raw_projection(candidates)

    raise ValueError(f"unsupported toy manifold type: {type_name!r}")


def _project_mean_to_manifold(
    mean: torch.Tensor,
    manifold: dict[str, Any],
    metadata: dict[str, Any],
) -> _AmbientProjection:
    """Project an ambient mean onto one planted noiseless manifold instance."""
    mean = mean.detach().cpu().double().reshape(-1)
    type_id = int(manifold["type_id"])
    embedding = manifold["embedding"].detach().cpu().double()
    offset = manifold["position"].detach().cpu().double()
    calibration_mean = metadata["calibration_means"][type_id].detach().cpu().double()
    calibration_scale = float(metadata["calibration_scales"][type_id])
    if calibration_scale <= 0.0 or not math.isfinite(calibration_scale):
        raise ValueError("toy-manifold calibration scale must be finite and positive")
    if mean.numel() != embedding.shape[1] or offset.shape != mean.shape:
        raise ValueError("manifold embedding or offset does not match the model dimension")

    raw_target = (mean - offset) @ embedding.T
    raw_target = raw_target * calibration_scale + calibration_mean
    raw = _project_raw_point(
        raw_target,
        str(manifold["type_name"]),
        metadata["config"],
    )
    ambient_point = (
        (raw.point - calibration_mean) / calibration_scale
    ) @ embedding + offset
    ambient_tangent = embedding.T @ raw.tangent / calibration_scale
    tangent_basis, full_rank = _orthonormal_basis(ambient_tangent)
    distance_squared = float((mean - ambient_point).square().sum())
    return _AmbientProjection(
        point=ambient_point,
        tangent=tangent_basis,
        distance_squared=distance_squared,
        unique=raw.unique and full_rank,
    )


def _nearest_manifold_projection(
    mean: torch.Tensor,
    metadata: dict[str, Any],
) -> _AmbientProjection:
    projections = [
        _project_mean_to_manifold(mean, manifold, metadata)
        for manifold in metadata["manifolds"]
    ]
    if not projections:
        raise ValueError("toy-manifold metadata contains no manifold instances")
    projections.sort(key=lambda projection: projection.distance_squared)
    best = projections[0]
    tied = [
        projection
        for projection in projections
        if _distance_tied(projection.distance_squared, best.distance_squared)
    ]
    unique = best.unique and len(tied) == 1
    return _AmbientProjection(
        point=best.point,
        tangent=best.tangent,
        distance_squared=best.distance_squared,
        unique=unique,
    )


__all__ = [
    "_AmbientProjection",
    "_distance_tied",
    "_nearest_manifold_projection",
    "_orthonormal_basis",
    "_project_mean_to_manifold",
    "_project_raw_point",
]
