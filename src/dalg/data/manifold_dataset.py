"""Synthetic toy manifolds for local-geometry experiments."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import TensorDataset


MANIFOLD_NAMES = (
    "segment",
    "circle",
    "flat_disk",
    "sphere",
    "torus",
    "mobius",
    "swiss_roll",
    "helix",
    "hypersphere_10d",
    "product_torus_12d",
)
INTRINSIC_DIMS = (1, 1, 2, 2, 2, 2, 2, 1, 10, 12)
EMBEDDING_DIMS = (1, 2, 2, 3, 3, 3, 3, 3, 11, 24)


@dataclass(frozen=True)
class ToyManifoldConfig:
    """Configuration for the synthetic manifold-instance dataset.

    There are ``manifolds_per_type`` independently embedded instances of each
    type selected by ``manifold_types``. ``offset_radius=0`` places every
    normalized instance at the origin. A positive radius places their centers
    at random directions on the corresponding ambient-space sphere without
    changing their local geometry.
    ``noise_ratio`` is the ratio between a type's normalized radius of
    curvature and the per-coordinate standard deviation of its ambient
    Gaussian observation noise.
    """

    ambient_dim: int = 128
    n_samples: int = 400_000
    calibration_size: int = 50_000
    manifolds_per_type: int = 8
    manifold_types: tuple[str, ...] = MANIFOLD_NAMES
    offset_radius: float = 4.0
    noise_ratio: float = 10_000.0
    seed: int = 0

    segment_min: float = -1.0
    segment_max: float = 1.0
    torus_major_radius: float = 2.0
    torus_minor_radius: float = 1.0
    mobius_half_width: float = 0.5
    swiss_theta_min: float = 1.5 * math.pi
    swiss_theta_max: float = 4.5 * math.pi
    swiss_height_min: float = 0.0
    swiss_height_max: float = 21.0
    helix_theta_min: float = 0.0
    helix_theta_max: float = 4.0 * math.pi
    helix_alpha: float = 0.2


def _validate_config(config: ToyManifoldConfig) -> None:
    for name in (
        "ambient_dim",
        "n_samples",
        "calibration_size",
        "manifolds_per_type",
    ):
        value = getattr(config, name)
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{name} must be an integer")
    if config.ambient_dim < 3:
        raise ValueError("ambient_dim must be at least 3")
    if config.n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if config.calibration_size < 2:
        raise ValueError("calibration_size must be at least 2")
    if config.manifolds_per_type <= 0:
        raise ValueError("manifolds_per_type must be positive")
    if not isinstance(config.manifold_types, tuple):
        raise TypeError("manifold_types must be a tuple")
    if not config.manifold_types:
        raise ValueError("manifold_types must not be empty")
    if len(set(config.manifold_types)) != len(config.manifold_types):
        raise ValueError("manifold_types must be unique")
    unknown_types = set(config.manifold_types) - set(MANIFOLD_NAMES)
    if unknown_types:
        raise ValueError(f"unknown manifold types: {sorted(unknown_types)}")
    required_ambient_dim = max(
        3,
        max(
            EMBEDDING_DIMS[MANIFOLD_NAMES.index(name)]
            for name in config.manifold_types
        ),
    )
    if config.ambient_dim < required_ambient_dim:
        raise ValueError(
            "ambient_dim must be at least the largest selected native embedding "
            f"dimension ({required_ambient_dim})"
        )
    if not isinstance(config.seed, int) or isinstance(config.seed, bool):
        raise TypeError("seed must be an integer")

    finite_values = {
        "offset_radius": config.offset_radius,
        "noise_ratio": config.noise_ratio,
        "segment_min": config.segment_min,
        "segment_max": config.segment_max,
        "torus_major_radius": config.torus_major_radius,
        "torus_minor_radius": config.torus_minor_radius,
        "mobius_half_width": config.mobius_half_width,
        "swiss_theta_min": config.swiss_theta_min,
        "swiss_theta_max": config.swiss_theta_max,
        "swiss_height_min": config.swiss_height_min,
        "swiss_height_max": config.swiss_height_max,
        "helix_theta_min": config.helix_theta_min,
        "helix_theta_max": config.helix_theta_max,
        "helix_alpha": config.helix_alpha,
    }
    for name, value in finite_values.items():
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")

    if config.offset_radius < 0:
        raise ValueError("offset_radius must be non-negative")
    if config.noise_ratio <= 0:
        raise ValueError("noise_ratio must be positive")
    if config.segment_min >= config.segment_max:
        raise ValueError("segment_min must be smaller than segment_max")
    if not (
        config.torus_major_radius > config.torus_minor_radius > 0
    ):
        raise ValueError("torus radii must satisfy major_radius > minor_radius > 0")
    if not 0 < config.mobius_half_width < 1:
        raise ValueError("mobius_half_width must lie in (0, 1)")
    if config.swiss_theta_min >= config.swiss_theta_max:
        raise ValueError("swiss_theta_min must be smaller than swiss_theta_max")
    if config.swiss_height_min >= config.swiss_height_max:
        raise ValueError("swiss_height_min must be smaller than swiss_height_max")
    if config.helix_theta_min >= config.helix_theta_max:
        raise ValueError("helix_theta_min must be smaller than helix_theta_max")
    if config.helix_alpha <= 0:
        raise ValueError("helix_alpha must be positive")


def _generator(seed: int, stream: int) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed((int(seed) + stream * 1_000_003) % (2**63 - 1))
    return generator


def _uniform(
    n: int,
    low: float,
    high: float,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    return low + (high - low) * torch.rand(
        n, generator=generator, dtype=torch.float64
    )


def _sample_segment(
    n: int, generator: torch.Generator, config: ToyManifoldConfig
) -> torch.Tensor:
    t = _uniform(
        n, config.segment_min, config.segment_max, generator=generator
    )
    return t[:, None]


def _sample_circle(
    n: int, generator: torch.Generator, _config: ToyManifoldConfig
) -> torch.Tensor:
    theta = _uniform(n, 0.0, 2.0 * math.pi, generator=generator)
    return torch.stack((torch.cos(theta), torch.sin(theta)), dim=1)


def _sample_flat_disk(
    n: int, generator: torch.Generator, _config: ToyManifoldConfig
) -> torch.Tensor:
    theta = _uniform(n, 0.0, 2.0 * math.pi, generator=generator)
    radius = torch.rand(n, generator=generator, dtype=torch.float64).sqrt()
    return torch.stack((radius * torch.cos(theta), radius * torch.sin(theta)), dim=1)


def _sample_sphere(
    n: int, generator: torch.Generator, _config: ToyManifoldConfig
) -> torch.Tensor:
    theta = _uniform(n, 0.0, 2.0 * math.pi, generator=generator)
    z = _uniform(n, -1.0, 1.0, generator=generator)
    radius = (1.0 - z.square()).clamp_min(0.0).sqrt()
    return torch.stack(
        (radius * torch.cos(theta), radius * torch.sin(theta), z), dim=1
    )


def _sample_torus(
    n: int, generator: torch.Generator, config: ToyManifoldConfig
) -> torch.Tensor:
    theta = _uniform(n, 0.0, 2.0 * math.pi, generator=generator)
    phi = _uniform(n, 0.0, 2.0 * math.pi, generator=generator)
    tube = config.torus_major_radius + config.torus_minor_radius * torch.cos(phi)
    return torch.stack(
        (
            tube * torch.cos(theta),
            tube * torch.sin(theta),
            config.torus_minor_radius * torch.sin(phi),
        ),
        dim=1,
    )


def _sample_mobius(
    n: int, generator: torch.Generator, config: ToyManifoldConfig
) -> torch.Tensor:
    phi = _uniform(n, 0.0, 2.0 * math.pi, generator=generator)
    t = _uniform(
        n, -config.mobius_half_width, config.mobius_half_width, generator=generator
    )
    half_phi = 0.5 * phi
    radius = 1.0 + t * torch.cos(half_phi)
    return torch.stack(
        (
            radius * torch.cos(phi),
            radius * torch.sin(phi),
            t * torch.sin(half_phi),
        ),
        dim=1,
    )


def _sample_swiss_roll(
    n: int, generator: torch.Generator, config: ToyManifoldConfig
) -> torch.Tensor:
    theta = _uniform(
        n, config.swiss_theta_min, config.swiss_theta_max, generator=generator
    )
    height = _uniform(
        n, config.swiss_height_min, config.swiss_height_max, generator=generator
    )
    return torch.stack(
        (theta * torch.cos(theta), height, theta * torch.sin(theta)), dim=1
    )


def _sample_helix(
    n: int, generator: torch.Generator, config: ToyManifoldConfig
) -> torch.Tensor:
    theta = _uniform(
        n, config.helix_theta_min, config.helix_theta_max, generator=generator
    )
    return torch.stack(
        (torch.cos(theta), torch.sin(theta), config.helix_alpha * theta), dim=1
    )


def _sample_hypersphere_10d(
    n: int, generator: torch.Generator, _config: ToyManifoldConfig
) -> torch.Tensor:
    points = torch.randn(n, 11, generator=generator, dtype=torch.float64)
    norms = points.norm(dim=1, keepdim=True)
    if torch.any(norms == 0.0):
        raise RuntimeError("hypersphere sampler produced a zero direction")
    return points / norms


def _sample_product_torus_12d(
    n: int, generator: torch.Generator, _config: ToyManifoldConfig
) -> torch.Tensor:
    angles = 2.0 * math.pi * torch.rand(
        n,
        12,
        generator=generator,
        dtype=torch.float64,
    )
    return torch.stack((torch.cos(angles), torch.sin(angles)), dim=2).reshape(n, 24)


def _surface_max_abs_principal_curvature(
    xu: torch.Tensor,
    xv: torch.Tensor,
    xuu: torch.Tensor,
    xuv: torch.Tensor,
    xvv: torch.Tensor,
) -> torch.Tensor:
    """Return ``max(|k1|, |k2|)`` from a parametric surface's derivatives."""

    normal = torch.linalg.cross(xu, xv, dim=-1)
    normal = normal / normal.norm(dim=-1, keepdim=True)

    first_uu = (xu * xu).sum(dim=-1)
    first_uv = (xu * xv).sum(dim=-1)
    first_vv = (xv * xv).sum(dim=-1)
    second_uu = (xuu * normal).sum(dim=-1)
    second_uv = (xuv * normal).sum(dim=-1)
    second_vv = (xvv * normal).sum(dim=-1)

    det_first = first_uu * first_vv - first_uv.square()
    mean = (
        second_uu * first_vv - 2.0 * second_uv * first_uv + second_vv * first_uu
    ) / (2.0 * det_first)
    gaussian = (second_uu * second_vv - second_uv.square()) / det_first
    half_gap = (mean.square() - gaussian).clamp_min(0.0).sqrt()
    return mean.abs() + half_gap


def _mobius_max_abs_curvature(config: ToyManifoldConfig) -> torch.Tensor:
    """Numerically maximize the Mobius strip's principal curvature."""

    phi_values = torch.linspace(0.0, 2.0 * math.pi, 4_097, dtype=torch.float64)
    width_values = torch.linspace(
        -config.mobius_half_width,
        config.mobius_half_width,
        33,
        dtype=torch.float64,
    )
    phi, width = torch.meshgrid(phi_values, width_values, indexing="ij")
    phi = phi.flatten()
    width = width.flatten()

    cos_phi = torch.cos(phi)
    sin_phi = torch.sin(phi)
    cos_half = torch.cos(0.5 * phi)
    sin_half = torch.sin(0.5 * phi)
    radius = 1.0 + width * cos_half
    radius_u = -0.5 * width * sin_half
    radius_uu = -0.25 * width * cos_half

    xu = torch.stack(
        (
            radius_u * cos_phi - radius * sin_phi,
            radius_u * sin_phi + radius * cos_phi,
            0.5 * width * cos_half,
        ),
        dim=-1,
    )
    xv = torch.stack((cos_half * cos_phi, cos_half * sin_phi, sin_half), dim=-1)
    xuu = torch.stack(
        (
            radius_uu * cos_phi - 2.0 * radius_u * sin_phi - radius * cos_phi,
            radius_uu * sin_phi + 2.0 * radius_u * cos_phi - radius * sin_phi,
            -0.25 * width * sin_half,
        ),
        dim=-1,
    )
    xuv = torch.stack(
        (
            -0.5 * sin_half * cos_phi - cos_half * sin_phi,
            -0.5 * sin_half * sin_phi + cos_half * cos_phi,
            0.5 * cos_half,
        ),
        dim=-1,
    )
    xvv = torch.zeros_like(xu)
    return _surface_max_abs_principal_curvature(xu, xv, xuu, xuv, xvv).max()


def _raw_max_abs_curvatures(config: ToyManifoldConfig) -> torch.Tensor:
    """Maximum extrinsic principal curvature of every unnormalized type."""

    torus_curvature = max(
        1.0 / config.torus_minor_radius,
        1.0 / (config.torus_major_radius - config.torus_minor_radius),
    )
    closest_swiss_theta = (
        0.0
        if config.swiss_theta_min <= 0.0 <= config.swiss_theta_max
        else min(abs(config.swiss_theta_min), abs(config.swiss_theta_max))
    )
    swiss_curvature = (closest_swiss_theta**2 + 2.0) / (
        closest_swiss_theta**2 + 1.0
    ) ** 1.5
    helix_curvature = 1.0 / (1.0 + config.helix_alpha**2)

    return torch.tensor(
        (
            0.0,
            1.0,
            0.0,
            1.0,
            torus_curvature,
            _mobius_max_abs_curvature(config),
            swiss_curvature,
            helix_curvature,
            1.0,
            1.0,
        ),
        dtype=torch.float64,
    )


_Sampler = Callable[[int, torch.Generator, ToyManifoldConfig], torch.Tensor]
_SAMPLERS: tuple[_Sampler, ...] = (
    _sample_segment,
    _sample_circle,
    _sample_flat_disk,
    _sample_sphere,
    _sample_torus,
    _sample_mobius,
    _sample_swiss_roll,
    _sample_helix,
    _sample_hypersphere_10d,
    _sample_product_torus_12d,
)


def _orthonormal_embedding(
    ambient_dim: int,
    local_dim: int,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    raw = torch.randn(
        ambient_dim, local_dim, generator=generator, dtype=torch.float64
    )
    q, _ = torch.linalg.qr(raw, mode="reduced")
    return q.T.contiguous()


def _offset_directions(
    config: ToyManifoldConfig, num_manifolds: int
) -> torch.Tensor:
    raw = torch.randn(
        num_manifolds,
        config.ambient_dim,
        generator=_generator(config.seed, 300),
        dtype=torch.float64,
    )
    return raw / raw.norm(dim=1, keepdim=True)


def _balanced_counts(total: int, num_manifolds: int) -> list[int]:
    base, remainder = divmod(total, num_manifolds)
    return [
        base + int(manifold_id < remainder)
        for manifold_id in range(num_manifolds)
    ]


def _make_dataset(
    n_samples: int,
    *,
    config: ToyManifoldConfig,
    stream: int,
    means: tuple[torch.Tensor, ...],
    scales: torch.Tensor,
    noise_stds: torch.Tensor,
    samplers: tuple[_Sampler, ...],
    embeddings: tuple[torch.Tensor, ...],
    offsets: torch.Tensor,
    manifold_type_ids: torch.Tensor,
) -> TensorDataset:
    samples = []
    manifold_ids = []
    num_manifolds = len(embeddings)
    for manifold_id, count in enumerate(
        _balanced_counts(n_samples, num_manifolds)
    ):
        if count == 0:
            continue
        type_id = int(manifold_type_ids[manifold_id])
        raw = samplers[type_id](
            count, _generator(config.seed, stream + manifold_id), config
        )
        normalized = (raw - means[type_id]) / scales[type_id]
        ambient = normalized @ embeddings[manifold_id] + offsets[manifold_id]
        ambient += noise_stds[type_id] * torch.randn(
            count,
            config.ambient_dim,
            generator=_generator(config.seed, stream + 2_000 + manifold_id),
            dtype=torch.float64,
        )
        samples.append(ambient)
        manifold_ids.append(
            torch.full((count,), manifold_id, dtype=torch.long)
        )

    x = torch.cat(samples, dim=0)
    y = torch.cat(manifold_ids, dim=0)
    permutation = torch.randperm(
        n_samples, generator=_generator(config.seed, stream + 1_000)
    )
    return TensorDataset(x[permutation].float(), y[permutation])


def make_toy_manifold_dataset(
    config: ToyManifoldConfig | None = None,
) -> tuple[TensorDataset, dict[str, object]]:
    """Generate points sampled from shared manifold instances.

    The first tensor is the ambient observation and the second is its
    manifold-instance ID. Observation noise is isotropic in the ambient space,
    with a type-specific standard deviation set by ``config.noise_ratio``.
    The activation-shard training path creates the train/validation split.
    """

    config = ToyManifoldConfig() if config is None else config
    if not isinstance(config, ToyManifoldConfig):
        raise TypeError("config must be a ToyManifoldConfig")
    _validate_config(config)

    selected_type_ids = tuple(
        MANIFOLD_NAMES.index(name) for name in config.manifold_types
    )
    samplers = tuple(_SAMPLERS[type_id] for type_id in selected_type_ids)
    intrinsic_dims = tuple(INTRINSIC_DIMS[type_id] for type_id in selected_type_ids)
    embedding_dims = tuple(EMBEDDING_DIMS[type_id] for type_id in selected_type_ids)

    means = []
    scales = []
    for type_id, sampler in zip(selected_type_ids, samplers, strict=True):
        calibration = sampler(
            config.calibration_size,
            _generator(config.seed, 100 + type_id),
            config,
        )
        mean = calibration.mean(dim=0)
        centered = calibration - mean
        scale = centered.square().sum(dim=1).mean().sqrt()
        if not torch.isfinite(scale) or scale <= 0:
            raise ValueError(
                f"{MANIFOLD_NAMES[type_id]} has zero or non-finite RMS"
            )
        means.append(mean)
        scales.append(scale)
    means_tuple = tuple(means)
    scales_tensor = torch.stack(scales)

    raw_max_abs_curvatures = _raw_max_abs_curvatures(config)[list(selected_type_ids)]
    max_abs_curvatures = raw_max_abs_curvatures * scales_tensor
    # A flat manifold has infinite curvature radius. Its normalized RMS size is
    # the only finite geometric scale available for adding nonzero noise.
    curvature_radii = torch.where(
        max_abs_curvatures > 0.0,
        max_abs_curvatures.reciprocal(),
        torch.ones_like(max_abs_curvatures),
    )
    noise_stds = curvature_radii / float(config.noise_ratio)

    num_manifolds = len(config.manifold_types) * config.manifolds_per_type
    manifold_type_ids = torch.arange(len(config.manifold_types)).repeat_interleave(
        config.manifolds_per_type
    )
    embeddings = []
    for manifold_id, type_id_tensor in enumerate(manifold_type_ids):
        type_id = int(type_id_tensor)
        embeddings.append(
            _orthonormal_embedding(
                config.ambient_dim,
                embedding_dims[type_id],
                generator=_generator(config.seed, 200 + manifold_id),
            )
        )

    embeddings_tuple = tuple(embeddings)
    offset_directions = _offset_directions(config, num_manifolds)
    offsets = float(config.offset_radius) * offset_directions

    dataset = _make_dataset(
        config.n_samples,
        config=config,
        stream=400,
        means=means_tuple,
        scales=scales_tensor,
        noise_stds=noise_stds,
        samplers=samplers,
        embeddings=embeddings_tuple,
        offsets=offsets,
        manifold_type_ids=manifold_type_ids,
    )
    manifolds = []
    for manifold_id, type_id_tensor in enumerate(manifold_type_ids):
        type_id = int(type_id_tensor)
        manifolds.append(
            {
                "manifold_id": manifold_id,
                "type_id": type_id,
                "type_name": config.manifold_types[type_id],
                "intrinsic_dim": intrinsic_dims[type_id],
                "embedding_dim": embedding_dims[type_id],
                "max_abs_curvature": max_abs_curvatures[type_id],
                "curvature_radius": curvature_radii[type_id],
                "noise_std": noise_stds[type_id],
                "position": offsets[manifold_id],
                "embedding": embeddings_tuple[manifold_id],
            }
        )
    metadata: dict[str, object] = {
        "config": asdict(config),
        "num_manifolds": num_manifolds,
        "manifold_types": config.manifold_types,
        "type_id_to_name": dict(enumerate(config.manifold_types)),
        "type_name_to_id": {
            name: type_id for type_id, name in enumerate(config.manifold_types)
        },
        "intrinsic_dims": intrinsic_dims,
        "embedding_dims": embedding_dims,
        "manifold_type_ids": manifold_type_ids,
        "calibration_means": means_tuple,
        "calibration_scales": scales_tensor,
        "raw_max_abs_curvatures": raw_max_abs_curvatures,
        "max_abs_curvatures": max_abs_curvatures,
        "curvature_radii": curvature_radii,
        "noise_stds": noise_stds,
        "curvature_definition": "maximum absolute extrinsic principal curvature",
        "flat_radius_convention": "unit RMS radius",
        "embeddings": embeddings_tuple,
        "offset_directions": offset_directions,
        "offsets": offsets,
        "manifolds": manifolds,
    }
    return dataset, metadata


def save_toy_manifold_shards(
    output_dir: str | Path,
    config: ToyManifoldConfig | None = None,
    *,
    shard_size: int = 50_000,
    layer: int = 0,
) -> Path:
    """Write toy points in the activation-shard layout used by MFA training.

    Every point is represented as a one-position activation window, so layer
    shards have shape ``(rows, 1, ambient_dim)`` and ``drop_prefix`` is zero.
    Downstream training creates the train/validation split with the standard
    ``val_frac`` and ``split_seed`` arguments.

    Per-row JSON metadata records the manifold instance and type. The larger
    tensors describing embeddings, offsets, and point labels are stored in
    ``manifold_metadata.pt``. Token shards are intentionally omitted because
    synthetic points have no textual token identity.

    The destination must be absent or empty to avoid mixing shards from
    different generator configurations.
    """
    if not isinstance(shard_size, int) or isinstance(shard_size, bool):
        raise TypeError("shard_size must be an integer")
    if shard_size <= 0:
        raise ValueError("shard_size must be positive")
    if not isinstance(layer, int) or isinstance(layer, bool):
        raise TypeError("layer must be an integer")
    if layer < 0:
        raise ValueError("layer must be non-negative")

    root = Path(output_dir)
    if root.exists():
        if not root.is_dir():
            raise FileExistsError(f"output path exists and is not a directory: {root}")
        if any(root.iterdir()):
            raise FileExistsError(f"output directory is not empty: {root}")

    config = ToyManifoldConfig() if config is None else config
    dataset, metadata = make_toy_manifold_dataset(config)
    manifold_type_ids = metadata["manifold_type_ids"]
    manifold_types = metadata["manifold_types"]
    intrinsic_dims = metadata["intrinsic_dims"]

    root.mkdir(parents=True, exist_ok=True)
    layer_dir = root / f"layer{layer:02d}"
    meta_dir = root / "meta"
    layer_dir.mkdir()
    meta_dir.mkdir()

    global_row = 0
    shard_id = 0
    points, manifold_ids = dataset.tensors
    for start in range(0, len(dataset), shard_size):
        end = min(start + shard_size, len(dataset))
        shard_points = points[start:end].clone().unsqueeze(1)
        shard_manifold_ids = manifold_ids[start:end].tolist()
        row_indices = list(range(global_row, global_row + len(shard_points)))

        rows = []
        for manifold_id in shard_manifold_ids:
            type_id = int(manifold_type_ids[manifold_id])
            rows.append(
                {
                    "subset": manifold_types[type_id],
                    "manifold_id": int(manifold_id),
                    "manifold_type_id": type_id,
                    "intrinsic_dim": intrinsic_dims[type_id],
                }
            )

        shard_path = layer_dir / f"shard_{shard_id:05d}.pt"
        shard_tmp = shard_path.with_suffix(".pt.tmp")
        torch.save(shard_points, shard_tmp)
        shard_tmp.replace(shard_path)

        meta_path = meta_dir / f"shard_{shard_id:05d}.json"
        meta_tmp = meta_path.with_suffix(".json.tmp")
        meta_tmp.write_text(
            json.dumps(
                {
                    "start": global_row,
                    "end": global_row + len(shard_points),
                    "row_indices": row_indices,
                    "rows": rows,
                }
            )
        )
        meta_tmp.replace(meta_path)

        global_row += len(shard_points)
        shard_id += 1

    saved_metadata = dict(metadata)
    saved_metadata.update(
        {
            "row_manifold_ids": manifold_ids,
            "canonical_order": "generated",
            "layer": layer,
        }
    )
    metadata_path = root / "manifold_metadata.pt"
    metadata_tmp = metadata_path.with_suffix(".pt.tmp")
    torch.save(saved_metadata, metadata_tmp)
    metadata_tmp.replace(metadata_path)

    shard_config = {
        "model": "synthetic/toy_manifolds",
        "mode": "synthetic_observations",
        "source_kind": "toy_manifolds",
        "layers": [layer],
        "window": 1,
        "d_model": config.ambient_dim,
        "dtype": "float32",
        "prepend_bos": False,
        "shard_size": shard_size,
        "drop_prefix": 0,
        "num_rows": global_row,
        "num_shards": shard_id,
        "generator_config": asdict(config),
        "manifold_metadata": metadata_path.name,
    }
    config_path = root / "config.json"
    config_tmp = config_path.with_suffix(".json.tmp")
    config_tmp.write_text(json.dumps(shard_config, indent=2) + "\n")
    config_tmp.replace(config_path)
    return root


__all__ = [
    "MANIFOLD_NAMES",
    "ToyManifoldConfig",
    "make_toy_manifold_dataset",
    "save_toy_manifold_shards",
]
