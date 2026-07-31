"""Synthetic toy manifolds for local-geometry experiments."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
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
)
INTRINSIC_DIMS = (1, 1, 2, 2, 2, 2, 2, 1)
EMBEDDING_DIMS = (1, 2, 2, 3, 3, 3, 3, 3)


@dataclass(frozen=True)
class ToyManifoldConfig:
    """Configuration for the synthetic manifold-instance dataset.

    There are ``manifolds_per_type`` independently embedded instances of each
    manifold type. ``offset_radius=0`` places every normalized instance at the
    origin. A positive radius places their centers at random directions on the
    corresponding ambient-space sphere without changing their local geometry.
    """

    ambient_dim: int = 128
    n_train: int = 300_000
    n_val: int = 100_000
    calibration_size: int = 50_000
    manifolds_per_type: int = 8
    offset_radius: float = 4.0
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
        "n_train",
        "n_val",
        "calibration_size",
        "manifolds_per_type",
    ):
        value = getattr(config, name)
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{name} must be an integer")
    if config.ambient_dim < 3:
        raise ValueError("ambient_dim must be at least 3")
    if config.n_train <= 0 or config.n_val <= 0:
        raise ValueError("n_train and n_val must be positive")
    if config.calibration_size < 2:
        raise ValueError("calibration_size must be at least 2")
    if config.manifolds_per_type <= 0:
        raise ValueError("manifolds_per_type must be positive")
    if not isinstance(config.seed, int) or isinstance(config.seed, bool):
        raise TypeError("seed must be an integer")

    finite_values = {
        "offset_radius": config.offset_radius,
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


def _make_split(
    n_samples: int,
    *,
    config: ToyManifoldConfig,
    stream: int,
    means: tuple[torch.Tensor, ...],
    scales: torch.Tensor,
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
        raw = _SAMPLERS[type_id](
            count, _generator(config.seed, stream + manifold_id), config
        )
        normalized = (raw - means[type_id]) / scales[type_id]
        ambient = normalized @ embeddings[manifold_id] + offsets[manifold_id]
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


def make_toy_manifold_datasets(
    config: ToyManifoldConfig | None = None,
) -> tuple[TensorDataset, TensorDataset, dict[str, object]]:
    """Generate train and validation points from shared manifold instances.

    The first tensor in each dataset is the ambient observation and the second
    is its manifold-instance ID. Train and validation independently sample
    points from the same instances. This ordering lets the datasets pass
    directly through a normal ``DataLoader`` into ``train_nll`` while retaining
    ground-truth instance membership for evaluation.
    """

    config = ToyManifoldConfig() if config is None else config
    if not isinstance(config, ToyManifoldConfig):
        raise TypeError("config must be a ToyManifoldConfig")
    _validate_config(config)

    means = []
    scales = []
    for type_id, sampler in enumerate(_SAMPLERS):
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

    num_manifolds = len(MANIFOLD_NAMES) * config.manifolds_per_type
    manifold_type_ids = torch.arange(len(MANIFOLD_NAMES)).repeat_interleave(
        config.manifolds_per_type
    )
    embeddings = []
    for manifold_id, type_id_tensor in enumerate(manifold_type_ids):
        type_id = int(type_id_tensor)
        embeddings.append(
            _orthonormal_embedding(
                config.ambient_dim,
                EMBEDDING_DIMS[type_id],
                generator=_generator(config.seed, 200 + manifold_id),
            )
        )

    embeddings_tuple = tuple(embeddings)
    offset_directions = _offset_directions(config, num_manifolds)
    offsets = float(config.offset_radius) * offset_directions

    train_dataset = _make_split(
        config.n_train,
        config=config,
        stream=400,
        means=means_tuple,
        scales=scales_tensor,
        embeddings=embeddings_tuple,
        offsets=offsets,
        manifold_type_ids=manifold_type_ids,
    )
    val_dataset = _make_split(
        config.n_val,
        config=config,
        stream=500,
        means=means_tuple,
        scales=scales_tensor,
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
                "type_name": MANIFOLD_NAMES[type_id],
                "intrinsic_dim": INTRINSIC_DIMS[type_id],
                "embedding_dim": EMBEDDING_DIMS[type_id],
                "position": offsets[manifold_id],
                "embedding": embeddings_tuple[manifold_id],
            }
        )
    metadata: dict[str, object] = {
        "config": asdict(config),
        "num_manifolds": num_manifolds,
        "manifold_types": MANIFOLD_NAMES,
        "type_id_to_name": dict(enumerate(MANIFOLD_NAMES)),
        "type_name_to_id": {
            name: type_id for type_id, name in enumerate(MANIFOLD_NAMES)
        },
        "intrinsic_dims": INTRINSIC_DIMS,
        "embedding_dims": EMBEDDING_DIMS,
        "manifold_type_ids": manifold_type_ids,
        "calibration_means": means_tuple,
        "calibration_scales": scales_tensor,
        "embeddings": embeddings_tuple,
        "offset_directions": offset_directions,
        "offsets": offsets,
        "manifolds": manifolds,
    }
    return train_dataset, val_dataset, metadata


__all__ = [
    "MANIFOLD_NAMES",
    "ToyManifoldConfig",
    "make_toy_manifold_datasets",
]
