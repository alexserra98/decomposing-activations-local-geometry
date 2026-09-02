"""Build a K=200 oracle MFA-HDDC for the 20K circle/helix dataset.

The two planted curves are recovered from ``manifold_metadata.pt``.  Each
curve's normalized parameter interval [0, 1] is split into 100 equal tiles,
and one factor analyzer is fit to the points in each tile.  This is a direct
construction: there is no optimizer or training loop.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch

from dalg.data.shard_activations import (
    load_meta_index,
    per_subset_counts,
    stratified_split,
)
from dalg.models.adaptive_q.mfa_hddc import (
    MFA_HDDC,
    load_mfa_hddc,
    save_mfa_hddc,
)


COMPONENTS_PER_MANIFOLD = 100
EXPECTED_MANIFOLD_NAMES = ("circle", "helix")
EXPECTED_POINTS_PER_MANIFOLD = 10_000
EXPECTED_NUM_POINTS = 20_000
LAYER = 0
EPS_FLOOR = 1e-12
MAX_PROJECTION_RESIDUAL_NOISE_STDS = 8.0
TANGENT_ALIGNMENT_TOL = 0.999
VAL_FRAC = 0.05
SPLIT_SEED = 42


def _inverse_softplus(value: torch.Tensor) -> torch.Tensor:
    if torch.any(value <= 0):
        raise ValueError("inverse softplus requires strictly positive values")
    return torch.log(torch.expm1(value))


def _check_output_dir(output_dir: Path) -> None:
    if not output_dir.exists():
        return
    if not output_dir.is_dir():
        raise FileExistsError(f"output path exists and is not a directory: {output_dir}")
    if any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")


def _load_source(
    shard_dir: Path,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any], dict[str, Any]]:
    config_path = shard_dir / "config.json"
    shard_config = json.loads(config_path.read_text())
    required_config = {
        "source_kind": "toy_manifolds",
        "window": 1,
        "d_model": 128,
        "drop_prefix": 0,
        "num_rows": EXPECTED_NUM_POINTS,
    }
    for key, expected in required_config.items():
        actual = shard_config.get(key)
        if actual != expected:
            raise ValueError(
                f"expected source config {key}={expected!r}, got {actual!r}"
            )
    if shard_config.get("layers") != [LAYER]:
        raise ValueError(f"expected source layer [{LAYER}], got {shard_config.get('layers')}")

    metadata_path = shard_dir / shard_config["manifold_metadata"]
    metadata = torch.load(
        metadata_path,
        map_location="cpu",
        weights_only=True,
    )
    names = tuple(metadata["manifold_types"])
    if names != EXPECTED_MANIFOLD_NAMES:
        raise ValueError(
            f"expected manifolds {EXPECTED_MANIFOLD_NAMES}, got {names}"
        )
    if int(metadata["num_manifolds"]) != len(EXPECTED_MANIFOLD_NAMES):
        raise ValueError("expected exactly one circle and one helix instance")

    layer_dir = shard_dir / f"layer{LAYER:02d}"
    shard_paths = sorted(layer_dir.glob("shard_*.pt"))
    if len(shard_paths) != int(shard_config["num_shards"]):
        raise ValueError(
            f"expected {shard_config['num_shards']} activation shards, "
            f"found {len(shard_paths)}"
        )
    points = torch.cat(
        [
            torch.load(path, map_location="cpu", mmap=True, weights_only=True)[:, 0]
            for path in shard_paths
        ],
        dim=0,
    ).float()
    manifold_ids = metadata["row_manifold_ids"].reshape(-1).long()
    if points.shape != (EXPECTED_NUM_POINTS, 128):
        raise ValueError(
            f"expected points shape ({EXPECTED_NUM_POINTS}, 128), "
            f"got {tuple(points.shape)}"
        )
    if manifold_ids.shape != (EXPECTED_NUM_POINTS,):
        raise ValueError(
            f"expected row_manifold_ids shape ({EXPECTED_NUM_POINTS},), "
            f"got {tuple(manifold_ids.shape)}"
        )
    counts = torch.bincount(manifold_ids, minlength=2)
    if counts.tolist() != [EXPECTED_POINTS_PER_MANIFOLD] * 2:
        raise ValueError(
            f"expected {EXPECTED_POINTS_PER_MANIFOLD} points per manifold, "
            f"got {counts.tolist()}"
        )
    return points, manifold_ids, metadata, shard_config


def _raw_curve(
    manifold_name: str,
    u: torch.Tensor,
    generator_config: dict[str, Any],
) -> torch.Tensor:
    if manifold_name == "circle":
        theta = 2.0 * math.pi * u
        return torch.stack((torch.cos(theta), torch.sin(theta)), dim=1)
    if manifold_name == "helix":
        theta_min = float(generator_config["helix_theta_min"])
        theta_max = float(generator_config["helix_theta_max"])
        alpha = float(generator_config["helix_alpha"])
        theta = theta_min + (theta_max - theta_min) * u
        return torch.stack(
            (torch.cos(theta), torch.sin(theta), alpha * theta),
            dim=1,
        )
    raise ValueError(f"unsupported manifold: {manifold_name}")


def parameter_to_ambient(
    manifold_id: int,
    u: torch.Tensor,
    metadata: dict[str, Any],
) -> torch.Tensor:
    """Map normalized curve parameters to the saved ambient-space manifold."""
    name = metadata["manifold_types"][manifold_id]
    raw = _raw_curve(name, u.double(), metadata["config"])
    normalized = (
        raw - metadata["calibration_means"][manifold_id].double()
    ) / metadata["calibration_scales"][manifold_id].double()
    return (
        normalized @ metadata["embeddings"][manifold_id].double()
        + metadata["offsets"][manifold_id].double()
    )


def ambient_to_parameter(
    manifold_id: int,
    points: torch.Tensor,
    metadata: dict[str, Any],
) -> torch.Tensor:
    """Invert the saved orthonormal embedding and recover normalized u."""
    normalized = (
        points.double() - metadata["offsets"][manifold_id].double()
    ) @ metadata["embeddings"][manifold_id].double().T
    raw = (
        normalized * metadata["calibration_scales"][manifold_id].double()
        + metadata["calibration_means"][manifold_id].double()
    )
    name = metadata["manifold_types"][manifold_id]
    if name == "circle":
        theta = torch.atan2(raw[:, 1], raw[:, 0]).remainder(2.0 * math.pi)
        return theta / (2.0 * math.pi)
    if name == "helix":
        config = metadata["config"]
        theta = raw[:, 2] / float(config["helix_alpha"])
        theta_min = float(config["helix_theta_min"])
        theta_max = float(config["helix_theta_max"])
        return (theta - theta_min) / (theta_max - theta_min)
    raise ValueError(f"unsupported manifold: {name}")


def _analytic_tangent(
    manifold_id: int,
    u: float,
    metadata: dict[str, Any],
) -> torch.Tensor:
    name = metadata["manifold_types"][manifold_id]
    config = metadata["config"]
    if name == "circle":
        theta = 2.0 * math.pi * u
        local = torch.tensor(
            [-math.sin(theta), math.cos(theta)], dtype=torch.float64
        )
    elif name == "helix":
        theta_min = float(config["helix_theta_min"])
        theta_max = float(config["helix_theta_max"])
        theta = theta_min + (theta_max - theta_min) * u
        local = torch.tensor(
            [-math.sin(theta), math.cos(theta), float(config["helix_alpha"])],
            dtype=torch.float64,
        )
    else:
        raise ValueError(f"unsupported manifold: {name}")
    ambient = (
        local / metadata["calibration_scales"][manifold_id].double()
    ) @ metadata["embeddings"][manifold_id].double()
    return ambient / ambient.norm()


def _recover_tiles(
    points: torch.Tensor,
    manifold_ids: torch.Tensor,
    metadata: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, float]:
    parameters = torch.empty(len(points), dtype=torch.float64)
    reconstructed = torch.empty_like(points, dtype=torch.float64)
    max_projection_residual = 0.0
    for manifold_id in range(2):
        mask = manifold_ids == manifold_id
        u = ambient_to_parameter(manifold_id, points[mask], metadata)
        if float(u.min()) < -1e-7 or float(u.max()) > 1.0 + 1e-7:
            raise ValueError(
                f"recovered {metadata['manifold_types'][manifold_id]} parameters "
                f"outside [0, 1]: [{float(u.min())}, {float(u.max())}]"
            )
        u = u.clamp(0.0, 1.0)
        parameters[mask] = u
        reconstructed[mask] = parameter_to_ambient(manifold_id, u, metadata)
        projection_residual = float(
            (reconstructed[mask] - points[mask].double()).abs().max()
        )
        noise_std = float(metadata["noise_stds"][manifold_id])
        residual_limit = MAX_PROJECTION_RESIDUAL_NOISE_STDS * noise_std
        if projection_residual >= residual_limit:
            name = metadata["manifold_types"][manifold_id]
            raise ValueError(
                f"{name} manifold projection residual {projection_residual:.3g} "
                f"exceeds {MAX_PROJECTION_RESIDUAL_NOISE_STDS:g} noise standard "
                f"deviations ({residual_limit:.3g})"
            )
        max_projection_residual = max(
            max_projection_residual, projection_residual
        )

    local_tiles = torch.floor(parameters * COMPONENTS_PER_MANIFOLD).long()
    local_tiles.clamp_(max=COMPONENTS_PER_MANIFOLD - 1)
    assignments = manifold_ids * COMPONENTS_PER_MANIFOLD + local_tiles
    return parameters, assignments, max_projection_residual


def _fit_tangent_components(
    points: torch.Tensor,
    assignments: torch.Tensor,
    metadata: dict[str, Any],
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    K = len(EXPECTED_MANIFOLD_NAMES) * COMPONENTS_PER_MANIFOLD
    D = points.shape[1]
    means = torch.empty(K, D, dtype=torch.float32)
    directions = torch.empty(K, D, rank, dtype=torch.float32)
    retained_variances = torch.empty(K, rank, dtype=torch.float64)
    tangent_alignments = torch.empty(K, dtype=torch.float64)

    for component_id in range(K):
        component_points = points[assignments == component_id].double()
        if len(component_points) <= rank:
            raise ValueError(
                f"component {component_id} needs more than {rank} points, got "
                f"{len(component_points)}"
            )
        mean = component_points.mean(dim=0)
        centered = component_points - mean
        _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
        component_directions = vh[:rank].T.contiguous()
        component_variances = singular_values[:rank].square() / len(component_points)

        manifold_id, local_tile = divmod(component_id, COMPONENTS_PER_MANIFOLD)
        midpoint = (local_tile + 0.5) / COMPONENTS_PER_MANIFOLD
        tangent = _analytic_tangent(manifold_id, midpoint, metadata)
        signed_alignment = torch.dot(component_directions[:, 0], tangent)
        if signed_alignment < 0:
            component_directions[:, 0] = -component_directions[:, 0]
            signed_alignment = -signed_alignment

        means[component_id] = mean.float()
        directions[component_id] = component_directions.float()
        retained_variances[component_id] = component_variances
        tangent_alignments[component_id] = signed_alignment

    minimum_alignment = float(tangent_alignments.min())
    if minimum_alignment <= TANGENT_ALIGNMENT_TOL:
        raise ValueError(
            f"minimum empirical-PC/tangent alignment {minimum_alignment:.6f} "
            f"does not exceed {TANGENT_ALIGNMENT_TOL}"
        )
    return means, directions, retained_variances, tangent_alignments


def _component_noise_variances(metadata: dict[str, Any]) -> torch.Tensor:
    """Return the planted ambient noise variance b_i for every component."""
    manifold_noise_stds = metadata["noise_stds"].double()
    if manifold_noise_stds.shape != (len(EXPECTED_MANIFOLD_NAMES),):
        raise ValueError(
            "expected one noise standard deviation for the circle and helix"
        )
    component_manifold_ids = torch.arange(
        len(EXPECTED_MANIFOLD_NAMES) * COMPONENTS_PER_MANIFOLD
    ).div(COMPONENTS_PER_MANIFOLD, rounding_mode="floor")
    return manifold_noise_stds.square()[component_manifold_ids]


def _build_model(
    means: torch.Tensor,
    directions: torch.Tensor,
    retained_variances: torch.Tensor,
    noise_variances: torch.Tensor,
) -> MFA_HDDC:
    below_noise = retained_variances <= noise_variances[:, None]
    if torch.any(below_noise):
        bad = torch.nonzero(below_noise).tolist()
        raise ValueError(
            "retained variance does not exceed dataset noise b_i for "
            f"(component, column) pairs {bad}"
        )
    psi_targets = noise_variances - EPS_FLOOR
    if torch.any(psi_targets <= 0.0):
        raise ValueError("EPS_FLOOR must be smaller than every dataset noise variance")

    model = MFA_HDDC(
        means,
        rank=directions.shape[2],
        psi_init=float(noise_variances.min()),
        isotropic_psi=True,
        eps_floor=EPS_FLOOR,
    )
    loading_scales = (retained_variances - noise_variances[:, None]).sqrt()
    with torch.no_grad():
        model.dir_raw.copy_(directions)
        model.scale_rho.copy_(
            _inverse_softplus(loading_scales).float()
        )
        model.psi_rho.copy_(
            _inverse_softplus(psi_targets).float()[:, None]
        )
        model.rank_mask.fill_(1.0)
        model.pi_logits.zero_()
    model.eval()
    return model


@torch.no_grad()
def _validate_model(
    model: MFA_HDDC,
    points: torch.Tensor,
    retained_variances: torch.Tensor,
    noise_variances: torch.Tensor,
) -> float:
    rank = retained_variances.shape[1]
    if (model.K, model.D, model.q) != (200, 128, rank):
        raise ValueError(
            f"unexpected model dimensions: K={model.K}, D={model.D}, q={model.q}"
        )
    if not model.isotropic_psi or model.psi_rho.shape != (model.K, 1):
        raise ValueError("oracle does not store one isotropic b_i per component")
    expected_ranks = torch.full((model.K,), rank, dtype=torch.long)
    if not torch.equal(model.component_ranks, expected_ranks):
        raise ValueError(f"oracle components are not all rank {rank}")

    fitted_noise_variances = model._psi()[:, 0].double()
    if not torch.allclose(
        fitted_noise_variances,
        noise_variances,
        rtol=1e-6,
        atol=1e-14,
    ):
        raise ValueError("oracle b_i does not match the dataset noise variance")
    fitted_retained_variances = (
        model._scale().double().square() + fitted_noise_variances[:, None]
    )
    if not torch.allclose(
        fitted_retained_variances,
        retained_variances,
        rtol=1e-6,
        atol=1e-12,
    ):
        raise ValueError("W W^T + b_i does not recover the retained variances")

    weights = torch.softmax(model.pi_logits, dim=0)
    expected_weights = torch.full_like(weights, 1.0 / model.K)
    if not torch.equal(weights, expected_weights):
        raise ValueError("mixture weights are not exactly uniform")
    total_nll = 0.0
    with model.inference_cache():
        for batch in points.split(1_024):
            log_prob = model.log_prob(batch)
            if not torch.isfinite(log_prob).all():
                raise ValueError("model produced a non-finite log likelihood")
            total_nll -= float(log_prob.sum())
    return total_nll / len(points)


def _save_split_info(
    output_dir: Path,
    shard_dir: Path,
    shard_config: dict[str, Any],
) -> None:
    """Record the standard shard split in the training-run schema."""
    meta_index = load_meta_index(shard_dir, layer=LAYER)
    train_positions, val_positions = stratified_split(
        meta_index,
        val_frac=VAL_FRAC,
        seed=SPLIT_SEED,
    )

    split_info = {
        "seed": SPLIT_SEED,
        "val_frac": VAL_FRAC,
        "per_row_tokens": int(shard_config["window"])
        - int(shard_config["drop_prefix"]),
        "train_rows": len(train_positions),
        "val_rows": len(val_positions),
        "train_per_subset": per_subset_counts(meta_index, train_positions),
        "val_per_subset": per_subset_counts(meta_index, val_positions),
        "val_global_rows": [
            meta_index[position]["global_row"] for position in val_positions
        ],
        "world_size": 1,
        "training_mode": "parametric_oracle",
        "component_shard": False,
        "split_kind": "stratified_by_subset",
    }
    (output_dir / "val_indices.json").write_text(
        json.dumps(split_info, indent=2) + "\n"
    )


def _save_outputs(
    output_dir: Path,
    shard_dir: Path,
    shard_config: dict[str, Any],
    model: MFA_HDDC,
    means: torch.Tensor,
    assignments: torch.Tensor,
    parameters: torch.Tensor,
    retained_variances: torch.Tensor,
    noise_variances: torch.Tensor,
    tangent_alignments: torch.Tensor,
    max_projection_residual: float,
    mean_nll: float,
) -> None:
    K = model.K
    cluster_sizes = torch.bincount(assignments, minlength=K)
    if int(cluster_sizes.sum()) != EXPECTED_NUM_POINTS or torch.any(cluster_sizes == 0):
        raise ValueError("oracle assignments do not define 200 nonempty tiles")

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(means, output_dir / "centroids.pt")
    _save_split_info(output_dir, shard_dir, shard_config)
    save_mfa_hddc(
        model,
        str(output_dir / "mfa_model.pt"),
        extra={
            "construction": "parametric_oracle_equal_parameter_tiles",
            "source_shard_dir": str(shard_dir.resolve()),
            "components_per_manifold": COMPONENTS_PER_MANIFOLD,
            "component_order": "circle_000_099_then_helix_100_199",
            "fit": "empirical_mle_mean_and_retained_covariance_eigenpairs",
            "b_i_source": "square of manifold_metadata.noise_stds",
            "b_i": {
                "circle": float(noise_variances[0]),
                "helix": float(noise_variances[COMPONENTS_PER_MANIFOLD]),
            },
            "mixture_weights": "uniform_1_over_200",
        },
    )

    tile_bounds = torch.stack(
        (
            torch.arange(K, dtype=torch.float64).remainder(
                COMPONENTS_PER_MANIFOLD
            )
            / COMPONENTS_PER_MANIFOLD,
            (
                torch.arange(K, dtype=torch.float64).remainder(
                    COMPONENTS_PER_MANIFOLD
                )
                + 1
            )
            / COMPONENTS_PER_MANIFOLD,
        ),
        dim=1,
    )
    component_manifold_ids = torch.arange(K).div(
        COMPONENTS_PER_MANIFOLD, rounding_mode="floor"
    )
    torch.save(
        {
            "cluster_sizes": cluster_sizes,
            "assignments": assignments,
            "K": K,
            "subset_spec": None,
            "normalized_parameters": parameters,
            "tile_bounds": tile_bounds,
            "component_manifold_ids": component_manifold_ids,
            "source": {
                "kind": "parametric_equal_parameter_tiles",
                "shard_dir": str(shard_dir.resolve()),
                "layer": LAYER,
                "drop_prefix": int(shard_config["drop_prefix"]),
                "num_items": int(assignments.numel()),
                "canonical_order": "generated",
                "component_order": "circle_000_099_then_helix_100_199",
            },
        },
        output_dir / "oracle_tile_assignments.pt",
    )

    run_config = {
        "model": "MFA_HDDC",
        "construction": "parametric_oracle_equal_parameter_tiles",
        "K": K,
        "rank": model.q,
        "shard_dir": str(shard_dir.resolve()),
        "layer": LAYER,
        "window": int(shard_config["window"]),
        "d_model": int(shard_config["d_model"]),
        "drop_prefix": int(shard_config["drop_prefix"]),
        "num_points": EXPECTED_NUM_POINTS,
        "components_per_manifold": {
            "circle": COMPONENTS_PER_MANIFOLD,
            "helix": COMPONENTS_PER_MANIFOLD,
        },
        "component_order": {
            "circle": [0, 99],
            "helix": [100, 199],
        },
        "parameterization": {
            "normalized_interval": [0.0, 1.0],
            "tile_width": 1.0 / COMPONENTS_PER_MANIFOLD,
            "circle": "theta = 2*pi*u",
            "helix": "theta = 4*pi*u; raw = (cos(theta), sin(theta), 0.2*theta)",
            "ambient": "((raw - calibration_mean) / calibration_scale) @ embedding + offset",
        },
        "fit": {
            "moments": "existing_points_in_each_parameter_tile",
            "covariance_normalization": "maximum_likelihood_1_over_n",
            "retained_eigenpairs": model.q,
            "b_i": {
                "circle": float(noise_variances[0]),
                "helix": float(noise_variances[COMPONENTS_PER_MANIFOLD]),
            },
            "b_i_source": "square of manifold_metadata.noise_stds",
            "psi_kind": "per_component_isotropic_dataset_noise_variance",
            "loading_scale": "sqrt(empirical_retained_variance - b_i)",
            "eps_floor": EPS_FLOOR,
            "mixture_weights": "uniform_1_over_200",
        },
        "oracle_cluster_size_range": [
            int(cluster_sizes.min()),
            int(cluster_sizes.max()),
        ],
        "leading_variance_range": [
            float(retained_variances[:, 0].min()),
            float(retained_variances[:, 0].max()),
        ],
        "retained_variance_range": [
            float(retained_variances.min()),
            float(retained_variances.max()),
        ],
        "validation": {
            "manifold_projection_max_abs_residual": max_projection_residual,
            "projection_residual_limit_in_noise_stds": (
                MAX_PROJECTION_RESIDUAL_NOISE_STDS
            ),
            "minimum_tangent_alignment": float(tangent_alignments.min()),
            "mean_tangent_alignment": float(tangent_alignments.mean()),
            "mean_nll": mean_nll,
        },
    }
    (output_dir / "config.json").write_text(
        json.dumps(run_config, indent=2) + "\n"
    )


def build_oracle_hddc(shard_dir: Path, output_dir: Path, rank: int) -> None:
    _check_output_dir(output_dir)
    points, manifold_ids, metadata, shard_config = _load_source(shard_dir)
    parameters, assignments, max_projection_residual = _recover_tiles(
        points, manifold_ids, metadata
    )
    cluster_sizes = torch.bincount(assignments, minlength=200)
    if torch.any(cluster_sizes == 0):
        raise ValueError("at least one of the 200 parameter tiles is empty")

    means, directions, retained_variances, tangent_alignments = (
        _fit_tangent_components(points, assignments, metadata, rank)
    )
    noise_variances = _component_noise_variances(metadata)
    model = _build_model(
        means, directions, retained_variances, noise_variances
    )
    mean_nll = _validate_model(
        model, points, retained_variances, noise_variances
    )
    _save_outputs(
        output_dir,
        shard_dir,
        shard_config,
        model,
        means,
        assignments,
        parameters,
        retained_variances,
        noise_variances,
        tangent_alignments,
        max_projection_residual,
        mean_nll,
    )

    loaded = load_mfa_hddc(output_dir / "mfa_model.pt", map_location="cpu")
    reloaded_nll = _validate_model(
        loaded, points, retained_variances, noise_variances
    )
    if not math.isclose(mean_nll, reloaded_nll, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(
            f"round-trip checkpoint NLL changed from {mean_nll} to {reloaded_nll}"
        )

    print(f"Oracle MFA-HDDC saved to {output_dir}")
    print(
        f"K=200 D=128 q={rank}  tile sizes={int(cluster_sizes.min())}.."
        f"{int(cluster_sizes.max())}  mean NLL={mean_nll:.6f}"
    )
    print(
        f"manifold projection max residual={max_projection_residual:.3e}  "
        f"minimum tangent alignment={float(tangent_alignments.min()):.8f}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rank",
        type=int,
        default=1,
        help="Number of empirical covariance eigenpairs retained per tile.",
    )
    parser.add_argument(
        "--shard-dir",
        type=Path,
        required=True,
        help="The strict 20K circle/helix activation-shard directory.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="New or empty output directory for the oracle MFA-HDDC run.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    build_oracle_hddc(args.shard_dir, args.out_dir, args.rank)


if __name__ == "__main__":
    main()
