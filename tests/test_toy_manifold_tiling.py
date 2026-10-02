from __future__ import annotations

import json
import math
from itertools import combinations
from pathlib import Path

import pytest
import torch

from dalg.analysis.bic_improved import compute_improved_bic_details
from dalg.data.manifold_dataset import (
    MANIFOLD_NAMES,
    ToyManifoldConfig,
    make_toy_manifold_dataset,
    save_toy_manifold_shards,
)
from dalg.data.shard_activations import load_meta_index
from dalg.evaluation.toy_manifold_geometry import _project_mean_to_manifold
from dalg.evaluation.toy_manifold_metrics import (
    _best_subset_alignment,
    _leading_covariance_eigenspaces,
    _subspace_alignment,
    evaluate_toy_manifold_metrics,
)
from dalg.evaluation.toy_manifold_tiling import evaluate_toy_manifold_tiling
from dalg.models.adaptive_q.mfa_ard import MFA_ARD, save_mfa_ard
from dalg.models.adaptive_q.mfa_hddc import MFA_HDDC, save_mfa_hddc
from dalg.models.kmeans import KMeans
from dalg.models.mfa import MFA, save_mfa


@pytest.mark.parametrize("intrinsic_dim,rank", [(1, 1), (1, 3), (2, 2), (2, 4), (3, 5)])
@pytest.mark.parametrize("seed", [12, 42])
def test_best_subset_matches_exhaustive_search(intrinsic_dim, rank, seed):
    generator = torch.Generator().manual_seed(seed)
    tangent, _ = torch.linalg.qr(torch.randn(7, intrinsic_dim, generator=generator, dtype=torch.float64))
    principal, _ = torch.linalg.qr(torch.randn(7, rank, generator=generator, dtype=torch.float64))
    candidates = []
    for subset in combinations(range(rank), intrinsic_dim):
        cosines = torch.linalg.svdvals(tangent.T @ principal[:, list(subset)])
        candidates.append((float(cosines.square().mean()), float(cosines.min())))
    expected = max(candidates, key=lambda scores: scores[0])
    assert _best_subset_alignment(tangent, principal) == pytest.approx(expected)

    rotation, _ = torch.linalg.qr(torch.randn(
        intrinsic_dim, intrinsic_dim, generator=generator, dtype=torch.float64,
    ))
    signs = torch.where(torch.arange(rank) % 2 == 0, -1.0, 1.0)
    assert _best_subset_alignment(tangent @ rotation, principal * signs) == pytest.approx(expected)


def test_best_subset_breaks_contribution_ties_by_pc_index():
    # Every PC contributes 1/2; the first two span only one tangent direction.
    tangent = 0.5 * torch.tensor([
        [1.0, 1.0], [1.0, 1.0], [1.0, -1.0], [1.0, -1.0],
    ], dtype=torch.float64)
    principal = torch.eye(4, dtype=torch.float64)
    overlap, worst = _best_subset_alignment(tangent, principal)
    assert overlap == pytest.approx(0.5)
    assert worst == pytest.approx(0.0, abs=1e-12)
    # Another equally good overlap has a better worst cosine, but loses the tie.
    assert _subspace_alignment(tangent, principal[:, [0, 2]]) == pytest.approx((0.5, math.sqrt(0.5)))


def test_best_subset_optimizes_overlap_not_worst_direction():
    tangent = torch.tensor([
        [math.sqrt(0.6), 0.0], [math.sqrt(0.4), 0.0],
        [0.0, math.sqrt(0.35)], [0.0, math.sqrt(0.33)], [0.0, math.sqrt(0.32)],
    ], dtype=torch.float64)
    assert _best_subset_alignment(tangent, torch.eye(5, dtype=torch.float64)) == pytest.approx((0.5, 0.0))


@pytest.mark.parametrize("model_class", [MFA, MFA_ARD, MFA_HDDC, KMeans])
@pytest.mark.parametrize("case,rank,alignment,containment,worst", [
    ("later_pcs", 3, 0.5, 1.0, 1.0),
    ("spread_tangent", 3, 0.75, 0.75, math.sqrt(0.5)),
    ("matched_rank", 2, 0.75, 0.75, math.sqrt(0.5)),
])
def test_containment_selects_best_intrinsic_size_pc_span(
    model_class, case, rank, alignment, containment, worst,
):
    _, metadata = make_toy_manifold_dataset(ToyManifoldConfig(
        ambient_dim=4, n_samples=20, calibration_size=20,
        manifold_types=("flat_disk",), manifolds_per_type=1, noise_ratio=None,
    ))
    manifold = metadata["manifolds"][0]
    projection = _project_mean_to_manifold(manifold["position"], manifold, metadata)
    basis, _ = torch.linalg.qr(projection.tangent, mode="complete")
    if case == "later_pcs":
        principal = basis[:, [2, 0, 1, 3]]
    else:
        principal = basis.clone()
        principal[:, 1] = (basis[:, 1] + basis[:, 2]) / math.sqrt(2.0)
        principal[:, 2] = (basis[:, 1] - basis[:, 2]) / math.sqrt(2.0)
    scales = torch.tensor([3.0, 2.0, 1.0, 0.1], dtype=torch.float64)
    if rank == 2:
        scales[2:] = torch.tensor([0.1, 0.05])
    if model_class is KMeans:
        model = KMeans.from_centroids(projection.point[None])
        offsets = (principal * scales).T
        points = projection.point + torch.cat((offsets, -offsets))
        model.compute_pcs(points, threshold=0.1)
    else:
        model = model_class(
            projection.point[None], rank=rank, psi_init=0.5,
            init_directions=principal[None, :, :rank],
        )
        with torch.no_grad():
            model.scale_rho[0].copy_(scales[:rank].expm1().log())
    result = evaluate_toy_manifold_metrics(model, metadata, torch.ones(1, dtype=torch.bool))
    assert result["rank"]["mean_learned"] == rank
    basis_name = "pca" if model_class is KMeans else "covariance"
    assert result["tangent_containment"]["definition"] == (
        f"best_intrinsic_dim_subset_of_leading_rank_{basis_name}_principal_angles"
    )
    for summary in (result, result["per_manifold"][0]):
        assert summary["tangent_alignment"]["subspace_overlap"]["mean"] == pytest.approx(alignment, abs=1e-6)
        assert summary["tangent_adjusted_alignment"]["subspace_overlap"] == {
            "mean": pytest.approx(alignment, abs=1e-6),
            "valid_components": 1, "undefined_components": 0,
        }
        for name, expected in (("subspace_overlap", containment), ("worst_direction_cosine", worst)):
            assert summary["tangent_containment"][name] == {
                "mean": pytest.approx(expected, abs=1e-6),
                "valid_components": 1, "undefined_components": 0,
            }
        if rank == 2:
            assert summary["tangent_containment"]["worst_direction_cosine"]["mean"] == pytest.approx(
                summary["tangent_alignment"]["worst_direction_cosine"]["mean"], abs=1e-6,
            )
    if case == "spread_tangent":
        _, candidates = _leading_covariance_eigenspaces(model, rank)
        assert _subspace_alignment(projection.tangent, candidates[0])[0] == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("reason", ["eigengap", "projection"])
def test_full_containment_preserves_geometry_validity_checks(reason):
    manifold_type = "sphere" if reason == "projection" else "flat_disk"
    _, metadata = make_toy_manifold_dataset(ToyManifoldConfig(
        ambient_dim=3, n_samples=20, calibration_size=20,
        manifold_types=(manifold_type,), manifolds_per_type=1, noise_ratio=None,
    ))
    manifold = metadata["manifolds"][0]
    mean = manifold["position"]
    if reason == "projection":
        mean = (-metadata["calibration_means"][0] / metadata["calibration_scales"][0]) @ manifold["embedding"] + mean
    model = MFA_HDDC(
        mean[None], rank=2, scale_init=2.0 if reason == "projection" else 1e-8,
        psi_init=0.5,
    )
    result = evaluate_toy_manifold_metrics(model, metadata, torch.ones(1, dtype=torch.bool))
    assert result["rank"]["mean_learned"] == 2.0
    for summary in (result, result["per_manifold"][0]):
        assert summary["tangent_adjusted_alignment"]["subspace_overlap"] == {
            "mean": None, "valid_components": 0, "undefined_components": 1,
        }
        for name in ("subspace_overlap", "worst_direction_cosine"):
            assert summary["tangent_containment"][name] == {
                "mean": None, "valid_components": 0, "undefined_components": 1,
            }


def test_containment_accepts_tied_eigenvalues_inside_candidate_subspace():
    _, metadata = make_toy_manifold_dataset(ToyManifoldConfig(
        ambient_dim=3, n_samples=20, calibration_size=20,
        manifold_types=("segment",), manifolds_per_type=1, noise_ratio=None,
    ))
    manifold = metadata["manifolds"][0]
    projection = _project_mean_to_manifold(manifold["position"], manifold, metadata)
    basis, _ = torch.linalg.qr(projection.tangent, mode="complete")
    model = MFA_HDDC(
        projection.point[None], rank=2, init_directions=basis[None, :, :2],
        scale_init=2.0, psi_init=0.5,
    )
    result = evaluate_toy_manifold_metrics(model, metadata, torch.ones(1, dtype=torch.bool))
    for summary in (result, result["per_manifold"][0]):
        assert summary["tangent_alignment"]["subspace_overlap"]["mean"] is None
        assert summary["tangent_adjusted_alignment"]["subspace_overlap"] == {
            "mean": None, "valid_components": 0, "undefined_components": 1,
        }
        for score in ("subspace_overlap", "worst_direction_cosine"):
            assert summary["tangent_containment"][score]["valid_components"] == 1
            assert summary["tangent_containment"][score]["undefined_components"] == 0
            assert 0.0 <= summary["tangent_containment"][score]["mean"] <= 1.0


@pytest.mark.parametrize("model_class", [MFA, MFA_ARD, MFA_HDDC])
@pytest.mark.parametrize("ranks", [(0,), (1,), (2,), (3,), (0, 1, 2, 3)])
def test_tangent_metrics_require_rank_at_least_intrinsic_dim(model_class, ranks):
    _, metadata = make_toy_manifold_dataset(ToyManifoldConfig(
        ambient_dim=3, n_samples=20, calibration_size=20,
        manifold_types=("flat_disk",), manifolds_per_type=1, noise_ratio=None,
    ))
    manifold = metadata["manifolds"][0]
    projection = _project_mean_to_manifold(manifold["position"], manifold, metadata)
    basis, _ = torch.linalg.qr(projection.tangent, mode="complete")
    model = model_class(
        projection.point.repeat(len(ranks), 1), rank=3, psi_init=0.5,
        init_directions=basis.repeat(len(ranks), 1, 1),
    )
    with torch.no_grad():
        for k, rank in enumerate(ranks):
            scales = torch.tensor([3.0, 2.0, 1.0], dtype=model.mu.dtype)
            scales[rank:] = 0.1
            model.scale_rho[k].copy_(scales.expm1().log())
            if isinstance(model, MFA_HDDC):
                model.rank_mask[k] = torch.arange(3) < rank

    result = evaluate_toy_manifold_metrics(
        model, metadata, torch.zeros(len(ranks), dtype=torch.bool),
    )
    assert result["rank"]["components"] == len(ranks)
    assert result["rank"]["mean_learned"] == pytest.approx(sum(ranks) / len(ranks))
    for summary in (result, result["per_manifold"][0]):
        assert summary["tangent_adjusted_alignment"]["subspace_overlap"] == {
            "mean": pytest.approx(sum(min(rank, 2) / 2 for rank in ranks) / len(ranks), abs=1e-6),
            "valid_components": len(ranks), "undefined_components": 0,
        }
        for metric_name, valid in (
            ("tangent_alignment", sum(rank >= 2 for rank in ranks)),
            ("tangent_containment", sum(rank >= 2 for rank in ranks)),
            ("tangent_partial_containment", sum(0 < rank < 2 for rank in ranks)),
        ):
            for score_name in ("subspace_overlap", "worst_direction_cosine"):
                score = summary[metric_name][score_name]
                assert score["valid_components"] == valid
                assert score["undefined_components"] == len(ranks) - valid
                if valid:
                    assert score["mean"] == pytest.approx(1.0, abs=1e-6)
                else:
                    assert score["mean"] is None
                    assert json.loads(json.dumps(score))["mean"] is None


@pytest.mark.parametrize("model_class", [MFA, MFA_ARD, MFA_HDDC, KMeans])
@pytest.mark.parametrize("manifold_type,rank,angle,overlap,worst", [
    ("flat_disk", 1, 0.0, 1.0, 1.0),
    ("flat_disk", 1, math.pi / 4, 0.5, math.sqrt(0.5)),
    ("flat_disk", 1, math.pi / 2, 0.0, 0.0),
    ("hypersphere_10d", 2, math.pi / 2, 0.5, 0.0),
])
def test_partial_containment_scores_learned_directions(
    model_class, manifold_type, rank, angle, overlap, worst,
):
    dataset, metadata = make_toy_manifold_dataset(ToyManifoldConfig(
        ambient_dim=11, n_samples=20, calibration_size=20,
        manifold_types=(manifold_type,), manifolds_per_type=1, noise_ratio=None,
    ))
    manifold = metadata["manifolds"][0]
    projection = _project_mean_to_manifold(dataset.tensors[0][0], manifold, metadata)
    basis, _ = torch.linalg.qr(projection.tangent, mode="complete")
    directions = basis[:, :rank].clone()
    directions[:, -1] = (
        math.cos(angle) * basis[:, rank - 1]
        + math.sin(angle) * basis[:, manifold["intrinsic_dim"]]
    )
    if model_class is KMeans:
        model = KMeans.from_centroids(projection.point[None])
        points = projection.point + torch.cat((directions.T, -directions.T))
        model.compute_pcs(points, threshold=0.1)
    else:
        model = model_class(
            projection.point[None], rank=rank, init_directions=directions[None],
            scale_init=2.0, psi_init=0.5,
        )
    result = evaluate_toy_manifold_metrics(model, metadata, torch.ones(1, dtype=torch.bool))
    for summary in (result, result["per_manifold"][0]):
        assert summary["tangent_adjusted_alignment"]["subspace_overlap"] == {
            "mean": pytest.approx(rank / manifold["intrinsic_dim"] * overlap, abs=1e-6),
            "valid_components": 1, "undefined_components": 0,
        }
        metric = summary["tangent_partial_containment"]
        for score_name, expected in (("subspace_overlap", overlap), ("worst_direction_cosine", worst)):
            assert metric[score_name] == {
                "mean": pytest.approx(expected, abs=1e-6),
                "valid_components": 1, "undefined_components": 0,
            }
        assert summary["tangent_alignment"]["subspace_overlap"]["mean"] is None
        assert summary["tangent_containment"]["subspace_overlap"]["mean"] is None


@pytest.mark.parametrize("reason", ["eigengap", "projection"])
def test_partial_containment_requires_identifiable_geometry(reason):
    manifold_type = "sphere" if reason == "projection" else "flat_disk"
    _, metadata = make_toy_manifold_dataset(ToyManifoldConfig(
        ambient_dim=3, n_samples=20, calibration_size=20,
        manifold_types=(manifold_type,), manifolds_per_type=1, noise_ratio=None,
    ))
    manifold = metadata["manifolds"][0]
    mean = manifold["position"]
    if reason == "projection":
        mean = (
            -metadata["calibration_means"][0] / metadata["calibration_scales"][0]
        ) @ manifold["embedding"] + mean
        assert not _project_mean_to_manifold(mean, manifold, metadata).unique
    model = MFA_HDDC(mean[None], rank=1, scale_init=1e-8, psi_init=0.5)
    result = evaluate_toy_manifold_metrics(model, metadata, torch.ones(1, dtype=torch.bool))
    assert result["association"]["associated_components"] == 1
    assert result["rank"]["mean_learned"] == 1.0
    for summary in (result, result["per_manifold"][0]):
        assert summary["tangent_adjusted_alignment"]["subspace_overlap"] == {
            "mean": None, "valid_components": 0, "undefined_components": 1,
        }
        for score in ("subspace_overlap", "worst_direction_cosine"):
            assert summary["tangent_partial_containment"][score] == {
                "mean": None, "valid_components": 0, "undefined_components": 1,
            }


@pytest.mark.parametrize("model_class", [MFA, MFA_ARD, MFA_HDDC])
def test_adjusted_alignment_averages_component_scores_before_manifolds(model_class):
    _, metadata = make_toy_manifold_dataset(ToyManifoldConfig(
        ambient_dim=11, n_samples=40, calibration_size=20, offset_radius=50,
        manifold_types=("hypersphere_10d",), manifolds_per_type=2, noise_ratio=None,
    ))
    projections = [
        _project_mean_to_manifold(m["position"] + m["embedding"][0], m, metadata)
        for m in metadata["manifolds"]
    ]
    bases = [torch.linalg.qr(p.tangent, mode="complete")[0] for p in projections]
    means = torch.stack([projections[i].point for i in (0, 0, 0, 1)])
    directions = torch.stack([bases[i][:, :10] for i in (0, 0, 0, 1)])
    directions[2, :, 1] = bases[0][:, 10]  # One of two learned directions is normal.
    ranks = [0, 1, 2, 10]
    model = model_class(means, rank=10, init_directions=directions, psi_init=0.5)
    with torch.no_grad():
        for k, rank in enumerate(ranks):
            scales = torch.linspace(3.0, 1.0, 10, dtype=model.mu.dtype)
            scales[rank:] = 0.1
            model.scale_rho[k].copy_(scales.expm1().log())
            if isinstance(model, MFA_HDDC):
                model.rank_mask[k] = torch.arange(10) < rank
    result = evaluate_toy_manifold_metrics(
        model, metadata, torch.tensor([False, True, False, True]),
    )
    # Scores are [0, .1, .1, 1]. Assignment-dead components still contribute.
    for summary, mean, count in zip(
        (result, *result["per_manifold"]), (0.3, 0.2 / 3, 1.0), (4, 3, 1),
    ):
        assert summary["tangent_adjusted_alignment"]["subspace_overlap"] == {
            "mean": pytest.approx(mean, abs=1e-6),
            "valid_components": count, "undefined_components": 0,
        }
    assert result["per_manifold"][0]["tangent_partial_containment"]["subspace_overlap"]["mean"] == pytest.approx(0.75, abs=1e-6)


@pytest.mark.parametrize("case", ["undefined_tangent", "ambiguous", "outside_cutoff"])
@pytest.mark.parametrize("rank", [0, 1])
def test_adjusted_alignment_requires_association_and_tangent(case, rank):
    _, metadata = make_toy_manifold_dataset(ToyManifoldConfig(
        ambient_dim=3, n_samples=20, calibration_size=20,
        manifold_types=("sphere",), manifolds_per_type=1, noise_ratio=None,
    ))
    manifold = metadata["manifolds"][0]
    center = (-metadata["calibration_means"][0] / metadata["calibration_scales"][0]) @ manifold["embedding"] + manifold["position"]
    mean = center if case == "undefined_tangent" else center + 10 * manifold["embedding"][0]
    if case == "ambiguous":
        metadata["manifolds"].append({**manifold, "manifold_id": 1})
        metadata["num_manifolds"] = 2
    model = MFA_HDDC(mean[None], rank=1, scale_init=2.0, psi_init=0.5)
    with torch.no_grad():
        model.rank_mask[:] = bool(rank)
    result = evaluate_toy_manifold_metrics(
        model, metadata, torch.ones(1, dtype=torch.bool),
        max_mean_to_manifold_distance=0.01 if case == "outside_cutoff" else None,
    )
    association_key = {
        "undefined_tangent": "associated_components",
        "ambiguous": "ambiguous_components",
        "outside_cutoff": "outside_cutoff_components",
    }[case]
    assert result["association"][association_key] == 1
    for summary in (result, *result["per_manifold"]):
        score = summary["tangent_adjusted_alignment"]["subspace_overlap"]
        assert score == {
            "mean": None, "valid_components": 0,
            "undefined_components": int(case == "undefined_tangent"),
        }
        assert json.loads(json.dumps(score, allow_nan=False))["mean"] is None


def _save_model(model_kind: str, centroids: torch.Tensor, path: Path) -> None:
    if model_kind == "mfa":
        save_mfa(MFA(centroids, rank=2), str(path))
    elif model_kind == "ard":
        save_mfa_ard(MFA_ARD(centroids, rank=2), path)
    elif model_kind == "hddc":
        save_mfa_hddc(
            MFA_HDDC(centroids, rank=2, isotropic_psi=True),
            str(path),
        )
    else:
        raise AssertionError(f"unexpected model kind: {model_kind}")


def _build_evaluation_artifacts(tmp_path: Path, model_kind: str) -> tuple[Path, Path]:
    shard_dir = save_toy_manifold_shards(
        tmp_path / "toy_shards",
        ToyManifoldConfig(
            ambient_dim=32,
            n_samples=96,
            calibration_size=32,
            manifolds_per_type=1,
            offset_radius=3.0,
            seed=0,
        ),
        shard_size=24,
        layer=0,
    )
    metadata = torch.load(
        shard_dir / "manifold_metadata.pt",
        map_location="cpu",
        weights_only=True,
    )
    centroids = torch.stack(
        [
            _project_mean_to_manifold(
                manifold["position"],
                manifold,
                metadata,
            ).point.float()
            for manifold in metadata["manifolds"]
        ]
    )

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _save_model(model_kind, centroids, run_dir / "mfa_model.pt")
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "model_kind": model_kind,
                "shard_dir": str(shard_dir),
                "layer": 0,
                "window": 1,
                "drop_prefix": 0,
            }
        )
    )

    meta_index = load_meta_index(shard_dir, layer=0)
    val_positions = list(range(0, len(meta_index), 4))
    val_rows = [meta_index[position]["global_row"] for position in val_positions]
    (run_dir / "val_indices.json").write_text(
        json.dumps(
            {
                "train_rows": len(meta_index) - len(val_positions),
                "val_rows": len(val_positions),
                "val_global_rows": val_rows,
            }
        )
    )
    assignments = metadata["row_manifold_ids"].long()
    torch.save(
        {
            "K": len(centroids),
            "assignments": assignments,
            "cluster_sizes": torch.bincount(assignments, minlength=len(centroids)),
            "subset_spec": None,
        },
        run_dir / "mfa_model_assignments.pt",
    )
    return run_dir, shard_dir


@pytest.mark.parametrize("model_kind", ["mfa", "ard", "hddc"])
def test_toy_manifold_tiling_evaluation_supports_all_model_kinds(
    tmp_path: Path,
    model_kind: str,
) -> None:
    run_dir, shard_dir = _build_evaluation_artifacts(tmp_path, model_kind)

    metrics = evaluate_toy_manifold_tiling(
        run_dir,
        shard_dir=shard_dir,
        layer=0,
        model_kind=model_kind,
        batch_size=16,
        device="cpu",
        max_mean_to_manifold_distance=0.1,
    )

    assert metrics["evaluation"] == "toy_manifold_tiling"
    assert metrics["model_kind"] == model_kind
    assert metrics["K"] == len(MANIFOLD_NAMES)
    assert metrics["components"]["dead"] == 0
    assert metrics["association"] == {
        "rule": "unique_nearest_exact_projection_within_cutoff",
        "max_mean_to_manifold_distance": 0.1,
        "associated_components": metrics["K"],
        "outside_cutoff_components": 0,
        "ambiguous_components": 0,
    }
    assert len(metrics["per_manifold"]) == metrics["K"]
    assert all(
        manifold["components"]["associated"] == 1
        for manifold in metrics["per_manifold"]
    )
    assert metrics["rank"]["population"] == "proximity_associated_components"
    assert metrics["rank"]["components"] == metrics["K"]
    assert metrics["ambient_rank"]["population"] == (
        "proximity_associated_components"
    )
    assert metrics["ambient_rank"]["components"] == metrics["K"]
    assert all(
        manifold["ambient_rank"]["target_ambient_dim"]
        == manifold["embedding_dim"]
        for manifold in metrics["per_manifold"]
    )
    assert metrics["tangent_alignment"]["definition"] == (
        "leading_intrinsic_dim_covariance_subspace_principal_angles"
    )
    assert metrics["tangent_containment"]["definition"] == (
        "best_intrinsic_dim_subset_of_leading_rank_covariance_principal_angles"
    )
    if model_kind == "hddc":
        for rank_name in ("rank", "ambient_rank"):
            assert metrics[rank_name]["definition"] == "hddc_rank_mask_count"
            assert "threshold" not in metrics[rank_name]
            assert metrics[rank_name]["mean_learned"] == 2.0
    assert metrics["tangent_partial_containment"]["normalization"] == "effective_rank"
    adjusted = metrics["tangent_adjusted_alignment"]
    assert adjusted["definition"] == "leading_min_intrinsic_effective_rank_covariance_subspace_overlap"
    assert adjusted["normalization"] == "intrinsic_dim"
    assert adjusted["rank_requirement"] == "effective_rank_gte_zero"
    assert adjusted["zero_rank"] == "zero_if_tangent_defined"
    assert adjusted["aggregation"] == "unweighted_component_mean"
    assert adjusted["relative_boundary_eigengap_threshold"] == 1e-6
    assert "worst_direction_cosine" not in adjusted
    for summary in (metrics, *metrics["per_manifold"]):
        score = summary["tangent_adjusted_alignment"]["subspace_overlap"]
        associated = (summary["components"]["associated"] if "manifold_id" in summary
                      else summary["association"]["associated_components"])
        assert score["valid_components"] + score["undefined_components"] == associated
        assert score["mean"] is None or 0 <= score["mean"] <= 1
    for metric_name, requirement in (
        ("tangent_alignment", "effective_rank_gte_intrinsic_dim"),
        ("tangent_containment", "effective_rank_gte_intrinsic_dim"),
        ("tangent_partial_containment", "effective_rank_gt_zero_lt_intrinsic_dim"),
    ):
        assert metrics[metric_name]["rank_requirement"] == requirement
        for score_name in ("subspace_overlap", "worst_direction_cosine"):
            summary = metrics[metric_name][score_name]
            assert summary["valid_components"] + summary["undefined_components"] == (
                metrics["K"]
            )
            if summary["mean"] is not None:
                assert 0.0 <= summary["mean"] <= 1.0
    assert metrics["clustering"]["adjusted_rand_index"] == 1.0
    assert torch.isfinite(torch.tensor(metrics["nll"]["train"]))
    assert torch.isfinite(torch.tensor(metrics["nll"]["validation"]))
    assert metrics["schema_version"] == 2
    assert metrics["bic"]["n"] == metrics["dataset"]["train_rows"]
    assert metrics["bic"]["parameters"] > 0
    assert metrics["bic"]["split"] == "train"
    assert metrics["bic"]["convention"] == "higher_is_better"
    assert metrics["bic"]["formula"] == "-standard_bic / n + active_components"
    assert metrics["bic"]["assignment_rule"] == "hard_map_count_greater_than_zero"
    assert metrics["bic"]["active_components"] == metrics["K"]
    assert metrics["bic"]["inactive_components"] == 0
    assert metrics["bic"]["standard_bic"] == pytest.approx(
        2.0 * metrics["bic"]["n"] * metrics["nll"]["train"]
        + metrics["bic"]["parameters"] * math.log(metrics["bic"]["n"])
    )
    assert metrics["bic"]["value"] == pytest.approx(
        -metrics["bic"]["standard_bic"] / metrics["bic"]["n"] + metrics["K"]
    )


@pytest.mark.parametrize("model_kind", ["mfa", "ard", "hddc"])
def test_toy_manifold_tiling_augmented_bic_excludes_validation_activity(
    tmp_path: Path,
    model_kind: str,
) -> None:
    run_dir, shard_dir = _build_evaluation_artifacts(tmp_path, model_kind)
    split = json.loads((run_dir / "val_indices.json").read_text())
    validation_rows = set(split["val_global_rows"])
    meta_index = load_meta_index(shard_dir, layer=0)
    assignments = torch.tensor(
        [int(row["global_row"] in validation_rows) for row in meta_index]
    )
    assignments_path = run_dir / "custom_assignments.pt"
    torch.save(
        {
            "K": len(MANIFOLD_NAMES),
            "assignments": assignments,
            "cluster_sizes": torch.bincount(assignments, minlength=len(MANIFOLD_NAMES)),
            "subset_spec": None,
        },
        assignments_path,
    )

    metrics = evaluate_toy_manifold_tiling(
        run_dir,
        shard_dir=shard_dir,
        layer=0,
        model_kind=model_kind,
        assignments_path=assignments_path,
        batch_size=16,
        device="cpu",
    )
    standalone = compute_improved_bic_details(
        run_dir, assignments_path=assignments_path, batch_size=16
    )

    assert metrics["components"]["live"] == 2
    assert metrics["bic"]["active_components"] == 1
    assert metrics["bic"]["inactive_components"] == metrics["K"] - 1
    assert metrics["bic"]["n"] == split["train_rows"]
    assert metrics["bic"]["value"] == pytest.approx(
        -metrics["bic"]["standard_bic"] / split["train_rows"] + 1
    )
    for key, value in standalone.items():
        if isinstance(value, float):
            assert metrics["bic"][key] == pytest.approx(value)
        else:
            assert metrics["bic"][key] == value


@pytest.mark.parametrize("distance", [0.0, -0.1, float("inf"), float("nan")])
def test_toy_manifold_tiling_rejects_invalid_distance(
    tmp_path: Path,
    distance: float,
) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        evaluate_toy_manifold_tiling(
            tmp_path,
            shard_dir=tmp_path,
            layer=0,
            model_kind="mfa",
            device="cpu",
            max_mean_to_manifold_distance=distance,
        )
