from __future__ import annotations

import pytest
import torch

from dalg.data.manifold_dataset import ToyManifoldConfig, make_toy_manifold_dataset
from dalg.evaluation.toy_manifold_metrics import (
    _associate_component_means,
    _subspace_alignment,
    evaluate_toy_manifold_metrics,
)
from dalg.models.mfa import MFA


def _set_scales(model: MFA, scales: torch.Tensor) -> None:
    with torch.no_grad():
        model.scale_rho.copy_(torch.log(torch.expm1(scales)))


def _ambient_point(
    metadata: dict[str, object],
    manifold_index: int,
    raw_point: torch.Tensor,
) -> torch.Tensor:
    manifold = metadata["manifolds"][manifold_index]
    type_id = int(manifold["type_id"])
    return (
        (raw_point.double() - metadata["calibration_means"][type_id])
        / metadata["calibration_scales"][type_id]
    ) @ manifold["embedding"] + manifold["position"]


def _flat_disk_metadata(*, manifolds_per_type: int = 1):
    _, metadata = make_toy_manifold_dataset(
        ToyManifoldConfig(
            ambient_dim=4,
            n_samples=16,
            calibration_size=64,
            manifolds_per_type=manifolds_per_type,
            manifold_types=("flat_disk",),
            offset_radius=3.0,
            seed=11,
        )
    )
    return metadata


def _flat_disk_frame(metadata):
    manifold = metadata["manifolds"][0]
    tangent_0, tangent_1 = manifold["embedding"].float()
    _, _, vh = torch.linalg.svd(manifold["embedding"].float(), full_matrices=True)
    normal_0, normal_1 = vh[2], vh[3]
    mean = _ambient_point(metadata, 0, torch.zeros(2)).float()
    return mean, tangent_0, tangent_1, normal_0, normal_1


def test_subspace_alignment_is_basis_invariant() -> None:
    tangent = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=torch.float64
    )
    angle = torch.tensor(0.4, dtype=torch.float64)
    rotation = torch.stack(
        (
            torch.stack((torch.cos(angle), -torch.sin(angle))),
            torch.stack((torch.sin(angle), torch.cos(angle))),
        )
    )
    principal = tangent @ rotation

    overlap, worst = _subspace_alignment(tangent, principal)

    assert overlap == pytest.approx(1.0, abs=1e-12)
    assert worst == pytest.approx(1.0, abs=1e-12)


def test_subspace_alignment_reports_partial_and_missing_direction() -> None:
    tangent = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=torch.float64
    )
    principal = torch.tensor(
        [[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]], dtype=torch.float64
    )

    overlap, worst = _subspace_alignment(tangent, principal)

    assert overlap == pytest.approx(0.5, abs=1e-12)
    assert worst == pytest.approx(0.0, abs=1e-12)


def test_subspace_alignment_reports_containment_in_larger_space() -> None:
    tangent = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]],
        dtype=torch.float64,
    )
    principal = torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )

    overlap, worst = _subspace_alignment(tangent, principal)

    assert overlap == pytest.approx(1.0, abs=1e-12)
    assert worst == pytest.approx(1.0, abs=1e-12)


def test_subspace_alignment_penalizes_missing_tangent_dimensions() -> None:
    tangent = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=torch.float64
    )
    principal = torch.tensor([[1.0], [0.0], [0.0]], dtype=torch.float64)

    overlap, worst = _subspace_alignment(tangent, principal)

    assert overlap == pytest.approx(0.5, abs=1e-12)
    assert worst == pytest.approx(0.0, abs=1e-12)


def test_proximity_association_accepts_cutoff_and_rejects_far_mean() -> None:
    metadata = _flat_disk_metadata()
    mean, _, _, normal, _ = _flat_disk_frame(metadata)
    means = torch.stack((mean + 0.1 * normal, mean + 0.2 * normal))

    associations = _associate_component_means(
        means,
        metadata,
        max_mean_to_manifold_distance=0.1,
    )

    assert associations.manifold_indices.tolist() == [0, -1]
    assert associations.ambiguous.tolist() == [False, False]
    assert associations.associated.tolist() == [True, False]
    assert associations.outside_cutoff.tolist() == [False, True]
    assert associations.nearest_distances.tolist() == pytest.approx([0.1, 0.2])


def test_proximity_association_rejects_tied_nearest_manifolds() -> None:
    metadata = _flat_disk_metadata()
    mean, *_ = _flat_disk_frame(metadata)
    tied_metadata = dict(metadata)
    duplicate = dict(metadata["manifolds"][0])
    duplicate["manifold_id"] = 99
    tied_metadata["manifolds"] = [metadata["manifolds"][0], duplicate]
    tied_metadata["num_manifolds"] = 2

    associations = _associate_component_means(
        mean[None],
        tied_metadata,
        max_mean_to_manifold_distance=0.1,
    )

    assert associations.manifold_indices.tolist() == [-1]
    assert associations.ambiguous.tolist() == [True]
    assert not bool(associations.outside_cutoff[0])


def test_internal_top_eigenvalue_tie_keeps_two_dimensional_alignment_defined() -> None:
    metadata = _flat_disk_metadata()
    mean, tangent_0, tangent_1, _, _ = _flat_disk_frame(metadata)
    directions = torch.stack((tangent_0, tangent_1), dim=1)[None]
    model = MFA(mean[None], rank=2, init_directions=directions, psi_init=0.1)
    _set_scales(model, torch.tensor([[2.0, 2.0]]))

    metrics = evaluate_toy_manifold_metrics(
        model,
        metadata,
        torch.tensor([False]),
        max_mean_to_manifold_distance=0.1,
    )

    manifold = metrics["per_manifold"][0]
    assert manifold["components"] == {
        "associated": 1,
        "assignment_live": 0,
        "assignment_dead": 1,
    }
    assert manifold["rank"]["exact_match"] == 1.0
    assert manifold["tangent_alignment"]["subspace_overlap"] == {
        "mean": pytest.approx(1.0, abs=1e-10),
        "valid_components": 1,
        "undefined_components": 0,
    }
    assert manifold["tangent_alignment"]["worst_direction_cosine"]["mean"] == (
        pytest.approx(1.0, abs=1e-10)
    )


def test_later_tangent_pc_does_not_rescue_leading_plane_alignment() -> None:
    metadata = _flat_disk_metadata()
    mean, tangent_0, _, normal_0, _ = _flat_disk_frame(metadata)
    directions = torch.stack((normal_0, tangent_0), dim=1)[None]
    model = MFA(mean[None], rank=2, init_directions=directions, psi_init=0.1)
    _set_scales(model, torch.tensor([[3.0, 2.0]]))

    metrics = evaluate_toy_manifold_metrics(
        model,
        metadata,
        torch.tensor([True]),
        max_mean_to_manifold_distance=0.1,
    )

    alignment = metrics["per_manifold"][0]["tangent_alignment"]
    assert alignment["subspace_overlap"]["mean"] == pytest.approx(0.5, abs=1e-6)
    assert alignment["worst_direction_cosine"]["mean"] == pytest.approx(
        0.0, abs=1e-6
    )


def test_effective_rank_containment_uses_later_principal_components() -> None:
    metadata = _flat_disk_metadata()
    mean, tangent_0, tangent_1, normal_0, _ = _flat_disk_frame(metadata)
    directions = torch.stack((normal_0, tangent_0, tangent_1), dim=1)[None]
    model = MFA(mean[None], rank=3, init_directions=directions, psi_init=0.1)
    _set_scales(model, torch.tensor([[3.0, 2.0, 1.0]]))

    metrics = evaluate_toy_manifold_metrics(
        model,
        metadata,
        torch.tensor([True]),
        max_mean_to_manifold_distance=0.1,
    )

    manifold = metrics["per_manifold"][0]
    assert manifold["tangent_alignment"]["subspace_overlap"]["mean"] == (
        pytest.approx(0.5, abs=1e-6)
    )
    assert manifold["tangent_alignment"]["worst_direction_cosine"]["mean"] == (
        pytest.approx(0.0, abs=1e-6)
    )
    assert manifold["tangent_containment"]["subspace_overlap"]["mean"] == (
        pytest.approx(1.0, abs=1e-6)
    )
    assert manifold["tangent_containment"]["worst_direction_cosine"]["mean"] == (
        pytest.approx(1.0, abs=1e-6)
    )


def test_zero_effective_rank_has_zero_alignment_and_containment() -> None:
    metadata = _flat_disk_metadata()
    mean, tangent_0, tangent_1, _, _ = _flat_disk_frame(metadata)
    directions = torch.stack((tangent_0, tangent_1), dim=1)[None]
    model = MFA(mean[None], rank=2, init_directions=directions, psi_init=0.1)
    _set_scales(model, torch.tensor([[0.01, 0.01]]))

    metrics = evaluate_toy_manifold_metrics(
        model,
        metadata,
        torch.tensor([True]),
        max_mean_to_manifold_distance=0.1,
    )

    assert metrics["rank"]["mean_learned"] == 0.0
    for metric_name in ("tangent_alignment", "tangent_containment"):
        for score_name in ("subspace_overlap", "worst_direction_cosine"):
            assert metrics[metric_name][score_name] == {
                "mean": 0.0,
                "valid_components": 1,
                "undefined_components": 0,
            }


def test_tied_effective_rank_boundary_only_undefines_containment() -> None:
    _, metadata = make_toy_manifold_dataset(
        ToyManifoldConfig(
            ambient_dim=4,
            n_samples=16,
            calibration_size=64,
            manifolds_per_type=1,
            manifold_types=("segment",),
            offset_radius=3.0,
            seed=12,
        )
    )
    manifold = metadata["manifolds"][0]
    mean = _ambient_point(metadata, 0, torch.zeros(1)).float()
    tangent = manifold["embedding"][0].float()
    directions = torch.stack((tangent, tangent), dim=1)[None]
    model = MFA(mean[None], rank=2, init_directions=directions, psi_init=0.1)
    _set_scales(model, torch.tensor([[2.0, 2.0]]))

    metrics = evaluate_toy_manifold_metrics(
        model,
        metadata,
        torch.tensor([True]),
        max_mean_to_manifold_distance=0.1,
    )

    assert metrics["tangent_alignment"]["subspace_overlap"]["mean"] == (
        pytest.approx(1.0, abs=1e-6)
    )
    undefined = {
        "mean": None,
        "valid_components": 0,
        "undefined_components": 1,
    }
    assert metrics["tangent_containment"]["subspace_overlap"] == undefined
    assert metrics["tangent_containment"]["worst_direction_cosine"] == undefined


def test_full_ambient_effective_rank_has_defined_perfect_containment() -> None:
    metadata = _flat_disk_metadata()
    mean, *_ = _flat_disk_frame(metadata)
    model = MFA(
        mean[None],
        rank=4,
        init_directions=torch.eye(4)[None],
        psi_init=0.1,
    )
    _set_scales(model, torch.ones(1, 4))

    metrics = evaluate_toy_manifold_metrics(
        model,
        metadata,
        torch.tensor([True]),
        max_mean_to_manifold_distance=0.1,
    )

    assert metrics["tangent_containment"]["subspace_overlap"] == {
        "mean": pytest.approx(1.0, abs=1e-10),
        "valid_components": 1,
        "undefined_components": 0,
    }
    assert metrics["tangent_containment"]["worst_direction_cosine"]["mean"] == (
        pytest.approx(1.0, abs=1e-10)
    )


def test_tied_intrinsic_dimension_boundary_makes_alignment_undefined() -> None:
    metadata = _flat_disk_metadata()
    mean, tangent_0, _, _, _ = _flat_disk_frame(metadata)
    model = MFA(
        mean[None],
        rank=1,
        init_directions=tangent_0[None, :, None],
        psi_init=0.1,
    )
    _set_scales(model, torch.tensor([[2.0]]))

    metrics = evaluate_toy_manifold_metrics(
        model,
        metadata,
        torch.tensor([True]),
        max_mean_to_manifold_distance=0.1,
    )

    undefined = {
        "mean": None,
        "valid_components": 0,
        "undefined_components": 1,
    }
    alignment = metrics["per_manifold"][0]["tangent_alignment"]
    assert alignment["subspace_overlap"] == undefined
    assert alignment["worst_direction_cosine"] == undefined


def test_per_manifold_metrics_include_empty_instances() -> None:
    metadata = _flat_disk_metadata(manifolds_per_type=2)
    mean, tangent_0, tangent_1, _, _ = _flat_disk_frame(metadata)
    directions = torch.stack((tangent_0, tangent_1), dim=1)[None]
    model = MFA(mean[None], rank=2, init_directions=directions, psi_init=0.1)
    _set_scales(model, torch.tensor([[3.0, 2.0]]))

    metrics = evaluate_toy_manifold_metrics(
        model,
        metadata,
        torch.tensor([True]),
        max_mean_to_manifold_distance=0.1,
    )

    assert metrics["association"] == {
        "rule": "unique_nearest_exact_projection_within_cutoff",
        "max_mean_to_manifold_distance": 0.1,
        "associated_components": 1,
        "outside_cutoff_components": 0,
        "ambiguous_components": 0,
    }
    assert len(metrics["per_manifold"]) == 2
    assert metrics["per_manifold"][0]["components"]["associated"] == 1
    empty = metrics["per_manifold"][1]
    assert empty["components"] == {
        "associated": 0,
        "assignment_live": 0,
        "assignment_dead": 0,
    }
    assert empty["rank"]["mean_learned"] is None
    assert empty["tangent_alignment"]["subspace_overlap"] == {
        "mean": None,
        "valid_components": 0,
        "undefined_components": 0,
    }


def test_non_unique_tangent_is_associated_but_alignment_is_undefined() -> None:
    embedding = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    manifold = {
        "manifold_id": 0,
        "type_id": 0,
        "type_name": "circle",
        "intrinsic_dim": 1,
        "position": torch.zeros(4, dtype=torch.float64),
        "embedding": embedding,
    }
    metadata = {
        "num_manifolds": 1,
        "manifolds": [manifold],
        "calibration_means": (torch.zeros(2, dtype=torch.float64),),
        "calibration_scales": torch.ones(1, dtype=torch.float64),
        "config": {},
    }
    mean = torch.zeros(4)
    tangent = embedding[0].float()
    model = MFA(mean[None], rank=1, init_directions=tangent[None, :, None])

    metrics = evaluate_toy_manifold_metrics(
        model,
        metadata,
        torch.tensor([False]),
        max_mean_to_manifold_distance=2.0,
    )

    assert metrics["association"]["associated_components"] == 1
    assert metrics["tangent_alignment"]["subspace_overlap"] == {
        "mean": None,
        "valid_components": 0,
        "undefined_components": 1,
    }
