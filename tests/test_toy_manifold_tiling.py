from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from dalg.analysis.bic_improved import compute_improved_bic_details
from dalg.data.manifold_dataset import (
    MANIFOLD_NAMES,
    ToyManifoldConfig,
    save_toy_manifold_shards,
)
from dalg.data.shard_activations import load_meta_index
from dalg.evaluation.toy_manifold_geometry import _project_mean_to_manifold
from dalg.evaluation.toy_manifold_tiling import evaluate_toy_manifold_tiling
from dalg.models.adaptive_q.mfa_ard import MFA_ARD, save_mfa_ard
from dalg.models.adaptive_q.mfa_hddc import MFA_HDDC, save_mfa_hddc
from dalg.models.mfa import MFA, save_mfa


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
        "leading_learned_rank_covariance_subspace_principal_angles"
        if model_kind == "hddc"
        else "leading_effective_rank_covariance_subspace_principal_angles"
    )
    if model_kind == "hddc":
        for rank_name in ("rank", "ambient_rank"):
            assert metrics[rank_name]["definition"] == "hddc_rank_mask_count"
            assert "threshold" not in metrics[rank_name]
            assert metrics[rank_name]["mean_learned"] == 2.0
    for metric_name in ("tangent_alignment", "tangent_containment"):
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
