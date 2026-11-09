"""Evaluate how a model tiles a dataset with planted toy manifolds."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from sklearn.metrics import (
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
    normalized_mutual_info_score,
)

from dalg.evaluation.common import (
    evaluate_fit, load_evaluation_data, load_model, resolve_device,
)
from dalg.evaluation.subspaces import effective_component_ranks, leading_covariance_eigenspaces
from dalg.evaluation.tile_alignment import evaluate_tile_alignment, validate_tile_alignment_settings
from dalg.evaluation.toy_manifold_metrics import evaluate_toy_manifold_metrics
from dalg.evaluation.toy_manifold_coverage import evaluate_toy_test_coverage, validate_toy_test_split


def evaluate_toy_manifold_tiling(
    run_dir: str | Path,
    *,
    shard_dir: str | Path,
    layer: int,
    model_kind: str,
    assignments_path: str | Path | None = None,
    batch_size: int = 4096,
    device: str = "cuda",
    rank_threshold: float = 1.0,
    max_mean_to_manifold_distance: float | None = None,
    heldout_distribution_coverage: bool = True,
    tile_alignment: bool = True,
    tile_alignment_neighbors: int = 5,
) -> dict[str, Any]:
    """Evaluate one model run against planted toy-manifold structure."""
    if type(heldout_distribution_coverage) is not bool:
        raise ValueError("heldout_distribution_coverage must be true or false")
    if model_kind not in {"hddc", "kmeans"} and rank_threshold <= 0.0:
        raise ValueError("rank_threshold must be positive")
    if max_mean_to_manifold_distance is not None and (
        not math.isfinite(max_mean_to_manifold_distance)
        or max_mean_to_manifold_distance <= 0.0
    ):
        raise ValueError("max_mean_to_manifold_distance must be finite and positive")

    validate_tile_alignment_settings(tile_alignment, tile_alignment_neighbors)
    run_dir = Path(run_dir)
    test_source = validate_toy_test_split(shard_dir, layer=layer) if heldout_distribution_coverage else None
    model = load_model(run_dir, model_kind)
    data = load_evaluation_data(
        run_dir, shard_dir=shard_dir, layer=layer, model=model,
        model_kind=model_kind, assignments_path=assignments_path,
    )
    if data.config.get("source_kind") != "toy_manifolds":
        raise ValueError("toy_manifold_tiling requires shards from save_toy_manifold_shards")
    if int(data.config["window"]) != 1 or data.drop_prefix != 0:
        raise ValueError("toy_manifold_tiling expects one activation per row")
    manifold_metadata = torch.load(
        data.shard_dir / data.config["manifold_metadata"], map_location="cpu", weights_only=True,
    )
    all_manifold_ids = manifold_metadata["row_manifold_ids"].reshape(-1).long()
    if all_manifold_ids.numel() != len(data.meta_index):
        raise ValueError("toy manifold labels are not aligned with activation metadata")
    row_manifold_ids = all_manifold_ids[torch.as_tensor(data.positions)]
    resolved_device = resolve_device(device)
    model = model.to(resolved_device).eval()
    fit_metrics = evaluate_fit(
        model, data, model_kind=model_kind, layer=layer,
        batch_size=batch_size, device=resolved_device,
    )

    true_ids = row_manifold_ids.numpy()
    predicted_ids = data.assignments.numpy()
    clustering = {
        "homogeneity": float(homogeneity_score(true_ids, predicted_ids)),
        "completeness": float(completeness_score(true_ids, predicted_ids)),
        "adjusted_rand_index": float(adjusted_rand_score(true_ids, predicted_ids)),
        "normalized_mutual_information": float(
            normalized_mutual_info_score(true_ids, predicted_ids)
        ),
    }

    assignment_live = data.cluster_sizes > 0
    eigenspaces = None
    tile_metrics = {}
    if tile_alignment:
        ranks = effective_component_ranks(model, rank_threshold)
        eigenspaces = leading_covariance_eigenspaces(model, max(1, int(ranks.max())))
        tile_metrics["tile_alignment"] = evaluate_tile_alignment(
            model, assignment_live, neighbors=tile_alignment_neighbors,
            rank_threshold=rank_threshold, eigenspaces=eigenspaces,
        )
    manifold_metrics = evaluate_toy_manifold_metrics(
        model,
        manifold_metadata,
        assignment_live,
        rank_threshold=rank_threshold,
        max_mean_to_manifold_distance=max_mean_to_manifold_distance,
        eigenspaces=eigenspaces,
    )

    coverage_metrics = {}
    if test_source is not None:
        coverage_metrics["heldout_distribution_coverage"] = evaluate_toy_test_coverage(
            test_source, model.mu, data.train_cluster_sizes > 0, batch_size=batch_size,
        )

    return {
        "schema_version": 2,
        "evaluation": "toy_manifold_tiling",
        "model_kind": model_kind,
        "K": int(model.K),
        "q_capacity": int(model.q),
        "dataset": data.dataset_summary(layer),
        **fit_metrics,
        "clustering": clustering,
        "components": {
            "live": int(assignment_live.sum()),
            "dead": int((~assignment_live).sum()),
        },
        **manifold_metrics,
        **tile_metrics,
        **coverage_metrics,
    }


__all__ = ["evaluate_toy_manifold_tiling"]
