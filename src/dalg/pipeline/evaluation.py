"""Evaluation settings, report validation, and atomic report publication."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from dalg.data.subset_spec import split_shard_dir_spec
from dalg.evaluation.toy_manifold_coverage import coverage_report_valid
from dalg.evaluation.tile_alignment import tile_alignment_report_valid
from dalg.evaluation.reconstruction import reconstruction_report_valid
from dalg.evaluation.pc1_density import PC1_DENSITY_DEFAULTS

from .common import _write_json_atomic


_EVALUATION_DEFAULTS = {
    "enabled": False,
    "kind": None,
    "batch_size": 4096,
    "device": "cuda",
    "rank_threshold": 1.0,
    "max_mean_to_manifold_distance": None,
    "heldout_distribution_coverage": True,
    "tile_alignment": True,
    "tile_alignment_neighbors": 5,
    **PC1_DENSITY_DEFAULTS,
}


def _evaluation_artifact_valid(run: Mapping[str, Any]) -> bool:
    path = Path(run["run_dir"]) / "metrics.json"
    if not path.is_file():
        return False
    try:
        metrics = json.loads(path.read_text())
    except (OSError, TypeError, json.JSONDecodeError):
        return False
    return _evaluation_metrics_valid(run, metrics)


def _evaluation_metrics_valid(run: Mapping[str, Any], metrics: Any) -> bool:
    """Validate evaluation metrics before publishing them or reusing a report."""
    if run["evaluation"]["kind"] == "unsupervised":
        from dalg.evaluation.unsupervised import unsupervised_report_valid

        return unsupervised_report_valid(run, metrics)
    if not isinstance(metrics, Mapping):
        return False
    kind = run["training"]["model_kind"]
    if kind in {"hddc", "kmeans"} and not reconstruction_report_valid(
        metrics.get("reconstruction"), model_kind=kind, dataset=metrics.get("dataset"),
    ):
        return False
    if run["evaluation"].get("tile_alignment", True):
        if not tile_alignment_report_valid(
            metrics.get("tile_alignment"), components=run["training"]["arguments"]["K"],
            neighbors=run["evaluation"].get("tile_alignment_neighbors", 5),
        ):
            return False
    elif "tile_alignment" in metrics:
        return False
    if run["evaluation"].get("heldout_distribution_coverage", True):
        if not coverage_report_valid(
            metrics.get("heldout_distribution_coverage"),
            components=run["training"]["arguments"]["K"],
        ):
            return False
        source = metrics["heldout_distribution_coverage"]["source"]
        root, _ = split_shard_dir_spec(run["dataset"]["shard_dir"])
        if (Path(source["shard_dir"]).resolve() != (root / "test").resolve()
                or source["layer"] != run["dataset"]["layer"]):
            return False
    alignment = metrics.get("tangent_alignment")
    basis = "pca" if run["training"]["model_kind"] == "kmeans" else "covariance"
    if not isinstance(alignment, Mapping) or alignment.get("definition") != (
        f"leading_intrinsic_dim_{basis}_subspace_principal_angles"
    ):
        return False
    for metric_name in ("tangent_alignment", "tangent_containment"):
        metric = metrics.get(metric_name)
        if not isinstance(metric, Mapping) or (
            metric.get("rank_requirement") != "effective_rank_gte_intrinsic_dim"
        ):
            return False
    if metrics["tangent_containment"].get("definition") != (
        f"best_intrinsic_dim_subset_of_leading_rank_{basis}_principal_angles"
    ):
        return False
    partial_containment = metrics.get("tangent_partial_containment")
    if not isinstance(partial_containment, Mapping) or (
        partial_containment.get("definition")
        != f"leading_{basis}_subspace_within_tangent_principal_angles"
        or partial_containment.get("rank_requirement")
        != "effective_rank_gt_zero_lt_intrinsic_dim"
        or partial_containment.get("normalization") != "effective_rank"
    ):
        return False
    adjusted_alignment = metrics.get("tangent_adjusted_alignment")
    if not isinstance(adjusted_alignment, Mapping) or (
        adjusted_alignment.get("definition")
        != f"leading_min_intrinsic_effective_rank_{basis}_subspace_overlap"
        or adjusted_alignment.get("rank_requirement") != "effective_rank_gte_zero"
        or adjusted_alignment.get("normalization") != "intrinsic_dim"
        or adjusted_alignment.get("zero_rank") != "zero_if_tangent_defined"
        or adjusted_alignment.get("aggregation") != "unweighted_component_mean"
    ):
        return False
    if run["training"]["model_kind"] == "kmeans":
        quantization = metrics.get("quantization")
        if not isinstance(quantization, Mapping) or quantization.get("convention") != "lower_is_better":
            return False
        for name in ("train", "validation"):
            values = quantization.get(name)
            if not isinstance(values, Mapping) or type(values.get("n")) is not int or values["n"] < 0:
                return False
            if values["n"] == 0:
                if name == "train":
                    return False
                continue
            for key in ("sum_squared_distance", "mean_squared_distance"):
                value = values.get(key)
                if not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
                    return False
        geometry = metrics.get("pca_geometry")
        if not isinstance(geometry, Mapping) or geometry.get("version") != 1:
            return False
        counts = [geometry.get(name) for name in ("eligible_components", "excluded_components")]
        if any(type(count) is not int or count < 0 for count in counts):
            return False
        return (
            metrics.get("schema_version") == 2
            and metrics.get("model_kind") == "kmeans"
            and geometry.get("minimum_cluster_points") == 2
            and sum(counts) == run["training"]["arguments"]["K"]
            and metrics.get("evaluation") == run["evaluation"]["kind"]
            and metrics.get("identity_hash") == run["identity_hash"]
            and metrics.get("rank", {}).get("definition") == "kmeans_component_ranks"
            and metrics.get("ambient_rank", {}).get("definition") == "kmeans_component_ranks"
            and "nll" not in metrics and "bic" not in metrics
        )
    if run["training"]["model_kind"] == "hddc":
        for rank_name in ("rank", "ambient_rank"):
            rank = metrics.get(rank_name)
            if not isinstance(rank, Mapping) or (
                rank.get("definition") != "hddc_rank_mask_count"
                or "threshold" in rank
            ):
                return False
    bic_metrics = metrics.get("bic")
    if not isinstance(bic_metrics, Mapping):
        return False
    bic_value = bic_metrics.get("value")
    return (
        metrics.get("schema_version") == 2
        and bic_metrics.get("convention") == "higher_is_better"
        and bic_metrics.get("formula") == "-standard_bic / n + active_components"
        and isinstance(bic_value, (int, float))
        and math.isfinite(float(bic_value))
        and metrics.get("evaluation") == run["evaluation"]["kind"]
        and metrics.get("identity_hash") == run["identity_hash"]
    )


def compute_metrics(run: Mapping[str, Any]) -> dict[str, Any]:
    from dalg.evaluation import evaluate_pipeline_run

    return evaluate_pipeline_run(run)


def publish_metrics(run: Mapping[str, Any], path: Path, metrics: Mapping[str, Any]) -> None:
    """Validate before publishing; preserve existing reports on failure."""
    if not _evaluation_metrics_valid(run, metrics):
        raise ValueError(f"evaluator produced invalid metrics for {run['run_dir']}")
    json.dumps(metrics, allow_nan=False)
    _write_json_atomic(path, metrics)
