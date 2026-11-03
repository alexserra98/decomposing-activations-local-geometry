"""Numerical evaluation workflows for trained DALG models."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch

from dalg.evaluation.tile_alignment import validate_tile_alignment_settings
from dalg.evaluation.pc1_density import PC1_DENSITY_DEFAULTS, validate_pc1_density_settings


def validate_evaluation_settings(cfg: Mapping[str, Any], *, model_kind: str) -> None:
    validate_pc1_density_settings(cfg)
    if cfg.get("pc1_density", False) and (cfg["kind"] != "unsupervised" or model_kind not in {"kmeans", "hddc"}):
        raise ValueError("PC1 density requires unsupervised evaluation of KMeans or HDDC")
    validate_tile_alignment_settings(
        cfg.get("tile_alignment", True), cfg.get("tile_alignment_neighbors", 5),
    )
    if cfg["kind"] not in {"toy_manifold_tiling", "unsupervised"} and (
        cfg["kind"] is not None or cfg.get("enabled", True)
    ):
        raise ValueError(f"unsupported evaluation: {cfg['kind']!r}")
    if type(cfg.get("heldout_distribution_coverage", True)) is not bool:
        raise ValueError("evaluation.heldout_distribution_coverage must be true or false")
    if type(cfg["batch_size"]) is not int or cfg["batch_size"] <= 0:
        raise ValueError("evaluation.batch_size must be a positive integer")
    torch.device(cfg["device"])
    threshold = float(cfg["rank_threshold"])
    if model_kind not in {"hddc", "kmeans"} and (not math.isfinite(threshold) or threshold <= 0):
        raise ValueError("evaluation.rank_threshold must be finite and positive")
    cutoff = cfg.get("max_mean_to_manifold_distance")
    if cutoff is not None and (not math.isfinite(float(cutoff)) or float(cutoff) <= 0):
        raise ValueError("evaluation.max_mean_to_manifold_distance must be finite and positive")
    if cfg["kind"] == "unsupervised":
        if cutoff is not None:
            raise ValueError("max_mean_to_manifold_distance only applies to toy_manifold_tiling")
        if cfg.get("enabled", True) and cfg.get("heldout_distribution_coverage", True) and not cfg.get("test_shard_dir"):
            raise ValueError("unsupervised coverage requires evaluation.test_shard_dir")
    elif cfg.get("test_shard_dir") is not None:
        raise ValueError("test_shard_dir only applies to unsupervised evaluation")


def validate_evaluation_inputs(
    shard_dir: str, *, layer: int, model_kind: str, evaluation: Mapping[str, Any],
    drop_prefix: int | None = None,
) -> None:
    """Check coverage prerequisites before training or saved-run evaluation."""
    if not evaluation.get("heldout_distribution_coverage", True):
        return
    if evaluation["kind"] == "toy_manifold_tiling":
        from dalg.evaluation.toy_manifold_coverage import validate_toy_test_split

        validate_toy_test_split(shard_dir, layer=layer)
    elif evaluation["kind"] == "unsupervised":
        from dalg.evaluation.unsupervised import validate_test_shards

        validate_test_shards(shard_dir, evaluation["test_shard_dir"], layer=layer,
                             model_kind=model_kind, drop_prefix=drop_prefix)
    else:
        raise ValueError(f"unsupported evaluation: {evaluation['kind']!r}")


def evaluate_pipeline_run(
    run: Mapping[str, Any], *, evaluation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Dispatch a manifest row to its configured evaluator."""
    cfg = run["evaluation"] if evaluation is None else evaluation
    kind = run["training"]["model_kind"]
    validate_evaluation_settings(cfg, model_kind=kind)
    if cfg["kind"] == "toy_manifold_tiling":
        from dalg.evaluation.toy_manifold_tiling import evaluate_toy_manifold_tiling

        evaluator = evaluate_toy_manifold_tiling
        options = {"max_mean_to_manifold_distance": cfg["max_mean_to_manifold_distance"]}
    else:
        from dalg.evaluation.unsupervised import evaluate_unsupervised

        evaluator = evaluate_unsupervised
        options = {"test_shard_dir": cfg.get("test_shard_dir"),
                   **{key: cfg.get(key, value) for key, value in PC1_DENSITY_DEFAULTS.items()}}
    metrics = evaluator(
        run["run_dir"], shard_dir=run["dataset"]["shard_dir"],
        layer=int(run["dataset"]["layer"]), model_kind=kind,
        batch_size=int(cfg["batch_size"]), device=str(cfg["device"]),
        rank_threshold=float(cfg["rank_threshold"]),
        heldout_distribution_coverage=cfg.get("heldout_distribution_coverage", True),
        tile_alignment=cfg.get("tile_alignment", True),
        tile_alignment_neighbors=cfg.get("tile_alignment_neighbors", 5), **options,
    )
    metrics.update(run_id=run["run_id"], identity_hash=run["identity_hash"])
    return metrics

