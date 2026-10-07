"""Independent test-population coverage for the toy-manifold pipeline."""

from __future__ import annotations

import hashlib
import json
import math
import shlex
from collections.abc import Mapping
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dalg.data.shard_activations import ActivationBatchDataset, load_meta_index
from dalg.data.subset_spec import split_shard_dir_spec
from dalg.evaluation.coverage import (
    MAX_EMPIRICAL_COVERAGE_POINTS,
    check_empirical_coverage_size,
    nearest_live_centroid_distances,
    summarize_coverage_distances,
)


def validate_toy_test_split(shard_dir: str | Path, *, layer: int) -> dict:
    """Validate reserved or supplemental test data without loading activations."""
    root, _ = split_shard_dir_spec(str(shard_dir))
    test_dir = root / "test"
    if not test_dir.is_dir():
        raise FileNotFoundError(
            f"heldout_distribution_coverage requires {test_dir}. Generate an "
            "independent test split with: PYTHONPATH=src .venv/bin/python "
            "scripts/temporary/add_toy_manifold_test_split.py "
            f"{shlex.quote(str(root))}"
        )
    source = json.loads((root / "config.json").read_text())
    config = json.loads((test_dir / "config.json").read_text())
    count = config.get("num_rows")
    if type(count) is not int or count <= 0:
        raise ValueError(f"coverage test num_rows must be a positive integer: {test_dir}")
    check_empirical_coverage_size(count)
    for directory, cfg in ((root, source), (test_dir, config)):
        if (cfg.get("source_kind") != "toy_manifolds" or cfg.get("window") != 1
                or cfg.get("drop_prefix", 0) != 0 or layer not in cfg.get("layers", [])):
            raise ValueError(f"incompatible toy-manifold coverage dataset/layer: {directory}")
    if config.get("d_model") != source.get("d_model"):
        raise ValueError(f"coverage test ambient dimension differs from its parent: {test_dir}")
    partition = config.get("partition")
    if (not isinstance(partition, dict) or partition.get("version") != 1
            or partition.get("role") != "test"
            or partition.get("kind") not in {"reserved", "supplemental"}
            or partition.get("source_dir") != ".."):
        raise ValueError(f"invalid coverage test partition: {test_dir}")
    for key, path in (("config", root / "config.json"),
                      ("metadata", root / source["manifold_metadata"])):
        with path.open("rb") as handle:
            fingerprint = hashlib.file_digest(handle, "sha256").hexdigest()
        if partition.get(f"source_{key}_sha256") != fingerprint:
            raise ValueError(f"coverage test source {key} fingerprint mismatch: {test_dir}")
    metadata = torch.load(test_dir / config["manifold_metadata"], map_location="cpu", weights_only=True)
    if metadata.get("partition") != partition:
        raise ValueError(f"coverage test partition differs between config and metadata: {test_dir}")
    meta = load_meta_index(test_dir, layer=layer)
    check_empirical_coverage_size(len(meta))
    if (len(meta) != count or metadata["row_manifold_ids"].numel() != count
            or [row["global_row"] for row in meta] != list(range(count))):
        raise ValueError(f"coverage test row count/metadata mismatch: {test_dir}")
    return {"shard_dir": str(test_dir), "layer": layer, "num_rows": count, "partition": partition}


@torch.no_grad()
def evaluate_toy_test_coverage(
    source: dict, centroids: torch.Tensor, train_live: torch.Tensor, *, batch_size: int,
) -> dict:
    """Stream test points, keeping only their nearest training-live distances."""
    check_empirical_coverage_size(source["num_rows"])
    dataset = ActivationBatchDataset(
        source["shard_dir"], layer=source["layer"], batch_size=batch_size,
        drop_prefix=0, dtype=torch.float32, shuffle_shards=False, shuffle_within_shard=False,
    )
    chunks = []
    count = 0
    for batch in DataLoader(dataset, batch_size=None, num_workers=0):
        count += len(batch)
        check_empirical_coverage_size(count)
        if count > source["num_rows"]:
            raise ValueError("coverage test stream exceeds its declared row count")
        chunks.append(nearest_live_centroid_distances(
            batch, centroids, live_components=train_live, batch_size=batch_size,
        ))
    if count != source["num_rows"]:
        raise ValueError("coverage test stream does not match its declared row count")
    summary = summarize_coverage_distances(torch.cat(chunks), thresholds=None)
    summary.update(
        population="independent_test_split", components_total=int(centroids.shape[0]),
        components_live=int(train_live.sum()), liveness_split="train",
        curve_kind="full_empirical_cdf", source=source,
    )
    return summary


def coverage_report_valid(report: object, *, components: int) -> bool:
    """Check the bounded empirical-curve contract before publishing or reusing it."""
    if not isinstance(report, Mapping):
        return False
    if (report.get("definition") != "heldout_point_to_nearest_live_centroid_euclidean_distance"
            or report.get("population") != "independent_test_split"
            or report.get("liveness_split") != "train"
            or report.get("curve_kind") != "full_empirical_cdf"
            or report.get("convention") != "lower_distances_and_higher_coverage_fractions_are_better"
            or report.get("components_total") != components):
        return False
    count, live = report.get("reference_points"), report.get("components_live")
    if (type(count) is not int or not 0 < count <= MAX_EMPIRICAL_COVERAGE_POINTS
            or type(live) is not int or not 0 < live <= components):
        return False
    source = report.get("source")
    if not isinstance(source, Mapping) or source.get("num_rows") != count:
        return False
    partition = source.get("partition")
    if (not isinstance(partition, Mapping) or partition.get("version") != 1
            or partition.get("role") != "test" or partition.get("source_dir") != ".."
            or partition.get("kind") not in {"reserved", "supplemental"}
            or not isinstance(source.get("shard_dir"), str)
            or type(source.get("layer")) is not int):
        return False
    if any(not isinstance(partition.get(f"source_{key}_sha256"), str)
           or len(partition[f"source_{key}_sha256"]) != 64 for key in ("config", "metadata")):
        return False
    for key in ("mean", "root_mean_square", "median", "r90", "r95", "r99", "maximum"):
        value = report.get(key)
        if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
            return False
    curve = report.get("coverage_curve")
    if not isinstance(curve, list) or not 0 < len(curve) <= count:
        return False
    previous_radius, previous_fraction = -1.0, 0.0
    for point in curve:
        if not isinstance(point, Mapping):
            return False
        radius, fraction = point.get("radius"), point.get("fraction")
        if (type(radius) not in (float, int) or not math.isfinite(radius)
                or radius < 0 or radius <= previous_radius
                or type(fraction) not in (float, int) or not math.isfinite(fraction)
                or not previous_fraction < fraction <= 1
                or not math.isclose(fraction * count, round(fraction * count), abs_tol=1e-8)):
            return False
        previous_radius, previous_fraction = radius, fraction
    return previous_fraction == 1.0 and previous_radius == report["maximum"]
