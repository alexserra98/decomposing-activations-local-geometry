"""Reconstruction report definitions and completion validation."""

from __future__ import annotations

import math
from collections.abc import Mapping


def reconstruction_metadata(model_kind: str, dimension: int) -> dict:
    if model_kind not in {"hddc", "kmeans"}:
        raise ValueError(f"unsupported reconstruction model: {model_kind!r}")
    return {
        "version": 1,
        "method": "nearest_centroid_local_pca" if model_kind == "kmeans" else "mfa_posterior_mean",
        "normalization": "per_activation_dimension",
        "population": "samples_assigned_to_pca_valid_components" if model_kind == "kmeans" else "all_samples",
        "dimension": dimension,
        "convention": "lower_is_better",
    }


def reconstruction_report_valid(report, *, model_kind: str, dataset, allow_empty_splits: bool = False) -> bool:
    """Check definitions, split counts, and finite statistics before publication."""
    if not isinstance(report, Mapping) or not isinstance(dataset, Mapping):
        return False
    dimension = dataset.get("d_model")
    if type(dimension) is not int or dimension <= 0:
        return False
    if any(report.get(key) != value for key, value in reconstruction_metadata(model_kind, dimension).items()):
        return False
    if type(report.get("version")) is not int or type(report.get("dimension")) is not int:
        return False
    window, prefix = dataset.get("window"), dataset.get("drop_prefix")
    if type(window) is not int or type(prefix) is not int or not 0 <= prefix < window:
        return False
    for split in ("train", "validation"):
        values = report.get(split)
        if not isinstance(values, Mapping):
            return False
        n, excluded = values.get("n"), values.get("excluded_n")
        points = dataset.get(f"{split}_points")
        rows = dataset.get(f"{split}_rows")
        if (any(type(count) is not int or count < 0 for count in (n, excluded, points))
                or n + excluded != points or (split == "train" and points == 0 and not allow_empty_splits)):
            return False
        if type(rows) is not int or rows < 0 or rows * (window - prefix) != points:
            return False
        if model_kind == "hddc" and (excluded != 0 or (n == 0 and not allow_empty_splits)):
            return False
        for key, defined in (("mse", n > 0), ("standard_error", n > 1)):
            if key not in values:
                return False
            value = values[key]
            if defined:
                if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
                    return False
            elif value is not None:
                return False
    return True
