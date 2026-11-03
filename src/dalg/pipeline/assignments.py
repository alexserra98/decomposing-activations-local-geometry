"""Assignment settings, command construction, and partition validation."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from dalg.init.activation_selection import resolve_initialization_rows

from .common import _model_stem
from .training import _training_artifact_path


_ASSIGNMENT_DEFAULTS = {
    "enabled": True,
    "batch_size": 1024,
    "device": "cuda",
    "seed": None,
    "use_inference_cache": True,
}


def _assignments_path(run: Mapping[str, Any]) -> Path:
    return Path(run["run_dir"]) / f"{_model_stem(run)}_assignments.pt"


def _validate_assignment_bundle(path: Path, expected_k: int) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        bundle = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        assignments = bundle["assignments"].reshape(-1)
        sizes = bundle["cluster_sizes"].reshape(-1)
        saved_k = int(bundle["K"])
    except (EOFError, KeyError, OSError, TypeError, RuntimeError, ValueError):
        return False
    if assignments.numel() == 0:
        return False
    if int(assignments.min()) < 0 or int(assignments.max()) >= expected_k:
        return False
    return (
        saved_k == expected_k
        and sizes.numel() == expected_k
        and int(sizes.sum()) == assignments.numel()
        and torch.equal(torch.bincount(assignments.long(), minlength=expected_k), sizes.long())
    )


def _assignment_artifact_valid(run: Mapping[str, Any]) -> bool:
    """Validate a complete model partition, including the KMeans source stream."""
    path = _assignments_path(run)
    k = int(run["training"]["arguments"]["K"])
    if not _validate_assignment_bundle(path, k):
        return False
    if run["training"]["model_kind"] != "kmeans":
        return True
    try:
        bundle = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        root, _, _, selection = resolve_initialization_rows(
            run["dataset"]["shard_dir"], layer=run["dataset"]["layer"],
            val_frac=0.0, drop_prefix=run["training"]["arguments"].get("drop_prefix"),
        )
        source = bundle["source"]
        expected = selection["selected_activations"]
        return (
            bundle["assignments"].dtype == torch.long
            and bundle["cluster_sizes"].dtype == torch.long
            and bundle["assignments"].numel() == expected
            and bundle["max_responsibilities"].shape == (expected,)
            and bool((bundle["max_responsibilities"] == 1).all())
            and bundle["model_type"] == "kmeans"
            and Path(bundle["model_path"]).resolve() == _training_artifact_path(run).resolve()
            and bundle.get("subset_spec") == selection["subset_spec"]
            and Path(source["shard_dir"]).resolve() == root.resolve()
            and source["layer"] == run["dataset"]["layer"]
            and source["drop_prefix"] == selection["drop_prefix"]
            and source["num_items"] == expected
        )
    except (EOFError, KeyError, OSError, TypeError, RuntimeError, ValueError):
        return False


def _assignment_command(run: Mapping[str, Any], path: Path) -> list[str]:
    cfg = run["assignments"]
    command = [
        sys.executable,
        "-m",
        "dalg.cli.run_metrics",
        "assignments",
        "--data-dir",
        run["run_dir"],
        "--shard-dir",
        run["dataset"]["shard_dir"],
        "--layer",
        str(run["dataset"]["layer"]),
        "--batch-size",
        str(cfg["batch_size"]),
        "--device",
        str(cfg["device"]),
        "--seed",
        str(cfg["seed"]),
        "--model-type",
        "mfa" if run["training"]["model_kind"] == "ard" else run["training"]["model_kind"],
        "--save-path",
        str(path),
    ]
    if not cfg["use_inference_cache"]:
        command.append("--no-inference-cache")
    drop_prefix = run["training"]["arguments"].get("drop_prefix")
    if drop_prefix is not None:
        command.extend(["--drop-prefix", str(drop_prefix)])
    return command
