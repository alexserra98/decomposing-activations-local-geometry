"""Execution prerequisites and ownership of the persisted run specification."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from dalg.evaluation import validate_evaluation_inputs

from .common import PipelineConfigError, _write_json_atomic
from .initialization import _check_initialization_version


def _ensure_run_spec(run: Mapping[str, Any]) -> Path:
    run_dir = Path(run["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "run_spec.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if (
            saved.get("identity_hash") != run["identity_hash"]
            or saved.get("identity") != run["identity"]
        ):
            raise PipelineConfigError(
                f"run directory belongs to a different configuration: {run_dir}"
            )
        return path
    _write_json_atomic(path, run)
    return path


def validate_prerequisites(run: Mapping[str, Any]) -> None:
    """Check inputs and manifest compatibility before creating a run directory."""
    if run["evaluation"]["enabled"] and run["evaluation"].get("heldout_distribution_coverage", True):
        validate_evaluation_inputs(
            run["dataset"]["shard_dir"], layer=run["dataset"]["layer"],
            model_kind=run["training"]["model_kind"], evaluation=run["evaluation"],
            drop_prefix=run["training"]["arguments"].get("drop_prefix"),
        )
    if "initialization" in run:
        _check_initialization_version(run)
    if run["training"]["model_kind"] == "kmeans":
        arguments = run["training"]["arguments"]
        if "pca_method" in arguments or arguments.get("pca_purpose") != "geometry":
            raise PipelineConfigError("incompatible KMeans PCA manifest; plan a new run with cluster geometry")

def prepare_run(run: Mapping[str, Any]) -> Path:
    """Check prerequisites, then create or verify run_spec.json; return run_dir."""
    validate_prerequisites(run)
    _ensure_run_spec(run)
    return Path(run["run_dir"])