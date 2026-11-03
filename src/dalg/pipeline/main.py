"""Read the pipeline here: planning first, then the phases of one run.

Planning passes dictionaries from YAML through sweep expansion and resolution to
an immutable JSONL manifest. Execution consumes one manifest row. Its phases
share that resolved specification and exchange model/assignment/report files in
``run_dir``; tensors and model instances stay inside the existing trainers and
evaluators. Each phase returns its artifact path, or None when it is disabled.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from . import (
    assignments,
    common,
    completion,
    configuration,
    evaluation,
    initialization,
    manifest,
    preparation,
    resolution,
    sweep,
    training,
)
from .common import PipelineConfigError


# Planning: YAML -> expanded configurations -> resolved runs -> manifest.

def map_experiments_to_manifest(
    path: str | Path, *, check_inputs: bool = True,
) -> list[dict[str, Any]]:
    """Map YAML configurations to manifest rows without writing files or fitting models."""
    config = configuration.load_experiment(path)
    configurations = sweep.expand_sweep(config)
    runs = [resolution.map_run_to_manifest(item, check_inputs=check_inputs) for item in configurations]
    run_ids = [run["run_id"] for run in runs]
    if len(run_ids) != len(set(run_ids)):
        raise PipelineConfigError("the sweep expands to duplicate run configurations")
    return runs


def save_experiment_manifest_to_disk(
    config_path: str | Path,
    manifest_path: str | Path | None = None,
    *,
    check_inputs: bool = True,
) -> tuple[Path, list[dict[str, Any]]]:
    """Resolve YAML and save the manifest used by local or Slurm workers."""
    runs = map_experiments_to_manifest(config_path, check_inputs=check_inputs)
    path = Path(manifest_path).resolve() if manifest_path else manifest.default_manifest_path(runs)
    return manifest.write_manifest(runs, path), runs


# Execution: one resolved run -> persisted artifacts -> completion.

def prepare_run(run: Mapping[str, Any]) -> Path:
    """Check prerequisites, then create or verify run_spec.json; return run_dir."""
    preparation.validate_prerequisites(run)
    preparation._ensure_run_spec(run)
    return Path(run["run_dir"])


def initialize_run(run: Mapping[str, Any]) -> Path | None:
    """Produce/reuse automatic initialization; the trainer reads its saved path."""
    if "initialization" not in run:
        return None
    initialization._check_initialization_version(run)
    directory = Path(run["run_dir"]) / "initialization"
    if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
        if not initialization._initialization_artifact_valid(run):
            raise RuntimeError(f"refusing to overwrite invalid initialization artifact: {directory}")
        print(f"[{run['run_id']}] initialization artifact is complete; skipping initialization")
    else:
        print(f"[{run['run_id']}] initialization", flush=True)
        common._run_command(initialization._initialization_command(run))
        initialization._validate_initialization(run)
    path = directory / "kmeans_model.pt"
    completion._mark_stage(Path(run["run_dir"]), "initialization", path)
    return path


def train_run(run: Mapping[str, Any]) -> Path:
    """Produce/reuse the final model and saved split consumed by later phases."""
    if training._training_artifacts_valid(run):
        print(f"[{run['run_id']}] training artifact is complete; skipping training")
    else:
        model_path = training._training_artifact_path(run)
        if run["training"]["model_kind"] == "kmeans" and model_path.exists():
            raise RuntimeError(f"refusing to overwrite invalid model artifact: {model_path}")
        print(f"[{run['run_id']}] training", flush=True)
        common._run_command(training._training_command(run))
        if not training._training_artifacts_valid(run):
            raise RuntimeError("training command finished without valid final model artifacts")
    path = training._training_artifact_path(run)
    completion._mark_stage(Path(run["run_dir"]), "training", path)
    return path


def assign_run(run: Mapping[str, Any]) -> Path | None:
    """Read the saved model and produce/reuse the selected stream's partition."""
    if not run["assignments"]["enabled"]:
        return None
    path = assignments._assignments_path(run)
    if assignments._assignment_artifact_valid(run):
        print(f"[{run['run_id']}] assignment artifact is complete; skipping assignments")
    else:
        if path.exists():
            raise RuntimeError(f"refusing to overwrite invalid assignment artifact: {path}")
        print(f"[{run['run_id']}] assignments", flush=True)
        common._run_command(assignments._assignment_command(run, path))
        if not assignments._assignment_artifact_valid(run):
            raise RuntimeError("assignment command finished without a valid bundle")
    completion._mark_stage(Path(run["run_dir"]), "assignments", path)
    return path


def evaluate_run(run: Mapping[str, Any]) -> Path | None:
    """Read saved model, split, and assignments; produce/reuse metrics.json."""
    if not run["evaluation"]["enabled"]:
        return None
    path = Path(run["run_dir"]) / "metrics.json"
    if evaluation._evaluation_artifact_valid(run):
        print(f"[{run['run_id']}] evaluation artifact is complete; skipping evaluation")
    else:
        if path.exists():
            raise RuntimeError(
                f"refusing to overwrite invalid evaluation artifact: {path}; "
                "use dalg-run-pipeline evaluate --manifest <manifest> to reevaluate"
            )
        print(f"[{run['run_id']}] evaluation", flush=True)
        metrics = evaluation.compute_metrics(run)
        evaluation.publish_metrics(run, path, metrics)
    completion._mark_stage(Path(run["run_dir"]), "evaluation", path)
    return path


def complete_run(run: Mapping[str, Any]) -> Path:
    """Mark the pipeline complete only after all configured phases succeeded."""
    run_dir = Path(run["run_dir"])
    metrics_path = run_dir / "metrics.json"
    completion._mark_stage(run_dir, "pipeline", metrics_path if metrics_path.exists() else None)
    print(f"[{run['run_id']}] pipeline complete: {run_dir}", flush=True)
    return run_dir


def execute_run(run: Mapping[str, Any]) -> Path:
    """Execute one manifest row; each phase reuses its valid saved artifacts."""
    prepare_run(run)
    initialize_run(run)
    train_run(run)
    assign_run(run)
    evaluate_run(run)
    return complete_run(run)
