"""Resource normalization, grouping, and Slurm array command construction."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .common import REPO_ROOT, PipelineConfigError, _canonical_json, _slug


_RESOURCE_DEFAULTS = {
    "partition": "H100",
    "account": "LADE",
    "nodes": 1,
    "ntasks_per_node": 1,
    "cpus_per_task": 8,
    "gpus": 1,
    "gpu_type": "H100",
    "memory": "80G",
    "time": "23:00:00",
    "max_parallel": 4,
}


_RESOURCE_KEYS = set(_RESOURCE_DEFAULTS)


def _normalise_resources(raw: Mapping[str, Any]) -> dict[str, Any]:
    unknown = sorted(set(raw) - _RESOURCE_KEYS)
    if unknown:
        raise PipelineConfigError(f"unknown resource parameters: {unknown}")
    resources = {**_RESOURCE_DEFAULTS, **dict(raw)}
    for key in ("nodes", "ntasks_per_node", "cpus_per_task", "gpus", "max_parallel"):
        resources[key] = int(resources[key])
    if resources["nodes"] <= 0 or resources["ntasks_per_node"] <= 0:
        raise PipelineConfigError("resource nodes and tasks must be positive")
    if resources["cpus_per_task"] <= 0 or resources["gpus"] < 0:
        raise PipelineConfigError("resource CPUs must be positive and GPUs non-negative")
    if resources["max_parallel"] <= 0:
        raise PipelineConfigError("resources.max_parallel must be positive")
    return resources


def group_by_resources(runs: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        key = _canonical_json(run["resources"])
        groups.setdefault(key, []).append(run)
    return list(groups.values())


def sbatch_command(
    manifest_path: Path,
    runs: list[dict[str, Any]],
    *,
    worker_path: Path,
    log_dir: Path | None = None,
    create_log_dir: bool = True,
) -> list[str]:
    resources = runs[0]["resources"]
    count = len(runs)
    array = f"0-{count - 1}%{resources['max_parallel']}"
    experiment = _slug(runs[0]["identity"]["experiment"])
    log_dir = log_dir if log_dir is not None else REPO_ROOT / "logs" / "experiments" / experiment
    if create_log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "sbatch",
        "--parsable",
        f"--nodes={resources['nodes']}",
        f"--ntasks-per-node={resources['ntasks_per_node']}",
        f"--cpus-per-task={resources['cpus_per_task']}",
        f"--mem={resources['memory']}",
        f"--time={resources['time']}",
        f"--array={array}",
        f"--job-name=dalg-{experiment}",
        f"--output={log_dir}/pipeline_%A_%a.out",
    ]
    if resources.get("partition"):
        command.append(f"--partition={resources['partition']}")
    if resources.get("account"):
        command.append(f"--account={resources['account']}")
    if int(resources["gpus"]) > 0:
        gpu_type = str(resources.get("gpu_type") or "").strip()
        gres = f"gpu:{gpu_type}:{resources['gpus']}" if gpu_type else f"gpu:{resources['gpus']}"
        command.append(f"--gres={gres}")
    command.extend([str(worker_path), str(manifest_path)])
    return command
