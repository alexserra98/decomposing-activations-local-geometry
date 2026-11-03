"""Completion markers for individual stages and the whole pipeline."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .common import _write_json_atomic


def _mark_stage(run_dir: Path, stage: str, artifact: Path | None = None) -> None:
    payload: dict[str, Any] = {"stage": stage, "completed": True}
    if artifact is not None:
        payload["artifact"] = str(artifact)
    slurm_job = os.environ.get("SLURM_JOB_ID")
    slurm_task = os.environ.get("SLURM_ARRAY_TASK_ID")
    if slurm_job:
        payload["slurm_job_id"] = slurm_job
    if slurm_task:
        payload["slurm_array_task_id"] = slurm_task
    _write_json_atomic(run_dir / f"{stage.upper()}_COMPLETED.json", payload)
