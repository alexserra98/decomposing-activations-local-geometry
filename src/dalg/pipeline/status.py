"""Read stage artifacts and report completion without executing work."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .initialization import _initialization_artifact_valid
from .training import _training_artifacts_valid
from .assignments import _assignment_artifact_valid
from .evaluation import _evaluation_artifact_valid


def pipeline_status(runs: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        run_dir = Path(run["run_dir"])
        initialization_complete = _initialization_artifact_valid(run)
        training_complete = _training_artifacts_valid(run)
        assignments_complete = (
            not run["assignments"]["enabled"]
            or _assignment_artifact_valid(run)
        )
        evaluation_complete = (
            not run["evaluation"]["enabled"] or _evaluation_artifact_valid(run)
        )
        rows.append(
            {
                "run_id": run["run_id"],
                "initialization": initialization_complete,
                "training": training_complete,
                "assignments": assignments_complete,
                "evaluation": evaluation_complete,
                "pipeline": (
                    initialization_complete
                    and training_complete
                    and assignments_complete
                    and evaluation_complete
                    and (run_dir / "PIPELINE_COMPLETED.json").is_file()
                ),
                "run_dir": str(run_dir),
            }
        )
    return rows
