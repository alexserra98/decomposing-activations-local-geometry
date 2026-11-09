"""Immutable JSONL persistence and manifest discovery."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .common import REPO_ROOT, PipelineConfigError, _canonical_json, _slug


def default_manifest_path(runs: list[dict[str, Any]]) -> Path:
    experiment_name = runs[0]["identity"]["experiment"]
    manifest_digest = hashlib.sha256(
        "\n".join(_canonical_json(run) for run in runs).encode()
    ).hexdigest()[:10]
    return REPO_ROOT / "outputs" / "experiments" / _slug(experiment_name) / (
        f"manifest_{manifest_digest}.jsonl"
    )


def read_manifest(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise PipelineConfigError(f"manifest is empty: {path}")
    return rows


def persist_manifest(runs: list[dict[str, Any]], path: str | Path) -> Path:
    path = Path(path).resolve()
    payload = "".join(_canonical_json(run) + "\n" for run in runs)
    if path.exists():
        if path.read_text() != payload:
            raise PipelineConfigError(f"refusing to overwrite a different manifest: {path}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(payload)
    tmp.replace(path)
    return path

def write_manifest(runs: list[dict[str, Any]], path: str | Path) -> Path:
    """Persist resolved runs, refusing to replace a different existing plan."""
    return persist_manifest(runs, path)
