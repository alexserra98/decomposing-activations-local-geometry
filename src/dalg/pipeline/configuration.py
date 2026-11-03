"""Helpers for loading and validating the shape of experiment YAML."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from .common import PipelineConfigError, _require_mapping


_TOP_LEVEL_KEYS = {
    "experiment",
    "experimental",
    "dataset",
    "model",
    "training",
    "initialization",
    "assignments",
    "evaluation",
    "resources",
    "sweep",
}


def read_yaml(path: str | Path) -> Any:
    return yaml.safe_load(Path(path).read_text())


def validate_sections(payload: Any) -> dict[str, Any]:
    """Check section names and types without expanding the sweep."""
    if not isinstance(payload, Mapping):
        raise PipelineConfigError("experiment YAML must contain a top-level mapping")
    payload = dict(payload)
    unknown = sorted(set(payload) - _TOP_LEVEL_KEYS)
    if unknown:
        raise PipelineConfigError(f"unknown top-level keys: {unknown}")
    for key in _TOP_LEVEL_KEYS - {"sweep"}:
        if key in payload:
            payload[key] = _require_mapping(payload[key], key)
    return payload

def load_experiment(path: str | Path) -> dict[str, Any]:
    """Read one YAML configuration and validate its section structure."""
    payload = read_yaml(path)
    return validate_sections(payload)