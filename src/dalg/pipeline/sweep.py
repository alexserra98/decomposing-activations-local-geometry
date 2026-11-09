"""Cartesian expansion of explicit, existing dotted configuration fields."""

from __future__ import annotations

import copy
import itertools
from typing import Any

from .common import PipelineConfigError, _require_mapping


def _set_dotted(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    if len(parts) < 2 or any(not part for part in parts):
        raise PipelineConfigError(
            f"sweep key {dotted_key!r} must be a dotted path such as 'training.seed'"
        )
    node: dict[str, Any] = config
    for part in parts[:-1]:
        child = node.get(part)
        if not isinstance(child, dict):
            raise PipelineConfigError(f"sweep path {dotted_key!r} does not exist")
        node = child
    if parts[-1] not in node:
        raise PipelineConfigError(f"sweep path {dotted_key!r} does not exist")
    node[parts[-1]] = value


def expand_configurations(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand a simple Cartesian product over explicit dotted YAML fields."""
    sweep = _require_mapping(config.get("sweep"), "sweep")
    if not sweep:
        item = copy.deepcopy(config)
        item.pop("sweep", None)
        return [item]
    keys = list(sweep)
    values = []
    for key in keys:
        axis = sweep[key]
        if not isinstance(axis, list) or not axis:
            raise PipelineConfigError(f"sweep axis {key!r} must be a non-empty list")
        values.append(axis)
    expanded = []
    for combination in itertools.product(*values):
        item = copy.deepcopy(config)
        item.pop("sweep", None)
        for key, value in zip(keys, combination):
            _set_dotted(item, key, value)
        expanded.append(item)
    return expanded

def expand_sweep(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Return one independent configuration per Cartesian sweep combination."""
    return expand_configurations(config)
