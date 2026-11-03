"""Shared serialization, CLI encoding, and filesystem execution primitives."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]


class PipelineConfigError(ValueError):
    """Raised when an experiment cannot be resolved safely."""


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise PipelineConfigError(f"{name} must be a mapping")
    return dict(value)


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip()).strip("-").lower()
    return slug or "experiment"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _action_by_dest(parser: argparse.ArgumentParser) -> dict[str, argparse.Action]:
    return {
        action.dest: action
        for action in parser._actions
        if action.dest != argparse.SUPPRESS and action.option_strings
    }


def _preferred_option(action: argparse.Action, *, negative: bool = False) -> str:
    long_options = [item for item in action.option_strings if item.startswith("--")]
    options = long_options or list(action.option_strings)
    if negative:
        candidates = [item for item in options if item.startswith("--no-")]
    else:
        candidates = [item for item in options if not item.startswith("--no-")]
    if not candidates:
        raise PipelineConfigError(f"cannot encode CLI option for {action.dest!r}")
    return candidates[0]


def _mapping_to_argv(
    parser: argparse.ArgumentParser,
    values: Mapping[str, Any],
) -> list[str]:
    actions = _action_by_dest(parser)
    unknown = sorted(set(values) - set(actions))
    if unknown:
        raise PipelineConfigError(f"unknown training parameters: {unknown}")
    argv: list[str] = []
    for key, value in values.items():
        if value is None:
            continue
        action = actions[key]
        if isinstance(action, argparse.BooleanOptionalAction):
            if not isinstance(value, bool):
                raise PipelineConfigError(f"{key} must be true or false")
            argv.append(_preferred_option(action, negative=not value))
        elif action.nargs == 0:
            if value == action.const:
                argv.append(_preferred_option(action))
            elif value != action.default:
                raise PipelineConfigError(
                    f"{key}={value!r} cannot be represented by its CLI flag"
                )
        else:
            argv.append(_preferred_option(action))
            if isinstance(value, (list, tuple)):
                argv.extend(str(item) for item in value)
            else:
                argv.append(str(value))
    return argv


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def _run_command(command: list[str]) -> None:
    env = dict(os.environ)
    source_path = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = source_path + os.pathsep + env.get("PYTHONPATH", "")
    print(f"$ {shlex.join(command)}", flush=True)
    subprocess.run(command, check=True, cwd=REPO_ROOT, env=env)


def _model_stem(run: Mapping[str, Any]) -> str:
    return "kmeans_model" if run["training"]["model_kind"] == "kmeans" else "mfa_model"
