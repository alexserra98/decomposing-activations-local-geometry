"""Collect saved experiment metrics into run and per-manifold tables."""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import pandas as pd


_EXCLUDED_DIRS = {"centroids", "archive", "archived", "checkpoints", "snapshots"}
_IDENTIFIERS = ["source_path", "run_id", "evaluation_id"]
_ALIASES = {
    "model_kind": "config.training.model_kind",
    "K": "config.training.arguments.K",
    "surgery_threshold": "config.training.arguments.surgery_threshold",
}


def _read_object(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f"{path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _flatten(value: dict, prefix: str = "") -> dict:
    result = {}

    def visit(mapping: dict, parent: str) -> None:
        for key, item in mapping.items():
            name = f"{parent}.{key}" if parent else key
            if isinstance(item, dict) and item:
                visit(item, name)
            else:
                if name in result:
                    raise ValueError(f"flattened-column collision: {name}")
                result[name] = (
                    json.dumps(item, sort_keys=True)
                    if isinstance(item, (list, dict)) else item
                )

    visit(value, prefix)
    return result


def _merge(target: dict, values: dict) -> None:
    for key, value in values.items():
        if key in target and target[key] is not None and value is not None:
            if target[key] != value:
                raise ValueError(f"conflicting values for {key}")
        if key not in target or target[key] is None:
            target[key] = value


def _report_rows(path: Path, root: Path) -> tuple[dict, list[dict]]:
    report = _read_object(path)
    spec_path = path.with_name("run_spec.json")
    spec = _read_object(spec_path) if spec_path.exists() else {}
    try:
        config = _flatten(spec, "config")
    except ValueError as exc:
        raise ValueError(f"{spec_path}: {exc}") from exc
    try:
        per_manifold = report.pop("per_manifold", [])
        if not isinstance(per_manifold, list):
            raise ValueError("per_manifold must be a list")
        general = _flatten(report)
        if set(general) & set(config):
            raise ValueError("flattened-column collision between metrics and config")
        source = path.relative_to(root).as_posix()
        context = {
            "source_path": source,
            "run_id": report.get("run_id"),
            "evaluation_id": source,
        }
        _merge(context, {"run_id": spec.get("run_id")})
        if context["run_id"] is None:
            context["run_id"] = path.parent.relative_to(root).as_posix()
        for name, config_name in _ALIASES.items():
            context[name] = general.get(name)
            _merge(context, {name: config.get(config_name)})
        context.update(config)
        _merge(general, context)

        rows = []
        seen = set()
        for manifold in per_manifold:
            if not isinstance(manifold, dict):
                raise ValueError("per_manifold entries must be objects")
            manifold_id = manifold.get("manifold_id")
            if type(manifold_id) is not int:
                raise ValueError("manifold_id must be an integer")
            if manifold_id in seen:
                raise ValueError(f"duplicate manifold_id: {manifold_id}")
            seen.add(manifold_id)
            row = _flatten(manifold)
            if any(name.startswith("config.") for name in row):
                raise ValueError("flattened-column collision between manifold and config")
            _merge(row, context)
            rows.append(row)
        rows.sort(key=lambda row: row["manifold_id"])
        return general, rows
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from exc


def _dataframe(rows: list[dict], identifiers: list[str]) -> pd.DataFrame:
    columns = set(_ALIASES)
    for row in rows:
        columns.update(row)
    ordered = identifiers + sorted(columns - set(identifiers))
    # Construct nullable arrays directly: float inference would round large ints.
    return pd.DataFrame({name: pd.array([row.get(name) for row in rows]) for name in ordered})


def aggregate_metrics(
    folder: str | Path, *, skip_regex: str | None = None,
    skip_paths: list[str | Path] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read reports without writing files; return general and manifold DataFrames.

    Each report contributes one general row and one row per saved manifold.
    Nested fields use dotted columns and run specifications use ``config.*``.
    ``evaluation_id`` (the relative report path) joins the tables. Deprecated
    centroid reports, history directories, and nested directory symlinks are
    excluded. ``skip_regex`` additionally skips subdirectories whose names
    match via ``re.search``. ``skip_paths`` skips specific subdirectories using
    absolute paths or paths relative to ``folder``. Both filters exclude entire
    subtrees; the root folder is never filtered. Invalid reports raise
    ValueError with the source path.
    """
    root = Path(folder).resolve()
    if not root.is_dir():
        raise ValueError(f"directory does not exist: {root}")
    try:
        skip_pattern = re.compile(skip_regex) if skip_regex is not None else None
    except re.error as exc:
        raise ValueError(f"invalid skip_regex {skip_regex!r}: {exc}") from exc
    excluded_paths = {(root / path).resolve() for path in (skip_paths or [])}

    def walk_error(exc: OSError) -> None:
        raise exc

    paths = []
    for directory, dirs, files in os.walk(root, followlinks=False, onerror=walk_error):
        dirs[:] = [
            name for name in dirs
            if name not in _EXCLUDED_DIRS and not (Path(directory) / name).is_symlink()
            and Path(directory) / name not in excluded_paths
            and (skip_pattern is None or skip_pattern.search(name) is None)
        ]
        if "metrics.json" in files:
            paths.append(Path(directory) / "metrics.json")
    if not paths:
        raise ValueError(f"{root}: no metrics.json reports")

    general_rows, manifold_rows = [], []
    for path in sorted(paths, key=lambda path: path.relative_to(root).as_posix()):
        general, manifolds = _report_rows(path, root)
        general_rows.append(general)
        manifold_rows.extend(manifolds)
    return (
        _dataframe(general_rows, _IDENTIFIERS),
        _dataframe(manifold_rows, _IDENTIFIERS + ["manifold_id"]),
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path, help="Experiment folder to read and save CSVs inside")
    parser.add_argument(
        "--skip-regex",
        help="Skip subfolders whose names match this regex (re.search), including all descendants",
    )
    parser.add_argument(
        "--skip-paths", nargs="+", type=Path,
        help="Subfolder paths to skip, including descendants (absolute or relative to folder)",
    )
    args = parser.parse_args(argv)
    try:
        general, manifolds = aggregate_metrics(
            args.folder, skip_regex=args.skip_regex, skip_paths=args.skip_paths,
        )
        paths = [args.folder / "general_metrics.csv", args.folder / "manifold_metrics.csv"]
        for frame, path in zip((general, manifolds), paths):
            frame.to_csv(path, index=False)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"{exc}\n")
    print(f"Aggregated {len(general)} reports into {len(general)} general rows and {len(manifolds)} manifold rows")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
