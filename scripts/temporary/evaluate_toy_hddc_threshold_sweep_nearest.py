"""Re-evaluate an existing HDDC threshold sweep without an association cutoff."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch

from dalg.evaluation.toy_manifold_tiling import evaluate_toy_manifold_tiling
from dalg.pipeline import _write_json_atomic, read_manifest, write_manifest


SOURCE_METRICS_NAME = "metrics.json"
OUTPUT_METRICS_NAME = "metrics_nearest.json"
EXPECTED_RUNS = 12
EXPECTED_K = 1000
EXPECTED_ROWS = 150_000
SOURCE_CUTOFF = 0.1
FLOAT_TOLERANCE = 1e-10


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _same_path(left: str | Path, right: str | Path) -> bool:
    return Path(left).expanduser().resolve() == Path(right).expanduser().resolve()


def _validate_source_run(run_dir: Path) -> dict[str, Any]:
    required = [
        run_dir / "run_spec.json",
        run_dir / "config.json",
        run_dir / "val_indices.json",
        run_dir / "mfa_model.pt",
        run_dir / "mfa_model_assignments.pt",
        run_dir / SOURCE_METRICS_NAME,
    ]
    missing = [str(path) for path in required if not path.is_file() or not path.stat().st_size]
    if missing:
        raise FileNotFoundError(f"missing or empty source artifacts: {missing}")

    run_spec = _load_json(run_dir / "run_spec.json")
    if not _same_path(run_spec["run_dir"], run_dir):
        raise ValueError(f"run_spec.json points to a different run directory: {run_dir}")
    if run_spec["training"]["model_kind"] != "hddc":
        raise ValueError(f"source run is not HDDC: {run_dir}")
    if int(run_spec["training"]["arguments"]["K"]) != EXPECTED_K:
        raise ValueError(f"source run does not have K={EXPECTED_K}: {run_dir}")

    evaluation = run_spec["evaluation"]
    if evaluation["kind"] != "toy_manifold_tiling" or not evaluation["enabled"]:
        raise ValueError(f"source run lacks toy-manifold evaluation: {run_dir}")
    if not math.isclose(float(evaluation["max_mean_to_manifold_distance"]), SOURCE_CUTOFF):
        raise ValueError(f"source run does not use cutoff {SOURCE_CUTOFF}: {run_dir}")

    shard_dir = Path(run_spec["dataset"]["shard_dir"])
    shard_config = _load_json(shard_dir / "config.json")
    if shard_config.get("source_kind") != "toy_manifolds":
        raise ValueError(f"source dataset is not a toy-manifold dataset: {shard_dir}")
    if int(shard_config["num_rows"]) != EXPECTED_ROWS:
        raise ValueError(f"source dataset does not have {EXPECTED_ROWS} rows: {shard_dir}")

    assignments_path = run_dir / "mfa_model_assignments.pt"
    bundle = torch.load(assignments_path, map_location="cpu", mmap=True, weights_only=True)
    assignments = bundle["assignments"].reshape(-1).long()
    cluster_sizes = bundle["cluster_sizes"].reshape(-1).long()
    if int(bundle["K"]) != EXPECTED_K or cluster_sizes.numel() != EXPECTED_K:
        raise ValueError(f"assignment K does not match {EXPECTED_K}: {assignments_path}")
    if assignments.numel() != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS} assignments: {assignments_path}")
    if assignments.numel() and (
        int(assignments.min()) < 0 or int(assignments.max()) >= EXPECTED_K
    ):
        raise ValueError(f"assignments lie outside [0, {EXPECTED_K - 1}]: {assignments_path}")
    if not torch.equal(torch.bincount(assignments, minlength=EXPECTED_K), cluster_sizes):
        raise ValueError(f"cluster sizes are inconsistent with assignments: {assignments_path}")

    source_metrics = _load_json(run_dir / SOURCE_METRICS_NAME)
    if source_metrics.get("identity_hash") != run_spec["identity_hash"]:
        raise ValueError(f"source metrics identity does not match run_spec.json: {run_dir}")
    if int(source_metrics["K"]) != EXPECTED_K:
        raise ValueError(f"source metrics K does not match {EXPECTED_K}: {run_dir}")
    if int(source_metrics["dataset"]["selected_rows"]) != EXPECTED_ROWS:
        raise ValueError(f"source metrics do not cover {EXPECTED_ROWS} rows: {run_dir}")
    if not math.isclose(
        float(source_metrics["association"]["max_mean_to_manifold_distance"]),
        SOURCE_CUTOFF,
    ):
        raise ValueError(f"source metrics do not use cutoff {SOURCE_CUTOFF}: {run_dir}")
    return run_spec


def _manifest_row(run_dir: Path, run_spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "source_run_id": run_spec["run_id"],
        "source_identity_hash": run_spec["identity_hash"],
        "surgery_threshold": float(run_spec["training"]["arguments"]["surgery_threshold"]),
        "run_dir": str(run_dir.resolve()),
        "shard_dir": str(Path(run_spec["dataset"]["shard_dir"]).resolve()),
        "layer": int(run_spec["dataset"]["layer"]),
        "assignments_path": str((run_dir / "mfa_model_assignments.pt").resolve()),
        "source_metrics_path": str((run_dir / SOURCE_METRICS_NAME).resolve()),
        "output_metrics_path": str((run_dir / OUTPUT_METRICS_NAME).resolve()),
        "batch_size": int(run_spec["evaluation"]["batch_size"]),
        "device": str(run_spec["evaluation"]["device"]),
        "rank_threshold": float(run_spec["evaluation"]["rank_threshold"]),
        "source_cutoff": SOURCE_CUTOFF,
        "target_association": "unique_nearest_exact_projection",
    }


def prepare_manifest(sweep_dir: Path, manifest_path: Path) -> None:
    run_dirs = sorted(
        path for path in sweep_dir.iterdir() if path.is_dir() and (path / "run_spec.json").is_file()
    )
    if len(run_dirs) != EXPECTED_RUNS:
        raise ValueError(f"expected {EXPECTED_RUNS} runs, found {len(run_dirs)} in {sweep_dir}")
    rows = [_manifest_row(run_dir, _validate_source_run(run_dir)) for run_dir in run_dirs]
    rows.sort(key=lambda row: row["surgery_threshold"])
    thresholds = [row["surgery_threshold"] for row in rows]
    if len(set(thresholds)) != EXPECTED_RUNS:
        raise ValueError(f"surgery thresholds are not unique: {thresholds}")
    written = write_manifest(rows, manifest_path)
    print(f"Validated {len(rows)} source runs")
    print(f"Surgery thresholds: {thresholds}")
    print(f"Evaluation manifest: {written}")


def _assert_close(label: str, actual: float, expected: float) -> None:
    if not math.isclose(
        float(actual), float(expected), rel_tol=FLOAT_TOLERANCE, abs_tol=FLOAT_TOLERANCE
    ):
        raise ValueError(f"{label} changed: source={expected}, nearest={actual}")


def validate_nearest(row: dict[str, Any], metrics: dict[str, Any]) -> None:
    source = _load_json(Path(row["source_metrics_path"]))
    if metrics.get("schema_version") != 1 or metrics.get("evaluation") != "toy_manifold_tiling":
        raise ValueError("unexpected nearest-association metric schema")
    if int(metrics["K"]) != EXPECTED_K or metrics.get("model_kind") != "hddc":
        raise ValueError("nearest-association model metadata does not match the source run")
    if metrics.get("source_run_id") != row["source_run_id"]:
        raise ValueError("nearest-association source_run_id does not match the manifest")
    if metrics.get("source_identity_hash") != row["source_identity_hash"]:
        raise ValueError("nearest-association source identity does not match the manifest")
    if metrics.get("evaluation_variant", {}).get("association") != row["target_association"]:
        raise ValueError("nearest-association provenance is missing")

    association = metrics["association"]
    if association != {
        "rule": "unique_nearest_exact_projection",
        "max_mean_to_manifold_distance": None,
        "associated_components": EXPECTED_K - int(association["ambiguous_components"]),
        "outside_cutoff_components": 0,
        "ambiguous_components": int(association["ambiguous_components"]),
    }:
        raise ValueError(f"unexpected nearest-association summary: {association}")
    if sum(int(item["components"]["associated"]) for item in metrics["per_manifold"]) != int(
        association["associated_components"]
    ):
        raise ValueError("per-manifold association counts do not match the global count")

    if metrics["dataset"] != source["dataset"]:
        raise ValueError("dataset summary changed during evaluation-only execution")
    if metrics["components"] != source["components"]:
        raise ValueError("live/dead component counts changed during evaluation-only execution")
    if metrics["clustering"] != source["clustering"]:
        raise ValueError("clustering metrics changed during evaluation-only execution")
    if metrics["bic"]["parameters"] != source["bic"]["parameters"]:
        raise ValueError("BIC parameter count changed during evaluation-only execution")
    for split in ("train", "validation"):
        _assert_close(f"nll.{split}", metrics["nll"][split], source["nll"][split])
    _assert_close("bic.value", metrics["bic"]["value"], source["bic"]["value"])


def evaluate_row(manifest_path: Path, index: int) -> None:
    rows = read_manifest(manifest_path)
    if not 0 <= index < len(rows):
        raise IndexError(f"manifest index {index} is outside [0, {len(rows) - 1}]")
    row = rows[index]
    run_dir = Path(row["run_dir"])
    run_spec = _validate_source_run(run_dir)
    if run_spec["identity_hash"] != row["source_identity_hash"]:
        raise ValueError("source run identity changed after the evaluation manifest was made")
    output_path = Path(row["output_metrics_path"])
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing nearest metrics: {output_path}")

    metrics = evaluate_toy_manifold_tiling(
        run_dir,
        shard_dir=row["shard_dir"],
        layer=int(row["layer"]),
        model_kind="hddc",
        assignments_path=row["assignments_path"],
        batch_size=int(row["batch_size"]),
        device=row["device"],
        rank_threshold=float(row["rank_threshold"]),
        max_mean_to_manifold_distance=None,
    )
    metrics["source_run_id"] = row["source_run_id"]
    metrics["source_identity_hash"] = row["source_identity_hash"]
    metrics["evaluation_variant"] = {
        "association": row["target_association"],
        "source_max_mean_to_manifold_distance": row["source_cutoff"],
        "source_metrics": SOURCE_METRICS_NAME,
        "surgery_threshold": float(row["surgery_threshold"]),
    }
    metrics["dataset"]["shard_dir"] = run_spec["dataset"]["shard_dir"]
    validate_nearest(row, metrics)
    _write_json_atomic(output_path, metrics)
    print(f"Saved threshold={row['surgery_threshold']:g}: {output_path}")


def validate_outputs(manifest_path: Path) -> None:
    rows = read_manifest(manifest_path)
    for row in rows:
        _validate_source_run(Path(row["run_dir"]))
        output_path = Path(row["output_metrics_path"])
        if not output_path.is_file() or not output_path.stat().st_size:
            raise FileNotFoundError(f"missing nearest metrics: {output_path}")
        validate_nearest(row, _load_json(output_path))
        print(f"OK threshold={row['surgery_threshold']:g}: {output_path}")
    print(f"Validated {len(rows)} nearest-association metric files")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--sweep-dir", type=Path, required=True)
    prepare.add_argument("--manifest", type=Path, required=True)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--manifest", type=Path, required=True)
    evaluate.add_argument("--index", type=int, required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        prepare_manifest(args.sweep_dir, args.manifest)
    elif args.command == "evaluate":
        evaluate_row(args.manifest, args.index)
    else:
        validate_outputs(args.manifest)


if __name__ == "__main__":
    main()
