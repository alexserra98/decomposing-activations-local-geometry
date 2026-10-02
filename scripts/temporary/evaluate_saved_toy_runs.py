"""Evaluate saved toy-manifold runs without invoking initialization or training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from dalg.evaluation.toy_manifold_tiling import evaluate_toy_manifold_tiling
from dalg.pipeline import (
    _assignment_artifact_valid,
    _assignments_path,
    _evaluation_artifact_valid,
    _mark_stage,
    _training_artifacts_valid,
    _write_json_atomic,
    read_manifest,
)


def evaluate(manifest: Path, index: int) -> None:
    runs = read_manifest(manifest)
    if not 0 <= index < len(runs):
        raise ValueError(f"index outside [0, {len(runs)}): {index}")
    run = runs[index]
    run_dir = Path(run["run_dir"])
    if json.loads((run_dir / "run_spec.json").read_text()) != run:
        raise ValueError(f"saved run specification differs from manifest: {run_dir}")
    if not _training_artifacts_valid(run) or not _assignment_artifact_valid(run):
        raise ValueError(f"evaluation requires valid saved training and assignments: {run_dir}")
    cfg = run["evaluation"]
    if not cfg["enabled"] or cfg["kind"] != "toy_manifold_tiling":
        raise ValueError(f"unsupported evaluation configuration: {cfg}")
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        if not _evaluation_artifact_valid(run):
            raise ValueError(f"refusing to overwrite invalid metrics: {metrics_path}")
        print(f"Evaluation already complete: {run_dir}", flush=True)
    else:
        print(f"Evaluating saved model: {run_dir}", flush=True)
        metrics = evaluate_toy_manifold_tiling(
            run_dir,
            shard_dir=run["dataset"]["shard_dir"],
            layer=int(run["dataset"]["layer"]),
            model_kind=run["training"]["model_kind"],
            assignments_path=_assignments_path(run),
            batch_size=int(cfg["batch_size"]),
            device=cfg["device"],
            rank_threshold=float(cfg["rank_threshold"]),
            max_mean_to_manifold_distance=cfg["max_mean_to_manifold_distance"],
        )
        metrics.update(run_id=run["run_id"], identity_hash=run["identity_hash"])
        _write_json_atomic(metrics_path, metrics)
        if not _evaluation_artifact_valid(run):
            raise ValueError(f"evaluation produced invalid metrics: {metrics_path}")
    _mark_stage(run_dir, "evaluation", metrics_path)
    _mark_stage(run_dir, "pipeline", metrics_path)
    print(f"Saved {metrics_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("index", type=int)
    args = parser.parse_args()
    evaluate(args.manifest, args.index)
