"""Stage reevaluations, then replace the original completed-run metrics."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import time


def write_json(path: Path, value: object) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def plan(root: Path, output: Path) -> None:
    root, output = root.resolve(), output.resolve()
    rows, skipped = [], []
    for spec_path in sorted(root.glob("noise_ratio_*/models/*/*/run_spec.json")):
        run_dir = spec_path.parent
        spec = json.loads(spec_path.read_text())
        kind = spec["training"]["model_kind"]
        stem = "kmeans_model" if kind == "kmeans" else "mfa_model"
        required = [
            "config.json", "val_indices.json", f"{stem}.pt",
            f"{stem}_assignments.pt", "TRAINING_COMPLETED.json", "metrics.json",
        ]
        missing = [name for name in required if not (run_dir / name).is_file()]
        if missing:
            skipped.append({"run_dir": str(run_dir), "model_kind": kind, "missing": missing})
            continue
        if spec["evaluation"]["kind"] != "toy_manifold_tiling":
            raise ValueError(f"unsupported evaluation: {spec_path}")
        shard_dir = Path(spec["dataset"]["shard_dir"])
        dataset = json.loads((shard_dir / "config.json").read_text())
        if not (shard_dir / dataset["manifold_metadata"]).is_file():
            raise FileNotFoundError(shard_dir / dataset["manifold_metadata"])
        rows.append({
            "run_dir": str(run_dir),
            "output_dir": str(output / run_dir.relative_to(root)),
            "assignments_path": str(run_dir / f"{stem}_assignments.pt"),
            "spec": spec,
        })
    counts = Counter(row["spec"]["training"]["model_kind"] for row in rows)
    # This launcher is scoped to the inspected 228-run experiment.
    if counts != {"kmeans": 108, "mfa": 12, "hddc": 94}:
        raise ValueError(f"unexpected completed-run inventory: {counts}")
    if len(skipped) != 14 or any(row["model_kind"] != "hddc" for row in skipped):
        raise ValueError(f"unexpected excluded runs: {skipped}")
    output.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[2]
    shutil.copytree(repo / "src" / "dalg", output / "code" / "dalg",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(__file__, output / "runner.py")
    write_json(output / "manifest.json", rows)
    write_json(output / "skipped.json", skipped)
    write_json(output / "plan.json", {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_root": str(root), "counts": dict(counts), "skipped": len(skipped),
    })
    print(f"Planned {len(rows)} evaluations; excluded {len(skipped)} HDDC runs")
    print(output / "manifest.json")


def run(manifest: Path, index: int) -> None:
    from dalg.evaluation.toy_manifold_tiling import evaluate_toy_manifold_tiling

    rows = json.loads(manifest.read_text())
    if not 0 <= index < len(rows):
        raise ValueError(f"index outside [0, {len(rows)}): {index}")
    row = rows[index]
    spec = row["spec"]
    output = Path(row["output_dir"])
    if output.exists():
        raise FileExistsError(f"refusing to overwrite evaluation output: {output}")
    evaluation = spec["evaluation"]
    print(f"Evaluating {index}: {row['run_dir']}", flush=True)
    started = time.monotonic()
    metrics = evaluate_toy_manifold_tiling(
        row["run_dir"],
        shard_dir=spec["dataset"]["shard_dir"],
        layer=int(spec["dataset"]["layer"]),
        model_kind=spec["training"]["model_kind"],
        assignments_path=row["assignments_path"],
        batch_size=int(evaluation["batch_size"]),
        device=evaluation["device"],
        rank_threshold=float(evaluation["rank_threshold"]),
        max_mean_to_manifold_distance=evaluation["max_mean_to_manifold_distance"],
    )
    metrics.update(run_id=spec["run_id"], identity_hash=spec["identity_hash"])
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "run_spec.json", spec)
    write_json(output / "metrics.json", metrics)
    write_json(output / "EVALUATION_COMPLETED.json", {
        "source_run_dir": row["run_dir"], "manifest": str(manifest.resolve()),
        "index": index, "elapsed_seconds": time.monotonic() - started,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    })
    print(f"Saved {output / 'metrics.json'}", flush=True)


def collect(manifest: Path) -> None:
    from dalg.analysis.aggregate_metrics import aggregate_metrics

    rows = json.loads(manifest.read_text())
    source_root = Path(json.loads((manifest.parent / "plan.json").read_text())["source_root"])
    reports = []
    for row in rows:
        staged = Path(row["output_dir"])
        original = Path(row["run_dir"])
        completed = json.loads((staged / "EVALUATION_COMPLETED.json").read_text())
        metrics_text = (staged / "metrics.json").read_text()
        metrics = json.loads(metrics_text)
        original_spec = json.loads((original / "run_spec.json").read_text())
        for key in ("run_id", "identity_hash"):
            if metrics[key] != row["spec"][key] or metrics[key] != original_spec[key]:
                raise ValueError(f"reevaluation identity mismatch: {original}")
        if original.relative_to(source_root) != staged.relative_to(manifest.parent):
            raise ValueError(f"reevaluation output path mismatch: {original}")
        if completed["source_run_dir"] != str(original):
            raise ValueError(f"reevaluation completion mismatch: {original}")
        marker = {
            "artifact": str(original / "metrics.json"),
            "completed": True, "stage": "evaluation",
            "completed_at": completed["completed_at"],
            "slurm_job_id": completed["slurm_job_id"],
            "slurm_array_task_id": str(completed["index"]),
            "reevaluation_manifest": str(manifest.resolve()),
        }
        reports.append((original, metrics_text, marker))

    # The staged tree mirrors the original relative paths and run specifications.
    general, manifolds = aggregate_metrics(manifest.parent)
    if len(general) != len(rows):
        raise ValueError(f"expected {len(rows)} aggregate rows, got {len(general)}")
    for original, metrics_text, marker in reports:
        for name, text in (
            ("metrics.json", metrics_text),
            ("EVALUATION_COMPLETED.json", json.dumps(marker, indent=2) + "\n"),
        ):
            temporary = original / f".{name}.reevaluation.tmp"
            temporary.write_text(text)
            temporary.replace(original / name)
    for frame, name in ((general, "general_metrics.csv"), (manifolds, "manifold_metrics.csv")):
        temporary = source_root / f".{name}.reevaluation.tmp"
        frame.to_csv(temporary, index=False)
        temporary.replace(source_root / name)
    write_json(manifest.parent / "ORIGINAL_RESULTS_UPDATED.json", {
        "source_root": str(source_root), "updated_runs": len(reports),
        "general_rows": len(general), "manifold_rows": len(manifolds),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    })
    print(f"Replaced {len(reports)} original metrics and rebuilt both CSVs in {source_root}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    planner = commands.add_parser("plan")
    planner.add_argument("root", type=Path)
    planner.add_argument("output", type=Path)
    worker = commands.add_parser("run")
    worker.add_argument("manifest", type=Path)
    worker.add_argument("index", type=int)
    collector = commands.add_parser("collect")
    collector.add_argument("manifest", type=Path)
    args = parser.parse_args()
    if args.command == "plan":
        plan(args.root, args.output)
    elif args.command == "run":
        run(args.manifest, args.index)
    else:
        collect(args.manifest)


if __name__ == "__main__":
    main()
