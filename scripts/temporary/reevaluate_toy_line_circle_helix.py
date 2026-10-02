"""Reevaluate the saved line/circle/helix sweep with archived original reports."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import reevaluate_toy_noise_sweep as sweep


def plan(root: Path, output: Path) -> None:
    root, output = root.resolve(), output.resolve()
    rows = []
    for spec_path in sorted(root.glob("*/*/*/run_spec.json")):
        original = spec_path.parent
        spec = json.loads(spec_path.read_text())
        if Path(spec["run_dir"]).resolve() != original:
            raise ValueError(f"run directory mismatch: {spec_path}")
        kind = spec["training"]["model_kind"]
        stem = "kmeans_model" if kind == "kmeans" else "mfa_model"
        for name in (
            "config.json", "val_indices.json", f"{stem}.pt",
            f"{stem}_assignments.pt", "TRAINING_COMPLETED.json", "metrics.json",
        ):
            if not (original / name).is_file():
                raise FileNotFoundError(original / name)
        if not spec["evaluation"]["enabled"] or spec["evaluation"]["kind"] != "toy_manifold_tiling":
            raise ValueError(f"unsupported evaluation: {spec_path}")
        shards = Path(spec["dataset"]["shard_dir"])
        dataset = json.loads((shards / "config.json").read_text())
        if not (shards / dataset["manifold_metadata"]).is_file():
            raise FileNotFoundError(shards / dataset["manifold_metadata"])
        rows.append({
            "run_dir": str(original),
            "output_dir": str(output / original.relative_to(root)),
            "assignments_path": str(original / f"{stem}_assignments.pt"),
            "spec": spec,
        })
    counts = Counter(row["spec"]["training"]["model_kind"] for row in rows)
    if counts != {"hddc": 96, "kmeans": 96}:
        raise ValueError(f"unexpected completed-run inventory: {counts}")
    output.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[2]
    shutil.copytree(repo / "src" / "dalg", output / "code" / "dalg",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(__file__, output / "runner.py")
    shutil.copy2(Path(sweep.__file__), output / "reevaluate_toy_noise_sweep.py")
    for row in rows:
        original = Path(row["run_dir"])
        archive = output / "archive" / original.relative_to(root)
        archive.mkdir(parents=True)
        for name in ("run_spec.json", "metrics.json", "EVALUATION_COMPLETED.json", "PIPELINE_COMPLETED.json"):
            if (original / name).is_file():
                shutil.copy2(original / name, archive / name)
    for name in ("general_metrics.csv", "manifold_metrics.csv"):
        if (root / name).is_file():
            shutil.copy2(root / name, output / "archive" / name)
    sweep.write_json(output / "manifest.json", rows)
    sweep.write_json(output / "plan.json", {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_root": str(root), "counts": dict(counts), "skipped": 0,
        "metric": "best_intrinsic_dim_subset_of_leading_rank",
    })
    print(f"Planned {len(rows)} evaluations; archived original reports in {output / 'archive'}")
    print(output / "manifest.json")


def collect(manifest: Path) -> None:
    from dalg.pipeline import _evaluation_artifact_valid

    rows = json.loads(manifest.read_text())
    root = Path(json.loads((manifest.parent / "plan.json").read_text())["source_root"])
    for row in rows:
        original = Path(row["run_dir"])
        staged_spec = {**row["spec"], "run_dir": row["output_dir"]}
        if not _evaluation_artifact_valid(staged_spec):
            raise ValueError(f"invalid reevaluated metrics: {row['output_dir']}")
        archive = manifest.parent / "archive" / original.relative_to(root)
        for name in ("run_spec.json", "metrics.json"):
            if (original / name).read_bytes() != (archive / name).read_bytes():
                raise ValueError(f"original changed after planning: {original / name}")
    sweep.collect(manifest)
    for row in rows:
        if not _evaluation_artifact_valid(row["spec"]):
            raise ValueError(f"invalid installed metrics: {row['run_dir']}")
    print(f"Validated all {len(rows)} updated reports", flush=True)


if __name__ == "__main__":
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
        sweep.run(args.manifest, args.index)
    else:
        collect(args.manifest)
