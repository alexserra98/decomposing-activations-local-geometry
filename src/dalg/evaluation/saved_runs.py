"""Recompute metrics in saved pipeline runs and refresh experiment summaries."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import shlex
import subprocess
from uuid import uuid4
from pathlib import Path

import torch
import yaml

from dalg.data.shard_activations import load_meta_index
from dalg.data.subset_spec import resolve_spec_positions, split_shard_dir_spec
from dalg.pipeline import (
    REPO_ROOT, _EVALUATION_DEFAULTS,
    _assignment_artifact_valid, _assignments_path, _canonical_json,
    _evaluation_artifact_valid, _evaluation_metrics_valid, _mark_stage, _model_stem,
    _normalise_resources, _training_artifacts_valid, _write_json_atomic,
    group_by_resources, read_manifest, sbatch_command,
)


def _read_object(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _settings(run: dict, overrides: dict) -> dict:
    cfg = {**_EVALUATION_DEFAULTS, **run.get("evaluation", {}), **overrides}
    cfg["enabled"] = True
    cfg["kind"] = cfg["kind"] or "toy_manifold_tiling"
    if cfg["kind"] != "toy_manifold_tiling":
        raise ValueError(f"unsupported evaluation: {cfg['kind']}")
    if int(cfg["batch_size"]) <= 0:
        raise ValueError("evaluation batch_size must be positive")
    torch.device(cfg["device"])
    threshold = float(cfg["rank_threshold"])
    if not math.isfinite(threshold) or (
        run["training"]["model_kind"] not in {"kmeans", "hddc"} and threshold <= 0
    ):
        raise ValueError("invalid evaluation rank_threshold")
    cutoff = cfg["max_mean_to_manifold_distance"]
    if cutoff is not None and (not math.isfinite(float(cutoff)) or float(cutoff) <= 0):
        raise ValueError("evaluation distance cutoff must be finite and positive, or none")
    return cfg


def _resources(run: dict, cfg: dict, overrides: dict) -> dict:
    resources = _normalise_resources({
        **run["resources"], "nodes": 1, "ntasks_per_node": 1,
        "gpus": int(torch.device(cfg["device"]).type == "cuda"), **overrides,
    })
    if resources["nodes"] != 1 or resources["ntasks_per_node"] != 1:
        raise ValueError("evaluation requires one node and one task")
    if torch.device(cfg["device"]).type == "cuda" and resources["gpus"] < 1:
        raise ValueError("CUDA evaluation requires a GPU allocation")
    return resources


def _prerequisites(run: dict) -> tuple[list[str], dict | None]:
    """Report absent run artifacts; reject existing invalid artifacts."""
    directory = Path(run["run_dir"])
    spec_path = directory / "run_spec.json"
    spec = _read_object(spec_path) if spec_path.exists() else None
    if spec is not None and any(spec.get(k) != run.get(k) for k in ("identity", "identity_hash", "run_id")):
        raise ValueError(f"source run identity mismatch: {spec_path}")
    required = [spec_path, directory / "config.json", directory / "val_indices.json",
                directory / f"{_model_stem(run)}.pt", _assignments_path(run)]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        if not required[3].is_file() and (directory / "mfa_model_shards.json").exists():
            missing.append("unsupported shard-only model: consolidated checkpoint required")
        return missing, spec
    _read_object(directory / "config.json")
    split = _read_object(directory / "val_indices.json")
    if not isinstance(split.get("val_global_rows"), list):
        raise ValueError(f"invalid validation split: {directory / 'val_indices.json'}")
    root, subset = split_shard_dir_spec(run["dataset"]["shard_dir"])
    dataset = _read_object(root / "config.json")
    if dataset.get("source_kind") != "toy_manifolds" or dataset.get("window") != 1 or dataset.get("drop_prefix", 0) != 0:
        raise ValueError(f"incompatible toy-manifold dataset: {root}")
    metadata = torch.load(root / dataset["manifold_metadata"], map_location="cpu", weights_only=True)
    meta = load_meta_index(root, layer=run["dataset"]["layer"])
    if metadata["row_manifold_ids"].numel() != len(meta):
        raise ValueError(f"toy-manifold metadata does not match the stream: {root}")
    positions = resolve_spec_positions(meta, subset, window=1, drop_prefix=0)
    if not _training_artifacts_valid(run):
        raise ValueError(f"invalid saved model artifacts: {directory}")
    from dalg.evaluation.toy_manifold_tiling import _load_model
    model = _load_model(directory, run["training"]["model_kind"])
    if model.K != run["training"]["arguments"]["K"]:
        raise ValueError(f"checkpoint K does not match the manifest: {directory}")
    del model
    if not _assignment_artifact_valid(run):
        raise ValueError(f"invalid saved assignments: {_assignments_path(run)}")
    bundle = torch.load(_assignments_path(run), map_location="cpu", mmap=True, weights_only=True)
    if bundle["assignments"].numel() != len(positions) or bundle.get("subset_spec") != subset:
        raise ValueError(f"assignments do not cover the selected stream: {_assignments_path(run)}")
    return [], spec


def plan_evaluation(manifests, *, indices=None, overrides=None, resources=None) -> dict:
    paths = [Path(path).resolve() for path in manifests]
    rows = [read_manifest(path) for path in paths]
    selected = None
    if indices is not None:
        selected = set()
        for token in indices:
            parts = str(token).split(":")
            if len(parts) == 1 and len(paths) == 1:
                parts.insert(0, "0")
            if len(parts) != 2:
                raise ValueError("multiple manifests require indices as manifest-position:row-index")
            m, r = map(int, parts)
            if not 0 <= m < len(rows) or not 0 <= r < len(rows[m]):
                raise ValueError(f"manifest index out of range: {token}")
            selected.add((m, r))
    unique = {}
    for m, manifest_rows in enumerate(rows):
        for r, source in enumerate(manifest_rows):
            if selected is not None and (m, r) not in selected:
                continue
            run = copy.deepcopy(source)
            directory = Path(run["run_dir"]).resolve()
            run["run_dir"] = str(directory)
            cfg = _settings(run, overrides or {})
            ref = {"manifest": str(paths[m]), "manifest_position": m, "row_index": r}
            if str(directory) in unique:
                previous = unique[str(directory)]
                if any(previous["run"].get(k) != run.get(k) for k in ("identity", "identity_hash", "run_id", "dataset", "training")) or previous["evaluation"] != cfg:
                    raise ValueError(f"conflicting references to saved run: {directory}")
                previous["sources"].append(ref)
                continue
            unique[str(directory)] = {
                "run": run, "evaluation": cfg, "sources": [ref],
                "resources": _resources(run, cfg, resources or {}),
            }
    eligible, skipped = [], []
    for entry in unique.values():
        missing, spec = _prerequisites(entry["run"])
        if missing:
            skipped.append({"run_dir": entry["run"]["run_dir"], "sources": entry["sources"], "reasons": missing})
            continue
        index = len(eligible)
        entry.update(index=index, source_spec=spec)
        eligible.append(entry)
    summary_dirs = set()
    for entry in eligible:
        directory = Path(entry["run"]["run_dir"])
        existing = [parent for parent in directory.parents if any(
            (parent / name).is_file() for name in ("general_metrics.csv", "manifold_metrics.csv")
        )]
        summary_dirs.update(existing or [directory.parent])
    job_dir = REPO_ROOT / "outputs" / "evaluations" / uuid4().hex
    plan = {
        "version": 2, "manifests": list(map(str, paths)), "job_dir": str(job_dir),
        "runs": eligible, "skipped": skipped,
        "summary_dirs": sorted(map(str, summary_dirs)),
    }
    plan["plan_hash"] = hashlib.sha256(_canonical_json(plan).encode()).hexdigest()
    return plan


def save_plan(plan: dict) -> Path:
    """Save an immutable job plan; reports remain in the original run directories."""
    path = Path(plan["job_dir"]) / "evaluation_plan.json"
    if path.exists():
        if _read_object(path) != plan:
            raise ValueError(f"conflicting evaluation plan: {path}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(path, plan)
    _write_json_atomic(path.parent / "skipped_runs.json", {"skipped": plan["skipped"]})
    return path


def read_evaluation_plan(path: Path) -> dict:
    plan = _read_object(path)
    payload = {key: value for key, value in plan.items() if key != "plan_hash"}
    if plan.get("version") != 2 or plan.get("plan_hash") != hashlib.sha256(_canonical_json(payload).encode()).hexdigest():
        raise ValueError(f"invalid or modified evaluation plan: {path}")
    if path.resolve() != Path(plan["job_dir"]) / "evaluation_plan.json":
        raise ValueError(f"evaluation plan location differs from its job directory: {path}")
    return plan


def _provenance(plan: dict, entry: dict) -> dict:
    return {"plan_hash": plan["plan_hash"], "source_run_dir": entry["run"]["run_dir"],
            "sources": entry["sources"]}


def _result_valid(plan: dict, entry: dict) -> bool:
    """Require a valid report produced by this invocation before collection."""
    directory = Path(entry["run"]["run_dir"])
    metrics_path = directory / "metrics.json"
    if not metrics_path.is_file():
        return False
    effective = {**entry["run"], "evaluation": entry["evaluation"]}
    if not _evaluation_artifact_valid(effective):
        raise ValueError(f"invalid evaluation: {metrics_path}")
    metrics = _read_object(metrics_path)
    if (metrics.get("evaluation_config") != entry["evaluation"]
        or metrics.get("evaluation_provenance") != _provenance(plan, entry)
        or metrics.get("run_id") != entry["run"]["run_id"]
        or _read_object(directory / "run_spec.json") != entry["source_spec"]):
        raise ValueError(f"evaluation provenance mismatch: {directory}")
    marker_path = directory / "EVALUATION_COMPLETED.json"
    marker = _read_object(marker_path)
    if (marker.get("completed") is not True or marker.get("stage") != "evaluation"
        or marker.get("artifact") != str(metrics_path)):
        raise ValueError(f"invalid completion marker: {marker_path}")
    return True


def evaluate_saved_run(plan: dict, index: int) -> None:
    if not 0 <= index < len(plan["runs"]):
        raise ValueError(f"evaluation index out of range: {index}")
    entry = plan["runs"][index]
    directory = Path(entry["run"]["run_dir"])
    missing, spec = _prerequisites(entry["run"])
    if missing or spec != entry["source_spec"]:
        raise ValueError(f"source artifacts changed or are missing: {directory}; {missing}")
    from dalg.evaluation.toy_manifold_tiling import evaluate_pipeline_run
    print(f"Reevaluating {index}: {directory}", flush=True)
    metrics = evaluate_pipeline_run(entry["run"], evaluation=entry["evaluation"])
    metrics.update(evaluation_config=entry["evaluation"], evaluation_provenance=_provenance(plan, entry))
    effective = {**entry["run"], "evaluation": entry["evaluation"]}
    if not _evaluation_metrics_valid(effective, metrics):
        raise ValueError(f"evaluator produced invalid metrics for {directory}")
    json.dumps(metrics, allow_nan=False)
    _write_json_atomic(directory / "metrics.json", metrics)
    _mark_stage(directory, "evaluation", directory / "metrics.json")
    print(f"Replaced {directory / 'metrics.json'}", flush=True)


def collect_evaluations(plan: dict) -> None:
    from dalg.analysis.aggregate_metrics import aggregate_metrics
    if not plan["runs"]:
        raise ValueError("no eligible saved runs to evaluate")
    for entry in plan["runs"]:
        if not _result_valid(plan, entry):
            raise ValueError(f"missing evaluation: {entry['run']['run_dir']}")
    # Aggregate whole experiment folders so unselected runs stay in the summaries.
    summaries = [(Path(root), *aggregate_metrics(root)) for root in plan["summary_dirs"]]
    for root, general, manifolds in summaries:
        for frame, name in ((general, "general_metrics.csv"), (manifolds, "manifold_metrics.csv")):
            temporary = root / f".{name}.{plan['plan_hash'][:12]}.tmp"
            frame.to_csv(temporary, index=False)
            temporary.replace(root / name)
        print(f"Refreshed {len(general)} runs: {root / 'general_metrics.csv'}, {root / 'manifold_metrics.csv'}")
    _mark_stage(Path(plan["job_dir"]), "evaluation")


def _slurm_jobs(plan: dict) -> list[list[str]]:
    root = Path(plan["job_dir"])
    plan_path = root / "evaluation_plan.json"
    worker = REPO_ROOT / "scripts/slurm/sbatch_evaluation_pipeline.sh"
    jobs = []
    for group in group_by_resources(plan["runs"]):
        command = sbatch_command(plan_path, [{**entry["run"], "resources": entry["resources"]} for entry in group],
                                 worker_path=worker, log_dir=root / "logs", create_log_dir=False)
        indices = ",".join(str(entry["index"]) for entry in group)
        command = [f"--array={indices}%{group[0]['resources']['max_parallel']}" if arg.startswith("--array=") else arg for arg in command]
        command = ["--job-name=dalg-evaluation" if arg.startswith("--job-name=") else arg for arg in command]
        jobs.append(command + ["run"])
    resources = {**plan["runs"][0]["resources"], "gpus": 0, "nodes": 1, "ntasks_per_node": 1}
    collector = sbatch_command(plan_path, [{**plan["runs"][0]["run"], "resources": resources}],
                               worker_path=worker, log_dir=root / "logs", create_log_dir=False)
    collector = [arg for arg in collector if not arg.startswith("--array=")]
    collector = ["--job-name=dalg-evaluation-collect" if arg.startswith("--job-name=") else arg for arg in collector]
    jobs.append(collector + ["collect"])
    return jobs


def submit_evaluations(plan: dict, *, dry_run: bool = False) -> None:
    jobs = _slurm_jobs(plan)
    ids = []
    if not dry_run:
        (Path(plan["job_dir"]) / "logs").mkdir(exist_ok=True)
    for index, command in enumerate(jobs):
        if index == len(jobs) - 1:
            dependency = ":".join(ids) if ids else "<evaluation-array-job-ids>"
            command = [command[0], f"--dependency=afterok:{dependency}", *command[1:]]
        print(f"$ {shlex.join(command)}")
        if not dry_run:
            result = subprocess.run(command, cwd=REPO_ROOT, check=True, text=True, capture_output=True)
            job_id = result.stdout.strip().split(";")[0]
            if not job_id.isdigit():
                raise RuntimeError(f"unexpected sbatch job ID: {result.stdout!r}")
            ids.append(job_id)
            print(f"Submitted {'collection' if index == len(jobs) - 1 else 'evaluation array'} job {job_id}")


def command_evaluate(args) -> None:
    overrides = {key: getattr(args, key) for key in (
        "device", "batch_size", "rank_threshold", "max_mean_to_manifold_distance"
    ) if hasattr(args, key)}
    resources = {}
    if args.resources:
        resources = yaml.safe_load(Path(args.resources).read_text())
        if not isinstance(resources, dict):
            raise ValueError("resource YAML must contain a mapping of resource keys")
        _normalise_resources(resources)
    plan = plan_evaluation(args.manifest, indices=args.indices,
                           overrides=overrides, resources=resources)
    print(f"Eligible: {len(plan['runs'])}; skipped: {len(plan['skipped'])}")
    print(f"Evaluation plan: {Path(plan['job_dir']) / 'evaluation_plan.json'}")
    for root in plan["summary_dirs"]:
        print(f"Refresh summaries: {root}")
    for entry in plan["skipped"]:
        print(f"Skipped {entry['run_dir']}: {', '.join(entry['reasons'])}")
    for entry in plan["runs"]:
        print(f"[{entry['index']}] overwrite {entry['run']['run_dir']}/metrics.json\n  evaluation={entry['evaluation']} resources={entry['resources']}")
    if not args.dry_run:
        save_plan(plan)
    if not plan["runs"]:
        raise ValueError("no eligible saved runs to evaluate")
    if args.submit:
        submit_evaluations(plan, dry_run=args.dry_run)
    elif not args.dry_run:
        for entry in plan["runs"]:
            evaluate_saved_run(plan, entry["index"])
        collect_evaluations(plan)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "collect"))
    parser.add_argument("plan", type=Path)
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    plan = read_evaluation_plan(args.plan)
    if args.mode == "run":
        if args.index is None:
            parser.error("run requires --index")
        evaluate_saved_run(plan, args.index)
    else:
        collect_evaluations(plan)


if __name__ == "__main__":
    main()
