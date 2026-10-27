"""Estimate 2NN and multiscale GRIDE IDs separately in saved train/test clusters.

Run every command, including planning and aggregation, in a Slurm allocation
on Orfeo. The input datasets, fitted models, and saved assignments are read-only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
from itertools import chain
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader

from dalg.analysis.cluster_assignments import compute_assignments
from dalg.data.shard_activations import ActivationBatchDataset, load_meta_index
from dalg.evaluation.toy_manifold_coverage import validate_toy_test_split


REPO_ROOT = Path(__file__).resolve().parents[2]
HOME_CACHE = "/u/dssc/zenocosini/decomposing-activations-local-geometry/dalg-cache/"
SETTINGS = {
    "min_population": 500,
    "gride_range_max": 64,
    "two_nn": {"algorithm": "base", "mu_fraction": 0.9, "data_fraction": 1},
    "splits": ["train", "test"],
    "subsampling": False,
    "assignment_rule": "native_model",
    "coordinate_space": "original_ambient",
}


def settings_for_population(min_population):
    """Keep the estimator contract fixed while varying cluster eligibility."""
    if type(min_population) is not int or min_population < 3:
        raise ValueError("minimum cluster population must be an integer >= 3")
    return {**SETTINGS, "min_population": min_population}


def select_runs(plan, *, K, thresholds, noise_levels, models):
    """Keep model/configuration identities distinct, including both EM variants."""
    selected = []
    for run in plan["runs"]:
        noise = ("noiseless" if run["condition"] == "noiseless"
                 else int(run["condition"].removeprefix("noise_ratio_")))
        if run["K"] != K or noise not in noise_levels:
            continue
        config = read_json(Path(run["run_dir"]) / "config.json")
        model = run["model_kind"]
        if model == "hddc" and config["fit_method"] == "em":
            model = "em" if config["shared_b"] else "em_bk"
        if model not in models:
            continue
        threshold = config.get("surgery_threshold")
        if model != "mfa" and threshold != thresholds[model]:
            continue
        selected.append({"run": run, "model": model, "noise_ratio": noise,
                         "surgery_threshold": threshold})
    keys = [(item["noise_ratio"], item["model"]) for item in selected]
    if len(keys) != len(set(keys)):
        raise ValueError("Select exactly one fitted configuration per model/noise cell")
    if not selected:
        raise ValueError("No saved runs match the requested selection")
    return sorted(selected, key=lambda item: (
        noise_levels.index(item["noise_ratio"]), models.index(item["model"]),
    ))


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def source_path(value, cache_root):
    """Resolve the recorded home alias through the explicitly selected cache."""
    value = str(value)
    if value.startswith(HOME_CACHE):
        return (Path(cache_root) / value[len(HOME_CACHE):]).resolve()
    if value.startswith("dalg-cache/"):
        return (Path(cache_root) / value[len("dalg-cache/"):]).resolve()
    path = Path(value).expanduser()
    return (path if path.is_absolute() else REPO_ROOT / path).resolve()


def file_stat(path):
    stat = Path(path).stat()
    return {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def provenance():
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()

    return {
        "repository": str(REPO_ROOT), "commit": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"), "status": git("status", "--porcelain"),
        "script_sha256": sha256(__file__), "python": sys.version,
        "packages": {name: importlib.metadata.version(name) for name in
                     ("torch", "numpy", "scipy", "scikit-learn", "dadapy")},
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }


def describe_run(run_dir, cache_root):
    run_dir = Path(run_dir).resolve()
    config = read_json(run_dir / "config.json")
    kind = config.get("model_kind") or {
        "MFA": "mfa", "MFA_HDDC": "hddc",
    }.get(config.get("model", "MFA"))
    if config.get("method") == "kmeans":
        kind = "kmeans"
    if kind not in {"kmeans", "mfa", "hddc"}:
        raise ValueError(f"unsupported model kind {kind!r}: {run_dir}")
    if config.get("subset_spec") is not None or "#" in config["shard_dir"]:
        raise ValueError(f"this experiment requires unsliced toy datasets: {run_dir}")
    if config.get("drop_prefix", 0) != 0:
        raise ValueError("toy rows must have drop_prefix=0")
    root = source_path(config["shard_dir"], cache_root)
    stem = "kmeans_model" if kind == "kmeans" else "mfa_model"
    files = [run_dir / name for name in (
        "config.json", "val_indices.json", f"{stem}.pt", f"{stem}_assignments.pt",
    )]
    files += [root / "config.json", root / "test/config.json"]
    missing = [str(path) for path in files if not path.is_file()]
    return {
        "run_dir": str(run_dir), "shard_dir": str(root), "model_kind": kind,
        "layer": int(config["layer"]), "K": int(config["K"]),
        "model_path": str(run_dir / f"{stem}.pt"),
        "assignments_path": str(run_dir / f"{stem}_assignments.pt"),
        "ready": not missing, "missing": missing,
        "input_stats": {str(path): file_stat(path) for path in files if path.is_file()},
    }


def make_plan(results_root, cache_root, output, *, min_population=500,
              K=None, surgery_threshold=None, models=None):
    """List every run at the expected sweep depth; never silently drop a run."""
    results_root, cache_root = Path(results_root).resolve(), Path(cache_root).resolve()
    runs = []
    for directory in sorted(results_root.glob("*/models/*/*")):
        if not directory.is_dir():
            continue
        relative = directory.relative_to(results_root)
        record = describe_run(directory, cache_root)
        record.update(index=len(runs), relative_run=str(relative),
                      condition=relative.parts[0], family=relative.parts[2])
        runs.append(record)
    if not runs:
        raise ValueError(f"no model runs found in {results_root}")
    settings = settings_for_population(min_population)
    plan = {"schema_version": 1, "results_root": str(results_root),
            "cache_root": str(cache_root), "settings": settings, "runs": runs,
            "provenance": provenance()}
    if any(value is not None for value in (K, surgery_threshold, models)):
        if any(value is None for value in (K, surgery_threshold, models)):
            raise ValueError("supply K, surgery threshold, and models together")
        noise_levels = ["noiseless", 1000, 100, 10]
        selected = select_runs(plan, K=K, thresholds={model: surgery_threshold for model in models},
                               noise_levels=noise_levels, models=models)
        if {(item["noise_ratio"], item["model"]) for item in selected} != {
            (noise, model) for noise in noise_levels for model in models
        }:
            raise ValueError("missing a requested model/noise configuration")
        runs = [{**item["run"], "index": index} for index, item in enumerate(selected)]
        plan.update(runs=runs, selection={"K": K, "surgery_threshold": surgery_threshold,
                                          "models": models, "noise_levels": noise_levels})
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    write_json(output, plan)
    print(json.dumps({"manifest": str(output), "runs": len(runs),
                      "ready": sum(run["ready"] for run in runs),
                      "not_ready": [run["relative_run"] for run in runs if not run["ready"]]},
                     indent=2), flush=True)
    return plan


def load_points(root, layer, batch_size):
    dataset = ActivationBatchDataset(
        root, layer=layer, batch_size=batch_size, drop_prefix=0, dtype=torch.float32,
        shuffle_shards=False, shuffle_within_shard=False, return_metadata=True,
    )
    chunks, row_chunks = [], []
    for points, rows, positions in DataLoader(dataset, batch_size=None, num_workers=0):
        if bool((positions != 0).any()):
            raise ValueError("expected exactly one activation per toy row")
        chunks.append(points)
        row_chunks.append(rows)
    return torch.cat(chunks), torch.cat(row_chunks).long()


def training_mask(global_rows, split):
    validation = torch.tensor(split["val_global_rows"], dtype=torch.long)
    if global_rows.unique().numel() != global_rows.numel():
        raise ValueError("duplicate global row IDs in the development stream")
    if (validation.unique().numel() != validation.numel()
            or not bool(torch.isin(validation, global_rows).all())):
        raise ValueError("invalid saved validation row IDs")
    mask = ~torch.isin(global_rows, validation)
    if (int(mask.sum()) != split["train_rows"]
            or int((~mask).sum()) != split["val_rows"]):
        raise ValueError("saved train/validation counts do not match the stream")
    return mask


def load_training(record, cache_root, batch_size):
    root, layer, K = Path(record["shard_dir"]), record["layer"], record["K"]
    config = read_json(root / "config.json")
    if (config.get("source_kind") != "toy_manifolds" or config["window"] != 1
            or config.get("drop_prefix", 0) != 0):
        raise ValueError("expected an activation-compatible toy-manifold dataset")
    points, rows = load_points(root, layer, batch_size)
    meta = load_meta_index(root, layer=layer)
    expected_rows = torch.tensor([row["global_row"] for row in meta])
    if not torch.equal(rows, expected_rows) or len(rows) != config["num_rows"]:
        raise ValueError("development stream is not aligned with its metadata")
    bundle = torch.load(record["assignments_path"], map_location="cpu", weights_only=True)
    labels, sizes = bundle["assignments"].reshape(-1), bundle["cluster_sizes"].reshape(-1)
    if (labels.dtype != torch.int64 or len(labels) != len(points) or len(sizes) != K
            or bundle.get("K", K) != K or bundle.get("subset_spec") is not None
            or bool((labels < 0).any()) or bool((labels >= K).any())
            or not torch.equal(torch.bincount(labels, minlength=K), sizes)):
        raise ValueError("saved assignments do not match the development stream/K")
    if "model_path" in bundle and source_path(bundle["model_path"], cache_root) != Path(record["model_path"]):
        raise ValueError("assignment model_path differs from the selected model")
    if "source" in bundle:
        source = bundle["source"]
        if (source_path(source["shard_dir"], cache_root) != root or source["layer"] != layer
                or source["drop_prefix"] != 0 or source["num_items"] != len(points)):
            raise ValueError("saved assignment source differs from the selected dataset")
    split = read_json(Path(record["run_dir"]) / "val_indices.json")
    mask = training_mask(rows, split)
    return points[mask], labels[mask], split


def estimate_cluster(points, *, min_population=500, range_max=64):
    """Call both DADApy estimators with shared Euclidean neighbour distances."""
    result = {
        "n_points": len(points), "two_nn_status": "below_min_population",
        "two_nn_id": None, "two_nn_error": None, "two_nn_scale": None,
        "gride_status": "below_min_population", "gride_ids": [],
        "gride_errors": [], "gride_scales": [], "gride_ranks": [], "warnings": [],
    }
    if len(points) < min_population:
        return result
    if not bool(torch.isfinite(points).all()):
        result.update(two_nn_status="non_finite_input", gride_status="non_finite_input")
        return result
    from dadapy import Data

    maximum = min(range_max, len(points) - 1)
    # Promotion preserves the stored coordinates and avoids float32 distance cancellation.
    data = Data(coordinates=points.detach().cpu().double().numpy(), maxk=maximum, n_jobs=1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        data.compute_distances()
        if np.any(data.distances[:, 1] <= 0):
            result.update(two_nn_status="zero_neighbour_distance", gride_status="zero_neighbour_distance")
        else:
            two_nn = np.asarray(data.compute_id_2NN(**SETTINGS["two_nn"]), dtype=float)
            if np.all(np.isfinite(two_nn)) and two_nn[0] > 0:
                result.update(two_nn_status="ok", two_nn_id=float(two_nn[0]),
                              two_nn_error=float(two_nn[1]), two_nn_scale=float(two_nn[2]))
            else:
                result["two_nn_status"] = "non_finite_or_nonpositive_estimate"
            ids, errors, scales = data.return_id_scaling_gride(range_max=maximum)
            arrays = [np.asarray(values, dtype=float).reshape(-1) for values in (ids, errors, scales)]
            if len({len(values) for values in arrays}) != 1 or not len(arrays[0]):
                raise ValueError("DADApy returned inconsistent GRIDE arrays")
            result["gride_status"] = (
                "ok" if all(np.isfinite(values).all() for values in arrays)
                and np.all(arrays[0] > 0) else "non_finite_or_nonpositive_estimate"
            )
            for name, values in zip(("gride_ids", "gride_errors", "gride_scales"), arrays):
                result[name] = [float(value) if np.isfinite(value) else None for value in values]
            result["gride_ranks"] = [2 ** (index + 1) for index in range(len(arrays[0]))]
        result["warnings"] = sorted({str(item.message) for item in caught})
    return result


def estimate_split(points, labels, K, split, *, min_population=500, range_max=64):
    if len(points) != len(labels) or bool((labels < 0).any()) or bool((labels >= K).any()):
        raise ValueError("split assignments are not aligned with points")
    counts = torch.bincount(labels, minlength=K)
    order = torch.argsort(labels, stable=True)
    offset, results = 0, []
    for cluster, count in enumerate(counts.tolist()):
        indices = order[offset:offset + count]
        offset += count
        result = estimate_cluster(points[indices], min_population=min_population, range_max=range_max)
        result.update(split=split, cluster_id=cluster)
        results.append(result)
        if count >= min_population:
            print(f"{split} cluster={cluster} n={count} 2NN={result['two_nn_status']} "
                  f"GRIDE={result['gride_status']}", flush=True)
    return results


def write_cluster_csv(path, rows):
    rows = iter(rows)
    first = next(rows)
    with Path(path).open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(first))
        writer.writeheader()
        for row in chain((first,), rows):
            writer.writerow({key: json.dumps(value) if isinstance(value, list) else value
                             for key, value in row.items()})


def run_one(record, cache_root, output, *, device="cpu", batch_size=1024, min_population=500):
    settings = settings_for_population(min_population)
    if not record["ready"]:
        raise FileNotFoundError(f"incomplete input run: {record['missing']}")
    for path, expected in record["input_stats"].items():
        if file_stat(path) != expected:
            raise ValueError(f"input changed since planning: {path}")
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    runtime = provenance()
    fingerprints = {path: sha256(path) for path in record["input_stats"]}
    train_points, train_labels, split = load_training(record, cache_root, batch_size)
    source = validate_toy_test_split(record["shard_dir"], layer=record["layer"])
    test_points, test_rows = load_points(source["shard_dir"], record["layer"], batch_size)
    if len(test_points) != source["num_rows"] or not torch.equal(test_rows, torch.arange(len(test_rows))):
        raise ValueError("test stream is not aligned with the independent test metadata")
    test_sizes, test_labels, max_resp, peakedness = compute_assignments(
        Path(record["model_path"]), DataLoader(test_points, batch_size=batch_size, shuffle=False),
        device=device, model_type=record["model_kind"],
    )
    if len(test_sizes) != record["K"] or len(test_labels) != len(test_points):
        raise ValueError("test assignments do not match the model/test population")
    torch.save({
        "K": record["K"], "assignments": test_labels, "cluster_sizes": test_sizes,
        "max_responsibilities": max_resp, "peakedness": peakedness,
        "model_type": record["model_kind"], "model_path": record["model_path"],
        "subset_spec": None, "split": "test", "source": {
            "shard_dir": source["shard_dir"], "layer": record["layer"],
            "drop_prefix": 0, "num_items": len(test_points),
        },
    }, output / "test_assignments.pt")
    clusters = []
    for name, points, labels in (("train", train_points, train_labels), ("test", test_points, test_labels)):
        clusters.extend(estimate_split(
            points, labels, record["K"], name,
            min_population=settings["min_population"], range_max=settings["gride_range_max"],
        ))
    report = {"schema_version": 1, "run": record, "settings": settings,
              "provenance": runtime, "input_sha256": fingerprints, "test_source": source,
              "populations": {"train": len(train_points), "validation_excluded": split["val_rows"],
                              "test": len(test_points)}, "clusters": clusters,
              "elapsed_seconds": time.monotonic() - started}
    for path, expected in record["input_stats"].items():
        if file_stat(path) != expected:
            raise ValueError(f"input changed during evaluation: {path}")
    write_json(output / "results.json", report)
    write_cluster_csv(output / "clusters.csv", clusters)
    write_json(output / "COMPLETED.json", {"results_sha256": sha256(output / "results.json")})
    print(f"Completed {output}", flush=True)
    return report


def aggregate(plan, output_root):
    """Combine completed results without replacing source metrics or selecting scales."""
    settings = plan["settings"]
    if settings != settings_for_population(settings["min_population"]):
        raise ValueError("unsupported manifest estimator settings")
    summary = []

    def combined_rows():
        for record in plan["runs"]:
            directory = Path(output_root) / "runs" / record["relative_run"]
            complete = read_json(directory / "COMPLETED.json")
            if sha256(directory / "results.json") != complete["results_sha256"]:
                raise ValueError(f"result fingerprint mismatch: {directory}")
            report = read_json(directory / "results.json")
            if report["run"] != record or report["settings"] != plan["settings"]:
                raise ValueError(f"result differs from the selected plan: {directory}")
            identity = {key: record[key] for key in ("condition", "family", "relative_run", "model_kind", "K")}
            for row in report["clusters"]:
                yield {**identity, **row}
            for split in settings["splits"]:
                selected = [row for row in report["clusters"] if row["split"] == split]
                summary.append({**identity, "split": split, "clusters": len(selected),
                                "points": sum(row["n_points"] for row in selected),
                                "eligible_clusters": sum(row["n_points"] >= settings["min_population"] for row in selected),
                                "two_nn_valid": sum(row["two_nn_status"] == "ok" for row in selected),
                                "gride_valid": sum(row["gride_status"] == "ok" for row in selected)})

    destination = Path(output_root) / "summary"
    destination.mkdir(exist_ok=False)
    write_cluster_csv(destination / "clusters.csv", combined_rows())
    write_cluster_csv(destination / "runs.csv", summary)
    print(f"Aggregated {len(plan['runs'])} runs into {destination}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan_parser = commands.add_parser("plan")
    plan_parser.add_argument("--results-root", type=Path, required=True)
    plan_parser.add_argument("--cache-root", type=Path, required=True)
    plan_parser.add_argument("--output", type=Path, required=True)
    plan_parser.add_argument("--min-population", type=int, default=500)
    plan_parser.add_argument("--K", type=int)
    plan_parser.add_argument("--surgery-threshold", type=float)
    plan_parser.add_argument("--models", nargs="+", choices=("kmeans", "mfa", "hddc", "em", "em_bk"))
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--manifest", type=Path, required=True)
    run_parser.add_argument("--index", type=int, required=True)
    run_parser.add_argument("--output-root", type=Path, required=True)
    run_parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    run_parser.add_argument("--batch-size", type=int, default=1024)
    aggregate_parser = commands.add_parser("aggregate")
    aggregate_parser.add_argument("--manifest", type=Path, required=True)
    aggregate_parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "plan":
        make_plan(args.results_root, args.cache_root, args.output, min_population=args.min_population,
                  K=args.K, surgery_threshold=args.surgery_threshold, models=args.models)
        return
    plan = read_json(args.manifest)
    if plan["settings"] != settings_for_population(plan["settings"]["min_population"]):
        raise ValueError("manifest estimator settings differ from this experiment")
    if args.command == "aggregate":
        aggregate(plan, args.output_root)
    else:
        if not 0 <= args.index < len(plan["runs"]) or args.batch_size <= 0:
            raise ValueError("invalid run index or batch size")
        record = plan["runs"][args.index]
        run_one(record, plan["cache_root"], args.output_root / "runs" / record["relative_run"],
                device=args.device, batch_size=args.batch_size,
                min_population=plan["settings"]["min_population"])


if __name__ == "__main__":
    main()
