"""Run or audit full-data toy-noise KMeans fits and geometry evaluations.

From the repository: PYTHONPATH=src:. .venv/bin/python <this file>
{preflight,run,check-training,check-evaluation}. Audits update the reference
table and exit nonzero when incomplete; they never modify input artifacts.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys

import torch

from dalg.init.activation_selection import resolve_initialization_rows
from dalg.init.centroid_artifact import load_centroid_artifact, validate_centroid_artifact
from dalg.evaluation.toy_manifold_metrics import _alignment_summary, _rank_summary
from scripts.temporary.evaluate_toy_kmeans_geometry import _cattell_rank_sweep


REPO = Path(__file__).resolve().parents[2]
BASE = REPO / "dalg-cache/toy_manifolds_10types_1each_D128_30Keach_noise_sweep_seed0"
STATUS = REPO / "outputs/experiments/toy_noise_kmeans_k_sweep/status.md"
NOISES = (10, 100, 1000)
KS = (250, 500, 2000)
N, D, Q, MIN_POPULATION = 300000, 128, 32, 33
THRESHOLDS = (0.00025, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.5, 1.0)


@dataclass(frozen=True)
class Run:
    root: Path
    noise: int
    k: int

    @property
    def shards(self) -> Path:
        return self.root / f"noise_ratio_{self.noise}/dataset"

    @property
    def directory(self) -> Path:
        return self.root / f"noise_ratio_{self.noise}/centroids/kmeans_k{self.k}_full"

    @property
    def evaluation(self) -> Path:
        return self.directory / "initialization_evaluation"

    @property
    def assignments(self) -> Path:
        return self.evaluation / "nearest_centroid_assignments.pt"


def runs(root: Path, noises=NOISES, ks=KS) -> list[Run]:
    return [Run(root, noise, k) for noise in noises for k in ks]


def require(condition, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def load_bundle(path: Path) -> dict:
    return torch.load(path, map_location="cpu", mmap=True, weights_only=True)


def fields(actual: dict, expected: dict, label: str) -> None:
    for key, value in expected.items():
        require(key in actual and actual[key] == value,
                f"{label}.{key}: expected {value!r}, got {actual.get(key)!r}")


def same_path(actual: str, expected: Path) -> None:
    require(Path(actual).resolve() == expected.resolve(),
            f"wrong provenance: {actual}; expected {expected}")


def tensor(value, shape: tuple, label: str) -> torch.Tensor:
    require(isinstance(value, torch.Tensor) and tuple(value.shape) == shape,
            f"{label}: expected tensor shape {shape}")
    require(torch.isfinite(value).all(), f"{label}: non-finite values")
    return value


def mask(value, shape: tuple, label: str) -> torch.Tensor:
    value = tensor(value, shape, label)
    require(value.dtype == torch.bool, f"{label}: expected boolean mask")
    return value


def preflight(root: Path, noises=NOISES, ks=KS) -> dict[int, dict]:
    selections = {}
    for noise in noises:
        shards = root / f"noise_ratio_{noise}/dataset"
        _, config, _, selection = resolve_initialization_rows(
            shards, layer=0, val_frac=0.0, split_seed=42, drop_prefix=0,
        )
        fields(config, {"source_kind": "toy_manifolds", "num_rows": N,
                        "d_model": D, "window": 1, "drop_prefix": 0,
                        "layers": [0]}, str(shards))
        fields(selection, {"subset_spec": None, "selected_rows": N,
                           "train_rows": N, "train_activations": N}, "selection")
        metadata = load_bundle(shards / config["manifold_metadata"])
        ids = tensor(metadata["row_manifold_ids"], (N,), "manifold row IDs")
        require(len(metadata["manifolds"]) == 10 and len(ids.unique()) == 10,
                "expected ten planted manifolds")
        require(max(m["intrinsic_dim"] for m in metadata["manifolds"]) <= Q,
                "PCA capacity is below the planted intrinsic dimension")
        selections[noise] = selection
    for run in runs(root, noises, ks):
        require(not run.directory.exists() or run.directory.is_dir(),
                f"destination is not a directory: {run.directory}")
        print(f"noise={run.noise} K={run.k}: {run.directory}", flush=True)
    return selections


def validate_training(run: Run, selection: dict) -> dict:
    config = read_json(run.directory / "config.json")
    fields(config, {
        "method": "kmeans", "metric": "euclidean", "K": run.k,
        "shape": [N, D], "uses_all_rows": True, "uses_all_training_rows": True,
        "rows_used": N, "source_rows": N, "sample_fraction_requested": 1.0,
        "sample_fraction_actual": 1.0, "sample_seed": 0, "seed": 0,
        "restarts": 10, "max_iter": 1000, "tol": 1e-6, "layer": 0,
        "centroid_artifact_format": "dalg_centroids_v1",
        "principal_components": None, "selection": selection,
    }, "centroid config")
    same_path(config["source_shard_dir"], run.shards)
    centroids, pcs = load_centroid_artifact(run.directory / "centroids.pt", mmap=True)
    validate_centroid_artifact(centroids, pcs, expected_k=run.k, expected_d=D)
    tensor(centroids, (run.k, D), "centroids")
    require(pcs is None, "expected centroid means only")
    sizes = torch.as_tensor(config["cluster_sizes"])
    tensor(sizes, (run.k,), "fit cluster sizes")
    require(sizes.dtype == torch.int64 and (sizes > 0).all() and int(sizes.sum()) == N,
            "fit cluster sizes must be positive integers totaling the full dataset")
    require(math.isfinite(config["inertia"]) and config["inertia"] > 0,
            "invalid fit inertia")
    return config


def validate_assignments(run: Run, config: dict) -> dict:
    bundle = load_bundle(run.assignments)
    fields(bundle, {"K": run.k, "subset_spec": None}, "assignments")
    same_path(bundle["centroids_path"], run.directory / "centroids.pt")
    same_path(bundle["source"]["shard_dir"], run.shards)
    fields(bundle["source"], {"layer": 0, "drop_prefix": 0, "num_items": N}, "source")
    labels = tensor(bundle["assignments"], (N,), "assignments")
    sizes = tensor(bundle["cluster_sizes"], (run.k,), "assignment sizes")
    require(labels.dtype in (torch.int32, torch.int64) and sizes.dtype == torch.int64,
            "assignments and sizes must be integers")
    require((labels >= 0).all() and (labels < run.k).all(), "assignment IDs out of range")
    require(torch.equal(torch.bincount(labels.long(), minlength=run.k), sizes),
            "assignment counts disagree")
    require(torch.equal(sizes, torch.tensor(config["cluster_sizes"])),
            "assignment counts differ from the fit")
    distances = tensor(bundle["min_distances"], (N,), "min_distances")
    require((distances >= 0).all(), "negative distances")
    inertia = float(distances.double().square().sum())
    require(math.isclose(inertia, config["inertia"], rel_tol=1e-5),
            "assignment inertia differs from fit inertia")
    return bundle


def validate_evaluation(run: Run, config: dict) -> None:
    bundle = validate_assignments(run, config)
    metrics = read_json(run.evaluation / "metrics.json")
    details = load_bundle(run.evaluation / "component_metrics.pt")
    validate_evaluation_reports(run, bundle, metrics, details)


def validate_evaluation_reports(run: Run, bundle: dict, metrics: dict, details: dict) -> None:
    k, t = run.k, len(THRESHOLDS)
    fields(metrics, {"schema_version": 2, "evaluation": "toy_kmeans_initialization_cattell_sweep",
                     "partition_kind": "nearest_euclidean_centroid", "K": k,
                     "q_max": Q, "pca_capacity": Q}, "metrics")
    fields(metrics["dataset"], {"selected_rows": N, "layer": 0, "subset_spec": None,
                                "in_sample": True}, "dataset")
    same_path(metrics["dataset"]["shard_dir"], run.shards)
    for key, path in {"centroids_path": run.directory / "centroids.pt",
                      "assignments_path": run.assignments,
                      "component_details_path": run.evaluation / "component_metrics.pt"}.items():
        same_path(metrics["artifacts"][key], path)
    fields(details, {"schema_version": 1, "K": k, "q_max": Q,
                     "min_population": MIN_POPULATION,
                     "evaluation": metrics["evaluation"]}, "sidecar")
    sizes = tensor(details["cluster_sizes"], (k,), "sidecar sizes")
    require(torch.equal(sizes, bundle["cluster_sizes"]), "sidecar sizes differ")
    eligible = mask(details["eligible"], (k,), "eligible")
    require(torch.equal(eligible, sizes >= MIN_POPULATION) and eligible.any(),
            "incorrect geometry eligibility")
    fields(metrics["eligibility"], {
        "min_population": MIN_POPULATION, "eligible_components": int(eligible.sum()),
        "excluded_components": int((~eligible).sum()),
        "eligible_points": int(sizes[eligible].sum()),
        "excluded_points": int(sizes[~eligible].sum()),
    }, "eligibility")
    fields(metrics["cattell"], {"thresholds": list(THRESHOLDS), "q_max": Q,
                               "comparison": "strict_greater_than",
                               "shared_noise_active_set_applied": False}, "Cattell")
    require(torch.equal(details["cattell_thresholds"], torch.tensor(THRESHOLDS, dtype=torch.float64)),
            "sidecar threshold list differs")
    spectrum = tensor(details["leading_eigenvalues"], (k, Q + 1), "spectrum")
    require((spectrum >= -1e-12).all() and
            (spectrum[:, :-1] >= spectrum[:, 1:] - 1e-12).all(), "invalid covariance spectrum")
    gaps, ranks = _cattell_rank_sweep(spectrum, eligible, q_max=Q, thresholds=THRESHOLDS)
    require(torch.allclose(tensor(details["normalized_cattell_gaps"], (k, Q), "gaps"), gaps,
                           atol=1e-12, rtol=1e-10), "Cattell gaps differ from spectrum")
    require(torch.equal(tensor(details["cattell_ranks"], (t, k), "ranks"), ranks),
            "Cattell ranks disagree with spectrum, thresholds, or eligibility")
    pcs = tensor(details["principal_components"], (k, D, Q), "PCs")
    require(torch.equal(mask(details["principal_components_defined"], (k,), "PC mask"), eligible),
            "PC validity mask differs from eligibility")
    require(torch.count_nonzero(pcs[~eligible]) == 0, "excluded PCs must be zero placeholders")
    gram = pcs[eligible].double().transpose(1, 2) @ pcs[eligible].double()
    require((gram - torch.eye(Q)).abs().max() <= 1e-4, "PCs are not orthonormal")
    pca = metrics["pca_validation"]
    fields(pca, {"source": "computed_from_assignments", "covariance_center": "empirical_cluster_mean",
                 "centroid_mean_agreement_required": False, "pca_tolerance": 1e-4,
                 "centroid_mean_l2_tolerance": 1e-5}, "PCA")
    for key in ("max_orthonormal_error", "max_relative_subspace_residual", "max_relative_eigenvalue_error"):
        require(math.isfinite(pca[key]) and 0 <= pca[key] <= 1e-4, f"PCA validation failed: {key}")
    inertia = float(bundle["min_distances"].double().square().sum())
    require(math.isclose(metrics["quantization"]["inertia"], inertia, rel_tol=1e-12),
            "reported inertia differs from assignments")
    for name, value in metrics["clustering"].items():
        require(math.isfinite(value) and (-1 if name == "adjusted_rand_index" else 0) <= value <= 1,
                f"invalid clustering score: {name}")

    associated = mask(details["associated"], (k,), "associated")
    population = eligible & associated
    association = metrics["association"]
    fields(association, {"rule": "unique_nearest_exact_projection",
                         "max_mean_to_manifold_distance": None, "outside_cutoff_components": 0,
                         "associated_components": int(associated.sum()),
                         "eligible_associated_components": int(population.sum()),
                         "ambiguous_components": k - int(associated.sum())}, "association")
    indices = tensor(details["associated_manifold_indices"], (k,), "manifold indices")
    targets = tensor(details["target_intrinsic_dims"], (k,), "target ranks")
    require(((indices[associated] >= 0) & (indices[associated] < 10)).all(), "invalid association IDs")
    require(((targets[associated] >= 1) & (targets[associated] <= Q)).all(), "invalid target ranks")
    require(len(metrics["per_manifold"]) == 10 and
            sum(m["components"]["associated"] for m in metrics["per_manifold"]) == int(associated.sum()) and
            sum(m["components"]["eligible_associated"] for m in metrics["per_manifold"]) == int(population.sum()),
            "per-manifold populations disagree")
    sweep = metrics["threshold_sweep"]
    require([m["cattell_threshold"] for m in sweep] == list(THRESHOLDS), "incorrect threshold sweep")
    for prefix, shape in (("alignment", (k,)), ("containment", (t, k))):
        defined = mask(details[f"{prefix}_defined"], shape, prefix)
        require((~defined | population).all(), f"{prefix} defined outside evaluation population")
        overlap = tensor(details[f"{prefix}_overlap"], shape, prefix)
        worst = tensor(details[f"{prefix}_worst_direction_cosine"], shape, prefix)
        for values in (overlap, worst):
            require(((values[defined] >= -1e-6) & (values[defined] <= 1 + 1e-6)).all(),
                    f"{prefix}: defined scores outside [0, 1]")
        for i, report in enumerate([metrics["tangent_alignment"]] if prefix == "alignment"
                                   else [m["tangent_containment"] for m in sweep]):
            a, b, valid = (overlap, worst, defined) if prefix == "alignment" else (overlap[i], worst[i], defined[i])
            fields(report, _alignment_summary(a, b, valid, population), prefix)
            fields(report, {"relative_boundary_eigengap_threshold": 1e-6}, prefix)
    for i, report in enumerate(sweep):
        fields(report["rank"], _rank_summary(ranks[i], targets, population), "rank recovery")


def command(args: list) -> None:
    args = [str(arg) for arg in args]
    print("Running: " + shlex.join(args), flush=True)
    subprocess.run(args, cwd=REPO, check=True)


def fit(run: Run, selection: dict) -> None:
    if run.directory.exists() and any(run.directory.iterdir()):
        validate_training(run, selection)
        print(f"Reusing validated fit: {run.directory}", flush=True)
        return
    command([sys.executable, REPO / "scripts/temporary/build_toy_kmeans_centroids.py",
             "--shard-dir", run.shards, "--layer", 0, "--drop-prefix", 0, "--val-frac", 0,
             "--split-seed", 42, "--K", run.k, "--out-dir", run.directory,
             "--max-iter", 1000, "--restarts", 10, "--tol", "1e-6", "--seed", 0,
             "--sample-fraction", 1.0, "--sample-seed", 0, "--pca-rank", 0,
             "--device", "cuda", "--load-batch-size", 20000, "--block-x", 8192, "--block-c", 8192])
    validate_training(run, selection)


def evaluate(run: Run, selection: dict) -> None:
    config = validate_training(run, selection)
    if any((run.evaluation / name).exists() for name in ("metrics.json", "component_metrics.pt")):
        validate_evaluation(run, config)
        print(f"Reusing validated evaluation: {run.evaluation}", flush=True)
        return
    if run.assignments.exists():
        validate_assignments(run, config)
    else:
        command([REPO / ".venv/bin/dalg-run-metrics", "assignments", "--centroids-path",
                 run.directory / "centroids.pt", "--shard-dir", run.shards,
                 "--layer", 0, "--drop-prefix", 0, "--batch-size", 8192,
                 "--device", "cuda", "--save-path", run.assignments])
        validate_assignments(run, config)
    command([sys.executable, REPO / "scripts/temporary/evaluate_toy_kmeans_geometry.py",
             "--centroids-path", run.directory / "centroids.pt", "--assignments-path", run.assignments,
             "--shard-dir", run.shards, "--layer", 0, "--device", "cuda", "--batch-size", 10000,
             "--compute-pca", "--q-max", Q, "--min-population", MIN_POPULATION,
             "--cattell-thresholds", *THRESHOLDS, "--relative-boundary-eigengap-threshold", "1e-6",
             "--output-path", run.evaluation / "metrics.json"])
    validate_evaluation(run, config)


def audit(rows: list[Run], selections: dict[int, dict]) -> list[dict]:
    results = []
    for run in rows:
        result = {"noise": run.noise, "k": run.k, "training": False, "evaluation": False, "errors": []}
        try:
            config = validate_training(run, selections[run.noise])
            result["training"] = True
        except Exception as error:
            result["errors"].append(f"Training: {type(error).__name__}: {error}")
        if result["training"]:
            try:
                validate_evaluation(run, config)
                result["evaluation"] = True
            except Exception as error:
                result["errors"].append(f"Evaluation: {type(error).__name__}: {error}")
        results.append(result)
    return results


def write_status(path: Path, rows: list[Run], results: list[dict], phase: str,
                 failures: list[str]) -> None:
    state_path = path.with_suffix(".json")
    previous = read_json(state_path) if state_path.exists() else {}
    trained = sum(row["training"] for row in results)
    evaluated = sum(row["evaluation"] for row in results)
    state = {"updated_at": datetime.now(timezone.utc).isoformat(), "phase": phase,
             "slurm_job_id": (os.environ.get("SLURM_JOB_ID") if phase == "before fitting"
                              else previous.get("slurm_job_id")),
             "training_gate_passed": trained == len(rows), "training_completed": trained,
             "evaluation_completed": evaluated, "rows": results, "failures": failures}
    audit_args = shlex.join([
        "--root", str(rows[0].root), "--noises", *map(str, dict.fromkeys(r.noise for r in rows)),
        "--ks", *map(str, dict.fromkeys(r.k for r in rows)), "--status-path", str(path),
    ])
    lines = ["# Toy-noise KMeans K sweep", "", f"Updated: {state['updated_at']}",
             f"Slurm job: {state['slurm_job_id'] or 'not submitted'}; phase: {phase}", "",
             "Full data: 300,000 points, D=128, layer=0, drop_prefix=0, val_frac=0.",
             "Euclidean KMeans: seed=0, restarts=10, max_iter=1000, tol=1e-6; means only.",
             "Evaluation: computed cluster PCA, q_max=32, minimum population=33, no distance cutoff.",
             "Cattell thresholds: " + ", ".join(map(str, THRESHOLDS)) + ".", "",
             f"Training: **{trained}/{len(rows)}**; evaluation: **{evaluated}/{len(rows)}**; "
             f"training gate: **{'PASSED' if state['training_gate_passed'] else 'BLOCKED'}**.", "",
             "| Noise ratio | K | Training completed | Evaluation completed |",
             "|---:|---:|:---:|:---:|"]
    for run, result in zip(rows, results, strict=True):
        directory = os.path.relpath(run.directory, path.parent)
        metrics = os.path.relpath(run.evaluation / "metrics.json", path.parent)
        lines.append(f"| {run.noise} | [{run.k}]({directory}/) | "
                     f"{'☑' if result['training'] else '☐'} | "
                     f"[{'☑' if result['evaluation'] else '☐'}]({metrics}) |")
    lines += ["", "Checkboxes indicate validated artifacts. Missing reports remain unchecked.", "",
              "Audit commands (from the repository root):", "", "```bash",
              "PYTHONPATH=src:. .venv/bin/python scripts/temporary/run_toy_noise_kmeans_k_sweep.py check-training " + audit_args,
              "PYTHONPATH=src:. .venv/bin/python scripts/temporary/run_toy_noise_kmeans_k_sweep.py check-evaluation " + audit_args,
              "```", "", "Validation findings:", ""]
    for result in results:
        lines.extend(f"- noise={result['noise']}, K={result['k']}: {error}" for error in result["errors"])
    lines.extend(f"- {failure}" for failure in failures)
    if not failures and not any(row["errors"] for row in results):
        lines.append("All required artifacts passed validation.")
    path.parent.mkdir(parents=True, exist_ok=True)
    for output, content in ((state_path, json.dumps(state, indent=2) + "\n"),
                            (path, "\n".join(lines) + "\n")):
        temporary = output.with_name(output.name + f".tmp.{os.getpid()}")
        temporary.write_text(content)
        temporary.replace(output)
    print(f"{phase}: training {trained}/{len(rows)}, evaluation {evaluated}/{len(rows)}", flush=True)


def execute(rows: list[Run], selections: dict[int, dict], status: Path) -> int:
    failures = []
    write_status(status, rows, audit(rows, selections), "before fitting", failures)
    for run in rows:
        try:
            fit(run, selections[run.noise])
        except Exception as error:
            failures.append(f"Fit noise={run.noise}, K={run.k}: {type(error).__name__}: {error}")
            print(failures[-1], flush=True)
        write_status(status, rows, audit(rows, selections), "fitting", failures)
    results = audit(rows, selections)
    write_status(status, rows, results, "global training audit", failures)
    if not all(row["training"] for row in results):
        print("Training gate failed; no evaluations will start.", flush=True)
        return 1
    print(f"TRAINING GATE PASSED: all {len(rows)} centroid configurations validated.", flush=True)
    for run in rows:
        try:
            evaluate(run, selections[run.noise])
        except Exception as error:
            failures.append(f"Evaluation noise={run.noise}, K={run.k}: {type(error).__name__}: {error}")
            print(failures[-1], flush=True)
        write_status(status, rows, audit(rows, selections), "evaluating", failures)
    results = audit(rows, selections)
    write_status(status, rows, results, "final audit", failures)
    return 0 if all(row["training"] and row["evaluation"] for row in results) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("preflight", "run", "check-training", "check-evaluation"))
    parser.add_argument("--root", type=Path, default=BASE)
    parser.add_argument("--noises", type=int, nargs="+", default=NOISES)
    parser.add_argument("--ks", type=int, nargs="+", default=KS)
    parser.add_argument("--status-path", type=Path)
    args = parser.parse_args()
    for name, values in (("noises", args.noises), ("ks", args.ks)):
        if any(value <= 0 for value in values) or len(set(values)) != len(values):
            parser.error(f"--{name} requires unique positive integers")
    if args.status_path is None:
        if tuple(args.noises) != NOISES or tuple(args.ks) != KS:
            parser.error("--status-path is required for a custom sweep")
        args.status_path = STATUS
    torch.set_num_threads(4)
    args.status_path.parent.mkdir(parents=True, exist_ok=True)
    with args.status_path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        selections = preflight(args.root, args.noises, args.ks)
        rows = runs(args.root, args.noises, args.ks)
        if args.stage == "run":
            return execute(rows, selections, args.status_path)
        results = audit(rows, selections)
        write_status(args.status_path, rows, results, args.stage, [])
        if args.stage == "preflight":
            return 0
        key = "training" if args.stage == "check-training" else "evaluation"
        return 0 if all(row[key] for row in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
