"""Read-only completion checks for the noise-100/noise-1000 PCA32 experiment.

Run from the repository with ``PYTHONPATH=src python <this script> --stage
centroids`` or ``--stage models``. Prints one compact JSON record per validated
condition/run and fails on missing, incomplete, or inconsistent artifacts.
Model validation reports every incomplete pipeline and checks every completed
pipeline before exiting unsuccessfully if any remain incomplete.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from dalg.init.centroid_artifact import load_centroid_artifact, validate_centroid_artifact
from dalg.models.adaptive_q.mfa_hddc import load_mfa_hddc
from dalg.models.mfa import load_mfa
from dalg.pipeline import default_manifest_path, pipeline_status, read_manifest, resolve_experiment


REPO = Path(__file__).resolve().parents[2]
BASE = REPO / "dalg-cache/toy_manifolds_10types_1each_D128_30Keach_noise_sweep_seed0"
THRESHOLDS = [0.00025, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.5, 1.0]
K, D, Q, N = 1000, 128, 32, 300000


def require(condition, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def load_bundle(path: Path) -> dict:
    return torch.load(path, map_location="cpu", mmap=True, weights_only=True)


def fields(actual: dict, expected: dict, label: str) -> None:
    for key, value in expected.items():
        require(actual.get(key) == value, f"{label}.{key}: expected {value!r}, got {actual.get(key)!r}")


def same_path(actual: str, expected: Path) -> None:
    require(Path(actual).resolve() == expected.resolve(), f"wrong path: {actual}; expected {expected}")


def tensor(value: torch.Tensor, shape: tuple, label: str) -> torch.Tensor:
    require(isinstance(value, torch.Tensor) and tuple(value.shape) == shape, f"{label}: expected shape {shape}")
    require(torch.isfinite(value).all(), f"{label}: non-finite values")
    return value


def score_report(report: dict, population: int) -> None:
    for name in ("subspace_overlap", "worst_direction_cosine"):
        score = report[name]
        valid, undefined = score["valid_components"], score["undefined_components"]
        require(valid >= 0 and undefined >= 0 and valid + undefined == population, f"{name}: invalid population counts")
        if valid:
            require(score["mean"] is not None and math.isfinite(score["mean"]) and -1e-6 <= score["mean"] <= 1 + 1e-6, f"{name}: invalid mean")
        else:
            require(score["mean"] is None, f"{name}: empty population must report null")


def association_count(report: dict) -> int:
    fields(report, {"rule": "unique_nearest_exact_projection", "max_mean_to_manifold_distance": None, "outside_cutoff_components": 0}, "association")
    counts = [report[key] for key in ("associated_components", "outside_cutoff_components", "ambiguous_components")]
    require(all(count >= 0 for count in counts) and sum(counts) == K, "association counts must partition K")
    return report["associated_components"]


def assignments(path: Path) -> dict:
    bundle = load_bundle(path)
    fields(bundle, {"K": K, "subset_spec": None}, str(path))
    labels = tensor(bundle["assignments"], (N,), "assignments")
    sizes = tensor(bundle["cluster_sizes"], (K,), "cluster_sizes")
    require(labels.dtype in (torch.int32, torch.int64), "assignment IDs must be integers")
    require(int(labels.min()) >= 0 and int(labels.max()) < K, "assignment IDs out of range")
    require(int(sizes.sum()) == N and torch.equal(torch.bincount(labels.long(), minlength=K), sizes.long()), "assignment cluster counts disagree")
    return bundle


def validate_centroids(noise: int) -> dict:
    condition = BASE / f"noise_ratio_{noise}"
    directory = condition / "centroids/kmeans_k1000_full_pca32"
    evaluation = directory / "initialization_evaluation"
    config = read_json(directory / "config.json")
    fields(config, {"method": "kmeans", "metric": "euclidean", "K": K, "shape": [N, D], "uses_all_rows": True, "rows_used": N, "source_rows": N, "sample_fraction_requested": 1.0, "sample_fraction_actual": 1.0, "seed": 0, "restarts": 10, "max_iter": 1000, "tol": 1e-6, "layer": 0, "centroid_artifact_format": "dalg_centroids_v1"}, "centroid config")
    same_path(config["source_shard_dir"], condition / "dataset")
    fields(config["principal_components"], {"rank": Q, "shape": [K, D, Q], "uses_all_rows": True, "center": "stored_kmeans_centroid", "covariance_accumulator_dtype": "float64"}, "PCA config")
    centroids, pcs = load_centroid_artifact(directory / "centroids.pt", map_location="cpu", mmap=True)
    validate_centroid_artifact(centroids, pcs, expected_k=K, expected_d=D)
    tensor(centroids, (K, D), "centroids")
    tensor(pcs, (K, D, Q), "principal_components")
    gram_error = float((pcs.double().transpose(1, 2) @ pcs.double() - torch.eye(Q, dtype=torch.float64)).abs().max())
    require(gram_error <= 1e-4, "stored PCs are not orthonormal")

    bundle = assignments(evaluation / "nearest_centroid_assignments.pt")
    sizes = bundle["cluster_sizes"].long()
    require(int(sizes.min()) >= Q + 1, "a cluster has fewer than 33 points")
    require(torch.equal(sizes, torch.tensor(config["cluster_sizes"])), "fit and assignment cluster sizes differ")
    same_path(bundle["centroids_path"], directory / "centroids.pt")
    same_path(bundle["source"]["shard_dir"], condition / "dataset")
    fields(bundle["source"], {"layer": 0, "drop_prefix": 0, "num_items": N}, "assignment source")
    distances = tensor(bundle["min_distances"], (N,), "min_distances")
    require((distances >= 0).all(), "negative distances")
    inertia = float(distances.double().square().sum())
    require(math.isfinite(config["inertia"]) and config["inertia"] > 0, "invalid fit inertia")
    require(abs(inertia - config["inertia"]) / config["inertia"] <= 1e-5, "assignment and fit inertia differ")

    metrics = read_json(evaluation / "metrics.json")
    fields(metrics, {"schema_version": 2, "evaluation": "toy_kmeans_initialization_cattell_sweep", "partition_kind": "nearest_euclidean_centroid", "K": K, "q_max": Q, "pca_capacity": Q}, "centroid metrics")
    fields(metrics["dataset"], {"selected_rows": N, "layer": 0, "subset_spec": None, "in_sample": True}, "dataset")
    same_path(metrics["dataset"]["shard_dir"], condition / "dataset")
    for name, path in {"centroids_path": directory / "centroids.pt", "assignments_path": evaluation / "nearest_centroid_assignments.pt", "component_details_path": evaluation / "component_metrics.pt"}.items():
        same_path(metrics["artifacts"][name], path)
    fields(metrics["eligibility"], {"min_population": 33, "eligible_components": K, "excluded_components": 0, "eligible_points": N, "excluded_points": 0}, "eligibility")
    fields(metrics["cattell"], {"thresholds": THRESHOLDS, "q_max": Q, "comparison": "strict_greater_than", "shared_noise_active_set_applied": False}, "Cattell")
    pca = metrics["pca_validation"]
    fields(pca, {"source": "centroid_artifact", "covariance_center": "saved_centroid", "centroid_mean_agreement_required": True, "pca_tolerance": 1e-4, "centroid_mean_l2_tolerance": 1e-5}, "PCA validation")
    for key, tolerance in {"max_centroid_mean_l2_error": 1e-5, "max_orthonormal_error": 1e-4, "max_relative_subspace_residual": 1e-4, "max_relative_eigenvalue_error": 1e-4}.items():
        require(math.isfinite(pca[key]) and 0 <= pca[key] <= tolerance, f"PCA validation failed: {key}")
    require(math.isclose(metrics["quantization"]["inertia"], inertia, rel_tol=1e-12), "report inertia differs from assignments")

    details = load_bundle(evaluation / "component_metrics.pt")
    fields(details, {"schema_version": 1, "K": K, "q_max": Q, "min_population": 33}, "component details")
    require(torch.equal(details["cluster_sizes"], sizes), "component-detail cluster sizes differ")
    require(tensor(details["eligible"], (K,), "eligible").bool().all(), "not all clusters eligible")
    require(torch.equal(tensor(details["cattell_thresholds"], (9,), "thresholds"), torch.tensor(THRESHOLDS, dtype=torch.float64)), "component thresholds differ")
    spectrum = tensor(details["leading_eigenvalues"], (K, Q + 1), "leading_eigenvalues")
    gaps = tensor(details["normalized_cattell_gaps"], (K, Q), "normalized_cattell_gaps")
    require((spectrum >= -1e-12).all() and (spectrum[:, :-1] >= spectrum[:, 1:] - 1e-12).all(), "invalid ordered covariance spectrum")
    expected_gaps = (spectrum[:, :-1].clamp_min(0) - spectrum[:, 1:].clamp_min(0)) / spectrum[:, :1].clamp_min(torch.finfo(spectrum.dtype).tiny)
    require(torch.allclose(gaps, expected_gaps, atol=1e-12, rtol=1e-10), "saved Cattell gaps differ from spectra")
    ranks = tensor(details["cattell_ranks"], (9, K), "cattell_ranks")
    require(((ranks >= 1) & (ranks <= Q)).all() and (ranks[1:] <= ranks[:-1]).all(), "invalid or nonmonotone Cattell ranks")
    for index, threshold in enumerate(THRESHOLDS):
        expected = torch.where(gaps > threshold, torch.arange(1, Q + 1)[None, :], 0).max(dim=1).values.clamp_min(1)
        require(torch.equal(ranks[index], expected), f"Cattell ranks disagree at threshold {threshold}")

    population = association_count(metrics["association"])
    associated = tensor(details["associated"], (K,), "associated").bool()
    require(int(associated.sum()) == population, "association sidecar count differs")
    score_report(metrics["tangent_alignment"], population)
    require(len(metrics["per_manifold"]) == 10 and sum(item["components"]["associated"] for item in metrics["per_manifold"]) == population, "per-manifold association coverage differs")
    sweep = metrics["threshold_sweep"]
    require([row["cattell_threshold"] for row in sweep] == THRESHOLDS, "reported sweep thresholds differ")
    for prefix, shape in (("alignment", (K,)), ("containment", (9, K))):
        defined = tensor(details[f"{prefix}_defined"], shape, f"{prefix}_defined").bool()
        require((~defined | associated).all(), f"{prefix}: defined for unassociated component")
        for suffix in ("overlap", "worst_direction_cosine"):
            values = tensor(details[f"{prefix}_{suffix}"], shape, f"{prefix}_{suffix}")
            require(((values[defined] >= -1e-6) & (values[defined] <= 1 + 1e-6)).all(), f"{prefix}_{suffix}: out of range")
    require(metrics["tangent_alignment"]["subspace_overlap"]["valid_components"] == int(details["alignment_defined"].sum()), "alignment validity count differs")
    for index, row in enumerate(sweep):
        fields(row["rank"], {"components": population}, "Cattell rank summary")
        score_report(row["tangent_containment"], population)
        require(row["tangent_containment"]["subspace_overlap"]["valid_components"] == int(details["containment_defined"][index].sum()), "containment validity count differs")
    return {"stage": "centroids", "noise_ratio": noise, "status": "validated", "directory": str(directory), "rows": N, "min_cluster_size": int(sizes.min()), "inertia": inertia, "max_centroid_mean_l2_error": pca["max_centroid_mean_l2_error"], "max_pc_orthonormal_error": gram_error, "tangent_alignment": metrics["tangent_alignment"]["subspace_overlap"], "thresholds": THRESHOLDS, "mean_cattell_ranks": ranks.double().mean(dim=1).tolist()}


def validate_model(run: dict, noise: int, kind: str, manifest: Path) -> dict:
    directory = Path(run["run_dir"])
    condition = BASE / f"noise_ratio_{noise}"
    require(directory.resolve().is_relative_to((condition / "models").resolve()), "run directory is outside its condition")
    same_path(run["dataset"]["shard_dir"], condition / "dataset")
    arguments = run["training"]["arguments"]
    same_path(arguments["centroids_path"], condition / "centroids/kmeans_k1000_full_pca32/centroids.pt")
    mode = "single_process" if kind == "hddc" else "vanilla"
    fields(arguments, {"K": K, "rank": Q, "direction_init": "cluster_pca", "training_mode": mode, "epochs": 1000, "lr": 0.001, "early_stop_patience": 10, "val_frac": 0.1, "split_seed": 42, "seed": 42}, "training arguments")
    fields(run["evaluation"], {"enabled": True, "kind": "toy_manifold_tiling", "rank_threshold": 1.0, "max_mean_to_manifold_distance": None}, "evaluation config")
    require(run["assignments"]["enabled"], "assignments disabled")
    require(read_json(directory / "run_spec.json") == run, "saved run spec differs from immutable manifest")
    for stage in ("training", "assignments", "evaluation", "pipeline"):
        fields(read_json(directory / f"{stage.upper()}_COMPLETED.json"), {"stage": stage, "completed": True}, f"{stage} completion marker")
    config = read_json(directory / "config.json")
    fields(config, {"K": K, "rank": Q, "d_model": D, "layer": 0, "drop_prefix": 0, "direction_init": "cluster_pca", "training_mode": mode, "val_frac": 0.1, "split_seed": 42}, "saved model config")
    same_path(config["shard_dir"], condition / "dataset")
    split = read_json(directory / "val_indices.json")
    fields(split, {"seed": 42, "val_frac": 0.1, "train_rows": 270000, "val_rows": 30000, "per_row_tokens": 1}, "split")
    require(len(split["val_global_rows"]) == len(set(split["val_global_rows"])) == 30000, "invalid validation row IDs")
    require(min(split["val_global_rows"]) >= 0 and max(split["val_global_rows"]) < N, "validation row IDs outside dataset")
    loader = load_mfa_hddc if kind == "hddc" else load_mfa
    checkpoint = load_bundle(directory / "mfa_model.pt")
    fields(checkpoint["meta"], {"K": K, "D": D, "q": Q}, "checkpoint metadata")
    model = loader(directory / "mfa_model.pt", map_location="cpu")
    require((model.K, model.D, model.q) == (K, D, Q), "checkpoint dimensions differ")
    for name, value in model.state_dict().items():
        require(torch.isfinite(value).all(), f"checkpoint has non-finite state: {name}")
    require(torch.isfinite(model.W).all() and torch.isfinite(model._psi()).all() and (model._psi() > 0).all(), "invalid derived covariance parameters")
    if kind == "hddc":
        expected_hddc = {"shared_b": True, "fit_method": "adam", "surgery_every_epochs": 0.5, "surgery_min_count": 64, "surgery_threshold": arguments["surgery_threshold"]}
        fields(arguments, expected_hddc, "HDDC arguments")
        fields(config, expected_hddc, "saved HDDC config")
        require(model.shared_b, "checkpoint is not shared-b HDDC")
        mask = tensor(model.rank_mask, (K, Q), "rank_mask")
        require(torch.equal(mask, checkpoint["state_dict"]["rank_mask"]), "loaded rank mask differs from saved mask")
        require(((mask == 0) | (mask == 1)).all(), "nonbinary rank mask")
        ranks = model.component_ranks
        require(torch.equal(ranks, mask.sum(-1).long()) and ((ranks >= 1) & (ranks <= Q)).all(), "invalid saved HDDC ranks")

    bundle = assignments(directory / "mfa_model_assignments.pt")
    responsibilities = tensor(bundle["max_responsibilities"], (N,), "max_responsibilities")
    require(((responsibilities >= 0) & (responsibilities <= 1 + 1e-6)).all(), "invalid max responsibilities")
    metrics = read_json(directory / "metrics.json")
    fields(metrics, {"schema_version": 2, "evaluation": "toy_manifold_tiling", "model_kind": kind, "K": K, "q_capacity": Q, "identity_hash": run["identity_hash"], "run_id": run["run_id"]}, "model metrics")
    fields(metrics["dataset"], {"selected_rows": N, "train_rows": 270000, "validation_rows": 30000, "subset_spec": None, "layer": 0}, "evaluated dataset")
    same_path(metrics["dataset"]["shard_dir"], condition / "dataset")
    require(all(math.isfinite(value) for value in metrics["nll"].values()), "non-finite NLL")
    bic = metrics["bic"]
    fields(bic, {"n": 270000, "K": K, "split": "train", "formula": "-standard_bic / n + active_components", "convention": "higher_is_better"}, "BIC")
    train_mask = torch.ones(N, dtype=torch.bool)
    train_mask[torch.tensor(split["val_global_rows"], dtype=torch.long)] = False
    train_sizes = torch.bincount(bundle["assignments"][train_mask].long(), minlength=K)
    active = int((train_sizes > 0).sum())
    fields(bic, {"active_components": active, "inactive_components": K - active, "activity_reward": active}, "BIC training activity")
    require(math.isfinite(bic["value"]) and math.isclose(bic["value"], -bic["standard_bic"] / bic["n"] + bic["active_components"], rel_tol=1e-10, abs_tol=1e-10), "inconsistent active BIC")
    live = int((bundle["cluster_sizes"] > 0).sum())
    fields(metrics["components"], {"live": live, "dead": K - live}, "components")
    population = association_count(metrics["association"])
    for rank_name in ("rank", "ambient_rank"):
        rank = metrics[rank_name]
        fields(rank, {"components": population, "definition": "hddc_rank_mask_count" if kind == "hddc" else "loading_variance_above_noise_threshold"}, rank_name)
        require("threshold" not in rank if kind == "hddc" else rank["threshold"] == 1.0, "incorrect rank threshold semantics")
    for name in ("tangent_alignment", "tangent_containment"):
        score_report(metrics[name], population)
    per_manifold = metrics["per_manifold"]
    require(len(per_manifold) == 10 and sum(item["components"]["associated"] for item in per_manifold) == population, "per-manifold coverage differs")
    for item in per_manifold:
        count = item["components"]["associated"]
        for name in ("tangent_alignment", "tangent_containment"):
            score_report(item[name], count)
    return {"stage": "models", "noise_ratio": noise, "model_kind": kind, "surgery_threshold": arguments.get("surgery_threshold"), "status": "validated", "manifest": str(manifest), "run_id": run["run_id"], "run_dir": str(directory), "rows": N, "nll": metrics["nll"], "active_bic": bic["value"], "live_components": live, "rank": metrics["rank"], "tangent_alignment": metrics["tangent_alignment"]["subspace_overlap"], "tangent_containment": metrics["tangent_containment"]["subspace_overlap"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("centroids", "models"), required=True)
    args = parser.parse_args()
    torch.set_num_threads(min(torch.get_num_threads(), 8))
    if args.stage == "centroids":
        for noise in (100, 1000):
            print(json.dumps(validate_centroids(noise), allow_nan=False), flush=True)
        return

    all_runs = []
    for noise in (100, 1000):
        for kind in ("hddc", "mfa"):
            config_path = REPO / f"configs/experiments/toy_noise{noise}_k1000_{kind}_cluster_pca.yaml"
            resolved = resolve_experiment(config_path, check_inputs=True)
            manifest = default_manifest_path(resolved)
            runs = read_manifest(manifest)
            require(runs == resolved, f"manifest differs from config: {manifest}")
            require(len(runs) == (9 if kind == "hddc" else 1), f"wrong run count: {manifest}")
            if kind == "hddc":
                require([run["training"]["arguments"]["surgery_threshold"] for run in runs] == THRESHOLDS, "wrong HDDC threshold sweep")
            all_runs.extend((noise, kind, manifest, run, status) for run, status in zip(runs, pipeline_status(runs)))
    require(len(all_runs) == len({run["run_dir"] for _, _, _, run, _ in all_runs}) == 20, "expected 20 unique model runs")
    incomplete = 0
    for noise, kind, manifest, run, status in all_runs:
        if status["pipeline"]:
            record = validate_model(run, noise, kind, manifest)
        else:
            incomplete += 1
            record = {"stage": "models", "noise_ratio": noise, "model_kind": kind,
                      "surgery_threshold": run["training"]["arguments"].get("surgery_threshold"),
                      "status": "incomplete", "manifest": str(manifest),
                      "run_id": run["run_id"], "run_dir": run["run_dir"], "pipeline_status": status}
        print(json.dumps(record, allow_nan=False), flush=True)
    if incomplete:
        raise SystemExit(f"{incomplete}/20 pipelines are incomplete.")


if __name__ == "__main__":
    main()
