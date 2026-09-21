"""Exercise the all-fit gate, provenance checks, resume, and sparse geometry audit."""

import copy
import json

import pytest
import torch

from dalg.init.centroid_artifact import save_centroid_artifact
from scripts.temporary import run_toy_noise_kmeans_k_sweep as sweep


@pytest.fixture
def small_dimensions(monkeypatch):
    monkeypatch.setattr(sweep, "N", 6000)
    monkeypatch.setattr(sweep, "D", 4)
    monkeypatch.setattr(sweep, "Q", 2)
    monkeypatch.setattr(sweep, "MIN_POPULATION", 3)
    return {"val_frac": 0.0, "drop_prefix": 0, "train_rows": 6000}


def training_artifact(run, selection):
    run.directory.mkdir(parents=True)
    sizes = [2] * run.k
    sizes[0] += sweep.N - sum(sizes)
    config = {
        "method": "kmeans", "metric": "euclidean", "K": run.k,
        "shape": [sweep.N, sweep.D], "uses_all_rows": True,
        "uses_all_training_rows": True, "rows_used": sweep.N, "source_rows": sweep.N,
        "sample_fraction_requested": 1.0, "sample_fraction_actual": 1.0,
        "sample_seed": 0, "seed": 0, "restarts": 10, "max_iter": 1000,
        "tol": 1e-6, "layer": 0, "centroid_artifact_format": "dalg_centroids_v1",
        "principal_components": None, "selection": selection,
        "source_shard_dir": str(run.shards), "cluster_sizes": sizes,
        "inertia": float(sweep.N),
    }
    (run.directory / "config.json").write_text(json.dumps(config))
    save_centroid_artifact(run.directory / "centroids.pt", torch.randn(run.k, sweep.D), None)
    return config


def test_exact_matrix_and_dataset_pairing(tmp_path):
    rows = sweep.runs(tmp_path)
    assert [(r.noise, r.k) for r in rows] == [
        (10, 250), (10, 500), (10, 2000),
        (100, 250), (100, 500), (100, 2000),
        (1000, 250), (1000, 500), (1000, 2000),
    ]
    assert len({r.directory for r in rows}) == 9
    for run in rows:
        assert run.shards == tmp_path / f"noise_ratio_{run.noise}/dataset"
        assert run.directory == tmp_path / f"noise_ratio_{run.noise}/centroids/kmeans_k{run.k}_full"


@pytest.mark.parametrize("damage", ["missing", "corrupt", "wrong_noise", "wrong_k"])
def test_training_failure_blocks_every_evaluation(tmp_path, monkeypatch, small_dimensions, damage):
    rows = sweep.runs(tmp_path)
    for run in rows:
        training_artifact(run, small_dimensions)
    broken = rows[4]
    path = broken.directory / "centroids.pt"
    if damage == "missing":
        path.unlink()
    elif damage == "corrupt":
        path.write_bytes(b"not a torch artifact")
    else:
        config_path = broken.directory / "config.json"
        config = json.loads(config_path.read_text())
        if damage == "wrong_noise":
            config["source_shard_dir"] = str(rows[0].shards)
        else:
            config["K"] = 1000
        config_path.write_text(json.dumps(config))
    attempted = []
    monkeypatch.setattr(sweep, "fit", lambda run, selection: attempted.append(run))
    monkeypatch.setattr(sweep, "evaluate", lambda *args: pytest.fail("evaluation crossed a failed training gate"))
    status = tmp_path / "status.md"
    assert sweep.execute(rows, dict.fromkeys(sweep.NOISES, small_dimensions), status) == 1
    assert attempted == rows
    state = json.loads(status.with_suffix(".json").read_text())
    assert state["training_completed"] == 8
    assert state["evaluation_completed"] == 0
    assert not state["training_gate_passed"]
    assert not state["rows"][4]["training"]


def test_resume_valid_fit_and_refuse_incompatible_output(tmp_path, monkeypatch, small_dimensions):
    run = sweep.Run(tmp_path, 10, 250)
    training_artifact(run, small_dimensions)
    monkeypatch.setattr(sweep, "command", lambda *args: pytest.fail("attempted to overwrite fit"))
    before = (run.directory / "centroids.pt").read_bytes()
    sweep.fit(run, small_dimensions)
    config_path = run.directory / "config.json"
    config = json.loads(config_path.read_text())
    config["max_iter"] = 100
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="max_iter"):
        sweep.fit(run, small_dimensions)
    assert (run.directory / "centroids.pt").read_bytes() == before


def test_partial_evaluation_unchecked_and_not_overwritten(tmp_path, monkeypatch, small_dimensions):
    run = sweep.Run(tmp_path, 100, 250)
    training_artifact(run, small_dimensions)
    run.evaluation.mkdir()
    (run.evaluation / "metrics.json").write_text("{}")
    monkeypatch.setattr(sweep, "command", lambda *args: pytest.fail("overwrote partial evaluation"))
    result = sweep.audit([run], {100: small_dimensions})[0]
    assert result["training"] and not result["evaluation"]
    with pytest.raises(FileNotFoundError):
        sweep.evaluate(run, small_dimensions)
    assert (run.evaluation / "metrics.json").read_text() == "{}"


def test_reuse_assignments_and_completed_evaluation(tmp_path, monkeypatch, small_dimensions):
    run = sweep.Run(tmp_path, 100, 250)
    training_artifact(run, small_dimensions)
    run.evaluation.mkdir()
    run.assignments.touch()
    calls = []
    monkeypatch.setattr(sweep, "validate_assignments", lambda *args: calls.append("assignments checked"))
    monkeypatch.setattr(sweep, "validate_evaluation", lambda *args: calls.append("evaluation checked"))
    monkeypatch.setattr(sweep, "command", lambda args: calls.append(str(args[1])))
    sweep.evaluate(run, small_dimensions)
    assert calls == ["assignments checked", str(sweep.REPO / "scripts/temporary/evaluate_toy_kmeans_geometry.py"),
                     "evaluation checked"]
    (run.evaluation / "metrics.json").touch()
    calls.clear()
    sweep.evaluate(run, small_dimensions)
    assert calls == ["evaluation checked"]


def test_successful_gate_order_and_job_id_survives_audit(tmp_path, monkeypatch):
    rows = sweep.runs(tmp_path)
    trained, evaluated = set(), set()
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    def fit(run, selection):
        trained.add(run)
    def evaluate(run, selection):
        assert trained == set(rows)
        evaluated.add(run)
    def audit(runs, selections):
        return [{"noise": r.noise, "k": r.k, "training": r in trained,
                 "evaluation": r in evaluated, "errors": []} for r in runs]
    monkeypatch.setattr(sweep, "fit", fit)
    monkeypatch.setattr(sweep, "evaluate", evaluate)
    monkeypatch.setattr(sweep, "audit", audit)
    status = tmp_path / "status.md"
    selections = dict.fromkeys(sweep.NOISES, {})
    assert sweep.execute(rows, selections, status) == 0
    monkeypatch.setenv("SLURM_JOB_ID", "unrelated-interactive-job")
    sweep.write_status(status, rows, audit(rows, selections), "check-evaluation", [])
    state = json.loads(status.with_suffix(".json").read_text())
    assert state["slurm_job_id"] == "12345"
    assert state["training_completed"] == state["evaluation_completed"] == 9
    assert status.read_text().count("☑") == 18


def geometry_reports(run):
    k, q, t = run.k, sweep.Q, len(sweep.THRESHOLDS)
    sizes = torch.tensor([sweep.N - 8, 7, 1])
    eligible = torch.tensor([True, True, False])
    associated = torch.ones(k, dtype=torch.bool)
    spectrum = torch.tensor([[2., .5, .1], [1., .6, .2], [0., 0., 0.]], dtype=torch.float64)
    gaps, ranks = sweep._cattell_rank_sweep(spectrum, eligible, q_max=q, thresholds=sweep.THRESHOLDS)
    pcs = torch.zeros(k, sweep.D, q, dtype=torch.float64)
    pcs[eligible] = torch.eye(sweep.D, dtype=torch.float64)[:, :q]
    targets = torch.tensor([1, 2, 1])
    # The second eligible cluster has undefined tangent scores; that is valid.
    defined = torch.tensor([True, False, False])
    scores = torch.tensor([.75, 0., 0.], dtype=torch.float64)
    alignment = {**sweep._alignment_summary(scores, scores, defined, eligible),
                 "relative_boundary_eigengap_threshold": 1e-6}
    details = {
        "schema_version": 1, "evaluation": "toy_kmeans_initialization_cattell_sweep",
        "K": k, "q_max": q, "min_population": sweep.MIN_POPULATION,
        "cluster_sizes": sizes, "eligible": eligible,
        "cattell_thresholds": torch.tensor(sweep.THRESHOLDS, dtype=torch.float64),
        "leading_eigenvalues": spectrum, "normalized_cattell_gaps": gaps,
        "cattell_ranks": ranks, "principal_components": pcs,
        "principal_components_defined": eligible.clone(), "associated": associated,
        "associated_manifold_indices": torch.arange(k), "target_intrinsic_dims": targets,
        "alignment_defined": defined, "alignment_overlap": scores,
        "alignment_worst_direction_cosine": scores,
        "containment_defined": defined.repeat(t, 1), "containment_overlap": scores.repeat(t, 1),
        "containment_worst_direction_cosine": scores.repeat(t, 1),
    }
    bundle = {"cluster_sizes": sizes, "min_distances": torch.ones(sweep.N)}
    metrics = {
        "schema_version": 2, "evaluation": details["evaluation"], "K": k,
        "partition_kind": "nearest_euclidean_centroid", "q_max": q, "pca_capacity": q,
        "dataset": {"selected_rows": sweep.N, "layer": 0, "subset_spec": None,
                    "in_sample": True, "shard_dir": str(run.shards)},
        "artifacts": {"centroids_path": str(run.directory / "centroids.pt"),
                      "assignments_path": str(run.assignments),
                      "component_details_path": str(run.evaluation / "component_metrics.pt")},
        "eligibility": {"min_population": sweep.MIN_POPULATION, "eligible_components": 2,
                        "excluded_components": 1, "eligible_points": sweep.N - 1, "excluded_points": 1},
        "cattell": {"thresholds": list(sweep.THRESHOLDS), "q_max": q,
                    "comparison": "strict_greater_than", "shared_noise_active_set_applied": False},
        "pca_validation": {"source": "computed_from_assignments", "covariance_center": "empirical_cluster_mean",
                           "centroid_mean_agreement_required": False, "pca_tolerance": 1e-4,
                           "centroid_mean_l2_tolerance": 1e-5, "max_orthonormal_error": 0.,
                           "max_relative_subspace_residual": 0., "max_relative_eigenvalue_error": 0.},
        "quantization": {"inertia": float(sweep.N)}, "clustering": {"homogeneity": .5},
        "association": {"rule": "unique_nearest_exact_projection", "max_mean_to_manifold_distance": None,
                        "outside_cutoff_components": 0, "associated_components": k,
                        "eligible_associated_components": 2, "ambiguous_components": 0},
        "per_manifold": [{"components": {"associated": int(i < k),
                                          "eligible_associated": int(i < 2)}} for i in range(10)],
        "tangent_alignment": alignment,
        "threshold_sweep": [{"cattell_threshold": threshold, "tangent_containment": copy.deepcopy(alignment),
                             "rank": sweep._rank_summary(ranks[i], targets, eligible)}
                            for i, threshold in enumerate(sweep.THRESHOLDS)],
    }
    return bundle, metrics, details


def test_sparse_pca_and_undefined_geometry_accepted(tmp_path, small_dimensions):
    run = sweep.Run(tmp_path, 10, 3)
    sweep.validate_evaluation_reports(run, *geometry_reports(run))


@pytest.mark.parametrize("damage", ["eligible", "pc_mask", "rank", "scores", "spectrum", "counts", "thresholds"])
def test_geometry_audit_rejects_inconsistent_reports(tmp_path, small_dimensions, damage):
    run = sweep.Run(tmp_path, 10, 3)
    bundle, metrics, details = geometry_reports(run)
    if damage == "eligible":
        details["eligible"][2] = True
    elif damage == "pc_mask":
        details["principal_components_defined"][2] = True
    elif damage == "rank":
        details["cattell_ranks"][0, 2] = 1
    elif damage == "scores":
        details["alignment_defined"][2] = True
    elif damage == "spectrum":
        details["leading_eigenvalues"][0, 0] = float("nan")
    elif damage == "counts":
        metrics["tangent_alignment"]["subspace_overlap"]["undefined_components"] = 0
    else:
        metrics["cattell"]["thresholds"] = [.1]
    with pytest.raises(ValueError):
        sweep.validate_evaluation_reports(run, bundle, metrics, details)
