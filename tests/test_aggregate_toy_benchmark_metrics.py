from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/temporary/aggregate_toy_benchmark_metrics.py"
)
SPEC = importlib.util.spec_from_file_location("aggregate_toy_benchmark_metrics", SCRIPT)
aggregator = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = aggregator
SPEC.loader.exec_module(aggregator)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _identity(manifold_id: int, *, embedding: bool = True) -> dict:
    result = {
        "manifold_id": manifold_id,
        "type_id": manifold_id,
        "type_name": ["segment", "circle"][manifold_id],
        "intrinsic_dim": 1,
    }
    if embedding:
        result["embedding_dim"] = manifold_id + 1
    return result


@pytest.fixture
def benchmark(tmp_path: Path) -> dict:
    root = tmp_path / "benchmark"
    condition = root / "noise_ratio_10"
    dataset = condition / "dataset"
    dataset_config = {
        "source_kind": "toy_manifolds",
        "d_model": 128,
        "num_rows": 80,
        "window": 1,
        "drop_prefix": 0,
        "layers": [0],
        "generator_config": {
            "ambient_dim": 128,
            "noise_ratio": 10.0,
            "seed": 0,
            "manifold_types": ["segment", "circle"],
            "manifolds_per_type": 1,
        },
    }
    _write_json(dataset / "config.json", dataset_config)
    run = condition / "models" / "experiment" / "hddc_run"
    arguments = {
        "K": 4,
        "rank": 32,
        "seed": 42,
        "split_seed": 43,
        "shard_dir": str(dataset),
        "layer": 0,
        "surgery_threshold": 0.15,
        "lr": 0.001,
    }
    spec = {
        "run_id": "hddc_run",
        "identity_hash": "fixture_identity",
        "training": {"model_kind": "hddc", "arguments": arguments},
        "dataset": {"shard_dir": str(dataset), "layer": 0},
        "evaluation": {"kind": "toy_manifold_tiling", "rank_threshold": 1.0},
    }
    config = {**arguments, "d_model": 128}
    trained_metrics = {
        "schema_version": 2,
        "evaluation": "toy_manifold_tiling",
        "run_id": "hddc_run",
        "identity_hash": "fixture_identity",
        "model_kind": "hddc",
        "K": 4,
        "q_capacity": 32,
        "dataset": {
            "shard_dir": str(dataset),
            "layer": 0,
            "selected_rows": 80,
            "train_rows": 72,
            "validation_rows": 8,
            "subset_spec": None,
        },
        "nll": {"train": -132.33384859969527, "validation": -130.3822494232178},
        "bic": {"value": 1244.8576658501852, "convention": "higher_is_better"},
        "clustering": {"adjusted_rand_index": 0.031147368706097977},
        "rank": {"mean_learned": 2.25, "definition": "hddc_rank_mask_count"},
        "components": {"live": 4, "dead": 0},
        "diagnostic": {"undefined": None, "large_count": 9007199254740993},
        "per_manifold": [
            {
                **_identity(i),
                "components": {"associated": i + 1},
                "rank": {"mean_learned": float(i + 1)},
                "tangent_alignment": {
                    "subspace_overlap": {"mean": None if i == 0 else 0.9876543210123456}
                },
            }
            for i in range(2)
        ],
    }
    _write_json(run / "run_spec.json", spec)
    _write_json(run / "config.json", config)
    _write_json(run / "metrics.json", trained_metrics)

    centroid_dir = condition / "centroids" / "kmeans_k4_full"
    baseline = centroid_dir / "initialization_evaluation"
    initialization_config = {
        "method": "kmeans",
        "K": 4,
        "seed": 0,
        "source_shard_dir": str(dataset),
        "layer": 0,
        "rows_used": 80,
        "shape": [80, 128],
        "cluster_sizes": [10, 20, 30, 20],
    }
    kmeans_metrics = {
        "schema_version": 2,
        "evaluation": "toy_kmeans_initialization_cattell_sweep",
        "partition_kind": "nearest_euclidean_centroid",
        "K": 4,
        "q_max": 16,
        "pca_capacity": 16,
        "dataset": {
            "shard_dir": str(dataset),
            "layer": 0,
            "selected_rows": 80,
            "in_sample": True,
            "subset_spec": None,
        },
        "artifacts": {"centroids_path": str(centroid_dir / "centroids.pt")},
        "eligibility": {"min_population": 26, "eligible_components": 2},
        "clustering": {"adjusted_rand_index": 0.031147368706097977},
        "quantization": {"inertia": 264964.09405542695},
        "pca_validation": {"max_relative_eigenvalue_error": 2.893451791084183e-14},
        "cattell": {
            "definition": "max_j_with_normalized_consecutive_eigengap_above_threshold",
            "thresholds": [0.01, 0.1],
        },
        "omitted_metrics": {"nll": "requires a probabilistic density"},
        "per_manifold": [
            {
                **_identity(i, embedding=False),
                "components": {"associated": 10 + i},
                "mean_to_manifold_distance": {"mean": 0.125 + i},
            }
            for i in range(2)
        ],
        "threshold_sweep": [
            {
                "cattell_threshold": threshold,
                "eligible_cluster_rank_distribution": {"1": 1, "2": 0, "16": 1},
                "rank": {
                    "mean_learned": 5.25 - position,
                    "definition": "raw_cattell_scree_rank",
                },
                "per_manifold": [
                    {
                        **_identity(i, embedding=False),
                        "rank": {"mean_learned": float(10 * position + i + 1)},
                        "tangent_containment": {"subspace_overlap": {"mean": None}},
                    }
                    for i in reversed(range(2))
                ],
            }
            for position, threshold in enumerate([0.01, 0.1])
        ],
    }
    _write_json(centroid_dir / "config.json", initialization_config)
    _write_json(baseline / "metrics.json", kmeans_metrics)
    paths_file = tmp_path / "paths.md"
    paths_file.write_text(f"- {run.parent}\n- {baseline}\n", encoding="utf-8")
    return {
        "root": root,
        "dataset": dataset,
        "run": run,
        "spec": spec,
        "config": config,
        "trained_metrics": trained_metrics,
        "baseline": baseline,
        "kmeans_metrics": kmeans_metrics,
        "paths_file": paths_file,
    }


def _aggregate(benchmark: dict):
    return aggregator.aggregate_benchmark(benchmark["paths_file"], benchmark["root"])


def _assert_leaves(row: pd.Series, values: dict, prefix: tuple[str, ...] = ()) -> None:
    for key, value in values.items():
        path = (*prefix, key)
        if isinstance(value, dict):
            _assert_leaves(row, value, path)
        elif not isinstance(value, list):
            actual = row["__".join(path)]
            if value is None:
                assert pd.isna(actual), path
            else:
                assert actual == value, path


def test_all_metric_leaves_and_scopes_are_preserved(benchmark: dict) -> None:
    frame, manifest, columns = _aggregate(benchmark)
    assert len(frame) == 9
    assert frame.evaluation_id.nunique() == 3
    assert len(manifest["sources"]) == 2
    assert not manifest["skipped_runs"]
    assert columns
    assert frame.groupby("evaluation_id").scope.apply(list).apply(
        lambda values: values.count("overall") == 1 and values.count("manifold") == 2
    ).all()

    trained = frame.loc[frame.model_kind == "hddc"]
    assert trained.run_id.eq("hddc_run").all()
    assert trained.noise_ratio.eq(10).all()
    assert trained.K.eq(4).all()
    assert trained.q_capacity.eq(32).all()
    assert trained.seed.eq(42).all()
    assert trained.split_seed.eq(43).all()
    assert trained.dataset_seed.eq(0).all()
    assert trained["training__surgery_threshold"].eq(0.15).all()
    assert trained.surgery_threshold.eq(0.15).all()
    assert trained["evaluation_config__rank_threshold"].eq(1.0).all()
    assert trained.cattell_threshold.isna().all()
    overall = trained.loc[trained.scope == "overall"].iloc[0]
    for key in ("nll", "bic", "clustering", "rank", "components", "diagnostic"):
        _assert_leaves(overall, {key: benchmark["trained_metrics"][key]})
    for source in benchmark["trained_metrics"]["per_manifold"]:
        row = trained.loc[trained.manifold_id == source["manifold_id"]].iloc[0]
        _assert_leaves(row, source)
        assert pd.isna(row["nll__validation"])
        assert pd.isna(row["clustering__adjusted_rand_index"])
        assert pd.isna(row["bic__value"])


def test_kmeans_thresholds_join_by_identity_and_keep_common_values(benchmark: dict) -> None:
    frame, _, _ = _aggregate(benchmark)
    baseline = frame.loc[frame.model_kind == "kmeans"]
    assert baseline.q_capacity.eq(16).all()
    assert baseline["initialization__seed"].eq(0).all()
    assert baseline["training__surgery_threshold"].isna().all()
    assert set(baseline.cattell_threshold) == {0.01, 0.1}
    report = benchmark["kmeans_metrics"]
    for setting in report["threshold_sweep"]:
        rows = baseline.loc[baseline.cattell_threshold == setting["cattell_threshold"]]
        assert rows.evaluation_id.nunique() == 1
        overall = rows.loc[rows.scope == "overall"].iloc[0]
        for key in (
            "clustering", "quantization", "pca_validation", "eligibility",
            "cattell", "omitted_metrics",
        ):
            _assert_leaves(overall, {key: report[key]})
        _assert_leaves(
            overall,
            {key: value for key, value in setting.items() if key != "per_manifold"},
        )
        for source in setting["per_manifold"]:
            row = rows.loc[rows.manifold_id == source["manifold_id"]].iloc[0]
            common = next(
                item for item in report["per_manifold"]
                if item["manifold_id"] == source["manifold_id"]
            )
            _assert_leaves(row, common)
            _assert_leaves(row, source)
            assert pd.isna(row["quantization__inertia"])
            assert pd.isna(row["clustering__adjusted_rand_index"])
    ids_before = baseline.groupby("cattell_threshold").evaluation_id.first().to_dict()
    report["threshold_sweep"].reverse()
    report["cattell"]["thresholds"].reverse()
    report["per_manifold"].reverse()
    for setting in report["threshold_sweep"]:
        setting["per_manifold"].reverse()
    _write_json(benchmark["baseline"] / "metrics.json", report)
    reordered, _, _ = _aggregate(benchmark)
    ids_after = (
        reordered.loc[reordered.model_kind == "kmeans"]
        .groupby("cattell_threshold").evaluation_id.first().to_dict()
    )
    assert ids_before == ids_after


def test_duplicate_and_symlink_sources_are_deduplicated(benchmark: dict) -> None:
    expected, _, _ = _aggregate(benchmark)
    alias = benchmark["paths_file"].parent / "baseline_alias"
    alias.symlink_to(benchmark["baseline"], target_is_directory=True)
    with benchmark["paths_file"].open("a", encoding="utf-8") as handle:
        handle.write(f"- {alias}\n- {benchmark['run']}\n")
    actual, manifest, _ = _aggregate(benchmark)
    assert len(manifest["sources"]) == 2
    assert_frame_equal(actual, expected)


def test_missing_reports_are_audited_and_backups_excluded(benchmark: dict, tmp_path: Path) -> None:
    missing = benchmark["run"].parent / "unfinished"
    _write_json(missing / "run_spec.json", {**benchmark["spec"], "run_id": "unfinished"})
    _write_json(missing / "config.json", benchmark["config"])
    (missing / "metrics.json.bak").write_text("invalid historical report")
    (benchmark["run"] / "metrics.before_augmented_bic.json").write_text(
        "invalid historical report"
    )
    for historical in ("archive", "epoch_0001"):
        _write_json(
            benchmark["run"] / historical / "metrics.json",
            {**benchmark["trained_metrics"], "run_id": "historical_run"},
        )
    frame, manifest, columns = _aggregate(benchmark)
    assert len(frame) == 9
    assert len(manifest["skipped_runs"]) == 1
    assert str(missing) in json.dumps(manifest["skipped_runs"])
    output = tmp_path / "aggregate"
    aggregator.write_outputs(frame, manifest, columns, output)
    with sqlite3.connect(output / "metrics.sqlite") as connection:
        assert connection.execute(
            "SELECT run_id, model_kind, noise_ratio, K, surgery_threshold, reason FROM skipped_runs"
        ).fetchall() == [("unfinished", "hddc", 10.0, 4, 0.15, "missing_metrics")]


@pytest.mark.parametrize(
    "field,value",
    [("run_id", "different_run"), ("identity_hash", "different_identity"),
     ("K", 99), ("q_capacity", 99), ("model_kind", "mfa")],
)
def test_inconsistent_trained_provenance_fails(
    benchmark: dict, field: str, value
) -> None:
    report = benchmark["trained_metrics"]
    report[field] = value
    _write_json(benchmark["run"] / "metrics.json", report)
    with pytest.raises(ValueError):
        _aggregate(benchmark)


@pytest.mark.parametrize(
    "change",
    ["wrong_type", "wrong_dimension", "missing_manifold", "duplicate_manifold",
     "overlapping_metric"],
)
def test_conflicting_kmeans_manifold_joins_fail(benchmark: dict, change: str) -> None:
    report = benchmark["kmeans_metrics"]
    manifold_rows = report["threshold_sweep"][0]["per_manifold"]
    if change == "wrong_type":
        manifold_rows[0]["type_name"] = "sphere"
    elif change == "wrong_dimension":
        manifold_rows[0]["intrinsic_dim"] = 2
    elif change == "missing_manifold":
        manifold_rows.pop()
    elif change == "duplicate_manifold":
        manifold_rows[1] = copy.deepcopy(manifold_rows[0])
    else:
        manifold_rows[0]["components"] = {"associated": 999}
    _write_json(benchmark["baseline"] / "metrics.json", report)
    with pytest.raises(ValueError):
        _aggregate(benchmark)


@pytest.mark.parametrize(
    "change",
    ["wrong_schema", "wrong_evaluation", "duplicate_threshold",
     "undeclared_threshold", "invalid_json"],
)
def test_malformed_reports_fail(benchmark: dict, change: str) -> None:
    report = benchmark["kmeans_metrics"]
    path = benchmark["baseline"] / "metrics.json"
    if change == "wrong_schema":
        report["schema_version"] = 999
    elif change == "wrong_evaluation":
        report["evaluation"] = "unrelated_evaluation"
    elif change == "duplicate_threshold":
        report["threshold_sweep"][1]["cattell_threshold"] = 0.01
    elif change == "undeclared_threshold":
        report["threshold_sweep"][1]["cattell_threshold"] = 0.5
    _write_json(path, report)
    if change == "invalid_json":
        path.write_text("{ malformed JSON")
    with pytest.raises(ValueError):
        _aggregate(benchmark)


def test_column_separator_collision_fails(benchmark: dict) -> None:
    report = benchmark["trained_metrics"]
    report["extra"] = {"a__b": 1, "a": {"b": 2}}
    _write_json(benchmark["run"] / "metrics.json", report)
    with pytest.raises(ValueError):
        _aggregate(benchmark)


def test_source_mappings_hashes_and_configuration_lists_are_recorded(
    benchmark: dict,
) -> None:
    frame, manifest, columns = _aggregate(benchmark)
    assert set(columns) == set(frame.columns)
    assert columns["nll__validation"]["scopes"] == ["overall"]
    assert set(columns["rank__mean_learned"]["scopes"]) == {"overall", "manifold"}
    assert {"role": "metrics", "json_path": "$.nll.validation"} in (
        columns["nll__validation"]["sources"]
    )
    for source in manifest["input_files"]:
        assert source["sha256"] == hashlib.sha256(
            Path(source["path"]).read_bytes()
        ).hexdigest()
    initialization = manifest["configuration_metadata"][
        str(benchmark["baseline"].parent / "config.json")
    ]
    assert initialization["shape"] == [80, 128]
    assert initialization["cluster_sizes"] == [10, 20, 30, 20]


def test_changed_source_is_rejected_before_writing(
    benchmark: dict, tmp_path: Path
) -> None:
    frame, manifest, columns = _aggregate(benchmark)
    source = benchmark["run"] / "metrics.json"
    source.write_text(source.read_text() + "\n")
    output_dir = tmp_path / "aggregate"
    with pytest.raises(ValueError, match="Input changed"):
        aggregator.write_outputs(frame, manifest, columns, output_dir)
    assert not output_dir.exists()


def test_source_change_during_writing_preserves_existing_outputs(
    benchmark: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame, manifest, columns = _aggregate(benchmark)
    output_dir = tmp_path / "aggregate"
    aggregator.write_outputs(frame, manifest, columns, output_dir)
    previous = {path.name: path.read_bytes() for path in output_dir.iterdir()}
    original_to_parquet = pd.DataFrame.to_parquet

    def to_parquet_then_change_source(self, *args, **kwargs):
        result = original_to_parquet(self, *args, **kwargs)
        source = benchmark["run"] / "metrics.json"
        source.write_text(source.read_text() + "\n")
        return result

    monkeypatch.setattr(pd.DataFrame, "to_parquet", to_parquet_then_change_source)
    with pytest.raises(ValueError, match="Input changed"):
        aggregator.write_outputs(frame, manifest, columns, output_dir)
    assert {path.name: path.read_bytes() for path in output_dir.iterdir()} == previous


def test_other_benchmark_dataset_is_rejected(benchmark: dict) -> None:
    foreign = (
        benchmark["root"].parent / "other_benchmark" / "noise_ratio_10" / "dataset"
    )
    _write_json(
        foreign / "config.json",
        json.loads((benchmark["dataset"] / "config.json").read_text()),
    )
    report = benchmark["trained_metrics"]
    report["dataset"]["shard_dir"] = str(foreign)
    _write_json(benchmark["run"] / "metrics.json", report)
    with pytest.raises(ValueError):
        _aggregate(benchmark)


@pytest.mark.parametrize("matching_config", [True, False])
def test_declared_legacy_dataset_alias_requires_matching_configuration(
    benchmark: dict, matching_config: bool
) -> None:
    legacy = benchmark["root"].parent / "assets" / "legacy_noise_ratio_10"
    config = json.loads((benchmark["dataset"] / "config.json").read_text())
    if not matching_config:
        config["generator_config"]["seed"] = 999
    _write_json(legacy / "config.json", config)
    _write_json(
        benchmark["root"] / "experiment.json",
        {"conditions": [{"noise_ratio": 10.0, "dataset": str(legacy)}]},
    )
    report = benchmark["trained_metrics"]
    report["dataset"]["shard_dir"] = str(legacy)
    _write_json(benchmark["run"] / "metrics.json", report)
    if matching_config:
        frame, manifest, _ = _aggregate(benchmark)
        assert len(frame) == 9
        assert frame.loc[frame.model_kind == "hddc", "dataset__shard_dir"].eq(
            str(legacy)
        ).all()
        assert str(legacy / "config.json") in manifest["configuration_metadata"]
    else:
        with pytest.raises(ValueError, match="Dataset provenance mismatch"):
            _aggregate(benchmark)


def test_mixed_integer_float_metric_rejects_loss_of_precision(benchmark: dict) -> None:
    report = benchmark["kmeans_metrics"]
    report["diagnostic"] = {"large_count": 0.25}
    _write_json(benchmark["baseline"] / "metrics.json", report)
    with pytest.raises(ValueError, match="Value changed"):
        _aggregate(benchmark)


@pytest.mark.parametrize("destination", ["benchmark", "trained", "baseline", "paths_parent"])
def test_output_directory_cannot_overwrite_source_locations(
    benchmark: dict, destination: str
) -> None:
    frame, manifest, columns = _aggregate(benchmark)
    output_dir = {
        "benchmark": benchmark["root"],
        "trained": benchmark["run"],
        "baseline": benchmark["baseline"],
        "paths_parent": benchmark["paths_file"].parent,
    }[destination]
    before = {path: path.read_bytes() for path in benchmark["root"].rglob("*.json")}
    with pytest.raises(ValueError):
        aggregator.write_outputs(frame, manifest, columns, output_dir)
    assert not (output_dir / "metrics.parquet").exists()
    assert all(path.read_bytes() == content for path, content in before.items())


def test_parquet_round_trip_preserves_precision_and_nullable_integers(
    benchmark: dict, tmp_path: Path
) -> None:
    source_bytes = {
        path: path.read_bytes() for path in benchmark["root"].rglob("*.json")
    }
    frame, manifest, columns = _aggregate(benchmark)
    output_dir = tmp_path / "aggregate"
    aggregator.write_outputs(frame, manifest, columns, output_dir)
    restored = pd.read_parquet(output_dir / "metrics.parquet")
    assert_frame_equal(restored, frame, check_exact=True)
    assert pd.api.types.is_integer_dtype(restored.manifold_id.dtype)
    assert pd.api.types.is_integer_dtype(restored["diagnostic__large_count"].dtype)
    assert restored["diagnostic__large_count"].dropna().iloc[0] == 9007199254740993
    assert restored["nll__validation"].dropna().iloc[0] == -130.3822494232178
    assert (
        restored["pca_validation__max_relative_eigenvalue_error"].dropna().iloc[0]
        == 2.893451791084183e-14
    )
    assert restored["diagnostic__undefined"].isna().all()
    assert {"metrics.parquet", "metrics.sqlite", "manifest.json", "columns.json", "README.md"} <= {
        path.name for path in output_dir.iterdir()
    }
    assert (
        json.loads((output_dir / "manifest.json").read_text())["sources"]
        == manifest["sources"]
    )
    assert json.loads((output_dir / "columns.json").read_text()) == columns
    assert all(path.read_bytes() == content for path, content in source_bytes.items())
    with sqlite3.connect(output_dir / "metrics.sqlite") as connection:
        assert connection.execute("SELECT count(*) FROM overall_metrics").fetchone() == (3,)
        assert connection.execute("SELECT count(*) FROM manifold_metrics").fetchone() == (6,)
        assert connection.execute("SELECT count(*) FROM skipped_runs").fetchone() == (0,)
        assert connection.execute(
            "SELECT diagnostic__large_count, typeof(diagnostic__large_count), nll__validation "
            "FROM overall_metrics WHERE model_kind = 'hddc'"
        ).fetchone() == (9007199254740993, "integer", -130.3822494232178)
        assert json.loads(connection.execute(
            "SELECT value_json FROM metadata WHERE name = 'columns'"
        ).fetchone()[0]) == columns
    saved_manifest = json.loads((output_dir / "manifest.json").read_text())
    assert saved_manifest["sqlite_sha256"] == hashlib.sha256((output_dir / "metrics.sqlite").read_bytes()).hexdigest()
    assert str(output_dir / "metrics.parquet") in (output_dir / "README.md").read_text()


def test_full_benchmark_discovery_preserves_all_sweep_axes(benchmark: dict) -> None:
    for index, (noise, k, threshold) in enumerate([(10, 8, 0.15), (10, 4, 0.25), (100, 4, 0.15)]):
        condition = benchmark["root"] / f"noise_ratio_{noise}"
        dataset = condition / "dataset"
        dataset_config = json.loads((benchmark["dataset"] / "config.json").read_text())
        dataset_config["generator_config"]["noise_ratio"] = noise
        _write_json(dataset / "config.json", dataset_config)
        run_id = f"new_run_{index}"
        run = condition / "models" / "new_sweep" / run_id
        spec = copy.deepcopy(benchmark["spec"])
        spec["run_id"] = run_id
        spec["dataset"]["shard_dir"] = str(dataset)
        spec["training"]["arguments"].update(K=k, surgery_threshold=threshold, shard_dir=str(dataset))
        report = copy.deepcopy(benchmark["trained_metrics"])
        report.update(K=k, run_id=run_id)
        report["dataset"]["shard_dir"] = str(dataset)
        _write_json(run / "run_spec.json", spec)
        _write_json(run / "config.json", {**spec["training"]["arguments"], "d_model": 128})
        _write_json(run / "metrics.json", report)

    restricted, _, _ = _aggregate(benchmark)
    assert len(restricted) == 9
    frame, manifest, _ = aggregator.aggregate_benchmark(None, benchmark["root"])
    assert len(frame) == 18
    assert manifest["paths_file"] is None
    assert manifest["source_roots"] == [str(benchmark["root"])]
    assert manifest["coverage"]["reports"] == 5
    assert manifest["sweep_values"]["K"] == [4, 8]
    assert manifest["sweep_values"]["noise_ratio"] == [10, 100]
    assert manifest["sweep_values"]["surgery_threshold"] == [0.15, 0.25]
    trained = frame.query("scope == 'overall' and model_kind == 'hddc'")
    assert set(trained[["noise_ratio", "K", "surgery_threshold"]].itertuples(index=False, name=None)) == {
        (10, 4, 0.15), (10, 8, 0.15), (10, 4, 0.25), (100, 4, 0.15),
    }
    manifest["output_dir"] = str(benchmark["root"].parent / "aggregate")
    assert "--paths-file" not in aggregator.handoff_readme(manifest).split("```bash")[1].split("```")[0]


@pytest.mark.parametrize("corruption", [None, "path", "plan", "training"])
def test_pipeline_initialization_identity_is_resolved_strictly(benchmark: dict, corruption: str | None) -> None:
    spec = copy.deepcopy(benchmark["spec"])
    spec["initialization"] = {"method": "kmeans_pca", "pca_rank": 32}
    spec["identity"] = {
        "dataset": copy.deepcopy(spec["dataset"]),
        "model_kind": "hddc",
        "training_args": {**spec["training"]["arguments"], "centroids_path": None},
        "initialization": copy.deepcopy(spec["initialization"]),
    }
    init_dir = benchmark["run"] / ("wrong_directory" if corruption == "path" else "initialization")
    spec["training"]["arguments"]["centroids_path"] = str(init_dir / "centroids.pt")
    _write_json(init_dir / "config.json", {
        "K": 4, "source_shard_dir": str(benchmark["dataset"]), "layer": 0,
    })
    if corruption == "plan":
        spec["identity"]["initialization"]["pca_rank"] = 16
    elif corruption == "training":
        spec["identity"]["training_args"]["surgery_threshold"] = 0.5
    _write_json(benchmark["run"] / "run_spec.json", spec)
    _write_json(benchmark["run"] / "config.json", {**spec["training"]["arguments"], "d_model": 128})
    if corruption is None:
        frame, _, _ = _aggregate(benchmark)
        assert len(frame) == 9
    else:
        with pytest.raises(ValueError, match="identity|initialization path"):
            _aggregate(benchmark)
