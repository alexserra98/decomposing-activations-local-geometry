from __future__ import annotations

import json

import pandas as pd
import pytest

from dalg.analysis.aggregate_metrics import aggregate_metrics, main


def _write_report(directory, report, spec=None):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "metrics.json").write_text(json.dumps(report))
    if spec is not None:
        (directory / "run_spec.json").write_text(json.dumps(spec))


def _mfa_report():
    return {
        "run_id": "saved-run",
        "evaluation": "toy_manifold_tiling",
        "model_kind": "hddc",
        "K": 100,
        "nll": {"train": -12.0, "validation": -10.0},
        "rank": {"mean_learned": 2.5},
        "per_manifold": [
            {
                "manifold_id": 1, "type_name": "circle", "intrinsic_dim": 1,
                "components": {"associated": 0},
                "rank": {"mean_learned": None},
            },
            {
                "manifold_id": 0, "type_name": "line", "intrinsic_dim": 1,
                "components": {"associated": 100},
                "rank": {"mean_learned": 2.5},
            },
        ],
    }


def _spec(threshold=0.1, k=100, seed=42):
    return {
        "run_id": "saved-run",
        "training": {
            "model_kind": "hddc",
            "arguments": {"K": k, "surgery_threshold": threshold, "seed": seed},
        },
        "evaluation": {"enabled": True, "rank_threshold": 1.0},
        "tags": ["example", "sweep"],
    }


def _kmeans_report():
    return {
        "evaluation": "toy_manifold_tiling",
        "model_kind": "kmeans",
        "K": 20,
        "clustering": {"homogeneity": 0.9},
        "rank": {"mean_learned": 3.1},
        "rank_distribution": {"1": 5, "2": 15},
        "per_manifold": [
            {
                "manifold_id": 1, "type_name": "circle",
                "components": {"associated": 8}, "rank": {"mean_learned": 4.1},
            },
            {
                "manifold_id": 0, "type_name": "line",
                "components": {"associated": 12}, "rank": {"mean_learned": 2.1},
            },
        ],
    }


def test_mfa_configurations_remain_separate_and_repeat_config(tmp_path):
    settings = [(100, 0.1, 42), (100, 0.5, 42), (300, 0.1, 42), (100, 0.1, 43)]
    for index, (k, threshold, seed) in enumerate(settings):
        report = _mfa_report()
        report["K"] = k
        _write_report(tmp_path / f"condition_{index}" / "run", report, _spec(threshold, k, seed))

    general, manifolds = aggregate_metrics(tmp_path)

    assert len(general) == 4
    assert len(manifolds) == 8
    assert general["run_id"].tolist() == ["saved-run"] * 4
    assert general["evaluation_id"].is_unique
    assert list(general[["K", "surgery_threshold", "config.training.arguments.seed"]]
                .itertuples(index=False, name=None)) == settings
    assert general.columns[:3].tolist() == ["source_path", "run_id", "evaluation_id"]
    assert general["nll.validation"].tolist() == [-10.0] * 4
    assert not any(name.startswith("per_manifold") for name in general)
    assert "nll.validation" not in manifolds
    assert manifolds["manifold_id"].tolist() == [0, 1] * 4
    assert manifolds.loc[manifolds.manifold_id == 1, "components.associated"].tolist() == [0] * 4
    assert manifolds.loc[manifolds.manifold_id == 1, "rank.mean_learned"].isna().all()
    for row in general.to_dict("records"):
        children = manifolds.loc[manifolds.evaluation_id == row["evaluation_id"]]
        for column in ["K", "surgery_threshold", *[c for c in general if c.startswith("config.")]]:
            assert children[column].tolist() == [row[column]] * 2
    assert not list(tmp_path.glob("*.csv"))


def test_mixed_reports_keep_one_row_per_run_and_join_manifolds_by_id(tmp_path):
    _write_report(tmp_path / "hddc", _mfa_report(), _spec())
    _write_report(tmp_path / "kmeans", _kmeans_report())
    mfa = _mfa_report()
    mfa["model_kind"] = "mfa"
    _write_report(tmp_path / "mfa", mfa)

    general, manifolds = aggregate_metrics(tmp_path)

    assert len(general) == 3
    assert len(manifolds) == 6
    assert general.model_kind.tolist() == ["hddc", "kmeans", "mfa"]
    kmeans = general.loc[general.model_kind == "kmeans"]
    assert kmeans["rank.mean_learned"].tolist() == [3.1]
    assert kmeans["clustering.homogeneity"].tolist() == [0.9]
    assert kmeans["rank_distribution.1"].tolist() == [5]
    assert kmeans["run_id"].tolist() == ["kmeans"]
    assert kmeans["nll.validation"].isna().all()
    assert kmeans["config.training.arguments.seed"].isna().all()
    children = manifolds.loc[manifolds.model_kind == "kmeans"]
    assert children["manifold_id"].tolist() == [0, 1]
    assert children["rank.mean_learned"].tolist() == [2.1, 4.1]
    assert children["components.associated"].tolist() == [12, 8]
    assert children["type_name"].tolist() == ["line", "circle"]
    assert set(children.evaluation_id) == set(kmeans.evaluation_id)
    pd.testing.assert_frame_equal(general, aggregate_metrics(tmp_path)[0])
    pd.testing.assert_frame_equal(manifolds, aggregate_metrics(tmp_path)[1])


def test_discovery_includes_root_and_nested_reports_but_not_history_or_symlinks(tmp_path):
    root = tmp_path / "experiment"
    _write_report(root, {"score": 1})
    _write_report(root / "nested" / "run", {"score": 2})
    for excluded in ["centroids", "archive", "archived", "checkpoints", "snapshots"]:
        _write_report(root / "nested" / excluded / "run", {})
        (root / "nested" / excluded / "run" / "metrics.json").write_text("{broken")
    external = tmp_path / "external"
    _write_report(external, {"score": 3})
    (root / "linked-run").symlink_to(external, target_is_directory=True)
    (root / "cycle").symlink_to(root, target_is_directory=True)
    alias = tmp_path / "alias"
    alias.symlink_to(root, target_is_directory=True)

    general, manifolds = aggregate_metrics(alias)

    assert general["source_path"].tolist() == ["metrics.json", "nested/run/metrics.json"]
    assert general["run_id"].tolist() == [".", "nested/run"]
    assert manifolds.empty
    assert "manifold_id" in manifolds


def test_optional_config_lists_and_missing_values_are_preserved(tmp_path):
    large_count = 2**60 + 1
    _write_report(tmp_path / "a", {
        "count": large_count, "nullable": None, "items": [{"x": 1}, 2], "empty": {},
    }, {"training": {"model_kind": "mfa", "arguments": {"K": 10}}})
    _write_report(tmp_path / "b", {"count": None, "other": 3})

    general, manifolds = aggregate_metrics(tmp_path)

    assert general.loc[0, "count"] == large_count
    assert pd.isna(general.loc[1, "count"])
    assert general["nullable"].isna().all()
    assert pd.isna(general.loc[0, "other"])
    assert json.loads(general.loc[0, "items"]) == [{"x": 1}, 2]
    assert general.loc[0, "empty"] == "{}"
    assert general.loc[0, "model_kind"] == "mfa"
    assert general.loc[0, "K"] == 10
    assert general["surgery_threshold"].isna().all()
    assert manifolds.empty


@pytest.mark.parametrize("case, message", [
    ("duplicate_manifold", "duplicate manifold_id"),
    ("flatten_collision", "flattened-column collision"),
    ("config_collision", "flattened-column collision"),
    ("config_metric_collision", "flattened-column collision"),
    ("wrong_config", "conflicting values for K"),
    ("wrong_model", "conflicting values for model_kind"),
    ("wrong_run_id", "conflicting values for run_id"),
    ("invalid_manifolds", "per_manifold must be a list"),
    ("invalid_entry", "per_manifold entries must be objects"),
    ("missing_id", "manifold_id must be an integer"),
    ("manifold_collision", "flattened-column collision"),
])
def test_invalid_reports_fail_with_source_path(tmp_path, case, message):
    report, spec = _kmeans_report(), None
    if case == "duplicate_manifold":
        report["per_manifold"].append(report["per_manifold"][0])
    elif case == "flatten_collision":
        report["clustering.homogeneity"] = 0.9
    elif case == "config_collision":
        spec = {"training": {"seed": 42}, "training.seed": 42}
    elif case == "config_metric_collision":
        spec = {"seed": 42}
        report["config.seed"] = 42
    elif case == "wrong_config":
        spec = {"training": {"arguments": {"K": 999}}}
    elif case == "wrong_model":
        spec = {"training": {"model_kind": "mfa"}}
    elif case == "wrong_run_id":
        spec = {"run_id": "different"}
        report["run_id"] = "saved"
    elif case == "invalid_manifolds":
        report["per_manifold"] = None
    elif case == "invalid_entry":
        report["per_manifold"] = [None]
    elif case == "missing_id":
        del report["per_manifold"][0]["manifold_id"]
    elif case == "manifold_collision":
        report["per_manifold"][0]["rank.mean_learned"] = 0
    _write_report(tmp_path / "run", report, spec)
    with pytest.raises(ValueError, match=message) as error:
        aggregate_metrics(tmp_path)
    filename = "run_spec.json" if case == "config_collision" else "metrics.json"
    assert str(tmp_path / "run" / filename) in str(error.value)


def test_missing_or_empty_experiment(tmp_path):
    with pytest.raises(ValueError, match="directory does not exist"):
        aggregate_metrics(tmp_path / "missing")
    with pytest.raises(ValueError, match="no metrics.json reports"):
        aggregate_metrics(tmp_path)


@pytest.mark.parametrize("filename, contents", [
    ("metrics.json", "{broken"),
    ("metrics.json", "[]"),
    ("run_spec.json", "{broken"),
])
def test_malformed_json_does_not_replace_existing_outputs(tmp_path, capsys, filename, contents):
    _write_report(tmp_path / "a", _mfa_report(), _spec())
    main([str(tmp_path)])
    previous = {p: p.read_bytes() for p in tmp_path.glob("*.csv")}
    _write_report(tmp_path / "z", _mfa_report())
    broken = tmp_path / "z" / filename
    broken.write_text(contents)

    with pytest.raises(SystemExit) as error:
        main([str(tmp_path)])

    assert error.value.code == 1
    assert str(broken) in capsys.readouterr().err
    assert {p: p.read_bytes() for p in previous} == previous


def test_cli_csv_output_and_rerun(tmp_path, capsys):
    _write_report(tmp_path / "mfa", _mfa_report(), _spec())
    _write_report(tmp_path / "kmeans", _kmeans_report())
    inputs = {p: p.read_bytes() for p in tmp_path.rglob("*.json")}

    main([str(tmp_path)])

    output = capsys.readouterr().out
    assert "2 reports into 2 general rows and 4 manifold rows" in output
    general = pd.read_csv(tmp_path / "general_metrics.csv")
    manifolds = pd.read_csv(tmp_path / "manifold_metrics.csv")
    assert len(general) == 2
    assert len(manifolds) == 4
    assert general.columns[0] == "source_path"
    assert not any(c.startswith("Unnamed:") for c in general)
    assert general.loc[general.model_kind == "mfa", "nll.validation"].empty
    assert general.loc[general.model_kind == "hddc", "nll.validation"].tolist() == [-10.0]
    assert json.loads(general.loc[general.model_kind == "hddc", "config.tags"].iloc[0]) == ["example", "sweep"]
    before = {p: p.read_bytes() for p in tmp_path.glob("*.csv")}
    main([str(tmp_path)])
    assert {p: p.read_bytes() for p in before} == before
    _write_report(tmp_path / "second_mfa", _mfa_report(), _spec(0.5))
    main([str(tmp_path)])
    assert len(pd.read_csv(tmp_path / "general_metrics.csv")) == 3
    assert len(pd.read_csv(tmp_path / "manifold_metrics.csv")) == 6
    assert {p: p.read_bytes() for p in inputs} == inputs


def test_cli_writes_readable_empty_manifold_table(tmp_path):
    _write_report(tmp_path, {"nll": {"validation": 2.0}})
    main([str(tmp_path)])
    manifolds = pd.read_csv(tmp_path / "manifold_metrics.csv")
    assert manifolds.empty
    assert {"evaluation_id", "manifold_id"} <= set(manifolds.columns)


def test_pipeline_kmeans_single_report_preserves_quantization_and_selected_ranks(tmp_path):
    report = {
        "schema_version": 2,
        "evaluation": "toy_manifold_tiling",
        "model_kind": "kmeans",
        "K": 20,
        "q_capacity": 4,
        "dataset": {"selected_rows": 100, "train_rows": 80, "validation_rows": 20},
        "quantization": {
            "train": {"sum_squared_distance": 40.0, "mean_squared_distance": 0.5, "n": 80},
            "validation": {"sum_squared_distance": 12.0, "mean_squared_distance": 0.6, "n": 20},
            "convention": "lower_is_better",
        },
        "rank": {"definition": "kmeans_component_ranks", "mean_learned": 2.5},
        "ambient_rank": {"definition": "kmeans_component_ranks", "mean_absolute_error": 0.5},
        "per_manifold": [
            {
                "manifold_id": 0, "type_name": "segment", "intrinsic_dim": 1,
                "rank": {"mean_learned": 1.0},
                "components": {"associated": 10},
            },
            {
                "manifold_id": 1, "type_name": "flat_disk", "intrinsic_dim": 2,
                "rank": {"mean_learned": 4.0},
                "components": {"associated": 10},
            },
        ],
    }
    spec = {
        "training": {
            "model_kind": "kmeans",
            "arguments": {"K": 20, "rank": 4, "surgery_threshold": 0.1, "pca_purpose": "geometry"},
        },
    }
    report["pca_geometry"] = {
        "version": 1, "minimum_cluster_points": 5,
        "eligible_components": 12, "excluded_components": 8,
        "eligible_associated_components": 12, "excluded_associated_components": 8,
    }
    _write_report(tmp_path / "kmeans", report, spec)
    mfa = _mfa_report()
    mfa["model_kind"] = "mfa"
    mfa["bic"] = {"value": 105.0, "convention": "higher_is_better"}
    _write_report(tmp_path / "mfa", mfa)

    general, manifolds = aggregate_metrics(tmp_path)

    assert len(general) == 2
    assert len(manifolds) == 4
    kmeans = general.loc[general.model_kind == "kmeans"]
    assert len(kmeans) == 1
    row = kmeans.iloc[0]
    assert row["evaluation_id"] == "kmeans/metrics.json"
    assert row["rank.definition"] == "kmeans_component_ranks"
    assert row["rank.mean_learned"] == 2.5
    assert row["ambient_rank.definition"] == "kmeans_component_ranks"
    assert row["q_capacity"] == 4
    assert row["config.training.arguments.rank"] == 4
    assert row["config.training.arguments.surgery_threshold"] == 0.1
    assert row["config.training.arguments.pca_purpose"] == "geometry"
    assert row["pca_geometry.eligible_components"] == 12
    assert row["pca_geometry.excluded_components"] == 8
    assert row["quantization.convention"] == "lower_is_better"
    for split, values in report["quantization"].items():
        if split != "convention":
            for metric, value in values.items():
                assert row[f"quantization.{split}.{metric}"] == value
    assert kmeans[["nll.train", "nll.validation", "bic.value"]].isna().all().all()
    assert general.loc[general.model_kind == "mfa", "quantization.train.mean_squared_distance"].isna().all()
    children = manifolds.loc[manifolds.model_kind == "kmeans"]
    assert children["rank.mean_learned"].tolist() == [1.0, 4.0]
    assert children["config.training.arguments.surgery_threshold"].tolist() == [0.1, 0.1]
    assert set(children.evaluation_id) == {row["evaluation_id"]}
    assert not any(column.startswith("quantization.") for column in manifolds)


def test_run_id_from_spec_and_columns_are_deterministic(tmp_path):
    _write_report(tmp_path, {"z": 1, "a": 2}, {"run_id": "from-spec"})
    general, _ = aggregate_metrics(tmp_path)
    assert general.loc[0, "run_id"] == "from-spec"
    assert general.loc[0, "evaluation_id"] == "metrics.json"
    assert general.columns[3:].tolist() == sorted(general.columns[3:])


def test_adjusted_alignment_exports_global_and_manifold_summaries(tmp_path):
    report = _mfa_report()
    score = {"mean": 0.4, "valid_components": 80, "undefined_components": 20}
    report["tangent_adjusted_alignment"] = {
        "normalization": "intrinsic_dim", "subspace_overlap": score,
    }
    report["per_manifold"][0]["tangent_adjusted_alignment"] = {
        "subspace_overlap": {"mean": None, "valid_components": 0, "undefined_components": 0},
    }
    report["per_manifold"][1]["tangent_adjusted_alignment"] = {"subspace_overlap": score}
    _write_report(tmp_path / "run", report, _spec())
    main([str(tmp_path)])
    general = pd.read_csv(tmp_path / "general_metrics.csv")
    manifolds = pd.read_csv(tmp_path / "manifold_metrics.csv").set_index("manifold_id")
    prefix = "tangent_adjusted_alignment.subspace_overlap"
    for field, value in score.items():
        assert general.loc[0, f"{prefix}.{field}"] == value
        assert manifolds.loc[0, f"{prefix}.{field}"] == value
    assert pd.isna(manifolds.loc[1, f"{prefix}.mean"])
    assert manifolds.loc[1, f"{prefix}.valid_components"] == 0
    assert manifolds.loc[1, f"{prefix}.undefined_components"] == 0
