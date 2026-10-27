"""ID estimates preserve the saved training split and native test memberships."""

import importlib.util
import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from dadapy import Data

from dalg.evaluation.toy_manifold_tiling import _load_model
from tests.test_kmeans_evaluation import _build_kmeans_run
from tests.test_toy_manifold_tiling import _build_evaluation_artifacts


spec = importlib.util.spec_from_file_location(
    "cluster_id_experiment",
    Path(__file__).parents[1] / "scripts/temporary/run_cluster_id_experiment.py",
)
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)


def test_ids_match_direct_dadapy_and_include_every_gride_scale():
    points = torch.randn(600, 4, generator=torch.Generator().manual_seed(91))
    result = experiment.estimate_cluster(points)
    reference = Data(coordinates=points.double().numpy(), n_jobs=1, maxk=64)
    two_nn = reference.compute_id_2NN(algorithm="base", mu_fraction=0.9, data_fraction=1)
    ids, errors, scales = reference.return_id_scaling_gride(range_max=64)
    assert result["two_nn_status"] == result["gride_status"] == "ok"
    np.testing.assert_allclose(
        [result["two_nn_id"], result["two_nn_error"], result["two_nn_scale"]], two_nn,
    )
    for key, expected in (("gride_ids", ids), ("gride_errors", errors), ("gride_scales", scales)):
        np.testing.assert_allclose(result[key], expected)
    assert result["gride_ranks"] == [2, 4, 8, 16, 32, 64]
    assert result["n_points"] == len(points)


@pytest.mark.parametrize("minimum", [65, 200, 500])
def test_population_threshold_is_inclusive_and_keeps_empty_clusters(minimum):
    points = torch.randn(2 * minimum - 1, 3, generator=torch.Generator().manual_seed(1))
    labels = torch.cat((torch.zeros(minimum - 1, dtype=torch.long), torch.full((minimum,), 2)))
    rows = experiment.estimate_split(points, labels, 3, "train", min_population=minimum)
    assert [row["cluster_id"] for row in rows] == [0, 1, 2]
    assert [row["n_points"] for row in rows] == [minimum - 1, 0, minimum]
    assert rows[0]["two_nn_status"] == rows[1]["gride_status"] == "below_min_population"
    assert rows[2]["two_nn_status"] == rows[2]["gride_status"] == "ok"
    assert rows[2]["gride_ranks"] == [2, 4, 8, 16, 32, 64]


def test_degenerate_cluster_preserves_population_and_records_reason():
    result = experiment.estimate_cluster(torch.ones(500, 3))
    assert result["n_points"] == 500
    assert result["two_nn_status"] == result["gride_status"] == "zero_neighbour_distance"
    assert result["two_nn_id"] is None
    assert result["gride_ids"] == []


def test_training_membership_uses_saved_global_rows_not_row_positions():
    rows = torch.tensor([80, 20, 60, 40])
    split = {"val_global_rows": [20, 40], "train_rows": 2, "val_rows": 2}
    assert experiment.training_mask(rows, split).tolist() == [True, False, True, False]
    with pytest.raises(ValueError, match="validation row IDs"):
        experiment.training_mask(rows, {**split, "val_global_rows": [1, 2]})


@pytest.mark.parametrize("kind", ["kmeans", "mfa", "hddc"])
@pytest.mark.parametrize("minimum, test_minimum", [(200, None), (500, None), (200, 65)])
def test_saved_run_separates_train_and_test_and_preserves_inputs(tmp_path, kind, minimum, test_minimum):
    if kind == "kmeans":
        run, root, *_ = _build_kmeans_run(tmp_path, "cluster", False)
    else:
        run, root = _build_evaluation_artifacts(tmp_path, kind)
    model = _load_model(run, kind)
    config = experiment.read_json(run / "config.json")
    config.update(model_kind=kind, K=model.K, shard_dir=str(root), layer=0)
    (run / "config.json").write_text(json.dumps(config))
    stem = "kmeans_model" if kind == "kmeans" else "mfa_model"
    path = run / f"{stem}_assignments.pt"
    bundle = torch.load(path, weights_only=True)
    split = experiment.read_json(run / "val_indices.json")
    bundle["assignments"].zero_()
    bundle["assignments"][split["val_global_rows"]] = 1
    bundle["cluster_sizes"] = torch.bincount(bundle["assignments"], minlength=model.K)
    torch.save(bundle, path)
    before = {p: p.read_bytes() for p in run.iterdir() if p.is_file()}
    record = experiment.describe_run(run, tmp_path)
    output = tmp_path / "id_results"
    report = experiment.run_one(record, tmp_path, output, batch_size=7, min_population=minimum,
                                test_min_population=test_minimum)
    assert report["settings"] == experiment.settings_for_population(minimum, test_minimum)
    train = [row for row in report["clusters"] if row["split"] == "train"]
    test = [row for row in report["clusters"] if row["split"] == "test"]
    assert len(train) == len(test) == model.K
    assert train[0]["n_points"] == split["train_rows"]
    assert all(row["n_points"] == 0 for row in train[1:])
    assert report["populations"]["validation_excluded"] == split["val_rows"]
    test_points, _ = experiment.load_points(root / "test", 0, 7)
    with torch.no_grad():
        expected = (model.predict(test_points) if kind == "kmeans"
                    else model.responsibilities(test_points).argmax(1))
    saved = torch.load(output / "test_assignments.pt", weights_only=True)
    torch.testing.assert_close(saved["assignments"], expected)
    assert [row["n_points"] for row in test] == torch.bincount(expected, minlength=model.K).tolist()
    assert all(row["two_nn_status"] == "below_min_population" for row in report["clusters"])
    assert experiment.read_json(output / "COMPLETED.json")["results_sha256"] == experiment.sha256(output / "results.json")
    assert before == {p: p.read_bytes() for p in run.iterdir() if p.is_file()}
    with pytest.raises(FileExistsError):
        experiment.run_one(record, tmp_path, output, batch_size=7)


def test_recorded_home_alias_is_resolved_only_through_selected_cache(tmp_path):
    value = experiment.HOME_CACHE + "assets/example/noiseless"
    assert experiment.source_path(value, tmp_path) == tmp_path / "assets/example/noiseless"


@pytest.mark.parametrize("test_minimum", [None, 65])
def test_aggregation_uses_the_manifest_population_threshold(tmp_path, test_minimum):
    run = {"condition": "noiseless", "family": "example", "relative_run": "example",
           "model_kind": "kmeans", "K": 5}
    rows = []
    for split in ["train", "test"]:
        minimum = test_minimum if split == "test" and test_minimum is not None else 200
        for index, size in enumerate([64, 65, 199, 200, 500]):
            status = "ok" if size >= minimum else "below_min_population"
            rows.append({"cluster_id": index, "split": split, "n_points": size,
                         "two_nn_status": status, "gride_status": status})
    plan = {"runs": [run], "settings": experiment.settings_for_population(200, test_minimum)}
    directory = tmp_path / "runs/example"
    directory.mkdir(parents=True)
    experiment.write_json(directory / "results.json", {
        "run": run, "settings": plan["settings"], "clusters": rows,
    })
    experiment.write_json(directory / "COMPLETED.json", {
        "results_sha256": experiment.sha256(directory / "results.json"),
    })
    experiment.aggregate(plan, tmp_path)
    with (tmp_path / "summary/runs.csv").open() as handle:
        summary = list(csv.DictReader(handle))
    expected = [2, 4 if test_minimum == 65 else 2]
    assert [int(row["eligible_clusters"]) for row in summary] == expected
    assert [int(row["two_nn_valid"]) for row in summary] == expected
    assert [int(row["gride_valid"]) for row in summary] == expected
    assert experiment.SETTINGS["min_population"] == 500
