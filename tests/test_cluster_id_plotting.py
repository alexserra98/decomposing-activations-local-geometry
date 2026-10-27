"""Regression checks for population and averaging choices in the ID figures."""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts/temporary"))
from plot_cluster_id_experiment import GRIDE_RANKS, macro_summary, select_runs, summarize_split


def cluster(index, size, value=None):
    status = "ok" if value is not None else (
        "below_min_population" if size < 500 else "zero_neighbour_distance")
    return {"cluster_id": index, "n_points": size,
            "two_nn_status": status, "two_nn_id": value, "gride_status": status,
            "gride_ranks": list(GRIDE_RANKS) if value is not None else [],
            "gride_ids": [value + i for i in range(6)] if value is not None else []}


def manifold(index, count):
    return {"manifold_id": index, "type_name": f"manifold_{index}",
            "intrinsic_dim": 1, "embedding_dim": 2, "components": {"associated": count}}


def test_equal_cluster_then_equal_manifold_means_keep_gride_scales_separate():
    rows = [cluster(0, 500, 2), cluster(1, 1500, 6), cluster(2, 500, 10),
            cluster(3, 499), cluster(4, 700)]
    result = summarize_split(rows, {0: 0, 1: 0, 2: 1, 4: 1},
                             [manifold(0, 2), manifold(1, 3)])
    assert result[0]["two_nn_mean"] == 4  # Not weighted by point population.
    assert result[1]["two_nn_mean"] == 10
    assert result[1]["eligible_clusters"] == 2
    assert result[1]["two_nn_n"] == 1
    assert [result[0][f"gride_{rank}_mean"] for rank in GRIDE_RANKS] == [4, 5, 6, 7, 8, 9]
    frame = pd.DataFrame([{**row, "relative_run": "one", "split": "train", "K": 5,
                           "model": "kmeans", "noise_ratio": "noiseless"} for row in result])
    macro = macro_summary(frame).set_index("estimator")
    assert macro.loc["two_nn", "mean_id"] == 7  # Not the pooled-cluster mean of 6.
    assert macro.loc["two_nn", "valid_clusters"] == 3
    assert macro.loc["two_nn", "valid_manifolds"] == 2
    assert macro.loc["gride_64", "mean_id"] == 12


def test_empty_or_invalid_manifolds_remain_undefined_and_counted():
    result = summarize_split([cluster(0, 500), cluster(1, 499)], {0: 0},
                             [manifold(0, 1), manifold(1, 1)])
    assert [row["two_nn_mean"] for row in result] == [None, None]
    assert [row["two_nn_n"] for row in result] == [0, 0]
    assert [row["eligible_clusters"] for row in result] == [1, 0]
    frame = pd.DataFrame([{**row, "relative_run": "one", "split": "test", "K": 2,
                           "model": "kmeans", "noise_ratio": 10} for row in result])
    summary = macro_summary(frame)
    assert summary["mean_id"].isna().all()
    assert summary["valid_manifolds"].eq(0).all()
    assert summary["total_manifolds"].eq(2).all()


def test_gride_scale_mismatch_is_rejected():
    row = cluster(0, 500, 2)
    row["gride_ranks"] = [2, 4, 8, 16, 32, 128]
    with pytest.raises(AssertionError):
        summarize_split([row], {0: 0}, [manifold(0, 1)])


def test_em_variants_are_selected_separately(tmp_path):
    runs = []
    for shared in (True, False):
        directory = tmp_path / str(shared)
        directory.mkdir()
        (directory / "config.json").write_text(json.dumps({
            "fit_method": "em", "shared_b": shared, "surgery_threshold": 0.05,
        }))
        runs.append({"K": 2000, "condition": "noiseless", "model_kind": "hddc",
                     "run_dir": str(directory)})
    selection = select_runs({"runs": runs}, K=2000, thresholds={"em": 0.05, "em_bk": 0.05},
                            noise_levels=["noiseless"], models=["em", "em_bk"])
    assert [item["model"] for item in selection] == ["em", "em_bk"]


@pytest.mark.parametrize("minimum, expected, count", [(65, 6, 5), (200, 8, 3), (500, 10, 1)])
def test_manifold_means_include_newly_eligible_clusters(minimum, expected, count):
    rows = [cluster(0, 64), cluster(1, 65, 2), cluster(2, 199, 4),
            cluster(3, 200, 6), cluster(4, 499, 8), cluster(5, 500, 10)]
    result = summarize_split(rows, {index: 0 for index in range(1, 6)}, [manifold(0, 6)],
                             min_population=minimum)
    assert result[0]["two_nn_mean"] == expected
    assert result[0]["two_nn_n"] == count
    assert result[0]["gride_64_n"] == count
    assert result[0]["gride_64_mean"] == expected + 5
