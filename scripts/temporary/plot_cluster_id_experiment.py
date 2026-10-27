"""Join saved cluster IDs to the original toy-manifold geometry for plotting.

Used by notebooks/toy_manifold_noise_sweep_intrinsic_dimension.ipynb. Run its
preparation and rendering inside Slurm; no ID estimator is rerun here.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import torch

from dalg.evaluation.toy_manifold_metrics import _associate_component_means
from dalg.evaluation.toy_manifold_tiling import _load_model
from run_cluster_id_experiment import (
    file_stat, read_json, select_runs, settings_for_population, sha256,
)


GRIDE_RANKS = (2, 4, 8, 16, 32, 64)
ESTIMATORS = {"two_nn": "2NN", **{
    f"gride_{rank}": f"GRIDE ({rank // 2}, {rank})" for rank in GRIDE_RANKS
}}
MODEL_NAMES = {
    "kmeans": "K-means + PCA", "mfa": "Vanilla MFA", "hddc": "HDDC Adam",
    "em": "HDDC EM (shared b)", "em_bk": "HDDC EM (component b)",
}


def summarize_split(rows, associations, manifolds, *, min_population=500):
    """Average valid IDs equally within each manifold, separately for one split."""
    result = []
    for manifold in manifolds:
        eligible = [row for row in rows if row["n_points"] >= min_population
                    and associations[row["cluster_id"]] == manifold["manifold_id"]]
        record = {key: manifold[key] for key in
                  ("manifold_id", "type_name", "intrinsic_dim", "embedding_dim")}
        record.update(associated_clusters=manifold["components"]["associated"],
                      eligible_clusters=len(eligible))
        assert record["eligible_clusters"] <= record["associated_clusters"]
        values = {name: [] for name in ESTIMATORS}
        for row in eligible:
            if row["two_nn_status"] == "ok":
                values["two_nn"].append(row["two_nn_id"])
            if row["gride_status"] == "ok":
                assert row["gride_ranks"] == list(GRIDE_RANKS)
                assert len(row["gride_ids"]) == len(GRIDE_RANKS)
                for rank, value in zip(GRIDE_RANKS, row["gride_ids"]):
                    values[f"gride_{rank}"].append(value)
        for name, estimates in values.items():
            assert all(math.isfinite(value) and value > 0 for value in estimates)
            record[f"{name}_n"] = len(estimates)
            record[f"{name}_mean"] = (
                math.fsum(estimates) / len(estimates) if estimates else None
            )
        result.append(record)
    return result


def build_manifold_tables(plan, selected, id_root, output):
    """Associate eligible component means with the same noiseless manifolds.

    Only the union of clusters eligible in either split needs a new geometric
    projection. Original per-manifold metrics supply the all-component counts.
    Source and result fingerprints are recorded with the derived tables.
    """
    min_population = plan["settings"]["min_population"]
    assert plan["settings"] == settings_for_population(min_population)
    id_root, output = Path(id_root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    metadata_cache, provenance, tables, membership_rows = {}, [], [], []
    for item in selected:
        run = item["run"]
        directory = id_root / "runs" / run["relative_run"]
        results_path = directory / "results.json"
        report = read_json(results_path)
        digest = sha256(results_path)
        assert digest == read_json(directory / "COMPLETED.json")["results_sha256"]
        assert report["run"] == run and report["settings"] == plan["settings"]
        for path in (run["model_path"], str(Path(run["run_dir"]) / "config.json")):
            assert file_stat(path) == run["input_stats"][path], f"Changed input: {path}"
            assert sha256(path) == report["input_sha256"][path], f"Changed input: {path}"
        metrics_path = Path(run["run_dir"]) / "metrics.json"
        metrics = read_json(metrics_path)
        assert metrics["K"] == run["K"] and metrics["run_id"] == Path(run["run_dir"]).name
        assert metrics["association"]["rule"] == "unique_nearest_exact_projection"
        cutoff = metrics["association"]["max_mean_to_manifold_distance"]
        root = Path(run["shard_dir"])
        if root not in metadata_cache:
            config = read_json(root / "config.json")
            metadata_path = root / config["manifold_metadata"]
            metadata_cache[root] = (torch.load(metadata_path, map_location="cpu", weights_only=True),
                                    str(metadata_path), sha256(metadata_path))
        metadata, metadata_path, metadata_digest = metadata_cache[root]
        manifolds = metrics["per_manifold"]
        assert len({m["type_name"] for m in manifolds}) == len(manifolds) == 10
        assert {(m["manifold_id"], m["type_name"], m["intrinsic_dim"], m["embedding_dim"])
                for m in manifolds} == {
                    (m["manifold_id"], m["type_name"], m["intrinsic_dim"], m["embedding_dim"])
                    for m in metadata["manifolds"]}
        split_rows = {split: [row for row in report["clusters"] if row["split"] == split]
                      for split in ("train", "test")}
        for split, rows in split_rows.items():
            assert [row["cluster_id"] for row in rows] == list(range(run["K"]))
            assert sum(row["n_points"] for row in rows) == report["populations"][split]
            assert all(row["n_points"] >= min_population or
                       row["two_nn_status"] == row["gride_status"] == "below_min_population"
                       for row in rows)
        eligible = sorted({row["cluster_id"] for row in report["clusters"]
                           if row["n_points"] >= min_population})
        print(f"{run['condition']} {item['model']} K={run['K']}: "
              f"associate {len(eligible)} eligible component means", flush=True)
        associations = {}
        if eligible:
            model = _load_model(Path(run["run_dir"]), run["model_kind"])
            assert model.K == run["K"]
            projection = _associate_component_means(
                model.mu[eligible], metadata, max_mean_to_manifold_distance=cutoff,
            )
            for cluster, index, distance, ambiguous in zip(
                eligible, projection.manifold_indices.tolist(),
                projection.nearest_distances.tolist(), projection.ambiguous.tolist(),
            ):
                manifold_id = metadata["manifolds"][index]["manifold_id"] if index >= 0 else None
                associations[cluster] = manifold_id
                membership_rows.append({"relative_run": run["relative_run"], "cluster_id": cluster,
                    "manifold_id": manifold_id, "nearest_distance": distance, "ambiguous": ambiguous})
            del model, projection
        identity = {"relative_run": run["relative_run"], "K": run["K"],
                    "condition": run["condition"], "model": item["model"],
                    "noise_ratio": item["noise_ratio"], "surgery_threshold": item["surgery_threshold"],
                    "min_population": min_population}
        for split, rows in split_rows.items():
            tables.extend({**identity, "split": split, **row}
                          for row in summarize_split(rows, associations, manifolds,
                                                     min_population=min_population))
        provenance.append({**identity, "results_sha256": digest,
            "model_sha256": report["input_sha256"][run["model_path"]],
            "metrics_path": str(metrics_path), "metrics_sha256": sha256(metrics_path),
            "metadata_path": metadata_path, "metadata_sha256": metadata_digest,
            "max_mean_to_manifold_distance": cutoff,
            "eligible_union": len(eligible),
            "unassociated_eligible_union": sum(value is None for value in associations.values())})
    frame = pd.DataFrame(tables)
    assert not frame.duplicated(["relative_run", "split", "manifold_id"]).any()
    frame.to_csv(output / "manifold_ids.csv", index=False)
    pd.DataFrame(membership_rows).to_csv(output / "eligible_component_manifolds.csv", index=False)
    (output / "sources.json").write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n")
    return frame


def macro_summary(frame):
    """Equal-manifold means; retain how many manifolds and clusters contribute."""
    summaries = []
    for (run, split), rows in frame.groupby(["relative_run", "split"], sort=False):
        identity = {key: rows.iloc[0][key] for key in ("model", "noise_ratio", "K")}
        for name, label in ESTIMATORS.items():
            values = rows[f"{name}_mean"].dropna()
            summaries.append({**identity, "relative_run": run, "split": split,
                "estimator": name, "label": label,
                "mean_id": values.mean() if len(values) else float("nan"),
                "valid_manifolds": len(values), "total_manifolds": len(rows),
                "valid_clusters": int(rows[f"{name}_n"].sum()),
                "associated_clusters": int(rows["associated_clusters"].sum())})
    return pd.DataFrame(summaries)
