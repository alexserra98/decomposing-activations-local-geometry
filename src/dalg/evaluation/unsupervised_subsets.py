"""Prompt-subset populations and validation for unsupervised manifold reports."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from contextlib import nullcontext

import torch
from torch.utils.data import DataLoader

from dalg.analysis.reconstruction import ReconstructionMoments, reconstruction_errors
from dalg.data.shard_activations import ActivationBatchDataset
from dalg.evaluation.common import EvaluationData
from dalg.evaluation.coverage import coverage_report_valid
from dalg.evaluation.pc1_density import pc1_density_report_valid
from dalg.evaluation.reconstruction import reconstruction_metadata, reconstruction_report_valid
from dalg.evaluation.tile_alignment import tile_alignment_report_valid


def subset_populations(data: EvaluationData) -> tuple[list[tuple[str, EvaluationData]], dict]:
    """Slice canonical token assignments and label components using training only."""
    names = sorted({data.meta_index[p]["subset"] for p in data.positions})
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("manifold subsets must be nonempty strings")
    train = set(data.train_positions)
    validation = set(data.val_positions)
    assignments = data.assignments.reshape(len(data.positions), data.tokens_per_row)
    groups = []
    for name in names:
        rows = [i for i, p in enumerate(data.positions) if data.meta_index[p]["subset"] == name]
        positions = [data.positions[i] for i in rows]
        selected = assignments[rows]
        training_rows = torch.tensor([p in train for p in positions], dtype=torch.bool)
        groups.append((name, replace(
            data, positions=positions,
            train_positions=[p for p in positions if p in train],
            val_positions=[p for p in positions if p in validation],
            assignments=selected.reshape(-1),
            cluster_sizes=torch.bincount(selected.reshape(-1), minlength=len(data.cluster_sizes)),
            train_cluster_sizes=torch.bincount(selected[training_rows].reshape(-1),
                                               minlength=len(data.cluster_sizes)),
        )))
    counts = torch.stack([group.train_cluster_sizes for _, group in groups])
    totals = counts.sum(0)
    maxima, winners = counts.max(0)
    tied = (counts == maxima).sum(0) > 1
    records = []
    for component in range(len(totals)):
        total = int(totals[component])
        reason = "no_training_assignments" if total == 0 else "tied_training_counts" if tied[component] else None
        records.append({
            "component_id": component,
            "manifold_id": int(winners[component]) if reason is None else None,
            "train_points": total, "purity": int(maxima[component]) / total if total else None,
            "train_counts": counts[:, component].tolist(), "exclusion_reason": reason,
        })
    return groups, {"version": 1, "label_field": "subset",
                    "association": "unique_training_majority", "manifold_names": names,
                    "per_component": records}


@torch.no_grad()
def subset_reconstruction(model, data: EvaluationData, *, model_kind: str, layer: int,
                          batch_size: int, device: torch.device) -> dict:
    """Score the full fitted model on each split, including empty subgroup splits."""
    report = reconstruction_metadata(model_kind, model.D)
    for name, positions in (("train", data.train_positions), ("validation", data.val_positions)):
        moments = ReconstructionMoments()
        if positions:
            dataset = ActivationBatchDataset(
                data.shard_dir, layer=layer, row_subset=positions, batch_size=batch_size,
                drop_prefix=data.drop_prefix, dtype=torch.float32,
                shuffle_shards=False, shuffle_within_shard=False,
            )
            with model.inference_cache() if model_kind == "hddc" else nullcontext():
                for batch in DataLoader(dataset, batch_size=None, num_workers=0):
                    moments.update(*reconstruction_errors(model, batch.to(device), model_kind=model_kind))
        report[name] = moments.summary()
    return report


def subset_reports_valid(metrics: Mapping, cfg: Mapping, *, components: int, model_kind: str) -> bool:
    """Validate the partition, training-majority associations, and metric populations."""
    stratification, groups = metrics.get("stratification"), metrics.get("per_manifold")
    if not isinstance(stratification, Mapping) or not isinstance(groups, list) or not groups:
        return False
    names = stratification.get("manifold_names")
    if (stratification.get("version") != 1 or stratification.get("label_field") != "subset"
            or stratification.get("association") != "unique_training_majority"
            or not isinstance(names, list) or any(not isinstance(n, str) or not n for n in names)
            or names != sorted(set(names)) or len(names) != len(groups)):
        return False
    records = stratification.get("per_component")
    if not isinstance(records, list) or len(records) != components:
        return False
    group_totals = [0] * len(groups)
    associated = [[] for _ in groups]
    for index, record in enumerate(records):
        if not isinstance(record, Mapping) or record.get("component_id") != index:
            return False
        counts = record.get("train_counts")
        if (not isinstance(counts, list) or len(counts) != len(groups)
                or any(type(c) is not int or c < 0 for c in counts)):
            return False
        total, maximum = sum(counts), max(counts)
        reason = "no_training_assignments" if total == 0 else "tied_training_counts" if counts.count(maximum) > 1 else None
        winner = counts.index(maximum) if reason is None else None
        if (record.get("train_points") != total or record.get("exclusion_reason") != reason
                or record.get("manifold_id") != winner
                or record.get("purity") != (maximum / total if total else None)):
            return False
        if winner is not None:
            associated[winner].append(index)
        group_totals = [a + b for a, b in zip(group_totals, counts)]
    parent = metrics["dataset"]
    count_fields = [f"{split}_{unit}" for split in ("selected", "train", "validation") for unit in ("rows", "points")]
    sums = dict.fromkeys(count_fields, 0)
    test_points = 0
    for index, group in enumerate(groups):
        if (not isinstance(group, Mapping) or group.get("manifold_id") != index
                or group.get("subset") != names[index] or group.get("component_ids") != associated[index]):
            return False
        dataset = group.get("dataset")
        if not isinstance(dataset, Mapping):
            return False
        for key in ("shard_dir", "subset_spec", "layer", "d_model", "window", "drop_prefix"):
            if dataset.get(key) != parent.get(key):
                return False
        for key in count_fields:
            value = dataset.get(key)
            if type(value) is not int or value < 0:
                return False
            sums[key] += value
        if (dataset["train_points"] != group_totals[index]
                or dataset["selected_rows"] <= 0
                or dataset["selected_rows"] != dataset["train_rows"] + dataset["validation_rows"]
                or any(dataset[f"{s}_points"] != dataset[f"{s}_rows"] * (dataset["window"] - dataset["drop_prefix"])
                       for s in ("selected", "train", "validation"))):
            return False
        if not reconstruction_report_valid(group.get("reconstruction"), model_kind=model_kind,
                                           dataset=dataset, allow_empty_splits=True):
            return False
        if cfg.get("tile_alignment", True):
            alignment = group.get("tile_alignment")
            if not tile_alignment_report_valid(alignment, components=components, neighbors=cfg.get("tile_alignment_neighbors", 5)):
                return False
            if any(tile["component_id"] not in associated[index] for tile in alignment["per_tile"]):
                return False
        elif "tile_alignment" in group:
            return False
        if cfg.get("pc1_density", False):
            density = group.get("pc1_density")
            if not pc1_density_report_valid(density, components=components, cfg=cfg, dataset=dataset):
                return False
            if any(r["component_id"] not in associated[index] for r in density["per_component"]):
                return False
            outside = {r["component_id"] for r in density["exclusions"] if r["reason"] == "outside_manifold"}
            if outside != set(range(components)) - set(associated[index]):
                return False
        elif "pc1_density" in group:
            return False
        if cfg.get("heldout_distribution_coverage", True):
            coverage = group.get("heldout_distribution_coverage")
            if not coverage_report_valid(coverage, components=components):
                return False
            source = coverage.get("source")
            pooled = metrics["heldout_distribution_coverage"]
            if (not isinstance(source, Mapping) or source.get("manifold_subset") != names[index]
                    or coverage["components_live"] != pooled["components_live"]
                    or any(source.get(k) != pooled["source"].get(k) for k in
                           ("shard_dir", "subset_spec", "layer", "window", "drop_prefix", "d_model"))
                    or type(source.get("num_rows")) is not int or source["num_rows"] <= 0
                    or source.get("num_points") != coverage["reference_points"]
                    or source["num_rows"] * (source["window"] - source["drop_prefix"]) != source["num_points"]):
                return False
            test_points += coverage["reference_points"]
        elif "heldout_distribution_coverage" in group:
            return False
    return (all(sums[k] == parent[k] for k in count_fields)
            and (not cfg.get("heldout_distribution_coverage", True)
                 or test_points == metrics["heldout_distribution_coverage"]["reference_points"]))
