"""Evaluate saved models without ground-truth labels or manifold geometry."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from dalg.data.shard_activations import ActivationBatchDataset, load_meta_index
from dalg.data.subset_spec import resolve_spec_positions, split_shard_dir_spec
from dalg.evaluation.common import evaluate_fit, load_evaluation_data, load_model, resolve_device
from dalg.evaluation.coverage import (
    check_empirical_coverage_size, coverage_report_valid, evaluate_stream_coverage,
)
from dalg.evaluation.subspaces import effective_component_ranks, leading_covariance_eigenspaces
from dalg.evaluation.unsupervised_subsets import (
    subset_populations, subset_reconstruction, subset_reports_valid,
)
from dalg.evaluation.reconstruction import reconstruction_report_valid
from dalg.evaluation.pc1_density import (
    evaluate_pc1_density, pc1_density_report_valid, validate_pc1_density_settings,
)
from dalg.evaluation.tile_alignment import (
    evaluate_tile_alignment, tile_alignment_report_valid, validate_tile_alignment_settings,
)


def validate_test_shards(
    shard_dir: str | Path, test_shard_dir: str | Path, *, layer: int,
    model_kind: str, drop_prefix: int | None = None,
) -> dict:
    """Check an explicitly supplied test population without loading activations."""
    root, _ = split_shard_dir_spec(str(shard_dir))
    test_root, subset = split_shard_dir_spec(str(test_shard_dir))
    if root.resolve() == test_root.resolve():
        raise ValueError("test_shard_dir must be separate from the training shard directory")
    parent = json.loads((root / "config.json").read_text())
    config = json.loads((test_root / "config.json").read_text())
    if drop_prefix is None:
        drop_prefix = int(parent.get("drop_prefix", 0 if model_kind == "kmeans" else 32))
    window = int(parent["window"])
    if not 0 <= drop_prefix < window:
        raise ValueError("coverage drop_prefix must leave at least one token per row")
    if (config.get("d_model") != parent["d_model"] or config.get("window") != window
            or int(config.get("drop_prefix", drop_prefix)) != drop_prefix
            or layer not in config.get("layers", [])):
        raise ValueError("test shards must match the training dimension, window, prefix, and layer")
    meta = load_meta_index(test_root, layer=layer)
    positions = resolve_spec_positions(meta, subset, window=window, drop_prefix=drop_prefix)
    count = len(positions) * (window - drop_prefix)
    if count == 0:
        raise ValueError("coverage requires a nonempty test stream")
    check_empirical_coverage_size(count)
    return {
        "shard_dir": str(test_root.resolve()), "subset_spec": subset, "layer": layer,
        "num_rows": len(positions), "num_points": count, "window": window,
        "drop_prefix": drop_prefix, "d_model": int(config["d_model"]),
    }


def _test_coverage(source: dict, centroids: torch.Tensor, train_live: torch.Tensor, *, batch_size: int) -> dict:
    meta = load_meta_index(source["shard_dir"], layer=source["layer"])
    positions = resolve_spec_positions(
        meta, source["subset_spec"], window=source["window"], drop_prefix=source["drop_prefix"],
    )
    if "manifold_subset" in source:
        positions = [p for p in positions if meta[p]["subset"] == source["manifold_subset"]]
    dataset = ActivationBatchDataset(
        source["shard_dir"], layer=source["layer"], row_subset=positions,
        batch_size=batch_size, drop_prefix=source["drop_prefix"], dtype=torch.float32,
        shuffle_shards=False, shuffle_within_shard=False,
    )
    summary = evaluate_stream_coverage(
        DataLoader(dataset, batch_size=None, num_workers=0), centroids, train_live,
        expected_points=source["num_points"], batch_size=batch_size,
    )
    summary["source"] = source
    return summary


@torch.no_grad()
def evaluate_unsupervised(
    run_dir: str | Path, *, shard_dir: str | Path, layer: int, model_kind: str,
    assignments_path: str | Path | None = None, test_shard_dir: str | Path | None = None,
    batch_size: int = 4096, device: str = "cuda", rank_threshold: float = 1.0,
    heldout_distribution_coverage: bool = True,
    tile_alignment: bool = True,
    tile_alignment_neighbors: int = 5,
    pc1_density: bool = False, pc1_density_space: str = "pca75",
    pc1_density_neighbors: int = 20, pc1_density_random_directions: int = 32,
    pc1_density_seed: int = 42,
) -> dict[str, Any]:
    """Compute fit, component, rank, coverage, and optional geometry summaries.

    Tile alignment computes covariance PCs when enabled but saves only scores
    and neighbor comparisons, not a PC artifact.
    """
    validate_pc1_density_settings({
        "pc1_density": pc1_density, "pc1_density_space": pc1_density_space,
        "pc1_density_neighbors": pc1_density_neighbors,
        "pc1_density_random_directions": pc1_density_random_directions,
        "pc1_density_seed": pc1_density_seed,
    })
    if pc1_density and model_kind not in {"kmeans", "hddc"}:
        raise ValueError("PC1 density supports only KMeans and HDDC")
    validate_tile_alignment_settings(tile_alignment, tile_alignment_neighbors)
    if type(heldout_distribution_coverage) is not bool:
        raise ValueError("heldout_distribution_coverage must be true or false")
    if batch_size <= 0:
        raise ValueError("evaluation batch_size must be positive")
    if model_kind not in {"hddc", "kmeans"} and (not math.isfinite(rank_threshold) or rank_threshold <= 0):
        raise ValueError("rank_threshold must be finite and positive")
    if heldout_distribution_coverage and test_shard_dir is None:
        raise ValueError("unsupervised coverage requires evaluation.test_shard_dir")
    model = load_model(Path(run_dir), model_kind)
    data = load_evaluation_data(
        run_dir, shard_dir=shard_dir, layer=layer, model=model,
        model_kind=model_kind, assignments_path=assignments_path,
    )
    source = validate_test_shards(
        shard_dir, test_shard_dir, layer=layer, model_kind=model_kind, drop_prefix=data.drop_prefix,
    ) if heldout_distribution_coverage else None
    if model_kind in {"hddc", "kmeans"}:
        groups, association = subset_populations(data)
        test_counts = {}
        if source is not None:
            meta = load_meta_index(source["shard_dir"], layer=layer)
            positions = resolve_spec_positions(meta, source["subset_spec"],
                                               window=source["window"], drop_prefix=source["drop_prefix"])
            for p in positions:
                name = meta[p]["subset"]
                test_counts[name] = test_counts.get(name, 0) + 1
            if set(test_counts) != set(association["manifold_names"]):
                raise ValueError("coverage test subsets must match the selected development subsets")
    resolved_device = resolve_device(device)
    model = model.to(resolved_device).eval()
    fit = evaluate_fit(model, data, model_kind=model_kind, layer=layer,
                       batch_size=batch_size, device=resolved_device)
    ranks = effective_component_ranks(model, rank_threshold)
    valid = model.pca_valid.cpu() if model_kind == "kmeans" else torch.ones(model.K, dtype=torch.bool)
    if model_kind == "kmeans":
        definition = {"definition": "kmeans_component_ranks"}
    elif model_kind == "hddc":
        definition = {"definition": "hddc_rank_mask_count"}
    else:
        definition = {"definition": "loading_variance_above_noise_threshold", "threshold": float(rank_threshold)}
    live = data.cluster_sizes > 0
    report = {
        "schema_version": 2, "evaluation": "unsupervised", "model_kind": model_kind,
        "K": int(model.K), "q_capacity": int(model.q),
        "dataset": data.dataset_summary(layer),
        **fit,
        "components": {"live": int(live.sum()), "dead": int((~live).sum())},
        "rank": {
            **definition, "population": "pca_valid_components" if model_kind == "kmeans" else "all_components",
            "components": int(valid.sum()),
            "mean_learned": float(ranks[valid].float().mean()) if bool(valid.any()) else None,
        },
    }
    if model_kind == "kmeans":
        report["pca_geometry"] = {
            "version": 1, "minimum_cluster_points": 2,
            "eligible_components": int(valid.sum()), "excluded_components": int((~valid).sum()),
        }
    eigenspaces = None
    if tile_alignment and bool((live & valid & (ranks > 0)).any()):
        eigenspaces = leading_covariance_eigenspaces(model, int(ranks[live & valid].max()))
    if tile_alignment:
        report["tile_alignment"] = evaluate_tile_alignment(
            model, live, neighbors=tile_alignment_neighbors, rank_threshold=rank_threshold,
            eigenspaces=eigenspaces,
        )
    if pc1_density:
        report["pc1_density"] = evaluate_pc1_density(
            model, data, model_kind=model_kind, layer=layer, batch_size=batch_size,
            space=pc1_density_space, neighbors=pc1_density_neighbors,
            random_directions=pc1_density_random_directions, seed=pc1_density_seed,
        )
    if source is not None:
        report["heldout_distribution_coverage"] = _test_coverage(
            source, model.mu, data.train_cluster_sizes > 0, batch_size=batch_size,
        )
    if model_kind in {"hddc", "kmeans"}:
        report["stratification"] = association
        report["per_manifold"] = []
        for manifold_id, (name, group_data) in enumerate(groups):
            component_mask = torch.tensor([
                r["manifold_id"] == manifold_id for r in association["per_component"]
            ], dtype=torch.bool)
            group = {
                "manifold_id": manifold_id, "subset": name,
                "component_ids": torch.where(component_mask)[0].tolist(),
                "dataset": group_data.dataset_summary(layer),
                "reconstruction": subset_reconstruction(
                    model, group_data, model_kind=model_kind, layer=layer,
                    batch_size=batch_size, device=resolved_device,
                ),
            }
            if tile_alignment:
                group["tile_alignment"] = evaluate_tile_alignment(
                    model, component_mask, neighbors=tile_alignment_neighbors,
                    rank_threshold=rank_threshold, eigenspaces=eigenspaces,
                )
            if pc1_density:
                try:
                    group["pc1_density"] = evaluate_pc1_density(
                        model, group_data, model_kind=model_kind, layer=layer, batch_size=batch_size,
                        space=pc1_density_space, neighbors=pc1_density_neighbors,
                        random_directions=pc1_density_random_directions, seed=pc1_density_seed,
                        component_mask=component_mask,
                    )
                except ValueError as exc:
                    raise ValueError(f"PC1 density for subset {name!r}: {exc}") from exc
            if source is not None:
                group_source = dict(source, manifold_subset=name, num_rows=test_counts[name],
                                    num_points=test_counts[name] * data.tokens_per_row)
                group["heldout_distribution_coverage"] = _test_coverage(
                    group_source, model.mu, data.train_cluster_sizes > 0, batch_size=batch_size,
                )
            report["per_manifold"].append(group)
    return report


def unsupervised_report_valid(run: Mapping[str, Any], metrics: Any) -> bool:
    """Validate the report required for pipeline completion and reevaluation."""
    if not isinstance(metrics, Mapping):
        return False
    k = run["training"]["arguments"]["K"]
    kind = run["training"]["model_kind"]
    if (metrics.get("schema_version") != 2 or metrics.get("evaluation") != "unsupervised"
            or metrics.get("identity_hash") != run["identity_hash"]
            or metrics.get("model_kind") != kind or metrics.get("K") != k):
        return False
    if run["evaluation"].get("tile_alignment", True):
        if not tile_alignment_report_valid(
            metrics.get("tile_alignment"), components=k,
            neighbors=run["evaluation"].get("tile_alignment_neighbors", 5),
        ):
            return False
    elif "tile_alignment" in metrics:
        return False
    required_blocks = ("dataset", "components", "rank", "nll", "bic")
    if kind == "kmeans":
        required_blocks += ("pca_geometry", "gaussian_surrogate")
    if any(not isinstance(metrics.get(key), Mapping) for key in required_blocks):
        return False
    dataset = metrics["dataset"]
    if kind in {"hddc", "kmeans"} and not reconstruction_report_valid(
        metrics.get("reconstruction"), model_kind="hddc", dataset=dataset,
    ):
        return False
    root, subset = split_shard_dir_spec(run["dataset"]["shard_dir"])
    if (not isinstance(dataset.get("shard_dir"), str)
            or Path(dataset["shard_dir"]).resolve() != root.resolve()
            or dataset.get("subset_spec") != subset or dataset.get("layer") != run["dataset"]["layer"]):
        return False
    for field in ("selected_rows", "train_rows", "validation_rows", "selected_points",
                  "train_points", "validation_points", "window", "drop_prefix"):
        if type(dataset.get(field)) is not int or dataset[field] < 0:
            return False
    tokens_per_row = dataset["window"] - dataset["drop_prefix"]
    if (tokens_per_row <= 0 or dataset["train_rows"] == 0
            or dataset["train_rows"] + dataset["validation_rows"] != dataset["selected_rows"]):
        return False
    if any(dataset[f"{split}_points"] != dataset[f"{split}_rows"] * tokens_per_row
           for split in ("selected", "train", "validation")):
        return False
    if run["evaluation"].get("pc1_density", False):
        if not pc1_density_report_valid(metrics.get("pc1_density"), components=k,
                                        cfg=run["evaluation"], dataset=dataset):
            return False
    elif "pc1_density" in metrics:
        return False
    components = metrics.get("components", {})
    if (any(type(components.get(key)) is not int or components[key] < 0 for key in ("live", "dead"))
            or components["live"] + components["dead"] != k):
        return False
    rank = metrics.get("rank", {})
    expected_definition = {"kmeans": "kmeans_component_ranks", "hddc": "hddc_rank_mask_count"}.get(
        kind, "loading_variance_above_noise_threshold",
    )
    count, mean = rank.get("components"), rank.get("mean_learned")
    if (rank.get("definition") != expected_definition or type(count) is not int or not 0 <= count <= k
            or (count == 0 and mean is not None)
            or (count > 0 and (type(mean) not in (float, int) or not math.isfinite(mean) or mean < 0))):
        return False
    population = "pca_valid_components" if kind == "kmeans" else "all_components"
    if rank.get("population") != population or (kind != "kmeans" and count != k):
        return False
    if kind not in {"hddc", "kmeans"} and rank.get("threshold") != run["evaluation"]["rank_threshold"]:
        return False
    if kind == "kmeans":
        geometry = metrics.get("pca_geometry", {})
        if (geometry.get("version") != 1 or geometry.get("minimum_cluster_points") != 2
                or geometry.get("eligible_components") != count or geometry.get("excluded_components") != k - count):
            return False
        if "quantization" in metrics or metrics["gaussian_surrogate"].get("source_model") != "kmeans":
            return False
    nll, bic = metrics.get("nll", {}), metrics.get("bic", {})
    if any(type(nll.get(split)) not in (float, int) or not math.isfinite(nll[split])
           for split in ("train", "validation")):
        return False
    if (bic.get("formula") != "-standard_bic / n + active_components"
            or bic.get("convention") != "higher_is_better"
            or bic.get("n") != dataset["train_points"]
            or type(bic.get("value")) not in (float, int) or not math.isfinite(bic["value"])):
        return False
    if run["evaluation"].get("heldout_distribution_coverage", True):
        coverage = metrics.get("heldout_distribution_coverage")
        if not coverage_report_valid(coverage, components=k):
            return False
        source = coverage.get("source", {})
        test_path = run["evaluation"].get("test_shard_dir")
        if not test_path or not isinstance(source, Mapping) or not isinstance(source.get("shard_dir"), str):
            return False
        root, subset = split_shard_dir_spec(test_path)
        if (Path(source["shard_dir"]).resolve() != root.resolve()
                or source.get("subset_spec") != subset or source.get("layer") != run["dataset"]["layer"]
                or source.get("num_points") != coverage["reference_points"]):
            return False
        rows, window, prefix = (source.get(key) for key in ("num_rows", "window", "drop_prefix"))
        if (any(type(value) is not int for value in (rows, window, prefix))
                or rows <= 0 or window != dataset["window"] or prefix != dataset["drop_prefix"]
                or rows * (window - prefix) != source["num_points"]):
            return False
    elif "heldout_distribution_coverage" in metrics:
        return False
    if kind in {"kmeans", "hddc"} and not subset_reports_valid(
        metrics, run["evaluation"], components=k, model_kind=kind,
    ):
        return False
    return True


__all__ = ["evaluate_unsupervised"]
