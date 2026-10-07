"""Evaluate how a model tiles a dataset with planted toy manifolds."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch
from sklearn.metrics import (
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
    normalized_mutual_info_score,
)
from torch.utils.data import DataLoader

from dalg.analysis.bic import bic_from_mean_nll, model_parameter_count
from dalg.analysis.bic_improved import active_bic_from_standard
from dalg.data.shard_activations import ActivationBatchDataset, load_meta_index
from dalg.data.subset_spec import resolve_spec_positions, split_shard_dir_spec
from dalg.evaluation.toy_manifold_metrics import evaluate_toy_manifold_metrics
from dalg.evaluation.toy_manifold_coverage import evaluate_toy_test_coverage, validate_toy_test_split


def _resolve_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("toy-manifold tiling requested CUDA, but CUDA is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("toy-manifold tiling requested MPS, but MPS is unavailable")
    return device


def _load_model(run_dir: Path, model_kind: str):
    if model_kind == "kmeans":
        from dalg.models.kmeans import load_kmeans

        return load_kmeans(run_dir / "kmeans_model.pt", map_location="cpu")
    model_path = run_dir / "mfa_model.pt"
    if model_kind == "mfa":
        from dalg.models.mfa import load_mfa

        return load_mfa(model_path, map_location="cpu")
    if model_kind == "ard":
        from dalg.models.adaptive_q.mfa_ard import load_mfa_ard

        return load_mfa_ard(model_path, map_location="cpu")
    if model_kind == "hddc":
        from dalg.models.adaptive_q.mfa_hddc import load_mfa_hddc

        return load_mfa_hddc(model_path, map_location="cpu")
    raise ValueError(f"toy_manifold_tiling does not support model.kind {model_kind!r}")


@torch.no_grad()
def _mean_nll(
    model,
    *,
    shard_dir: Path,
    layer: int,
    positions: list[int],
    batch_size: int,
    drop_prefix: int,
    device: torch.device,
) -> float:
    if not positions:
        raise ValueError("cannot evaluate NLL on an empty split")
    dataset = ActivationBatchDataset(
        shard_dir,
        layer=layer,
        row_subset=positions,
        batch_size=batch_size,
        drop_prefix=drop_prefix,
        dtype=torch.float32,
        shuffle_shards=False,
        shuffle_within_shard=False,
        seed=0,
    )
    loader = DataLoader(dataset, batch_size=None, num_workers=0)
    total_nll = 0.0
    total_points = 0
    with model.inference_cache():
        for batch in loader:
            x = batch.to(device, non_blocking=(device.type == "cuda"))
            total_nll += float(model.nll(x).item()) * int(x.shape[0])
            total_points += int(x.shape[0])
    if total_points == 0:
        raise ValueError("cannot evaluate NLL on an empty split")
    return total_nll / total_points


@torch.no_grad()
def _quantization_error(
    model,
    *,
    shard_dir: Path,
    layer: int,
    positions: list[int],
    batch_size: int,
    drop_prefix: int,
    device: torch.device,
) -> dict[str, float | int | None]:
    """Measure squared distances, reporting no mean for an empty split."""
    if not positions:
        return {"sum_squared_distance": 0.0, "mean_squared_distance": None, "n": 0}
    dataset = ActivationBatchDataset(
        shard_dir,
        layer=layer,
        row_subset=positions,
        batch_size=batch_size,
        drop_prefix=drop_prefix,
        dtype=torch.float32,
        shuffle_shards=False,
        shuffle_within_shard=False,
        seed=0,
    )
    total_squared_distance = 0.0
    total_points = 0
    for batch in DataLoader(dataset, batch_size=None, num_workers=0):
        x = batch.to(device, non_blocking=(device.type == "cuda"))
        assigned_means = model.mu[model.predict(x)]
        total_squared_distance += float((x.double() - assigned_means.double()).square().sum())
        total_points += len(x)
    if total_points == 0:
        raise ValueError("cannot evaluate quantization error on an empty split")
    return {
        "sum_squared_distance": total_squared_distance,
        "mean_squared_distance": total_squared_distance / total_points,
        "n": total_points,
    }


def evaluate_toy_manifold_tiling(
    run_dir: str | Path,
    *,
    shard_dir: str | Path,
    layer: int,
    model_kind: str,
    assignments_path: str | Path | None = None,
    batch_size: int = 4096,
    device: str = "cuda",
    rank_threshold: float = 1.0,
    max_mean_to_manifold_distance: float | None = None,
    heldout_distribution_coverage: bool = True,
) -> dict[str, Any]:
    """Evaluate one model run against planted toy-manifold structure."""
    if type(heldout_distribution_coverage) is not bool:
        raise ValueError("heldout_distribution_coverage must be true or false")
    if model_kind not in {"hddc", "kmeans"} and rank_threshold <= 0.0:
        raise ValueError("rank_threshold must be positive")
    if max_mean_to_manifold_distance is not None and (
        not math.isfinite(max_mean_to_manifold_distance)
        or max_mean_to_manifold_distance <= 0.0
    ):
        raise ValueError("max_mean_to_manifold_distance must be finite and positive")

    run_dir = Path(run_dir)
    test_source = validate_toy_test_split(shard_dir, layer=layer) if heldout_distribution_coverage else None
    model_stem = "kmeans_model" if model_kind == "kmeans" else "mfa_model"
    assignments_path = (
        Path(assignments_path)
        if assignments_path is not None
        else run_dir / f"{model_stem}_assignments.pt"
    )
    required = [
        run_dir / "config.json",
        run_dir / "val_indices.json",
        run_dir / f"{model_stem}.pt",
        assignments_path,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing toy-manifold tiling artifacts: {missing}")

    clean_shard_dir, subset_spec = split_shard_dir_spec(str(shard_dir))
    shard_config = json.loads((clean_shard_dir / "config.json").read_text())
    if shard_config.get("source_kind") != "toy_manifolds":
        raise ValueError(
            "toy_manifold_tiling requires shards from save_toy_manifold_shards"
        )
    window = int(shard_config["window"])
    drop_prefix = int(shard_config.get("drop_prefix", 0))
    if window != 1 or drop_prefix != 0:
        raise ValueError("toy_manifold_tiling expects one activation per row")

    metadata_path = clean_shard_dir / shard_config["manifold_metadata"]
    manifold_metadata = torch.load(
        metadata_path,
        map_location="cpu",
        weights_only=True,
    )
    all_manifold_ids = manifold_metadata["row_manifold_ids"].reshape(-1).long()

    meta_index = load_meta_index(clean_shard_dir, layer=layer)
    positions = resolve_spec_positions(
        meta_index,
        subset_spec,
        window=window,
        drop_prefix=drop_prefix,
    )
    if all_manifold_ids.numel() != len(meta_index):
        raise ValueError(
            "toy manifold labels are not aligned with activation metadata: "
            f"labels={all_manifold_ids.numel()}, rows={len(meta_index)}"
        )
    row_manifold_ids = all_manifold_ids[torch.as_tensor(positions, dtype=torch.long)]

    assignment_bundle = torch.load(
        assignments_path,
        map_location="cpu",
        mmap=True,
        weights_only=True,
    )
    if model_kind == "kmeans":
        integer_dtypes = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
        for field in ("assignments", "cluster_sizes"):
            value = assignment_bundle.get(field)
            if not isinstance(value, torch.Tensor) or value.dtype not in integer_dtypes:
                raise ValueError(f"KMeans assignment {field} must contain integer values")
        if assignment_bundle.get("model_type") != "kmeans":
            raise ValueError("KMeans assignment model_type must be 'kmeans'")
        saved_model_path = assignment_bundle.get("model_path")
        if not isinstance(saved_model_path, str) or (
            Path(saved_model_path).resolve() != (run_dir / "kmeans_model.pt").resolve()
        ):
            raise ValueError("KMeans assignment model_path does not match the evaluated checkpoint")
        if "subset_spec" not in assignment_bundle:
            raise ValueError("KMeans assignments must record subset_spec")
        source = assignment_bundle.get("source")
        if not isinstance(source, Mapping):
            raise ValueError("KMeans assignments must record source provenance")
        saved_shard_dir = source.get("shard_dir")
        if not isinstance(saved_shard_dir, str) or (
            Path(saved_shard_dir).resolve() != clean_shard_dir.resolve()
        ):
            raise ValueError("KMeans assignment source.shard_dir does not match the evaluation dataset")
        for field, expected in (
            ("layer", layer), ("drop_prefix", drop_prefix), ("num_items", len(positions))
        ):
            if type(source.get(field)) is not int or source[field] != expected:
                raise ValueError(f"KMeans assignment source.{field} does not match the evaluation dataset")

    assignments = assignment_bundle["assignments"].reshape(-1).long()
    cluster_sizes = assignment_bundle["cluster_sizes"].reshape(-1).long()
    if assignment_bundle.get("subset_spec") != subset_spec:
        raise ValueError(
            "assignment subset does not match evaluation dataset: "
            f"assignments={assignment_bundle.get('subset_spec')!r}, "
            f"evaluation={subset_spec!r}"
        )
    if assignments.numel() != len(positions):
        raise ValueError(
            "assignments must cover the selected canonical stream: "
            f"assignments={assignments.numel()}, selected rows={len(positions)}"
        )
    if int(cluster_sizes.sum()) != assignments.numel():
        raise ValueError("cluster_sizes does not sum to the assignment count")

    model = _load_model(run_dir, model_kind)
    if cluster_sizes.numel() != model.K or int(assignment_bundle["K"]) != model.K:
        raise ValueError("assignment K does not match the loaded model")
    if int(shard_config["d_model"]) != model.D:
        raise ValueError("toy-manifold shard dimension does not match the loaded model")
    if not torch.equal(torch.bincount(assignments, minlength=model.K), cluster_sizes):
        raise ValueError("cluster_sizes is inconsistent with assignments")

    split_info = json.loads((run_dir / "val_indices.json").read_text())
    val_global_rows = set(split_info["val_global_rows"])
    val_positions = [
        position
        for position in positions
        if meta_index[position]["global_row"] in val_global_rows
    ]
    val_position_set = set(val_positions)
    train_positions = [
        position for position in positions if position not in val_position_set
    ]
    if len(train_positions) != int(split_info["train_rows"]):
        raise ValueError("reconstructed training split does not match val_indices.json")
    if len(val_positions) != int(split_info["val_rows"]):
        raise ValueError("reconstructed validation split does not match val_indices.json")
    train_mask = torch.tensor(
        [position not in val_position_set for position in positions], dtype=torch.bool,
    )
    train_cluster_sizes = torch.bincount(assignments[train_mask], minlength=model.K)

    resolved_device = _resolve_device(device)
    model = model.to(resolved_device).eval()
    if model_kind == "kmeans":
        if not train_positions:
            raise ValueError("cannot evaluate KMeans quantization error on an empty training split")
        fit_metrics = {
            "quantization": {
                name: _quantization_error(
                    model,
                    shard_dir=clean_shard_dir,
                    layer=layer,
                    positions=split_positions,
                    batch_size=batch_size,
                    drop_prefix=drop_prefix,
                    device=resolved_device,
                )
                for name, split_positions in (
                    ("train", train_positions), ("validation", val_positions)
                )
            }
        }
        fit_metrics["quantization"]["convention"] = "lower_is_better"
    else:
        train_nll = _mean_nll(
            model,
            shard_dir=clean_shard_dir,
            layer=layer,
            positions=train_positions,
            batch_size=batch_size,
            drop_prefix=drop_prefix,
            device=resolved_device,
        )
        val_nll = _mean_nll(
            model,
            shard_dir=clean_shard_dir,
            layer=layer,
            positions=val_positions,
            batch_size=batch_size,
            drop_prefix=drop_prefix,
            device=resolved_device,
        )
        n_train = len(train_positions) * (window - drop_prefix)
        bic_parameters = model_parameter_count(model, model_kind)
        standard_bic = bic_from_mean_nll(
            model,
            model_kind,
            mean_nll=train_nll,
            n=n_train,
        )
        active_components = int((train_cluster_sizes > 0).sum())
        bic = active_bic_from_standard(
            standard_bic,
            n=n_train,
            active_components=active_components,
            K=model.K,
        )

        fit_metrics = {
            "nll": {"train": train_nll, "validation": val_nll},
            "bic": {
                "value": bic,
                "standard_bic": standard_bic,
                "standard_bic_per_sample_reward": -standard_bic / n_train,
                "activity_reward": active_components,
                "active_components": active_components,
                "inactive_components": int(model.K) - active_components,
                "K": int(model.K),
                "parameters": bic_parameters,
                "n": n_train,
                "split": "train",
                "assignment_rule": "hard_map_count_greater_than_zero",
                "formula": "-standard_bic / n + active_components",
                "convention": "higher_is_better",
            },
        }

    true_ids = row_manifold_ids.numpy()
    predicted_ids = assignments.numpy()
    clustering = {
        "homogeneity": float(homogeneity_score(true_ids, predicted_ids)),
        "completeness": float(completeness_score(true_ids, predicted_ids)),
        "adjusted_rand_index": float(adjusted_rand_score(true_ids, predicted_ids)),
        "normalized_mutual_information": float(
            normalized_mutual_info_score(true_ids, predicted_ids)
        ),
    }

    assignment_live = cluster_sizes > 0
    manifold_metrics = evaluate_toy_manifold_metrics(
        model,
        manifold_metadata,
        assignment_live,
        rank_threshold=rank_threshold,
        max_mean_to_manifold_distance=max_mean_to_manifold_distance,
    )

    coverage_metrics = {}
    if test_source is not None:
        coverage_metrics["heldout_distribution_coverage"] = evaluate_toy_test_coverage(
            test_source, model.mu, train_cluster_sizes > 0, batch_size=batch_size,
        )

    return {
        "schema_version": 2,
        "evaluation": "toy_manifold_tiling",
        "model_kind": model_kind,
        "K": int(model.K),
        "q_capacity": int(model.q),
        "dataset": {
            "shard_dir": str(clean_shard_dir),
            "subset_spec": subset_spec,
            "layer": int(layer),
            "selected_rows": len(positions),
            "train_rows": len(train_positions),
            "validation_rows": len(val_positions),
        },
        **fit_metrics,
        "clustering": clustering,
        "components": {
            "live": int(assignment_live.sum()),
            "dead": int((~assignment_live).sum()),
        },
        **manifold_metrics,
        **coverage_metrics,
    }


__all__ = ["evaluate_toy_manifold_tiling"]


def evaluate_pipeline_run(run: Mapping[str, Any], *, evaluation: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Evaluate a manifest row using its saved model and assignment artifacts."""
    cfg = run["evaluation"] if evaluation is None else evaluation
    if cfg["kind"] != "toy_manifold_tiling":
        raise ValueError(f"unsupported evaluator: {cfg['kind']!r}")
    stem = "kmeans_model" if run["training"]["model_kind"] == "kmeans" else "mfa_model"
    metrics = evaluate_toy_manifold_tiling(
        run["run_dir"], shard_dir=run["dataset"]["shard_dir"],
        layer=int(run["dataset"]["layer"]), model_kind=run["training"]["model_kind"],
        assignments_path=Path(run["run_dir"]) / f"{stem}_assignments.pt",
        batch_size=int(cfg["batch_size"]), device=str(cfg["device"]),
        rank_threshold=float(cfg["rank_threshold"]),
        max_mean_to_manifold_distance=(None if cfg["max_mean_to_manifold_distance"] is None
                                      else float(cfg["max_mean_to_manifold_distance"])),
        heldout_distribution_coverage=cfg.get("heldout_distribution_coverage", True),
    )
    metrics.update(run_id=run["run_id"], identity_hash=run["identity_hash"])
    return metrics
