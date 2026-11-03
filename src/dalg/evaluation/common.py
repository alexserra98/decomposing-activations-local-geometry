"""Shared loading and fit metrics for model evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from dalg.analysis.bic import bic_from_mean_nll, model_parameter_count
from dalg.analysis.bic_improved import active_bic_from_standard
from dalg.analysis.reconstruction import ReconstructionMoments, reconstruction_errors
from dalg.data.shard_activations import ActivationBatchDataset, load_meta_index
from dalg.data.subset_spec import resolve_spec_positions, split_shard_dir_spec
from dalg.evaluation.reconstruction import reconstruction_metadata


@dataclass
class EvaluationData:
    shard_dir: Path
    subset_spec: str | None
    config: dict[str, Any]
    meta_index: list[dict]
    positions: list[int]
    train_positions: list[int]
    val_positions: list[int]
    drop_prefix: int
    assignments: torch.Tensor
    cluster_sizes: torch.Tensor
    train_cluster_sizes: torch.Tensor

    @property
    def tokens_per_row(self) -> int:
        return int(self.config["window"]) - self.drop_prefix

    def dataset_summary(self, layer: int) -> dict:
        return {
            "shard_dir": str(self.shard_dir), "subset_spec": self.subset_spec,
            "layer": int(layer), "selected_rows": len(self.positions),
            "train_rows": len(self.train_positions), "validation_rows": len(self.val_positions),
            "d_model": int(self.config["d_model"]),
            "window": int(self.config["window"]), "drop_prefix": self.drop_prefix,
            "selected_points": len(self.positions) * self.tokens_per_row,
            "train_points": len(self.train_positions) * self.tokens_per_row,
            "validation_points": len(self.val_positions) * self.tokens_per_row,
        }


def load_evaluation_data(
    run_dir: str | Path, *, shard_dir: str | Path, layer: int,
    model, model_kind: str, assignments_path: str | Path | None = None,
) -> EvaluationData:
    """Load a complete canonical partition and reconstruct training token occupancy."""
    run_dir = Path(run_dir)
    root, subset = split_shard_dir_spec(str(shard_dir))
    config = json.loads((root / "config.json").read_text())
    run_config = json.loads((run_dir / "config.json").read_text())
    split = json.loads((run_dir / "val_indices.json").read_text())
    window = int(config["window"])
    drop_prefix = int(run_config.get("drop_prefix", config.get("drop_prefix", 0 if model_kind == "kmeans" else 32)))
    if not 0 <= drop_prefix < window:
        raise ValueError("evaluation drop_prefix must leave at least one token per row")
    tokens_per_row = window - drop_prefix
    if "per_row_tokens" in split and split["per_row_tokens"] != tokens_per_row:
        raise ValueError("saved split token count differs from evaluation preprocessing")
    if int(config["d_model"]) != model.D:
        raise ValueError("shard dimension does not match the loaded model")
    meta = load_meta_index(root, layer=layer)
    positions = resolve_spec_positions(meta, subset, window=window, drop_prefix=drop_prefix)
    if not positions:
        raise ValueError("evaluation requires a nonempty selected stream")
    val_rows = set(split["val_global_rows"])
    selected_rows = {meta[p]["global_row"] for p in positions}
    if not val_rows <= selected_rows:
        raise ValueError("saved validation rows are outside the selected stream")
    val_positions = [p for p in positions if meta[p]["global_row"] in val_rows]
    train_positions = [p for p in positions if meta[p]["global_row"] not in val_rows]
    if len(train_positions) != split["train_rows"] or len(val_positions) != split["val_rows"]:
        raise ValueError("reconstructed training/validation split does not match val_indices.json")

    stem = "kmeans_model" if model_kind == "kmeans" else "mfa_model"
    path = Path(assignments_path) if assignments_path is not None else run_dir / f"{stem}_assignments.pt"
    bundle = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    for field in ("assignments", "cluster_sizes"):
        value = bundle[field]
        if not isinstance(value, torch.Tensor) or value.dtype not in (
            torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
        ):
            raise ValueError(f"assignment {field} must contain integer values")
    assignments = bundle["assignments"].reshape(-1).long()
    sizes = bundle["cluster_sizes"].reshape(-1).long()
    expected = len(positions) * tokens_per_row
    if assignments.numel() != expected or bundle.get("subset_spec") != subset:
        raise ValueError("assignments must cover the selected canonical token stream")
    if sizes.numel() != model.K or bundle["K"] != model.K:
        raise ValueError("assignment K does not match the loaded model")
    if int(assignments.min()) < 0 or int(assignments.max()) >= model.K:
        raise ValueError("assignments contain an out-of-range component ID")
    if not torch.equal(torch.bincount(assignments, minlength=model.K), sizes):
        raise ValueError("cluster_sizes is inconsistent with assignments")
    expected_type = "mfa" if model_kind == "ard" else model_kind
    if bundle["model_type"] != expected_type:
        raise ValueError("assignment model_type does not match the evaluated model")
    if Path(bundle["model_path"]).resolve() != (run_dir / f"{stem}.pt").resolve():
        raise ValueError("assignment model_path does not match the evaluated checkpoint")
    source = bundle["source"]
    if Path(source["shard_dir"]).resolve() != root.resolve():
        raise ValueError("assignment source.shard_dir does not match the evaluation dataset")
    for field, value in (("layer", layer), ("drop_prefix", drop_prefix), ("num_items", expected)):
        if source[field] != value:
            raise ValueError(f"assignment source.{field} does not match the evaluation stream")

    # Canonical order is shard, row, then retained token position.
    train_mask = torch.tensor([meta[p]["global_row"] not in val_rows for p in positions])
    train_assignments = assignments.reshape(len(positions), tokens_per_row)[train_mask].reshape(-1)
    train_sizes = torch.bincount(train_assignments, minlength=model.K)
    return EvaluationData(root, subset, config, meta, positions, train_positions,
                          val_positions, drop_prefix, assignments, sizes, train_sizes)


def resolve_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("evaluation requested CUDA, but CUDA is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("evaluation requested MPS, but MPS is unavailable")
    return device


def load_model(run_dir: Path, model_kind: str):
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
    raise ValueError(f"evaluation does not support model.kind {model_kind!r}")


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
    reconstruction: ReconstructionMoments | None = None,
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
            if reconstruction is not None:
                reconstruction.update(*reconstruction_errors(model, x, model_kind="hddc"))
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
    reconstruction: ReconstructionMoments | None = None,
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
        labels = model.predict(x)
        assigned_means = model.mu[labels]
        total_squared_distance += float((x.double() - assigned_means.double()).square().sum())
        total_points += len(x)
        if reconstruction is not None:
            reconstruction.update(*reconstruction_errors(model, x, model_kind="kmeans", labels=labels))
    if total_points == 0:
        raise ValueError("cannot evaluate quantization error on an empty split")
    return {
        "sum_squared_distance": total_squared_distance,
        "mean_squared_distance": total_squared_distance / total_points,
        "n": total_points,
    }


@torch.no_grad()
def evaluate_fit(model, data: EvaluationData, *, model_kind: str, layer: int,
                 batch_size: int, device: torch.device) -> dict[str, Any]:
    """Evaluate fit and reconstruction metrics using the saved partition."""
    clean_shard_dir = data.shard_dir
    train_positions, val_positions = data.train_positions, data.val_positions
    window, drop_prefix = int(data.config["window"]), data.drop_prefix
    train_cluster_sizes = data.train_cluster_sizes
    resolved_device = device
    reconstruction = {
        name: ReconstructionMoments() for name in ("train", "validation")
    } if model_kind in {"hddc", "kmeans"} else {}
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
                    reconstruction=reconstruction[name],
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
            reconstruction=reconstruction.get("train"),
        )
        val_nll = _mean_nll(
            model,
            shard_dir=clean_shard_dir,
            layer=layer,
            positions=val_positions,
            batch_size=batch_size,
            drop_prefix=drop_prefix,
            device=resolved_device,
            reconstruction=reconstruction.get("validation"),
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

    if reconstruction:
        fit_metrics["reconstruction"] = {
            **reconstruction_metadata(model_kind, model.D),
            **{name: moments.summary() for name, moments in reconstruction.items()},
        }
    return fit_metrics
