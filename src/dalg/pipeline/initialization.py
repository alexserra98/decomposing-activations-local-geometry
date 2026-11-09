"""Automatic KMeans initialization commands and training-split provenance checks."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from dalg.init.activation_selection import resolve_initialization_rows

from .common import PipelineConfigError


def _initialization_selection(run: Mapping[str, Any]) -> tuple[Path, dict, list[int], dict]:
    cfg = run["initialization"]
    return resolve_initialization_rows(
        run["dataset"]["shard_dir"], layer=int(run["dataset"]["layer"]),
        val_frac=cfg["val_frac"], split_seed=cfg["split_seed"],
        default_drop_prefix=cfg["drop_prefix_default"],
    )


def _initialization_command(run: Mapping[str, Any]) -> list[str]:
    _check_initialization_version(run)
    cfg = run["initialization"]
    _root, _source, _positions, selection = _initialization_selection(run)
    options = {
        "shard_dir": run["dataset"]["shard_dir"],
        "layer": run["dataset"]["layer"],
        "K": run["training"]["arguments"]["K"],
        "out_dir": str(Path(run["run_dir"]) / "initialization"),
        "drop_prefix": selection["drop_prefix"],
        **{key: value for key, value in cfg.items()
           if key not in {"method", "version", "drop_prefix_default"}},
    }
    command = [sys.executable, "-m", "dalg.cli.run_training_kmeans"]
    for key, value in options.items():
        command.extend(["--" + key.replace("_", "-"), str(value)])
    return command


def _check_initialization_version(run: Mapping[str, Any]) -> None:
    cfg = run["initialization"]
    if cfg.get("version") != 4 or cfg.get("method") != "kmeans_model" or cfg.get("pca_purpose") != "initialization":
        raise PipelineConfigError(
            "legacy centroid initialization and older KMeans initialization manifests are incompatible; plan a new run "
            "using the KMeans model checkpoint contract (initialization version 4)"
        )


def _validate_initialization(run: Mapping[str, Any]) -> None:
    """Check the KMeans checkpoint and its exact training-split provenance."""
    from dalg.models.kmeans import load_kmeans

    _check_initialization_version(run)
    directory = Path(run["run_dir"]) / "initialization"
    cfg = run["initialization"]
    root, source, _positions, selection = _initialization_selection(run)
    path = directory / "kmeans_model.pt"
    model = load_kmeans(path, map_location="cpu")
    k = int(run["training"]["arguments"]["K"])
    if (model.K, model.D, model.q_init) != (k, int(source["d_model"]), cfg["rank"]):
        raise ValueError("initialization model dimensions do not match the manifest")
    checkpoint = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    metadata = checkpoint["meta"]["extra"]
    expected = {
        "method": "kmeans", "K": k, "layer": run["dataset"]["layer"],
        "source_shard_dir": str(root.resolve()), "selection": selection,
        "rows_used": selection["train_activations"], "uses_all_training_rows": True,
        "device": torch.device(cfg["device"]).type,
        "sample_fraction_requested": cfg["sample_fraction"], "sample_seed": cfg["sample_seed"],
        **{key: cfg[key] for key in ("max_iter", "restarts", "tol", "seed")},
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"initialization metadata mismatch for {key}: {path}")
    sizes = metadata["cluster_sizes"]
    if len(sizes) != k or sum(sizes) != selection["train_activations"] or min(sizes) < 0:
        raise ValueError("initialization cluster sizes violate the training-split contract")
    if model.q or model.surgery_threshold is not None:
        raise ValueError("initialization checkpoint must contain initialization PCs only")
    if model.init_pca_neighbors != cfg["pca_neighbors"] or model.init_pca_n_samples != selection["train_activations"]:
        raise ValueError("initialization PCA neighborhood does not match the manifest")


def _initialization_artifact_valid(run: Mapping[str, Any]) -> bool:
    if "initialization" not in run:
        return True
    try:
        _validate_initialization(run)
    except (EOFError, KeyError, OSError, TypeError, RuntimeError, ValueError):
        return False
    return True
