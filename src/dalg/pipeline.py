"""Small, manifest-driven training pipeline for DALG experiments.

This module intentionally orchestrates the existing CLIs instead of replacing
their training or analysis logic. A YAML experiment is expanded once into an
immutable JSONL manifest; each manifest row is one independently resumable
initialization -> train -> assignments -> evaluation run.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import math
import os
import re
import shlex
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import yaml

from dalg.data.subset_spec import split_shard_dir_spec
from dalg.init.activation_selection import resolve_initialization_rows
from dalg.evaluation.toy_manifold_coverage import coverage_report_valid, validate_toy_test_split


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = 1
_TOP_LEVEL_KEYS = {
    "experiment",
    "experimental",
    "dataset",
    "model",
    "training",
    "initialization",
    "assignments",
    "evaluation",
    "resources",
    "sweep",
}
_MODEL_MODULES = {
    "mfa": "dalg.cli.run_training",
    "ard": "dalg.cli.adaptive_q.run_training_ard",
    "hddc": "dalg.cli.adaptive_q.run_training_hddc",
    "kmeans": "dalg.cli.run_training_kmeans",
}
_RESOURCE_DEFAULTS = {
    "partition": "H100",
    "account": "LADE",
    "nodes": 1,
    "ntasks_per_node": 1,
    "cpus_per_task": 8,
    "gpus": 1,
    "gpu_type": "H100",
    "memory": "80G",
    "time": "23:00:00",
    "max_parallel": 4,
}
_RESOURCE_KEYS = set(_RESOURCE_DEFAULTS)
_ASSIGNMENT_DEFAULTS = {
    "enabled": True,
    "batch_size": 1024,
    "device": "cuda",
    "seed": None,
    "use_inference_cache": True,
}
_EVALUATION_DEFAULTS = {
    "enabled": False,
    "kind": None,
    "batch_size": 4096,
    "device": "cuda",
    "rank_threshold": 1.0,
    "max_mean_to_manifold_distance": None,
    "heldout_distribution_coverage": True,
}


class PipelineConfigError(ValueError):
    """Raised when an experiment cannot be resolved safely."""


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise PipelineConfigError(f"{name} must be a mapping")
    return dict(value)


def load_experiment(path: str | Path) -> dict[str, Any]:
    """Load one YAML experiment without applying sweep expansion."""
    path = Path(path)
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, Mapping):
        raise PipelineConfigError("experiment YAML must contain a top-level mapping")
    payload = dict(payload)
    unknown = sorted(set(payload) - _TOP_LEVEL_KEYS)
    if unknown:
        raise PipelineConfigError(f"unknown top-level keys: {unknown}")
    for key in _TOP_LEVEL_KEYS - {"sweep"}:
        if key in payload:
            payload[key] = _require_mapping(payload[key], key)
    return payload


def _set_dotted(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    if len(parts) < 2 or any(not part for part in parts):
        raise PipelineConfigError(
            f"sweep key {dotted_key!r} must be a dotted path such as 'training.seed'"
        )
    node: dict[str, Any] = config
    for part in parts[:-1]:
        child = node.get(part)
        if not isinstance(child, dict):
            raise PipelineConfigError(f"sweep path {dotted_key!r} does not exist")
        node = child
    if parts[-1] not in node:
        raise PipelineConfigError(f"sweep path {dotted_key!r} does not exist")
    node[parts[-1]] = value


def expand_sweep(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand a simple Cartesian product over explicit dotted YAML fields."""
    sweep = _require_mapping(config.get("sweep"), "sweep")
    if not sweep:
        item = copy.deepcopy(config)
        item.pop("sweep", None)
        return [item]
    keys = list(sweep)
    values = []
    for key in keys:
        axis = sweep[key]
        if not isinstance(axis, list) or not axis:
            raise PipelineConfigError(f"sweep axis {key!r} must be a non-empty list")
        values.append(axis)
    expanded = []
    for combination in itertools.product(*values):
        item = copy.deepcopy(config)
        item.pop("sweep", None)
        for key, value in zip(keys, combination):
            _set_dotted(item, key, value)
        expanded.append(item)
    return expanded


def _parser_and_validator(model_kind: str):
    if model_kind == "mfa":
        from dalg.cli.run_training import build_parser, validate_args
    elif model_kind == "ard":
        from dalg.cli.adaptive_q.run_training_ard import build_parser, validate_args
    elif model_kind == "hddc":
        from dalg.cli.adaptive_q.run_training_hddc import build_parser, validate_args
    elif model_kind == "kmeans":
        from dalg.cli.run_training_kmeans import build_parser, validate_args
    else:
        raise PipelineConfigError(
            f"model.kind must be one of {sorted(_MODEL_MODULES)}, got {model_kind!r}"
        )
    return build_parser(), validate_args


def _action_by_dest(parser: argparse.ArgumentParser) -> dict[str, argparse.Action]:
    return {
        action.dest: action
        for action in parser._actions
        if action.dest != argparse.SUPPRESS and action.option_strings
    }


def _preferred_option(action: argparse.Action, *, negative: bool = False) -> str:
    long_options = [item for item in action.option_strings if item.startswith("--")]
    options = long_options or list(action.option_strings)
    if negative:
        candidates = [item for item in options if item.startswith("--no-")]
    else:
        candidates = [item for item in options if not item.startswith("--no-")]
    if not candidates:
        raise PipelineConfigError(f"cannot encode CLI option for {action.dest!r}")
    return candidates[0]


def _mapping_to_argv(
    parser: argparse.ArgumentParser,
    values: Mapping[str, Any],
) -> list[str]:
    actions = _action_by_dest(parser)
    unknown = sorted(set(values) - set(actions))
    if unknown:
        raise PipelineConfigError(f"unknown training parameters: {unknown}")
    argv: list[str] = []
    for key, value in values.items():
        if value is None:
            continue
        action = actions[key]
        if isinstance(action, argparse.BooleanOptionalAction):
            if not isinstance(value, bool):
                raise PipelineConfigError(f"{key} must be true or false")
            argv.append(_preferred_option(action, negative=not value))
        elif action.nargs == 0:
            if value == action.const:
                argv.append(_preferred_option(action))
            elif value != action.default:
                raise PipelineConfigError(
                    f"{key}={value!r} cannot be represented by its CLI flag"
                )
        else:
            argv.append(_preferred_option(action))
            if isinstance(value, (list, tuple)):
                argv.extend(str(item) for item in value)
            else:
                argv.append(str(value))
    return argv


def _parse_training_args(
    model_kind: str,
    values: dict[str, Any],
    *,
    world_size: int,
    generated_kmeans: bool = False,
) -> dict[str, Any]:
    parser, validator = _parser_and_validator(model_kind)
    try:
        args = parser.parse_args(_mapping_to_argv(parser, values))
    except SystemExit as exc:
        raise PipelineConfigError(f"invalid {model_kind} training parameters") from exc

    previous_world_size = os.environ.get("WORLD_SIZE")
    os.environ["WORLD_SIZE"] = str(world_size)
    # The pipeline will supply this artifact before invoking the unchanged CLI.
    if generated_kmeans:
        args.kmeans_model_path = "<pipeline-initialization>/kmeans_model.pt"
    try:
        validator(args)
    except (SystemExit, ValueError) as exc:
        raise PipelineConfigError(str(exc)) from exc
    finally:
        if generated_kmeans:
            args.kmeans_model_path = None
        if previous_world_size is None:
            os.environ.pop("WORLD_SIZE", None)
        else:
            os.environ["WORLD_SIZE"] = previous_world_size
    return vars(args)


def _resolve_shard_dir(value: str, subset: str | None) -> str:
    clean_path, inline_subset = split_shard_dir_spec(value)
    if subset and inline_subset:
        raise PipelineConfigError(
            "dataset subset is specified both in shard_dir and dataset.subset"
        )
    subset_spec = subset or inline_subset
    if not clean_path.is_absolute():
        clean_path = (REPO_ROOT / clean_path).resolve()
    return f"{clean_path}#{subset_spec}" if subset_spec else str(clean_path)


def _resolve_optional_path(value: Any) -> Any:
    if value in (None, ""):
        return value
    path = Path(str(value)).expanduser()
    return str(path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve())


def _resolve_kmeans_model_path(value: Any) -> str | None:
    """Resolve an explicit KMeans checkpoint used for initialization."""
    resolved = _resolve_optional_path(value)
    if not resolved:
        return None
    path = Path(resolved)
    if path.suffix != ".pt" or (path.exists() and not path.is_file()):
        raise PipelineConfigError("training.kmeans_model_path must point directly to a .pt file")
    return str(path)


def _resolve_init_model_path(value: Any) -> str | None:
    """Resolve the direct HDDC model file used for epoch-0 initialization."""
    resolved = _resolve_optional_path(value)
    if not resolved:
        return None
    path = Path(resolved)
    if path.suffix != ".pt":
        raise PipelineConfigError(
            "training.init_model_path must point directly to a .pt file"
        )
    if path.exists() and not path.is_file():
        raise PipelineConfigError(
            f"training.init_model_path must be a file, got: {path}"
        )
    return str(path)


def _validate_kmeans_model(
    path_value: str,
    *,
    expected_k: int,
    expected_rank: int,
    direction_init: str,
    shard_dir_arg: str,
) -> None:
    """Check the actual model checkpoint, rather than a legacy centroid bundle."""
    from dalg.models.kmeans import load_kmeans

    path = Path(path_value)
    if not path.is_file():
        raise PipelineConfigError(f"KMeans model file not found: {path}")
    try:
        model = load_kmeans(path, map_location="cpu")
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        raise PipelineConfigError(f"invalid KMeans model checkpoint: {path}: {exc}") from exc
    shard_dir, _ = split_shard_dir_spec(shard_dir_arg)
    expected_d = int(json.loads((shard_dir / "config.json").read_text())["d_model"])
    if model.K != expected_k or model.D != expected_d:
        raise PipelineConfigError(
            f"KMeans checkpoint shape {(model.K, model.D)} does not match "
            f"requested K,D={(expected_k, expected_d)}: {path}"
        )
    if direction_init == "cluster_pca" and model.q_init < expected_rank:
        raise PipelineConfigError(
            f"KMeans checkpoint stores {model.q_init} initialization principal components but "
            f"model rank/q_max={expected_rank} was requested: {path}"
        )


def _validate_hddc_init_model(
    path_value: str,
    *,
    expected_k: int,
    expected_rank: int,
    expected_isotropic_psi: bool,
    expected_shared_b: bool,
    shard_dir_arg: str,
) -> None:
    """Fail at planning time when an HDDC model cannot seed the requested run."""
    from dalg.models.adaptive_q.mfa_hddc import load_mfa_hddc

    path = Path(path_value)
    if not path.is_file():
        raise PipelineConfigError(f"initial model not found: {path}")
    try:
        model = load_mfa_hddc(path, map_location="cpu")
    except Exception as exc:
        raise PipelineConfigError(f"could not load initial HDDC model: {path}: {exc}") from exc

    shard_dir, _ = split_shard_dir_spec(shard_dir_arg)
    try:
        shard_config = json.loads((shard_dir / "config.json").read_text())
        expected_dim = int(shard_config["d_model"])
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise PipelineConfigError(
            f"could not read activation dimension from {shard_dir / 'config.json'}"
        ) from exc

    if model.K != expected_k or model.D != expected_dim:
        raise PipelineConfigError(
            f"initial model has (K={model.K}, D={model.D}), expected "
            f"(K={expected_k}, D={expected_dim}): {path}"
        )
    if model.q != expected_rank:
        raise PipelineConfigError(
            f"initial model rank q={model.q} does not match model.q_max={expected_rank}: {path}"
        )
    if (
        model.isotropic_psi != expected_isotropic_psi
        or getattr(model, "shared_b", False) != expected_shared_b
    ):
        raise PipelineConfigError(
            "initial model and requested HDDC run must use the same Psi noise mode"
        )
    if getattr(model, "_rotation_on", False):
        raise PipelineConfigError("initial HDDC model must not have an active factor rotation")


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip()).strip("-").lower()
    return slug or "experiment"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _identity_hash(identity: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(identity).encode()).hexdigest()


def _normalise_resources(raw: Mapping[str, Any]) -> dict[str, Any]:
    unknown = sorted(set(raw) - _RESOURCE_KEYS)
    if unknown:
        raise PipelineConfigError(f"unknown resource parameters: {unknown}")
    resources = {**_RESOURCE_DEFAULTS, **dict(raw)}
    for key in ("nodes", "ntasks_per_node", "cpus_per_task", "gpus", "max_parallel"):
        resources[key] = int(resources[key])
    if resources["nodes"] <= 0 or resources["ntasks_per_node"] <= 0:
        raise PipelineConfigError("resource nodes and tasks must be positive")
    if resources["cpus_per_task"] <= 0 or resources["gpus"] < 0:
        raise PipelineConfigError("resource CPUs must be positive and GPUs non-negative")
    if resources["max_parallel"] <= 0:
        raise PipelineConfigError("resources.max_parallel must be positive")
    return resources


def _normalise_stage_config(
    raw: Mapping[str, Any],
    defaults: Mapping[str, Any],
    *,
    name: str,
) -> dict[str, Any]:
    unknown = sorted(set(raw) - set(defaults))
    if unknown:
        raise PipelineConfigError(f"unknown {name} parameters: {unknown}")
    return {**defaults, **dict(raw)}


def _validate_inputs(shard_dir_arg: str, layer: int) -> None:
    shard_dir, _ = split_shard_dir_spec(shard_dir_arg)
    if not (shard_dir / "config.json").is_file():
        raise PipelineConfigError(f"activation shard config not found: {shard_dir / 'config.json'}")
    if not (shard_dir / f"layer{layer:02d}").is_dir():
        raise PipelineConfigError(
            f"activation layer directory not found: {shard_dir / f'layer{layer:02d}'}"
        )


def resolve_run(config: Mapping[str, Any], *, check_inputs: bool = True) -> dict[str, Any]:
    """Resolve one already-expanded YAML configuration into a manifest row."""
    experiment = _require_mapping(config.get("experiment"), "experiment")
    dataset = _require_mapping(config.get("dataset"), "dataset")
    model = _require_mapping(config.get("model"), "model")
    training = _require_mapping(config.get("training"), "training")
    experimental = _require_mapping(config.get("experimental"), "experimental")
    unknown_experimental = sorted(set(experimental) - {"hard_assignment_covariance"})
    if unknown_experimental:
        raise PipelineConfigError(f"unknown experimental parameters: {unknown_experimental}")
    hard_covariance = experimental.get("hard_assignment_covariance", False)
    if not isinstance(hard_covariance, bool):
        raise PipelineConfigError("experimental.hard_assignment_covariance must be true or false")
    initialization_raw = _require_mapping(config.get("initialization"), "initialization")
    unknown_initialization = sorted(set(initialization_raw) - {"pca_neighbors"})
    if unknown_initialization:
        raise PipelineConfigError(f"unknown initialization parameters: {unknown_initialization}")
    pca_neighbors = initialization_raw.get("pca_neighbors", 64)
    if type(pca_neighbors) is not int or pca_neighbors <= 0:
        raise PipelineConfigError("initialization.pca_neighbors must be a positive integer")
    assignments_raw = _require_mapping(config.get("assignments"), "assignments")
    evaluation_raw = _require_mapping(config.get("evaluation"), "evaluation")
    resources = _normalise_resources(_require_mapping(config.get("resources"), "resources"))

    name = str(experiment.get("name", "")).strip()
    output_root_raw = experiment.get("output_root")
    if not name or output_root_raw in (None, ""):
        raise PipelineConfigError("experiment.name and experiment.output_root are required")
    output_root = Path(str(output_root_raw)).expanduser()
    if not output_root.is_absolute():
        output_root = (REPO_ROOT / output_root).resolve()

    if "shard_dir" not in dataset or "layer" not in dataset:
        raise PipelineConfigError("dataset.shard_dir and dataset.layer are required")
    shard_dir = _resolve_shard_dir(
        str(dataset["shard_dir"]),
        None if dataset.get("subset") is None else str(dataset["subset"]),
    )
    layer = int(dataset["layer"])
    dataset_id = str(dataset.get("id") or Path(split_shard_dir_spec(shard_dir)[0]).name)
    unknown_dataset = sorted(set(dataset) - {"id", "shard_dir", "subset", "layer"})
    if unknown_dataset:
        raise PipelineConfigError(f"unknown dataset parameters: {unknown_dataset}")
    if check_inputs:
        _validate_inputs(shard_dir, layer)

    model_kind = str(model.get("kind", "")).lower()
    model_values = dict(model)
    model_values.pop("kind", None)
    if "q_max" in model_values:
        if "rank" in model_values:
            raise PipelineConfigError("set only one of model.rank and model.q_max")
        model_values["rank"] = model_values.pop("q_max")
    overlap = sorted(set(model_values) & set(training))
    if overlap:
        raise PipelineConfigError(
            f"parameters must appear in only one of model or training: {overlap}"
        )
    if "out_dir" in model_values or "out_dir" in training:
        raise PipelineConfigError("out_dir is derived by the pipeline and cannot be set")
    values = {
        "shard_dir": shard_dir,
        "layer": layer,
        **model_values,
        **training,
    }
    if "hard_assignment_covariance" in values:
        raise PipelineConfigError("hard_assignment_covariance belongs in experimental")
    if hard_covariance and model_kind != "hddc":
        raise PipelineConfigError("experimental.hard_assignment_covariance requires model.kind: hddc")
    if model_kind == "hddc":
        values["hard_assignment_covariance"] = hard_covariance
    if "centroids_path" in values:
        raise PipelineConfigError(
            "training.centroids_path is deprecated in the pipeline; use "
            "training.kmeans_model_path with a KMeans model checkpoint"
        )
    if "kmeans_model_path" in values:
        values["kmeans_model_path"] = _resolve_kmeans_model_path(values["kmeans_model_path"])
    if "init_model_path" in values:
        values["init_model_path"] = _resolve_init_model_path(values["init_model_path"])

    default_training_mode = "single_process" if model_kind in {"hddc", "kmeans"} else "vanilla"
    training_mode = str(values.get("training_mode", default_training_mode))
    world_size = resources["gpus"] if training_mode == "component_shard" else 1
    if training_mode == "component_shard" and resources["gpus"] <= 1:
        raise PipelineConfigError(
            "component_shard training requires resources.gpus greater than one"
        )
    if model_kind == "ard" and training_mode == "component_shard":
        raise PipelineConfigError("ARD training does not support component sharding")
    generated_kmeans = (
        model_kind in {"mfa", "ard", "hddc"}
        and not values.get("kmeans_model_path")
        and not values.get("init_model_path")
        and not (model_kind == "hddc" and values.get("fit_method") == "em")
    )
    if initialization_raw and not generated_kmeans:
        raise PipelineConfigError(
            "initialization options require automatic KMeans/PCA; omit kmeans_model_path "
            "and init_model_path, and use an MFA, ARD, or Adam-based HDDC run"
        )
    if generated_kmeans:
        values.setdefault("direction_init", "cluster_pca")
    training_args = _parse_training_args(
        model_kind, values, world_size=world_size,
        generated_kmeans=generated_kmeans,
    )
    training_args["out_dir"] = None
    if model_kind == "kmeans":
        if training_args["pca_purpose"] != "geometry":
            raise PipelineConfigError("KMeans pipeline models require cluster geometry; initialization PCA belongs to MFA initialization")
        if training_args["pca_only"]:
            raise PipelineConfigError("pca_only is a standalone checkpoint operation; pipeline runs fit a new model")
        if training_args["rank"] is not None:
            raise PipelineConfigError("KMeans geometry learns rank with Cattell; omit model.rank")
        if resources["gpus"] > 1 or resources["ntasks_per_node"] != 1 or resources["nodes"] != 1:
            raise PipelineConfigError("KMeans supports one CPU or CUDA process")
        if check_inputs:
            _, source, _, selection = resolve_initialization_rows(
                shard_dir, layer=layer, val_frac=training_args["val_frac"],
                split_seed=training_args["split_seed"], drop_prefix=training_args["drop_prefix"],
            )
            n_fit = int(selection["train_activations"] * training_args["sample_fraction"])
            if not 1 <= training_args["K"] < n_fit:
                raise PipelineConfigError("KMeans requires 1 <= K < fitting activations")
    initialization = None
    if generated_kmeans:
        initialization = {
            "method": "kmeans_model",
            "version": 4,
            "pca_purpose": "initialization",
            "pca_neighbors": pca_neighbors,
            "max_iter": 100,
            "restarts": 10,
            "tol": 1e-6,
            "seed": training_args.get("seed") or 0,
            "device": training_args["device"],
            "rank": training_args["rank"],
            "val_frac": training_args["val_frac"],
            "split_seed": training_args["split_seed"],
            "drop_prefix_default": 32,
            "sample_fraction": 1.0,
            "sample_seed": 0,
            "load_batch_size": 20_000,
            "block_x": 8192,
            "block_c": 8192,
            "pca_chunk_elems": 1 << 23,
            "pca_eig_batch_size": 256,
        }
        if pca_neighbors <= training_args["rank"]:
            raise PipelineConfigError("initialization.pca_neighbors must exceed rank/q_max")
        if check_inputs:
            _, source, _, selection = resolve_initialization_rows(
                shard_dir, layer=layer,
                val_frac=initialization["val_frac"],
                split_seed=initialization["split_seed"],
                default_drop_prefix=initialization["drop_prefix_default"],
            )
            if not 1 <= training_args["rank"] <= int(source["d_model"]):
                raise PipelineConfigError("initialization PCA rank must be in [1, d_model]")
            if not 1 <= training_args["K"] < selection["train_activations"]:
                raise PipelineConfigError("initialization requires 1 <= K < training activations")
            if pca_neighbors > selection["train_activations"]:
                raise PipelineConfigError("initialization.pca_neighbors exceeds training activations")
    if check_inputs and training_args.get("kmeans_model_path"):
        _validate_kmeans_model(
            training_args["kmeans_model_path"],
            expected_k=int(training_args["K"]),
            expected_rank=int(training_args["rank"]),
            direction_init=str(training_args.get("direction_init", "random")),
            shard_dir_arg=shard_dir,
        )
    if check_inputs and training_args.get("init_model_path"):
        _validate_hddc_init_model(
            training_args["init_model_path"],
            expected_k=int(training_args["K"]),
            expected_rank=int(training_args["rank"]),
            expected_isotropic_psi=bool(training_args["isotropic_psi"]),
            expected_shared_b=bool(training_args["shared_b"]),
            shard_dir_arg=shard_dir,
        )

    assignments = _normalise_stage_config(
        assignments_raw,
        _ASSIGNMENT_DEFAULTS,
        name="assignments",
    )
    evaluation = _normalise_stage_config(
        evaluation_raw,
        _EVALUATION_DEFAULTS,
        name="evaluation",
    )
    assignments["enabled"] = bool(assignments["enabled"])
    evaluation["enabled"] = bool(evaluation["enabled"])
    if evaluation["enabled"] and not assignments["enabled"]:
        raise PipelineConfigError("evaluation requires assignments.enabled: true")
    if evaluation["enabled"] and evaluation["kind"] != "toy_manifold_tiling":
        raise PipelineConfigError(
            "the pipeline supports evaluation.kind: toy_manifold_tiling"
        )
    if model_kind not in {"hddc", "kmeans"} and float(evaluation["rank_threshold"]) <= 0:
        raise PipelineConfigError("evaluation.rank_threshold must be positive")
    max_mean_distance = evaluation["max_mean_to_manifold_distance"]
    if max_mean_distance is not None:
        max_mean_distance = float(max_mean_distance)
        if not math.isfinite(max_mean_distance) or max_mean_distance <= 0:
            raise PipelineConfigError(
                "evaluation.max_mean_to_manifold_distance must be finite and positive"
            )
    if assignments["seed"] is None:
        assignments["seed"] = training_args.get("seed") or 0
    if type(evaluation["heldout_distribution_coverage"]) is not bool:
        raise PipelineConfigError("evaluation.heldout_distribution_coverage must be true or false")
    if check_inputs and evaluation["enabled"] and evaluation["heldout_distribution_coverage"]:
        validate_toy_test_split(shard_dir, layer=layer)

    dataset_spec = {
        "id": dataset_id,
        "shard_dir": shard_dir,
        "layer": layer,
    }
    identity = {
        "schema_version": SCHEMA_VERSION,
        "experiment": name,
        "dataset": dataset_spec,
        "model_kind": model_kind,
        "training_args": {key: value for key, value in training_args.items() if key != "out_dir"},
        "assignments": assignments,
        "evaluation": evaluation,
    }
    if initialization is not None:
        identity["initialization"] = initialization
    digest = _identity_hash(identity)
    rank = training_args.get("rank")
    run_name = "__".join(
        [
            _slug(model_kind),
            _slug(dataset_id),
            f"l{layer:02d}",
            f"k{int(training_args['K'])}",
            "cattell" if model_kind == "kmeans" else f"q{int(rank)}",
            f"s{int(training_args.get('seed') or 0)}",
            digest[:8],
        ]
    )
    run_dir = output_root / _slug(name) / run_name
    training_args["out_dir"] = str(run_dir)
    if initialization is not None:
        training_args["kmeans_model_path"] = str(run_dir / "initialization" / "kmeans_model.pt")
    run = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_name,
        "identity_hash": digest,
        "identity": identity,
        "run_dir": str(run_dir),
        "resources": resources,
        "training": {
            "model_kind": model_kind,
            "module": _MODEL_MODULES[model_kind],
            "arguments": training_args,
        },
        "dataset": dataset_spec,
        "assignments": assignments,
        "evaluation": evaluation,
    }
    if initialization is not None:
        run["initialization"] = initialization
    return run


def resolve_experiment(path: str | Path, *, check_inputs: bool = True) -> list[dict[str, Any]]:
    config = load_experiment(path)
    runs = [resolve_run(item, check_inputs=check_inputs) for item in expand_sweep(config)]
    run_ids = [run["run_id"] for run in runs]
    if len(run_ids) != len(set(run_ids)):
        raise PipelineConfigError("the sweep expands to duplicate run configurations")
    return runs


def default_manifest_path(runs: list[dict[str, Any]]) -> Path:
    experiment_name = runs[0]["identity"]["experiment"]
    manifest_digest = hashlib.sha256(
        "\n".join(_canonical_json(run) for run in runs).encode()
    ).hexdigest()[:10]
    return REPO_ROOT / "outputs" / "experiments" / _slug(experiment_name) / (
        f"manifest_{manifest_digest}.jsonl"
    )


def write_manifest(runs: list[dict[str, Any]], path: str | Path) -> Path:
    path = Path(path).resolve()
    payload = "".join(_canonical_json(run) + "\n" for run in runs)
    if path.exists():
        if path.read_text() != payload:
            raise PipelineConfigError(f"refusing to overwrite a different manifest: {path}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(payload)
    tmp.replace(path)
    return path


def read_manifest(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise PipelineConfigError(f"manifest is empty: {path}")
    return rows


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def _run_command(command: list[str]) -> None:
    env = dict(os.environ)
    source_path = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = source_path + os.pathsep + env.get("PYTHONPATH", "")
    print(f"$ {shlex.join(command)}", flush=True)
    subprocess.run(command, check=True, cwd=REPO_ROOT, env=env)


def _model_stem(run: Mapping[str, Any]) -> str:
    return "kmeans_model" if run["training"]["model_kind"] == "kmeans" else "mfa_model"


def _assignments_path(run: Mapping[str, Any]) -> Path:
    return Path(run["run_dir"]) / f"{_model_stem(run)}_assignments.pt"


def _training_artifacts_valid(run: Mapping[str, Any]) -> bool:
    run_dir = Path(run["run_dir"])
    kind = run["training"]["model_kind"]
    if not (run_dir / "config.json").is_file() or not (run_dir / "val_indices.json").is_file():
        return False
    model_path = run_dir / f"{_model_stem(run)}.pt"
    if kind == "kmeans":
        from dalg.models.kmeans import load_kmeans

        try:
            model = load_kmeans(model_path, map_location="cpu")
            args = run["training"]["arguments"]
            saved = json.loads((run_dir / "config.json").read_text())
            split = json.loads((run_dir / "val_indices.json").read_text())
            root, source, positions, selection = resolve_initialization_rows(
                run["dataset"]["shard_dir"], layer=run["dataset"]["layer"],
                val_frac=args["val_frac"], split_seed=args["split_seed"],
                drop_prefix=args["drop_prefix"],
            )
            expected = {
                "selection": selection, "source_shard_dir": str(root.resolve()),
                "layer": run["dataset"]["layer"],
                "sample_fraction_requested": args["sample_fraction"],
                "sample_seed": args["sample_seed"],
                **{key: args[key] for key in ("max_iter", "restarts", "tol", "seed")},
            }
            if saved != model.checkpoint_extra or any(saved.get(k) != v for k, v in expected.items()):
                return False
            if any(getattr(model, key) != args[key] for key in ("max_iter", "restarts", "tol", "seed", "block_x", "block_c")):
                return False
            return (
                model.K == args["K"] and model.q == max(1, int(model.component_ranks.max()))
                and model.surgery_threshold == args.get("surgery_threshold", args.get("cattell_threshold"))
                and model.q_init == 0
                and int(model.cluster_counts.sum()) == saved["rows_used"]
                and model.cluster_counts.tolist() == saved["cluster_sizes"]
                and saved.get("K") == model.K
                and model.D == int(source["d_model"])
                and split["train_rows"] == selection["train_rows"]
                and split["val_rows"] == selection["selected_rows"] - selection["train_rows"]
            )
        except (EOFError, KeyError, OSError, TypeError, RuntimeError, ValueError):
            return False
    if model_path.is_file() and model_path.stat().st_size > 0:
        return True
    manifest_name = "mfa_model_shards.json"
    manifest_path = run_dir / manifest_name
    if not manifest_path.is_file() or kind == "ard":
        return False
    try:
        manifest = json.loads(manifest_path.read_text())
        shards = manifest["shards"]
    except (KeyError, OSError, TypeError, json.JSONDecodeError):
        return False
    return isinstance(shards, list) and bool(shards) and all(
        isinstance(name, str)
        and name
        and (run_dir / name).is_file()
        and (run_dir / name).stat().st_size > 0
        for name in shards
    )


def _training_artifact_path(run: Mapping[str, Any]) -> Path:
    run_dir = Path(run["run_dir"])
    model_path = run_dir / f"{_model_stem(run)}.pt"
    if model_path.is_file() or run["training"]["model_kind"] == "kmeans":
        return model_path
    return run_dir / "mfa_model_shards.json"


def _training_command(run: Mapping[str, Any]) -> list[str]:
    kind = run["training"]["model_kind"]
    module = run["training"]["module"]
    parser, _ = _parser_and_validator(kind)
    arguments = dict(run["training"]["arguments"])
    if arguments.get("wandb") and not arguments.get("wandb_name"):
        arguments["wandb_name"] = run["run_id"]
    argv = _mapping_to_argv(parser, arguments)
    if arguments.get("training_mode") == "component_shard":
        nproc = int(run["resources"]["gpus"])
        return [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            f"--nproc_per_node={nproc}",
            "-m",
            module,
            *argv,
        ]
    return [sys.executable, "-m", module, *argv]


def _validate_assignment_bundle(path: Path, expected_k: int) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        bundle = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        assignments = bundle["assignments"].reshape(-1)
        sizes = bundle["cluster_sizes"].reshape(-1)
        saved_k = int(bundle["K"])
    except (EOFError, KeyError, OSError, TypeError, RuntimeError, ValueError):
        return False
    if assignments.numel() == 0:
        return False
    if int(assignments.min()) < 0 or int(assignments.max()) >= expected_k:
        return False
    return (
        saved_k == expected_k
        and sizes.numel() == expected_k
        and int(sizes.sum()) == assignments.numel()
        and torch.equal(torch.bincount(assignments.long(), minlength=expected_k), sizes.long())
    )


def _assignment_artifact_valid(run: Mapping[str, Any]) -> bool:
    """Validate a complete model partition, including the KMeans source stream."""
    path = _assignments_path(run)
    k = int(run["training"]["arguments"]["K"])
    if not _validate_assignment_bundle(path, k):
        return False
    if run["training"]["model_kind"] != "kmeans":
        return True
    try:
        bundle = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        root, _, _, selection = resolve_initialization_rows(
            run["dataset"]["shard_dir"], layer=run["dataset"]["layer"],
            val_frac=0.0, drop_prefix=run["training"]["arguments"].get("drop_prefix"),
        )
        source = bundle["source"]
        expected = selection["selected_activations"]
        return (
            bundle["assignments"].dtype == torch.long
            and bundle["cluster_sizes"].dtype == torch.long
            and bundle["assignments"].numel() == expected
            and bundle["max_responsibilities"].shape == (expected,)
            and bool((bundle["max_responsibilities"] == 1).all())
            and bundle["model_type"] == "kmeans"
            and Path(bundle["model_path"]).resolve() == _training_artifact_path(run).resolve()
            and bundle.get("subset_spec") == selection["subset_spec"]
            and Path(source["shard_dir"]).resolve() == root.resolve()
            and source["layer"] == run["dataset"]["layer"]
            and source["drop_prefix"] == selection["drop_prefix"]
            and source["num_items"] == expected
        )
    except (EOFError, KeyError, OSError, TypeError, RuntimeError, ValueError):
        return False


def _evaluation_artifact_valid(run: Mapping[str, Any]) -> bool:
    path = Path(run["run_dir"]) / "metrics.json"
    if not path.is_file():
        return False
    try:
        metrics = json.loads(path.read_text())
    except (OSError, TypeError, json.JSONDecodeError):
        return False
    return _evaluation_metrics_valid(run, metrics)


def _evaluation_metrics_valid(run: Mapping[str, Any], metrics: Any) -> bool:
    """Validate evaluation metrics before publishing them or reusing a report."""
    if not isinstance(metrics, Mapping):
        return False
    if run["evaluation"].get("heldout_distribution_coverage", True):
        if not coverage_report_valid(
            metrics.get("heldout_distribution_coverage"),
            components=run["training"]["arguments"]["K"],
        ):
            return False
        source = metrics["heldout_distribution_coverage"]["source"]
        root, _ = split_shard_dir_spec(run["dataset"]["shard_dir"])
        if (Path(source["shard_dir"]).resolve() != (root / "test").resolve()
                or source["layer"] != run["dataset"]["layer"]):
            return False
    alignment = metrics.get("tangent_alignment")
    basis = "pca" if run["training"]["model_kind"] == "kmeans" else "covariance"
    if not isinstance(alignment, Mapping) or alignment.get("definition") != (
        f"leading_intrinsic_dim_{basis}_subspace_principal_angles"
    ):
        return False
    for metric_name in ("tangent_alignment", "tangent_containment"):
        metric = metrics.get(metric_name)
        if not isinstance(metric, Mapping) or (
            metric.get("rank_requirement") != "effective_rank_gte_intrinsic_dim"
        ):
            return False
    if metrics["tangent_containment"].get("definition") != (
        f"best_intrinsic_dim_subset_of_leading_rank_{basis}_principal_angles"
    ):
        return False
    partial_containment = metrics.get("tangent_partial_containment")
    if not isinstance(partial_containment, Mapping) or (
        partial_containment.get("definition")
        != f"leading_{basis}_subspace_within_tangent_principal_angles"
        or partial_containment.get("rank_requirement")
        != "effective_rank_gt_zero_lt_intrinsic_dim"
        or partial_containment.get("normalization") != "effective_rank"
    ):
        return False
    adjusted_alignment = metrics.get("tangent_adjusted_alignment")
    if not isinstance(adjusted_alignment, Mapping) or (
        adjusted_alignment.get("definition")
        != f"leading_min_intrinsic_effective_rank_{basis}_subspace_overlap"
        or adjusted_alignment.get("rank_requirement") != "effective_rank_gte_zero"
        or adjusted_alignment.get("normalization") != "intrinsic_dim"
        or adjusted_alignment.get("zero_rank") != "zero_if_tangent_defined"
        or adjusted_alignment.get("aggregation") != "unweighted_component_mean"
    ):
        return False
    if run["training"]["model_kind"] == "kmeans":
        quantization = metrics.get("quantization")
        if not isinstance(quantization, Mapping) or quantization.get("convention") != "lower_is_better":
            return False
        for name in ("train", "validation"):
            values = quantization.get(name)
            if not isinstance(values, Mapping) or type(values.get("n")) is not int or values["n"] < 0:
                return False
            if values["n"] == 0:
                if name == "train":
                    return False
                continue
            for key in ("sum_squared_distance", "mean_squared_distance"):
                value = values.get(key)
                if not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
                    return False
        geometry = metrics.get("pca_geometry")
        if not isinstance(geometry, Mapping) or geometry.get("version") != 1:
            return False
        counts = [geometry.get(name) for name in ("eligible_components", "excluded_components")]
        if any(type(count) is not int or count < 0 for count in counts):
            return False
        return (
            metrics.get("schema_version") == 2
            and metrics.get("model_kind") == "kmeans"
            and geometry.get("minimum_cluster_points") == 2
            and sum(counts) == run["training"]["arguments"]["K"]
            and metrics.get("evaluation") == run["evaluation"]["kind"]
            and metrics.get("identity_hash") == run["identity_hash"]
            and metrics.get("rank", {}).get("definition") == "kmeans_component_ranks"
            and metrics.get("ambient_rank", {}).get("definition") == "kmeans_component_ranks"
            and "nll" not in metrics and "bic" not in metrics
        )
    if run["training"]["model_kind"] == "hddc":
        for rank_name in ("rank", "ambient_rank"):
            rank = metrics.get(rank_name)
            if not isinstance(rank, Mapping) or (
                rank.get("definition") != "hddc_rank_mask_count"
                or "threshold" in rank
            ):
                return False
    bic_metrics = metrics.get("bic")
    if not isinstance(bic_metrics, Mapping):
        return False
    bic_value = bic_metrics.get("value")
    return (
        metrics.get("schema_version") == 2
        and bic_metrics.get("convention") == "higher_is_better"
        and bic_metrics.get("formula") == "-standard_bic / n + active_components"
        and isinstance(bic_value, (int, float))
        and math.isfinite(float(bic_value))
        and metrics.get("evaluation") == run["evaluation"]["kind"]
        and metrics.get("identity_hash") == run["identity_hash"]
    )


def _assignment_command(run: Mapping[str, Any], path: Path) -> list[str]:
    cfg = run["assignments"]
    command = [
        sys.executable,
        "-m",
        "dalg.cli.run_metrics",
        "assignments",
        "--data-dir",
        run["run_dir"],
        "--shard-dir",
        run["dataset"]["shard_dir"],
        "--layer",
        str(run["dataset"]["layer"]),
        "--batch-size",
        str(cfg["batch_size"]),
        "--device",
        str(cfg["device"]),
        "--seed",
        str(cfg["seed"]),
        "--model-type",
        "mfa" if run["training"]["model_kind"] == "ard" else run["training"]["model_kind"],
        "--save-path",
        str(path),
    ]
    if not cfg["use_inference_cache"]:
        command.append("--no-inference-cache")
    drop_prefix = run["training"]["arguments"].get("drop_prefix")
    if drop_prefix is not None:
        command.extend(["--drop-prefix", str(drop_prefix)])
    return command


def _ensure_run_spec(run: Mapping[str, Any]) -> Path:
    run_dir = Path(run["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "run_spec.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if (
            saved.get("identity_hash") != run["identity_hash"]
            or saved.get("identity") != run["identity"]
        ):
            raise PipelineConfigError(
                f"run directory belongs to a different configuration: {run_dir}"
            )
        return path
    _write_json_atomic(path, run)
    return path


def _mark_stage(run_dir: Path, stage: str, artifact: Path | None = None) -> None:
    payload: dict[str, Any] = {"stage": stage, "completed": True}
    if artifact is not None:
        payload["artifact"] = str(artifact)
    slurm_job = os.environ.get("SLURM_JOB_ID")
    slurm_task = os.environ.get("SLURM_ARRAY_TASK_ID")
    if slurm_job:
        payload["slurm_job_id"] = slurm_job
    if slurm_task:
        payload["slurm_array_task_id"] = slurm_task
    _write_json_atomic(run_dir / f"{stage.upper()}_COMPLETED.json", payload)


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


def _ensure_initialization(run: Mapping[str, Any]) -> None:
    _check_initialization_version(run)
    directory = Path(run["run_dir"]) / "initialization"
    if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
        if not _initialization_artifact_valid(run):
            raise RuntimeError(f"refusing to overwrite invalid initialization artifact: {directory}")
        print(f"[{run['run_id']}] initialization artifact is complete; skipping initialization")
    else:
        print(f"[{run['run_id']}] initialization", flush=True)
        _run_command(_initialization_command(run))
        _validate_initialization(run)
    _mark_stage(Path(run["run_dir"]), "initialization", directory / "kmeans_model.pt")


def execute_run(run: Mapping[str, Any]) -> Path:
    """Execute one manifest row, resuming at the first incomplete stage."""
    if run["evaluation"]["enabled"] and run["evaluation"].get("heldout_distribution_coverage", True):
        validate_toy_test_split(run["dataset"]["shard_dir"], layer=run["dataset"]["layer"])
    if "initialization" in run:
        _check_initialization_version(run)
    if run["training"]["model_kind"] == "kmeans":
        arguments = run["training"]["arguments"]
        if "pca_method" in arguments or arguments.get("pca_purpose") != "geometry":
            raise PipelineConfigError("incompatible KMeans PCA manifest; plan a new run with cluster geometry")
    _ensure_run_spec(run)
    run_dir = Path(run["run_dir"])
    if "initialization" in run:
        _ensure_initialization(run)

    if _training_artifacts_valid(run):
        print(f"[{run['run_id']}] training artifact is complete; skipping training")
    else:
        model_path = _training_artifact_path(run)
        if run["training"]["model_kind"] == "kmeans" and model_path.exists():
            raise RuntimeError(f"refusing to overwrite invalid model artifact: {model_path}")
        print(f"[{run['run_id']}] training", flush=True)
        _run_command(_training_command(run))
        if not _training_artifacts_valid(run):
            raise RuntimeError("training command finished without valid final model artifacts")
    _mark_stage(run_dir, "training", _training_artifact_path(run))

    assignments_path = _assignments_path(run)
    if run["assignments"]["enabled"]:
        if _assignment_artifact_valid(run):
            print(f"[{run['run_id']}] assignment artifact is complete; skipping assignments")
        else:
            if assignments_path.exists():
                raise RuntimeError(
                    f"refusing to overwrite invalid assignment artifact: {assignments_path}"
                )
            print(f"[{run['run_id']}] assignments", flush=True)
            _run_command(_assignment_command(run, assignments_path))
            if not _assignment_artifact_valid(run):
                raise RuntimeError("assignment command finished without a valid bundle")
        _mark_stage(run_dir, "assignments", assignments_path)

    metrics_path = run_dir / "metrics.json"
    if run["evaluation"]["enabled"]:
        if _evaluation_artifact_valid(run):
            print(f"[{run['run_id']}] evaluation artifact is complete; skipping evaluation")
        else:
            if metrics_path.exists():
                raise RuntimeError(
                    f"refusing to overwrite invalid evaluation artifact: {metrics_path}; "
                    "use dalg-run-pipeline evaluate --manifest <manifest> to reevaluate"
                )
            print(f"[{run['run_id']}] evaluation", flush=True)
            from dalg.evaluation.toy_manifold_tiling import evaluate_pipeline_run

            metrics = evaluate_pipeline_run(run)
            if not _evaluation_metrics_valid(run, metrics):
                raise ValueError(f"evaluator produced invalid metrics for {run_dir}")
            json.dumps(metrics, allow_nan=False)
            _write_json_atomic(metrics_path, metrics)
        _mark_stage(run_dir, "evaluation", metrics_path)

    _mark_stage(run_dir, "pipeline", metrics_path if metrics_path.exists() else None)
    print(f"[{run['run_id']}] pipeline complete: {run_dir}", flush=True)
    return run_dir


def pipeline_status(runs: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        run_dir = Path(run["run_dir"])
        initialization_complete = _initialization_artifact_valid(run)
        training_complete = _training_artifacts_valid(run)
        assignments_complete = (
            not run["assignments"]["enabled"]
            or _assignment_artifact_valid(run)
        )
        evaluation_complete = (
            not run["evaluation"]["enabled"] or _evaluation_artifact_valid(run)
        )
        rows.append(
            {
                "run_id": run["run_id"],
                "initialization": initialization_complete,
                "training": training_complete,
                "assignments": assignments_complete,
                "evaluation": evaluation_complete,
                "pipeline": (
                    initialization_complete
                    and training_complete
                    and assignments_complete
                    and evaluation_complete
                    and (run_dir / "PIPELINE_COMPLETED.json").is_file()
                ),
                "run_dir": str(run_dir),
            }
        )
    return rows


def group_by_resources(runs: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        key = _canonical_json(run["resources"])
        groups.setdefault(key, []).append(run)
    return list(groups.values())


def sbatch_command(
    manifest_path: Path,
    runs: list[dict[str, Any]],
    *,
    worker_path: Path,
    log_dir: Path | None = None,
    create_log_dir: bool = True,
) -> list[str]:
    resources = runs[0]["resources"]
    count = len(runs)
    array = f"0-{count - 1}%{resources['max_parallel']}"
    experiment = _slug(runs[0]["identity"]["experiment"])
    log_dir = log_dir if log_dir is not None else REPO_ROOT / "logs" / "experiments" / experiment
    if create_log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "sbatch",
        "--parsable",
        f"--nodes={resources['nodes']}",
        f"--ntasks-per-node={resources['ntasks_per_node']}",
        f"--cpus-per-task={resources['cpus_per_task']}",
        f"--mem={resources['memory']}",
        f"--time={resources['time']}",
        f"--array={array}",
        f"--job-name=dalg-{experiment}",
        f"--output={log_dir}/pipeline_%A_%a.out",
    ]
    if resources.get("partition"):
        command.append(f"--partition={resources['partition']}")
    if resources.get("account"):
        command.append(f"--account={resources['account']}")
    if int(resources["gpus"]) > 0:
        gpu_type = str(resources.get("gpu_type") or "").strip()
        gres = f"gpu:{gpu_type}:{resources['gpus']}" if gpu_type else f"gpu:{resources['gpus']}"
        command.append(f"--gres={gres}")
    command.extend([str(worker_path), str(manifest_path)])
    return command
