"""Small, manifest-driven training pipeline for DALG experiments.

This module intentionally orchestrates the existing CLIs instead of replacing
their training or analysis logic. A YAML experiment is expanded once into an
immutable JSONL manifest; each manifest row is one independently resumable
train -> assignments -> evaluation run.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
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


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = 1
_TOP_LEVEL_KEYS = {
    "experiment",
    "dataset",
    "model",
    "training",
    "assignments",
    "evaluation",
    "resources",
    "sweep",
}
_MODEL_MODULES = {
    "mfa": "dalg.cli.run_training",
    "ard": "dalg.cli.adaptive_q.run_training_ard",
    "hddc": "dalg.cli.adaptive_q.run_training_hddc",
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
) -> dict[str, Any]:
    parser, validator = _parser_and_validator(model_kind)
    try:
        args = parser.parse_args(_mapping_to_argv(parser, values))
    except SystemExit as exc:
        raise PipelineConfigError(f"invalid {model_kind} training parameters") from exc

    previous_world_size = os.environ.get("WORLD_SIZE")
    os.environ["WORLD_SIZE"] = str(world_size)
    try:
        validator(args)
    except SystemExit as exc:
        raise PipelineConfigError(str(exc)) from exc
    finally:
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


def _resolve_centroids_path(value: Any) -> str | None:
    """Resolve a shared centroid tensor path and reject ambiguous inputs."""
    resolved = _resolve_optional_path(value)
    if not resolved:
        return None
    path = Path(resolved)
    if path.suffix != ".pt":
        raise PipelineConfigError(
            "training.centroids_path must point directly to a .pt file"
        )
    if path.exists() and not path.is_file():
        raise PipelineConfigError(
            f"training.centroids_path must be a file, got: {path}"
        )
    return str(path)


def _validate_centroids(
    path_value: str,
    *,
    expected_k: int,
    shard_dir_arg: str,
) -> None:
    """Fail at planning time when shared centroids cannot initialize the run."""
    path = Path(path_value)
    if not path.is_file():
        raise PipelineConfigError(f"centroids file not found: {path}")
    try:
        centroids = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    except Exception as exc:
        raise PipelineConfigError(f"could not load centroids tensor: {path}: {exc}") from exc
    if not isinstance(centroids, torch.Tensor) or centroids.ndim != 2:
        raise PipelineConfigError(
            f"centroids must be a rank-2 tensor, got {type(centroids).__name__} "
            f"with shape {getattr(centroids, 'shape', None)}"
        )
    if centroids.shape[0] != expected_k:
        raise PipelineConfigError(
            f"centroids K={centroids.shape[0]} does not match model.K={expected_k}: {path}"
        )

    shard_dir, _ = split_shard_dir_spec(shard_dir_arg)
    try:
        shard_config = json.loads((shard_dir / "config.json").read_text())
        expected_dim = int(shard_config["d_model"])
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise PipelineConfigError(
            f"could not read activation dimension from {shard_dir / 'config.json'}"
        ) from exc
    if centroids.shape[1] != expected_dim:
        raise PipelineConfigError(
            f"centroid dimension D={centroids.shape[1]} does not match activation "
            f"d_model={expected_dim}: {path}"
        )


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
    if "centroids_path" in values:
        values["centroids_path"] = _resolve_centroids_path(values["centroids_path"])

    training_mode = str(values.get("training_mode", "vanilla"))
    world_size = resources["gpus"] if training_mode == "component_shard" else 1
    if training_mode == "component_shard" and resources["gpus"] <= 1:
        raise PipelineConfigError(
            "component_shard training requires resources.gpus greater than one"
        )
    if model_kind == "ard" and training_mode == "component_shard":
        raise PipelineConfigError("ARD training does not support component sharding")
    training_args = _parse_training_args(model_kind, values, world_size=world_size)
    training_args["out_dir"] = None
    if check_inputs and training_args.get("centroids_path"):
        _validate_centroids(
            training_args["centroids_path"],
            expected_k=int(training_args["K"]),
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
    if evaluation["enabled"] and evaluation["kind"] != "adaptive_q_toy":
        raise PipelineConfigError(
            "the first pipeline version supports evaluation.kind: adaptive_q_toy"
        )
    if evaluation["kind"] == "adaptive_q_toy" and model_kind not in {"ard", "hddc"}:
        raise PipelineConfigError("adaptive_q_toy evaluation requires an ARD or HDDC model")
    if assignments["seed"] is None:
        assignments["seed"] = training_args.get("seed") or 0

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
    digest = _identity_hash(identity)
    rank = training_args.get("rank")
    run_name = "__".join(
        [
            _slug(model_kind),
            _slug(dataset_id),
            f"l{layer:02d}",
            f"k{int(training_args['K'])}",
            f"q{int(rank)}",
            f"s{int(training_args.get('seed') or 0)}",
            digest[:8],
        ]
    )
    run_dir = output_root / _slug(name) / run_name
    training_args["out_dir"] = str(run_dir)
    return {
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


def _training_artifacts_valid(run: Mapping[str, Any]) -> bool:
    run_dir = Path(run["run_dir"])
    kind = run["training"]["model_kind"]
    if not (run_dir / "config.json").is_file() or not (run_dir / "val_indices.json").is_file():
        return False
    model_path = run_dir / "mfa_model.pt"
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
    model_path = run_dir / "mfa_model.pt"
    if model_path.is_file():
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


def _evaluation_artifact_valid(run: Mapping[str, Any]) -> bool:
    path = Path(run["run_dir"]) / "metrics.json"
    if not path.is_file():
        return False
    try:
        metrics = json.loads(path.read_text())
    except (OSError, TypeError, json.JSONDecodeError):
        return False
    return (
        metrics.get("evaluation") == run["evaluation"]["kind"]
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
        "hddc" if run["training"]["model_kind"] == "hddc" else "mfa",
        "--save-path",
        str(path),
    ]
    if not cfg["use_inference_cache"]:
        command.append("--no-inference-cache")
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


def execute_run(run: Mapping[str, Any]) -> Path:
    """Execute one manifest row, resuming at the first incomplete stage."""
    _ensure_run_spec(run)
    run_dir = Path(run["run_dir"])
    expected_k = int(run["training"]["arguments"]["K"])

    if _training_artifacts_valid(run):
        print(f"[{run['run_id']}] training artifact is complete; skipping training")
    else:
        print(f"[{run['run_id']}] training", flush=True)
        _run_command(_training_command(run))
        if not _training_artifacts_valid(run):
            raise RuntimeError("training command finished without valid final model artifacts")
    _mark_stage(run_dir, "training", _training_artifact_path(run))

    assignments_path = run_dir / "mfa_model_assignments.pt"
    if run["assignments"]["enabled"]:
        if _validate_assignment_bundle(assignments_path, expected_k):
            print(f"[{run['run_id']}] assignment artifact is complete; skipping assignments")
        else:
            if assignments_path.exists():
                raise RuntimeError(
                    f"refusing to overwrite invalid assignment artifact: {assignments_path}"
                )
            print(f"[{run['run_id']}] assignments", flush=True)
            _run_command(_assignment_command(run, assignments_path))
            if not _validate_assignment_bundle(assignments_path, expected_k):
                raise RuntimeError("assignment command finished without a valid bundle")
        _mark_stage(run_dir, "assignments", assignments_path)

    metrics_path = run_dir / "metrics.json"
    if run["evaluation"]["enabled"]:
        if _evaluation_artifact_valid(run):
            print(f"[{run['run_id']}] evaluation artifact is complete; skipping evaluation")
        else:
            if metrics_path.exists():
                raise RuntimeError(
                    f"refusing to overwrite invalid evaluation artifact: {metrics_path}"
                )
            print(f"[{run['run_id']}] evaluation", flush=True)
            if run["evaluation"]["kind"] == "adaptive_q_toy":
                from dalg.analysis.adaptive_q_evaluation import evaluate_adaptive_q_toy

                metrics = evaluate_adaptive_q_toy(
                    run_dir,
                    shard_dir=run["dataset"]["shard_dir"],
                    layer=int(run["dataset"]["layer"]),
                    model_kind=run["training"]["model_kind"],
                    assignments_path=assignments_path,
                    batch_size=int(run["evaluation"]["batch_size"]),
                    device=str(run["evaluation"]["device"]),
                )
            else:
                raise PipelineConfigError(
                    f"unsupported evaluator: {run['evaluation']['kind']!r}"
                )
            metrics["run_id"] = run["run_id"]
            metrics["identity_hash"] = run["identity_hash"]
            _write_json_atomic(metrics_path, metrics)
        _mark_stage(run_dir, "evaluation", metrics_path)

    _mark_stage(run_dir, "pipeline", metrics_path if metrics_path.exists() else None)
    print(f"[{run['run_id']}] pipeline complete: {run_dir}", flush=True)
    return run_dir


def pipeline_status(runs: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        run_dir = Path(run["run_dir"])
        training_complete = _training_artifacts_valid(run)
        assignments_complete = (
            not run["assignments"]["enabled"]
            or _validate_assignment_bundle(
                run_dir / "mfa_model_assignments.pt",
                int(run["training"]["arguments"]["K"]),
            )
        )
        evaluation_complete = (
            not run["evaluation"]["enabled"] or _evaluation_artifact_valid(run)
        )
        rows.append(
            {
                "run_id": run["run_id"],
                "training": training_complete,
                "assignments": assignments_complete,
                "evaluation": evaluation_complete,
                "pipeline": (
                    training_complete
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
) -> list[str]:
    resources = runs[0]["resources"]
    count = len(runs)
    array = f"0-{count - 1}%{resources['max_parallel']}"
    experiment = _slug(runs[0]["identity"]["experiment"])
    log_dir = REPO_ROOT / "logs" / "experiments" / experiment
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
