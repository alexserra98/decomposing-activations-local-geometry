"""Resolve normalized configuration into an identified, fully defaulted run."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from dalg.data.subset_spec import split_shard_dir_spec
from dalg.init.activation_selection import resolve_initialization_rows
from dalg.evaluation import validate_evaluation_inputs, validate_evaluation_settings

from .common import REPO_ROOT, PipelineConfigError, _require_mapping, _canonical_json, _slug
from .training import _MODEL_MODULES, _parse_training_args
from .assignments import _ASSIGNMENT_DEFAULTS
from .evaluation import _EVALUATION_DEFAULTS
from .slurm import _normalise_resources


SCHEMA_VERSION = 1


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


def _identity_hash(identity: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(identity).encode()).hexdigest()


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


def normalise_sections(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate section shapes and the pipeline-specific controls."""
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

    return {
        "experiment": experiment,
        "dataset": dataset,
        "model": model,
        "training": training,
        "hard_covariance": hard_covariance,
        "initialization_raw": initialization_raw,
        "pca_neighbors": pca_neighbors,
        "assignments_raw": assignments_raw,
        "evaluation_raw": evaluation_raw,
        "resources": resources,
    }


def resolve_inputs(sections: Mapping[str, Any], *, check_inputs: bool) -> tuple[str, Path, dict[str, Any]]:
    """Resolve experiment location and dataset identity, without fitting anything."""
    experiment = sections["experiment"]
    dataset = sections["dataset"]
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

    dataset_spec = {
        "id": dataset_id,
        "shard_dir": shard_dir,
        "layer": layer,
    }
    return name, output_root, dataset_spec


def resolve_training(
    sections: Mapping[str, Any], dataset: Mapping[str, Any], *, check_inputs: bool,
) -> tuple[str, dict[str, Any], dict[str, Any] | None]:
    """Resolve trainer arguments and optional generated-initialization settings."""
    model = sections["model"]
    training = sections["training"]
    resources = sections["resources"]
    hard_covariance = sections["hard_covariance"]
    initialization_raw = sections["initialization_raw"]
    pca_neighbors = sections["pca_neighbors"]
    shard_dir, layer = dataset["shard_dir"], dataset["layer"]
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

    return model_kind, training_args, initialization


def resolve_stages(
    sections: Mapping[str, Any], dataset: Mapping[str, Any],
    model_kind: str, training_args: Mapping[str, Any], *, check_inputs: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fill assignment/evaluation defaults and check their dependencies and inputs."""
    assignments_raw = sections["assignments_raw"]
    evaluation_raw = sections["evaluation_raw"]
    shard_dir, layer = dataset["shard_dir"], dataset["layer"]
    assignments = _normalise_stage_config(
        assignments_raw,
        _ASSIGNMENT_DEFAULTS,
        name="assignments",
    )
    evaluation_defaults = dict(_EVALUATION_DEFAULTS)
    if evaluation_raw.get("kind") == "unsupervised":
        evaluation_defaults["test_shard_dir"] = None
    evaluation = _normalise_stage_config(evaluation_raw, evaluation_defaults, name="evaluation")
    assignments["enabled"] = bool(assignments["enabled"])
    evaluation["enabled"] = bool(evaluation["enabled"])
    if evaluation["enabled"] and not assignments["enabled"]:
        raise PipelineConfigError("evaluation requires assignments.enabled: true")
    if evaluation.get("test_shard_dir") is not None:
        evaluation["test_shard_dir"] = _resolve_shard_dir(str(evaluation["test_shard_dir"]), None)
    try:
        validate_evaluation_settings(evaluation, model_kind=model_kind)
    except ValueError as exc:
        raise PipelineConfigError(str(exc)) from exc
    if assignments["seed"] is None:
        assignments["seed"] = training_args.get("seed") or 0
    if check_inputs and evaluation["enabled"] and evaluation["heldout_distribution_coverage"]:
        validate_evaluation_inputs(
            shard_dir, layer=layer, model_kind=model_kind, evaluation=evaluation,
            drop_prefix=training_args.get("drop_prefix"),
        )

    return assignments, evaluation


def build_run(
    *, name: str, output_root: Path, dataset_spec: dict[str, Any],
    resources: dict[str, Any], model_kind: str, training_args: dict[str, Any],
    initialization: dict[str, Any] | None, assignments: dict[str, Any],
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    """Hash configuration identity before injecting derived artifact paths."""
    dataset_id, layer = dataset_spec["id"], dataset_spec["layer"]
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

def map_run_to_manifest(config: Mapping[str, Any], *, check_inputs: bool = True) -> dict[str, Any]:
    """Turn one expanded configuration into a fully specified manifest row."""
    sections = normalise_sections(config)
    name, output_root, dataset = resolve_inputs(sections, check_inputs=check_inputs)
    model_kind, training_args, init = resolve_training(
        sections, dataset, check_inputs=check_inputs,
    )
    assignment_config, evaluation_config = resolve_stages(
        sections, dataset, model_kind, training_args, check_inputs=check_inputs,
    )
    return build_run(
        name=name, output_root=output_root, dataset_spec=dataset,
        resources=sections["resources"], model_kind=model_kind,
        training_args=training_args, initialization=init,
        assignments=assignment_config, evaluation=evaluation_config,
    )
