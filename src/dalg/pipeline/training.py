"""Trainer interfaces, command construction, and final-model validation."""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from dalg.init.activation_selection import resolve_initialization_rows

from .common import PipelineConfigError, _mapping_to_argv, _model_stem


_MODEL_MODULES = {
    "mfa": "dalg.cli.run_training",
    "ard": "dalg.cli.adaptive_q.run_training_ard",
    "hddc": "dalg.cli.adaptive_q.run_training_hddc",
    "kmeans": "dalg.cli.run_training_kmeans",
}


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
