"""CLI entrypoint for MFA training on activation shards.

Training modes:
- vanilla: one process, one full MFA model.
- component_shard: N processes, each owns a slice of the MFA components.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from datetime import timedelta
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader


# Dataset setup


def _resolve_activation_data(args, *, log) -> dict:
    """Read shard metadata and build the train/validation row split."""
    from dalg.data.shard_activations import (
        load_meta_index,
        per_subset_counts,
        stratified_split,
    )

    shard_dir_arg = getattr(args, "shard_dir", None)
    if shard_dir_arg is None:
        raise SystemExit("train: --shard-dir is required")
    if getattr(args, "layer", None) is None:
        raise SystemExit("train: --layer is required")

    val_frac = getattr(args, "val_frac", 0.05)
    split_seed = getattr(args, "split_seed", 42)

    shard_dir = Path(shard_dir_arg)
    extract_cfg = json.loads((shard_dir / "config.json").read_text())
    window = int(extract_cfg["window"])
    d_model = int(extract_cfg["d_model"])
    drop_prefix = int(extract_cfg.get("drop_prefix", 32))
    per_row_tokens = window - drop_prefix
    if per_row_tokens <= 0:
        raise SystemExit(
            f"drop_prefix={drop_prefix} leaves no trainable tokens for window={window}"
        )

    meta_index = load_meta_index(shard_dir, layer=args.layer)
    train_pos_full, val_pos = stratified_split(
        meta_index,
        val_frac=val_frac,
        seed=split_seed,
    )
    n_train_tokens = len(train_pos_full) * per_row_tokens

    out_dir_arg = getattr(args, "out_dir", None)
    if out_dir_arg:
        out_dir = Path(out_dir_arg)
    elif getattr(args, "training_mode", "vanilla") == "vae":
        out_dir = shard_dir / f"layer{args.layer:02d}_vae"
    else:
        out_dir = shard_dir / f"layer{args.layer:02d}_{args.K}_mfa"

    log(f"shard_dir={shard_dir}  layer={args.layer}  out_dir={out_dir}")
    log(f"window={window}  d_model={d_model}  drop_prefix={drop_prefix}")
    log(f"split: train rows={len(train_pos_full):,}  val rows={len(val_pos):,}")
    log(f"       train tokens~={n_train_tokens:,}  val tokens~={len(val_pos) * per_row_tokens:,}")

    return {
        "out_dir": out_dir,
        "shard_dir": shard_dir,
        "layer": args.layer,
        "window": window,
        "d_model": d_model,
        "drop_prefix": drop_prefix,
        "n_train_tokens": n_train_tokens,
        "meta_index": meta_index,
        "train_pos_full": train_pos_full,
        "val_pos": val_pos,
        "val_frac": val_frac,
        "split_seed": split_seed,
        "train_counts": per_subset_counts(meta_index, train_pos_full),
        "val_counts": per_subset_counts(meta_index, val_pos),
    }


def _loader_num_workers(args) -> int:
    return max(0, int(getattr(args, "num_workers", 0)))


def _parse_dims(value: str | None, *, default: tuple[int, ...]) -> tuple[int, ...]:
    if value is None:
        return default
    dims = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not dims:
        return default
    if any(dim <= 0 for dim in dims):
        raise SystemExit(f"hidden dimensions must be positive, got {dims}")
    return dims


def _build_data_loader(dataset, args, *, device: str):
    # num_workers: N subprocesses prefetch batches in parallel (0 = load in main process).
    # persistent_workers: keep those workers alive across epochs so we don't re-spawn and
    # re-open shard files every epoch; only meaningful when num_workers > 0.
    num_workers = _loader_num_workers(args)
    return DataLoader(
        dataset,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=(device != "cpu"),
        persistent_workers=(num_workers > 0),
    )


def _build_train_loader(
    data: dict,
    args,
    *,
    device: str,
):
    """Build the train loader and return (loader, steps_per_epoch, row_positions)."""
    from dalg.data.shard_activations import ActivationBatchDataset

    train_pos = data["train_pos_full"]
    train_ds = ActivationBatchDataset(
        data["shard_dir"],
        layer=data["layer"],
        row_subset=train_pos,
        batch_size=args.batch_size,
        drop_prefix=data["drop_prefix"],
        shuffle_shards=True,
        shuffle_within_shard=True,
        seed=(args.seed or 0),
    )
    return _build_data_loader(train_ds, args, device=device), len(train_ds), train_pos


def _build_val_loader(data: dict, args, *, device: str):
    """Build a deterministic val DataLoader (no shuffling).

    With shuffles disabled, ``ActivationBatchDataset`` only partitions shards
    by DataLoader ``worker_id``, never by distributed rank. Independent loaders
    on every rank therefore yield the same batch sequence — exactly what we
    need for component-sharded validation, where every rank must call
    ``model.nll`` on the same batch so the distributed logsumexp inside
    ``ComponentShardedMFA.log_prob`` completes symmetrically.
    """
    from dalg.data.shard_activations import ActivationBatchDataset

    if not data["val_pos"]:
        return None
    val_ds = ActivationBatchDataset(
        data["shard_dir"],
        layer=data["layer"],
        row_subset=data["val_pos"],
        batch_size=args.batch_size,
        drop_prefix=data["drop_prefix"],
        shuffle_shards=False,
        shuffle_within_shard=False,
        seed=(args.seed or 0),
    )
    return _build_data_loader(val_ds, args, device=device)


@torch.no_grad()
def _materialize_val_tensor(
    shard_dir,
    val_pos,
    *,
    layer: int,
    batch_size: int,
    drop_prefix: int,
    val_on_gpu: bool,
    seed: int,
    device: str,
    num_workers: int,
) -> Optional[torch.Tensor]:
    """Stream validation rows into one tensor. Rank 0 only.

    The result is a single contiguous tensor holding the entire validation
    split, parked on the training device (when `val_on_gpu`) or in pinned CPU
    memory. It is passed to `train_nll` as `val_tensor`, which is the fast
    eval path: chunked iteration over an in-memory tensor avoids per-epoch
    shard I/O and DataLoader worker startup. `train_nll` also accepts a
    `val_loader` for cases where the val set is too large to materialize,
    but the shard-training CLI uses this materialized path whenever a val
    split exists.
    """
    import time

    from dalg.data.shard_activations import ActivationBatchDataset

    on_gpu = val_on_gpu and device != "cpu"
    where = f"{device} memory" if on_gpu else "pinned CPU memory"
    print(f"[val] streaming {len(val_pos):,} rows into {where}...")
    t0 = time.time()

    val_ds = ActivationBatchDataset(
        shard_dir,
        layer=layer,
        row_subset=val_pos,
        batch_size=batch_size,
        drop_prefix=drop_prefix,
        shuffle_shards=False,
        shuffle_within_shard=False,
        seed=seed,
        dtype=torch.float16,
    )
    val_prefetch = DataLoader(
        val_ds,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=(device != "cpu" and not on_gpu),
        persistent_workers=(num_workers > 0),
    )

    chunks = []
    for xb in val_prefetch:
        if on_gpu:
            xb = xb.to(device, non_blocking=True)
        chunks.append(xb)

    if not chunks:
        print(f"[val] skipped: empty validation split in {time.time() - t0:.1f}s")
        return None

    val_tensor = torch.cat(chunks, dim=0).contiguous()
    if not on_gpu and device != "cpu":
        val_tensor = val_tensor.pin_memory()
    print(
        f"[val] done: shape={tuple(val_tensor.shape)} dtype={val_tensor.dtype} "
        f"on {val_tensor.device} in {time.time() - t0:.1f}s"
    )
    return val_tensor


def _build_val_tensor_for_main(
    data: dict,
    args,
    *,
    device: str,
) -> Optional[torch.Tensor]:
    if not data["val_pos"]:
        return None
    return _materialize_val_tensor(
        data["shard_dir"],
        data["val_pos"],
        layer=data["layer"],
        batch_size=args.batch_size,
        drop_prefix=data["drop_prefix"],
        val_on_gpu=bool(args.val_on_gpu),
        seed=(args.seed or 0),
        device=device,
        num_workers=_loader_num_workers(args),
    )


def _fit_and_save_centroids(
    centroids_path: Path,
    data: dict,
    args,
    *,
    device: str,
) -> None:
    """Fit ReservoirKMeans from streamed activation shards."""
    from dalg.data.shard_activations import ActivationBatchDataset
    from dalg.init.projected_knn import ReservoirKMeans

    activation_ds = ActivationBatchDataset(
        data["shard_dir"],
        layer=data["layer"],
        row_subset=data["train_pos_full"],
        batch_size=args.batch_size,
        drop_prefix=data["drop_prefix"],
        shuffle_shards=True,
        shuffle_within_shard=True,
        seed=(args.seed or 0),
    )
    activation_loader = _build_data_loader(activation_ds, args, device=device)

    pool_base = data["n_train_tokens"]
    pool_size = args.pool_size
    max_pool = args.max_pool_size or 2_000_000
    if pool_size is not None and pool_size > 0:
        pool_size = int(pool_size)
    else:
        pool_size = min(max(args.K * 2, pool_base // 5), max_pool)
    pool_size = min(pool_size, pool_base)
    print(f"Reservoir pool_size: {pool_size:,} (n_train_tokens={pool_base:,})")

    if args.seed is not None:
        torch.manual_seed(args.seed)

    knn = ReservoirKMeans(
        n_clusters=args.K,
        pool_size=pool_size,
        vocab_size=args.vocab_size,
        device=device,
        proj_dim=args.proj_dim,
        seed=args.seed,
    )
    centroids = knn.fit(
        activation_loader,
        token_loader=None,
        refine_epochs=args.refine_epochs,
    )
    torch.save(centroids.cpu(), centroids_path)
    print(f"Centroids: {tuple(centroids.shape)} saved to {centroids_path}")


def _ensure_centroids(
    data: dict,
    args,
    *,
    out_dir: Path,
    is_main: bool,
    device: str,
    barrier: bool,
) -> torch.Tensor:
    """Fit missing centroids on rank 0, then load them on every rank."""
    centroids_path = out_dir / "centroids.pt"
    if is_main and not centroids_path.exists():
        _fit_and_save_centroids(
            centroids_path,
            data,
            args,
            device=device,
        )
    if barrier and dist.is_available() and dist.is_initialized():
        dist.barrier()
    centroids = torch.load(centroids_path, map_location=device, weights_only=True)
    if centroids.shape[0] != args.K:
        raise SystemExit(
            f"Cached centroids K={centroids.shape[0]} != --K {args.K}; "
            f"delete {centroids_path} to recompute."
        )
    return centroids


def _write_split_info(
    data: dict,
    out_dir: Path,
    *,
    args,
    training_mode: str,
    world_size: int,
) -> None:
    """Persist train/validation composition for downstream analysis."""
    split_info = {
        "seed": data["split_seed"],
        "val_frac": data["val_frac"],
        "per_row_tokens": data["window"] - data["drop_prefix"],
        "train_rows": len(data["train_pos_full"]),
        "val_rows": len(data["val_pos"]),
        "train_per_subset": data["train_counts"],
        "val_per_subset": data["val_counts"],
        "val_global_rows": [data["meta_index"][p]["global_row"] for p in data["val_pos"]],
        "world_size": world_size,
        "training_mode": training_mode,
        "component_shard": training_mode == "component_shard",
    }
    (out_dir / "val_indices.json").write_text(json.dumps(split_info, indent=2))


def _write_run_config(
    data: dict,
    out_dir: Path,
    *,
    args,
    training_mode: str,
    world_size: int,
) -> None:
    """Persist the run config for downstream tools."""
    cfg = {
        "K": args.K,
        "rank": args.rank,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "num_workers": _loader_num_workers(args),
        "training_mode": training_mode,
        "world_size": world_size,
        "shard_dir": str(data["shard_dir"]),
        "layer": args.layer,
        "window": data["window"],
        "d_model": data["d_model"],
        "drop_prefix": data["drop_prefix"],
        "val_frac": data["val_frac"],
        "split_seed": data["split_seed"],
        "pool_size": args.pool_size,
        "refine_epochs": args.refine_epochs,
    }
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))


def _write_vae_run_config(
    data: dict,
    out_dir: Path,
    *,
    args,
) -> None:
    """Persist VAE training settings next to the saved model."""
    cfg = {
        "training_mode": "vae",
        "world_size": 1,
        "shard_dir": str(data["shard_dir"]),
        "layer": args.layer,
        "window": data["window"],
        "d_model": data["d_model"],
        "drop_prefix": data["drop_prefix"],
        "val_frac": data["val_frac"],
        "split_seed": data["split_seed"],
        "epochs": args.epochs,
        "max_steps": args.max_steps,
        "lr": args.lr,
        "weight_decay": args.vae_weight_decay,
        "grad_clip": args.grad_clip,
        "batch_size": args.batch_size,
        "num_workers": _loader_num_workers(args),
        "latent_dim": args.vae_latent_dim,
        "enc_hidden_dims": list(_parse_dims(args.vae_enc_hidden_dims, default=(1024, 512))),
        "dec_hidden_dims": list(_parse_dims(args.vae_dec_hidden_dims, default=(512, 1024))),
        "prior": args.vae_prior,
        "prior_components": args.vae_prior_components,
        "beta": args.vae_beta,
        "beta_warmup_steps": args.vae_beta_warmup_steps,
        "dropout": args.vae_dropout,
        "layer_norm": args.vae_layer_norm,
    }
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))


def _maybe_init_wandb(args, data: dict, *, training_mode: str, world_size: int, is_main: bool):
    """Initialize a W&B run on rank 0 only. Returns the run, or None.

    The training loop checks `wandb.run` rather than this return value, so the
    rest of the code remains agnostic about whether logging is on.
    """
    if not (getattr(args, "wandb", False) and is_main):
        return None
    import wandb

    run_config = {
        "K": getattr(args, "K", None),
        "rank": getattr(args, "rank", None),
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "grad_clip": args.grad_clip,
        "training_mode": training_mode,
        "world_size": world_size,
        "layer": args.layer,
        "shard_dir": str(data["shard_dir"]),
        "d_model": data["d_model"],
        "window": data["window"],
        "drop_prefix": data["drop_prefix"],
        "n_train_tokens": data["n_train_tokens"],
        "val_frac": data["val_frac"],
    }
    if training_mode == "vae":
        run_config.update({
            "latent_dim": args.vae_latent_dim,
            "prior": args.vae_prior,
            "prior_components": args.vae_prior_components,
            "beta": args.vae_beta,
            "beta_warmup_steps": args.vae_beta_warmup_steps,
            "weight_decay": args.vae_weight_decay,
        })
    return wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        config=run_config,
    )


def _finish_wandb(run) -> None:
    if run is None:
        return
    import wandb
    wandb.finish()


# Top-level training commands


def cmd_train(args):
    """Single-process MFA training on activation shards."""
    from dalg.models.mfa import MFA, save_mfa
    from dalg.models.train import train_nll

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    data = _resolve_activation_data(args, log=print)
    out_dir = data["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    wandb_run = _maybe_init_wandb(
        args, data, training_mode="vanilla", world_size=1, is_main=True,
    )

    train_loader, steps_per_epoch, _ = _build_train_loader(
        data,
        args,
        device=args.device,
    )
    val_tensor = _build_val_tensor_for_main(
        data,
        args,
        device=args.device,
    )
    _write_split_info(
        data,
        out_dir,
        args=args,
        training_mode="vanilla",
        world_size=1,
    )

    centroids = _ensure_centroids(
        data,
        args,
        out_dir=out_dir,
        is_main=True,
        device=args.device,
        barrier=False,
    )

    model = MFA(
        centroids=centroids,
        rank=args.rank,
    ).to(args.device)
    if getattr(args, "compile", False):
        print("Compiling model with torch.compile...")
        model = torch.compile(model)

    train_nll(
        model,
        train_loader,
        val_tensor=val_tensor,
        epochs=args.epochs,
        lr=args.lr,
        grad_clip=args.grad_clip,
        save_path=str(out_dir / "mfa_model.pt"),
        save_func=save_mfa,
        ckpt_path=str(out_dir / "checkpoint.pt"),
        steps_per_epoch=steps_per_epoch,
        track_best=True,
        max_steps=args.max_steps,
    )

    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    save_mfa(raw_model, str(out_dir / "mfa_model.pt"))
    _write_run_config(
        data,
        out_dir,
        args=args,
        training_mode="vanilla",
        world_size=1,
    )
    print(f"Model saved to {out_dir}/mfa_model.pt")
    _finish_wandb(wandb_run)


def cmd_train_vae(args):
    """Single-process VAE training on activation shards."""
    from dalg.models.train import train_vae
    from dalg.models.vae import VAE, build_prior, save_vae

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    if args.seed is not None:
        torch.manual_seed(args.seed)

    data = _resolve_activation_data(args, log=print)
    out_dir = data["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    wandb_run = _maybe_init_wandb(
        args, data, training_mode="vae", world_size=1, is_main=True,
    )

    train_loader, steps_per_epoch, _ = _build_train_loader(
        data,
        args,
        device=args.device,
    )
    val_tensor = _build_val_tensor_for_main(
        data,
        args,
        device=args.device,
    )
    _write_split_info(
        data,
        out_dir,
        args=args,
        training_mode="vae",
        world_size=1,
    )

    enc_hidden_dims = _parse_dims(args.vae_enc_hidden_dims, default=(1024, 512))
    dec_hidden_dims = _parse_dims(args.vae_dec_hidden_dims, default=(512, 1024))
    prior = build_prior(
        args.vae_prior,
        args.vae_latent_dim,
        args.vae_prior_components,
        input_dim=data["d_model"],
    )
    model = VAE(
        input_dim=data["d_model"],
        latent_dim=args.vae_latent_dim,
        enc_hidden_dims=enc_hidden_dims,
        dec_hidden_dims=dec_hidden_dims,
        prior=prior,
        dropout=args.vae_dropout,
        layer_norm=args.vae_layer_norm,
        beta=args.vae_beta,
    ).to(args.device)
    if getattr(args, "compile", False):
        print("Compiling VAE with torch.compile...")
        model = torch.compile(model)

    train_vae(
        model,
        train_loader,
        val_tensor=val_tensor,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.vae_weight_decay,
        grad_clip=args.grad_clip,
        save_path=str(out_dir / "vae_model.pt"),
        save_func=save_vae,
        ckpt_path=str(out_dir / "checkpoint.pt"),
        steps_per_epoch=steps_per_epoch,
        track_best=True,
        max_steps=args.max_steps,
        beta_warmup_steps=args.vae_beta_warmup_steps,
        log_interval=args.log_interval,
    )

    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    save_vae(raw_model, str(out_dir / "vae_model.pt"))
    _write_vae_run_config(data, out_dir, args=args)
    print(f"VAE saved to {out_dir}/vae_model.pt")
    _finish_wandb(wandb_run)


def cmd_train_component_shard(args):
    """Component-sharded MFA training.

    Each rank owns a contiguous slice of the K components. Every rank consumes
    identical activation batches, so rank 0 loads each batch and broadcasts it.
    """
    from dalg.models.mfa import ComponentShardedMFA, save_component_shard
    from dalg.models.train import train_nll

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_main = rank == 0

    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(minutes=60),
        device_id=torch.device(device),
    )
    torch.set_float32_matmul_precision("high")
    if is_main:
        print(f"[component_shard] world_size={world_size} backend=nccl")

    log = print if is_main else (lambda *_a, **_k: None)
    data = _resolve_activation_data(args, log=log)
    out_dir = data["out_dir"]
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    wandb_run = _maybe_init_wandb(
        args, data, training_mode="component_shard", world_size=world_size, is_main=is_main,
    )

    base_loader, steps_per_epoch, train_pos = _build_train_loader(
        data,
        args,
        device=device,
    )
    log(f"each rank sees all {len(train_pos):,} train rows")

    train_loader = _ComponentShardLoader(
        base_loader,
        steps_per_epoch,
        rank,
        world_size,
        device,
    )

    val_loader = _build_val_loader(data, args, device=device)
    if val_loader is not None:
        log("[val] using val_loader on every rank (deterministic, identical batches)")
    else:
        log("[val] no validation rows; best epoch will fall back to train NLL")

    if is_main:
        _write_split_info(
            data,
            out_dir,
            args=args,
            training_mode="component_shard",
            world_size=world_size,
        )

    centroids = _ensure_centroids(
        data,
        args,
        out_dir=out_dir,
        is_main=is_main,
        device=device,
        barrier=True,
    )
    log(f"Loaded centroids: {tuple(centroids.shape)}")

    model = ComponentShardedMFA.from_global_centroids(
        centroids,
        rank=args.rank,
        dist_rank=rank,
        world_size=world_size,
    ).to(device)
    log(
        f"Component sharding: rank {rank}/{world_size} owns "
        f"[{model.component_start}, {model.component_end})"
    )
    if getattr(args, "compile", False):
        log("Compiling model with torch.compile...")
        model = torch.compile(model)

    ckpt_path = str(out_dir / f"checkpoint_rank{rank:04d}.pt")
    if is_main:
        ckpt_manifest = {
            "format": "component_sharded_checkpoint",
            "global_K": args.K,
            "rank": args.rank,
            "world_size": world_size,
            "checkpoints": [f"checkpoint_rank{r:04d}.pt" for r in range(world_size)],
        }
        (out_dir / "checkpoint_shards.json").write_text(json.dumps(ckpt_manifest, indent=2))
    dist.barrier()

    train_nll(
        model,
        train_loader,
        val_loader=val_loader,
        val_tensor=None,
        epochs=args.epochs,
        lr=args.lr,
        grad_clip=args.grad_clip,
        save_path=None,
        save_func=None,
        ckpt_path=ckpt_path,
        steps_per_epoch=steps_per_epoch,
        track_best=True,
        checkpoint_all_ranks=True,
        max_steps=args.max_steps,
    )

    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    shard_path = out_dir / f"mfa_model_rank{rank:04d}.pt"
    save_component_shard(raw_model, shard_path)
    dist.barrier()
    if is_main:
        manifest = {
            "format": "component_sharded_mfa",
            "global_K": args.K,
            "rank": args.rank,
            "world_size": world_size,
            "shards": [f"mfa_model_rank{r:04d}.pt" for r in range(world_size)],
        }
        (out_dir / "mfa_model_shards.json").write_text(json.dumps(manifest, indent=2))
        _write_run_config(
            data,
            out_dir,
            args=args,
            training_mode="component_shard",
            world_size=world_size,
        )
        print(f"Component-sharded model shards saved to {out_dir}")

    dist.barrier()
    _finish_wandb(wandb_run)
    dist.destroy_process_group()


class _ComponentShardLoader:
    """Rank 0 loads batches and broadcasts them; all ranks consume identical data."""

    def __init__(self, loader, steps, rank, world_size, device):
        self.loader = loader
        self.steps = steps
        self.rank = rank
        self.world_size = world_size
        self.device = device

    def __len__(self):
        return self.steps

    def __iter__(self):
        shape = torch.zeros(2, dtype=torch.long, device=self.device)
        if self.rank == 0:
            for batch in itertools.islice(self.loader, self.steps):
                batch = batch.to(self.device)
                shape[0], shape[1] = batch.shape
                dist.broadcast(shape, src=0)
                dist.broadcast(batch.contiguous(), src=0)
                yield batch
        else:
            for _ in range(self.steps):
                dist.broadcast(shape, src=0)
                batch = torch.empty(
                    int(shape[0]),
                    int(shape[1]),
                    dtype=torch.float32,
                    device=self.device,
                )
                dist.broadcast(batch, src=0)
                yield batch


# CLI parsing / dispatch


def validate_args(args) -> None:
    """Validate CLI arguments and the requested training mode."""
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if args.layer is None:
        raise SystemExit("train: --layer is required")

    mode = args.training_mode
    if mode in {"vanilla", "component_shard"} and args.K is None:
        raise SystemExit(f"train: --K is required for --training-mode {mode}")
    if mode == "vanilla" and world_size > 1:
        raise SystemExit(
            "train: --training-mode vanilla was requested under torchrun; "
            "run a single process or use --training-mode component_shard"
        )
    if mode == "vae" and world_size > 1:
        raise SystemExit("train: --training-mode vae currently supports a single process")
    if mode == "component_shard":
        if world_size <= 1:
            raise SystemExit(f"train: --training-mode {mode} requires torchrun with WORLD_SIZE>1")
        if args.device != "cuda":
            raise SystemExit("train: component_shard requires --device cuda")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train MFA on activation shards")
    p.add_argument("--device", default="cuda", help="Device (cuda/cpu/mps)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--shard-dir", required=True, help="Activation shard root from extract-windows")
    p.add_argument("--layer", type=int, required=True, help="Layer to train on")
    p.add_argument("--out-dir", default=None, help="Where to save centroids/model")
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--val-on-gpu", action="store_true")
    p.add_argument("--K", type=int, default=None, help="Number of MFA components")
    p.add_argument("--rank", type=int, default=10, help="MFA rank (q)")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=None,
                   help="Hard cap on total optimizer steps; if reached, training stops early. "
                        "Useful for bisect/smoke runs.")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--grad-clip", type=float, default=None)
    p.add_argument("--log-interval", type=int, default=100)
    p.add_argument("--proj-dim", type=int, default=32)
    p.add_argument("--refine-epochs", type=int, default=25)
    p.add_argument("--vocab-size", type=int, default=50257)
    p.add_argument("--pool-size", type=int, default=None)
    p.add_argument("--max-pool-size", type=int, default=2_000_000)
    p.add_argument(
        "--training-mode",
        default="vanilla",
        choices=["vanilla", "component_shard", "vae"],
    )
    p.add_argument("--vae-latent-dim", type=int, default=64)
    p.add_argument("--vae-enc-hidden-dims", default="1024,512")
    p.add_argument("--vae-dec-hidden-dims", default="512,1024")
    p.add_argument("--vae-prior", choices=["standard", "mog", "vamp"], default="standard")
    p.add_argument("--vae-prior-components", type=int, default=100)
    p.add_argument("--vae-beta", type=float, default=1.0)
    p.add_argument("--vae-beta-warmup-steps", type=int, default=0)
    p.add_argument("--vae-weight-decay", type=float, default=1e-4)
    p.add_argument("--vae-dropout", type=float, default=0.0)
    p.add_argument("--vae-layer-norm", action="store_true")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--wandb", action="store_true", help="Log training to Weights & Biases (rank 0 only)")
    p.add_argument("--wandb-project", default=None, help="W&B project name")
    p.add_argument("--wandb-name", default=None, help="W&B run name")
    return p


_DISPATCH = {
    "vanilla": cmd_train,
    "component_shard": cmd_train_component_shard,
    "vae": cmd_train_vae,
}


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)
    _DISPATCH[args.training_mode](args)


if __name__ == "__main__":
    main()
