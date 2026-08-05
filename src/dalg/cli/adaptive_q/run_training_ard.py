"""CLI entrypoint for ARD-regularized MFA training on activation shards.

Deliberately redundant with `run_training.py` so the ARD research path can
diverge without touching the production one. Single-process (vanilla) only:
there is no component-sharded ARD variant.

The model is `dalg.models.mfa_ard.MFA_ARD`, whose ARD prior shrinks the columns
of each W_k. `--rank` is therefore the *maximum* rank per component: set it
generously and read the learned q_k off the `q_eff` logs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader


# Dataset setup


def _resolve_activation_data(args, *, log) -> dict:
    """Read shard metadata and build the train/validation row split."""
    from dalg.data.shard_activations import (
        load_meta_index,
        per_subset_counts,
        stratified_split,
    )
    from dalg.data.subset_spec import resolve_spec_positions, split_shard_dir_spec

    shard_dir_arg = getattr(args, "shard_dir", None)
    if shard_dir_arg is None:
        raise SystemExit("train: --shard-dir is required")
    if getattr(args, "layer", None) is None:
        raise SystemExit("train: --layer is required")

    val_frac = getattr(args, "val_frac", 0.05)
    split_seed = getattr(args, "split_seed", 42)

    shard_dir, subset_spec = split_shard_dir_spec(shard_dir_arg)
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
    keep = resolve_spec_positions(
        meta_index, subset_spec, window=window, drop_prefix=drop_prefix
    )
    if subset_spec:
        log(f"subset spec={subset_spec!r}: {len(keep):,}/{len(meta_index):,} rows selected")
    train_pos_full, val_pos = stratified_split(
        meta_index,
        val_frac=val_frac,
        seed=split_seed,
        positions=keep,
    )
    n_train_tokens = len(train_pos_full) * per_row_tokens

    out_dir_arg = getattr(args, "out_dir", None)
    out_dir = Path(out_dir_arg) if out_dir_arg else (
        shard_dir / f"layer{args.layer:02d}_{args.K}_mfa_ard"
    )

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


def _build_data_loader(dataset, args, *, device: str):
    num_workers = _loader_num_workers(args)
    return DataLoader(
        dataset,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=(device != "cpu"),
        persistent_workers=(num_workers > 0),
    )


def _build_train_loader(data: dict, args, *, device: str):
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


def _limit_steps_per_epoch(steps_per_epoch: int, args, *, log) -> int:
    limit = getattr(args, "steps_per_epoch", None)
    if limit is None:
        return steps_per_epoch
    limited = min(steps_per_epoch, int(limit))
    log(f"[debug] limiting each epoch to {limited:,}/{steps_per_epoch:,} batches")
    return limited


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
    """Stream validation rows into one tensor for fast per-epoch eval."""
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


def _build_val_tensor_for_main(data: dict, args, *, device: str) -> Optional[torch.Tensor]:
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


def _fit_and_save_centroids(centroids_path: Path, data: dict, args, *, device: str) -> None:
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


def _ensure_centroids(data: dict, args, *, out_dir: Path, device: str) -> torch.Tensor:
    """Resolve centroids for the run: cached, provided, or freshly fitted."""
    centroids_path = out_dir / "centroids.pt"
    if not centroids_path.exists():
        provided = getattr(args, "centroids_path", None)
        if provided:
            src = Path(provided)
            if src.is_dir():
                src = src / "centroids.pt"
            if not src.is_file():
                raise SystemExit(f"--centroids-path not found: {provided}")
            import shutil
            shutil.copyfile(src, centroids_path)
            print(f"Centroids: copied from {src} to {centroids_path}")
        else:
            _fit_and_save_centroids(centroids_path, data, args, device=device)
    centroids = torch.load(centroids_path, map_location=device, weights_only=True)
    if centroids.shape[0] != args.K:
        raise SystemExit(
            f"Cached centroids K={centroids.shape[0]} != --K {args.K}; "
            f"delete {centroids_path} to recompute."
        )
    return centroids


def _write_split_info(data: dict, out_dir: Path) -> None:
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
        "world_size": 1,
        "training_mode": "vanilla_ard",
        "component_shard": False,
    }
    (out_dir / "val_indices.json").write_text(json.dumps(split_info, indent=2))


def _write_run_config(data: dict, out_dir: Path, *, args, ard_weight: float) -> None:
    """Persist the run config for downstream tools."""
    cfg = {
        "model": "MFA_ARD",
        "K": args.K,
        "rank": args.rank,
        "alpha0": args.alpha0,
        "b0": args.b0,
        "ard_lambda": args.ard_lambda,
        "ard_weight": ard_weight,
        "ard_warmup_frac": args.ard_warmup_frac,
        "ard_ramp_frac": args.ard_ramp_frac,
        "ard_schedule_epochs": args.ard_schedule_epochs,
        "rank_threshold": args.rank_threshold,
        "pruned_at_end": bool(args.prune_at_end),
        "epochs": args.epochs,
        "early_stop_delta": args.early_stop_delta,
        "steps_per_epoch": args.steps_per_epoch,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "num_workers": _loader_num_workers(args),
        "training_mode": "vanilla_ard",
        "world_size": 1,
        "shard_dir": str(data["shard_dir"]),
        "layer": args.layer,
        "window": data["window"],
        "d_model": data["d_model"],
        "drop_prefix": data["drop_prefix"],
        "n_train_tokens": data["n_train_tokens"],
        "val_frac": data["val_frac"],
        "split_seed": data["split_seed"],
        "pool_size": args.pool_size,
        "refine_epochs": args.refine_epochs,
    }
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))


def _maybe_init_wandb(args, data: dict, *, ard_weight: float):
    """Initialize a W&B run. Returns the run, or None."""
    if not getattr(args, "wandb", False):
        return None
    import wandb

    run_config = {
        "model": "MFA_ARD",
        "K": args.K,
        "rank": args.rank,
        "alpha0": args.alpha0,
        "b0": args.b0,
        "ard_lambda": args.ard_lambda,
        "ard_weight": ard_weight,
        "ard_warmup_frac": args.ard_warmup_frac,
        "ard_ramp_frac": args.ard_ramp_frac,
        "rank_threshold": args.rank_threshold,
        "epochs": args.epochs,
        "early_stop_delta": args.early_stop_delta,
        "steps_per_epoch": args.steps_per_epoch,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "grad_clip": args.grad_clip,
        "training_mode": "vanilla_ard",
        "world_size": 1,
        "layer": args.layer,
        "shard_dir": str(data["shard_dir"]),
        "d_model": data["d_model"],
        "window": data["window"],
        "drop_prefix": data["drop_prefix"],
        "n_train_tokens": data["n_train_tokens"],
        "val_frac": data["val_frac"],
    }
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


# Top-level training command


def cmd_train_ard(args):
    """Single-process ARD-MFA training on activation shards."""
    from dalg.models.adaptive_q.mfa_ard import MFA_ARD, save_mfa_ard
    from dalg.models.adaptive_q.train_ard import train_nll_ard

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    data = _resolve_activation_data(args, log=print)
    out_dir = data["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # The ARD penalty is a prior over parameters — it applies once per dataset,
    # while the training loss is a per-sample mean. Hence the 1/N_train scale;
    # --ard-lambda is the knob for dialing the pressure up or down.
    n_train = max(1, int(data["n_train_tokens"]))
    ard_weight = args.ard_lambda / n_train
    log_coeff = data["d_model"] / 2.0 + args.alpha0 - 1.0
    print(
        f"[ard] alpha0={args.alpha0}  b0={args.b0}  lambda={args.ard_lambda}  "
        f"n_train={n_train:,}  ard_weight={ard_weight:.6g}  "
        f"log_coeff=D/2+alpha0-1={log_coeff:.4f}  rank_threshold={args.rank_threshold}"
    )

    wandb_run = _maybe_init_wandb(args, data, ard_weight=ard_weight)

    train_loader, steps_per_epoch, _ = _build_train_loader(data, args, device=args.device)
    steps_per_epoch = _limit_steps_per_epoch(steps_per_epoch, args, log=print)
    val_tensor = _build_val_tensor_for_main(data, args, device=args.device)
    _write_split_info(data, out_dir)

    centroids = _ensure_centroids(data, args, out_dir=out_dir, device=args.device)

    model = MFA_ARD(
        centroids=centroids,
        rank=args.rank,
        alpha0=args.alpha0,
        b0=args.b0,
        ard_weight=ard_weight,
        rank_threshold=args.rank_threshold,
    ).to(args.device)
    if getattr(args, "compile", False):
        print("Compiling model with torch.compile...")
        model = torch.compile(model)

    def _epoch_snapshot(snapshot_model, ep):
        snap_dir = out_dir / f"epoch_{ep:04d}"
        snap_dir.mkdir(parents=True, exist_ok=True)
        save_mfa_ard(snapshot_model, str(snap_dir / "mfa_model.pt"))

    info = train_nll_ard(
        model,
        train_loader,
        val_tensor=val_tensor,
        epochs=args.epochs,
        lr=args.lr,
        grad_clip=args.grad_clip,
        save_path=str(out_dir / "mfa_model.pt"),
        save_func=save_mfa_ard,
        ckpt_path=str(out_dir / "checkpoint.pt"),
        steps_per_epoch=steps_per_epoch,
        track_best=True,
        max_steps=args.max_steps,
        early_stop_delta=args.early_stop_delta,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
        epoch_snapshot_func=_epoch_snapshot,
        epoch_snapshot_every=args.epoch_snapshot_every,
        ard_warmup_frac=args.ard_warmup_frac,
        ard_ramp_frac=args.ard_ramp_frac,
        ard_schedule_epochs=args.ard_schedule_epochs,
    )

    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    q_eff = info.get("q_eff_mean")
    if q_eff is not None:
        print(f"Mean effective rank q_eff={q_eff:.2f} (max rank {args.rank}), "
              f"dead components={info.get('dead_components')}")

    # Pruning happens here and only here — after training, after the best-epoch
    # rollback — so the columns being zeroed are the ones the selected model
    # actually left below the noise floor.
    if args.prune_at_end:
        _prune_and_save(raw_model, out_dir, args=args, val_tensor=val_tensor)
    else:
        save_mfa_ard(raw_model, str(out_dir / "mfa_model.pt"))
        print(f"Model saved to {out_dir}/mfa_model.pt (not pruned)")

    _write_run_config(data, out_dir, args=args, ard_weight=ard_weight)
    _finish_wandb(wandb_run)


def _prune_and_save(raw_model, out_dir: Path, *, args, val_tensor) -> None:
    """Zero sub-threshold columns, then save both the pruned and raw models."""
    from dalg.models.adaptive_q.mfa_ard import save_mfa_ard
    from dalg.models.adaptive_q.train_ard import _eval_nll_tensor

    unpruned_path = out_dir / "mfa_model_unpruned.pt"
    save_mfa_ard(raw_model, str(unpruned_path))

    before = None
    if val_tensor is not None:
        before = _eval_nll_tensor(raw_model, val_tensor, args.device)

    q_before = raw_model.effective_ranks()
    kept = raw_model.prune_columns(threshold=args.rank_threshold)
    zeroed = int((raw_model.q - kept).sum().item())

    after = None
    if val_tensor is not None:
        after = _eval_nll_tensor(raw_model, val_tensor, args.device)

    save_mfa_ard(raw_model, str(out_dir / "mfa_model.pt"), pruned=True)

    print(
        f"[prune] threshold={args.rank_threshold} (x mean Psi) | "
        f"zeroed {zeroed:,}/{raw_model.K * raw_model.q:,} columns | "
        f"q_k mean {q_before.float().mean().item():.2f} -> {kept.float().mean().item():.2f} "
        f"[{int(kept.min())}..{int(kept.max())}] | "
        f"fully collapsed components={int((kept == 0).sum().item())}"
    )
    if before is not None:
        # The pruned columns should have been inert; a material jump here means
        # --rank-threshold was too aggressive for this run.
        print(f"[prune] val NLL {before:.6f} -> {after:.6f} (delta {after - before:+.6f})")
    print(f"Pruned model saved to {out_dir}/mfa_model.pt "
          f"(pre-prune copy at {unpruned_path.name})")


# CLI parsing / dispatch


def validate_args(args) -> None:
    """Validate CLI arguments."""
    if args.layer is None:
        raise SystemExit("train: --layer is required")
    if args.steps_per_epoch is not None and args.steps_per_epoch <= 0:
        raise SystemExit("train: --steps-per-epoch must be positive")
    if args.b0 <= 0:
        raise SystemExit("train: --b0 must be positive")
    if args.ard_lambda < 0:
        raise SystemExit("train: --ard-lambda must be non-negative")
    if args.rank_threshold <= 0:
        raise SystemExit("train: --rank-threshold must be positive")
    if not (0.0 <= args.ard_warmup_frac <= 1.0 and 0.0 <= args.ard_ramp_frac <= 1.0):
        raise SystemExit("train: --ard-warmup-frac and --ard-ramp-frac must be in [0, 1]")
    if args.ard_warmup_frac + args.ard_ramp_frac > 1.0:
        raise SystemExit("train: --ard-warmup-frac + --ard-ramp-frac must not exceed 1")
    if args.ard_schedule_epochs is not None and args.ard_schedule_epochs <= 0:
        raise SystemExit("train: --ard-schedule-epochs must be positive")
    if args.epochs <= 0 and args.ard_schedule_epochs is None and args.ard_lambda > 0:
        # Without a horizon the beta schedule cannot be expressed as fractions,
        # and falling back to beta=1 would reintroduce the cold-start collapse.
        raise SystemExit(
            "train: --epochs <= 0 needs an explicit --ard-schedule-epochs so the "
            "ARD warmup/ramp has a horizon (or set --ard-lambda 0)"
        )
    if (
        args.epochs <= 0
        and args.max_steps is None
        and (args.val_frac <= 0 or args.early_stop_delta <= 0)
    ):
        raise SystemExit(
            "train: --epochs <= 0 needs validation early stopping or --max-steps"
        )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train ARD-regularized MFA on activation shards")
    p.add_argument("--device", default="cuda", help="Device (cuda/cpu/mps)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--shard-dir", required=True, help="Activation shard root from extract-windows")
    p.add_argument("--layer", type=int, required=True, help="Layer to train on")
    p.add_argument("--out-dir", default=None, help="Where to save centroids/model")
    p.add_argument(
        "--centroids-path",
        default=None,
        help="Path to a pre-computed centroids.pt (or a directory containing one) "
             "to use instead of fitting KMeans. Copied into <out_dir>/centroids.pt.",
    )
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--val-on-gpu", action="store_true")
    p.add_argument("--K", type=int, required=True, help="Number of components")
    p.add_argument("--rank", type=int, default=64,
                   help="Maximum MFA rank (q) per component; ARD prunes below it, "
                        "so set this generously.")
    p.add_argument("--alpha0", type=float, default=1.0,
                   help="Gamma shape alpha0 of the ARD prior on nu.")
    p.add_argument("--b0", type=float, default=1e-4,
                   help="Gamma rate b0 of the ARD prior on nu. Smaller values weaken "
                        "shrinkage for columns already near zero.")
    p.add_argument("--ard-lambda", type=float, default=1.0,
                   help="Multiplier on the ARD penalty. The applied weight is "
                        "lambda / n_train_tokens; lambda=1 is the plain MAP objective.")
    p.add_argument("--ard-warmup-frac", type=float, default=0.15,
                   help="Fraction of the epoch budget trained with ard_beta=0. Full "
                        "pressure from a cold start collapses columns into the "
                        "penalty's s->0 well before they align with the data.")
    p.add_argument("--ard-ramp-frac", type=float, default=0.20,
                   help="Fraction of the epoch budget over which ard_beta ramps "
                        "linearly from 0 to 1 after the warmup.")
    p.add_argument("--ard-schedule-epochs", type=int, default=None,
                   help="Epoch horizon the beta schedule is computed against. "
                        "Defaults to --epochs; required when --epochs <= 0.")
    p.add_argument("--prune-at-end", action=argparse.BooleanOptionalAction, default=True,
                   help="After training, zero out every loading column below "
                        "--rank-threshold. The pre-prune model is kept alongside as "
                        "mfa_model_unpruned.pt.")
    p.add_argument("--rank-threshold", type=float, default=1.0,
                   help="A loading column counts toward q_k when its variance exceeds "
                        "this multiple of the component's mean unique variance (Psi). "
                        "1.0 = a factor must explain more than the noise it sits on.")
    p.add_argument("--epochs", type=int, default=10,
                   help="Epoch cap. Set <=0 to run until early stopping, max-steps, or walltime.")
    p.add_argument("--steps-per-epoch", type=int, default=None,
                   help="Debug/smoke option: cap batches per epoch.")
    p.add_argument("--early-stop-delta", type=float, default=1e-3,
                   help="Stop once consecutive validation NLLs differ by less than this. "
                        "Set <=0 to disable.")
    p.add_argument("--early-stop-patience", type=int, default=None,
                   help="Stop after this many epochs without improving the best validation "
                        "NLL by at least --early-stop-min-delta. Off by default.")
    p.add_argument("--early-stop-min-delta", type=float, default=0.0,
                   help="Minimum validation-NLL improvement to count as progress for patience.")
    p.add_argument("--epoch-snapshot-every", type=int, default=5,
                   help="Save a full model snapshot every N epochs (plus epoch 1) under "
                        "<out_dir>/epoch_XXXX/. Set 0 to disable snapshots.")
    p.add_argument("--max-steps", type=int, default=None,
                   help="Hard cap on total optimizer steps; if reached, training stops early.")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--grad-clip", type=float, default=None)
    p.add_argument("--proj-dim", type=int, default=32)
    p.add_argument("--refine-epochs", type=int, default=25,
                   help="Number of additional epochs to run with token assignments fixed to "
                        "the nearest centroid; only applies when fitting centroids from scratch.")
    p.add_argument("--vocab-size", type=int, default=50257)
    p.add_argument("--pool-size", type=int, default=None)
    p.add_argument("--max-pool-size", type=int, default=2_000_000)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--wandb", action="store_true", help="Log training to Weights & Biases")
    p.add_argument("--wandb-project", default=None, help="W&B project name")
    p.add_argument("--wandb-name", default=None, help="W&B run name")
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)
    cmd_train_ard(args)


if __name__ == "__main__":
    main()
