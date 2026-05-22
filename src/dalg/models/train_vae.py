from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from dalg.data.shard_activations import (
    ShardActivationDataset,
    load_meta_index,
    stratified_split,
)
from dalg.models.vae import VAEConfig, build_lightning_vae, build_prior


def _parse_dims(value: str | None) -> tuple[int, ...]:
    if value is None:
        return ()
    dims = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not dims:
        raise ValueError("Dimension list must not be empty.")
    if any(dim <= 0 for dim in dims):
        raise ValueError(f"All dimensions must be positive; got {dims}.")
    return dims


def adapt_loader_batch(batch: Any) -> tuple[torch.Tensor, torch.Tensor | None, Any | None]:
    """Normalize current/future loader outputs to (x, tok, metadata).

    Supported shapes:
    - (x, tok)
    - (x, tok, metadata)
    - ((x, tok), metadata)
    - ((x, tok, metadata),)
    """
    x = None
    tok = None
    meta = None

    if isinstance(batch, torch.Tensor):
        x = batch
    elif isinstance(batch, (tuple, list)):
        if len(batch) == 2 and isinstance(batch[0], (tuple, list)):
            inner = batch[0]
            meta = batch[1]
            if len(inner) >= 2:
                x, tok = inner[0], inner[1]
            elif len(inner) == 1:
                x = inner[0]
        elif len(batch) >= 3:
            x, tok, meta = batch[0], batch[1], batch[2]
        elif len(batch) == 2:
            x, tok = batch[0], batch[1]
        elif len(batch) == 1 and isinstance(batch[0], (tuple, list)):
            inner = batch[0]
            if len(inner) >= 3:
                x, tok, meta = inner[0], inner[1], inner[2]
            elif len(inner) == 2:
                x, tok = inner[0], inner[1]
            elif len(inner) == 1:
                x = inner[0]

    if x is None or not torch.is_tensor(x):
        raise ValueError("Unable to extract activations tensor from batch.")

    if x.ndim == 1:
        x = x.unsqueeze(0)
    if x.ndim >= 3:
        x = x.reshape(-1, x.shape[-1])
    elif x.ndim != 2:
        raise ValueError(f"Expected x to be rank-2 or rank-3+, got shape {tuple(x.shape)}")

    if x.shape[-1] != 2048:
        raise ValueError(f"Expected activation dim 2048, got {x.shape[-1]}")

    if tok is not None:
        if not torch.is_tensor(tok):
            tok = None
        else:
            tok = tok.reshape(-1)
            if tok.shape[0] != x.shape[0]:
                # If contract carries per-window tokens but x flattened to per-token,
                # keep only x for objective and drop incompatible token vector.
                tok = None

    return x, tok, meta


@torch.no_grad()
def compute_feature_stats_from_loader(loader: DataLoader, max_tokens: int = 0) -> tuple[torch.Tensor, torch.Tensor, int]:
    total = 0
    fsum = torch.zeros(2048, dtype=torch.float64)
    fsumsq = torch.zeros(2048, dtype=torch.float64)

    for batch in loader:
        x, _tok, _meta = adapt_loader_batch(batch)
        x64 = x.to(torch.float64)
        fsum += x64.sum(dim=0)
        fsumsq += x64.square().sum(dim=0)
        total += x64.shape[0]
        if max_tokens > 0 and total >= max_tokens:
            break

    if total == 0:
        raise ValueError("No tokens observed while computing normalization stats.")

    mean = fsum / total
    var = fsumsq / total - mean.square()
    std = torch.sqrt(var.clamp_min(1e-12))
    return mean.to(torch.float32), std.to(torch.float32), total


def _build_shard_loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader | None, int, int]:
    meta_index = load_meta_index(args.shard_dir)
    train_pos, val_pos = stratified_split(meta_index, val_frac=args.val_frac, seed=args.split_seed)

    train_ds = ShardActivationDataset(
        args.shard_dir,
        layer=args.layer,
        row_subset=train_pos,
        drop_prefix=args.drop_prefix,
        shuffle_shards=True,
        shuffle_within_shard=True,
        seed=args.seed,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=(args.num_workers > 0),
    )

    val_loader = None
    if not args.no_val and len(val_pos) > 0:
        val_ds = ShardActivationDataset(
            args.shard_dir,
            layer=args.layer,
            row_subset=val_pos,
            drop_prefix=args.drop_prefix,
            shuffle_shards=False,
            shuffle_within_shard=False,
            seed=args.seed,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            num_workers=max(0, args.num_workers // 2),
            pin_memory=args.pin_memory,
            persistent_workers=(args.num_workers // 2 > 0),
        )

    return train_loader, val_loader, len(train_pos), len(val_pos)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Train a VAE on sharded activations.")
    ap.add_argument("--shard-dir", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--drop-prefix", type=int, default=32)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-val", action="store_true")

    ap.add_argument("--latent-dim", type=int, default=64)
    ap.add_argument("--enc-hidden-dims", default="1024,512")
    ap.add_argument("--dec-hidden-dims", default="512,1024")
    ap.add_argument("--prior", choices=("standard", "mog", "vamp"), default="standard")
    ap.add_argument("--prior-components", type=int, default=100)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--beta-warmup-steps", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)

    ap.add_argument("--normalize", choices=("none", "meanstd"), default="none")
    ap.add_argument("--input-clip", type=float, default=5.0)
    ap.add_argument("--norm-max-tokens", type=int, default=0)

    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--accelerator", default="auto")
    ap.add_argument("--devices", default=1)
    ap.add_argument("--gradient-clip-val", type=float, default=1.0)
    ap.add_argument("--logger", choices=("csv", "none"), default="csv")
    ap.add_argument("--resume-from", default=None)
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    pl.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision("high")

    train_loader, val_loader, n_train_rows, n_val_rows = _build_shard_loaders(args)

    out_dir = Path(args.out_dir or (Path(args.shard_dir) / f"layer{args.layer:02d}_vae"))
    run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = out_dir / run_name
    ckpt_dir = run_dir / "checkpoints"
    log_dir = run_dir / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    input_mean = None
    input_std = None
    norm_stats_path = None

    if args.normalize == "meanstd":
        stats_dir = out_dir / "feature_stats"
        stats_dir.mkdir(parents=True, exist_ok=True)
        norm_stats_path = stats_dir / f"layer{args.layer:02d}_meanstd.pt"
        if norm_stats_path.exists():
            payload = torch.load(norm_stats_path, map_location="cpu", weights_only=True)
            input_mean = payload["mean"]
            input_std = payload["std"]
        else:
            mean, std, n_tok = compute_feature_stats_from_loader(train_loader, max_tokens=args.norm_max_tokens)
            torch.save({"mean": mean, "std": std, "tokens": n_tok}, norm_stats_path)
            input_mean, input_std = mean, std

    config = VAEConfig(
        input_dim=2048,
        latent_dim=args.latent_dim,
        enc_hidden_dims=_parse_dims(args.enc_hidden_dims) or (1024, 512),
        dec_hidden_dims=_parse_dims(args.dec_hidden_dims) or (512, 1024),
        lr=args.lr,
        weight_decay=args.weight_decay,
        beta=args.beta,
        beta_warmup_steps=args.beta_warmup_steps,
        input_mean=input_mean,
        input_std=input_std,
        input_clip=(args.input_clip if args.normalize == "meanstd" else None),
    )

    prior = build_prior(args.prior, args.latent_dim, args.prior_components)
    model = build_lightning_vae(config, prior=prior)

    callbacks: list[Any] = []
    if val_loader is not None:
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="best",
            monitor="val/loss",
            mode="min",
            save_top_k=1,
            save_last=True,
            auto_insert_metric_name=False,
        )
    else:
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="best",
            save_top_k=0,
            save_last=True,
        )
    callbacks.append(checkpoint_callback)

    logger = None
    if args.logger == "csv":
        from pytorch_lightning.loggers import CSVLogger

        logger = CSVLogger(save_dir=log_dir, name="metrics")

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        default_root_dir=run_dir,
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=args.gradient_clip_val,
        gradient_clip_algorithm="norm",
    )

    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=args.resume_from,
    )

    final_ckpt = ckpt_dir / "final.ckpt"
    trainer.save_checkpoint(str(final_ckpt))

    manifest = {
        "shard_dir": str(args.shard_dir),
        "layer": args.layer,
        "train_rows": n_train_rows,
        "val_rows": n_val_rows,
        "batch_size": args.batch_size,
        "drop_prefix": args.drop_prefix,
        "latent_dim": args.latent_dim,
        "enc_hidden_dims": list(config.enc_hidden_dims),
        "dec_hidden_dims": list(config.dec_hidden_dims),
        "prior": args.prior,
        "prior_components": args.prior_components,
        "beta": args.beta,
        "beta_warmup_steps": args.beta_warmup_steps,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "normalize": args.normalize,
        "input_clip": args.input_clip if args.normalize == "meanstd" else None,
        "norm_stats_path": str(norm_stats_path) if norm_stats_path is not None else None,
        "seed": args.seed,
        "split_seed": args.split_seed,
        "val_frac": args.val_frac,
        "epochs": args.epochs,
        "run_dir": str(run_dir),
        "checkpoint_dir": str(ckpt_dir),
        "final_checkpoint": str(final_ckpt),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
