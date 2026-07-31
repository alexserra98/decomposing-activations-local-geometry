"""ARD-MFA training on the toy-manifold `.pt` datasets.

`dalg.cli.run_training_ard` reads activation shards; these datasets are single
`.pt` files holding in-memory tensors, so this script provides the data path and
reuses everything else (`MFA_ARD`, `train_nll_ard`, `save_mfa_ard`) unchanged.

The whole train split fits on the GPU (300k x 128 float32 = 153 MB), so batches
are sliced directly from a device-resident tensor instead of going through a
DataLoader. Batch order uses the global RNG, which `train_nll_ard` checkpoints
and restores, so a resume replays the same order.

Example:

    PYTHONPATH=src python scripts/temporary/train_ard_toy_manifolds.py \
        --dataset dalg-cache/assets/toy_manifolds_centered.pt \
        --out-dir dalg-cache/toy_manifold_models/centered_K32000_q50_mfa_ard \
        --K 32000 --rank 50 --epochs 50 --batch-size 512 --device cuda
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dalg.models.mfa_ard import MFA_ARD, save_mfa_ard  # noqa: E402
from dalg.models.train_ard import _eval_nll, train_nll_ard  # noqa: E402


# Data


class TensorBatches:
    """Iterable of device-resident batches sliced from one tensor.

    Reshuffles on every `__iter__` using the global RNG (not a private
    generator) so that `train_nll_ard`'s checkpointed RNG state reproduces the
    batch order on resume.
    """

    def __init__(self, x: torch.Tensor, batch_size: int, *, shuffle: bool):
        self.x = x
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)

    def __len__(self) -> int:
        return math.ceil(self.x.shape[0] / self.batch_size)

    def __iter__(self):
        n = self.x.shape[0]
        order = torch.randperm(n, device=self.x.device) if self.shuffle else None
        for start in range(0, n, self.batch_size):
            if order is None:
                yield self.x[start:start + self.batch_size]
            else:
                yield self.x[order[start:start + self.batch_size]]


def load_dataset(path: Path, device: str) -> dict:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("x_train", "x_val"):
        if key not in blob:
            raise SystemExit(f"{path} has no '{key}'; not a toy-manifold dataset")
    x_train = blob["x_train"].float().to(device)
    x_val = blob["x_val"].float().to(device)
    return {
        "x_train": x_train,
        "x_val": x_val,
        "config": blob.get("config", {}),
        "n_train": int(x_train.shape[0]),
        "n_val": int(x_val.shape[0]),
        "D": int(x_train.shape[1]),
    }


# Centroid initialization


@torch.no_grad()
def _assign(x: torch.Tensor, c: torch.Tensor, block_x: int, block_c: int) -> torch.Tensor:
    """Nearest-centroid labels, blocked over both points and centroids."""
    n = x.shape[0]
    labels = torch.empty(n, dtype=torch.long, device=x.device)
    c_sq = (c * c).sum(dim=1)
    for s in range(0, n, block_x):
        xb = x[s:s + block_x]
        best = torch.full((xb.shape[0],), float("inf"), device=x.device)
        best_idx = torch.zeros(xb.shape[0], dtype=torch.long, device=x.device)
        for t in range(0, c.shape[0], block_c):
            cb = c[t:t + block_c]
            # ||x||^2 is constant per row, so it is dropped from the argmin.
            d2 = c_sq[t:t + block_c][None, :] - 2.0 * (xb @ cb.T)
            vals, idx = d2.min(dim=1)
            better = vals < best
            best = torch.where(better, vals, best)
            best_idx = torch.where(better, idx + t, best_idx)
        labels[s:s + xb.shape[0]] = best_idx
    return labels


@torch.no_grad()
def kmeans(
    x: torch.Tensor,
    k: int,
    *,
    iters: int,
    tol: float = 1e-4,
    block_x: int = 8192,
    block_c: int = 8192,
    log=print,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Lloyd's algorithm from a random-sample init. Returns (centroids, counts).

    k-means++ is O(k) sequential passes, which is minutes of launch overhead at
    k=32000; a random sample of distinct data points is the init k-means++
    approximates anyway, and Lloyd's converges from it in a handful of blocked
    passes. Empty clusters keep their previous position rather than being
    reseeded, so K stays fixed and the centroids stay on data.
    """
    n = x.shape[0]
    if k > n:
        raise SystemExit(f"K={k} exceeds the number of training points ({n})")
    centroids = x[torch.randperm(n, device=x.device)[:k]].clone()

    counts = torch.zeros(k, device=x.device)
    for it in range(1, iters + 1):
        t0 = time.time()
        labels = _assign(x, centroids, block_x, block_c)
        sums = torch.zeros_like(centroids)
        counts = torch.zeros(k, device=x.device)
        sums.index_add_(0, labels, x)
        counts.index_add_(0, labels, torch.ones(n, device=x.device))

        occupied = counts > 0
        updated = centroids.clone()
        updated[occupied] = sums[occupied] / counts[occupied][:, None]
        shift = (updated - centroids).norm(dim=1).max().item()
        centroids = updated

        log(
            f"  kmeans it {it:02d}/{iters} | max shift={shift:.5f} | "
            f"empty={int((~occupied).sum().item()):,}/{k:,} | {time.time() - t0:.1f}s"
        )
        if shift < tol:
            log(f"  kmeans converged at iteration {it}")
            break

    return centroids, counts


@torch.no_grad()
def residual_variance(x: torch.Tensor, centroids: torch.Tensor, *, block_x: int = 8192) -> float:
    """Mean per-dimension variance of points around their nearest centroid.

    This is the natural scale for `psi_init`: the global variance overstates it
    badly once K is large, and starting Psi orders of magnitude above the true
    noise floor makes the ARD rank criterion (`s^2 > threshold * mean Psi`)
    meaningless for the first epochs.
    """
    labels = _assign(x, centroids, block_x, centroids.shape[0])
    total = 0.0
    for s in range(0, x.shape[0], block_x):
        xb = x[s:s + block_x]
        residual = xb - centroids[labels[s:s + block_x]]
        total += float(residual.pow(2).sum().item())
    return total / (x.shape[0] * x.shape[1])


def ensure_centroids(data: dict, args, out_dir: Path) -> torch.Tensor:
    """Resolve centroids: cached, provided, or freshly fitted."""
    path = out_dir / "centroids.pt"
    if not path.exists():
        source = getattr(args, "centroids_path", None)
        if source:
            src = Path(source)
            if src.is_dir():
                src = src / "centroids.pt"
            if not src.is_file():
                raise SystemExit(f"--centroids-path not found: {source}")
            import shutil
            shutil.copyfile(src, path)
            print(f"Centroids: copied from {src}")
        else:
            print(f"Fitting {args.K:,} centroids on {data['n_train']:,} points...")
            centroids, counts = kmeans(
                data["x_train"], args.K, iters=args.kmeans_iters
            )
            occupancy = counts[counts > 0]
            print(
                f"Centroids: {tuple(centroids.shape)} | occupied="
                f"{int((counts > 0).sum().item()):,}/{args.K:,} | points per occupied "
                f"cluster mean={occupancy.mean().item():.1f} max={int(occupancy.max().item())}"
            )
            torch.save(centroids.cpu(), path)

    centroids = torch.load(path, map_location=args.device, weights_only=True)
    if centroids.shape[0] != args.K:
        raise SystemExit(
            f"Cached centroids K={centroids.shape[0]} != --K {args.K}; "
            f"delete {path} to recompute."
        )
    return centroids


# Run bookkeeping


def maybe_init_wandb(args, data: dict, *, ard_weight: float, psi_init: float, scale_init: float):
    if not args.wandb:
        return None
    import wandb

    return wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        config={
            "model": "MFA_ARD",
            "dataset": str(args.dataset),
            "dataset_config": data["config"],
            "K": args.K,
            "rank": args.rank,
            "alpha0": args.alpha0,
            "b0": args.b0,
            "ard_lambda": args.ard_lambda,
            "ard_weight": ard_weight,
            "ard_warmup_frac": args.ard_warmup_frac,
            "ard_ramp_frac": args.ard_ramp_frac,
            "rank_threshold": args.rank_threshold,
            "psi_init": psi_init,
            "scale_init": scale_init,
            "psi_per_component": args.psi_per_component,
            "epochs": args.epochs,
            "lr": args.lr,
            "grad_clip": args.grad_clip,
            "batch_size": args.batch_size,
            "early_stop_delta": args.early_stop_delta,
            "n_train": data["n_train"],
            "n_val": data["n_val"],
            "d_model": data["D"],
            "training_mode": "vanilla_ard_toy_manifolds",
        },
    )


def write_run_config(data: dict, out_dir: Path, *, args, ard_weight, psi_init, scale_init) -> None:
    cfg = {
        "model": "MFA_ARD",
        "dataset": str(args.dataset),
        "dataset_config": data["config"],
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
        "psi_init": psi_init,
        "scale_init": scale_init,
        "psi_per_component": bool(args.psi_per_component),
        "epochs": args.epochs,
        "lr": args.lr,
        "grad_clip": args.grad_clip,
        "batch_size": args.batch_size,
        "early_stop_delta": args.early_stop_delta,
        "steps_per_epoch": args.steps_per_epoch,
        "kmeans_iters": args.kmeans_iters,
        "seed": args.seed,
        "n_train": data["n_train"],
        "n_val": data["n_val"],
        "d_model": data["D"],
        "training_mode": "vanilla_ard_toy_manifolds",
        "world_size": 1,
    }
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))


def prune_and_save(model, out_dir: Path, *, args, val_batches) -> None:
    """Zero sub-threshold columns after training, keeping a pre-prune copy."""
    unpruned = out_dir / "mfa_model_unpruned.pt"
    save_mfa_ard(model, str(unpruned))

    before = _eval_nll(model, val_batches, args.device)
    q_before = model.effective_ranks()
    kept = model.prune_columns(threshold=args.rank_threshold)
    zeroed = int((model.q - kept).sum().item())
    after = _eval_nll(model, val_batches, args.device)

    save_mfa_ard(model, str(out_dir / "mfa_model.pt"), pruned=True)
    print(
        f"[prune] threshold={args.rank_threshold} (x mean Psi) | "
        f"zeroed {zeroed:,}/{model.K * model.q:,} columns | "
        f"q_k mean {q_before.float().mean().item():.2f} -> {kept.float().mean().item():.2f} "
        f"[{int(kept.min())}..{int(kept.max())}] | "
        f"fully collapsed components={int((kept == 0).sum().item())}"
    )
    print(f"[prune] val NLL {before:.6f} -> {after:.6f} (delta {after - before:+.6f})")
    print(f"Pruned model saved to {out_dir}/mfa_model.pt (pre-prune copy at {unpruned.name})")


# Main


def main() -> None:
    args = build_parser().parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is not available")
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    dataset_path = Path(args.dataset)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_dataset(dataset_path, args.device)
    print(f"dataset={dataset_path}  out_dir={out_dir}")
    print(f"train={data['n_train']:,}  val={data['n_val']:,}  D={data['D']}  "
          f"config={data['config']}")

    if args.val_subset and args.val_subset < data["n_val"]:
        data["x_val"] = data["x_val"][:args.val_subset]
        print(f"validating on the first {args.val_subset:,} val points")

    train_batches = TensorBatches(data["x_train"], args.batch_size, shuffle=True)
    val_batches = TensorBatches(data["x_val"], args.val_batch_size or args.batch_size,
                                shuffle=False)
    steps_per_epoch = len(train_batches)
    if args.steps_per_epoch:
        steps_per_epoch = min(steps_per_epoch, args.steps_per_epoch)
    print(f"steps_per_epoch={steps_per_epoch:,}  batch_size={args.batch_size}")

    centroids = ensure_centroids(data, args, out_dir)

    # Psi and the loading scales are initialized from the data rather than from
    # MFA's defaults (1.0 each), which assume activation-scale inputs. Here the
    # per-dimension variance is ~1e-2, so the defaults would start the model
    # three orders of magnitude too wide.
    psi_init = args.psi_init
    if psi_init is None:
        psi_init = residual_variance(data["x_train"], centroids)
    scale_init = args.scale_init
    if scale_init is None:
        # q columns of norm s contribute q*s^2/D per-dimension variance; match Psi.
        scale_init = math.sqrt(max(psi_init, 1e-12) * data["D"] / args.rank)
    print(f"init: psi_init={psi_init:.6g}  scale_init={scale_init:.6g}  "
          f"psi_per_component={bool(args.psi_per_component)}")

    # The ARD penalty is a prior over parameters — it applies once per dataset,
    # while the loss is a per-sample mean. --ard-lambda dials the pressure.
    ard_weight = args.ard_lambda / max(1, data["n_train"])
    log_coeff = data["D"] / 2.0 + args.alpha0 - 1.0
    print(
        f"[ard] alpha0={args.alpha0}  b0={args.b0}  lambda={args.ard_lambda}  "
        f"n_train={data['n_train']:,}  ard_weight={ard_weight:.6g}  "
        f"log_coeff=D/2+alpha0-1={log_coeff:.4f}  rank_threshold={args.rank_threshold}"
    )

    wandb_run = maybe_init_wandb(
        args, data, ard_weight=ard_weight, psi_init=psi_init, scale_init=scale_init
    )

    model = MFA_ARD(
        centroids=centroids,
        rank=args.rank,
        alpha0=args.alpha0,
        b0=args.b0,
        ard_weight=ard_weight,
        rank_threshold=args.rank_threshold,
        psi_init=psi_init,
        scale_init=scale_init,
        psi_per_component=bool(args.psi_per_component),
    ).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: K={model.K:,} D={model.D} q={model.q} | {n_params / 1e6:.1f}M parameters")
    if args.device.startswith("cuda"):
        print(f"cuda memory after model init: "
              f"{torch.cuda.memory_allocated() / 2**30:.2f} GiB allocated")

    def epoch_snapshot(snapshot_model, ep):
        snap_dir = out_dir / f"epoch_{ep:04d}"
        snap_dir.mkdir(parents=True, exist_ok=True)
        save_mfa_ard(snapshot_model, str(snap_dir / "mfa_model.pt"))

    info = train_nll_ard(
        model,
        train_batches,
        val_loader=val_batches,
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
        epoch_snapshot_func=epoch_snapshot if args.epoch_snapshot_every else None,
        epoch_snapshot_every=args.epoch_snapshot_every,
        ard_warmup_frac=args.ard_warmup_frac,
        ard_ramp_frac=args.ard_ramp_frac,
        ard_schedule_epochs=args.ard_schedule_epochs,
    )

    q_eff = info.get("q_eff_mean")
    if q_eff is not None:
        print(f"Mean effective rank q_eff={q_eff:.2f} (max rank {args.rank}), "
              f"dead components={info.get('dead_components')}")

    if args.prune_at_end:
        prune_and_save(model, out_dir, args=args, val_batches=val_batches)
    else:
        save_mfa_ard(model, str(out_dir / "mfa_model.pt"))
        print(f"Model saved to {out_dir}/mfa_model.pt (not pruned)")

    write_run_config(
        data, out_dir, args=args, ard_weight=ard_weight,
        psi_init=psi_init, scale_init=scale_init,
    )
    if wandb_run is not None:
        import wandb
        wandb.finish()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, help="Path to a toy-manifold .pt dataset")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--K", type=int, required=True, help="Number of components")
    p.add_argument("--rank", type=int, required=True,
                   help="MAXIMUM rank per component; ARD prunes below it")
    p.add_argument("--psi-init", type=float, default=None,
                   help="Default: mean within-cluster residual variance")
    p.add_argument("--scale-init", type=float, default=None,
                   help="Default: sqrt(psi_init * D / rank)")
    p.add_argument("--psi-per-component", action="store_true",
                   help="Per-component Psi_k instead of one shared Psi")
    p.add_argument("--centroids-path", default=None,
                   help="Reuse centroids from a file or run directory")
    p.add_argument("--kmeans-iters", type=int, default=30)

    p.add_argument("--alpha0", type=float, default=1.0, help="Gamma shape of the prior on nu")
    p.add_argument("--b0", type=float, default=1e-4, help="Gamma rate of the prior on nu")
    p.add_argument("--ard-lambda", type=float, default=1.0,
                   help="Applied ARD weight = lambda / n_train")
    p.add_argument("--rank-threshold", type=float, default=1.0,
                   help="Column counts toward q_k when s^2 > threshold * mean(Psi_k)")
    p.add_argument("--ard-warmup-frac", type=float, default=0.15)
    p.add_argument("--ard-ramp-frac", type=float, default=0.20)
    p.add_argument("--ard-schedule-epochs", type=int, default=None,
                   help="Horizon the schedule fractions are measured against")
    p.add_argument("--prune-at-end", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--val-batch-size", type=int, default=None)
    p.add_argument("--val-subset", type=int, default=0,
                   help="Validate on the first N points only (0 = all)")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--steps-per-epoch", type=int, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    # Off by default: an early stop shortens the run without shortening the beta
    # ramp, so the schedule would be sized against epochs that never happen.
    p.add_argument("--early-stop-delta", type=float, default=0.0)
    p.add_argument("--early-stop-patience", type=int, default=None)
    p.add_argument("--early-stop-min-delta", type=float, default=0.0)
    p.add_argument("--epoch-snapshot-every", type=int, default=0,
                   help="0 disables per-epoch model snapshots (they are ~1.5 GiB each)")

    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", default="dalg-mfa")
    p.add_argument("--wandb-name", default=None)
    return p


if __name__ == "__main__":
    main()
