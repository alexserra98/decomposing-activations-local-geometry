"""Training loop for ARD-regularized MFA (`MFA_ARD`).

Deliberately redundant with `train.py`: this is a self-contained copy of
`train_nll` so the ARD research path can diverge without touching the
production training path. Differences from `train_nll`:

- the step loss is `nll + ard_weight * ard_penalty` via `model.loss_terms(x)`,
  and the two terms are tracked separately;
- the ARD pressure is ramped in over epochs by `ard_beta` (see
  `ard_beta_schedule`). Applying full pressure from a cold start collapses
  every column into the stiff `s -> 0` well of the penalty before the loadings
  have aligned with the data, and that state is unrecoverable — measurably
  *worse* on the ARD objective itself than the same lambda reached via a warmup;
- per-epoch effective-rank statistics (the learned q_k) are logged;
- validation stays *pure NLL* and remains the model-selection metric, so ARD
  runs are directly comparable to baseline MFA runs and the prior term does not
  bias early stopping;
- the distributed / component-shard machinery is dropped. This path is
  single-process (vanilla) only.
"""

from __future__ import annotations

import math
import os
import time
from contextlib import contextmanager
from typing import Optional

import torch

try:
    import wandb as _wandb
except ImportError:
    _wandb = None


def _wandb_active() -> bool:
    return _wandb is not None and getattr(_wandb, "run", None) is not None


def _fmt_eta(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        return "?"
    s = int(seconds)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def _unwrap_model(model):
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def _cpu_state_dict(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


@contextmanager
def _strict_validation_matmul(device):
    """Use full float32 matmul precision for validation likelihoods."""
    dev = torch.device(device)
    if dev.type != "cuda" or not torch.cuda.is_available():
        yield
        return

    old_precision = torch.get_float32_matmul_precision()
    old_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(old_precision)
        torch.backends.cuda.matmul.allow_tf32 = old_matmul_tf32
        torch.backends.cudnn.allow_tf32 = old_cudnn_tf32


@torch.no_grad()
def _eval_nll(model, loader, device):
    """Validation NLL only — the ARD penalty is excluded on purpose."""
    model.eval()
    tot_nll, tot_n = 0.0, 0
    with _strict_validation_matmul(device):
        for batch in loader:
            x = batch[0] if isinstance(batch, (tuple, list)) else batch
            x = x.view(x.size(0), -1).to(device)
            nll = model.nll(x)                  # mean over batch
            B = x.size(0)
            tot_nll += nll.item() * B
            tot_n   += B
    return tot_nll / tot_n


@torch.no_grad()
def _eval_nll_tensor(model, X, device, chunk=8192):
    """Chunk-evaluate NLL on a (CPU/pinned) tensor; move each chunk to `device`."""
    model.eval()
    N = X.shape[0]
    tot = 0.0
    with _strict_validation_matmul(device):
        for i in range(0, N, chunk):
            xb = X[i:i + chunk].to(device, non_blocking=True)
            xb = xb.view(xb.size(0), -1).float()
            tot += float(model.nll(xb).item()) * xb.size(0)
    return tot / max(N, 1)


def _atomic_torch_save(obj, path):
    tmp = f"{path}.tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _rng_state(device):
    state = {"torch": torch.random.get_rng_state()}
    if torch.cuda.is_available() and torch.device(device).type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state(device)
    return state


def _restore_rng_state(state, device):
    if not state:
        return
    if "torch" in state:
        torch.random.set_rng_state(state["torch"].cpu())
    if "cuda" in state and torch.cuda.is_available() and torch.device(device).type == "cuda":
        torch.cuda.set_rng_state(state["cuda"].cpu(), device)


def ard_beta_schedule(
    epoch: int,
    total_epochs: Optional[int],
    *,
    warmup_frac: float = 0.15,
    ramp_frac: float = 0.20,
) -> float:
    """Multiplier on `ard_weight` for a 1-indexed epoch.

    Zero for the first `warmup_frac` of training, then linear from 0 to 1 across
    the next `ramp_frac`, then 1 for the rest. The warmup exists because the
    penalty's `s -> 0` well has stiffness `lambda * c / b0`: a column that dips
    into it before it has aligned with any data direction cannot climb back out,
    since its likelihood value vanishes as s^2 while the restoring force is
    linear in s. Letting the loadings grow first makes the pruning decision on
    columns that have had a chance to earn their place.

    With no known horizon (`total_epochs is None`) the schedule is off and this
    returns 1.0.
    """
    if total_epochs is None or total_epochs <= 0:
        return 1.0
    if not (0.0 <= warmup_frac <= 1.0 and 0.0 <= ramp_frac <= 1.0):
        raise ValueError("warmup_frac and ramp_frac must be in [0, 1]")
    if warmup_frac + ramp_frac > 1.0:
        raise ValueError("warmup_frac + ramp_frac must not exceed 1")

    done = max(0, int(epoch) - 1)          # epochs completed before this one
    warmup_end = warmup_frac * total_epochs
    ramp_end = (warmup_frac + ramp_frac) * total_epochs
    if done < warmup_end:
        return 0.0
    if done >= ramp_end:
        return 1.0
    span = ramp_end - warmup_end
    if span <= 0:
        return 1.0
    # +1 so the ramp's last epoch lands exactly on 1.0: the warmup fraction is
    # entirely at beta=0 and the ramp fraction is entirely spent ramping.
    return float(min(1.0, (done - warmup_end + 1) / span))


_HORIZON_KEY = "ard_schedule_epochs"


def _check_schedule_horizon(ckpt: dict, horizon, target_ard_weight: float, ckpt_path: str) -> None:
    """Refuse to resume onto a different beta schedule than the one saved.

    `ard_beta` is a function of (epoch, horizon). Resuming with a different
    horizon silently rescales the ramp, so the epochs already completed were
    trained under one schedule and the remaining ones under another — e.g. a run
    checkpointed at epoch 12 of 20 (beta already 1.0) that resumes with
    `epochs=60` drops back to beta=0 and un-does the pruning pressure.
    """
    if target_ard_weight <= 0:
        return  # no ARD pressure: the schedule cannot affect this run
    if _HORIZON_KEY not in ckpt:
        print(
            f"[ckpt] warning: {ckpt_path} predates ARD schedule tracking; "
            f"cannot verify its horizon against the current one ({horizon})",
            flush=True,
        )
        return

    saved = ckpt[_HORIZON_KEY]
    if saved == horizon:
        return
    raise RuntimeError(
        f"ARD schedule horizon changed on resume: {ckpt_path} was written with "
        f"{_HORIZON_KEY}={saved}, but this run computes {horizon}. The completed "
        f"epochs were trained on the old ramp, so continuing would apply a "
        f"different ard_beta to the same epoch indices. Either restore the "
        f"original epoch count, or pass ard_schedule_epochs={saved} "
        f"(--ard-schedule-epochs {saved}) to keep the original schedule while "
        f"changing the epoch cap."
    )


@torch.no_grad()
def _rank_stats(model) -> dict:
    """Effective-rank summary of an MFA_ARD, or {} for models without ARD."""
    effective_ranks = getattr(model, "effective_ranks", None)
    if not callable(effective_ranks):
        return {}
    q_eff = effective_ranks().float()
    scales = model.column_scales()
    # Psi is the earliest collapse signal: when ARD strips the loadings, the
    # unique variance inflates to absorb what W gave up, and it moves before
    # q_eff finishes falling.
    psi_mean = float(model._psi().mean().item())
    return {
        "q_eff": q_eff,
        "q_eff_mean": float(q_eff.mean().item()),
        "q_eff_median": float(q_eff.median().item()),
        "q_eff_min": float(q_eff.min().item()),
        "q_eff_max": float(q_eff.max().item()),
        "scale_mean": float(scales.mean().item()),
        "psi_mean": psi_mean,
        "dead_components": int((q_eff == 0).sum().item()),
    }


def train_nll_ard(
    model,
    loader,
    *,
    val_loader=None,
    val_tensor=None,
    epochs=5,
    lr=1e-3,
    grad_clip=None,
    save_path=None,
    save_func=None,
    log_interval=100,
    steps_per_epoch=None,
    ckpt_path=None,
    track_best=True,
    max_steps=None,
    early_stop_delta=1e-3,
    early_stop_patience=None,
    early_stop_min_delta=0.0,
    epoch_snapshot_func=None,
    epoch_snapshot_every=5,
    ard_warmup_frac=0.15,
    ard_ramp_frac=0.20,
    ard_schedule_epochs=None,
):
    """Train an `MFA_ARD` on NLL + ARD penalty, keeping the best-by-val-NLL model.

    The step loss is `nll + ard_beta * ard_weight * ard_penalty`, where
    `ard_beta` follows `ard_beta_schedule` over the epoch horizon
    (`ard_schedule_epochs`, defaulting to `epochs`). *Selection and early
    stopping use validation NLL alone*, so runs stay comparable to non-ARD MFA
    training. Validation follows the same two paths as `train_nll`:
    `val_tensor` (pre-materialized, fast) takes priority over `val_loader`;
    with neither, the metric falls back to the epoch's training NLL.

    Passing `epochs <= 0` removes the epoch cap; training then runs until early
    stopping, `max_steps`, or the surrounding job limit stops it.

    If `ckpt_path` is given, a full training checkpoint (model + optimizer +
    epoch + best state + RNG) is written atomically after every epoch, and
    reloaded on startup to resume from the next epoch. `ard_beta` is recomputed
    from the epoch index rather than stored, so a resume rejoins the schedule
    where it left off. The horizon it was computed against *is* stored, and a
    resume that would change it raises rather than silently re-scheduling the
    remaining epochs — pass `ard_schedule_epochs` equal to the stored value to
    change the epoch cap while keeping the original ramp.

    Known gap: early stopping ends a run before `epochs`, so the ramp is sized
    against the requested budget rather than the realized one. Left as-is for
    now.
    """
    epoch_limit = None if epochs is None or int(epochs) <= 0 else int(epochs)
    epoch_label = "unbounded" if epoch_limit is None else f"{epoch_limit:02d}"

    raw_model = _unwrap_model(model)
    device = next(model.parameters()).device
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # Target ARD pressure and the horizon the beta schedule is measured against.
    # Both are needed before the checkpoint load so a resume can be validated
    # against the schedule the checkpoint was written under.
    target_ard_weight = float(getattr(raw_model, "ard_weight", 0.0))
    horizon = ard_schedule_epochs if ard_schedule_epochs else epoch_limit

    best_metric = float("inf")
    best_state  = _cpu_state_dict(raw_model) if track_best else None
    best_epoch  = 0
    start_epoch = 1
    last_val_metric = None
    epochs_without_improvement = 0

    if ckpt_path and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        _check_schedule_horizon(ckpt, horizon, target_ard_weight, ckpt_path)
        raw_model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        if track_best:
            best_metric = ckpt["best_metric"]
            best_state  = ckpt["best_state"]
            best_epoch  = ckpt["best_epoch"]
        last_val_metric = ckpt.get("last_val_metric")
        epochs_without_improvement = int(
            ckpt.get("epochs_without_improvement", max(0, ckpt["epoch"] - best_epoch))
        )
        start_epoch = ckpt["epoch"] + 1
        _restore_rng_state(ckpt.get("rng_state"), device)
        if track_best:
            print(f"[ckpt] resumed from epoch {ckpt['epoch']:02d}  "
                  f"best_metric={best_metric:.6f}  best_epoch={best_epoch:02d}  "
                  f"next={start_epoch:02d}/{epoch_label}", flush=True)
        else:
            print(f"[ckpt] resumed from epoch {ckpt['epoch']:02d}  "
                  f"next={start_epoch:02d}/{epoch_label}", flush=True)

    global_step = (start_epoch - 1) * (steps_per_epoch or 0)
    wandb_on = _wandb_active()

    def log(msg: str) -> None:
        print(msg, flush=True)

    # `ard_beta` below ramps the model's live ard_weight up to target_ard_weight.
    # That attribute is a plain float, not a buffer, so scheduling it never
    # touches the state_dict.
    log(f"[ard] ard_weight={target_ard_weight:.6g}  "
        f"alpha0={getattr(raw_model, 'alpha0', None)}  "
        f"b0={getattr(raw_model, 'b0', None)}  "
        f"log_coeff={getattr(raw_model, 'log_coeff', None)}")
    if horizon:
        log(f"[ard] beta schedule over {horizon} epochs: 0 for the first "
            f"{ard_warmup_frac:.0%}, ramp to 1 across the next {ard_ramp_frac:.0%}, "
            f"then constant")
    elif target_ard_weight > 0:
        log("[ard] no epoch horizon — beta fixed at 1.0 (no warmup; "
            "cold-start column collapse is likely)")

    stats = _rank_stats(raw_model)
    ep = start_epoch
    while epoch_limit is None or ep <= epoch_limit:
        model.train()
        total_nll, total_pen, total_n = 0.0, 0.0, 0

        # Recomputed from the epoch index, so resume picks the schedule back up
        # exactly where it left off.
        ard_beta = ard_beta_schedule(
            ep, horizon, warmup_frac=ard_warmup_frac, ramp_frac=ard_ramp_frac
        )
        raw_model.ard_weight = target_ard_weight * ard_beta

        ep_start = time.time()
        win_start = ep_start
        win_loss_sum = 0.0
        win_n = 0
        total_str = str(steps_per_epoch) if steps_per_epoch else "?"
        log(f"[epoch {ep:02d}/{epoch_label}] start — {total_str} steps | "
            f"ard_beta={ard_beta:.3f} (weight={raw_model.ard_weight:.6g})")

        for batch_idx, batch in enumerate(loader, 1):
            x = batch[0] if isinstance(batch, (tuple, list)) else batch
            x = x.view(x.size(0), -1).to(device)
            opt.zero_grad(set_to_none=True)
            loss, nll, penalty = model.loss_terms(x)
            loss.backward()

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            opt.step()

            B = x.size(0)
            loss_val = float(loss.item())
            nll_val = float(nll.item())
            pen_val = float(penalty.item())
            total_nll += nll_val * B
            total_pen += pen_val * B
            total_n   += B
            win_loss_sum += nll_val * B
            win_n += B
            global_step += 1

            if wandb_on:
                _wandb.log(
                    {
                        "train/loss": loss_val,
                        "train/nll": nll_val,
                        "train/ard_penalty": pen_val,
                        "ard/beta": ard_beta,
                        "epoch": ep,
                    },
                    step=global_step,
                )

            if (batch_idx % log_interval) == 0:
                now = time.time()
                win_dt = max(now - win_start, 1e-6)
                steps_per_sec = log_interval / win_dt
                window_nll = win_loss_sum / max(1, win_n)
                if steps_per_epoch is not None:
                    remaining = steps_per_epoch - batch_idx
                    eta = remaining / steps_per_sec if steps_per_sec > 0 else float("inf")
                    pct = 100.0 * batch_idx / steps_per_epoch
                    log(
                        f"  ep {ep:02d} step {batch_idx:>6d}/{steps_per_epoch} "
                        f"({pct:5.1f}%) | nll={window_nll:.4f} | pen={pen_val:.4g} | "
                        f"{steps_per_sec:5.2f} it/s | eta {_fmt_eta(eta)}"
                    )
                else:
                    log(
                        f"  ep {ep:02d} step {batch_idx:>6d} | "
                        f"nll={window_nll:.4f} | pen={pen_val:.4g} | "
                        f"{steps_per_sec:5.2f} it/s"
                    )
                win_start = now
                win_loss_sum = 0.0
                win_n = 0

            if steps_per_epoch is not None and batch_idx >= steps_per_epoch:
                break
            if max_steps is not None and global_step >= max_steps:
                break

            del x, loss, nll, penalty

        avg_train_nll = total_nll / total_n if total_n else float("nan")
        avg_penalty = total_pen / total_n if total_n else float("nan")
        epoch_time = time.time() - ep_start

        val_t0 = time.time()
        if val_tensor is not None:
            val_nll = _eval_nll_tensor(raw_model, val_tensor, device)
            select_metric = val_nll
            has_val_metric = True
        elif val_loader is not None:
            val_nll = _eval_nll(raw_model, val_loader, device)
            select_metric = val_nll
            has_val_metric = True
        else:
            val_nll = float("nan")
            select_metric = avg_train_nll
            has_val_metric = False
        val_time = time.time() - val_t0

        stats = _rank_stats(raw_model)

        val_delta = None
        stop_for_val_delta = False
        if (
            early_stop_delta is not None
            and early_stop_delta > 0
            and has_val_metric
            and last_val_metric is not None
            and not math.isnan(select_metric)
        ):
            val_delta = abs(select_metric - last_val_metric)
            stop_for_val_delta = val_delta < early_stop_delta
        if has_val_metric and not math.isnan(select_metric):
            last_val_metric = select_metric

        min_delta = 0.0 if early_stop_min_delta is None else float(early_stop_min_delta)
        improved = False
        if track_best and not math.isnan(select_metric):
            improved = select_metric < (best_metric - max(0.0, min_delta))

        # Periodic full-model snapshots for monitoring how q_k evolves across
        # training; independent of the best-model save_path/save_func below.
        if (
            epoch_snapshot_func is not None
            and epoch_snapshot_every
            and (ep % epoch_snapshot_every == 0 or ep == 1)
        ):
            epoch_snapshot_func(raw_model, ep)

        if improved:
            best_metric = select_metric
            epochs_without_improvement = 0
            if track_best:
                best_state  = _cpu_state_dict(raw_model)
                best_epoch  = ep
            if save_path and save_func:
                save_func(raw_model, save_path)
        elif has_val_metric and not math.isnan(select_metric) and best_epoch > 0:
            epochs_without_improvement += 1

        val_str = f"{val_nll:.6f}" if not math.isnan(val_nll) else "n/a"
        tag = " ** best **" if improved else ""
        patience_str = ""
        if early_stop_patience is not None and early_stop_patience > 0 and has_val_metric:
            patience_str = (
                f" | no_improve={epochs_without_improvement}/"
                f"{int(early_stop_patience)}"
            )
        val_time_str = f" (val {_fmt_eta(val_time)})" if not math.isnan(val_nll) else ""
        rank_str = ""
        if stats:
            rank_str = (
                f" | beta={ard_beta:.2f} | q_eff={stats['q_eff_mean']:.2f} "
                f"[{int(stats['q_eff_min'])}..{int(stats['q_eff_max'])}] "
                f"| psi={stats['psi_mean']:.4g}"
            )
        log(
            f"[epoch {ep:02d}/{epoch_label}] done in {_fmt_eta(epoch_time)}{val_time_str} | "
            f"train_nll={avg_train_nll:.6f} | ard_pen={avg_penalty:.6g} | "
            f"val_nll={val_str} | "
            f"best_nll={best_metric:.6f} @ ep{best_epoch:02d}{tag}"
            f"{rank_str}{patience_str}"
        )

        if wandb_on:
            payload = {
                "epoch": ep,
                "ard/beta": ard_beta,
                "ard/weight": raw_model.ard_weight,
                "train/epoch_nll": avg_train_nll,
                "train/epoch_ard_penalty": avg_penalty,
                "val/nll": val_nll,
                "val/delta": val_delta,
                "val/time_s": val_time,
                "epoch/time_s": epoch_time,
                "best/metric": best_metric,
                "best/epoch": best_epoch,
                "early_stop/no_improve_epochs": epochs_without_improvement,
            }
            if stats:
                payload.update({
                    "ard/q_eff_mean": stats["q_eff_mean"],
                    "ard/q_eff_median": stats["q_eff_median"],
                    "ard/q_eff_min": stats["q_eff_min"],
                    "ard/q_eff_max": stats["q_eff_max"],
                    "ard/scale_mean": stats["scale_mean"],
                    "ard/psi_mean": stats["psi_mean"],
                    "ard/dead_components": stats["dead_components"],
                    "ard/q_eff_hist": _wandb.Histogram(stats["q_eff"].cpu().numpy()),
                })
            _wandb.log(payload, step=global_step)

        if ckpt_path:
            _atomic_torch_save({
                "epoch": ep,
                "model": raw_model.state_dict(),
                "optimizer": opt.state_dict(),
                "best_metric": best_metric,
                "best_state": best_state,
                "best_epoch": best_epoch,
                "last_val_metric": last_val_metric,
                "epochs_without_improvement": epochs_without_improvement,
                "rng_state": _rng_state(device),
                # Pins the beta ramp this run was trained under; a resume that
                # computes a different horizon is rejected rather than silently
                # re-scheduled.
                _HORIZON_KEY: horizon,
            }, ckpt_path)

        if max_steps is not None and global_step >= max_steps:
            log(f"[max-steps] reached global_step={global_step} >= max_steps={max_steps}; stopping early.")
            break
        if stop_for_val_delta:
            log(
                f"[early-stop] validation NLL changed by {val_delta:.6g} "
                f"< {early_stop_delta:.6g}; stopping at epoch {ep:02d}."
            )
            break
        if (
            early_stop_patience is not None
            and early_stop_patience > 0
            and has_val_metric
            and best_epoch > 0
            and epochs_without_improvement >= int(early_stop_patience)
        ):
            log(
                f"[early-stop] validation NLL did not improve by "
                f"{max(0.0, min_delta):.6g} for {epochs_without_improvement} "
                f"epochs; stopping at epoch {ep:02d}."
            )
            break
        ep += 1

    if track_best and best_state is not None:
        raw_model.load_state_dict(best_state)
        print(f"Restored best model from epoch {best_epoch:02d} with metric={best_metric:.6f}")
        stats = _rank_stats(raw_model)

    # Leave the model carrying its configured pressure, not the last scheduled
    # value, so what gets saved in checkpoint meta is what was asked for.
    raw_model.ard_weight = target_ard_weight

    # NOTE: no pruning here. Zeroing columns is a post-training step the caller
    # performs explicitly via `MFA_ARD.prune_columns`; doing it inside the loop
    # would silently change the model the best-epoch tracking selected.
    return dict(
        best_epoch=best_epoch,
        best_metric=best_metric,
        q_eff_mean=stats.get("q_eff_mean"),
        dead_components=stats.get("dead_components"),
    )


__all__ = ["train_nll_ard", "ard_beta_schedule"]
