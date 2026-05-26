import os
import math
import time
import torch
import torch.distributed as dist

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


def _distributed_state():
    on = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if on else 0
    return on, rank


def _unwrap_model(model):
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def _cpu_state_dict(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


@torch.no_grad()
def _eval_nll(model, loader, device):
    model.eval()
    tot_nll, tot_n = 0.0, 0
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

#TODO refactor to use pytorch lightining
def train_nll(
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
    checkpoint_all_ranks=False,
    max_steps=None,
    early_stop_delta=1e-3,
):
    """
    Train with NLL, keep the best (lowest) NLL model.

    Validation can be supplied two ways, with `val_tensor` taking priority:
    - `val_tensor`: a single pre-materialized tensor holding all validation
      rows (typically built once on rank 0, parked on GPU or in pinned CPU
      memory). Eval iterates contiguous chunks of it, so there is no shard
      I/O or worker startup per epoch. This is the fast path used by the
      shard-training CLI when the val set fits in memory.
    - `val_loader`: a regular DataLoader over the val set, re-streamed from
      disk each epoch. Fallback for val sets too large to materialize.
    If both are None, the best-model metric falls back to training NLL.
    When validation is available, training stops once the absolute change in
    validation NLL between consecutive epochs is below `early_stop_delta`.
    Passing `epochs <= 0` removes the epoch cap; in that case training runs
    until early stopping, `max_steps`, or the surrounding job limit stops it.

    Distributed-aware for component-sharded MFA: when `torch.distributed` is
    initialized, all ranks run the same train batches. Model-parallel models
    also run the same validation batches on every rank because their forward
    pass contains collectives. Rank 0 chooses the metric and broadcasts that
    decision, while every rank can checkpoint/restore its own component shard.

    If `ckpt_path` is given, a full training checkpoint (model + optimizer +
    epoch + best state) is written atomically after every epoch. On startup,
    if that file exists it is loaded and training resumes from the next epoch.
    """
    dist_on, rank = _distributed_state()
    is_main = (rank == 0)
    epoch_limit = None if epochs is None or int(epochs) <= 0 else int(epochs)
    epoch_label = "unbounded" if epoch_limit is None else f"{epoch_limit:02d}"

    raw_model = _unwrap_model(model)
    device = next(model.parameters()).device
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    keep_best_on_this_rank = track_best and (is_main or checkpoint_all_ranks)

    best_metric = float("inf")
    best_state  = _cpu_state_dict(raw_model) if keep_best_on_this_rank else None
    best_epoch  = 0
    start_epoch = 1
    last_val_metric = None

    load_ckpt = bool(ckpt_path) and os.path.exists(ckpt_path) and (
        is_main or checkpoint_all_ranks
    )
    if load_ckpt:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        raw_model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        if track_best:
            best_metric = ckpt["best_metric"]
            best_state  = ckpt["best_state"] if keep_best_on_this_rank else None
            best_epoch  = ckpt["best_epoch"]
        last_val_metric = ckpt.get("last_val_metric")
        start_epoch = ckpt["epoch"] + 1
        _restore_rng_state(ckpt.get("rng_state"), device)
        if is_main:
            if track_best:
                print(f"[ckpt] resumed from epoch {ckpt['epoch']:02d}  "
                      f"best_metric={best_metric:.6f}  best_epoch={best_epoch:02d}  "
                      f"next={start_epoch:02d}/{epoch_label}")
            else:
                print(f"[ckpt] resumed from epoch {ckpt['epoch']:02d}  "
                      f"next={start_epoch:02d}/{epoch_label}")

    if dist_on:
        starts = torch.tensor([start_epoch, start_epoch], device=device, dtype=torch.long)
        dist.all_reduce(starts[:1], op=dist.ReduceOp.MIN)
        dist.all_reduce(starts[1:], op=dist.ReduceOp.MAX)
        if int(starts[0].item()) != int(starts[1].item()):
            raise RuntimeError(
                "checkpoint epoch mismatch across ranks: "
                f"min next epoch={int(starts[0].item())}, "
                f"max next epoch={int(starts[1].item())}"
            )

    global_step = (start_epoch - 1) * (steps_per_epoch or 0)
    wandb_on = is_main and _wandb_active()

    def log(msg: str) -> None:
        if is_main:
            print(msg, flush=True)

    ep = start_epoch
    while epoch_limit is None or ep <= epoch_limit:
        model.train()
        total_nll, total_n = 0.0, 0

        ep_start = time.time()
        win_start = ep_start
        win_loss_sum = 0.0
        win_n = 0
        total_str = str(steps_per_epoch) if steps_per_epoch else "?"
        log(f"[epoch {ep:02d}/{epoch_label}] start — {total_str} steps")

        for batch_idx, batch in enumerate(loader, 1):
            x = batch[0] if isinstance(batch, (tuple, list)) else batch
            x = x.view(x.size(0), -1).to(device)
            opt.zero_grad(set_to_none=True)
            loss = model(x)
            loss.backward()

            sync_replicated_grads = getattr(raw_model, "sync_replicated_grads", None)
            if callable(sync_replicated_grads):
                sync_replicated_grads()

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            opt.step()

            B = x.size(0)
            loss_val = float(loss.item())
            total_nll += loss_val * B
            total_n   += B
            win_loss_sum += loss_val * B
            win_n += B
            global_step += 1

            if wandb_on:
                _wandb.log(
                    {"train/loss": loss_val, "epoch": ep},
                    step=global_step,
                )

            if is_main and (batch_idx % log_interval) == 0:
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
                        f"({pct:5.1f}%) | nll={window_nll:.4f} | "
                        f"{steps_per_sec:5.2f} it/s | eta {_fmt_eta(eta)}"
                    )
                else:
                    log(
                        f"  ep {ep:02d} step {batch_idx:>6d} | "
                        f"nll={window_nll:.4f} | {steps_per_sec:5.2f} it/s"
                    )
                win_start = now
                win_loss_sum = 0.0
                win_n = 0

            if steps_per_epoch is not None and batch_idx >= steps_per_epoch:
                break
            if max_steps is not None and global_step >= max_steps:
                break

            del x, loss

        avg_train_nll = total_nll / total_n if total_n else float("nan")
        epoch_time = time.time() - ep_start

        # Validation: every rank participates so model-parallel collectives
        # inside `model.nll` (e.g. ComponentShardedMFA's distributed
        # logsumexp) complete symmetrically. Callers must therefore supply
        # `val_tensor` / `val_loader` on every rank, or on none.
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

        if dist_on:
            t = torch.tensor([select_metric], device=device, dtype=torch.float64)
            dist.broadcast(t, src=0)
            select_metric = float(t[0].item())
            has_val = torch.tensor([int(has_val_metric)], device=device, dtype=torch.long)
            dist.broadcast(has_val, src=0)
            has_val_metric = bool(has_val.item())

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

        improved = (
            track_best and (select_metric < best_metric)
            if not math.isnan(select_metric) else False
        )
        if improved:
            best_metric = select_metric
            if keep_best_on_this_rank:
                best_state  = _cpu_state_dict(raw_model)
                best_epoch  = ep
            if is_main:
                if save_path and save_func:
                    save_func(raw_model, save_path)

        if is_main:
            val_str = f"{val_nll:.6f}" if not math.isnan(val_nll) else "n/a"
            tag = " ** best **" if improved else ""
            val_time_str = (
                f" (val {_fmt_eta(val_time)})" if not math.isnan(val_nll) else ""
            )
            log(
                f"[epoch {ep:02d}/{epoch_label}] done in {_fmt_eta(epoch_time)}{val_time_str} | "
                f"train_nll={avg_train_nll:.6f} | val_nll={val_str} | "
                f"best_nll={best_metric:.6f} @ ep{best_epoch:02d}{tag}"
            )

        if wandb_on:
            _wandb.log(
                {
                    "epoch": ep,
                    "train/epoch_nll": avg_train_nll,
                    "val/nll": val_nll,
                    "val/delta": val_delta,
                    "val/time_s": val_time,
                    "epoch/time_s": epoch_time,
                    "best/metric": best_metric,
                    "best/epoch": best_epoch,
                },
                step=global_step,
            )

        if ckpt_path and (is_main or checkpoint_all_ranks):
            _atomic_torch_save({
                "epoch": ep,
                "model": raw_model.state_dict(),
                "optimizer": opt.state_dict(),
                "best_metric": best_metric,
                "best_state": best_state,
                "best_epoch": best_epoch,
                "last_val_metric": last_val_metric,
                "rng_state": _rng_state(device),
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
        ep += 1

    if keep_best_on_this_rank and best_state is not None:
        raw_model.load_state_dict(best_state)
        if is_main:
            print(f"Restored best model from epoch {best_epoch:02d} with metric={best_metric:.6f}")

    return dict(best_epoch=best_epoch, best_metric=best_metric)
