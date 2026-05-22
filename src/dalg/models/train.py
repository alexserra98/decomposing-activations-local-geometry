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

    raw_model = _unwrap_model(model)
    device = next(model.parameters()).device
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    keep_best_on_this_rank = track_best and (is_main or checkpoint_all_ranks)

    best_metric = float("inf")
    best_state  = _cpu_state_dict(raw_model) if keep_best_on_this_rank else None
    best_epoch  = 0
    start_epoch = 1

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
        start_epoch = ckpt["epoch"] + 1
        _restore_rng_state(ckpt.get("rng_state"), device)
        if is_main:
            if track_best:
                print(f"[ckpt] resumed from epoch {ckpt['epoch']:02d}  "
                      f"best_metric={best_metric:.6f}  best_epoch={best_epoch:02d}  "
                      f"next={start_epoch:02d}/{epochs:02d}")
            else:
                print(f"[ckpt] resumed from epoch {ckpt['epoch']:02d}  "
                      f"next={start_epoch:02d}/{epochs:02d}")

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

    for ep in range(start_epoch, epochs + 1):
        model.train()
        total_nll, total_n = 0.0, 0

        ep_start = time.time()
        win_start = ep_start
        win_loss_sum = 0.0
        win_n = 0
        total_str = str(steps_per_epoch) if steps_per_epoch else "?"
        log(f"[epoch {ep:02d}/{epochs:02d}] start — {total_str} steps")

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
        elif val_loader is not None:
            val_nll = _eval_nll(raw_model, val_loader, device)
            select_metric = val_nll
        else:
            val_nll = float("nan")
            select_metric = avg_train_nll
        val_time = time.time() - val_t0

        if dist_on:
            t = torch.tensor([select_metric], device=device, dtype=torch.float64)
            dist.broadcast(t, src=0)
            select_metric = float(t[0].item())

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
                f"[epoch {ep:02d}/{epochs:02d}] done in {_fmt_eta(epoch_time)}{val_time_str} | "
                f"train_nll={avg_train_nll:.6f} | val_nll={val_str} | "
                f"best_nll={best_metric:.6f} @ ep{best_epoch:02d}{tag}"
            )

        if wandb_on:
            _wandb.log(
                {
                    "epoch": ep,
                    "train/epoch_nll": avg_train_nll,
                    "val/nll": val_nll,
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
                "rng_state": _rng_state(device),
            }, ckpt_path)

        if max_steps is not None and global_step >= max_steps:
            log(f"[max-steps] reached global_step={global_step} >= max_steps={max_steps}; stopping early.")
            break

    if keep_best_on_this_rank and best_state is not None:
        raw_model.load_state_dict(best_state)
        if is_main:
            print(f"Restored best model from epoch {best_epoch:02d} with metric={best_metric:.6f}")

    return dict(best_epoch=best_epoch, best_metric=best_metric)


def _vae_batch_loss(model, x, *, beta):
    raw_model = _unwrap_model(model)
    old_beta = raw_model.beta
    raw_model.beta = float(beta)
    try:
        out = model(x)
        losses = raw_model.loss(x, out)
    finally:
        raw_model.beta = old_beta
    return losses


def _vae_beta(raw_model, global_step: int, beta_warmup_steps: int) -> float:
    target = float(raw_model.beta)
    if beta_warmup_steps <= 0:
        return target
    progress = min(1.0, float(global_step + 1) / float(beta_warmup_steps))
    return target * progress


@torch.no_grad()
def _eval_vae_loader(model, loader, device, *, beta):
    from dalg.models.vae import adapt_activation_batch

    raw_model = _unwrap_model(model)
    model.eval()
    tot_loss, tot_rec, tot_kl, tot_n = 0.0, 0.0, 0.0, 0
    for batch in loader:
        x = adapt_activation_batch(batch, input_dim=raw_model.input_dim)
        x = x.to(device, non_blocking=True).float()
        losses = _vae_batch_loss(model, x, beta=beta)
        B = x.size(0)
        tot_loss += float(losses["loss"].item()) * B
        tot_rec += float(losses["rec_loss"].item()) * B
        tot_kl += float(losses["kl_loss"].item()) * B
        tot_n += B
    denom = max(tot_n, 1)
    return {
        "loss": tot_loss / denom,
        "rec_loss": tot_rec / denom,
        "kl_loss": tot_kl / denom,
    }


@torch.no_grad()
def _eval_vae_tensor(model, X, device, *, beta, chunk=8192):
    raw_model = _unwrap_model(model)
    model.eval()
    N = X.shape[0]
    tot_loss, tot_rec, tot_kl = 0.0, 0.0, 0.0
    for i in range(0, N, chunk):
        xb = X[i:i + chunk].to(device, non_blocking=True).float()
        xb = xb.reshape(xb.size(0), -1)
        if xb.shape[-1] != raw_model.input_dim:
            raise ValueError(f"expected activation dim {raw_model.input_dim}, got {xb.shape[-1]}")
        losses = _vae_batch_loss(model, xb, beta=beta)
        B = xb.size(0)
        tot_loss += float(losses["loss"].item()) * B
        tot_rec += float(losses["rec_loss"].item()) * B
        tot_kl += float(losses["kl_loss"].item()) * B
    denom = max(N, 1)
    return {
        "loss": tot_loss / denom,
        "rec_loss": tot_rec / denom,
        "kl_loss": tot_kl / denom,
    }


def train_vae(
    model,
    loader,
    *,
    val_loader=None,
    val_tensor=None,
    epochs=5,
    lr=1e-3,
    weight_decay=1e-4,
    grad_clip=None,
    save_path=None,
    save_func=None,
    log_interval=100,
    steps_per_epoch=None,
    ckpt_path=None,
    track_best=True,
    max_steps=None,
    beta_warmup_steps=0,
    val_chunk_size=8192,
):
    """Train a VAE on activation batches from ``ActivationBatchDataset``.

    The loader is expected to yield already-batched activation tensors, with
    optional metadata tuples such as ``(x, global_rows, tok_pos)``. Validation
    follows the same convention as ``train_nll``: a materialized ``val_tensor``
    takes priority over a streaming ``val_loader``.
    """
    from dalg.models.vae import adapt_activation_batch, save_vae

    dist_on, rank = _distributed_state()
    if dist_on:
        raise RuntimeError("train_vae currently supports single-process training only")
    is_main = rank == 0

    raw_model = _unwrap_model(model)
    device = next(model.parameters()).device
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    save_func = save_vae if save_func is None else save_func

    best_metric = float("inf")
    best_state = _cpu_state_dict(raw_model) if track_best else None
    best_epoch = 0
    start_epoch = 1

    if ckpt_path and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        raw_model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        if track_best:
            best_metric = ckpt["best_metric"]
            best_state = ckpt["best_state"]
            best_epoch = ckpt["best_epoch"]
        start_epoch = ckpt["epoch"] + 1
        _restore_rng_state(ckpt.get("rng_state"), device)
        if is_main:
            print(
                f"[vae ckpt] resumed from epoch {ckpt['epoch']:02d} "
                f"next={start_epoch:02d}/{epochs:02d}"
            )

    global_step = (start_epoch - 1) * (steps_per_epoch or 0)
    wandb_on = is_main and _wandb_active()

    def log(msg: str) -> None:
        if is_main:
            print(msg, flush=True)

    for ep in range(start_epoch, epochs + 1):
        model.train()
        total_loss, total_rec, total_kl, total_n = 0.0, 0.0, 0.0, 0

        ep_start = time.time()
        win_start = ep_start
        win_loss_sum = 0.0
        win_n = 0
        total_str = str(steps_per_epoch) if steps_per_epoch else "?"
        log(f"[vae epoch {ep:02d}/{epochs:02d}] start - {total_str} steps")

        for batch_idx, batch in enumerate(loader, 1):
            x = adapt_activation_batch(batch, input_dim=raw_model.input_dim)
            x = x.to(device, non_blocking=True).float()

            beta = _vae_beta(raw_model, global_step, int(beta_warmup_steps))
            opt.zero_grad(set_to_none=True)
            losses = _vae_batch_loss(model, x, beta=beta)
            loss = losses["loss"]
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite VAE loss: loss={loss.item()} "
                    f"rec={losses['rec_loss'].item()} kl={losses['kl_loss'].item()}"
                )
            loss.backward()

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            opt.step()

            B = x.size(0)
            loss_val = float(loss.item())
            rec_val = float(losses["rec_loss"].item())
            kl_val = float(losses["kl_loss"].item())
            total_loss += loss_val * B
            total_rec += rec_val * B
            total_kl += kl_val * B
            total_n += B
            win_loss_sum += loss_val * B
            win_n += B
            global_step += 1

            if wandb_on:
                _wandb.log(
                    {
                        "train/loss": loss_val,
                        "train/rec": rec_val,
                        "train/kl": kl_val,
                        "train/beta": beta,
                        "epoch": ep,
                    },
                    step=global_step,
                )

            if is_main and (batch_idx % log_interval) == 0:
                now = time.time()
                win_dt = max(now - win_start, 1e-6)
                steps_per_sec = log_interval / win_dt
                window_loss = win_loss_sum / max(1, win_n)
                if steps_per_epoch is not None:
                    remaining = steps_per_epoch - batch_idx
                    eta = remaining / steps_per_sec if steps_per_sec > 0 else float("inf")
                    pct = 100.0 * batch_idx / steps_per_epoch
                    log(
                        f"  vae ep {ep:02d} step {batch_idx:>6d}/{steps_per_epoch} "
                        f"({pct:5.1f}%) | loss={window_loss:.4f} | "
                        f"{steps_per_sec:5.2f} it/s | eta {_fmt_eta(eta)}"
                    )
                else:
                    log(
                        f"  vae ep {ep:02d} step {batch_idx:>6d} | "
                        f"loss={window_loss:.4f} | {steps_per_sec:5.2f} it/s"
                    )
                win_start = now
                win_loss_sum = 0.0
                win_n = 0

            if steps_per_epoch is not None and batch_idx >= steps_per_epoch:
                break
            if max_steps is not None and global_step >= max_steps:
                break

            del x, loss, losses

        denom = max(total_n, 1)
        avg_train_loss = total_loss / denom
        avg_train_rec = total_rec / denom
        avg_train_kl = total_kl / denom
        epoch_time = time.time() - ep_start

        val_t0 = time.time()
        eval_beta = float(raw_model.beta)
        if val_tensor is not None:
            val_metrics = _eval_vae_tensor(
                model,
                val_tensor,
                device,
                beta=eval_beta,
                chunk=val_chunk_size,
            )
            select_metric = val_metrics["loss"]
        elif val_loader is not None:
            val_metrics = _eval_vae_loader(model, val_loader, device, beta=eval_beta)
            select_metric = val_metrics["loss"]
        else:
            val_metrics = None
            select_metric = avg_train_loss
        val_time = time.time() - val_t0

        improved = track_best and select_metric < best_metric
        if improved:
            best_metric = select_metric
            best_state = _cpu_state_dict(raw_model)
            best_epoch = ep
            if save_path:
                save_func(raw_model, save_path)

        if is_main:
            if val_metrics is None:
                val_str = "n/a"
                val_time_str = ""
            else:
                val_str = f"{val_metrics['loss']:.6f}"
                val_time_str = f" (val {_fmt_eta(val_time)})"
            tag = " ** best **" if improved else ""
            log(
                f"[vae epoch {ep:02d}/{epochs:02d}] done in {_fmt_eta(epoch_time)}{val_time_str} | "
                f"train_loss={avg_train_loss:.6f} | train_rec={avg_train_rec:.6f} | "
                f"train_kl={avg_train_kl:.6f} | val_loss={val_str} | "
                f"best_loss={best_metric:.6f} @ ep{best_epoch:02d}{tag}"
            )

        if wandb_on:
            payload = {
                "epoch": ep,
                "train/epoch_loss": avg_train_loss,
                "train/epoch_rec": avg_train_rec,
                "train/epoch_kl": avg_train_kl,
                "val/time_s": val_time,
                "epoch/time_s": epoch_time,
                "best/metric": best_metric,
                "best/epoch": best_epoch,
            }
            if val_metrics is not None:
                payload.update({
                    "val/loss": val_metrics["loss"],
                    "val/rec": val_metrics["rec_loss"],
                    "val/kl": val_metrics["kl_loss"],
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
                "rng_state": _rng_state(device),
            }, ckpt_path)

        if max_steps is not None and global_step >= max_steps:
            log(f"[vae max-steps] reached global_step={global_step} >= max_steps={max_steps}; stopping early.")
            break

    if track_best and best_state is not None:
        raw_model.load_state_dict(best_state)
        if is_main:
            print(f"Restored best VAE from epoch {best_epoch:02d} with metric={best_metric:.6f}")

    if save_path:
        save_func(raw_model, save_path)

    return dict(best_epoch=best_epoch, best_metric=best_metric)
