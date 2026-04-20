import os
import math
import torch
import torch.nn.functional as F
import torch.distributed as dist
from modeling.mfa import save_mfa
from tqdm import tqdm


def _ddp_state():
    on = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if on else 0
    world = dist.get_world_size() if on else 1
    return on, rank, world


def _cpu_state_dict(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


@torch.no_grad()
def _eval_nll(model, loader, device):
    model.eval()
    tot_nll, tot_n = 0.0, 0
    for x, _ in loader:
        x = x.view(x.size(0), -1).to(device)
        nll = model.nll(x)                  # mean over batch
        B = x.size(0)
        tot_nll += nll.item() * B
        tot_n   += B
    return tot_nll / tot_n


@torch.no_grad()
def _eval_nll_tensor(model, X, device, chunk=8192):
    """Chunk-evaluate NLL on a tensor already on `device` (or move here)."""
    model.eval()
    X = X.to(device, non_blocking=True)
    N = X.shape[0]
    tot = 0.0
    for i in range(0, N, chunk):
        xb = X[i:i + chunk].view(X[i:i + chunk].size(0), -1)
        tot += float(model.nll(xb).item()) * xb.size(0)
    return tot / max(N, 1)


def _atomic_torch_save(obj, path):
    tmp = f"{path}.tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


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
):
    """
    Train with NLL, keep the best (lowest) NLL model.

    DDP-aware: if `torch.distributed` is initialized, all ranks participate
    in forward/backward (grads are all-reduced by DDP), but only rank 0 runs
    validation, tracks best-state, writes the model/checkpoint and prints.

    If `ckpt_path` is given, a full training checkpoint (model + optimizer +
    epoch + best state) is written atomically after every epoch. On startup,
    if that file exists it is loaded and training resumes from the next epoch.
    """
    ddp_on, rank, world = _ddp_state()
    is_main = (rank == 0)

    raw_model = model.module if hasattr(model, "module") else model
    device = next(model.parameters()).device
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best_metric = float("inf")
    best_state  = _cpu_state_dict(raw_model) if is_main else None
    best_epoch  = 0
    start_epoch = 1

    if ckpt_path and is_main and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        raw_model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        best_metric = ckpt["best_metric"]
        best_state  = ckpt["best_state"]
        best_epoch  = ckpt["best_epoch"]
        start_epoch = ckpt["epoch"] + 1
        print(f"[ckpt] resumed from epoch {ckpt['epoch']:02d}  "
              f"best_metric={best_metric:.6f}  best_epoch={best_epoch:02d}  "
              f"next={start_epoch:02d}/{epochs:02d}")

    # Sync params + resume metadata across ranks after the (possible) load.
    if ddp_on:
        for p in model.parameters():
            dist.broadcast(p.data, src=0)
        meta = torch.tensor([start_epoch, best_epoch, best_metric],
                            device=device, dtype=torch.float64)
        dist.broadcast(meta, src=0)
        start_epoch = int(meta[0].item())
        best_epoch = int(meta[1].item())
        best_metric = float(meta[2].item())

    for ep in range(start_epoch, epochs + 1):
        model.train()
        total_nll, total_n = 0.0, 0

        iterable = enumerate(loader, 1)
        pbar = tqdm(iterable, total=steps_per_epoch, disable=not is_main)

        for batch_idx, batch in pbar:
            x = batch[0] if isinstance(batch, (tuple, list)) else batch
            x = x.view(x.size(0), -1).to(device)
            opt.zero_grad(set_to_none=True)
            loss = model(x)     # goes through DDP.forward → MFA.forward = nll
            loss.backward()

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            opt.step()

            B = x.size(0)
            total_nll += float(loss.item()) * B
            total_n   += B

            if is_main and (batch_idx % log_interval) == 0:
                avg_so_far = total_nll / max(1, total_n)
                pbar.set_description(
                    f"Epoch {ep:02d} | Step {batch_idx:06d} Train NLL={avg_so_far:.6f}"
                )

            if steps_per_epoch is not None and batch_idx >= steps_per_epoch:
                break

            del x, loss

        # Aggregate train NLL across ranks for reporting.
        if ddp_on:
            t = torch.tensor([total_nll, float(total_n)],
                             device=device, dtype=torch.float64)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            total_nll = float(t[0].item())
            total_n   = int(t[1].item())

        avg_train_nll = total_nll / total_n if total_n else float("nan")

        # Validation: run only on rank 0 (others provide a placeholder).
        if is_main:
            if val_tensor is not None:
                val_nll = _eval_nll_tensor(raw_model, val_tensor, device)
                select_metric = val_nll
            elif val_loader is not None:
                val_nll = _eval_nll(raw_model, val_loader, device)
                select_metric = val_nll
            else:
                val_nll = float("nan")
                select_metric = avg_train_nll
        else:
            val_nll = float("nan")
            select_metric = float("nan")

        # Broadcast the selection metric so all ranks agree on the decision.
        if ddp_on:
            t = torch.tensor([select_metric], device=device, dtype=torch.float64)
            dist.broadcast(t, src=0)
            select_metric = float(t[0].item())

        improved = (select_metric < best_metric) if not math.isnan(select_metric) else False
        if improved:
            best_metric = select_metric
            if is_main:
                best_state  = _cpu_state_dict(raw_model)
                best_epoch  = ep
                if save_path and save_func:
                    save_func(raw_model, save_path)

        if is_main:
            print(
                f"[epoch {ep:02d}] "
                f"train NLL={avg_train_nll:.6f}  "
                f"val NLL={val_nll:.6f} "
                f"{'** best **' if improved else ''}"
            )

        if ckpt_path and is_main:
            _atomic_torch_save({
                "epoch": ep,
                "model": raw_model.state_dict(),
                "optimizer": opt.state_dict(),
                "best_metric": best_metric,
                "best_state": best_state,
                "best_epoch": best_epoch,
            }, ckpt_path)

    if is_main and best_state is not None:
        raw_model.load_state_dict(best_state)
        print(f"Restored best model from epoch {best_epoch:02d} with metric={best_metric:.6f}")

    return dict(best_epoch=best_epoch, best_metric=best_metric)
