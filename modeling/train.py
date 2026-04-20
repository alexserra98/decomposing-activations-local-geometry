import os
import math
import torch
import torch.nn.functional as F
from modeling.mfa import save_mfa
from tqdm import tqdm

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

    If `ckpt_path` is given, a full training checkpoint (model + optimizer +
    epoch + best state) is written atomically after every epoch. On startup,
    if that file exists it is loaded and training resumes from the next epoch.
    """
    device = next(model.parameters()).device
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best_metric = float("inf")
    best_state  = _cpu_state_dict(model)
    best_epoch  = 0
    start_epoch = 1

    if ckpt_path and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        best_metric = ckpt["best_metric"]
        best_state  = ckpt["best_state"]
        best_epoch  = ckpt["best_epoch"]
        start_epoch = ckpt["epoch"] + 1
        print(f"[ckpt] resumed from epoch {ckpt['epoch']:02d}  "
              f"best_metric={best_metric:.6f}  best_epoch={best_epoch:02d}  "
              f"next={start_epoch:02d}/{epochs:02d}")

    for ep in range(start_epoch, epochs + 1):
        model.train()
        total_nll, total_n = 0.0, 0

        iterable = enumerate(loader, 1)
        pbar = tqdm(iterable, total=steps_per_epoch)


        for batch_idx, batch in pbar:
            x = batch[0] if isinstance(batch, (tuple, list)) else batch
            x = x.view(x.size(0), -1).to(device)
            opt.zero_grad(set_to_none=True)
            nll = model.nll(x)     # mean over batch
            nll.backward()

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            opt.step()

            B = x.size(0)
            total_nll += float(nll.item()) * B
            total_n   += B

            if (batch_idx % log_interval) == 0:
                avg_so_far = total_nll / max(1, total_n)
                pbar.set_description(f"Epoch {ep:02d} | Step {batch_idx:06d} Train NLL={avg_so_far:.6f}")

            if steps_per_epoch is not None and batch_idx >= steps_per_epoch:
                break

            # free ASAP
            del x, nll

        if total_n == 0:
            avg_train_nll = float("nan")
        else:
            avg_train_nll = total_nll / total_n

        if val_tensor is not None:
            val_nll = _eval_nll_tensor(model, val_tensor, device)
            select_metric = val_nll
        elif val_loader is not None:
            val_nll = _eval_nll(model, val_loader, device)
            select_metric = val_nll
        else:
            val_nll = float("nan")
            select_metric = avg_train_nll

        improved = (select_metric < best_metric) if not math.isnan(select_metric) else False
        if improved:
            best_metric = select_metric
            best_state  = _cpu_state_dict(model)
            best_epoch  = ep
            if save_path and save_func:
                save_func(model, save_path)
        print(
            f"[epoch {ep:02d}] "
            f"train NLL={avg_train_nll:.6f}  "
            f"val NLL={val_nll:.6f} "
            f"{'** best **' if improved else ''}"
        )

        if ckpt_path:
            _atomic_torch_save({
                "epoch": ep,
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "best_metric": best_metric,
                "best_state": best_state,
                "best_epoch": best_epoch,
            }, ckpt_path)

    model.load_state_dict(best_state)
    print(f"Restored best model from epoch {best_epoch:02d} with metric={best_metric:.6f}")

    return dict(best_epoch=best_epoch, best_metric=best_metric)
