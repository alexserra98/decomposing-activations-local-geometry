"""Periodic HDDC-style covariance surgery for MFA, giving adaptive per-component rank.

SGD training of `train_nll` is left untouched. Every `T` epochs this module
applies the closed-form covariance update of the HDDC model `[a_ij b_i Q_i d_i]`
(Bouveyron, Girard & Schmid, arXiv:math/0604064) to re-estimate each component's
covariance with an adaptive rank `d_k <= q_max`, and rewrites the result in MFA
parameters. Between surgeries the columns beyond `d_k` are hard-masked by
`MFA.rank_mask`, so they contribute nothing to the likelihood and receive no
gradient.

Surgery touches covariances only — `dir_raw`, `scale_rho`, `psi_rho`,
`rank_mask`. `mu` and `pi_logits` stay whatever SGD has made them.

Three phases:

A. One E-pass over the train loader accumulating, in float64, the
   responsibility-weighted second moment of each component about its *current*
   model mean `mu_k`. Centering on `mu_k` rather than on the empirical
   responsibility-weighted mean is deliberate: `mu_k` is retained, so this is
   the ML covariance given the fixed mean — the coherent partial M-step. Pairing
   a covariance centered at `mu_hat_k` with a retained `mu_k` would leak the
   mean shift into the eigen-spectrum and inflate the apparent rank whenever the
   SGD means lag the data.

B. Per component: `eigh(S_k)`, a scale-free Cattell scree test on consecutive
   eigenvalue differences to pick `d_k`, the HDDC noise level
   `b_k = (Tr(S_k) - sum_{j<=d_k} lam_j) / (D - d_k)`, then the MFA
   reconstruction of `Sigma_k = W_k W_k^T + b_k I` with
   `scale_j = sqrt(lam_j - b_k)`. All `q_max` columns are written from the
   eigendecomposition and only the mask records `d_k`, so a later surgery can
   raise a component's rank with no revival logic.

C. Adam state for the rewritten tensors is dropped by the caller
   (`reset_optimizer_state`), optionally followed by a short LR warmup.

The whole path is a parallel stack that leaves the production files untouched:
`mfa_hddc.py`, `train_hddc.py`, this module, and
`dalg/cli/adaptive_q/run_training_hddc.py`. Removing the feature means deleting
those four files and the `dalg-run-training-hddc` entry in `pyproject.toml`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .mfa_hddc import ComponentShardedMFA_HDDC, _distributed_logsumexp


@dataclass
class SurgeryConfig:
    """Everything the periodic surgery needs. `every <= 0` disables it."""

    enabled: bool = False
    every: int = 0
    threshold: float = 0.01      # Cattell t, relative to lam_1 (scale-free)
    min_count: float = 0.0       # n_min in effective points; <=0 => max(5*q, 50)
    warmup_steps: int = 0        # linear LR warmup after each surgery
    psi_floor: float = 1e-6      # lower clamp on b_k
    eps: float = 1e-12           # clamp inside sqrt(lam_j - b_k)
    max_batches: Optional[int] = None  # cap the E-pass length (debug/smoke)

    def n_min(self, q: int) -> float:
        if self.min_count and self.min_count > 0:
            return float(self.min_count)
        return float(max(5 * q, 50))

    def active_at(self, epoch: int) -> bool:
        return bool(
            self.enabled and self.every > 0 and epoch > 0 and epoch % self.every == 0
        )


# Optimizer hygiene (phase C)


def surgery_params(model) -> List[torch.nn.Parameter]:
    """The parameters surgery rewrites, i.e. the ones whose Adam state is stale."""
    return [model.dir_raw, model.scale_rho, model.psi_rho]


def reset_optimizer_state(optimizer, params) -> int:
    """Drop `exp_avg`/`exp_avg_sq`/`step` for `params`; Adam re-initializes lazily.

    State for parameters not listed (`mu`, `pi_logits`) is preserved.
    """
    dropped = 0
    for p in params:
        if optimizer.state.pop(p, None) is not None:
            dropped += 1
    return dropped


# Phase A — streaming statistics


@torch.no_grad()
def _responsibilities(model, x: torch.Tensor) -> torch.Tensor:
    """(B, K_local) posterior component probabilities, globally normalized.

    For `ComponentShardedMFA_HDDC` the normalizer spans components held by other
    ranks, so it needs the same all-reduced logsumexp the NLL uses.
    """
    ll = model.log_prob_components(x)
    if isinstance(model, ComponentShardedMFA_HDDC):
        num = ll + model.local_log_pi()[None, :]
        den = _distributed_logsumexp(num, dim=1)
        return (num - den[:, None]).exp()
    log_pi = F.log_softmax(model.pi_logits, dim=0)
    return F.softmax(ll + log_pi[None, :], dim=1)


@torch.no_grad()
def accumulate_statistics(
    model,
    loader,
    *,
    device,
    max_batches: Optional[int] = None,
    chunk_elems: int = 1 << 23,
):
    """One E-pass: returns `(N_k, S_acc_k, n_rows)` in float64.

    `N_k = sum_n r_nk` and `S_acc_k = sum_n r_nk (x_n - mu_k)(x_n - mu_k)^T`,
    both for this rank's local components. The `(K, D, D)` accumulator is 65 KB
    per component at D=128; the large-D path (trace accumulation plus a
    randomized sketch for the top q_max+1 eigenpairs) is deliberately not
    implemented yet.

    TODO(large-D): at D ~ 2304 replace the explicit (K, D, D) accumulator with
    `sum_n r_nk ||x_n - mu_k||^2` for the trace plus top-(q_max+1) eigenpairs
    from a randomized sketch of the responsibility-weighted centered data.
    """
    was_training = model.training
    model.eval()
    K, D = model.K, model.D
    N = torch.zeros(K, dtype=torch.float64, device=device)
    S = torch.zeros(K, D, D, dtype=torch.float64, device=device)
    mu = model.mu.detach().to(torch.float64)
    n_rows = 0

    # Centering is done explicitly per component rather than by expanding
    # sum r x x^T - m mu^T - mu m^T + N mu mu^T: the expanded form cancels
    # ||mu||^2-sized terms against a much smaller covariance.
    rows_per_chunk = max(1, chunk_elems // max(1, K * D))

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        x = x.view(x.size(0), -1).to(device)
        r = _responsibilities(model, x).to(torch.float64)      # (B, K)
        x64 = x.to(torch.float64)
        N += r.sum(dim=0)
        for s in range(0, x64.shape[0], rows_per_chunk):
            xc = x64[s:s + rows_per_chunk, None, :] - mu[None, :, :]   # (b, K, D)
            rxc = r[s:s + rows_per_chunk, :, None] * xc                # (b, K, D)
            S += torch.einsum("bkd,bke->kde", rxc, xc)
        n_rows += x.shape[0]

    if was_training:
        model.train()
    return N, S, n_rows


# Phase B — rank selection and reconstruction


def _softplus_inverse(y: torch.Tensor) -> torch.Tensor:
    """rho with softplus(rho) == y, for y > 0. Stable at both ends."""
    y = y.clamp_min(1e-12)
    return y + torch.log(-torch.expm1(-y))


@torch.no_grad()
def reconstruct_components(model, N, S_acc, cfg: SurgeryConfig) -> Dict[str, Any]:
    """Rewrite each eligible component's covariance from its scatter matrix.

    Components with `N_k < n_min` keep every parameter and their mask untouched.
    """
    K, D, q = model.K, model.D, model.q
    device = model.mu.device
    n_min = cfg.n_min(q)

    d_k = model.rank_mask.sum(-1).to(torch.int64)          # unchanged where skipped
    b_k = torch.full((K,), float("nan"), dtype=torch.float64, device=device)

    eligible = (N >= n_min) & torch.isfinite(N)
    idx = eligible.nonzero(as_tuple=True)[0]
    if idx.numel() == 0:
        return {
            "d_k": d_k, "b_k": b_k, "eligible": eligible,
            "n_updated": 0, "n_skipped": int(K), "N_k": N,
        }

    S = S_acc[idx] / N[idx][:, None, None]                 # (n, D, D)
    S = 0.5 * (S + S.transpose(-1, -2))                    # symmetrize fp noise
    trace = torch.diagonal(S, dim1=-2, dim2=-1).sum(-1)    # (n,)

    lam_asc, Q_asc = torch.linalg.eigh(S)                  # ascending
    lam = lam_asc.flip(-1)                                 # (n, D) descending
    Q = Q_asc.flip(-1)                                     # (n, D, D) matching columns

    # Cattell scree test on consecutive differences, normalized by lam_1 so the
    # threshold is scale-free: d_k = max{ j <= q_max : (lam_j - lam_j+1)/lam_1 > t }.
    lam_top = lam[:, :q + 1].clamp_min(0.0)                # (n, q+1)
    denom = lam[:, 0].clamp_min(torch.finfo(lam.dtype).tiny)
    delta = (lam_top[:, :-1] - lam_top[:, 1:]) / denom[:, None]   # (n, q)
    above = delta > float(cfg.threshold)
    j = torch.arange(1, q + 1, device=device, dtype=torch.int64)[None, :]
    d_sel = torch.where(above, j, torch.zeros_like(j)).max(dim=1).values
    # Empty selection -> 1; never exceed q_max, and never leave zero noise dims.
    d_sel = d_sel.clamp(min=1, max=min(q, D - 1))

    # HDDC eq. 4: the noise level is the mean of the discarded eigenvalues, so
    # b_k > 0 and lam_j >= b_k for every retained j hold by construction.
    head = torch.cumsum(lam[:, :q].clamp_min(0.0), dim=1)
    kept = head.gather(1, (d_sel - 1)[:, None]).squeeze(1)
    b = (trace - kept) / (D - d_sel).to(trace.dtype)
    b = b.clamp_min(float(cfg.psi_floor))

    # Sigma_k = W_k W_k^T + b_k I with W_k's columns the top eigenvectors scaled
    # by sqrt(lam_j - b_k). All q_max columns are written; only the mask records
    # d_k, so a later surgery can raise the rank with no revival logic.
    scale = (lam[:, :q] - b[:, None]).clamp_min(float(cfg.eps)).sqrt()

    dtype = model.dir_raw.dtype
    model.dir_raw.data[idx] = Q[:, :, :q].to(dtype)
    model.scale_rho.data[idx] = _softplus_inverse(scale).to(dtype)
    # _psi() is softplus(psi_rho) + eps_floor, so invert against the offset.
    psi_target = (b - model._eps).clamp_min(1e-12)
    model.psi_rho.data[idx] = _softplus_inverse(psi_target).to(dtype)[:, None]
    model.rank_mask.data[idx] = (j <= d_sel[:, None]).to(model.rank_mask.dtype)

    d_k = d_k.clone()
    d_k[idx] = d_sel
    b_k[idx] = b

    return {
        "d_k": d_k,
        "b_k": b_k,
        "eligible": eligible,
        "n_updated": int(idx.numel()),
        "n_skipped": int(K - idx.numel()),
        "N_k": N,
    }


# Reporting


def _summarize(stats: Dict[str, Any], q: int, device) -> Dict[str, Any]:
    """Reduce per-component tensors to scalars, all-reduced when sharded."""
    d_k = stats["d_k"]
    b_k = stats["b_k"]
    eligible = stats["eligible"]

    hist = torch.bincount(d_k.clamp(0, q), minlength=q + 1).to(torch.float64)
    finite_b = b_k[torch.isfinite(b_k)]
    totals = torch.tensor(
        [
            float(d_k.numel()),
            float(d_k.sum().item()),
            float(stats["n_updated"]),
            float(stats["n_skipped"]),
            float(finite_b.sum().item()) if finite_b.numel() else 0.0,
            float(finite_b.numel()),
            float((d_k == q).sum().item()),
            float(stats["N_k"][eligible].sum().item()) if eligible.any() else 0.0,
        ],
        dtype=torch.float64,
        device=device,
    )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        hist = hist.to(device)
        dist.all_reduce(hist, op=dist.ReduceOp.SUM)

    K_total = max(1.0, float(totals[0].item()))
    n_b = max(1.0, float(totals[5].item()))
    # Order statistics come from the (all-reduced) histogram so they describe
    # every component, not just this rank's shard.
    counts = hist.cpu()
    occupied = counts.nonzero(as_tuple=True)[0]
    cumulative = counts.cumsum(0)
    median_idx = int((cumulative >= counts.sum() / 2.0).nonzero(as_tuple=True)[0][0]) \
        if counts.sum() > 0 else 0
    return {
        "d_k_mean": float(totals[1].item()) / K_total,
        "d_k_median": median_idx,
        "d_k_min": int(occupied[0].item()) if occupied.numel() else 0,
        "d_k_max": int(occupied[-1].item()) if occupied.numel() else 0,
        "d_k_hist": [int(v) for v in counts.tolist()],
        "b_k_mean": float(totals[4].item()) / n_b,
        "n_updated": int(totals[2].item()),
        "n_skipped": int(totals[3].item()),
        "n_components": int(totals[0].item()),
        "saturated_frac": float(totals[6].item()) / K_total,
    }


# Entry point


def hddc_surgery(model, loader, cfg: SurgeryConfig, *, device=None, log=None) -> Dict[str, Any]:
    """Run phases A and B in place. Returns a summary dict for logging.

    The caller is responsible for phase C (`reset_optimizer_state` on
    `surgery_params(model)`) and for any post-surgery LR warmup.
    """
    if not getattr(model, "isotropic_psi", False):
        raise ValueError(
            "hddc_surgery requires isotropic_psi=True: the HDDC reconstruction "
            "Sigma_k = W_k W_k^T + b_k I is exact only for isotropic Psi_k"
        )
    device = device if device is not None else model.mu.device

    N, S_acc, n_rows = accumulate_statistics(
        model, loader, device=device, max_batches=cfg.max_batches
    )
    stats = reconstruct_components(model, N, S_acc, cfg)
    summary = _summarize(stats, model.q, device)
    summary["n_rows"] = n_rows
    summary["threshold"] = float(cfg.threshold)
    summary["n_min"] = cfg.n_min(model.q)
    summary["d_k_per_component"] = [int(v) for v in stats["d_k"].tolist()]
    if log is not None:
        log(
            f"[surgery] rows={n_rows:,} | d_k mean={summary['d_k_mean']:.2f} "
            f"median={summary['d_k_median']} range=[{summary['d_k_min']}, "
            f"{summary['d_k_max']}] | b_k mean={summary['b_k_mean']:.4g} | "
            f"updated={summary['n_updated']} skipped={summary['n_skipped']} "
            f"(n_min={summary['n_min']:.0f}) | saturated@q={summary['saturated_frac']:.1%}"
        )
    return summary


def parameter_count(model) -> int:
    """Free parameters under the current rank mask, for post-hoc BIC curves.

    Per component: `mu` (D) + mixture weight (1) + one noise level (1 with
    isotropic Psi, else D) + the loading columns actually in use. A rank-d
    subspace of R^D has d(D - d) + d free parameters once the rotational
    redundancy inside the subspace is removed, matching HDDC's count.
    """
    K, D = model.K, model.D
    d = model.rank_mask.sum(-1).to(torch.int64)
    subspace = (d * (D - d) + d).sum().item()
    noise = K if getattr(model, "isotropic_psi", False) else (
        K * D if model.psi_per_component else D
    )
    return int(K * D + (K - 1) + noise + subspace)


__all__ = [
    "SurgeryConfig",
    "accumulate_statistics",
    "hddc_surgery",
    "parameter_count",
    "reconstruct_components",
    "reset_optimizer_state",
    "surgery_params",
]
