"""MFA with an ARD prior on the columns of W_k, for adaptive per-component rank.

Each loading column gets a Gaussian prior whose precision is itself Gamma-distributed:

    p(w_j^k | nu_j^k) = N(0, (nu_j^k)^-1 I_D)      nu_j^k ~ Gamma(alpha0, b0)

The MAP objective adds, on top of the MFA negative log-likelihood,

    sum_{k,j} [ 1/2 ||w_j^k||^2 nu_j^k + b0 nu_j^k - (D/2 + alpha0 - 1) log nu_j^k ]

Two facts make this cheap here:

1. `MFA._dir_hat()` normalizes each column of `dir_raw` over D, so
   ||w_j^k|| == softplus(scale_rho)[k, j] exactly. The penalty is a function of
   `scale_rho` alone -- W is never materialized.
2. The penalty is convex in nu with the exact minimizer

       nu*_{k,j} = c / (1/2 s_{k,j}^2 + b0),     c := D/2 + alpha0 - 1

   which we recompute each forward pass and detach. With nu detached the
   gradient w.r.t. s is  c * s / (1/2 s^2 + b0), exactly the gradient of the
   nu-eliminated penalty  c * log(1/2 s^2 + b0). So this is not an
   approximation: it is gradient descent on the profiled objective, with no
   extra parameters.

Because nu adds no parameters, an MFA_ARD state_dict is identical to a plain
MFA one. `dalg.models.mfa.load_mfa` therefore reads these checkpoints as-is,
and every downstream consumer (assignments, Gaussian overlap, intrinsic dim,
interpretation) works on ARD runs unchanged. Use `load_mfa_ard` when you also
want the ARD hyperparameters back.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

from ..mfa import MFA


# softplus(-120) underflows to exactly 0 in float32 while staying finite, so a
# pruned scale reads as zero without risking NaNs if an optimizer ever touches
# it again (its gradient, sigmoid(-120), is 0 — pruning is permanent).
_PRUNED_SCALE_RHO = -120.0


class MFA_ARD(MFA):
    """MFA whose per-component rank is shrunk by an ARD prior on W_k's columns.

    Args:
        centroids: (K, D) initial mu_k, as in `MFA`.
        rank: maximum rank q per component. ARD prunes columns below it, so
            set this generously rather than tightly.
        alpha0: Gamma shape of the prior on nu.
        b0: Gamma rate of the prior on nu. Smaller b0 weakens shrinkage for
            columns whose scale is already near zero.
        ard_weight: scalar multiplying the whole penalty. The penalty is a
            prior over parameters (applies once per dataset) while the training
            loss is a per-sample mean, so callers set this to
            ``lambda / N_train``.
        rank_threshold: how much variance a column must carry, relative to its
            component's mean unique variance, to count in `effective_ranks`.
    """

    def __init__(
        self,
        centroids: torch.Tensor,
        *,
        rank: int,
        alpha0: float = 1.0,
        b0: float = 1e-4,
        ard_weight: float = 1.0,
        rank_threshold: float = 1.0,
        **mfa_kwargs: Any,
    ):
        super().__init__(centroids, rank=rank, **mfa_kwargs)

        if not b0 > 0:
            raise ValueError("b0 must be positive")
        log_coeff = self.D / 2.0 + float(alpha0) - 1.0
        if not log_coeff > 0:
            raise ValueError(
                "D/2 + alpha0 - 1 must be positive; a non-positive coefficient "
                "turns the log term into anti-shrinkage "
                f"(D={self.D}, alpha0={alpha0})"
            )
        if not rank_threshold > 0:
            raise ValueError("rank_threshold must be positive")

        # Plain floats, not parameters/buffers: the state_dict stays identical
        # to a plain MFA. These are persisted in checkpoint meta instead.
        self.alpha0 = float(alpha0)
        self.b0 = float(b0)
        self.ard_weight = float(ard_weight)
        self.rank_threshold = float(rank_threshold)
        self.log_coeff = float(log_coeff)  # c = D/2 + alpha0 - 1

    def ard_precisions(self) -> torch.Tensor:
        """(K, q) closed-form nu*, detached — the exact minimizer given W."""
        s = self._scale()
        return (self.log_coeff / (0.5 * s.detach() ** 2 + self.b0))

    def ard_penalty(self) -> torch.Tensor:
        """Scalar ARD penalty summed over components and columns."""
        s = self._scale()                       # (K, q) == column norms of W_k
        nu = self.ard_precisions()              # detached
        return (
            0.5 * s ** 2 * nu
            + self.b0 * nu
            - self.log_coeff * nu.log()
        ).sum()

    def loss_terms(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(total, nll, penalty) so the training loop can log them separately."""
        nll = self.nll(x)
        penalty = self.ard_penalty()
        return nll + self.ard_weight * penalty, nll, penalty

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.loss_terms(x)[0]

    @torch.no_grad()
    def column_scales(self) -> torch.Tensor:
        """(K, q) column norms s_{k,j} of W_k, detached."""
        return self._scale().detach()

    @torch.no_grad()
    def alive_mask(self, threshold: Optional[float] = None) -> torch.Tensor:
        """(K, q) bool: columns carrying variance above the noise floor."""
        thr = self.rank_threshold if threshold is None else float(threshold)
        s2 = self.column_scales() ** 2                       # (K, q)
        noise_floor = self._psi().mean(dim=1, keepdim=True)  # (K, 1)
        return s2 > thr * noise_floor

    @torch.no_grad()
    def prune_columns(self, threshold: Optional[float] = None) -> torch.Tensor:
        """Zero every column below the noise floor. Returns (K,) surviving counts.

        **Run this only after training has finished.** Applying it mid-training
        would freeze pruning decisions the loadings can never undo, and would
        silently alter the model that best-epoch tracking selected.

        Columns are zeroed in place rather than deleted, so `W` keeps its
        (K, D, q) shape and every downstream consumer keeps working. Both halves
        of the factorization are killed: `scale_rho` goes to a large negative
        constant (so `column_scales` and `effective_ranks` read the column as
        dead) and `dir_raw` goes to zero (so `_dir_hat` returns exactly zero and
        the W column is exactly zero in any dtype, not merely underflowed).
        A pruned column contributes nothing to `C_k = W W^T + Psi`.

        Components can legitimately end at q_k = 0 — those are collapsed tiles,
        now plain diagonal Gaussians.
        """
        keep = self.alive_mask(threshold)
        self.scale_rho.masked_fill_(~keep, _PRUNED_SCALE_RHO)
        # keep: (K, q) -> (K, 1, q) to broadcast over the D axis of dir_raw.
        self.dir_raw.mul_(keep[:, None, :].to(self.dir_raw.dtype))
        return keep.sum(dim=1)

    @torch.no_grad()
    def effective_ranks(self, threshold: Optional[float] = None) -> torch.Tensor:
        """(K,) learned q_k: columns carrying variance above the noise floor.

        A column counts when ``s_{k,j}^2 > threshold * mean_d Psi_{k,d}`` — the
        classical factor-analysis rule that a factor must explain more than the
        unique variance it sits on.

        The reference is Psi rather than the component's largest column on
        purpose. A relative-to-peak cutoff is blind to the over-pruning failure
        mode: when ARD collapses *every* column, the ratios stay near 1 and the
        count would wrongly report full rank. Against Psi, a collapsed
        component correctly reports q_k = 0.
        """
        return self.alive_mask(threshold).sum(dim=1)


def _ard_meta(model: MFA_ARD) -> Dict[str, Any]:
    q_eff = model.effective_ranks()
    return {
        "alpha0": model.alpha0,
        "b0": model.b0,
        "ard_weight": model.ard_weight,
        "rank_threshold": model.rank_threshold,
        "nu_mode": "closed_form",
        # The learned per-component rank, so downstream tools do not have to
        # re-derive it from the scales.
        "effective_ranks": q_eff.tolist(),
        "q_eff_mean": float(q_eff.float().mean().item()),
        "dead_components": int((q_eff == 0).sum().item()),
    }


def save_mfa_ard(
    model: MFA_ARD,
    path: str | Path,
    *,
    pruned: bool = False,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Save an MFA_ARD model, recording the ARD hyperparameters in meta.

    The weights are a plain-MFA state_dict, so `dalg.models.mfa.load_mfa` also
    reads these files.
    """
    meta: Dict[str, Any] = {
        "K": model.K,
        "D": model.D,
        "q": model.q,
        "psi_per_component": model.psi_per_component,
        "eps_floor": model._eps,
        "dtype": str(model.mu.dtype),
        "version": 1,
        "model": "MFA_ARD",
        "ard": {**_ard_meta(model), "pruned": bool(pruned)},
        "rotation_on": bool(getattr(model, "_rotation_on", False)),
        "rotation_kind": getattr(model, "_rotation_kind", None),
        "rotation_params": getattr(model, "_rotation_params", {}),
    }
    if extra:
        meta["extra"] = extra

    torch.save({"state_dict": model.state_dict(), "meta": meta}, path)


def load_mfa_ard(
    path: str | Path,
    *,
    map_location: Optional[str | torch.device] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    strict: bool = True,
) -> MFA_ARD:
    """Load an MFA_ARD model, restoring its ARD hyperparameters from meta.

    Checkpoints written by `save_mfa` (no "ard" meta) load fine too; the ARD
    hyperparameters then fall back to the MFA_ARD defaults.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    ckpt = torch.load(path, map_location=map_location)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state: Dict[str, torch.Tensor] = ckpt["state_dict"]
        meta: Dict[str, Any] = ckpt.get("meta", {}) or {}
    else:
        state = ckpt
        meta = {}

    mu = state["mu"]                    # (K, D)
    dir_raw = state["dir_raw"]           # (K, D, q)
    K, D = mu.shape
    q = dir_raw.shape[-1]

    psi_rho = state["psi_rho"]           # (K, D) or (D,)
    psi_per_component = bool(
        meta.get("psi_per_component", psi_rho.ndim == 2 and psi_rho.shape[0] == K)
    )
    eps_floor = float(meta.get("eps_floor", 1e-8))
    ard = meta.get("ard", {}) or {}

    centroids = torch.zeros(K, D, dtype=mu.dtype)
    model = MFA_ARD(
        centroids=centroids,
        rank=q,
        alpha0=float(ard.get("alpha0", 1.0)),
        b0=float(ard.get("b0", 1e-4)),
        ard_weight=float(ard.get("ard_weight", 1.0)),
        rank_threshold=float(ard.get("rank_threshold", 1.0)),
        psi_per_component=psi_per_component,
        eps_floor=eps_floor,
    )

    if "_rot_T" not in state or "_rot_inv_Tt" not in state:
        eye = torch.eye(q, dtype=mu.dtype)
        state.setdefault("_rot_T", eye.repeat(K, 1, 1))
        state.setdefault("_rot_inv_Tt", eye.repeat(K, 1, 1))

    model.load_state_dict(state, strict=strict)

    model._rotation_on = bool(meta.get("rotation_on", False))
    model._rotation_kind = meta.get("rotation_kind", None)
    model._rotation_params = meta.get("rotation_params", {})

    if device is not None:
        model = model.to(device)
    if dtype is not None:
        model = model.to(dtype=dtype)
    return model


__all__ = ["MFA_ARD", "save_mfa_ard", "load_mfa_ard"]
