"""Periodic HDDC-style covariance surgery for MFA, giving adaptive per-component rank.

SGD training of `train_nll` is left untouched. Every `T` epochs this module
applies the closed-form covariance update of either the HDDC model
`[a_ij b_i Q_i d_i]` or its single-process shared-noise variant
`[a_ij b Q_i d_i]` (Bouveyron, Girard & Schmid, arXiv:math/0604064). It
re-estimates each component's covariance with an adaptive rank
`d_k <= q_max`, and rewrites the result in MFA parameters. Between surgeries
the columns beyond `d_k` are hard-masked by `MFA.rank_mask`, so they contribute
nothing to the likelihood and receive no gradient.

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
   eigenvalue differences to propose `d_k`, and the HDDC noise level. The
   default is `b_k = (Tr(S_k) - sum_{j<=d_k} lam_j) / (D - d_k)`; the
   shared-noise model pools those residuals with weights `N_k`. Because one
   common floor can overtake a weak retained eigenvalue from another component,
   shared-b surgery treats the Cattell ranks as caps and solves a small active-
   set problem that retains exactly the optional directions compatible with
   `lam_j > b`. The MFA reconstruction then uses
   `scale_j = sqrt(lam_j - b)` (or `b_k`). All `q_max` columns are written from
   the eigendecomposition and only the mask records `d_k`, so a later surgery
   can raise a component's rank with no revival logic.

C. Adam state for the rewritten tensors is dropped by the caller
   (`reset_optimizer_state`), optionally followed by a short LR warmup.

The whole path is a parallel stack that leaves the production files untouched:
`mfa_hddc.py`, `train_hddc.py`, this module, and
`dalg/cli/adaptive_q/run_training_hddc.py`. Removing the feature means deleting
those four files and the `dalg-run-training-hddc` entry in `pyproject.toml`.
"""

from __future__ import annotations

import math
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
    every: float = 0.0
    threshold: float = 0.01      # Cattell t, relative to lam_1 (scale-free)
    min_count: float = 0.0       # n_min in effective points; 0 disables the cutoff
    warmup_steps: int = 0        # linear LR warmup after each surgery
    psi_floor: float = 1e-6      # requested b floor; model._eps may impose a higher one
    eps: float = 1e-12           # clamp inside sqrt(lam_j - b_k)
    max_batches: Optional[int] = None  # cap the E-pass length (debug/smoke)

    def n_min(self) -> float:
        """Return the literal effective-membership cutoff used by surgery.

        ``min_count=0`` disables the cutoff, so every component with positive
        soft membership is eligible. A negative or non-finite cutoff is invalid;
        there is no sentinel value that silently selects a heuristic threshold.
        """
        value = float(self.min_count)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("surgery min_count must be finite and non-negative")
        return value

    def active_at(self, epoch: int) -> bool:
        if not self.enabled or self.every <= 0 or epoch <= 0:
            return False
        periods = float(epoch) / float(self.every)
        return math.isclose(periods, round(periods), rel_tol=0.0, abs_tol=1e-9)

    def active_after_batch(
        self,
        batch: int,
        steps_per_epoch: Optional[int],
        *,
        epoch: int = 1,
    ) -> bool:
        """Whether a sub-epoch cadence crosses a surgery boundary after `batch`.

        Optimizer steps discretize fractional epoch positions. Surgery therefore
        runs on the first completed batch at or beyond each multiple of `every`.
        The epoch is included in the calculation so non-divisor cadences such as
        0.3 continue across epoch boundaries without being rounded or reset.
        """
        if (
            not self.enabled
            or not 0 < float(self.every) < 1
            or steps_per_epoch is None
            or steps_per_epoch <= 0
            or epoch <= 0
            or batch <= 0
            or batch > steps_per_epoch
        ):
            return False

        previous = (epoch - 1) + (batch - 1) / steps_per_epoch
        current = (epoch - 1) + batch / steps_per_epoch
        previous_period = math.floor(previous / self.every + 1e-9)
        current_period = math.floor(current / self.every + 1e-9)
        crossed = current_period - previous_period

        # An exact epoch-boundary event runs after validation via `active_at`,
        # preserving the existing integer and half-epoch ordering.
        if batch == steps_per_epoch and self.active_at(epoch):
            crossed -= 1
        if crossed > 1:
            raise ValueError(
                "surgery cadence is shorter than one optimizer step; increase "
                "--surgery-every-epochs or --steps-per-epoch"
            )
        return crossed == 1


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
def _solve_shared_b_active_set(
    lam: torch.Tensor,
    trace: torch.Tensor,
    N: torch.Tensor,
    rank_cap: torch.Tensor,
    *,
    psi_floor: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Solve the optional Cattell active set under one shared isotropic floor.

    The shared-noise HDDC covariance of component ``k`` has eigenvalues

    ``lambda_kj = scale_kj**2 + b`` for ``j <= d_k`` and ``b`` otherwise.

    Consequently every retained direction must satisfy ``lambda_kj > b``.
    Equality is noise, not a rank-bearing direction: keeping it would make the
    loading variance zero while ``rank_mask`` still reported it as active.

    Cattell selection is independent for each component, whereas shared ``b``
    is coupled across all eligible components. A Cattell proposal can therefore
    be algebraically inconsistent even when each component's scree plot is
    sensible. This function treats the proposed ``rank_cap[k]`` as an *upper
    bound* and solves the resulting active-set problem without another E-pass or
    eigendecomposition.

    For provisional ranks ``r_k = rank_cap[k]``, directions ``j > r_k`` form
    the mandatory noise pool

    ``A = sum_k N_k sum_{j > r_k} lambda_kj``
    ``B = sum_k N_k (D - r_k)``,  so ``b = max(A / B, psi_floor)``.

    Directions ``2 <= j <= r_k`` are optional. They are sorted *globally* by
    increasing eigenvalue, retaining their component weight ``N_k``. Starting
    from the mandatory pool, the smallest optional direction is moved into the
    noise pool exactly when ``lambda_kj <= b``; then ``A``, ``B``, and ``b`` are
    updated. Once the next eigenvalue is strictly above the current floor, every
    remaining candidate is larger and is therefore retained. Processing one
    increasing prefix is important: discarding every violation against the
    initial floor in one batch can over-prune, because adding small eigenvalues
    to the pool lowers ``b`` and can make a larger candidate valid.

    The first direction is deliberately not optional. The HDDC path currently
    requires ``d_k >= 1``. If the final shared floor reaches ``lambda_k1``, the
    caller's retained-eigenvalue validation raises instead of silently changing
    the model to admit rank-zero spherical components.

    This is a feasibility/profile update conditional on the current
    responsibilities, fixed means, eligible components, and Cattell rank caps;
    it is not a new intrinsic-dimension criterion and does not replace model
    selection over the Cattell threshold. A later surgery recomputes the caps
    from scratch, so a direction removed here may be restored later.

    Args:
        lam: Descending empirical covariance eigenvalues with shape ``(n, D)``
            for the eligible components only.
        trace: Empirical covariance traces with shape ``(n,)``.
        N: Effective component memberships with shape ``(n,)``. Each discarded
            eigenvalue of component ``k`` receives weight ``N_k``.
        rank_cap: Per-component Cattell proposals with shape ``(n,)`` and values
            in ``[1, min(q_max, D - 1)]``.
        psi_floor: Lower numerical bound applied to the pooled floor after every
            update. The caller passes the larger of the configured surgery floor
            and ``model._eps + 1e-12``. The caller then round-trips the target
            through the model dtype and validates against the floor that will
            actually be written.

    Returns:
        ``(rank, b, b_at_cattell_cap)``. ``rank`` contains the rank-one-
        constrained candidate ranks after resolving every optional direction,
        ``b`` is the corresponding pooled floor, and
        ``b_at_cattell_cap`` is the floor before optional directions were moved
        into the noise pool. The latter is useful for diagnosing how strongly
        shared-floor consistency changed the scree proposals. The caller must
        still validate the mandatory first direction of every component.

    Raises:
        ValueError: If the inputs have incompatible shapes, contain no
            components, or contain a rank cap outside ``[1, D - 1]``.
        RuntimeError: If the pooled residual or residual degrees of freedom are
            non-finite, or the latter are non-positive. The caller separately
            reports an impossible mandatory first direction with its original
            component index.
    """
    n, D = lam.shape
    if trace.shape != (n,) or N.shape != (n,) or rank_cap.shape != (n,):
        raise ValueError("shared-b active-set inputs have incompatible shapes")
    if n == 0:
        raise ValueError("shared-b active-set requires at least one component")
    if bool(((rank_cap < 1) | (rank_cap >= D)).any()):
        raise ValueError("shared-b rank caps must lie in [1, D - 1]")

    q = int(rank_cap.max().item())
    j = torch.arange(1, q + 1, device=lam.device, dtype=torch.int64)[None, :]
    lam_cap = lam[:, :q].clamp_min(0.0)
    head = torch.cumsum(lam_cap, dim=1)
    kept_at_cap = head.gather(1, (rank_cap - 1)[:, None]).squeeze(1)
    residual_at_cap = trace - kept_at_cap

    numerator = (N * residual_at_cap).sum()
    denominator = (N * (D - rank_cap).to(N.dtype)).sum()
    if (
        not torch.isfinite(numerator)
        or not torch.isfinite(denominator)
        or float(denominator) <= 0.0
    ):
        raise RuntimeError(
            "shared-b surgery produced a non-finite pooled residual or "
            "non-positive residual degrees of freedom"
        )

    b_at_cap = (numerator / denominator).clamp_min(float(psi_floor))

    # Direction one remains mandatory. All later directions below each Cattell
    # cap are optional and participate in the globally coupled active set.
    optional = (j > 1) & (j <= rank_cap[:, None])
    candidate_lam = lam_cap[optional]
    if candidate_lam.numel() == 0:
        return rank_cap.clone(), b_at_cap, b_at_cap

    candidate_weight = N[:, None].expand(-1, q)[optional]
    candidate_component = torch.arange(n, device=lam.device)[:, None].expand(
        -1, q
    )[optional]
    order = torch.argsort(candidate_lam)
    candidate_lam = candidate_lam[order]
    candidate_weight = candidate_weight[order]
    candidate_component = candidate_component[order]

    weighted_lam_prefix = torch.cat(
        [
            torch.zeros(1, dtype=numerator.dtype, device=lam.device),
            torch.cumsum(candidate_weight * candidate_lam, dim=0),
        ]
    )
    weight_prefix = torch.cat(
        [
            torch.zeros(1, dtype=denominator.dtype, device=lam.device),
            torch.cumsum(candidate_weight, dim=0),
        ]
    )

    # b_before[i] is the floor after accepting candidates 0..i-1 as noise.
    # The first candidate strictly above that floor starts the retained suffix.
    b_before = (
        (numerator + weighted_lam_prefix[:-1])
        / (denominator + weight_prefix[:-1])
    ).clamp_min(float(psi_floor))
    retained_suffix = (candidate_lam > b_before).nonzero(as_tuple=True)[0]
    n_to_noise = (
        int(retained_suffix[0].item())
        if retained_suffix.numel()
        else int(candidate_lam.numel())
    )

    numerator = numerator + weighted_lam_prefix[n_to_noise]
    denominator = denominator + weight_prefix[n_to_noise]
    b = (numerator / denominator).clamp_min(float(psi_floor))

    # Derive ranks from the exact prefix used in A and B. Re-thresholding against
    # the rounded final b could disagree by one ULP at an equality boundary.
    discarded_per_component = torch.bincount(
        candidate_component[:n_to_noise], minlength=n
    )
    rank = rank_cap - discarded_per_component
    return rank, b, b_at_cap


@torch.no_grad()
def reconstruct_components(model, N, S_acc, cfg: SurgeryConfig) -> Dict[str, Any]:
    """Rewrite each eligible component's covariance from its scatter matrix.

    Components with `N_k < n_min` keep their loading parameters and mask. Setting
    `n_min = 0` disables that cutoff and makes every component eligible, including
    components with no hard assignments but positive soft responsibility mass.
    An exactly zero `N_k` cannot define `S_k / N_k` and raises explicitly instead
    of being silently skipped. In shared-b mode skipped components' covariance
    floor still changes when eligible components produce a new global b. For
    eligible shared-b components, the Cattell ranks are upper bounds:
    `_solve_shared_b_active_set` may lower them to make the final rank mask
    consistent with the common floor. That helper's docstring gives the complete
    derivation, sorted stopping rule, rank-one policy, and statistical scope of
    this extra step.

    All rank selection, pooled-floor estimation, and retained-eigenvalue
    validation finish before any parameter is mutated. A failed shared-b fit
    therefore leaves the model unchanged.
    """
    K, D, q = model.K, model.D, model.q
    device = model.mu.device
    n_min = cfg.n_min()
    # `_psi()` is softplus(psi_rho) + model._eps. Validate and reconstruct with
    # a floor the parameterization can actually represent, even when the surgery
    # config requests a smaller numerical floor.
    effective_psi_floor = max(float(cfg.psi_floor), float(model._eps) + 1e-12)

    d_k = model.rank_mask.sum(-1).to(torch.int64)          # unchanged where skipped
    b_k = torch.full((K,), float("nan"), dtype=torch.float64, device=device)
    shared_b = bool(getattr(model, "shared_b", False))

    eligible = (N >= n_min) & torch.isfinite(N)
    idx = eligible.nonzero(as_tuple=True)[0]
    if idx.numel() == 0:
        return {
            "d_k": d_k, "b_k": b_k, "eligible": eligible,
            "b_shared": None,
            "b_shared_at_cattell": None,
            "n_shared_b_pruned_components": 0,
            "n_shared_b_pruned_directions": 0,
            "n_updated": 0, "n_skipped": int(K), "N_k": N,
        }

    zero_support = idx[N[idx] <= 0.0]
    if zero_support.numel():
        component = int(zero_support[0].item())
        raise RuntimeError(
            "HDDC surgery cannot estimate a covariance for an eligible "
            f"component with non-positive effective membership: component={component}, "
            f"N_k={float(N[component]):.8g}. Set surgery_min_count above zero "
            "to skip unsupported components."
        )

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

    # HDDC eq. 4: a component-specific noise level is the mean of its discarded
    # eigenvalues. Equation 5 pools the same residual variance across components
    # for the [a_ij b Q_i d_i] model, weighted by effective membership N_k.
    # In the shared model the independently selected Cattell ranks are only caps:
    # the active-set solve may lower them so every reported signal direction has
    # strictly positive loading variance under the common floor.
    d_cattell = d_sel.clone()
    b_shared_at_cattell = None
    n_shared_b_pruned_components = 0
    n_shared_b_pruned_directions = 0
    if shared_b:
        d_sel, b_shared, b_shared_at_cattell = _solve_shared_b_active_set(
            lam,
            trace,
            N[idx],
            d_cattell,
            psi_floor=effective_psi_floor,
        )
        pruned = d_cattell - d_sel
        n_shared_b_pruned_components = int((pruned > 0).sum().item())
        n_shared_b_pruned_directions = int(pruned.sum().item())

        b = b_shared.expand(idx.numel())
    else:
        b_shared = None
        head = torch.cumsum(lam[:, :q].clamp_min(0.0), dim=1)
        kept = head.gather(1, (d_sel - 1)[:, None]).squeeze(1)
        residual = trace - kept
        b = (residual / (D - d_sel).to(trace.dtype)).clamp_min(
            effective_psi_floor
        )

    # Encode b exactly as the model writer will, then use the round-tripped value
    # for validation, scale reconstruction, and reporting. This closes the small
    # but real gap between a float64 pooled target and its stored model dtype.
    dtype = model.dir_raw.dtype
    psi_target = (b - model._eps).clamp_min(1e-12)
    if shared_b:
        psi_rho_new = _softplus_inverse(psi_target[:1]).to(dtype)
        b = (
            F.softplus(psi_rho_new).to(b.dtype) + model._eps
        ).expand(idx.numel())
        b_shared = b[0]
    else:
        psi_rho_new = _softplus_inverse(psi_target).to(dtype)
        b = F.softplus(psi_rho_new).to(b.dtype) + model._eps

    # Retained directions must be strictly above the floor that is actually
    # written. The shared active set guarantees this apart from a mandatory
    # first direction or a dtype-rounding boundary. Component-specific tail
    # means normally guarantee it algebraically; an imposed numerical floor can
    # still make that model infeasible, so both modes use the same strict check.
    retained = j <= d_sel[:, None]
    invalid = retained & (lam[:, :q] <= b[:, None])
    if invalid.any():
        row, col = invalid.nonzero()[0].tolist()
        component = int(idx[row].item())
        eigenvalue = float(lam[row, col].item())
        noise_mode = "shared-b" if shared_b else "component-b"
        raise RuntimeError(
            f"{noise_mode} surgery cannot reconstruct retained direction: "
            f"component={component}, direction={col + 1}, "
            f"lambda={eigenvalue:.8g} <= b={float(b[row]):.8g}"
        )

    # Sigma_k = W_k W_k^T + b_* I with W_k's columns the top eigenvectors scaled
    # by sqrt(lam_j - b_*). All q_max columns are written; only the mask records
    # d_k, so a later surgery can raise the rank with no revival logic.
    scale = (lam[:, :q] - b[:, None]).clamp_min(float(cfg.eps)).sqrt()

    model.dir_raw.data[idx] = Q[:, :, :q].to(dtype)
    model.scale_rho.data[idx] = _softplus_inverse(scale).to(dtype)
    if shared_b:
        model.psi_rho.data.copy_(psi_rho_new.reshape_as(model.psi_rho))
    else:
        model.psi_rho.data[idx] = psi_rho_new[:, None]
    model.rank_mask.data[idx] = (j <= d_sel[:, None]).to(model.rank_mask.dtype)

    d_k = d_k.clone()
    d_k[idx] = d_sel
    b_k[idx] = b

    return {
        "d_k": d_k,
        "b_k": b_k,
        "b_shared": b_shared,
        "b_shared_at_cattell": b_shared_at_cattell,
        "n_shared_b_pruned_components": n_shared_b_pruned_components,
        "n_shared_b_pruned_directions": n_shared_b_pruned_directions,
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
    summary = {
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
    b_shared = stats.get("b_shared")
    if b_shared is not None:
        summary["b_shared"] = float(b_shared.item())
        summary["b_shared_at_cattell"] = float(
            stats["b_shared_at_cattell"].item()
        )
        summary["n_shared_b_pruned_components"] = int(
            stats["n_shared_b_pruned_components"]
        )
        summary["n_shared_b_pruned_directions"] = int(
            stats["n_shared_b_pruned_directions"]
        )
    return summary


# Entry point


def hddc_surgery(model, loader, cfg: SurgeryConfig, *, device=None, log=None) -> Dict[str, Any]:
    """Run phases A and B in place. Returns a summary dict for logging.

    The caller is responsible for phase C (`reset_optimizer_state` on
    `surgery_params(model)`) and for any post-surgery LR warmup.
    """
    if getattr(model, "shared_b", False) and isinstance(
        model, ComponentShardedMFA_HDDC
    ):
        raise ValueError("shared-b surgery is supported only for a full MFA_HDDC model")
    if not (
        getattr(model, "isotropic_psi", False)
        or getattr(model, "shared_b", False)
    ):
        raise ValueError(
            "hddc_surgery requires isotropic_psi=True or shared_b=True: the "
            "HDDC reconstruction is exact only for isotropic noise"
        )
    device = device if device is not None else model.mu.device

    N, S_acc, n_rows = accumulate_statistics(
        model, loader, device=device, max_batches=cfg.max_batches
    )
    stats = reconstruct_components(model, N, S_acc, cfg)
    summary = _summarize(stats, model.q, device)
    summary["n_rows"] = n_rows
    summary["threshold"] = float(cfg.threshold)
    summary["n_min"] = cfg.n_min()
    summary["d_k_per_component"] = [int(v) for v in stats["d_k"].tolist()]
    if log is not None:
        noise_text = (
            f"b={summary['b_shared']:.4g}"
            if "b_shared" in summary
            else f"b_k mean={summary['b_k_mean']:.4g}"
        )
        active_set_text = ""
        if summary.get("n_shared_b_pruned_directions", 0) > 0:
            active_set_text = (
                f" | shared-b active set pruned="
                f"{summary['n_shared_b_pruned_directions']} dirs/"
                f"{summary['n_shared_b_pruned_components']} comps "
                f"(b at Cattell={summary['b_shared_at_cattell']:.4g})"
            )
        log(
            f"[surgery] rows={n_rows:,} | d_k mean={summary['d_k_mean']:.2f} "
            f"median={summary['d_k_median']} range=[{summary['d_k_min']}, "
            f"{summary['d_k_max']}] | {noise_text} | "
            f"updated={summary['n_updated']} skipped={summary['n_skipped']} "
            f"(n_min={summary['n_min']:.0f}) | saturated@q={summary['saturated_frac']:.1%}"
            f"{active_set_text}"
        )
    return summary


def parameter_count(model) -> int:
    """Free parameters under the current rank mask, for post-hoc BIC curves.

    Per component: `mu` (D) + mixture weight (1) + the loading columns actually
    in use, plus either one global shared-b parameter, K component-specific
    isotropic levels, or the diagonal Psi parameters. A rank-d subspace of R^D
    has d(D - d) + d free parameters once the rotational redundancy inside the
    subspace is removed, matching HDDC's count.
    """
    K, D = model.K, model.D
    d = model.rank_mask.sum(-1).to(torch.int64)
    subspace = (d * (D - d) + d).sum().item()
    if getattr(model, "shared_b", False):
        noise = 1
    elif getattr(model, "isotropic_psi", False):
        noise = K
    else:
        noise = K * D if model.psi_per_component else D
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
