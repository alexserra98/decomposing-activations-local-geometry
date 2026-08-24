from __future__ import annotations

import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass
from pathlib import Path
from contextlib import contextmanager

class MFA(nn.Module):
    def __init__(
        self,
        centroids: torch.Tensor, # (K, D) initial mu_k
        *,
        rank: int, # q
        init_directions: Optional[torch.Tensor] = None, # optional (K, D, q)
        psi_init: float = 1.0, # initial diagonal unique variance
        psi_per_component: bool = False, # True => Psi_k per component; False => shared Psi
        scale_init: float = 1.0, # initial loading scales s_{k,j}
        eps_floor: float = 1e-5, # numerical floor for positivity / norms
    ):
        super().__init__()
        if centroids.ndim != 2:
            raise ValueError("centroids must have shape (K, D)")
        K, D = centroids.shape
        if not (1 <= rank <= D):
            raise ValueError("rank must be in [1, D]")
        if init_directions is not None and tuple(init_directions.shape) != (K, D, rank):
            raise ValueError(
                "init_directions must have shape "
                f"{(K, D, rank)}, got {tuple(init_directions.shape)}"
            )

        self.K, self.D, self.q = K, D, int(rank)
        self._two_pi_logD = self.D * math.log(2.0 * math.pi)
        self._eps = float(eps_floor)

        # Means  (K, D)
        self.mu = nn.Parameter(centroids.clone())

        # Loadings W_k parameterized as direction * scale
        if init_directions is None:
            direction_values = (
                torch.randn(K, D, self.q, dtype=centroids.dtype, device=centroids.device)
                / math.sqrt(D)
            )
        else:
            direction_values = init_directions.to(
                device=centroids.device,
                dtype=centroids.dtype,
            ).clone()
        self.dir_raw = nn.Parameter(direction_values)  # (K, D, q)
        rho_s0 = math.log(math.exp(float(scale_init)) - 1.0)
        self.scale_rho = nn.Parameter(
            torch.full((K, self.q), rho_s0, dtype=centroids.dtype)
        )  # (K, q)

        # Diagonal unique variances Psi
        psi_shape = (K, D) if psi_per_component else (D,)
        rho0 = math.log(math.exp(float(psi_init)) - 1.0)
        self.psi_rho = nn.Parameter(torch.full(psi_shape, rho0, dtype=centroids.dtype))
        self.psi_per_component = bool(psi_per_component)

        # Mixture weights (K,)
        self.pi_logits = nn.Parameter(torch.zeros(K, dtype=centroids.dtype))

        eye = torch.eye(self.q, dtype=centroids.dtype)
        self.register_buffer("_rot_T", eye.repeat(K, 1, 1))        # (K,q,q)
        self.register_buffer("_rot_inv_Tt", eye.repeat(K, 1, 1))   # (K,q,q)
        self._rotation_on: bool = False
        self._rotation_kind: Optional[str] = None    # 'oblimin' or None
        self._rotation_params: dict = {}
        self._inference_cache: Optional[Dict[str, Any]] = None

    def _psi(self) -> torch.Tensor:
        psi = F.softplus(self.psi_rho) + self._eps
        if psi.ndim == 1:
            psi = psi[None, :].expand(self.K, self.D)
        return psi  # (K, D)

    def _dir_hat(self) -> torch.Tensor:
        d = self.dir_raw
        n = d.norm(dim=1, keepdim=True).clamp_min(self._eps)  # (K, 1, q)
        return d / n

    def _scale(self) -> torch.Tensor:
        return F.softplus(self.scale_rho)

    def _W(self) -> torch.Tensor:
        d_hat = self._dir_hat()                 # (K, D, q)
        s = self._scale()                       # (K, q)
        return d_hat * s[:, None, :]            # (K, D, q)

    def _W_rotated(self, W: torch.Tensor) -> torch.Tensor:
        # L = A @ inv(T.T)
        return torch.einsum("kdq,kqp->kdp", W, self._rot_inv_Tt)

    def _maybe_rotate_scores(self, Ez: torch.Tensor, Sz: torch.Tensor):
        if not self._rotation_on:
            return Ez, Sz
        T = self._rot_T  # (K,q,q)

        # z_rot = z @ T
        Ez_rot = torch.einsum("bkq,kqp->bkp", Ez, T)
        Tt = T.transpose(1, 2)
        Sz_rot = torch.matmul(Tt, torch.matmul(Sz, T))
        return Ez_rot, Sz_rot


    @property
    def W(self) -> torch.Tensor:
        W = self._W()
        return self._W_rotated(W) if self._rotation_on else W

    @torch.no_grad()
    def _build_inference_cache(self) -> Dict[str, Any]:
        """
        Precompute frozen MFA likelihood terms for repeated eval calls.

        This is intentionally opt-in via inference_cache(): the cached tensors
        can be large for big K, but they avoid rebuilding model-only quantities
        for every activation batch during analysis.
        """
        psi = self._psi()
        psi_inv = 1.0 / psi
        W = self._W()

        A = W * psi_inv[:, :, None].sqrt()
        M = torch.einsum("kdi,kdj->kij", A, A)
        Iq = torch.eye(self.q, dtype=W.dtype, device=W.device)
        M = M + Iq[None]
        L = torch.linalg.cholesky(M)
        Minv = torch.cholesky_solve(
            Iq.expand(self.K, self.q, self.q).clone(),
            L,
            upper=False,
        )

        PinvW = psi_inv[:, :, None] * W
        pinvw_flat = PinvW.permute(1, 0, 2).reshape(self.D, self.K * self.q).contiguous()
        wt_pinv_mu = torch.einsum("kd,kdq->kq", self.mu, PinvW)
        mu_pinv_t = (psi_inv * self.mu).T.contiguous()
        mu_quad = (self.mu ** 2 * psi_inv).sum(dim=-1)
        logdet_psi = torch.log(psi).sum(dim=-1)
        logdet_m = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(-1)

        return {
            "psi_inv": psi_inv,
            "shared_psi": not self.psi_per_component,
            "mu_pinv_t": mu_pinv_t,
            "mu_quad": mu_quad,
            "pinvw_flat": pinvw_flat,
            "wt_pinv_mu": wt_pinv_mu,
            "Minv": Minv,
            "logdet_c": logdet_psi + logdet_m,
        }

    @contextmanager
    def inference_cache(self, *, enabled: bool = True):
        """
        Temporarily cache model-only likelihood terms for repeated inference.

        Use this around large eval-only loops:

            model.eval()
            with torch.no_grad(), model.inference_cache():
                r = model.responsibilities(x)
        """
        if not enabled:
            yield self
            return

        old_cache = self._inference_cache
        self._inference_cache = self._build_inference_cache()
        try:
            yield self
        finally:
            self._inference_cache = old_cache

    def _cached_log_prob_components(self, x: torch.Tensor) -> torch.Tensor:
        cache = self._inference_cache
        if cache is None:
            raise RuntimeError("MFA inference cache is not active")

        B, D = x.shape
        if D != self.D:
            raise ValueError(f"expected input dim {self.D}, got {D}")

        K, q = self.K, self.q

        if cache["shared_psi"]:
            x_quad = torch.matmul(x ** 2, cache["psi_inv"][0])
            quad_Psi = x_quad[:, None]
        else:
            quad_Psi = torch.einsum("bd,kd->bk", x ** 2, cache["psi_inv"])
        quad_Psi = (
            quad_Psi
            - 2.0 * torch.matmul(x, cache["mu_pinv_t"])
            + cache["mu_quad"][None, :]
        )

        WT_Pinv_x = torch.matmul(x, cache["pinvw_flat"]).reshape(B, K, q)
        v = WT_Pinv_x - cache["wt_pinv_mu"][None, :, :]

        v = v.float()
        quad_Psi = quad_Psi.float()
        low_rank = (torch.einsum("bkq,kqr->bkr", v, cache["Minv"]) * v).sum(dim=-1)
        quad = quad_Psi - low_rank
        return -0.5 * (self._two_pi_logD + cache["logdet_c"][None, :] + quad)

    def _core(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Core E-step shared by all public methods. Computes log-likelihoods and
        posterior latents for every (sample, component) pair in one batched pass.

        The covariance of component k is  C_k = W_k W_k^T + Psi  (factor analyser).
        Instead of inverting the D×D matrix C_k, we work with the q×q inner matrix

            M_k = I_q + W_k^T Psi^{-1} W_k

        which is cheap to factor (q << D). All inverses of C_k are expressed via the
        Woodbury identity using the Cholesky L_k of M_k.

        Args:
            x: (B, D) batch of activation vectors.

        Returns:
            ll:   (B, K)    log p(x | k)  — per-sample, per-component log-likelihood.
            Ez:   (B, K, q) E[z | x, k]  — posterior mean of the latent z.
            Sz:   (K, q, q) Cov[z | x, k] = M_k^{-1}  — posterior covariance (batch-independent).
            L:    (K, q, q) Cholesky factor of M_k (lower-triangular).
            v:    (B, K, q) W_k^T Psi^{-1} (x - mu_k)  — RHS before the Cholesky solve.
            psi:  (K, D)    diagonal noise variances.
        """
        B, D = x.shape
        if D != self.D:
            raise ValueError(f"expected input dim {self.D}, got {D}")

        psi     = self._psi()       # (K, D)  diagonal noise Psi_k
        psi_inv = 1.0 / psi         # (K, D)
        W       = self._W()         # (K, D, q)  factor loadings W_k  (unrotated)

        # ------------------------------------------------------------------
        # Step 1 — Cholesky of M_k  (always float32 for numerical stability)
        # ------------------------------------------------------------------
        A  = W * psi_inv[:, :, None].sqrt()         # (K, D, q)  scaled loadings
        M  = torch.einsum("kdi,kdj->kij", A, A)     # (K, q, q)  W^T Psi^{-1} W
        Iq = torch.eye(self.q, dtype=W.dtype, device=W.device)
        M  = M + Iq[None]                           # (K, q, q)  I + W^T Psi^{-1} W
        L  = torch.linalg.cholesky(M)               # (K, q, q)  lower-triangular

        # ------------------------------------------------------------------
        # Steps 2–3 — Mahalanobis + posterior
        # ------------------------------------------------------------------
        xT_Pinv_x   = torch.einsum("bd,kd->bk", x ** 2,      psi_inv)          # (B, K)
        xT_Pinv_mu  = torch.einsum("bd,kd->bk", x,   psi_inv * self.mu)        # (B, K)
        muT_Pinv_mu = (self.mu ** 2 * psi_inv).sum(dim=-1)                      # (K,)
        quad_Psi    = xT_Pinv_x - 2.0 * xT_Pinv_mu + muT_Pinv_mu[None, :]     # (B, K)

        PinvW      = psi_inv[:, :, None] * W                                    # (K, D, q)
        WT_Pinv_x  = torch.einsum("bd,kdq->bkq", x,        PinvW)              # (B, K, q)
        WT_Pinv_mu = torch.einsum("kd,kdq->kq",  self.mu,  PinvW)              # (K, q)
        v          = WT_Pinv_x - WT_Pinv_mu[None, :, :]                        # (B, K, q)

        # Cholesky solve stays in float32
        v = v.float()
        quad_Psi = quad_Psi.float()

        Ez = torch.cholesky_solve(v.permute(1, 2, 0), L, upper=False)          # (K, q, B)
        Ez = Ez.permute(2, 0, 1)                                                # (B, K, q)

        Sz = torch.cholesky_solve(Iq.expand(self.K, self.q, self.q).clone(),
                                  L, upper=False)                               # (K, q, q)

        quad = quad_Psi - (v * Ez).sum(dim=-1)                                  # (B, K)

        # ------------------------------------------------------------------
        # Step 4 — Log-determinant via the matrix determinant lemma
        #
        # log|C_k| = log|Psi_k| + log|M_k|
        # log|M_k| = 2 * sum(log diag(L_k))   from the Cholesky factor.
        # ------------------------------------------------------------------
        logdet_Psi = torch.log(psi).sum(dim=-1)                                 # (K,)
        logdet_M   = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(-1)  # (K,)
        logdet_C   = logdet_Psi + logdet_M                                      # (K,)

        # ------------------------------------------------------------------
        # Step 5 — Log-likelihood
        #
        # log p(x | k) = -1/2 [ D log(2π) + log|C_k| + (x-mu_k)^T C_k^{-1} (x-mu_k) ]
        # ------------------------------------------------------------------
        ll = -0.5 * (self.D * math.log(2.0 * math.pi) + logdet_C[None, :] + quad)  # (B, K)

        return ll, Ez, Sz, L, v, psi

    def responsibilities(self, x: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
        if self._inference_cache is None:
            ll, *_ = self._core(x)
        else:
            ll = self._cached_log_prob_components(x)
        log_pi = F.log_softmax(self.pi_logits, dim=0)[None, :]
        return F.softmax((ll + log_pi) / float(tau), dim=1)

    def log_prob_components(self, x: torch.Tensor) -> torch.Tensor:
        if self._inference_cache is None:
            ll, *_ = self._core(x)
            return ll
        return self._cached_log_prob_components(x)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        if self._inference_cache is None:
            ll, *_ = self._core(x)
        else:
            ll = self._cached_log_prob_components(x)
        log_pi = F.log_softmax(self.pi_logits, dim=0)  # (K,)
        return torch.logsumexp(ll + log_pi[None, :], dim=1)

    def nll(self, x: torch.Tensor) -> torch.Tensor:
        return (-self.log_prob(x)).mean()

    def component_posterior(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Posterior mean and covariance of the latent z for each component.

        Returns:
            Ez: (B, K, q) E[z | x, k] — coordinates in the local subspace of component k.
            Sz: (K, q, q) Cov[z | x, k] — shared across the batch.
        """
        _ll, Ez, Sz, *_ = self._core(x)
        Ez, Sz = self._maybe_rotate_scores(Ez, Sz)
        return Ez, Sz

    def reconstruct(self, x: torch.Tensor, *, use_mixture_mean: bool = True) -> torch.Tensor:
        """
        Reconstruct activations from their posterior latent codes.

        Each component predicts x_hat_k = mu_k + W_k E[z|x,k]. If use_mixture_mean
        is True (default), these are averaged using the posterior mixture weights
        (responsibilities), giving a single (B, D) reconstruction. If False,
        returns all per-component reconstructions as (B, K, D).

        Args:
            x: (B, D) input activations.
            use_mixture_mean: Whether to collapse components with responsibility weights.

        Returns:
            (B, D) mixture-weighted reconstruction, or (B, K, D) per-component.
        """
        ll, Ez, _Sz, _L, _v, _psi = self._core(x)
        # Use rotated view if enabled
        W_eff = self.W
        if self._rotation_on:
            Ez, _ = self._maybe_rotate_scores(Ez, _Sz)
        comp = self.mu[None, :, :] + torch.einsum("kdq,bkq->bkd", W_eff, Ez) # (B,K,D)
        if not use_mixture_mean:
            return comp
        log_pi = F.log_softmax(self.pi_logits, dim=0)[None, :]
        alpha = F.softmax(ll + log_pi, dim=1) # (B,K)
        return torch.einsum("bk,bkd->bd", alpha, comp) # (B,D)

    def forward(self, x):
        return self.nll(x)


def component_shard_bounds(K: int, rank: int, world_size: int) -> tuple[int, int]:
    """Contiguous component range owned by one distributed rank."""
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if not (0 <= rank < world_size):
        raise ValueError("rank must be in [0, world_size)")
    base = K // world_size
    rem = K % world_size
    start = rank * base + min(rank, rem)
    end = start + base + (1 if rank < rem else 0)
    return start, end


def _distributed_logsumexp(local_values: torch.Tensor, dim: int) -> torch.Tensor:
    """
    logsumexp over a tensor dimension that is sharded across distributed ranks.

    Every rank must call this with the same non-sharded dimensions. Gradients
    flow into each rank's local values through the SUM all-reduce.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return torch.logsumexp(local_values, dim=dim)

    local_max = local_values.max(dim=dim).values.detach()
    global_max = local_max.clone()
    dist.all_reduce(global_max, op=dist.ReduceOp.MAX)
    shifted = local_values - global_max.unsqueeze(dim)
    local_sum = shifted.exp().sum(dim=dim)
    global_sum = local_sum.detach().clone()
    dist.all_reduce(global_sum, op=dist.ReduceOp.SUM)
    global_sum = global_sum.clamp_min(torch.finfo(global_sum.dtype).tiny)
    # The correction term is zero in the forward pass (subtracts a detached copy),
    # but carries gradient: d/d x_i = exp(x_i - global_max) / global_sum = softmax(x_i).
    # global_sum is detached so each rank's gradient depends only on its own local
    # values — no cross-rank implicit gradient coupling.
    return global_max + global_sum.log() + (local_sum - local_sum.detach()) / global_sum


class ComponentShardedMFA(MFA):
    """
    MFA variant where each distributed rank owns a contiguous shard of components.

    This is model-parallel over K, not data-parallel. All ranks must see the
    same activation batch in the same order. Each rank computes likelihoods for
    its local components, then the mixture log probability is assembled with a
    distributed logsumexp over components.
    """

    def __init__(
        self,
        centroids: torch.Tensor,
        *,
        rank: int,
        global_K: int,
        component_start: int,
        init_directions: Optional[torch.Tensor] = None,
        psi_init: float = 1.0,
        psi_per_component: bool = False,
        scale_init: float = 1.0,
        eps_floor: float = 1e-5,
    ):
        super().__init__(
            centroids,
            rank=rank,
            init_directions=init_directions,
            psi_init=psi_init,
            psi_per_component=psi_per_component,
            scale_init=scale_init,
            eps_floor=eps_floor,
        )
        self.global_K = int(global_K)
        self.component_start = int(component_start)
        self.component_end = self.component_start + self.K

    @classmethod
    def from_global_centroids(
        cls,
        centroids: torch.Tensor,
        *,
        rank: int,
        dist_rank: int,
        world_size: int,
        init_directions: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> "ComponentShardedMFA":
        start, end = component_shard_bounds(centroids.shape[0], dist_rank, world_size)
        return cls(
            centroids[start:end].contiguous(),
            rank=rank,
            global_K=centroids.shape[0],
            component_start=start,
            init_directions=(
                None
                if init_directions is None
                else init_directions[start:end].contiguous()
            ),
            **kwargs,
        )

    def local_log_pi(self) -> torch.Tensor:
        log_z = _distributed_logsumexp(self.pi_logits, dim=0)
        return self.pi_logits - log_z

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        if self._inference_cache is None:
            ll, *_ = self._core(x)
        else:
            ll = self._cached_log_prob_components(x)
        log_num = _distributed_logsumexp(ll + self.pi_logits[None, :], dim=1)
        log_den = _distributed_logsumexp(self.pi_logits, dim=0)
        return log_num - log_den

    def sync_replicated_grads(self) -> None:
        """
        Sum gradients for parameters that are replicated across component shards.

        With the default shared Psi, every rank has a copy of psi_rho but only
        sees the likelihood terms for its local components. Summing the local
        psi gradients recovers the serial full-K gradient.
        """
        if self.psi_per_component:
            return
        if not (dist.is_available() and dist.is_initialized()):
            return
        if self.psi_rho.grad is not None:
            dist.all_reduce(self.psi_rho.grad, op=dist.ReduceOp.SUM)


def save_component_shard(model: ComponentShardedMFA, path: str | Path) -> None:
    """Save one rank's component shard."""
    path = Path(path)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "meta": {
                "global_K": model.global_K,
                "component_start": model.component_start,
                "component_end": model.component_end,
                "local_K": model.K,
                "D": model.D,
                "q": model.q,
                "psi_per_component": model.psi_per_component,
                "eps_floor": model._eps,
                "dtype": str(model.mu.dtype),
                "version": 1,
                "format": "component_shard",
            },
        },
        path,
    )

def save_mfa(model: MFA, path: str, *, extra: Optional[Dict[str, Any]] = None) -> None:
    """
    Save an MFA model to disk.
    """
    meta = {
        "K": model.K,
        "D": model.D,
        "q": model.q,
        "psi_per_component": model.psi_per_component,
        "eps_floor": model._eps,
        "dtype": str(model.mu.dtype),
        "version": 1, 
        "rotation_on": bool(getattr(model, "_rotation_on", False)),
        "rotation_kind": getattr(model, "_rotation_kind", None),
        "rotation_params": getattr(model, "_rotation_params", {}),
    }
    if extra:
        meta["extra"] = extra

    torch.save(
        {
            "state_dict": model.state_dict(), # includes rotation buffers if present
            "meta": meta,
        },
        path,
    )


def load_mfa(
    path: str | Path,
    *,
    map_location: Optional[str | torch.device] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    strict: bool = True,
) -> MFA:
    path = Path(path)
    if not path.exists():
        shards_json = path.parent / "mfa_model_shards.json"
        if shards_json.exists():
            return load_component_shards(
                shards_json, map_location=map_location, device=device, dtype=dtype
            )
        raise FileNotFoundError(path)
    ckpt = torch.load(path, map_location=map_location)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state: Dict[str, torch.Tensor] = ckpt["state_dict"]
        meta: Dict[str, Any] = ckpt.get("meta", {}) or {}
    else:
        state = ckpt
        meta = {}

    # Infer shapes
    mu = state["mu"] # (K, D)
    dir_raw = state["dir_raw"] # (K, D, q)
    K, D = mu.shape
    q = dir_raw.shape[-1]

    psi_rho = state["psi_rho"] # (K, D) or (D,)
    psi_per_component = bool(meta.get("psi_per_component",
                                      psi_rho.ndim == 2 and psi_rho.shape[0] == K))
    eps_floor = float(meta.get("eps_floor", 1e-8))

    centroids = torch.zeros(K, D, dtype=mu.dtype)
    model = MFA(
        centroids=centroids,
        rank=q,
        psi_per_component=psi_per_component,
        eps_floor=eps_floor,
    )

    if "_rot_T" not in state or "_rot_inv_Tt" not in state:
        eye = torch.eye(q, dtype=mu.dtype)
        state.setdefault("_rot_T", eye.repeat(K, 1, 1))
        state.setdefault("_rot_inv_Tt", eye.repeat(K, 1, 1))

    # Load weights/buffers
    model.load_state_dict(state, strict=strict)

    model._rotation_on = bool(meta.get("rotation_on", False))
    model._rotation_kind = meta.get("rotation_kind", None)
    model._rotation_params = meta.get("rotation_params", {})

    if device is not None:
        model = model.to(device)
    if dtype is not None:
        model = model.to(dtype=dtype)

    return model


def load_component_shards(
    path: str | Path,
    *,
    map_location: Optional[str | torch.device] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> MFA:
    """Assemble a full MFA from per-rank component-shard files.

    Accepts either the directory containing mfa_model_shards.json or the
    manifest path directly.
    """
    path = Path(path)
    manifest_path = (path / "mfa_model_shards.json") if path.is_dir() else path
    manifest = json.loads(manifest_path.read_text())
    base_dir = manifest_path.parent

    per_component_keys = ["mu", "dir_raw", "scale_rho", "pi_logits", "_rot_T", "_rot_inv_Tt"]
    shard_states: List[Dict[str, torch.Tensor]] = []
    first_meta: Dict[str, Any] = {}

    for fname in manifest["shards"]:
        ckpt = torch.load(base_dir / fname, map_location=map_location, weights_only=True)
        shard_states.append(ckpt["state_dict"])
        if not first_meta:
            first_meta = ckpt.get("meta", {})

    state: Dict[str, torch.Tensor] = {}
    for k in per_component_keys:
        if k in shard_states[0]:
            state[k] = torch.cat([s[k] for s in shard_states], dim=0)

    # psi_rho: shared (D,) when psi_per_component=False, per-component (K_local, D) otherwise
    if "psi_rho" not in state:
        psi_rho_0 = shard_states[0]["psi_rho"]
        if psi_rho_0.ndim == 2:
            state["psi_rho"] = torch.cat([s["psi_rho"] for s in shard_states], dim=0)
        else:
            state["psi_rho"] = psi_rho_0  # replicated; all ranks identical

    global_K = manifest["global_K"]
    D = first_meta["D"]
    q = first_meta["q"]
    psi_per_component = bool(first_meta.get("psi_per_component", False))
    eps_floor = float(first_meta.get("eps_floor", 1e-8))

    centroids = torch.zeros(global_K, D, dtype=state["mu"].dtype)
    model = MFA(centroids=centroids, rank=q,
                psi_per_component=psi_per_component, eps_floor=eps_floor)
    model.load_state_dict(state, strict=True)

    if device is not None:
        model = model.to(device)
    if dtype is not None:
        model = model.to(dtype=dtype)
    return model


@dataclass
class EncodedBatch:
    """
    Encoded representation of a batch against an MFA dictionary.
    """
    coeffs: torch.Tensor # (B, K*(1+q))
    alpha: torch.Tensor # (B, K) responsibilities
    z: torch.Tensor # (B, K, q) posterior means z_k aligned with dictionary
    dictionary: torch.Tensor # (D, K*(1+q))  atoms: [mu_k | W_k columns] over k
    recon: torch.Tensor # (B, D) coeffs @ dictionary.T
    index_map: List[Tuple[int, Optional[int]]]


class MFAEncoderDecoder:
    """
    Dictionary-based encoder/decoder for an MFA model.

    Builds a shared dictionary D whose columns are [mu_k | W_k[:,0] | ... | W_k[:,q-1]]
    stacked over all K components. Encoding a batch x returns sparse-style coefficients
    c such that c @ D.T ≈ x, where c_k = [alpha_k, alpha_k * z_k] and alpha_k is the
    responsibility of component k.
    """
    def __init__(self, model):
        self.model = model

    @torch.no_grad()
    def _current_params(self) -> Tuple[torch.Tensor, torch.Tensor]:
        W = self.model.W if hasattr(self.model, "W") else self.model._W()
        mu = self.model.mu
        return W, mu

    @torch.no_grad()
    def build_dictionary(self) -> Tuple[torch.Tensor, List[Tuple[int, Optional[int]]], Optional[torch.Tensor]]:
        W, mu = self._current_params() # (K,D,q), (K,D)
        K, D, q = W.shape
        device, dtype = W.device, W.dtype

        cols = []
        index_map: List[Tuple[int, Optional[int]]] = []
        for k in range(K):
            cols.append(mu[k].reshape(D, 1))
            index_map.append((k, None))
            cols.append(W[k])
            index_map.extend((k, j) for j in range(q))

        Dmat = torch.cat(cols, dim=1).to(device=device, dtype=dtype)
        return Dmat, index_map, None

    @torch.no_grad()
    def encode(self, x: torch.Tensor, *, tau: float = 1.0) -> EncodedBatch:
        """
        Encode a batch x into coefficients on the shared dictionary.
        """
        B, D = x.shape
        if D != self.model.D:
            raise ValueError(f"expected input dim {self.model.D}, got {D}")

        # Responsibilities and posterior means
        alpha = self.model.responsibilities(x, tau=tau) # (B, K)
        Ez, _Sz = self.model.component_posterior(x) # (B, K, q)

        # Build dictionary
        Dmat, index_map, _ = self.build_dictionary() # (D, K*(1+q))

        # assemble coefficient blocks
        blocks = [] 
        for k in range(self.model.K):
            ak = alpha[:, k:k+1] # (B,1)
            zk = Ez[:, k, :] # (B,q)
            blocks.append(torch.cat([ak, ak * zk], dim=1)) # (B,1+q)
        coeffs = torch.cat(blocks, dim=1).to(Dmat.dtype) # (B, K*(1+q))

        # Decode via single matmul
        recon = (coeffs @ Dmat.T).to(x.dtype) # (B, D)

        return EncodedBatch(
            coeffs=coeffs,
            alpha=alpha,
            z=Ez,
            dictionary=Dmat,
            recon=recon,
            index_map=index_map,
        )

    @torch.no_grad()
    def decode(self, coeffs: torch.Tensor) -> torch.Tensor:
        """
        Decode coefficient matrix back to R^D using the current dictionary.
        """
        Dmat, _imap, _ = self.build_dictionary()
        return (coeffs.to(Dmat.dtype) @ Dmat.T).to(Dmat.dtype)
