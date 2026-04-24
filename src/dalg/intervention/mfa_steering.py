from __future__ import annotations

from typing import List, Optional, Union
import torch
from transformer_lens import HookedTransformer, utils


class MFASteerer:
    """
    Steers LLM residual-stream (or MLP) activations using an MFA model.

    Two intervention strategies are supported:

    1. **Mean steering** (`intervene` / `generate`):
       x' = (1 - alpha) * x + alpha * mu_k
       Interpolates the activation towards the centroid of component k.

    2. **Latent two-stage steering** (`intervene_latent` / `generate_latent`):
       Step 1: x1 = x + alpha_centroid * (mu_k - x)   (centroid pull)
       Step 2: x' = x1 + W_k @ z                       (within-subspace move)
       Combines centroid pull with an explicit displacement in the local subspace.
    """

    _SITES = {
        "resid_post": lambda L: f"blocks.{L}.hook_resid_post",
        "mlp_act":    lambda L: f"blocks.{L}.mlp.hook_post",
        "mlp_out":    lambda L: f"blocks.{L}.hook_mlp_out",
    }

    def __init__(self, model: HookedTransformer, mfa, intervention_type: str = "resid_post"):
        if intervention_type not in self._SITES:
            raise ValueError(f"Unsupported intervention_type: {intervention_type}")
        self.model = model
        self.mfa = mfa
        self.intervention_type = intervention_type

    # ---- internals ----

    def _site(self, layer: int) -> str:
        return self._SITES[self.intervention_type](layer)

    def _to_tokens(self, prompt_or_tokens: Union[str, torch.Tensor]) -> torch.Tensor:
        if isinstance(prompt_or_tokens, str):
            return self.model.to_tokens(prompt_or_tokens)
        return prompt_or_tokens

    def _make_hooks(self, hook_fn, layers: List[int]):
        return [(self._site(L), hook_fn) for L in layers]

    @staticmethod
    def _flatten(value: torch.Tensor):
        """Flatten (B, T, D) -> (B*T, D) for hook functions; no-op for (N, D)."""
        if value.ndim == 3:
            B, T, D = value.shape
            return value.reshape(B * T, D), value.shape
        return value, value.shape

    def _get_W(self) -> torch.Tensor:
        for attr in ("W", "Lambda", "loadings"):
            if hasattr(self.mfa, attr):
                return getattr(self.mfa, attr)
        raise AttributeError("MFA loadings not found: expected .W, .Lambda, or .loadings")

    def _responsibilities(self, flat: torch.Tensor) -> torch.Tensor:
        r = self.mfa.responsibilities(flat)  # (N, K)
        return r.to(device=flat.device, dtype=torch.float32)

    # ---- hook constructors ----

    def _hook_mean(self, alpha: float, k: Optional[int]):
        """Hook that pulls activations towards centroid mu_k (or mixture centroid if k=None)."""
        mu = self.mfa.mu  # (K, D)

        @torch.inference_mode()
        def hook_fn(value: torch.Tensor, hook) -> torch.Tensor:
            flat, orig_shape = self._flatten(value)
            device, dtype = flat.device, flat.dtype
            mu_local = mu.to(device=device, dtype=dtype)

            if k is None:
                r = self._responsibilities(flat)
                target = (r @ mu_local.float()).to(dtype=dtype)
            else:
                target = mu_local[k].expand_as(flat)

            return ((1.0 - alpha) * flat + alpha * target).reshape(orig_shape)

        return hook_fn

    def _hook_latent_two_stage(self, alpha_centroid: float, z: Union[torch.Tensor, list], k: Optional[int]):
        """
        Hook that first pulls towards the centroid, then moves along the local subspace.

        Step 1:  x1 = x + alpha_centroid * (mu_k - x)
        Step 2:  x' = x1 + W_k @ z
        """
        mu = self.mfa.mu
        W = self._get_W()

        @torch.inference_mode()
        def hook_fn(value: torch.Tensor, hook) -> torch.Tensor:
            flat, orig_shape = self._flatten(value)
            device, dtype = flat.device, flat.dtype
            mu_local = mu.to(device=device, dtype=dtype)
            W_local  = W.to(device=device, dtype=dtype)

            z_local = z if isinstance(z, torch.Tensor) else torch.tensor(z)
            z_local = z_local.to(device=device, dtype=dtype)

            K, D = mu_local.shape
            _, _, q = W_local.shape
            N = flat.shape[0]

            if k is not None:
                centroid = mu_local[k].unsqueeze(0).expand(N, D)
                x1 = flat + alpha_centroid * (centroid - flat)

                if z_local.ndim == 1:
                    delta = (W_local[k] @ z_local).view(1, D).expand(N, D)
                elif z_local.ndim == 2 and z_local.shape == (N, q):
                    delta = z_local @ W_local[k].T
                else:
                    raise ValueError(
                        f"When k is specified, z must be (q,) or (N,q)=({N},{q}). Got {tuple(z_local.shape)}"
                    )
                return (x1 + delta).reshape(orig_shape)

            # k is None: use responsibility-weighted centroid and loadings
            r = self._responsibilities(flat)  # (N, K) fp32

            centroid = (r @ mu_local.float()).to(dtype=dtype)
            x1 = flat + alpha_centroid * (centroid - flat)

            if z_local.ndim == 1:
                W_eff = torch.einsum("nk,kdq->ndq", r, W_local.float())
                delta = torch.einsum("ndq,q->nd", W_eff, z_local.float()).to(dtype=dtype)
            elif z_local.ndim == 2 and z_local.shape == (N, q):
                W_eff = torch.einsum("nk,kdq->ndq", r, W_local.float())
                delta = torch.einsum("ndq,nq->nd", W_eff, z_local.float()).to(dtype=dtype)
            elif z_local.ndim == 2 and z_local.shape == (K, q):
                Wz_k = torch.einsum("kdq,kq->kd", W_local.float(), z_local.float())
                delta = (r @ Wz_k).to(dtype=dtype)
            else:
                raise ValueError(
                    f"When k=None, z must be (q,), (N,q)=({N},{q}), or (K,q)=({K},{q}). "
                    f"Got {tuple(z_local.shape)}"
                )

            return (x1 + delta).reshape(orig_shape)

        return hook_fn

    # ---- public API ----

    @torch.inference_mode()
    def intervene(
        self,
        prompt_or_tokens: Union[str, torch.Tensor],
        layers: List[int],
        alpha: float,
        k: Optional[int] = None,
    ) -> torch.Tensor:
        """Run a single forward pass with mean steering; returns logits."""
        tokens = self._to_tokens(prompt_or_tokens)
        hooks = self._make_hooks(self._hook_mean(alpha, k), layers)
        return self.model.run_with_hooks(tokens, fwd_hooks=hooks)

    @torch.inference_mode()
    def generate(
        self,
        prompt_or_tokens: Union[str, torch.Tensor],
        layers: List[int],
        alpha: float,
        k: Optional[int] = None,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
    ) -> str:
        """Generate text with mean steering applied at every forward pass."""
        tokens = self._to_tokens(prompt_or_tokens)
        hooks = self._make_hooks(self._hook_mean(alpha, k), layers)
        out = self.model.generate(
            tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            fwd_hooks=hooks,
        )
        return self.model.to_string(out[0])

    @torch.inference_mode()
    def intervene_latent(
        self,
        prompt_or_tokens: Union[str, torch.Tensor],
        layers: List[int],
        alpha_centroid: float,
        z: Union[torch.Tensor, list],
        k: Optional[int] = None,
    ) -> torch.Tensor:
        """Run a single forward pass with latent two-stage steering; returns logits."""
        tokens = self._to_tokens(prompt_or_tokens)
        hooks = self._make_hooks(self._hook_latent_two_stage(alpha_centroid, z, k), layers)
        return self.model.run_with_hooks(tokens, fwd_hooks=hooks)

    @torch.inference_mode()
    def generate_latent(
        self,
        prompt_or_tokens: Union[str, torch.Tensor],
        layers: List[int],
        alpha_centroid: float,
        z: Union[torch.Tensor, list],
        k: Optional[int] = None,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
    ) -> str:
        """Generate text with latent two-stage steering applied at every forward pass."""
        tokens = self._to_tokens(prompt_or_tokens)
        hooks = self._make_hooks(self._hook_latent_two_stage(alpha_centroid, z, k), layers)
        out = self.model.generate(
            tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            fwd_hooks=hooks,
        )
        return self.model.to_string(out[0])

    @torch.inference_mode()
    def generate_latent_sampling(
        self,
        prompt: str,
        layers: List[int],
        alpha_centroid: float,
        z: Union[torch.Tensor, list],
        k: Optional[int] = None,
        *,
        max_new_tokens: int = 50,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        temperature: float = 1.0,
        freq_penalty: float = 0.0,
        m: int = 1,
        use_past_kv_cache: bool = True,
    ) -> List[str]:
        """
        Generate m independent samples with latent two-stage steering.

        Args:
            prompt: Input text.
            layers: Layers at which to apply the hook.
            alpha_centroid: Centroid pull strength.
            z: Latent displacement vector(s).
            k: Component index (None uses responsibility-weighted mixture).
            max_new_tokens: Maximum tokens to generate per sample.
            m: Number of independent samples to generate in parallel.
            use_past_kv_cache: Use KV cache for efficient autoregressive generation.

        Returns:
            List of m generated strings (each includes the prompt).
        """
        device = self.model.cfg.device
        tokens = self.model.to_tokens(prompt).to(device).repeat(m, 1)

        past_kv_cache = None
        if use_past_kv_cache:
            from transformer_lens import HookedTransformerKeyValueCache
            past_kv_cache = HookedTransformerKeyValueCache.init_cache(self.model.cfg, device, m)

        hook_fn = self._hook_latent_two_stage(alpha_centroid, z, k)
        fwd_hooks = self._make_hooks(hook_fn, layers)

        for i in range(max_new_tokens):
            inp = tokens if (i == 0 or not use_past_kv_cache) else tokens[:, -1:]
            logits = self.model.run_with_hooks(inp, fwd_hooks=fwd_hooks, past_kv_cache=past_kv_cache)

            next_tok = utils.sample_logits(
                logits[:, -1, :],
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                freq_penalty=freq_penalty,
                tokens=tokens,
            ).unsqueeze(1)
            tokens = torch.cat([tokens, next_tok], dim=1)

        return [self.model.to_string(tokens[i]) for i in range(m)]
