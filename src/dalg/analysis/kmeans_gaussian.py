"""Turn saved training-cluster PCA into an evaluation-only HDDC model.

Keep KMeans means and use its geometry PCA (never W_init). Saved eigenvalues
are scatter / N_k, so they already have the maximum-likelihood normalization.
No EM iterations or new PCA fit are needed.

evaluate_fit(..., model_kind="kmeans") always performs this conversion and
reuses HDDC NLL, BIC and posterior reconstruction. The BIC uses the repository's
HDDC parameter-count convention; this scores a hard-cluster fitted surrogate,
not an EM-optimized likelihood. Config and report identity remain KMeans.
"""

from __future__ import annotations

import torch

from dalg.models.adaptive_q.hddc_surgery import _softplus_inverse
from dalg.models.adaptive_q.mfa_hddc import MFA_HDDC
from dalg.models.kmeans import KMeans


def fit_mixture_logits(counts: torch.Tensor) -> torch.Tensor:
    """Estimate mixture logits from hard training membership counts.

    Input: nonnegative integer counts (K,), with positive total membership.
    Return: floating logits (K,) on the same device, whose softmax gives the
    maximum-likelihood mixture weights for these fixed hard assignments.
    Empty components must have exactly zero probability (logit -inf).
    No pseudocounts are added and the input is not modified.
    """
    counts_float = counts.to(torch.float64)
    class_probability = counts_float / counts_float.sum()
    return class_probability.log()


def fit_cluster_covariances(
    eigenvalues: torch.Tensor,
    ranks: torch.Tensor,
    *,
    q: int,
    noise_floor: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fit HDDC covariance parameters from each descending PCA spectrum.

    Inputs (same device): eigenvalues (K,D), proposed retained ranks (K,),
    allocated loading capacity q >= 1, and a positive variance noise_floor.
    Return (noise, scales, effective_ranks): shapes (K,), (K,q), (K,).
    Noise and scales are floating tensors; effective_ranks is integer.

    Covariance rules:
    - Cap each proposed rank at D-1, leaving a residual subspace. Keep existing
      directions; do not select new directions or increase a proposed rank.
    - Estimate isotropic residual variance from the discarded eigenvalues,
      respecting noise_floor.
    - Discard retained eigenvalues <= the fitted noise in one pass. With a
      descending spectrum, these are noise-floor violations or ties; removing
      them leaves the floored residual variance unchanged. Rank zero is allowed.
      D=1 therefore always has rank zero.
    - Find loading scales so W W.T + b I has the retained PCA eigenvalues.
      Remember that columns of KMeans.W are unit directions, not loadings.
    - Zero-pad scales beyond each effective rank to capacity q. A zero spectrum
      produces rank zero, noise_floor and zero scales (including empty clusters).

    Do not mutate inputs. The wrapper rejects occupied clusters without PCA,
    so missing singleton covariance is not silently treated as zero variance.
    """
    K, D = eigenvalues.shape
    ranks = ranks.clamp(max=D - 1)
    if q < 1 or bool((ranks > q).any()):
        raise ValueError("q must be positive and accommodate the retained ranks")
    dimensions = torch.arange(D, device=eigenvalues.device)

    retained = dimensions[None, :] < ranks[:, None]
    residual_sum = eigenvalues.masked_fill(retained, 0).sum(dim=-1)
    noise_per_component = (residual_sum / (D - ranks)).clamp_min(noise_floor)
    # Sorted spectra allow all noise-floor violations and ties to be removed at once.
    retained &= eigenvalues > noise_per_component[:, None]
    ranks = retained.sum(dim=-1)

    scales = eigenvalues.new_zeros(K, q)
    width = min(q, D)
    signal_variance = eigenvalues[:, :width] - noise_per_component[:, None]
    scales[:, :width] = signal_variance.clamp_min(0).sqrt().masked_fill(~retained[:, :width], 0)
    return noise_per_component, scales, ranks


@torch.no_grad()
def gaussian_from_cluster_pca(
    model: KMeans, *, train_cluster_sizes: torch.Tensor, noise_floor: float = 1e-4,
) -> MFA_HDDC:
    """Pack the fitted parameters into a fresh, frozen, component-noise HDDC model.

    The caller must supply a checkpoint whose PCA was fitted on the recorded
    training split. Matching counts catches split mismatches, but is not a
    substitute for that provenance requirement. Empty components keep their IDs
    with zero mixture weight; occupied components must have valid saved PCA.
    """
    directions = model.W
    counts = model.cluster_counts
    if not torch.equal(counts.cpu(), train_cluster_sizes.cpu()):
        raise ValueError("saved PCA counts differ from evaluation training counts")
    if int(counts.sum()) == 0:
        raise ValueError("Gaussian conversion requires training points")
    if bool(((counts > 0) & ~model.pca_valid).any()):
        raise ValueError("occupied clusters require valid PCA (at least two training points)")
    eps = 1e-5
    if not noise_floor > eps:
        raise ValueError("noise_floor must exceed the HDDC positivity offset (1e-5)")

    logits = fit_mixture_logits(counts)
    noise, scales, ranks = fit_cluster_covariances(
        model.eigenvalues, model.component_ranks, q=model.q, noise_floor=noise_floor,
    )
    if logits.shape != (model.K,) or noise.shape != (model.K,) or scales.shape != (model.K, model.q):
        raise ValueError("fitted parameters have incorrect shapes")
    if ranks.shape != (model.K,) or ranks.dtype != torch.long:
        raise ValueError("effective_ranks must be an int64 tensor of shape (K,)")
    if bool(((ranks < 0) | (ranks > model.component_ranks) | (ranks >= model.D)).any()):
        raise ValueError("effective ranks must not exceed saved ranks and must be below D")
    if not bool(torch.isfinite(noise).all() & (noise >= noise_floor).all()):
        raise ValueError("noise must be finite and at least noise_floor")
    if not bool(torch.isfinite(scales).all() & (scales >= 0).all()):
        raise ValueError("loading scales must be finite and nonnegative")
    if not bool(torch.isfinite(logits[counts > 0]).all() & torch.isneginf(logits[counts == 0]).all()):
        raise ValueError("only empty components must have -inf logits")

    gaussian = MFA_HDDC(
        model.mu.detach(), rank=model.q, init_directions=directions,
        isotropic_psi=True, eps_floor=eps,
    ).to(device=model.mu.device, dtype=model.mu.dtype)
    active = torch.arange(model.q, device=model.mu.device) < ranks[:, None]
    if bool((scales[active] <= 0).any()) or bool((scales[~active] != 0).any()):
        raise ValueError("scales must be positive on active columns and zero elsewhere")
    gaussian.rank_mask.copy_(active)
    gaussian.pi_logits.copy_(logits)
    gaussian.psi_rho.copy_(_softplus_inverse(noise[:, None] - eps))
    gaussian.scale_rho.copy_(_softplus_inverse(scales))
    return gaussian.requires_grad_(False).eval()
