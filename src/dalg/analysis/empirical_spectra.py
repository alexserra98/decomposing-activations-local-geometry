"""Full spectra of empirical covariances under frozen soft responsibilities."""

from __future__ import annotations

import torch

from dalg.models.adaptive_q.hddc_surgery import accumulate_statistics


def _spectra_from_statistics(mu, counts, residual_sum, scatter, *, eig_batch_size):
    """Center residual moments on their weighted means, then diagonalize."""
    if eig_batch_size <= 0:
        raise ValueError("eig_batch_size must be positive")
    if any(not torch.isfinite(t).all() for t in (mu, counts, residual_sum, scatter)):
        raise ValueError("Empirical covariance moments must be finite")
    if (counts < 0).any():
        raise ValueError("Effective counts must be nonnegative")
    K, D = mu.shape
    valid = counts > 0
    eigenvalues = torch.full((K, D), float("nan"), dtype=torch.float64)
    means = torch.full_like(eigenvalues, float("nan"))
    ids = valid.nonzero(as_tuple=True)[0]
    for batch_ids in ids.split(eig_batch_size):
        shift = residual_sum[batch_ids] / counts[batch_ids, None]
        second = scatter[batch_ids] / counts[batch_ids, None, None]
        correction = shift[:, :, None] * shift[:, None, :]
        covariance = second - correction
        covariance = 0.5 * (covariance + covariance.transpose(-1, -2))
        if not torch.isfinite(covariance).all():
            raise ValueError("Empirical covariance contains nonfinite values")
        values = torch.linalg.eigvalsh(covariance).flip(-1)
        # Account for both mean subtraction and symmetric eigensolver roundoff.
        scale = second.abs().amax(dim=(-2, -1)) + correction.abs().amax(dim=(-2, -1))
        tolerance = 100 * D * torch.finfo(torch.float64).eps * scale
        if not torch.isfinite(values).all() or (values < -tolerance[:, None]).any():
            raise ValueError("Empirical covariance has nonfinite or materially negative eigenvalues")
        local_means = mu[batch_ids] + shift
        if not torch.isfinite(local_means).all():
            raise ValueError("Empirical means contain nonfinite values")
        eigenvalues[batch_ids.cpu()] = values.clamp_min(0).cpu()
        means[batch_ids.cpu()] = local_means.cpu()
    return {
        "eigenvalues": eigenvalues,
        "effective_counts": counts.cpu(),
        "empirical_means": means,
        "valid": valid.cpu(),
    }


@torch.no_grad()
def compute_empirical_spectra(model, loader, *, device="cpu", eig_batch_size=32):
    """Stream full soft covariances; move and freeze the supplied model in float64.

    Uses O(K D^2) accumulator memory, intended for low-dimensional toy runs.
    Covariances are centered on weighted empirical means and divided by total
    responsibility (ML normalization). Zero-mass components have NaN outputs.
    """
    device = torch.device(device)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Float64 spectrum computation requires CPU or CUDA")
    if eig_batch_size <= 0:
        raise ValueError("eig_batch_size must be positive")
    model.to(device=device, dtype=torch.float64).eval().requires_grad_(False)

    def batches():
        for batch in loader:
            x = batch[0] if isinstance(batch, (tuple, list)) else batch
            if x.ndim != 2 or x.shape[1] != model.D or not torch.isfinite(x).all():
                raise ValueError(f"Expected finite activation batches of shape (B, {model.D})")
            yield x.to(device=device, dtype=torch.float64)

    with model.inference_cache():
        counts, residual_sum, scatter, n_rows = accumulate_statistics(
            model, batches(), device=device, hard_assignment_covariance=False,
        )
    result = _spectra_from_statistics(
        model.mu.detach(), counts, residual_sum, scatter, eig_batch_size=eig_batch_size,
    )
    result["n_activations"] = n_rows
    return result
