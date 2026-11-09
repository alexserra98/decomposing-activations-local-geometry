"""Per-activation reconstruction errors and streaming mean/standard error."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@torch.no_grad()
def reconstruction_errors(model, x: torch.Tensor, *, model_kind: str,
                          labels: torch.Tensor | None = None) -> tuple[torch.Tensor, int]:
    """Return dimension-normalized errors for eligible samples and excluded count.

    HDDC uses the posterior mixture mean. KMeans projects onto retained cluster
    PCs and excludes assignments to components without valid training PCA.
    ``labels`` optionally reuses nearest-centroid predictions for this batch.
    """
    if model_kind == "hddc":
        reconstructed = model.reconstruct(x, use_mixture_mean=True)
        return (x.double() - reconstructed.double()).square().mean(dim=1), 0
    if model_kind != "kmeans":
        raise ValueError(f"unsupported reconstruction model: {model_kind!r}")

    pcs = model.W
    labels = model.predict(x) if labels is None else labels
    eligible = model.pca_valid[labels]
    excluded = int((~eligible).sum())
    x, labels = x[eligible], labels[eligible]
    errors = torch.empty(len(x), device=x.device, dtype=torch.float64)
    # Group by component to avoid a (batch, D, q) copy of the local bases.
    for component in labels.unique().tolist():
        members = labels == component
        basis = pcs[component, :, :int(model.component_ranks[component])]
        centered = x[members] - model.mu[component]
        reconstructed = model.mu[component] + (centered @ basis) @ basis.T
        errors[members] = (x[members].double() - reconstructed.double()).square().mean(dim=1)
    return errors, excluded


@dataclass
class ReconstructionMoments:
    """Merge batch moments without storing errors for the whole population."""

    n: int = 0
    excluded_n: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, errors: torch.Tensor, excluded_n: int = 0) -> None:
        if errors.ndim != 1 or not bool(torch.isfinite(errors).all()) or bool((errors < 0).any()):
            raise ValueError("reconstruction errors must be a finite nonnegative vector")
        if type(excluded_n) is not int or excluded_n < 0:
            raise ValueError("excluded_n must be a nonnegative integer")
        self.excluded_n += excluded_n
        count = errors.numel()
        if not count:
            return
        errors = errors.double()
        mean = float(errors.mean())
        m2 = float((errors - mean).square().sum())
        total = self.n + count
        delta = mean - self.mean
        self.m2 += m2 + delta * delta * self.n * count / total
        self.mean += delta * count / total
        self.n = total

    def summary(self) -> dict[str, float | int | None]:
        return {
            "mse": self.mean if self.n else None,
            "standard_error": math.sqrt(self.m2 / (self.n - 1) / self.n) if self.n > 1 else None,
            "n": self.n,
            "excluded_n": self.excluded_n,
        }
