"""Density support along Gaussian PC1 intervals and matched random directions.

The ambient kNN score is a local support surrogate; it is not an estimate of
an angle to an unknown manifold. All density averages are evaluated in log space.
"""
from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
import torch


QUADRATURE_GRID_POINTS = (201, 401, 801, 1601, 3201, 6401)


def log_average_density(log_density: torch.Tensor) -> torch.Tensor:
    """Log trapezoidal average on a uniform grid, along the last dimension."""
    if log_density.shape[-1] < 3 or not torch.isfinite(log_density).all():
        raise ValueError("Need at least three finite log densities")
    weights = torch.ones(log_density.shape[-1], dtype=log_density.dtype,
                         device=log_density.device)
    weights[[0, -1]] = 0.5
    return torch.logsumexp(log_density + weights.log(), dim=-1) - math.log(len(weights) - 1)


def random_unit_directions(dimension: int, count: int, seed: int) -> torch.Tensor:
    """Seeded isotropic directions, generated on CPU for device independence."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    directions = torch.randn(count, dimension, generator=generator, dtype=torch.float64)
    return directions / directions.norm(dim=1, keepdim=True)


class KNNLogDensity:
    """k/(n V_D r_k**D) using Euclidean distance in the original coordinates."""

    def __init__(self, reference: torch.Tensor, k: int = 20):
        if reference.ndim != 2 or not torch.isfinite(reference).all():
            raise ValueError("Reference must be a finite (N,D) tensor")
        if not 1 <= k <= len(reference) or reference.shape[1] < 1:
            raise ValueError("Require 1 <= k <= N and D >= 1")
        self.reference = reference.to(dtype=torch.float64)
        self.k = k
        self.dimension = reference.shape[1]
        log_volume = (self.dimension / 2 * math.log(math.pi)
                      - math.lgamma(self.dimension / 2 + 1))
        self.log_constant = math.log(k / len(reference)) - log_volume

    def from_squared_radii(self, squared_radii: torch.Tensor) -> torch.Tensor:
        if not torch.isfinite(squared_radii).all() or (squared_radii <= 0).any():
            raise ValueError("kNN radii must be positive and finite; no radius floor is applied")
        return self.log_constant - self.dimension / 2 * squared_radii.log()

    @torch.no_grad()
    def line_profiles(self, center, directions, offsets):
        """Exact distances along lines, without materializing (queries,D).

        Each line shares a center. Expanding ||reference-center-t*u||**2
        makes the expensive ambient dot products independent of grid size.
        Returns log densities and kNN radii for (directions, offsets).
        """
        center = center.to(self.reference)
        directions = directions.to(self.reference)
        offsets = offsets.to(self.reference)
        if (center.shape != (self.dimension,) or directions.ndim != 2
                or directions.shape[1] != self.dimension or offsets.ndim != 1
                or not all(torch.isfinite(t).all() for t in (center, directions, offsets))):
            raise ValueError("Invalid line center, directions, or offsets")
        if not torch.allclose(directions.norm(dim=1), torch.ones_like(directions[:, 0]),
                              atol=1e-10, rtol=1e-10):
            raise ValueError("Line directions must be unit vectors")
        residual = self.reference - center
        norm2 = residual.square().sum(dim=1)
        projection = directions @ residual.T
        squared_radii = []
        # One direction at a time bounds temporary storage to O(grid_size * N).
        for dot in projection:
            distance2 = (norm2[None, :] - 2 * offsets[:, None] * dot[None, :]
                         + offsets[:, None].square())
            squared_radii.append(distance2.kthvalue(self.k, dim=-1).values)
        squared_radii = torch.stack(squared_radii)
        return self.from_squared_radii(squared_radii), squared_radii.sqrt()

    @torch.no_grad()
    def adaptive_log_averages(self, center, directions, half_length, positions, profiles):
        """Integrate rescaled densities with QUADPACK and explicit peak splits.

        A uniform pilot grid locates peaks. Bounded searches refine their
        positions. Each direction starts with coarse uniform subintervals,
        with additional splits at pilot positions and refined peaks. Only
        failed convergence or excess error triggers finer subintervals for
        that direction, at the same tolerance.
        Absolute error is controlled relative to a pilot integral.
        """
        from scipy.integrate import quad
        from scipy.optimize import minimize_scalar

        residual = self.reference - center.to(self.reference)
        norm2 = residual.square().sum(1).cpu().numpy()
        projection = (directions.to(self.reference) @ residual.T).cpu().numpy()
        positions = positions.cpu().numpy()
        profiles = profiles.cpu().numpy()
        averages, relative_errors, evaluations = [], [], []
        for dot, profile in zip(projection, profiles):
            def log_density(s):
                d2 = norm2 - 2 * half_length * s * dot + (half_length * s)**2
                r2 = np.partition(d2, self.k - 1)[self.k - 1]
                if not math.isfinite(r2) or r2 <= 0:
                    raise ValueError('Adaptive quadrature encountered an invalid kNN radius')
                return self.log_constant - self.dimension / 2 * math.log(r2)

            peaks = np.flatnonzero((profile[1:-1] >= profile[:-2])
                                   & (profile[1:-1] >= profile[2:])) + 1
            peak_positions = []
            shift = float(profile.max())
            for index in peaks:
                optimum = minimize_scalar(lambda s: -log_density(s), method='bounded',
                                          bounds=(positions[index - 1], positions[index + 1]),
                                          options={'xatol': 1e-10})
                if not optimum.success:
                    raise RuntimeError('Could not resolve a pilot density peak')
                peak_positions.append(float(optimum.x))
                shift = max(shift, -float(optimum.fun))
            peak_breaks = np.r_[positions[1:-1], peak_positions]

            def integrand(s):
                return math.exp(log_density(s) - shift)

            initial_breaks = np.unique(np.r_[peak_breaks,
                np.linspace(-1, 1, QUADRATURE_GRID_POINTS[0])[1:-1]])
            pilot, _ = quad(integrand, -1, 1, points=initial_breaks, epsabs=1e-10,
                            epsrel=1e-3, limit=10000)
            if not math.isfinite(pilot) or pilot <= 0:
                raise ValueError('Adaptive density integral is nonpositive or nonfinite')
            total_evaluations = 0
            for grid_points in QUADRATURE_GRID_POINTS:
                breaks = np.unique(np.r_[peak_breaks, np.linspace(-1, 1, grid_points)[1:-1]])
                result = quad(integrand, -1, 1, points=breaks, epsabs=pilot * 1e-6,
                              epsrel=1e-4, limit=max(10000, 3 * len(breaks)), full_output=1)
                value, error, info = result[:3]
                total_evaluations += info['neval']
                if (value <= 0 or not math.isfinite(value) or not math.isfinite(error)
                        or error < 0):
                    raise ValueError('Adaptive density integral is nonpositive or nonfinite')
                if len(result) == 3 and error / value <= 1e-4:
                    break
            else:
                reason = result[3] if len(result) > 3 else 'relative error tolerance not met'
                raise RuntimeError(f'Adaptive density integration failed after {grid_points} '
                                   f'grid points: {reason}')
            averages.append(shift + math.log(value / 2))
            relative_errors.append(error / value)
            evaluations.append(total_evaluations)
        return (torch.tensor(averages, dtype=torch.float64, device=self.reference.device),
                max(relative_errors), max(evaluations))


@torch.no_grad()
def leading_components(model, kind: str) -> dict:
    """Extract true covariance PC1 and mark inactive or undefined geometry."""
    centers = model.mu.detach().cpu().double()
    K, D = centers.shape
    vectors = torch.zeros_like(centers)
    eigenvalues = torch.zeros(K, dtype=torch.float64)
    gap = torch.zeros_like(eigenvalues)
    reasons = [""] * K
    if kind == "kmeans":
        counts = model.cluster_counts.detach().cpu().double()
        spectrum = model.eigenvalues.detach().cpu()
        ranks = model.component_ranks.detach().cpu()
        for component in range(K):
            if not model.pca_valid[component]:
                reasons[component] = "invalid_cluster_pca"
                continue
            vectors[component] = model.W[component, :, 0].detach().cpu().double()
            eigenvalues[component] = spectrum[component, 0]
            gap[component] = spectrum[component, 0] - (spectrum[component, 1] if D > 1 else 0)
    elif kind == "hddc":
        counts = model.pi_logits.detach().cpu().double().softmax(0)
        ranks = model.component_ranks.detach().cpu()
        # Evaluate parameter transforms in the checkpoint dtype, then upcast.
        loadings = model._W().detach().cpu().double()
        noise = model._psi().detach().cpu().double()
        if not torch.allclose(noise, noise[:, :1].expand_as(noise)):
            raise ValueError("This experiment requires isotropic HDDC noise")
        for component in range(K):
            if not torch.isfinite(model.pi_logits[component]):
                reasons[component] = "zero_mixture_weight"
                continue
            if ranks[component] == 0:
                reasons[component] = "isotropic_covariance"
                continue
            u, s, _ = torch.linalg.svd(loadings[component], full_matrices=False)
            vectors[component] = u[:, 0]
            eigenvalues[component] = s[0].square() + noise[component, 0]
            gap[component] = s[0].square() - s[1].square() if len(s) > 1 else s[0].square()
    else:
        raise ValueError("Only kmeans and hddc are supported")
    for component in range(K):
        if reasons[component]:
            continue
        if not torch.isfinite(eigenvalues[component]) or eigenvalues[component] <= 0:
            reasons[component] = "nonpositive_leading_variance"
        elif gap[component] <= 1e-10 * eigenvalues[component]:
            reasons[component] = "nonunique_pc1"
        else:
            vectors[component] /= vectors[component].norm()
    return dict(centers=centers, vectors=vectors, eigenvalues=eigenvalues,
                eigengap=gap, ranks=ranks, occupancy=counts, exclusion_reasons=reasons)


@torch.no_grad()
def score_component(estimator, center, pc1, eigenvalue, *, n_random=32,
                    grid_points=101, probability=0.8, seed=42, adaptive=False,
                    half_length=None):
    """Return log retention, log enrichment, profiles, and grid diagnostics.

    The requested grid and a twice-refined grid share every coarse point.
    With adaptive=True scores use peak-split adaptive quadrature; otherwise
    scores use the refined grid. Coarse-grid changes diagnose under-resolution.
    A shared center sample can otherwise make short density spikes look equal.
    """
    if grid_points < 3 or grid_points % 2 != 1 or n_random < 1:
        raise ValueError("Need an odd grid_points >= 3 and at least one random direction")
    if not 0 < probability < 1 or not math.isfinite(eigenvalue) or eigenvalue <= 0:
        raise ValueError("Invalid interval probability or leading variance")
    if half_length is None:
        half_length = NormalDist().inv_cdf((1 + probability) / 2) * math.sqrt(eigenvalue)
    if not math.isfinite(half_length) or half_length <= 0:
        raise ValueError("Interval half-length must be positive and finite")
    relative_positions = torch.linspace(-1, 1, 2 * grid_points - 1, dtype=torch.float64)
    directions = torch.cat([pc1.cpu().double()[None],
                            random_unit_directions(estimator.dimension, n_random, seed)])
    profile, radii = estimator.line_profiles(center, directions, relative_positions * half_length)
    midpoint = profile.shape[1] // 2
    center_log_density = profile[0, midpoint]
    relative_profile = profile - center_log_density
    grid_retention = log_average_density(relative_profile)
    retention = grid_retention
    quadrature_error, quadrature_evaluations = 0., 0
    if adaptive:
        average, quadrature_error, quadrature_evaluations = estimator.adaptive_log_averages(
            center, directions, half_length, relative_positions, profile)
        retention = average - center_log_density
    coarse_retention = log_average_density(relative_profile[:, ::2])
    random_mean = torch.logsumexp(retention[1:], dim=0) - math.log(n_random)
    enrichment = retention[0] - random_mean
    coarse_enrichment = (coarse_retention[0]
                         - torch.logsumexp(coarse_retention[1:], dim=0) + math.log(n_random))
    center_fraction = (-math.log(profile.shape[1] - 1) - grid_retention).exp()
    return {
        "half_length": half_length,
        "center_log_density": float(center_log_density),
        "center_knn_radius": float(radii[0, midpoint]),
        "log_retention": float(retention[0]),
        "log_random_mean_retention": float(random_mean),
        "log_enrichment": float(enrichment),
        "random_ge_pc1_fraction": float((retention[1:] >= retention[0]).double().mean()),
        "max_grid_log_retention_change": float((grid_retention - coarse_retention).abs().max()),
        "grid_log_enrichment_change": float((grid_retention[0]
             - torch.logsumexp(grid_retention[1:], dim=0) + math.log(n_random) - coarse_enrichment).abs()),
        "max_center_quadrature_fraction": float(center_fraction.max()),
        "adaptive_quadrature_relative_error": quadrature_error,
        "adaptive_quadrature_max_evaluations": quadrature_evaluations,
        "max_adaptive_vs_grid_log_retention_change": float((retention - grid_retention).abs().max()),
        "pc1_endpoint_radius_ratio_min": float((radii[0, [0, -1]] / radii[0, midpoint]).min()),
        "relative_positions": relative_positions,
        "log_relative_density_profiles": relative_profile.cpu(),
        "log_direction_retentions": retention.cpu(),
    }


@torch.no_grad()
def fit_global_variance_pca(batches, variance_fraction=0.75):
    """Exact centered PCA from streamed float64 covariance, without whitening.

    Merge centered batch scatters as in the existing training-PCA plotting
    helper, then diagonalize once. Retain the smallest prefix reaching the
    requested fraction of total variance; no intrinsic-dimension rule is used.
    Projection basis rows are unit principal directions (shape d,D).
    """
    if not 0 < variance_fraction <= 1:
        raise ValueError('Variance fraction must be in (0,1]')
    count = 0
    mean = scatter = None
    for batch in batches:
        x = batch.detach().to(device='cpu', dtype=torch.float64)
        if x.ndim != 2 or len(x) == 0 or not torch.isfinite(x).all():
            raise ValueError('PCA requires finite nonempty activation batches')
        if mean is None:
            mean = torch.zeros(x.shape[1], dtype=torch.float64)
            scatter = torch.zeros(x.shape[1], x.shape[1], dtype=torch.float64)
        if x.shape[1] != len(mean):
            raise ValueError('Inconsistent PCA feature dimension')
        n = len(x)
        batch_mean = x.mean(0)
        centered = x - batch_mean
        delta = batch_mean - mean
        scatter += centered.T @ centered + torch.outer(delta, delta) * (count * n / (count + n))
        mean += delta * (n / (count + n))
        count += n
    if count < 2:
        raise ValueError('PCA requires at least two training activations')
    covariance = scatter / (count - 1)
    covariance = .5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    tolerance = 100 * len(mean) * torch.finfo(torch.float64).eps * eigenvalues.abs().max()
    if not torch.isfinite(eigenvalues).all() or (eigenvalues < -tolerance).any():
        raise ValueError('Invalid PCA covariance spectrum')
    eigenvalues = eigenvalues.flip(0).clamp_min(0)
    total = eigenvalues.sum()
    if total <= 0:
        raise ValueError('PCA training data has zero variance')
    ratios = eigenvalues / total
    cumulative = ratios.cumsum(0)
    d = int(torch.searchsorted(cumulative, torch.tensor(variance_fraction, dtype=torch.float64))) + 1
    if d > len(mean):
        raise ValueError('Requested variance fraction was not reached')
    basis = eigenvectors[:, -d:].flip(1).T.contiguous()
    return dict(mean=mean, basis=basis, dimension=d, input_dimension=len(mean),
                eigenvalues=eigenvalues, explained_variance_ratio=ratios,
                cumulative_explained_variance=cumulative,
                retained_variance_fraction=float(cumulative[d - 1]),
                previous_variance_fraction=float(cumulative[d - 2]) if d > 1 else 0.,
                target_variance_fraction=variance_fraction, total_variance=float(total),
                covariance_trace=float(covariance.trace()), training_count=count,
                centering=True, standardization=False, whitening=False,
                method='float64 streamed centered covariance; exact symmetric eigendecomposition',
                orthonormality_max_error=float((basis @ basis.T - torch.eye(d)).abs().max()))


@torch.no_grad()
def project_pc1_segments(geometry, mean, basis, *, null_norm_tolerance=1e-12):
    """Project original covariance PC1 intervals, without finding new PCs.

    The reduced unit direction is Pv/||Pv||, while the interval half-length is
    the original Gaussian marginal half-length times ||Pv||. Normalizing its
    density integral by projected length preserves the original-t average.
    """
    mean, basis = mean.double(), basis.double()
    if (basis.ndim != 2 or mean.shape != (basis.shape[1],)
            or not torch.isfinite(mean).all() or not torch.isfinite(basis).all()
            or not torch.allclose(basis @ basis.T, torch.eye(len(basis), dtype=torch.float64),
                                  atol=1e-10, rtol=1e-10)):
        raise ValueError('Expected a finite mean and orthonormal row projection basis')
    centers = (geometry['centers'] - mean) @ basis.T
    projected = geometry['vectors'] @ basis.T
    norms = projected.norm(dim=1)
    alpha = norms.square()
    if (alpha > 1 + 1e-10).any():
        raise ValueError('Orthogonal projection increased a unit PC1 norm')
    reasons = list(geometry['exclusion_reasons'])
    for component in range(len(reasons)):
        if not reasons[component] and norms[component] <= null_norm_tolerance:
            reasons[component] = 'null_pc1_projection'
    vectors = torch.zeros_like(projected)
    eligible = torch.tensor([not reason for reason in reasons])
    vectors[eligible] = projected[eligible] / norms[eligible, None]
    original_half_lengths = NormalDist().inv_cdf(.9) * geometry['eigenvalues'].clamp_min(0).sqrt()
    return dict(centers=centers, vectors=vectors, alpha=alpha,
                original_half_lengths=original_half_lengths,
                half_lengths=original_half_lengths * norms, exclusion_reasons=reasons,
                null_norm_tolerance=null_norm_tolerance)
