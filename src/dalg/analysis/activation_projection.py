"""Global PCA preprocessing for activation-direction comparisons."""

from __future__ import annotations

import math

import torch
from sklearn.decomposition import PCA


@torch.no_grad()
def fit_activation_pca(activations: torch.Tensor, n_components: int = 3) -> dict:
    """Fit centered, unwhitened PCA on the supplied activation population.

    Returns a float64 CPU mean (D,), orthonormal row basis (d,D), and
    explained-variance metadata. No labels or model geometry enter this fit.
    The caller chooses the fitting population and reuses the basis across
    comparisons. A full SVD gives the leading PCs without randomized fitting.
    """
    if (activations.ndim != 2 or not activations.is_floating_point()
            or not torch.isfinite(activations).all()):
        raise ValueError("activations must be a finite floating (N,D) matrix")
    if type(n_components) is not int or not 1 <= n_components <= min(len(activations) - 1, activations.shape[1]):
        raise ValueError("Require 1 <= n_components <= min(N-1, D)")
    values = activations.detach().to(device="cpu", dtype=torch.float64)
    if not (values.var(dim=0).sum() > 0):
        raise ValueError("PCA fitting activations must have positive variance")
    pca = PCA(n_components=n_components, svd_solver="full", whiten=False)
    pca.fit(values.numpy())
    return dict(mean=torch.from_numpy(pca.mean_.copy()),
                basis=torch.from_numpy(pca.components_.copy()),
                explained_variance_ratio=torch.from_numpy(pca.explained_variance_ratio_.copy()),
                retained_variance_fraction=float(pca.explained_variance_ratio_.sum()),
                dimension=n_components, input_dimension=values.shape[1], fitting_count=len(values),
                centering=True, whitening=False, standardization=False, method="full SVD")


@torch.no_grad()
def project_alignment_inputs(activations: torch.Tensor, geometries: dict, projection: dict,
                             *, null_norm_tolerance: float = 1e-12) -> tuple[torch.Tensor, dict]:
    """Project activations and original model PC1s into the same PCA space.

    Compute native model assignments before this step: assignments and saved
    ranks are preserved. Directions are projected without centering, then
    normalized. A vanishing projection makes a PC1 ineligible; its assignment
    weight remains present. Original tensors and geometry dictionaries are not
    modified. ``pc1_projection_squared_norm`` records retained directional mass.
    """
    mean, basis = projection["mean"], projection["basis"]
    if (activations.ndim != 2 or not activations.is_floating_point()
            or activations.shape[1] != basis.shape[1] or not torch.isfinite(activations).all()):
        raise ValueError("activations must be finite and match the PCA input dimension")
    if not math.isfinite(null_norm_tolerance) or null_norm_tolerance < 0:
        raise ValueError("null_norm_tolerance must be finite and nonnegative")
    projected_activations = (activations.detach().to(mean) - mean) @ basis.T
    result = {}
    for kind, geometry in geometries.items():
        pc1 = geometry["pc1"].to(basis)
        eligible = geometry["identifiable"].clone()
        if pc1.shape != (len(eligible), basis.shape[1]):
            raise ValueError("Model PC1s must match the PCA input dimension")
        if not torch.isfinite(pc1[eligible]).all():
            raise ValueError("Eligible model PC1s must be finite")
        vectors = basis.new_zeros((len(pc1), len(basis)))
        vectors[eligible] = pc1[eligible] @ basis.T
        norms = vectors.norm(dim=1)
        null = eligible & (norms <= null_norm_tolerance)
        reasons = list(geometry["exclusion_reasons"])
        for k in null.nonzero(as_tuple=True)[0].tolist():
            reasons[k] = "null_pc1_projection"
        eligible &= ~null
        vectors[eligible] /= norms[eligible, None]
        vectors[~eligible] = 0
        result[kind] = dict(geometry, pc1=vectors, identifiable=eligible,
                            exclusion_reasons=reasons, pc1_projection_squared_norm=norms.square())
    return projected_activations, result


@torch.no_grad()
def project_gaussian_components(model, kind: str, projection: dict) -> dict:
    """Project component means and full covariances without forming D×D matrices.

    HDDC uses its fitted loadings and noise. KMeans uses the existing Gaussian
    approximation from saved cluster PCA, including fitted residual noise.
    The result describes the marginal Gaussian in the global PCA coordinates.
    """
    if kind == "kmeans":
        from dalg.analysis.kmeans_gaussian import gaussian_from_cluster_pca

        model = gaussian_from_cluster_pca(model, train_cluster_sizes=model.cluster_counts)
    elif kind != "hddc":
        raise ValueError("Only kmeans and hddc are supported")
    mean, basis = projection["mean"], projection["basis"]
    centers = (model.mu.detach().to(basis) - mean) @ basis.T
    loadings = basis @ model._W().detach().to(basis)
    noise = model._psi().detach().to(basis)
    covariances = loadings @ loadings.transpose(1, 2)
    covariances += torch.einsum("ad,kd,bd->kab", basis, noise, basis)
    return dict(centers=centers, covariances=covariances,
                valid=torch.isfinite(model.pi_logits.detach().cpu()))
