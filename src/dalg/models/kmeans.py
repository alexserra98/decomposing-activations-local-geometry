"""Euclidean KMeans with independently fitted local principal components."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from dalg.init.centroid_artifact import compute_cluster_pca_directions
from dalg.init.neighborhood_pca import compute_neighborhood_pca, nearest_neighbor_indices
from dalg.init.projected_knn import KMeansTorch


KMEANS_MODEL_FORMAT = "dalg_kmeans_v3"
KMEANS_MODEL_VERSION = 3


class KMeans(nn.Module):
    """Hard nearest-centroid model with ordered, unscaled local PC directions.

    ``fit`` estimates only the centroids. ``compute_pcs`` estimates the
    full local PCA basis, selects Cattell ranks, and retains only those directions
    in ``W``, with zero padding across components. All model tensors are non-trainable buffers.
    """

    def __init__(
        self,
        K: int,
        *,
        max_iter: int = 100,
        restarts: int = 10,
        tol: float = 1e-6,
        seed: int = 0,
        device: str | torch.device | None = None,
        block_x: int = 8192,
        block_c: int = 8192,
    ):
        super().__init__()
        if K < 1:
            raise ValueError("K must be positive")
        if max_iter < 1 or restarts < 1:
            raise ValueError("max_iter and restarts must be positive")
        if not math.isfinite(tol) or tol < 0:
            raise ValueError("tol must be finite and nonnegative")
        if block_x < 1 or block_c < 1:
            raise ValueError("block_x and block_c must be positive")
        self.K = int(K)
        self.max_iter = int(max_iter)
        self.restarts = int(restarts)
        self.tol = float(tol)
        self.seed = int(seed)
        self.block_x = int(block_x)
        self.block_c = int(block_c)
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.register_buffer("mu", torch.empty(self.K, 0, device=device))
        self.register_buffer("_pcs", torch.empty(self.K, 0, 0, device=device))
        self.register_buffer("_eigenvalues", torch.empty(self.K, 0, dtype=torch.float64, device=device))
        self.register_buffer("_component_ranks", torch.zeros(self.K, dtype=torch.long, device=device))
        self.register_buffer("_init_pcs", torch.empty(self.K, 0, 0, device=device))
        self.register_buffer("_pca_valid", torch.zeros(self.K, dtype=torch.bool, device=device))
        self.register_buffer("_cluster_counts", torch.zeros(self.K, dtype=torch.long, device=device))
        self.fit_metadata: dict[str, Any] = {}
        self.init_pca_neighbors: int | None = None
        self.init_pca_n_samples: int | None = None
        self.surgery_threshold: float | None = None
        self.checkpoint_extra: dict[str, Any] = {}

    @classmethod
    def from_centroids(cls, centroids: torch.Tensor, **fit_settings) -> KMeans:
        """Create a centroid-only model without refitting the supplied centers."""
        cls._validate_matrix(centroids, name="centroids")
        fit_settings.setdefault("device", centroids.device)
        model = cls(int(centroids.shape[0]), **fit_settings)
        model.mu = centroids.detach().to(device=model.mu.device).clone()
        model._invalidate_pca()
        return model

    @property
    def D(self) -> int:
        return int(self.mu.shape[1])

    @property
    def q(self) -> int:
        return int(self._pcs.shape[-1])

    @property
    def W(self) -> torch.Tensor:
        self._require_pca()
        return self._pcs

    @property
    def rank_mask(self) -> torch.Tensor:
        self._require_pca()
        return torch.arange(self.q, device=self.mu.device) < self._component_ranks[:, None]

    @property
    def eigenvalues(self) -> torch.Tensor:
        self._require_pca()
        return self._eigenvalues

    @property
    def component_ranks(self) -> torch.Tensor:
        self._require_pca()
        return self._component_ranks

    @property
    def q_init(self) -> int:
        return int(self._init_pcs.shape[-1])

    @property
    def W_init(self) -> torch.Tensor:
        if self.q_init == 0:
            raise RuntimeError("compute_init_pcs must be called before accessing initialization PCs")
        return self._init_pcs

    @property
    def pca_valid(self) -> torch.Tensor:
        self._require_pca()
        return self._pca_valid

    @property
    def cluster_counts(self) -> torch.Tensor:
        self._require_pca()
        return self._cluster_counts

    @staticmethod
    def _validate_matrix(X: torch.Tensor, *, name: str = "X") -> None:
        if X.ndim != 2 or min(X.shape) < 1:
            raise ValueError(f"{name} must be a nonempty (N, D) tensor")
        if not X.is_floating_point() or not bool(torch.isfinite(X).all()):
            raise ValueError(f"{name} must contain finite floating-point values")

    def _require_fitted(self) -> None:
        if self.D == 0:
            raise RuntimeError("fit KMeans or supply centroids before using the model")

    def _require_pca(self) -> None:
        if self.q == 0:
            raise RuntimeError("compute_pcs must be called before accessing PCA state")

    def _invalidate_pca(self) -> None:
        self._pcs = self.mu.new_empty(self.K, self.D, 0)
        self._eigenvalues = torch.empty(self.K, 0, dtype=torch.float64, device=self.mu.device)
        self._component_ranks = torch.zeros(self.K, dtype=torch.long, device=self.mu.device)
        self._init_pcs = self.mu.new_empty(self.K, self.D, 0)
        self._pca_valid = torch.zeros(self.K, dtype=torch.bool, device=self.mu.device)
        self._cluster_counts = torch.zeros(self.K, dtype=torch.long, device=self.mu.device)
        self.init_pca_neighbors = None
        self.init_pca_n_samples = None
        self.surgery_threshold = None

    def _solver(self) -> KMeansTorch:
        return KMeansTorch(
            k=self.K, metric="euclidean", n_iter=self.max_iter,
            restarts=self.restarts, tol=self.tol, seed=self.seed,
            device=self.mu.device, dtype=self.mu.dtype,
            block_x=self.block_x, block_c=self.block_c,
        )

    @torch.no_grad()
    def fit(self, X: torch.Tensor) -> KMeans:
        self._validate_matrix(X)
        if X.shape[0] < self.K:
            raise ValueError(f"K={self.K} exceeds the number of points N={X.shape[0]}")
        solver = self._solver()
        self.mu = solver.fit(X).detach().clone()
        self.fit_metadata = {
            "n_samples": int(X.shape[0]),
            "inertia": float(solver.inertia_),
            "n_iter_run": solver.n_iter_run_,
        }
        self.checkpoint_extra = {}
        self._invalidate_pca()
        return self

    @torch.no_grad()
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Return cluster IDs, choosing the first centroid when distances tie."""
        self._require_fitted()
        if X.ndim != 2 or X.shape[1] != self.D:
            raise ValueError(f"X must have shape (N, D={self.D})")
        X = X.to(device=self.mu.device, dtype=self.mu.dtype)
        return self._solver()._assign_streamed(X, self.mu)

    @torch.no_grad()
    def responsibilities(self, X: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
        """Return floating hard one-hot responsibilities; temperature is inert."""
        if not math.isfinite(tau) or tau <= 0:
            raise ValueError("tau must be finite and positive")
        return F.one_hot(self.predict(X), num_classes=self.K).to(dtype=self.mu.dtype)

    def _validate_pca_input(self, X: torch.Tensor, rank: int) -> None:
        self._require_fitted()
        self._validate_matrix(X)
        if X.shape[1] != self.D:
            raise ValueError(f"X must have dimension D={self.D}")
        if not 1 <= rank <= self.D:
            raise ValueError(f"rank must be in [1, {self.D}], got {rank}")

    @torch.no_grad()
    def compute_init_pcs(
        self, X: torch.Tensor, *, rank: int, neighbors: int = 64,
        eig_batch_size: int = 256,
    ) -> KMeans:
        """Store fixed-capacity KNN directions for MFA initialization only."""
        self._validate_pca_input(X, rank)
        if not rank < neighbors <= X.shape[0]:
            raise ValueError(f"KNN PCA requires rank={rank} < neighbors <= N={X.shape[0]}")
        indices, _ = nearest_neighbor_indices(
            X, self.mu, neighbors=neighbors, device=self.mu.device,
            point_block_size=self.block_x,
        )
        directions, _ = compute_neighborhood_pca(
            X, self.mu, indices, rank=rank, device=self.mu.device,
            eig_batch_size=eig_batch_size, return_eigenvalues=True,
        )
        self._init_pcs = directions.to(device=self.mu.device, dtype=self.mu.dtype)
        self.init_pca_neighbors = int(neighbors)
        self.init_pca_n_samples = int(X.shape[0])
        return self

    @torch.no_grad()
    def compute_pcs(
        self, X: torch.Tensor, *, threshold: float = 0.1,
        chunk_elems: int = 1 << 23, eig_batch_size: int = 256,
    ) -> KMeans:
        """Compute all cluster PCs and retain only the Cattell-selected directions.

        Clusters with fewer than two members keep their IDs and zero geometry.
        Selected ranks cannot exceed the sample-supported capacity N_k - 1.
        """
        self._validate_pca_input(X, self.D)
        threshold = self._validate_threshold(threshold)
        if chunk_elems <= 0 or eig_batch_size <= 0:
            raise ValueError("PCA chunk and batch sizes must be positive")
        points = X.to(device=self.mu.device)
        labels = self.predict(points)
        counts = torch.bincount(labels, minlength=self.K)
        valid = counts >= 2
        pcs = self.mu.new_zeros(self.K, self.D, self.D)
        spectrum = torch.zeros(self.K, self.D, dtype=torch.float64, device=self.mu.device)
        if bool(valid.any()):
            # Compact only the PCA input; public component IDs never change.
            ids = valid.nonzero(as_tuple=True)[0]
            remap = torch.full((self.K,), -1, dtype=torch.long, device=self.mu.device)
            remap[ids] = torch.arange(len(ids), device=self.mu.device)
            keep = valid[labels]
            directions, values = compute_cluster_pca_directions(
                points[keep], remap[labels[keep]], self.mu[ids],
                chunk_elems=chunk_elems, eig_batch_size=eig_batch_size,
                return_eigenvalues=True,
            )
            pcs[ids] = directions.to(dtype=self.mu.dtype)
            spectrum[ids] = values
        # The last passing consecutive gap selects the rank; no gap selects one.
        leading = spectrum[:, :1].clamp_min(torch.finfo(spectrum.dtype).tiny)
        gaps = (spectrum[:, :-1] - spectrum[:, 1:]) / leading
        candidates = torch.arange(1, self.D, device=self.mu.device)
        ranks = torch.ones(self.K, dtype=torch.long, device=self.mu.device)
        if candidates.numel():
            ranks = torch.where(gaps > threshold, candidates, 0).amax(dim=1).clamp_min(1)
        ranks = torch.minimum(ranks, (counts - 1).clamp_min(0))
        ranks = torch.where(valid, ranks, 0)
        # Keep one zero column when all clusters are invalid, distinguishing this
        # computed PCA state from a centroid-only model (q=0).
        q = max(1, int(ranks.max()))
        active = torch.arange(q, device=self.mu.device) < ranks[:, None]
        # Clone the slice so no full-basis backing storage survives.
        self._pcs = pcs[:, :, :q].clone()
        self._pcs.mul_(active[:, None, :])
        self._eigenvalues = spectrum
        self._cluster_counts = counts
        self._pca_valid = valid
        self._component_ranks = ranks
        self.surgery_threshold = threshold
        return self

    @staticmethod
    def _validate_threshold(threshold: float) -> float:
        if threshold is None:
            raise ValueError("Cattell threshold is required")
        threshold = float(threshold)
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("threshold must be finite and in [0, 1]")
        return threshold



def save_kmeans(model: KMeans, path: str | Path, *, extra: dict[str, Any] | None = None) -> None:
    """Atomically save model state and provenance; assignments are separate."""
    model._require_fitted()
    meta = {
        "model_type": "kmeans", "version": KMEANS_MODEL_VERSION,
        "K": model.K, "D": model.D, "q": model.q, "q_init": model.q_init,
        "fit_settings": {
            name: getattr(model, name)
            for name in ("max_iter", "restarts", "tol", "seed", "block_x", "block_c")
        },
        "fit_metadata": model.fit_metadata,
        "init_pca_neighbors": model.init_pca_neighbors,
        "init_pca_n_samples": model.init_pca_n_samples,
        "surgery_threshold": model.surgery_threshold,
        "extra": model.checkpoint_extra if extra is None else extra,
    }
    payload = {
        "format": KMEANS_MODEL_FORMAT,
        "meta": meta,
        "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
    }
    path = Path(path)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_kmeans(
    path: str | Path,
    *,
    map_location: str | torch.device | None = None,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> KMeans:
    """Load the versioned KMeans model format; reject legacy centroid bundles."""
    payload = torch.load(path, map_location=map_location or "cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != KMEANS_MODEL_FORMAT:
        raise ValueError("expected a dalg_kmeans_v3 model checkpoint; old checkpoints and legacy centroid artifacts are incompatible; regenerate the model")
    meta = payload["meta"]
    if meta.get("model_type") != "kmeans" or meta.get("version") != KMEANS_MODEL_VERSION:
        raise ValueError("unsupported KMeans checkpoint model type or version")
    state = payload["state_dict"]
    model = KMeans.from_centroids(state["mu"], **meta["fit_settings"])
    K, D, q = int(meta["K"]), int(meta["D"]), int(meta["q"])
    q_init = int(meta["q_init"])
    if (K, D) != tuple(model.mu.shape) or not 0 <= q <= D or not 0 <= q_init <= D:
        raise ValueError("KMeans checkpoint dimensions do not match metadata")
    expected_shapes = {
        "_pcs": (K, D, q),
        "_eigenvalues": (K, D if q else 0),
        "_component_ranks": (K,),
        "_init_pcs": (K, D, q_init),
        "_pca_valid": (K,),
        "_cluster_counts": (K,),
    }
    for name, shape in expected_shapes.items():
        value = state[name]
        if tuple(value.shape) != shape:
            raise ValueError(f"KMeans checkpoint {name} must have shape {shape}")
        setattr(model, name, torch.empty_like(value))
    ranks, valid, counts = (state[key] for key in ("_component_ranks", "_pca_valid", "_cluster_counts"))
    if valid.dtype != torch.bool or counts.dtype != torch.long or bool((counts < 0).any()):
        raise ValueError("invalid PCA validity mask or cluster counts")
    if q:
        if not torch.equal(valid, counts >= 2):
            raise ValueError("PCA validity mask disagrees with cluster counts")
    elif bool(valid.any()) or bool(counts.any()):
        raise ValueError("cluster PCA metadata exists without cluster PCs")
    if ranks.dtype != torch.long or bool(((ranks < 0) | (ranks > q)).any()) or bool((ranks[valid] < 1).any()) or bool(ranks[~valid].any()):
        raise ValueError("invalid component ranks in KMeans checkpoint")
    if bool((ranks[valid] >= counts[valid]).any()):
        raise ValueError("KMeans checkpoint ranks exceed sample-supported capacity")
    if q and q != max(1, int(ranks.max())):
        raise ValueError("KMeans checkpoint PC capacity must match the maximum selected rank")
    pcs, spectrum, init_pcs = (state[key] for key in ("_pcs", "_eigenvalues", "_init_pcs"))
    for name, value in (("PCs", pcs), ("spectrum", spectrum), ("initialization PCs", init_pcs)):
        if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
            raise ValueError(f"KMeans checkpoint {name} must contain finite floating-point values")
    if bool(pcs[~valid].any()) or bool(spectrum[~valid].any()):
        raise ValueError("invalid components must have zero-filled PCA state")
    if q:
        tolerance = max(1e-12, 10 * torch.finfo(spectrum.dtype).eps)
        spectral_tolerance = spectrum[:, :1].abs().clamp_min(1) * tolerance
        if bool((spectrum < -spectral_tolerance).any()) or bool(
            (spectrum[:, 1:] - spectrum[:, :-1] > spectral_tolerance).any()
        ):
            raise ValueError("KMeans checkpoint spectrum must be nonnegative and descending")
    active = torch.arange(q, device=pcs.device) < ranks[:, None]
    if bool(pcs.masked_select(~active[:, None, :]).any()):
        raise ValueError("KMeans checkpoint inactive PCs must be zero")
    for directions, mask in (
        (pcs, active),
        (init_pcs, torch.ones(K, q_init, dtype=torch.bool, device=init_pcs.device)),
    ):
        if directions.shape[-1] == 0:
            continue
        # Float64 validation is independent of global CUDA TF32 settings.
        tolerance = 1e-10 if directions.dtype == torch.float64 else max(1e-5, 8 * torch.finfo(directions.dtype).eps)
        for start in range(0, K, 256):
            batch = directions[start:start + 256].double()
            gram = batch.transpose(1, 2) @ batch
            expected = torch.diag_embed(mask[start:start + 256].double())
            if not torch.allclose(gram, expected, atol=tolerance, rtol=0):
                raise ValueError("KMeans checkpoint active PCs must be orthonormal")
    neighbors, n_samples = meta["init_pca_neighbors"], meta["init_pca_n_samples"]
    if q_init:
        if type(neighbors) is not int or type(n_samples) is not int or not q_init < neighbors <= n_samples:
            raise ValueError("invalid initialization PCA neighborhood metadata")
    elif neighbors is not None or n_samples is not None:
        raise ValueError("initialization PCA metadata exists without initialization PCs")
    threshold = meta["surgery_threshold"] if "surgery_threshold" in meta else meta["cattell_threshold"]
    if (q and threshold is None) or (not q and threshold is not None):
        raise ValueError("invalid cluster PCA Cattell threshold")
    if q:
        model._validate_threshold(threshold)
    model.load_state_dict(state)
    model.fit_metadata = meta["fit_metadata"]
    model.init_pca_neighbors = neighbors
    model.init_pca_n_samples = n_samples
    model.surgery_threshold = threshold
    model.checkpoint_extra = meta.get("extra", {})
    if device is not None:
        model = model.to(device=device)
    if dtype is not None:
        model = model.to(dtype=dtype)
    return model
