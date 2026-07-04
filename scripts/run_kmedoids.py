"""Run dependency-backed CLARA KMedoids on a saved activation array."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


def _nearest_to_centers(
    x: np.ndarray,
    centers: np.ndarray,
    *,
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        print(f"Requested device={dev}, but CUDA is not available; falling back to CPU.")
        dev = torch.device("cpu")
    C = torch.as_tensor(centers, dtype=torch.float32, device=dev)
    c2 = (C * C).sum(dim=1).unsqueeze(0)
    labels = np.empty(x.shape[0], dtype=np.int64)
    min_d2_all = np.empty(x.shape[0], dtype=np.float32)
    for start in tqdm(range(0, x.shape[0], batch_size), desc="assigning to medoids"):
        xb_np = np.array(x[start:start + batch_size], dtype=np.float32, copy=True)
        xb = torch.as_tensor(xb_np, dtype=torch.float32, device=dev)
        d2 = ((xb * xb).sum(dim=1, keepdim=True) + c2 - 2.0 * (xb @ C.T)).clamp_min_(0.0)
        min_d2, idx = d2.min(dim=1)
        end = start + xb.shape[0]
        labels[start:end] = idx.cpu().numpy()
        min_d2_all[start:end] = min_d2.cpu().numpy()
    return labels, min_d2_all, np.bincount(labels, minlength=centers.shape[0]).astype(np.int64)


def _pairwise_squared_euclidean(x: np.ndarray, *, device: str) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        dev = torch.device("cpu")
    xt = torch.as_tensor(x, dtype=torch.float32, device=dev)
    norms = (xt * xt).sum(dim=1)
    d2 = (norms[:, None] + norms[None, :] - 2.0 * (xt @ xt.T)).clamp_min_(0.0)
    return d2.cpu().numpy()


def _kmedoids_alternate_on_sample(
    x_sample: np.ndarray,
    *,
    K: int,
    max_iter: int,
    rng: np.random.Generator,
    device: str,
) -> np.ndarray:
    """Small alternate-update k-medoids on a sampled distance matrix."""
    n = int(x_sample.shape[0])
    if K >= n:
        raise ValueError(f"K={K} must be smaller than sample size {n}")
    d2 = _pairwise_squared_euclidean(x_sample, device=device)

    medoids = rng.choice(n, size=K, replace=False)
    for _ in range(max_iter):
        labels = np.argmin(d2[:, medoids], axis=1)
        next_medoids = medoids.copy()
        for k in range(K):
            members = np.flatnonzero(labels == k)
            if members.size == 0:
                distances_to_nearest = d2[:, medoids].min(axis=1)
                next_medoids[k] = int(np.argmax(distances_to_nearest))
                continue
            intra = d2[np.ix_(members, members)]
            next_medoids[k] = int(members[np.argmin(intra.sum(axis=1))])
        if np.array_equal(np.sort(next_medoids), np.sort(medoids)):
            medoids = next_medoids
            break
        medoids = next_medoids
    return medoids


def _local_clara(
    x: np.ndarray,
    *,
    K: int,
    n_sampling: int,
    n_sampling_iter: int,
    max_iter: int,
    random_state: int,
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(random_state)
    n = int(x.shape[0])
    sample_size = min(max(n_sampling, K + 1), n)
    best_centers: np.ndarray | None = None
    best_inertia = float("inf")
    for _ in tqdm(range(n_sampling_iter), desc="local CLARA samples"):
        sample_idxs = rng.choice(n, size=sample_size, replace=False)
        sample = np.asarray(x[sample_idxs], dtype=np.float32)
        sample_medoids = _kmedoids_alternate_on_sample(
            sample,
            K=K,
            max_iter=max_iter,
            rng=rng,
            device=device,
        )
        centers = np.asarray(sample[sample_medoids], dtype=np.float32)
        _labels, min_d2, _sizes = _nearest_to_centers(
            x,
            centers,
            batch_size=batch_size,
            device=device,
        )
        inertia = float(min_d2.sum())
        if inertia < best_inertia:
            best_inertia = inertia
            best_centers = centers
    assert best_centers is not None
    return best_centers, best_inertia


def _nearest_row_indices(
    x: np.ndarray,
    centers: np.ndarray,
    *,
    batch_size: int,
    device: str,
) -> np.ndarray:
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        dev = torch.device("cpu")
    C = torch.as_tensor(centers, dtype=torch.float32, device=dev)
    best_d2 = torch.full((C.shape[0],), float("inf"), dtype=torch.float32, device=dev)
    best_idx = torch.full((C.shape[0],), -1, dtype=torch.long, device=dev)
    for start in tqdm(range(0, x.shape[0], batch_size), desc="recovering medoid rows"):
        xb_np = np.array(x[start:start + batch_size], dtype=np.float32, copy=True)
        xb = torch.as_tensor(xb_np, dtype=torch.float32, device=dev)
        d2 = torch.cdist(C, xb).pow_(2)
        vals, idxs = d2.min(dim=1)
        better = vals < best_d2
        best_d2[better] = vals[better]
        best_idx[better] = idxs[better] + start
    return best_idx.cpu().numpy().astype(np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CLARA KMedoids over activations.npy")
    parser.add_argument("--activations-path", type=Path, required=True)
    parser.add_argument("--K", type=int, default=12)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--metric", type=str, default="euclidean")
    parser.add_argument("--n-sampling", type=int, default=5000)
    parser.add_argument("--n-sampling-iter", type=int, default=10)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--device", type=str, default="cpu", help="Device for post-fit assignment/index recovery")
    parser.add_argument(
        "--backend",
        choices=["auto", "sklearn-extra", "local"],
        default="auto",
        help="Prefer sklearn-extra CLARA, or force the dependency-free local fallback.",
    )
    args = parser.parse_args()

    x = np.load(args.activations_path, mmap_mode="r")
    if x.ndim != 2:
        raise ValueError(f"activations must have shape (N, D), got {x.shape}")
    if args.K <= 0 or args.K >= x.shape[0]:
        raise ValueError(f"K must be in [1, N), got K={args.K}, N={x.shape[0]}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.time()
    backend = args.backend
    sklearn_error = None
    if backend in {"auto", "sklearn-extra"}:
        try:
            from sklearn_extra.cluster import CLARA

            clara = CLARA(
                n_clusters=args.K,
                metric=args.metric,
                init="build",
                max_iter=args.max_iter,
                n_sampling=args.n_sampling,
                n_sampling_iter=args.n_sampling_iter,
                random_state=args.random_state,
            )
            clara.fit(x)
            centers = np.asarray(clara.cluster_centers_, dtype=np.float32)
            backend = "sklearn-extra"
        except Exception as exc:
            sklearn_error = repr(exc)
            if args.backend == "sklearn-extra":
                raise
            print(f"sklearn-extra CLARA unavailable, falling back to local CLARA: {exc}")
            backend = "local"
    if backend == "local":
        if args.metric != "euclidean":
            raise ValueError("local fallback currently supports only metric='euclidean'")
        centers, _local_inertia = _local_clara(
            x,
            K=args.K,
            n_sampling=args.n_sampling,
            n_sampling_iter=args.n_sampling_iter,
            max_iter=args.max_iter,
            random_state=args.random_state,
            batch_size=args.batch_size,
            device=args.device,
        )
    fit_seconds = time.time() - start_time

    medoid_indices = _nearest_row_indices(x, centers, batch_size=args.batch_size, device=args.device)
    medoids = np.asarray(x[medoid_indices], dtype=np.float32)
    labels, min_d2, cluster_sizes = _nearest_to_centers(
        x,
        medoids,
        batch_size=args.batch_size,
        device=args.device,
    )

    np.save(args.out_dir / "medoids.npy", medoids)
    np.save(args.out_dir / "medoid_indices.npy", medoid_indices)
    np.save(args.out_dir / "labels.npy", labels)
    np.save(args.out_dir / "min_squared_distances.npy", min_d2)
    config = {
        "activations_path": str(args.activations_path),
        "shape": list(x.shape),
        "K": int(args.K),
        "metric": args.metric,
        "n_sampling": int(args.n_sampling),
        "n_sampling_iter": int(args.n_sampling_iter),
        "max_iter": int(args.max_iter),
        "random_state": int(args.random_state),
        "backend": backend,
        "sklearn_extra_error": sklearn_error,
        "fit_seconds": fit_seconds,
        "inertia": float(min_d2.sum()),
        "cluster_sizes": cluster_sizes.tolist(),
        "medoids_path": str(args.out_dir / "medoids.npy"),
        "medoid_indices_path": str(args.out_dir / "medoid_indices.npy"),
        "labels_path": str(args.out_dir / "labels.npy"),
    }
    (args.out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"Saved medoids to {args.out_dir / 'medoids.npy'} with shape {medoids.shape}")
    print(f"Saved labels to {args.out_dir / 'labels.npy'} with shape {labels.shape}")


if __name__ == "__main__":
    main()
