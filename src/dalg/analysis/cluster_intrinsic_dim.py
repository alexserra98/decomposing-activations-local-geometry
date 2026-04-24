# %% [markdown]
# # Cluster Intrinsic Dimensionality
#
# For each MFA component:
# 1. Assign activations via hard argmax of responsibilities.
# 2. Fit PCA on the cluster's activations (via economy SVD).
# 3. Record the number of dimensions needed to explain 90% of the variance.
# 4. Plot and save the results.

# %%
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import argparse
import concurrent.futures as _futures
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from dalg.models.mfa import load_mfa


# ── Core computation ────────────────────────────────────────────────────


def intrinsic_dim_pca(
    X_cluster: torch.Tensor,
    threshold: float = 0.90,
    device: str | torch.device | None = None,
) -> int:
    """
    Number of PCA components needed to explain `threshold` of total variance.
    Uses `svdvals` on mean-centered data (cheaper than full SVD since we only
    need the singular values). If `device` is set, runs the SVD there.
    """
    if X_cluster.shape[0] < 2:
        return 1
    if device is not None:
        X_cluster = X_cluster.to(device, non_blocking=True)
    X = X_cluster.float()
    X_c = X - X.mean(dim=0)
    S = torch.linalg.svdvals(X_c)
    var = S ** 2
    cumvar = var.cumsum(0) / var.sum()
    above = (cumvar >= threshold).nonzero(as_tuple=True)[0]
    return int(above[0].item()) + 1 if len(above) > 0 else int(S.numel())


def _pca_loop(
    buffers, sizes, *, threshold, min_population, pca_device, pca_workers
):
    """Run intrinsic_dim_pca over all clusters. GPU path is sequential (kernel
    launches are cheap and the GPU is already saturated). CPU path uses a
    thread pool — torch SVD releases the GIL, so threads give real speedup
    without the RAM duplication of processes."""
    K = len(buffers)
    dims = torch.zeros(K, dtype=torch.long)
    num_skipped = 0

    todo = [k for k in range(K)
            if int(sizes[k]) >= min_population and buffers[k]]
    num_skipped = K - len(todo)

    def _one(k):
        X_k = torch.cat(buffers[k], dim=0)
        return k, intrinsic_dim_pca(X_k, threshold=threshold, device=pca_device)

    use_threads = (pca_workers and pca_workers > 1
                   and (pca_device is None or str(pca_device).startswith("cpu")))

    if use_threads:
        # Keep each SVD single-threaded so workers don't fight over cores.
        old_t = torch.get_num_threads()
        torch.set_num_threads(max(1, old_t // pca_workers))
        try:
            with _futures.ThreadPoolExecutor(max_workers=pca_workers) as pool:
                futs = [pool.submit(_one, k) for k in todo]
                for fut in tqdm(_futures.as_completed(futs),
                                total=len(futs),
                                desc=f"PCA per cluster (cpu×{pca_workers})"):
                    k, d = fut.result()
                    dims[k] = d
        finally:
            torch.set_num_threads(old_t)
    else:
        tag = pca_device if pca_device else "cpu"
        for k in tqdm(todo, desc=f"PCA per cluster ({tag})"):
            _, d = _one(k)
            dims[k] = d

    return dims, num_skipped


def compute_intrinsic_dims_from_loader(
    model_path,
    loader,
    *,
    device="cpu",
    variance_threshold=0.90,
    min_population=100,
    max_samples=10_000,
    store_dtype=torch.float16,
    pca_device=None,
    pca_workers=1,
):
    """
    Streaming intrinsic-dim computation. Works with any DataLoader yielding
    either (x, ...) tuples or plain x tensors — including the sharded
    `ShardActivationDataset` used at training time.

    Parallelism:
    - Assignment phase is driven by the caller's DataLoader (raise its
      num_workers to parallelize shard I/O). Bump `loader.num_workers`, not
      any arg here.
    - PCA phase runs on `pca_device` (defaults to `device`). On CUDA it's
      sequential because the GPU is already the parallelism. On CPU, set
      `pca_workers > 1` to use a thread pool (torch SVD releases the GIL).

    Memory footprint: peak ~K * max_samples * D * sizeof(store_dtype). With
    K=1000, max_samples=2000, D=2304, fp16 that's ~9 GB. Drop max_samples if
    you hit RAM limits.
    """
    model = load_mfa(model_path, map_location="cpu").to(device)
    model.eval()
    K, D, q = model.K, model.D, model.q
    print(f"MFA: K={K} components  rank={q}  D={D}")
    if pca_device is None:
        pca_device = device

    sizes = torch.zeros(K, dtype=torch.long)
    saved = torch.zeros(K, dtype=torch.long)
    buffers: list[list[torch.Tensor]] = [[] for _ in range(K)]
    all_assignments: list[torch.Tensor] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Streaming + assigning"):
            x_batch = batch[0] if isinstance(batch, (list, tuple)) else batch
            x_dev = x_batch.to(device, non_blocking=True)
            r = model.responsibilities(x_dev)
            assign = r.argmax(dim=1).cpu()
            sizes += torch.bincount(assign, minlength=K)
            all_assignments.append(assign)

            x_cpu = x_batch.to(store_dtype).cpu() if x_batch.device.type != "cpu" \
                    else x_batch.to(store_dtype)
            for k in assign.unique().tolist():
                already = int(saved[k])
                if already >= max_samples:
                    continue
                chunk = x_cpu[assign == k][: max_samples - already]
                if chunk.shape[0] > 0:
                    buffers[k].append(chunk)
                    saved[k] += chunk.shape[0]

    assignments = torch.cat(all_assignments)

    print(f"\nCluster sizes — min={sizes.min().item()}  "
          f"max={sizes.max().item()}  "
          f"mean={sizes.float().mean():.1f}  "
          f"median={sizes.float().median():.1f}")
    print(f"Empty clusters: {(sizes == 0).sum().item()}")

    dims, num_skipped = _pca_loop(
        buffers, sizes,
        threshold=variance_threshold,
        min_population=min_population,
        pca_device=pca_device,
        pca_workers=pca_workers,
    )

    valid = dims > 0
    print(f"\nIntrinsic dims at {variance_threshold*100:.0f}% variance threshold:")
    print(f"  mean   = {dims[valid].float().mean():.2f}")
    print(f"  median = {dims[valid].float().median():.2f}")
    print(f"  min    = {dims[valid].min().item()}")
    print(f"  max    = {dims[valid].max().item()}")
    print(f"  MFA rank (q) = {q}  (reference)")
    print(f"Skipped {num_skipped} clusters with population < {min_population}")

    return {
        "intrinsic_dims": dims,
        "cluster_sizes": sizes,
        "assignments": assignments,
        "variance_threshold": variance_threshold,
        "K": K,
        "rank": q,
        "D": D,
    }


def compute_intrinsic_dims(
    model_path,
    act_path,
    tok_path,
    *,
    device="cpu",
    batch_size=512,
    variance_threshold=0.90,
    min_population=100,
    max_samples=10_000,
    pca_device=None,
    pca_workers=1,
):
    """Monolithic-layout wrapper: loads activations.pt/tokens.pt and streams
    them through `compute_intrinsic_dims_from_loader`."""
    X = torch.load(act_path, weights_only=True)
    tok = torch.load(tok_path, weights_only=True)
    print(f"Activations: {X.shape}  dtype={X.dtype}")

    loader = DataLoader(TensorDataset(X, tok), batch_size=batch_size, shuffle=False)
    return compute_intrinsic_dims_from_loader(
        model_path, loader,
        device=device,
        variance_threshold=variance_threshold,
        min_population=min_population,
        max_samples=max_samples,
        pca_device=pca_device,
        pca_workers=pca_workers,
    )


# ── CLI entry point ─────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Intrinsic dimensionality per MFA cluster")
    parser.add_argument("--model-path", required=True, help="Path to mfa_model.pt")
    parser.add_argument("--act-path", required=True, help="Path to activations.pt")
    parser.add_argument("--tok-path", required=True, help="Path to tokens.pt")
    parser.add_argument("--save-path", default=None, help="Where to save results")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--variance-threshold", type=float, default=0.90)
    parser.add_argument("--min-population", type=int, default=100)
    parser.add_argument("--max-samples", type=int, default=10_000)
    args = parser.parse_args()

    results = compute_intrinsic_dims(
        args.model_path, args.act_path, args.tok_path,
        device=args.device,
        batch_size=args.batch_size,
        variance_threshold=args.variance_threshold,
        min_population=args.min_population,
        max_samples=args.max_samples,
    )

    save_path = args.save_path or os.path.join(
        os.path.dirname(args.model_path), "intrinsic_dims.pt"
    )
    torch.save(results, save_path)
    print(f"Results saved to {save_path}")


if __name__ == "__main__":
    main()
