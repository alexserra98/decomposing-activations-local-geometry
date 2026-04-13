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
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import argparse
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from modeling.mfa import load_mfa


# ── Core computation ────────────────────────────────────────────────────


def intrinsic_dim_pca(X_cluster: torch.Tensor, threshold: float = 0.90) -> int:
    """
    Number of PCA components needed to explain `threshold` of total variance.
    Uses economy SVD on mean-centered data.
    """
    if X_cluster.shape[0] < 2:
        return 1
    X_c = X_cluster.float() - X_cluster.float().mean(dim=0)
    _, S, _ = torch.linalg.svd(X_c, full_matrices=False)  # S: (min(N,D),)
    var = S ** 2
    cumvar = var.cumsum(0) / var.sum()
    above = (cumvar >= threshold).nonzero(as_tuple=True)[0]
    return int(above[0].item()) + 1 if len(above) > 0 else len(S)


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
):
    """
    Compute intrinsic dimensionality per MFA cluster.

    Args:
        model_path: Path to saved MFA model.
        act_path: Path to activations.pt (N, D).
        tok_path: Path to tokens.pt (N,).
        device: Device for responsibility computation.
        batch_size: Batch size for responsibility computation.
        variance_threshold: Fraction of variance to explain.
        min_population: Skip clusters smaller than this.
        max_samples: Cap samples per cluster for SVD.

    Returns:
        dict with keys: intrinsic_dims, cluster_sizes, assignments,
                        variance_threshold, K, rank, D.
    """
    X = torch.load(act_path, weights_only=True)
    tok = torch.load(tok_path, weights_only=True)
    print(f"Activations: {X.shape}  dtype={X.dtype}")

    model = load_mfa(model_path, map_location="cpu").to(device)
    model.eval()
    K, D, q = model.K, model.D, model.q
    print(f"MFA: K={K} components  rank={q}  D={D}")

    # Hard assignments via responsibilities
    loader = DataLoader(TensorDataset(X, tok), batch_size=batch_size, shuffle=False)
    all_assignments = []
    with torch.no_grad():
        for x_batch, _ in tqdm(loader, desc="Computing responsibilities"):
            r = model.responsibilities(x_batch.to(device))
            all_assignments.append(r.argmax(dim=1).cpu())

    assignments = torch.cat(all_assignments)
    sizes = torch.bincount(assignments, minlength=K)

    print(f"\nCluster sizes — min={sizes.min().item()}  "
          f"max={sizes.max().item()}  "
          f"mean={sizes.float().mean():.1f}  "
          f"median={sizes.float().median():.1f}")
    print(f"Empty clusters: {(sizes == 0).sum().item()}")

    # PCA intrinsic dimension per cluster
    dims = torch.zeros(K, dtype=torch.long)
    num_skipped = 0

    for k in tqdm(range(K), desc="PCA per cluster"):
        idx = (assignments == k).nonzero(as_tuple=True)[0]
        n = idx.numel()
        if n < min_population:
            dims[k] = 0
            num_skipped += 1
            continue
        if n > max_samples:
            idx = idx[torch.randperm(n)[:max_samples]]
        dims[k] = intrinsic_dim_pca(X[idx], threshold=variance_threshold)

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
