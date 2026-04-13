# %% [markdown]
# ## Analytical Overlap Between MFA Components
#
# Each MFA component $k$ defines a Gaussian $\mathcal{N}(\mu_k, C_k)$ where
# $C_k = W_k W_k^\top + \Psi_k$ (low-rank plus diagonal). We compute three
# pairwise metrics between all $\binom{K}{2}$ pairs of components.
#
# All D×D operations are avoided via the **Woodbury identity**, and the
# pairwise loop is **vectorized** over chunks of P pairs for GPU efficiency.
#
# ### Metrics
#
# | Output | What it captures | Range | Symmetric |
# |---|---|---|---|
# | $\mathrm{KL}_{\mathrm{sym}}$ | Full distributional divergence | $[0,\infty)$ | Yes |
# | $D_B$ | Bhattacharyya distance | $[0,\infty)$ | Yes |
# | $D_B^{\mathrm{mean}}$ | Centroid-separation contribution | $[0,\infty)$ | Yes |
# | $D_B^{\mathrm{cov}}$ | Covariance-mismatch contribution | $[0,\infty)$ | Yes |
# | $\mathrm{BC}$ | Bhattacharyya coefficient (affinity) | $[0,1]$ | Yes |

# %%
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import argparse
import torch
from tqdm import tqdm

from modeling.mfa import load_mfa


# ── Vectorized overlap computation ──────────────────────────────────────


def _batched_kl_one_way(
    mu_i, mu_j, W_i, W_j, psi_i, psi_j, B_i, B_j,
    L_i, L_j, Minv_i, Minv_j, logdet_i, logdet_j, D,
):
    """
    Batched KL(p_i || p_j) for P pairs simultaneously.

    All inputs have a leading P dimension (batch of pairs).
    Returns: (P,) tensor of KL divergences.
    """
    delta = mu_j - mu_i  # (P, D)

    # --- tr(C_j^{-1} C_i) = tr(C_j^{-1} Psi_i) + tr(C_j^{-1} W_i W_i^T) ---

    # Diagonal part: tr(C_j^{-1} Psi_i) = sum(psi_i/psi_j) - tr(Minv_j @ P_ji)
    #   P_ji = B_j^T diag(psi_i) B_j  →  (P, q, q)
    BjT_psi_i = B_j * psi_i[:, :, None]                        # (P, D, q)
    P_ji = torch.einsum("pdq,pdr->pqr", BjT_psi_i, B_j)       # (P, q, q)
    tr_diag = (psi_i / psi_j).sum(-1) - (Minv_j * P_ji).sum((-1, -2))  # (P,)

    # Low-rank part: tr(C_j^{-1} W_i W_i^T) = ||W_i||^2_{Psi_j^{-1}} - ||L_j^{-1}(B_j^T W_i)||^2_F
    tr_lr = (W_i ** 2 / psi_j[:, :, None]).sum((-1, -2))       # (P,)
    R_ji = torch.einsum("pdq,pdr->pqr", B_j, W_i)             # (P, q, q)
    X_ji = torch.linalg.solve_triangular(L_j, R_ji, upper=False)  # (P, q, q)
    tr_lr = tr_lr - (X_ji ** 2).sum((-1, -2))                  # (P,)

    trace_term = tr_diag + tr_lr                                # (P,)

    # --- Mahalanobis: delta^T C_j^{-1} delta ---
    maha = (delta ** 2 / psi_j).sum(-1)                         # (P,)
    v_j = torch.einsum("pdq,pd->pq", B_j, delta)               # (P, q)
    y_j = torch.linalg.solve_triangular(
        L_j, v_j.unsqueeze(-1), upper=False
    ).squeeze(-1)                                               # (P, q)
    maha = maha - (y_j ** 2).sum(-1)                            # (P,)

    return 0.5 * (trace_term + maha - D + logdet_j - logdet_i)  # (P,)


def _batched_bhattacharyya(
    mu_i, mu_j, W_i, W_j, psi_i, psi_j, logdet_i, logdet_j, q, eps=1e-6,
):
    """
    Batched Bhattacharyya distance for P pairs.

    Returns: (db_mean, db_cov, db, bc) each of shape (P,).
    """
    P, D = mu_i.shape
    delta = mu_j - mu_i                                         # (P, D)

    # Bar_C = (C_i + C_j)/2 = W_bar W_bar^T + Psi_bar
    # where W_bar = [W_i | W_j] / sqrt(2)  →  (P, D, 2q)
    psi_bar = 0.5 * (psi_i + psi_j)                            # (P, D)
    W_bar = torch.cat([W_i, W_j], dim=-1) * (0.5 ** 0.5)       # (P, D, 2q)

    # M_bar = I_{2q} + A_bar^T A_bar  where A_bar = Psi_bar^{-1/2} W_bar
    A_bar = W_bar * psi_bar[:, :, None].rsqrt()                 # (P, D, 2q)
    I2q = torch.eye(2 * q, device=mu_i.device, dtype=mu_i.dtype)
    M_bar = I2q[None] + torch.einsum("pdi,pdj->pij", A_bar, A_bar)  # (P, 2q, 2q)

    # Add small jitter for numerical stability with large K
    M_bar = M_bar + eps * I2q[None]

    L_bar = torch.linalg.cholesky(M_bar)                        # (P, 2q, 2q)

    # log|Bar_C| = log|Psi_bar| + log|M_bar|
    logdet_Cbar = (
        psi_bar.log().sum(-1)
        + 2.0 * L_bar.diagonal(dim1=-2, dim2=-1).log().sum(-1)
    )                                                           # (P,)

    db_cov = 0.5 * logdet_Cbar - 0.25 * (logdet_i + logdet_j)  # (P,)

    # Mahalanobis under Bar_C
    B_bar = W_bar / psi_bar[:, :, None]                         # (P, D, 2q)
    maha = (delta ** 2 / psi_bar).sum(-1)                       # (P,)
    v_bar = torch.einsum("pdq,pd->pq", B_bar, delta)           # (P, 2q)
    y_bar = torch.linalg.solve_triangular(
        L_bar, v_bar.unsqueeze(-1), upper=False
    ).squeeze(-1)                                               # (P, 2q)
    maha = maha - (y_bar ** 2).sum(-1)                          # (P,)

    db_mean = maha / 8.0                                        # (P,)
    db = db_mean + db_cov                                       # (P,)
    bc = torch.exp(-db)                                         # (P,)

    return db_mean, db_cov, db, bc


def compute_overlap(model_path, *, device="cpu", batch_pairs=4096):
    """
    Compute pairwise overlap metrics between all MFA components.

    Args:
        model_path: Path to saved MFA model (.pt file).
        device: Device for computation.
        batch_pairs: Number of pairs to process per batch (tune for GPU memory).

    Returns:
        dict with keys: kl_sym, db, db_mean, db_cov, bc — each (K, K) tensors.
    """
    model = load_mfa(model_path, map_location="cpu")
    model.eval()
    K, D, q = model.K, model.D, model.q
    print(f"MFA: K={K}  D={D}  q={q}")
    n_pairs = K * (K - 1) // 2
    print(f"Computing {n_pairs:,} pairwise metrics in chunks of {batch_pairs}")

    # Extract parameters once (on CPU, then move chunks to device)
    with torch.no_grad():
        mu  = model.mu.float()
        W   = model.W.float()
        psi = model._psi().float()

    # Precompute per-component Woodbury factors
    with torch.no_grad():
        Iq = torch.eye(q)
        A        = W * psi[:, :, None].rsqrt()                         # (K, D, q)
        B        = W / psi[:, :, None]                                  # (K, D, q)
        M        = Iq[None] + torch.einsum("kdi,kdj->kij", A, A)       # (K, q, q)
        L        = torch.linalg.cholesky(M)                             # (K, q, q)
        Minv     = torch.cholesky_solve(Iq.expand(K, q, q).clone(), L) # (K, q, q)
        logdet_C = (psi.log().sum(-1)
                    + 2.0 * L.diagonal(dim1=-2, dim2=-1).log().sum(-1)) # (K,)

    # All pair indices
    pairs = torch.triu_indices(K, K, offset=1)  # (2, n_pairs)

    # Flat result tensors
    kl_flat      = torch.zeros(n_pairs)
    db_mean_flat = torch.zeros(n_pairs)
    db_cov_flat  = torch.zeros(n_pairs)
    db_flat      = torch.zeros(n_pairs)
    bc_flat      = torch.zeros(n_pairs)

    with torch.no_grad():
        for start in tqdm(range(0, n_pairs, batch_pairs), desc="pairwise overlap"):
            end = min(start + batch_pairs, n_pairs)
            idx_i = pairs[0, start:end]
            idx_j = pairs[1, start:end]

            # Gather per-pair tensors and move to device
            mu_i = mu[idx_i].to(device)
            mu_j = mu[idx_j].to(device)
            W_i  = W[idx_i].to(device)
            W_j  = W[idx_j].to(device)
            psi_i = psi[idx_i].to(device)
            psi_j = psi[idx_j].to(device)
            B_i   = B[idx_i].to(device)
            B_j   = B[idx_j].to(device)
            L_i   = L[idx_i].to(device)
            L_j   = L[idx_j].to(device)
            Minv_i = Minv[idx_i].to(device)
            Minv_j = Minv[idx_j].to(device)
            logdet_i = logdet_C[idx_i].to(device)
            logdet_j = logdet_C[idx_j].to(device)

            # Symmetrised KL = (KL(i||j) + KL(j||i)) / 2
            kl_ij = _batched_kl_one_way(
                mu_i, mu_j, W_i, W_j, psi_i, psi_j, B_i, B_j,
                L_i, L_j, Minv_i, Minv_j, logdet_i, logdet_j, D,
            )
            kl_ji = _batched_kl_one_way(
                mu_j, mu_i, W_j, W_i, psi_j, psi_i, B_j, B_i,
                L_j, L_i, Minv_j, Minv_i, logdet_j, logdet_i, D,
            )
            kl_sym_chunk = 0.5 * (kl_ij + kl_ji)

            # Bhattacharyya
            db_mean_chunk, db_cov_chunk, db_chunk, bc_chunk = _batched_bhattacharyya(
                mu_i, mu_j, W_i, W_j, psi_i, psi_j, logdet_i, logdet_j, q,
            )

            # Store (move back to CPU)
            kl_flat[start:end]      = kl_sym_chunk.cpu()
            db_mean_flat[start:end] = db_mean_chunk.cpu()
            db_cov_flat[start:end]  = db_cov_chunk.cpu()
            db_flat[start:end]      = db_chunk.cpu()
            bc_flat[start:end]      = bc_chunk.cpu()

    # Unfold flat arrays into symmetric (K, K) matrices
    kl_sym  = torch.zeros(K, K)
    db_mean = torch.zeros(K, K)
    db_cov  = torch.zeros(K, K)
    db      = torch.zeros(K, K)
    bc      = torch.zeros(K, K)

    idx_i, idx_j = pairs[0], pairs[1]
    kl_sym[idx_i, idx_j]  = kl_flat;      kl_sym[idx_j, idx_i]  = kl_flat
    db_mean[idx_i, idx_j] = db_mean_flat;  db_mean[idx_j, idx_i] = db_mean_flat
    db_cov[idx_i, idx_j]  = db_cov_flat;   db_cov[idx_j, idx_i]  = db_cov_flat
    db[idx_i, idx_j]      = db_flat;        db[idx_j, idx_i]      = db_flat
    bc[idx_i, idx_j]      = bc_flat;        bc[idx_j, idx_i]      = bc_flat

    print(f"KL_sym  — min: {kl_flat[kl_flat>0].min():.3f}  max: {kl_flat.max():.3f}")
    print(f"D_B     — min: {db_flat[db_flat>0].min():.3f}  max: {db_flat.max():.3f}")
    print(f"BC      — min: {bc_flat[bc_flat<1].min():.4f}  max: {bc_flat[bc_flat<1].max():.4f}")

    return {"kl_sym": kl_sym, "db": db, "db_mean": db_mean, "db_cov": db_cov, "bc": bc}


# ── CLI entry point ─────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Pairwise overlap metrics for MFA components")
    parser.add_argument("--model-path", required=True, help="Path to mfa_model.pt")
    parser.add_argument("--save-path", default=None, help="Where to save results (default: next to model)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-pairs", type=int, default=4096,
                        help="Pairs per batch (tune for GPU memory)")
    args = parser.parse_args()

    results = compute_overlap(args.model_path, device=args.device, batch_pairs=args.batch_pairs)

    save_path = args.save_path or os.path.join(os.path.dirname(args.model_path), "overlap.pt")
    torch.save(results, save_path)
    print(f"Saved to {save_path}")


if __name__ == "__main__":
    main()


# %% [markdown]
# ## Visualization (interactive use)
#
# The cells below produce heatmaps and histograms.
# Run them in a notebook or after loading results from disk.

# %%
# To use the visualization cells, load results:
#
#   results = torch.load("overlap.pt")
#   kl_sym, db, db_mean, db_cov, bc = (
#       results["kl_sym"], results["db"], results["db_mean"],
#       results["db_cov"], results["bc"],
#   )
#   K = kl_sym.shape[0]
#
# Then run the cells below.

# %%
# --- Heatmaps (hierarchically ordered) ---
#
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.cluster.hierarchy import linkage, leaves_list
# from scipy.spatial.distance import squareform
#
# dist_condensed = squareform(np.clip(1.0 - bc.numpy(), 0.0, None), checks=False)
# Z     = linkage(dist_condensed, method="average")
# order = leaves_list(Z)
#
# fig, axes = plt.subplots(2, 2, figsize=(18, 5))
# for ax, mat, title in zip(
#     axes.flatten(),
#     [bc[order][:, order], db[order][:, order],
#      db_mean[order][:, order], db_cov[order][:, order]],
#     ["BC (affinity)", r"D_B",
#      r"$D_B^{\rm mean}$ (centroid sep.)", r"$D_B^{\rm cov}$ (shape mismatch)"],
# ):
#     im = ax.imshow(mat.numpy(), aspect="auto")
#     ax.set_title(title)
#     ax.set_xlabel("component j")
#     ax.set_ylabel("component i")
#     plt.colorbar(im, ax=ax)
# plt.suptitle("Pairwise overlap — reordered by hierarchical clustering", y=1.02)
# plt.tight_layout()
# plt.show()
