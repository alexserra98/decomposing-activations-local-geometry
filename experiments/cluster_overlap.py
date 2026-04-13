# %% [markdown]
# ## Analytical Overlap Between MFA Components
#
# Each MFA component $k$ defines a Gaussian $\mathcal{N}(\mu_k, C_k)$ where
# $C_k = W_k W_k^\top + \Psi_k$ (low-rank plus diagonal). We compute three
# pairwise metrics between all $\binom{K}{2}$ pairs of components, then
# explore the resulting $K \times K$ matrices to decide what further analyses
# or visualisations are most informative.
#
# ---
#
# ### Metric 1 — Symmetrised KL Divergence (Jeffreys Divergence)
#
# For two Gaussians the KL divergence has the closed form
# $$\mathrm{KL}(p_i \| p_j) = \tfrac{1}{2}\bigl[\mathrm{tr}(C_j^{-1} C_i)
#   + (\mu_j - \mu_i)^\top C_j^{-1}(\mu_j - \mu_i) - D
#   + \ln\tfrac{|C_j|}{|C_i|}\bigr].$$
# We report the symmetrised version
# $\mathrm{KL}_{\mathrm{sym}} = \tfrac{1}{2}[\mathrm{KL}(i\|j)+\mathrm{KL}(j\|i)]$.
#
# Every expensive $D\times D$ operation is avoided via the **Woodbury identity**:
#
# - $C_j^{-1} = \Psi_j^{-1} - \Psi_j^{-1} W_j\, M_j^{-1}\, W_j^\top \Psi_j^{-1}$
#   where $M_j = I_q + W_j^\top \Psi_j^{-1} W_j$ is $q\times q$.
# - $\mathrm{tr}(C_j^{-1} C_i)$ decomposes into traces involving only
#   $D$-vectors and $q\times q$ matrices (see implementation).
# - $\ln|C_k| = \ln|\Psi_k| + \ln|M_k|$, both cheap.
#
# This captures *full distributional* divergence — means, subspace orientation,
# loading scales, and noise.
#
# **Complexity:** $O(Dq^2)$ per pair.
#
# ### Metric 2 — Bhattacharyya Distance (with Decomposition)
#
# $$D_B(i,j) = \underbrace{\tfrac{1}{8}(\mu_i-\mu_j)^\top
#   \bar C^{-1}(\mu_i-\mu_j)}_{D_B^{\mathrm{mean}}}
#   + \underbrace{\tfrac{1}{2}\ln\frac{|\bar C|}
#   {\sqrt{|C_i||C_j|}}}_{D_B^{\mathrm{cov}}},$$
# where $\bar C = (C_i + C_j)/2$.
#
# We store the total $D_B$ and its two terms separately:
# - $D_B^{\mathrm{mean}}$: **centroid separation** — Mahalanobis distance
#   between means under the pooled covariance. Zero iff $\mu_i = \mu_j$.
# - $D_B^{\mathrm{cov}}$: **shape mismatch** — how different the covariance
#   ellipsoids are. Zero iff $C_i = C_j$.
#
# From $D_B$ we also derive the **Bhattacharyya coefficient**
# $\mathrm{BC}(i,j) = \exp(-D_B) \in [0,1]$ (1 = identical, 0 = separated).
# Unlike KL, this is bounded and symmetric — natural for heatmaps and
# thresholding.
#
# The averaged covariance $\bar C = \tfrac{1}{2}(W_i W_i^\top + W_j W_j^\top) + \bar\Psi$ is rank-$2q$ plus diagonal, so Woodbury applies with a
# $2q \times 2q$ inner matrix.
#
# **Complexity:** $O(Dq^2)$ per pair.
#
# ### Summary
#
# | Output | What it captures | Range | Symmetric |
# |---|---|---|---|
# | $\mathrm{KL}_{\mathrm{sym}}$ | Full distributional divergence | $[0,\infty)$ | Yes |
# | $D_B$ | Full distributional distance (bounded growth) | $[0,\infty)$ | Yes |
# | $D_B^{\mathrm{mean}}$ | Centroid-separation contribution | $[0,\infty)$ | Yes |
# | $D_B^{\mathrm{cov}}$ | Covariance-mismatch contribution | $[0,\infty)$ | Yes |
# | $\mathrm{BC}$ | Bounded affinity | $[0,1]$ | Yes |
#
# ### Workflow
#
# 1. Compute the five $K \times K$ matrices above for all pairs.
# 2. Inspect the results and brainstorm which further statistics or
#    visualisations (heatmaps, histograms of pairwise distances,
#    clustering of the overlap graph, etc.) are most informative.

# %%
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from tqdm import tqdm

from modeling.mfa import load_mfa

# %%
# --- Config ---
DATA_PATH = "/Users/alessandroserra/Desktop/phd/decomposing-activations-local-geometry/experiments/data"
DEVICE       = "cpu"
MODEL_PATH   = f"{DATA_PATH}/mfa_model.pt"
RESULTS_PATH = "cluster_overlap.pt"

# %%
# --- Load model ---
model = load_mfa(MODEL_PATH, map_location="cpu")
model.eval()
K, D, q = model.K, model.D, model.q
print(f"MFA: K={K}  D={D}  q={q}")

with torch.no_grad():
    mu  = model.mu.float()                              # (K, D)
    W   = model.W.float()                              # (K, D, q)
    psi = model._psi().float()                         # (K, D)
    pi  = torch.softmax(model.pi_logits, dim=0).float()  # (K,)

# %%
# --- Precompute per-component Woodbury factors ---
# For each component k:
#   A_k = Psi_k^{-1/2} W_k          (D, q)   scaled loadings
#   B_k = Psi_k^{-1}   W_k          (D, q)   = A_k / sqrt(psi_k)
#   M_k = I_q + A_k^T A_k           (q, q)   inner matrix
#   L_k = chol(M_k)                  (q, q)
#   Minv_k = M_k^{-1}               (q, q)   via Cholesky solve
#   logdet_Ck = log|C_k|            scalar   = log|Psi_k| + log|M_k|
with torch.no_grad():
    Iq = torch.eye(q)

    A        = W * psi[:, :, None].rsqrt()                              # (K, D, q)
    B        = W / psi[:, :, None]                                      # (K, D, q)
    M        = Iq[None] + torch.einsum("kdi,kdj->kij", A, A)           # (K, q, q)
    L        = torch.linalg.cholesky(M)                                 # (K, q, q)
    Minv     = torch.cholesky_solve(Iq.expand(K, q, q).clone(), L)     # (K, q, q)
    logdet_C = (psi.log().sum(-1)
                + 2.0 * L.diagonal(dim1=-2, dim2=-1).log().sum(-1))    # (K,)

# %%
# --- Pairwise metrics (all K*(K-1)/2 pairs) ---
#
# For each pair (i, j) we compute:
#   kl_sym  : symmetrised KL  = (KL(i||j) + KL(j||i)) / 2
#   db_mean : Bhattacharyya mean term  = 1/8 * delta^T Bar_C^{-1} delta
#   db_cov  : Bhattacharyya cov term   = 1/2 * (log|Bar_C| - 1/2*(log|Ci|+log|Cj|))
#   db      : db_mean + db_cov
#   bc      : exp(-db)   in [0, 1]
#
# All D×D operations are avoided via the Woodbury identity (see markdown).

def _kl_one_way(i, j, delta):
    """KL(p_i || p_j): divergence of component i from component j."""
    # tr(C_j^{-1} C_i) = tr(C_j^{-1} Psi_i) + tr(C_j^{-1} W_i W_i^T)
    # -- diagonal part: tr(C_j^{-1} Psi_i) = sum(psi_i/psi_j) - tr(Minv_j P_ji)
    #    where P_ji = B_j^T diag(psi_i) B_j
    P_ji    = (B[j] * psi[i][:, None]).T @ B[j]          # (q, q)
    tr_diag = (psi[i] / psi[j]).sum() - (Minv[j] * P_ji).sum()

    # -- low-rank part: tr(C_j^{-1} W_i W_i^T) = ||A_ji||_F^2 - ||X_ji||_F^2
    #    A_ji = Psi_j^{-1/2} W_i;  X_ji = L_j^{-1} (B_j^T W_i)
    tr_lr = (W[i] ** 2 / psi[j][:, None]).sum()
    R_ji  = B[j].T @ W[i]                                # (q, q)
    X_ji  = torch.linalg.solve_triangular(L[j], R_ji, upper=False)
    tr_lr -= (X_ji ** 2).sum()

    trace_term = tr_diag + tr_lr

    # Mahalanobis: delta^T C_j^{-1} delta = (delta^2/psi_j).sum() - ||y_j||^2
    #   y_j = L_j^{-1} (B_j^T delta)
    maha  = (delta ** 2 / psi[j]).sum()
    v_j   = B[j].T @ delta                               # (q,)
    y_j   = torch.linalg.solve_triangular(L[j], v_j[:, None], upper=False).squeeze(-1)
    maha -= (y_j ** 2).sum()

    return 0.5 * (trace_term + maha - D + logdet_C[j] - logdet_C[i])


kl_sym  = torch.zeros(K, K)
db_mean = torch.zeros(K, K)
db_cov  = torch.zeros(K, K)
db      = torch.zeros(K, K)
bc      = torch.zeros(K, K)

I2q = torch.eye(2 * q)

with torch.no_grad():
    for i in tqdm(range(K), desc="pairwise metrics"):
        for j in range(i + 1, K):
            delta = mu[j] - mu[i]                                       # (D,)

            # --- Symmetrised KL ---
            kl_val = 0.5 * (_kl_one_way(i, j, delta)
                             + _kl_one_way(j, i, delta))

            # --- Bhattacharyya ---
            # Bar_C = (C_i + C_j)/2 = W_bar W_bar^T + Psi_bar
            # where W_bar = [W_i | W_j] / sqrt(2)  (D, 2q)
            psi_bar = 0.5 * (psi[i] + psi[j])                          # (D,)
            W_bar   = torch.cat([W[i], W[j]], dim=-1) / (2.0 ** 0.5)   # (D, 2q)
            A_bar   = W_bar * psi_bar[:, None].rsqrt()                  # (D, 2q)
            M_bar   = I2q + A_bar.T @ A_bar                             # (2q, 2q)
            L_bar   = torch.linalg.cholesky(M_bar)                      # (2q, 2q)

            logdet_Cbar = (psi_bar.log().sum()
                           + 2.0 * L_bar.diagonal().log().sum())

            db_cov_val = 0.5 * logdet_Cbar - 0.25 * (logdet_C[i] + logdet_C[j])

            # Mahalanobis under Bar_C:
            #   delta^T Bar_C^{-1} delta = (delta^2/psi_bar).sum() - ||y_bar||^2
            B_bar   = W_bar / psi_bar[:, None]                          # (D, 2q)
            maha_b  = (delta ** 2 / psi_bar).sum()
            v_bar   = B_bar.T @ delta                                   # (2q,)
            y_bar   = torch.linalg.solve_triangular(
                          L_bar, v_bar[:, None], upper=False).squeeze(-1)
            maha_b -= (y_bar ** 2).sum()

            db_mean_val = maha_b / 8.0
            db_val      = db_mean_val + db_cov_val
            bc_val      = torch.exp(-db_val)

            # Fill symmetric matrices
            kl_sym [i, j] = kl_sym [j, i] = kl_val
            db_mean[i, j] = db_mean[j, i] = db_mean_val
            db_cov [i, j] = db_cov [j, i] = db_cov_val
            db     [i, j] = db     [j, i] = db_val
            bc     [i, j] = bc     [j, i] = bc_val

print(f"KL_sym  — min: {kl_sym[kl_sym>0].min():.3f}  max: {kl_sym.max():.3f}")
print(f"D_B     — min: {db[db>0].min():.3f}  max: {db.max():.3f}")
print(f"BC      — min: {bc[bc<1].min():.4f}  max: {bc[bc<1].max():.4f}")

# %%
# --- Save results ---
torch.save({
    "kl_sym": kl_sym, "db": db, "db_mean": db_mean,
    "db_cov": db_cov, "bc": bc,
}, RESULTS_PATH)
print(f"Saved to {RESULTS_PATH}")

# %% [markdown]
# ## A. Heatmaps (hierarchically ordered)
#
# Components are reordered by hierarchical clustering on the BC distance matrix
# (distance = 1 - BC) so that similar components are adjacent.
# We plot BC, D_B^mean, and D_B^cov side by side to see whether overlap
# is driven by centroid proximity or covariance similarity.

# %%
dist_condensed = squareform(np.clip(1.0 - bc.numpy(), 0.0, None), checks=False)
Z     = linkage(dist_condensed, method="average")
order = leaves_list(Z)

fig, axes = plt.subplots(2, 2, figsize=(18, 5))
for ax, mat, title in zip(
    axes.flatten(),
    [bc[order][:, order], db[order][:, order] ,db_mean[order][:, order], db_cov[order][:, order]],
    ["BC (affinity)", r"D_B", r"$D_B^{\rm mean}$ (centroid sep.)", r"$D_B^{\rm cov}$ (shape mismatch)"],
):
    im = ax.imshow(mat.numpy(), aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("component j")
    ax.set_ylabel("component i")
    plt.colorbar(im, ax=ax)

plt.suptitle("Pairwise overlap — components reordered by hierarchical clustering", y=1.02)
plt.tight_layout()
plt.savefig("heatmaps.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## B. Histograms of pairwise distances
#
# Distribution of all K*(K-1)/2 pairwise values.
# Shows whether overlap is rare (long tail near 0) or widespread.

# %%
idx      = torch.triu_indices(K, K, offset=1)
bc_vals  = bc     [idx[0], idx[1]].numpy()
db_vals  = db     [idx[0], idx[1]].numpy()
dbm_vals = db_mean[idx[0], idx[1]].numpy()
dbc_vals = db_cov [idx[0], idx[1]].numpy()
kl_vals  = kl_sym [idx[0], idx[1]].numpy()

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].hist(bc_vals, bins=50, edgecolor="k", linewidth=0.3)
axes[0].set_title("Bhattacharyya Coefficient BC")
axes[0].set_xlabel("BC  (1 = identical)")
axes[0].set_ylabel("# pairs")

axes[1].hist(db_vals,  bins=50, alpha=0.6, label=r"$D_B$ total",            edgecolor="k", linewidth=0.2)
axes[1].hist(dbm_vals, bins=50, alpha=0.6, label=r"$D_B^{\rm mean}$",       edgecolor="k", linewidth=0.2)
axes[1].hist(dbc_vals, bins=50, alpha=0.6, label=r"$D_B^{\rm cov}$",        edgecolor="k", linewidth=0.2)
axes[1].set_title("Bhattacharyya Distance (decomposed)")
axes[1].set_xlabel("distance")
axes[1].legend()

axes[2].hist(kl_vals, bins=50, edgecolor="k", linewidth=0.3)
axes[2].set_title(r"Symmetrised KL $\mathrm{KL}_{\rm sym}$")
axes[2].set_xlabel("divergence")

plt.tight_layout()
plt.savefig("histograms.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## C. D_B^mean vs D_B^cov scatter
#
# Each point is a pair (i, j), coloured by BC.
# Points on the horizontal axis: overlap driven purely by centroid proximity.
# Points on the vertical axis: overlap driven purely by covariance similarity.
# The diagonal: both contribute equally.

# %%
fig, ax = plt.subplots(figsize=(6, 6))
sc = ax.scatter(dbm_vals, dbc_vals, c=bc_vals, cmap="viridis_r",
                s=8, alpha=0.6, vmin=0, vmax=1)
plt.colorbar(sc, ax=ax, label="BC (affinity)")
ax.set_xlabel(r"$D_B^{\rm mean}$  (centroid separation)")
ax.set_ylabel(r"$D_B^{\rm cov}$  (shape mismatch)")
ax.set_title("What drives overlap: mean separation vs covariance mismatch")
plt.tight_layout()
plt.savefig("db_decomposition_scatter.png", dpi=150, bbox_inches="tight")
plt.show()

