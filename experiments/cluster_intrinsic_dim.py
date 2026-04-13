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

import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from modeling.mfa import load_mfa

# %%
# --- Config ---
DATA_PATH =  "/Users/alessandroserra/Desktop/phd/decomposing-activations-local-geometry/experiments/data"
DEVICE = "mps"          # change to "cuda" or "cpu" as needed
MODEL_PATH = f"{DATA_PATH}/mfa_model.pt"
RESULTS_PATH = f"{DATA_PATH}/cluster_intrinsic_dims.pt"
VARIANCE_THRESHOLD = 0.90
MAX_SAMPLES_PER_CLUSTER = 10_000   # cap to keep SVD tractable
BATCH_SIZE = 512
MIN_POPULATION = 100               # threshold for flagging low-population clusters

# %%
# --- Load data ---
X = torch.load(f"{DATA_PATH}/activations.pt")    # (N, D)
tok = torch.load(f"{DATA_PATH}/tokens.pt")        # (N,)

print(f"Activations : {X.shape}  dtype={X.dtype}")
print(f"Tokens      : {tok.shape}")

# %%
# --- Load model ---
model = load_mfa(MODEL_PATH, map_location="cpu").to(DEVICE)
model.eval()
K, D, q = model.K, model.D, model.q
print(f"MFA: K={K} components  rank={q}  D={D}")

# %%
# --- Assign each activation to its most likely cluster ---
loader = DataLoader(TensorDataset(X, tok), batch_size=BATCH_SIZE, shuffle=False)

all_assignments = []
with torch.no_grad():
    for x_batch, _ in tqdm(loader, desc="Computing responsibilities"):
        r = model.responsibilities(x_batch.to(DEVICE))  # (B, K)
        all_assignments.append(r.argmax(dim=1).cpu())   # hard assignment

assignments = torch.cat(all_assignments)   # (N,)
sizes = torch.bincount(assignments, minlength=K)

print(f"\nCluster sizes — min={sizes.min().item()}  "
      f"max={sizes.max().item()}  "
      f"mean={sizes.float().mean():.1f}  "
      f"median={sizes.float().median():.1f}")
print(f"Empty clusters: {(sizes == 0).sum().item()}")

# %%
# --- Plot 1: cluster population distribution ---
import numpy as np

sizes_np   = sizes.numpy()
sort_idx   = np.argsort(sizes_np)[::-1]
sizes_sorted = sizes_np[sort_idx]
colors     = [ "#1f77b4" for k in sort_idx]

vmin = sizes_np.min()
vmax = sizes_np.max()
bin_edges = vmin + (vmax - vmin) * np.arange(0.0, 1.1, 0.1)
counts, edges = np.histogram(sizes_np, bins=bin_edges)

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(
    edges[:-1],
    counts,
    width=np.diff(edges),
    align="edge",
    color="#1f77b4",
    edgecolor="black"
)

ax.set_xticks(edges)
ax.legend()
ax.set_title(
    f"Cluster population distribution  (K={K})\n"
)
ax.set_xlabel("Cluster population")
ax.set_ylabel("Number of clusters")
plt.tight_layout()
plt.show()
# %%
# --- PCA intrinsic dimension per cluster ---

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


dims = torch.zeros(K, dtype=torch.long)

num_skipped_clusters = 0
for k in tqdm(range(K), desc="PCA per cluster"):
    idx = (assignments == k).nonzero(as_tuple=True)[0]
    n = idx.numel()
    if n < MIN_POPULATION:
        dims[k] = 0
        num_skipped_clusters += 1
        continue
    #if n > MAX_SAMPLES_PER_CLUSTER:
    idx = idx[torch.randperm(n)]
    dims[k] = intrinsic_dim_pca(X[idx], threshold=VARIANCE_THRESHOLD)

valid = dims > 0
print(f"\nIntrinsic dims at {VARIANCE_THRESHOLD*100:.0f}% variance threshold:")
print(f"  mean   = {dims[valid].float().mean():.2f}")
print(f"  median = {dims[valid].float().median():.2f}")
print(f"  min    = {dims[valid].min().item()}")
print(f"  max    = {dims[valid].max().item()}")
print(f"  MFA rank (q) = {q}  (reference)")
print(f"Skipped {num_skipped_clusters} clusters with population < {MIN_POPULATION}")

# %%
# --- Save results ---
results = {
    "intrinsic_dims": dims,                 # (K,) int — 0 for empty clusters
    "cluster_sizes": sizes,                 # (K,) int
    "assignments": assignments,             # (N,) int
    "variance_threshold": VARIANCE_THRESHOLD,
    "K": K,
    "rank": q,
    "D": D,
}
torch.save(results, RESULTS_PATH)
print(f"\nResults saved to {RESULTS_PATH}")

# %%
# --- Plot ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

d_valid = dims[valid].numpy()
s_valid = sizes[valid].numpy()

# Left: histogram of intrinsic dims
ax = axes[0]
bins = range(1, int(d_valid.max()) + 2)
ax.hist(d_valid, bins=bins, edgecolor="white", linewidth=0.4, color="steelblue")
ax.axvline(d_valid.mean(), color="crimson", linestyle="--", linewidth=1.5,
           label=f"mean = {d_valid.mean():.1f}")
ax.axvline(q, color="darkorange", linestyle="--", linewidth=1.5,
           label=f"MFA rank = {q}")
ax.set_xlabel("Intrinsic dimension")
ax.set_ylabel("# clusters")
ax.set_title(f"Intrinsic dim distribution ({VARIANCE_THRESHOLD*100:.0f}% variance)")
ax.legend(framealpha=0.7)

# Right: intrinsic dim vs cluster size (log-scale x)
ax = axes[1]
ax.scatter(s_valid, d_valid, alpha=0.45, s=14, color="steelblue")
ax.set_xscale("log")
ax.set_xlabel("Cluster size (log scale)")
ax.set_ylabel("Intrinsic dimension")
ax.set_title("Intrinsic dim vs cluster size")
ax.axhline(d_valid.mean(), color="crimson", linestyle="--", linewidth=1,
           label=f"mean = {d_valid.mean():.1f}")
ax.legend(framealpha=0.7)

plt.suptitle(f"MFA cluster intrinsic dimensionality  (K={K}, rank={q}, D={D})",
             fontsize=12)
plt.tight_layout()
plt.show()


