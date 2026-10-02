# KMeans+PCA

> **Kind:** Model explanation · **Status:** Current · **Use when:** Fitting,
> initializing from, or evaluating the pipeline KMeans model. **Related:**
> [Pipeline workflow](../workflows/training-pipeline.md) and
> [YAML reference](../reference/training-pipeline-config.md).

`dalg.models.kmeans.KMeans` is a Euclidean hard-clustering model with two
independent PCA states. KNN directions initialize MFA loadings; hard-cluster
PCs describe KMeans geometry. Neither computation overwrites the other.
All tensors are non-trainable PyTorch buffers and support `.to(...)`.

## Model API

```python
from dalg.models.kmeans import KMeans, save_kmeans, load_kmeans

model = KMeans(K=100, seed=0, device="cuda").fit(training_points)
model.compute_init_pcs(training_points, rank=32, neighbors=64)  # MFA initialization
model.compute_pcs(training_points, threshold=0.1)              # Cattell geometry
model.select_ranks(threshold=0.2)  # optionally change the geometry threshold
save_kmeans(model, "kmeans_model.pt")
model = load_kmeans("kmeans_model.pt", map_location="cpu")
```

- `mu`: `(K, D)` centroids.
- `predict(X)`: integer nearest-centroid IDs `(N,)`; ties select the first centroid.
- `responsibilities(X, tau=1.0)`: floating one-hot `(N, K)` memberships.
  Positive finite temperature leaves assignments unchanged.
- `W_init`: `(K, D, q_init)` ordered unit KNN PCs for MFA initialization.
  There is no initialization rank selection.
- `W`: `(K, D, D)` ordered unit cluster PCs, hard-masked to each selected rank.
  Columns beyond that rank and invalid components are exactly zero. There is
  no variance scaling. The full orthonormal basis is stored internally.
- `rank_mask`: `(K, D)` boolean mask derived from `component_ranks`.
- `eigenvalues`: full descending cluster covariance spectra `(K, D)`;
  invalid components are zero-filled.
- `cluster_counts`: `(K,)` training membership counts used by cluster PCA.
- `pca_valid`: `(K,)` boolean mask, true when `cluster_counts >= 2`.
- `component_ranks`: `(K,)` selected geometry ranks; invalid components have
  sentinel rank zero and must be excluded, not scored as zero-dimensional fits.

`fit` estimates means only and invalidates both PCA states. `from_centroids`
constructs a fitted model without estimating new means. Each PCA state is
optional: accessing an unavailable state raises explicitly. `q` and `q_init`
report their respective capacities, or zero when absent. Geometry capacity
`q` is always `D`; effective ranks come from `component_ranks`.

## PCA populations and sparse clusters

`compute_init_pcs(X, rank=..., neighbors=64)` uses each centroid's nearest
training points. Require `rank < neighbors <= len(X)`. Neighborhoods may overlap
and include points assigned to other centroids. Even an empty hard cluster
receives all requested initialization directions.

`compute_pcs(X, threshold=0.1)` computes all PCs using only hard-assigned
training points and immediately selects Cattell ranks. A cluster with fewer
than two members is skipped. Its centroid and component ID
remain in the model and continue participating in prediction and assignments.
All clusters may be skipped; this is a valid checkpoint.

Both methods reuse the existing PCA helpers, accumulate covariance in float64,
and center on the stored centroid rather than a newly estimated local mean.
Pipeline fitting and both PCA computations exclude validation points and honor
canonical subset selection and prefix dropping.

## Cluster rank selection

`select_ranks(threshold=0.1)` tests every consecutive gap in each full cluster
spectrum. The threshold must be in `[0, 1]`; fixed-rank geometry is not supported:

```text
rank_k = max { j : (lambda_kj - lambda_k,j+1) / lambda_k1 > threshold }
```

No passing gap gives rank one, including D=1 or a zero spectrum. Selected ranks
are capped at `cluster_counts - 1`, the sample-supported centered PCA rank.
Invalid clusters keep rank zero. `W` uses a hard column mask, as in HDDC; raising
the threshold removes directions and lowering it can restore them from the
stored full basis without recomputing PCA. `W_init` is independent. HDDC's own
rank proposal and noise fitting are unchanged.

## Pipeline artifacts and evaluation

KMeans model runs compute cluster geometry before saving `kmeans_model.pt`.
Omit `model.rank`; `model.surgery_threshold` defaults to `0.1` and always selects
the effective geometry ranks. The standalone CLI uses `--surgery-threshold`;
`--cattell-threshold` remains an alias for existing commands. Saved v3 checkpoints
with the former `cattell_threshold` metadata key remain readable.
There is no interchangeable `pca_method` setting.

Automatic MFA-family initialization computes only KNN PCs. Configure
`initialization.pca_neighbors` (default 64); its capacity is MFA `rank/q_max`.
The existing `training.direction_init: cluster_pca` selector reads **`W_init`**
from `training.kmeans_model_path`. The selector name remains for compatibility;
it does not permit using evaluation `W` as initialization directions.
Centroid-only initialization and HDDC EM remain supported without PCA.

The shared worker accepts `--pca-purpose geometry` (default) or `initialization`.
Only initialization accepts `--pca-neighbors` and requires a positive `--rank`;
it rejects Cattell selection. For standalone centroid-only fitting, `--rank 0`
skips PCA. Geometry otherwise rejects `--rank` and uses Cattell selection.
Standalone `--pca-only` can add or recompute either state while preserving the other, with the same original training
population and fitting settings.

Checkpoints use **`dalg_kmeans_v3`**, recording both independently optional PCA
states and provenance, including the full geometry basis and selected ranks.
Automatic initialization uses manifest **version 4**.
Older checkpoints, including v2, require regeneration. Older initialization
manifest versions also require a new plan; they are not silently reinterpreted.
Assignments remain separate in `kmeans_model_assignments.pt`; no `centroids.pt` is exported.

Toy evaluation uses only `W`, cluster spectra, selected ranks, and `pca_valid`.
Sparse components are excluded from rank recovery and tangent alignment and
containment, globally and per manifold. Centroid associations, clustering,
and train/validation quantization errors retain the full partition. Reports
include eligible/excluded counts separately from undefined tangent comparisons.
When no geometry is eligible, its summaries are null with zero contributing
components. Tangent comparisons use the selected active directions.
KMeans has no NLL or BIC.
