# KNN-PCA Centroid Initialization

> **Kind:** Experimental workflow · **Status:** Temporary · **Use when:**
> Building the two `K=1000` centroid initializations that require 32 local
> directions even though some hard KMeans clusters contain fewer than 33
> points. **Related:**
> [KMeans initialization evaluation](kmeans-initialization-evaluation.md).

> **DEPRECATED KMEANS WORKFLOW:** The standalone centroid-bundle experiment
> below is historical. The temporary builder has been removed. New work uses
> [KMeans+PCA](../models/kmeans.md) through the training pipeline.

The numerical functions in `src/dalg/init/neighborhood_pca.py` are reused by the
model for MFA-family initialization only. Set `initialization.pca_neighbors`
to control its KNN neighborhood size. KMeans evaluation always uses cluster
members and excludes undersized clusters from geometry metrics.


## Why it exists

The `K=1000` full-data KMeans partition has clusters as small as 20 points, and
the 35% partition has clusters as small as 5 points. Exact rank-32 PCA of every
hard cluster is therefore impossible: 32 data-supported covariance directions
require at least 33 points under the standard cluster-PCA contract.

This workflow replaces hard-cluster membership only for the PCA stage. It does
not change the KMeans centroids.

## Experimental definition

For each fixed centroid `mu_k`:

1. Select its 64 nearest Euclidean points from the same population used to fit
   that centroid. The full condition searches all 150,000 points; the 35%
   condition searches only its deterministic 52,500-point subset with sample
   seed 0.
2. Form residuals around the stored centroid, `x_i - mu_k`.
3. Compute the float64 covariance of those 64 residuals.
4. Store its 32 leading eigenvectors as `principal_components[k]`.

Every centroid therefore has 32 supported directions, but neighborhoods may
overlap and points need not be assigned to that centroid by KMeans. These are
nearest-neighbor covariance directions, not hard-assignment cluster PCs. The
output directory names and `config.json` both record that distinction.

## Run and outputs

Run:

```bash
sbatch scripts/slurm/temporary/sbatch_build_toy_10types_knn64_pca32.sh
```

The two source KMeans fits and derived artifacts live under
`dalg-cache/toy_manifold_models_10types_1each_15Keach/centroids/`:

```text
kmeans_k1000/                                      # full, means only
kmeans_k1000_sample35pct/                          # 35%, means only
kmeans_k1000_full_knn64_pca32_experimental/        # full, 32 KNN PCs
kmeans_k1000_sample35pct_knn64_pca32_experimental/ # 35%, 32 KNN PCs
```

The derived `centroids.pt` files use the existing `dalg_centroids_v1` tensor
layout and can be selected explicitly through `centroids_path` with
`direction_init: cluster_pca`. Compatibility does not make the method a
pipeline default: the adjacent configuration identifies
`principal_components.method` as `nearest_neighbor_covariance`, sets
`hard_assignment_pca: false`, and records the source centroid checksum,
population, sample seed, neighborhood size, and validation results.

## Validation

The temporary builder refuses nonempty output directories and validates:

- source centroid shape and sampling provenance;
- exactly 64 unique neighbors per centroid;
- finite `(1000, 128, 32)` directions;
- per-centroid PC orthonormality;
- bit-identical centroids between each source and derived artifact.

## Removal

The standalone experiment can be removed by deleting the following; automatic
pipeline KNN PCA and its shared functions are independent of these launchers:

- `scripts/temporary/build_toy_knn_pca_centroids.py`;
- `scripts/slurm/temporary/sbatch_build_toy_10types_knn64_pca32.sh`;
- this page and its entry in `docs/README.md`;
- the two `*_knn64_pca32_experimental/` artifact directories.

Removing KNN PCA from the pipeline as well requires removing its optional
configuration and builder branch, `src/dalg/init/neighborhood_pca.py`, and the
corresponding tests and documentation.
