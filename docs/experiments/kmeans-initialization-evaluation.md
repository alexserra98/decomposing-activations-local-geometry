# KMeans Initialization Evaluation

> **Kind:** Experimental workflow · **Status:** Experimental · **Use when:**
> Reproducing or inspecting the Cattell-rank and tangent-geometry evaluation of
> the toy-manifold KMeans initialization. **Related:**
> [Toy-manifold tiling evaluation](../evaluation/toy-manifold-tiling.md) and
> [HDDC rank surgery](hddc-rank-surgery.md).

This is a removable experimental feature implemented entirely under
`scripts/temporary/` and `scripts/slurm/temporary/`. It is not part of the
public `dalg.evaluation` API or the YAML training pipeline, and it does not
change `src/` to support a centroid-only special case.

The current launcher evaluates the 30,000-point, three-manifold dataset with
the saved `K=300` KMeans initialization. The evaluator itself consumes a
compatible toy-manifold shard directory, centroid artifact, and nearest-centroid
assignment bundle, but the Slurm paths and expected experiment configuration
are deliberately explicit.

## Evaluation contract

The workflow uses only quantities defined for a KMeans partition:

1. Reuse the saved KMeans centroids as the cluster means `mu_k` and reuse the
   16 saved empirical PC directions. Centroids and PCs are never overwritten.
2. Recompute the nearest-Euclidean-centroid assignment of every toy point in
   canonical stream order. The launcher overwrites the previous assignment
   bundle.
3. Mark cluster `k` eligible for rank and tangent evaluation when
   `cluster_size[k] >= 26`. Clustering and quantization metrics still use every
   point.
4. Compute the first 17 empirical covariance eigenvalues for every eligible
   cluster. PC17 is not needed: only `lambda_17` is required to test the gap
   after PC16.
5. Learn a separate rank `q_k(t)` for every eligible cluster and Cattell
   threshold `t`, with `q_max=16`.
6. Associate each centroid with its unique nearest exact planted manifold,
   without a distance cutoff, and construct the exact tangent at its projected
   location.
7. Compute matched-dimensional tangent alignment once and Cattell-rank tangent
   containment separately for every threshold.

The rank is not fixed to 16. Sixteen is only the largest selectable rank and
the number of stored PC directions.

For centroid artifacts without saved PCs, pass `--compute-pca` to compute the
leading `q_max` empirical directions for eligible clusters during evaluation.
This reuses the evaluator's covariance accumulation and leaves `centroids.pt`
and its configuration unchanged. Directions are saved in `component_metrics.pt`
as `principal_components`, with `principal_components_defined` identifying the
eligible clusters; excluded clusters have zero placeholders. This permits
evaluation when small clusters cannot support rank-16 PCA. The default path
continues to require and validate the saved PC directions.

Computed PCA uses covariance centered on each assigned cluster's empirical mean.
Finite-iteration KMeans centroids can differ from those means: this discrepancy
is reported under `pca_validation`, while geometry projections continue to use
the original saved centroids. The saved-PC path retains its strict centroid-to-
empirical-mean agreement check and covariance around the saved centroid.

### Cattell rank

For descending empirical covariance eigenvalues, the evaluator applies the raw
Cattell rule used to propose ranks inside HDDC surgery:

\[
q_k(t) = \max\left\{
j \leq q_{\max}:
\frac{\lambda_{kj}-\lambda_{k,j+1}}{\lambda_{k1}} > t
\right\}.
\]

The comparison is strictly greater than the threshold. If no gap passes, the
rank is set to one. The sweep is

```text
0.005, 0.01, 0.05, 0.1, 0.15, 0.2,
0.25, 0.3, 0.35, 0.4, 0.5, 1.0
```

This experiment stops after raw Cattell selection. It does not estimate an
HDDC noise floor and does not apply the shared-`b` active-set pruning described
in [HDDC rank surgery](hddc-rank-surgery.md). Consequently, `q_k(t)` is an
empirical cluster rank, not the loading-scale effective rank used by an MFA
checkpoint.

### Geometry metrics

For a centroid associated with a manifold of intrinsic dimension `r`:

- `tangent_alignment` compares the exact `r`-dimensional tangent with saved
  cluster PCs `1..r`. It is independent of the Cattell threshold.
- `tangent_containment` compares the same tangent with saved cluster PCs
  `1..q_k(t)`. It therefore changes across the threshold sweep.
- `rank` compares `q_k(t)` with the associated manifold's planted intrinsic
  dimension.

Both tangent scores use the principal-angle definitions and boundary-eigengap
rules in the
[canonical tiling evaluation](../evaluation/toy-manifold-tiling.md#tangent-subspace-geometry).
Global and per-manifold geometry summaries are unweighted over eligible,
associated components.

NLL, BIC, and Gaussian overlap are deliberately omitted because bare KMeans
does not define a probabilistic component density or covariance-noise model.

## Current experiment paths

The launcher fixes the following configuration:

| Setting | Value |
| --- | --- |
| Toy shards | `dalg-cache/assets/toy_manifolds_circle_helix_torus_D128_30K_noise1e4_shards/` |
| Centroid directory | `dalg-cache/toy_manifold_models_30k/centroids/kmeans_k300/` |
| Rows | `30,000` |
| Ambient dimension | `128` |
| Clusters | `300` |
| Saved PC capacity | `16` |
| Cattell `q_max` | `16` |
| Evaluation minimum population | `26`, inclusive |
| Association | Unique nearest exact manifold, no distance cutoff |

Required inputs are:

- `centroids.pt`, containing `(K, D)` centroids and `(K, D, 16)` principal
  components;
- the centroid directory's `config.json` provenance;
- the toy shard `config.json`, activation shards, and referenced
  `manifold_metadata.pt`.

## Run the experiment

From the repository root:

```bash
sbatch scripts/slurm/temporary/sbatch_evaluate_toy_kmeans_geometry.sh
```

The launcher requests one H100 and runs two stages:

1. `dalg-run-metrics assignments --centroids-path ...` recomputes all nearest
   centroid assignments.
2. `scripts/temporary/evaluate_toy_kmeans_geometry.py` validates the saved
   centroids and PCs, performs the threshold sweep, and writes the reports.

The launcher intentionally overwrites
`initialization_evaluation/nearest_centroid_assignments.pt` and
`initialization_evaluation/metrics.json`. It also overwrites the component
sidecar when one already exists. It does not create backups. The centroid
artifact and its configuration are read-only inputs.

Monitor a submitted job with:

```bash
squeue -j <job_id>
sacct -j <job_id> --format=JobID,State,ExitCode,Elapsed,NodeList
tail -n 120 logs/jobs/toy30k_kmeans_geometry_<job_id>.out
```

The four-condition, 300,000-point noise sweep uses
`scripts/slurm/temporary/sbatch_evaluate_toy_noise_kmeans_geometry.sh`. It evaluates
only `kmeans_k1000_full` under each `noise_ratio_*` folder in
`dalg-cache/toy_manifolds_10types_1each_D128_30Keach_noise_sweep_seed0/`, using
`--compute-pca`, `q_max=16`, minimum population 26, and no distance cutoff.
It reuses existing assignment bundles and refuses to overwrite metric reports.

## Outputs

Outputs are stored under
`dalg-cache/toy_manifold_models_30k/centroids/kmeans_k300/initialization_evaluation/`:

| Artifact | Contents |
| --- | --- |
| `nearest_centroid_assignments.pt` | `assignments`, `cluster_sizes`, Euclidean `min_distances`, and source provenance |
| `metrics.json` | Clustering, quantization, eligibility, association, PCA validation, alignment, and per-threshold rank/containment summaries |
| `component_metrics.pt` | Per-cluster spectra, Cattell gaps and ranks, association targets, and alignment/containment tensors |

For `T=12` thresholds and `K=300`, the important component tensors are:

| Field | Shape | Meaning |
| --- | --- | --- |
| `leading_eigenvalues` | `(K, 17)` | Empirical spectrum through `lambda_(q_max+1)` |
| `normalized_cattell_gaps` | `(K, 16)` | Consecutive gaps divided by `lambda_1` |
| `cattell_ranks` | `(T, K)` | Learned rank for each threshold and cluster; `-1` marks an ineligible cluster |
| `alignment_overlap` | `(K,)` | Matched-dimensional tangent overlap |
| `containment_overlap` | `(T, K)` | Tangent containment at each learned rank |
| `alignment_defined` | `(K,)` | Alignment validity mask |
| `containment_defined` | `(T, K)` | Containment validity mask |

The row order of every `(T, ...)` tensor is recorded by
`cattell_thresholds`. Component-axis index `k` is the KMeans cluster ID.

## Built-in validation

The evaluator fails rather than silently changing the population or geometry
when an input contract is violated. It checks:

- centroid, assignment, shard, layer, and full-stream provenance;
- assignment count, cluster-size totals, and agreement with the KMeans fit;
- at least `q_max+1` points in every eligible cluster;
- agreement between each saved centroid and its assigned empirical mean;
- orthonormality and covariance-subspace agreement of the saved PCs;
- agreement between assignment inertia and the saved KMeans inertia;
- finite spectra and Cattell gaps;
- ranks in `[1, q_max]` that cannot increase as the threshold increases;
- complete validity counts and score ranges for alignment and containment.

After completion, confirm a zero Slurm exit code and inspect both reports. A
successful file write alone is not evidence that the assignment stream was
aligned; the evaluator's provenance and size checks must also pass.

## Verified run

Slurm job `1554739` completed successfully on 2026-09-04. The saved artifacts
passed post-run shape, finiteness, population, and Cattell monotonicity checks.
All 300 clusters were eligible and associated. Matched-dimensional tangent
overlap was approximately `0.999732`. Thresholds `0.05` through `0.35` selected
the planted rank exactly for every cluster in this particular dataset.

These are in-sample initialization results: the same 30,000 points were used to
fit KMeans, construct the saved PCs, and evaluate the partition. Treat them as
an initialization diagnostic rather than held-out model performance.

## Removal

Because this is experimental, removal should be local: delete
`scripts/temporary/evaluate_toy_kmeans_geometry.py`, its temporary Slurm
launchers, this page, and this page's entry in `docs/README.md`. No `src/`
rollback is required for this feature.
