# Unsupervised Evaluation

> **Kind:** Evaluation contract · **Status:** Current · **Use when:** Evaluating
> saved models without ground-truth labels or manifold geometry. **Related:**
> [Coverage](heldout-distribution-coverage.md),
> [pipeline workflow](../workflows/training-pipeline.md), and
> [configuration reference](../reference/training-pipeline-config.md).

## Pipeline use

Use the normal training, assignment, and evaluation pipeline with:

```yaml
evaluation:
  enabled: true
  kind: unsupervised
  test_shard_dir: dalg-cache/my_independent_test_activations
  device: cuda
  batch_size: 4096
  rank_threshold: 1.0
  tile_alignment: true
  tile_alignment_neighbors: 5
```

The evaluator supports KMeans, MFA, ARD, and HDDC. Both window activations and
last-token prompt activations use the existing shard reader. Synthetic shards
can also be evaluated without consulting their manifold metadata.

Coverage requires an explicitly supplied independent test directory. The caller
must ensure these samples influenced neither training nor model selection.
The test directory must differ from the training root and match its layer,
ambient dimension, window length, and effective prefix dropping. Directory
separation alone cannot establish statistical independence. No split is created
automatically. `test_shard_dir` accepts the existing `#pile_<subset>_<N>` suffix;
relative paths resolve against the repository root.

Set `heldout_distribution_coverage: false` to omit coverage and its test-data
requirement. `max_mean_to_manifold_distance` applies only to toy evaluation and
must remain `null` for this suite.

For saved runs, including runs whose evaluation was originally disabled:

```bash
uv run --locked dalg-run-pipeline evaluate \
  --manifest /path/to/manifest.jsonl \
  --evaluation-kind unsupervised \
  --test-shard-dir dalg-cache/my_independent_test_activations
```

The normal `--submit`, `--dry-run`, selection, and resource options apply.
Overrides leave source manifests and training artifacts unchanged. Reports
are written to the existing run directory after successful validation.

## Metrics and populations

The schema-version-2 report identifies `evaluation: unsupervised` and reuses
the existing metric definitions:

- MFA-family models: training/validation `nll` and the existing augmented `bic`.
- KMeans: training/validation `quantization`, including sum and mean squared
  distance and point count. Empty validation has a null mean. MFA NLL requires
  a nonempty validation split.
- HDDC and KMeans+PCA: training/validation `reconstruction` MSE and standard
  error, with included and excluded activation counts; see below.
- `components.live` and `.dead`: occupancy over the selected assignment stream,
  including validation, matching the toy suite's component-use diagnostic.
- `rank`: effective-rank definition, population, component count, and mean
  learned rank. MFA/ARD use `rank_threshold`; HDDC and KMeans use saved ranks.
  The population is all MFA-family components or PCA-valid KMeans components.
  No eligible KMeans components gives a null mean. KMeans also reports
  `pca_geometry` eligibility counts.
- `tile_alignment`: neighboring-tile leading-PC and best-subset PC alignment,
  enabled by default; see the shared definition below.
- `pc1_density`: optional PC1 density retention and random-direction enrichment
  for KMeans+PCA and MFA-EM (HDDC); disabled by default.
- `heldout_distribution_coverage`: the existing distance summaries and full
  empirical curve, using training-only component occupancy.

There are no label-recovery, manifold-association, target-rank-error, tangent,
or per-manifold results. CSV aggregation adds one general row per run and no
manifold rows.

Assignment bundles must cover the complete selected canonical token stream and
record the checkpoint, model type, subset, layer, prefix dropping, and item
count, as written by the current assignment CLI. Saved validation rows determine
training occupancy; all retained tokens in a window inherit that window's split.
Model fitting and coverage use the saved preprocessing settings.

For `(rows, window, D)` storage, the selected point count is
`rows * (window - drop_prefix)`. Last-token and synthetic datasets use
`window=1, drop_prefix=0`. Dataset summaries record both row and point counts.
Coverage retains only distances while streaming and keeps the existing
100,000-point limit. It never silently truncates the supplied test population.
Its `source` records `shard_dir`, `subset_spec`, `layer`, `num_rows`, `num_points`,
`window`, `drop_prefix`, and `d_model`.

## Reconstruction MSE

Both evaluation suites automatically report reconstruction error for HDDC and
KMeans+PCA on the saved training and validation splits. They use the same subset,
prefix dropping, device, and batch size as the existing fit metrics, during the
same shard passes. No independent test data or new evaluation setting is required.

Following [the paper, Appendix D, Eq. 18](https://arxiv.org/html/2602.02464v2#A4),
define `e_i = ||x_i - reconstructed_i||² / D`, where `D` is the ambient activation
dimension. Report `mse = mean(e_i)` and
`standard_error = std(e_i, correction=1) / sqrt(n)`. Moments accumulate in float64
without retaining all errors. Standard error treats activations as observations;
it is not corrected for correlation between tokens in a document or window.

- **HDDC:** reuse `model.reconstruct(x, use_mixture_mean=True)`. Each component
  contributes its centroid plus posterior-mean latent reconstruction, weighted
  by posterior responsibility. Noise and saved rank masks participate. This
  follows the paper's reconstruction equation, rather than selecting only the
  MAP component or applying an orthogonal projection. Every sample contributes.
- **KMeans+PCA:** assign the nearest centroid `mu_k` and reconstruct
  `mu_k + U_k U_k.T (x - mu_k)` using the saved retained cluster PCs, never
  initialization PCs. Samples assigned to `pca_valid=False` components are
  excluded and counted. This extends the paper's centroid-only KMeans baseline.
  Existing quantization still scores all samples.

Within the top-level schema version 2, the block is:

```text
reconstruction
  version: 1
  method: mfa_posterior_mean | nearest_centroid_local_pca
  normalization: per_activation_dimension
  population: all_samples | samples_assigned_to_pca_valid_components
  dimension: D
  convention: lower_is_better
  train / validation
    mse
    standard_error
    n
    excluded_n
```

The first method/population pair is HDDC's; the second is KMeans+PCA's.
`n` counts included activation vectors, not windows or coordinates.
`n + excluded_n` equals `dataset.train_points` or `dataset.validation_points`.
Dataset summaries also record `d_model`, `window`, and `drop_prefix` in both suites.
Empty eligible populations have null MSE and standard error; a single included
sample has null standard error. An empty KMeans validation split is allowed;
MFA NLL still requires nonempty validation. If every KMeans component lacks valid
PCA, both reconstruction means are null. When KMeans excludes samples, its MSE
and HDDC's MSE describe different populations.

CSV aggregation exports `reconstruction.train.*` and
`reconstruction.validation.*` automatically. Completion validation requires the
block for HDDC and KMeans. Older reports remain aggregatable; explicit saved-run
reevaluation replaces them with the new metrics. Ordinary resume preserves an
obsolete report and reports it as invalid. Legacy MFA/ARD evaluation is unchanged.

## Neighboring-tile alignment

Both evaluation suites include `tile_alignment` by default. Set
`evaluation.tile_alignment: false` to disable it, or change the positive integer
`evaluation.tile_alignment_neighbors` (default `5`). Saved-run evaluation accepts
`--tile-alignment`, `--no-tile-alignment`, and `--tile-alignment-neighbors N`.

The shared evaluator, `dalg.evaluation.tile_alignment.evaluate_tile_alignment`,
builds directed Euclidean centroid kNN in the model's coordinate system. Eligible
tiles are live in the selected assignment stream (including validation), have
positive learned rank, and, for KMeans, have valid PCA. Self-neighbors are excluded;
distance ties use ascending component ID. The effective neighbor count is capped
at the eligible population minus one. Reciprocal edges are retained for both tiles.
No manifold labels, associations, or tangent metadata influence this graph.

For neighboring ranks `r_i`, `r_j`, let `m = min(r_i, r_j)`:

- `leading_pc_alignment`: compare the first `m` PCs of both tiles.
- `best_subset_pc_alignment`: compare the smaller tile's retained basis with the
  best subset of `m` individual PCs among the larger tile's retained PCs.

Both scores are the mean squared principal-angle cosine, in `[0, 1]`, using
`subspace_alignment` and `best_subset_alignment`. The latter selects the PCs
with the largest squared projection onto the smaller basis; it does not search
arbitrary linear combinations of the larger basis. Scores coincide for equal
ranks. Each tile averages its outgoing comparisons, and the global score is an
unweighted mean of those tile means. These are orientation-consistency diagnostics,
not proof of correct geometry: identical subspaces everywhere score perfectly.

MFA-family PCs come from `W W.T + diag(psi)`; KMeans uses saved cluster PCs.
Adjacent eigenvalues within the retained range and at its boundary are checked
for relative gaps at most `1e-6`. An aggregated `RuntimeWarning` identifies
affected eligible components: tied eigenvalues can make individual PCs and
PC-subset scores unstable. Comparisons still use the returned PCs, with no
correction or new filtering. Ties entirely in unused noise directions or padded
PCs do not trigger the warning. Existing toy tangent-metric eligibility checks
are unchanged.

The top-level schema remains version 2. The `tile_alignment` block has `version: 1`
and records `definitions`, `graph`, `population`, `aggregation`, requested
`neighbors`, `effective_neighbors`, eligible/excluded component counts, and
`relative_eigengap_tolerance`. Each score has a global `mean`. `per_tile` records
component IDs, ranks, and both neighbor means. `comparisons` records directed
`source`/`target` IDs, Euclidean `distance`, both ranks, and both scores.
With fewer than two eligible tiles, comparisons are empty and means are null.
An isolated eligible tile still has a per-tile record with null means.

CSV aggregation exports the scalar summaries and serializes the detailed lists.
When enabled, completion validation requires the block and matching neighbor
settings. Use explicit saved-run reevaluation to replace an older report;
ordinary resume does not overwrite obsolete reports. Disabling alignment omits
its block. No PC artifact is saved.

## PC1 density support

Enable `evaluation.pc1_density: true` for KMeans+PCA or MFA-EM (HDDC). The default
`pc1_density_space: pca75` fits a centered global PCA on the saved **training**
partition, retaining the smallest prefix explaining at least 75% of variance.
There is no standardization or whitening. `pc1_density_space: ambient` skips
projection. These are distinct density spaces; compare models using the same
space, activation population, saved split, and settings.

The density reference is the complete **saved validation** token stream, with
its subset and prefix dropping preserved. No split is reconstructed from a seed.
Validation may have selected model checkpoints; this is a descriptive diagnostic,
not an untouched-test estimate. Coverage's `test_shard_dir` is independent of this
reference. At least `pc1_density_neighbors` validation points are required
(default 20); missing or insufficient validation raises an error.

For a query in density dimension `d`, the estimator is
`p_hat(x) = k / (n V_d r_k(x)^d)`, using Euclidean kNN radii. KMeans uses its
saved unit PC1 and leading cluster covariance eigenvalue. Isotropic HDDC uses
an SVD of its effective loadings and adds the noise variance to the leading
squared singular value, avoiding a dense ambient covariance decomposition.
Invalid KMeans PCA, inactive or isotropic HDDC components, nonpositive leading
variance, and relative leading eigengaps at most `1e-10` are excluded explicitly.
Eligibility follows model geometry, without additional assignment-live filtering.

With unit covariance PC1 `v`, define `a = Phi^-1(0.9) sqrt(lambda_1)`: the central
80% one-dimensional Gaussian marginal interval. In ambient space, score
`mu + t v` for `t` in `[-a,a]`. In PCA75 space, project this original segment:
`P(mu-m) + t Pv`. Normalize `Pv` and use half-length `a ||Pv||`; do not recompute
PC1 from projected cluster covariance. Record `alpha = ||Pv||^2`, and exclude
projected norms at most `1e-12`.

For a density-space unit direction `u` and segment half-length `L`:

```text
A(u) = (1 / (2L)) integral[-L,L] p_hat(center + t u) / p_hat(center) dt
E    = A(PC1) / arithmetic_mean_r A(random_r)
```

`log_retention = log A(PC1)` is primary; `log_enrichment = log E` is secondary.
The baseline uses `pc1_density_random_directions` isotropic unit directions
(default 32), generated directly in the density space at the same center and
length, with seed `pc1_density_seed + component_id` (default seed 42).

CPU float64 adaptive quadrature uses a 201-position pilot and relative error
tolerance `1e-4`. Integration failures and zero/nonfinite radii raise explicitly.
Coarse/refined grid differences remain diagnostics. Training covariance is
streamed, but costs O(D²) memory; the full validation reference is held in memory.
The metric is opt-in because PCA and per-component integration can be expensive.
The `device` option still controls the other evaluation metrics.

The top-level report stays at schema version 2. Its `pc1_density` block has
`version: 1`, definitions, settings, reference row IDs and point count, actual
density dimension, PCA training count and variance fractions, eligible/excluded
counts and reasons, and scalar diagnostics per component (including alpha and
original/projected lengths for PCA75). Global `mean` and `median` summaries use
natural-log scores with equal component weights; empty eligible populations
have null summaries. Exponentiating the median log score gives the historical
pilot's reported retention/enrichment summary. PCA bases and density profiles
are not saved by the pipeline; PCA is recomputed per evaluated run.

CSV aggregation exports these summaries and serializes detailed lists using
its existing generic flattener. Completion validation requires the enabled block,
matching settings, split sizes, and consistent component summaries. Disabling
the metric omits the block. Use explicit saved-run reevaluation to replace older
reports. Saved-run flags are `--pc1-density` / `--no-pc1-density`,
`--pc1-density-space {pca75,ambient}`, `--pc1-density-neighbors N`,
`--pc1-density-random-directions N`, and `--pc1-density-seed N`.

Retention can exceed one and is not a manifold-membership probability.
Enrichment depends on the baseline space and local scale. Projection can hide
deviations; alpha and center/length diagnostics should accompany comparisons.
The [PC1 density experiment context](../experiments/pc1-density-alignment.md)
contains current commands and the historical years/temperature findings.

## Component PCs on demand

The reusable geometry functions live in `dalg.evaluation.subspaces`:

```python
from dalg.evaluation.subspaces import (
    effective_component_ranks,
    leading_covariance_eigenspaces,
    subspace_alignment,
    best_subset_alignment,
    leading_subspace_is_identifiable,
)

ranks = effective_component_ranks(model, threshold=1.0)
eigenvalues, pcs = leading_covariance_eigenspaces(model, max_subspace_dim=model.q)
```

For MFA-family models, these are the PCs of `W @ W.T + diag(psi)`, computed on
CPU in float64. For KMeans they are the stored cluster PCs and eigenvalues.
The return shapes are `(K, min(max_subspace_dim + 1, D))` for eigenvalues and
`(K, D, max_subspace_dim)` for PCs, ordered from largest eigenvalue downward.
The extra eigenvalue supports the boundary-eigengap check.

KMeans preserves zero padding beyond each component's rank and for invalid
components; callers must respect `component_ranks` and `pca_valid` when comparing
subspaces. Alignment expects orthonormal basis columns and normalizes overlap
by the first basis's dimension. `best_subset_alignment` chooses that many PCs
to maximize overlap. These are the same functions used by toy evaluation.

Unsupervised pipeline evaluation computes PCs when tile alignment is enabled;
disabling it avoids that eigendecomposition. PC1 density, when enabled,
computes its own leading geometry and optional global PCA. No PC artifact is saved. Import the
functions above when PCs are needed for analysis; the former private imports
from `toy_manifold_metrics` are removed.
