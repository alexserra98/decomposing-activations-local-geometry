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
- `components.live` and `.dead`: occupancy over the selected assignment stream,
  including validation, matching the toy suite's component-use diagnostic.
- `rank`: effective-rank definition, population, component count, and mean
  learned rank. MFA/ARD use `rank_threshold`; HDDC and KMeans use saved ranks.
  The population is all MFA-family components or PCA-valid KMeans components.
  No eligible KMeans components gives a null mean. KMeans also reports
  `pca_geometry` eligibility counts.
- `tile_alignment`: neighboring-tile leading-PC and best-subset PC alignment,
  enabled by default; see the shared definition below.
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
disabling it avoids that eigendecomposition. No PC artifact is saved. Import the
functions above when PCs are needed for analysis; the former private imports
from `toy_manifold_metrics` are removed.
