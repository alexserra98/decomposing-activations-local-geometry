# Held-Out Distribution Coverage

> **Kind:** Evaluation contract · **Status:** Experimental · **Use when:**
> Measuring how well learned centroids cover unseen samples when the continuous
> ground-truth manifold is unavailable. **Related:**
> [Toy held-out coverage experiment](../experiments/toy-heldout-coverage.md) and
> [Toy-manifold tiling evaluation](toy-manifold-tiling.md)

Held-out distribution coverage asks whether future samples from the same data
distribution lie near a learned, assignment-live centroid. It does not claim
coverage of unobserved parts of a continuous manifold.

The public functions are:

```python
from dalg.evaluation.coverage import (
    evaluate_heldout_distribution_coverage,
    nearest_live_centroid_distances,
    split_fingerprint,
    stratified_three_way_split,
    summarize_coverage_distances,
)
```

## Population contract

Fit every model parameter on the training split. Use validation only for model
selection, including MFA early stopping and best-checkpoint restoration. Compute
coverage exactly once on a separate test split that influenced neither fit nor
selection.

For a fair method comparison:

- use one fixed train/validation/test partition for every method;
- fit KMeans centroids and cluster PCA only on training rows;
- fit MFA parameters only on training rows and select its checkpoint on
  validation rows;
- use the same test rows and distance thresholds for every method; and
- define liveness from hard assignments of training rows, never from test rows.

KMeans with a fixed configuration does not need the validation rows. They
remain excluded from fitting so its training population still matches MFA's.
`stratified_three_way_split` builds a deterministic row-position partition;
persist the returned tensors and `split_fingerprint` rather than reconstructing
the split independently for each method.

## Definition

Let $X_{test}=\{x_1,\ldots,x_n\}$ be the independent test set, let
$\mu_1,\ldots,\mu_K$ be learned centroids or MFA means, and let
$C\subseteq\{1,\ldots,K\}$ contain components with at least one hard training
assignment. For every test point,

\[
d(x_j)=\min_{k\in C}\lVert x_j-\mu_k\rVert_2.
\]

Both KMeans and MFA use ambient Euclidean distance. MFA responsibility,
covariance, PCA directions, and the ground-truth manifold are deliberately not
part of this metric. This isolates the spatial placement of the learned region
centers. The implementation forces PyTorch's direct Euclidean-distance kernel,
not its squared-norm matrix-multiplication identity, to avoid cancellation for
nearby points that share a large ambient offset and to avoid TF32-dependent
results on GPUs.

The empirical coverage curve is

\[
C(\epsilon)=\frac{1}{n}\sum_{j=1}^n
  \mathbf 1[d(x_j)\leq\epsilon].
\]

Higher coverage at a fixed radius is better. The distance summaries have the
opposite convention: lower is better.

## Reported summaries

`evaluate_heldout_distribution_coverage` returns the pointwise distances and a
JSON-compatible summary with:

```text
definition
population  # independent_test_split
reference_points
components_total
components_live
mean
root_mean_square
median
r90
r95
r99
maximum
coverage_curve[]
  radius
  fraction
convention
```

`r95`, for example, is the smallest empirical radius containing 95% of test
points. Tail radii use the inverse empirical CDF order statistic
`ceil(level * n) - 1` rather than interpolation, so the reported radius is the
smallest observed value that actually attains its named coverage on the finite
test set. The maximum is only a sample-based approximation to a worst-case
covering radius, so use `r95` or `r99` as the primary tail summaries.

## Interpretation boundary

This metric estimates probability-mass coverage under the test distribution.
Dense regions contribute more than rare regions, ambient observation noise
creates an irreducible distance floor, and geometry absent from the observed
data cannot be evaluated. Call the result `heldout_distribution_coverage`, not
`geometric_manifold_coverage`.

For planted synthetic manifolds, keep the existing exact tangent evaluation as
a separate diagnostic. Good held-out coverage does not imply correct local
directions, and good tangent scores do not rule out large uncovered regions.

The implementation is covered by `tests/test_coverage.py`.
