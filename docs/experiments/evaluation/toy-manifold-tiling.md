# Toy-Manifold Tiling Evaluation

> **Kind:** Evaluation contract · **Status:** Current · **Use when:** Interpreting
> or changing toy-manifold association, rank, tangent geometry, or output
> metrics. **Related:** [Dataset generator](../../reference/toy-manifold-dataset.md)
> and [YAML training workflow](../../workflows/training-pipeline.md)

The toy-manifold tiling evaluator measures how local components cover planted
manifold instances. It supports MFA, ARD, HDDC, and KMeans+PCA checkpoints and
writes clustering, rank, and tangent geometry to `metrics.json`. MFA-family
models also report NLL and augmented BIC; KMeans reports quantization error.
All models include [held-out distribution coverage](../../evaluation/heldout-distribution-coverage.md)
by default, evaluated on the independent `test/` population.

The public entry point is:

```python
from dalg.evaluation.toy_manifold_tiling import evaluate_toy_manifold_tiling
```

## Evaluation populations

The evaluator deliberately uses two different component populations:

- Hard model assignments define clustering recovery and whether a component is
  assignment-live or assignment-dead.
- Exact mean-to-manifold proximity defines which planted manifold, if any, a
  component represents. Effective-rank and tangent-geometry metrics use this
  population even when an associated component is assignment-dead.

This separation is important while assignment behavior is being investigated.
An assignment-dead Gaussian can still be geometrically close to a planted
manifold, and an assignment-live Gaussian is not assumed to represent the
manifold that supplies most of its assigned points.

Held-out coverage has its own population: every point in `<dataset>/test/`,
measured against centroids with at least one **training** assignment. Validation
assignments do not contribute to coverage liveness. KMeans PCA validity does
not restrict those centroids. See the
[coverage integration contract](../../evaluation/heldout-distribution-coverage.md#pipeline-integration).

## KMeans geometry and quantization

KMeans reads `kmeans_model.pt` and `kmeans_model_assignments.pt`. The latter uses
the normal model-assignment schema, including one-hot confidence values and
source provenance. Stream identity, model identity, integer assignments, item
counts, and subset selection must agree with the evaluation inputs.

Geometry uses the Cattell-masked ordered `W` columns and full covariance spectrum.
`component_ranks` determines containment and rank recovery, reported as
`definition: kmeans_component_ranks`; `evaluation.rank_threshold` is ignored.
Matched-dimensional alignment uses exactly the planted intrinsic dimension.
Alignment and full containment are undefined when the selected rank is smaller
than that dimension. Partial containment evaluates positive ranks below it;
alignment, full containment, and partial containment are undefined at rank zero.
Adjusted alignment scores eligible rank-zero estimates as zero when the tangent
is defined. The same boundary-eigengap rule and undefined-geometry counts apply
for positive ranks.

KMeans emits no `nll` or `bic`. It adds:

```text
quantization
  convention: lower_is_better
  train / validation
    n
    sum_squared_distance
    mean_squared_distance
```

Distances are squared Euclidean distances to the nearest centroid. An empty
validation split has `n: 0`, zero sum, and a null mean. See
[KMeans+PCA](../../models/kmeans.md) for fitting, PCA populations, and Cattell rules.

Geometry always comes from hard-assigned training members. `W_init` is KNN
initialization state and is never used by this evaluator. Cluster PCs are
computed during model training and saved before evaluation.

Components with fewer than two training points have `pca_valid=False`.
Exclude them from global and per-manifold rank and all tangent-metric
populations, while retaining centroid associations, clustering scores, and
quantization errors. Their sentinel rank zero is not an estimated rank.

`pca_geometry` records `version: 1`, `minimum_cluster_points: 2`, total
`eligible_components` / `excluded_components`, and corresponding
`eligible_associated_components` / `excluded_associated_components`.
Each per-manifold `components` record adds `pca_eligible` and `pca_excluded`.
Sparse exclusions are separate from the existing `undefined_components` count,
which now describes eligible components lacking an identifiable tangent
comparison. All-excluded geometry completes with null means and zero contributors.
The stored geometry tensor has shape `(K, D, q)`, with
`q = max(1, max(component_ranks))` and zero padding beyond each selected rank.
The full eigenvalue spectrum is retained; discarded directions are not.
Effective ranks determine comparisons. See [KMeans+PCA](../../models/kmeans.md)
for storage and PCA recomputation.

## Augmented BIC

For MFA-family models, the reported `bic.value` is the utilization-adjusted, or active-BIC, score
implemented in [`analysis/bic_improved.py`](../../../src/dalg/analysis/bic_improved.py):

\[
S_{\mathrm{active\text{-}BIC}}
  = -\frac{\operatorname{BIC}_{\mathrm{standard}}}{n} + K_{\mathrm{active}}.
\]

**Higher is better.** `K_active` counts components with at least one hard MAP
assignment on the exact recorded training split. The evaluator slices the saved
assignment bundle to those training rows; validation-only assignments do not
earn an activity reward. Each additional active component adds exactly one to
the score at fixed likelihood and parameter count. The separate global
`components.live` diagnostic still counts activity over the full selected
stream, including validation.

This implements the utilization preference recorded in the
[research backlog](../../research/backlog.md). Compare runs on the same dataset and
split with the same nominal `K`: changing `K` changes the maximum activity
reward. Dividing standard BIC by `n` keeps the activity term on a scale that
does not vanish as the dataset grows. The unit reward is a research preference,
and a one-point MAP winner counts as active.

The underlying standard BIC uses the saved model's log likelihood on that same
training split:

\[
\operatorname{BIC}_{\mathrm{standard}} = -2\log L + p\log n
                       = 2n\,\overline{\operatorname{NLL}} + p\log n.
\]

Here `n` is the number of training activation vectors and `p` is the
identifiable model parameter count. Loading matrices are counted after removing
their rotational redundancy. Vanilla MFA uses its fixed configured rank; ARD
uses its saved effective component ranks; and HDDC uses `component_ranks`, with
one noise parameter for shared `b` or one per component for `b_k`. Adaptive-rank
models count their per-component selected dimensions, while vanilla MFA counts
its one common rank. Shared diagonal unique variance contributes `D` parameters
and component-specific diagonal variance contributes `K * D`.

The output stores the augmented score, `p`, `n`, split, activity counts,
formula, and `higher_is_better` convention explicitly. `bic.standard_bic`
retains the standard minimizing BIC as a diagnostic;
`bic.standard_bic_per_sample_reward` and `bic.activity_reward` sum to
`bic.value`. Validation NLL remains a separate held-out fit metric and is not
used in either BIC calculation.

## Exact proximity association

For component mean \(\mu_k\) and planted manifold instance \(M_i\), the geometry
module computes the closest point

\[
p_{ki} = \operatorname*{argmin}_{p \in M_i} \|\mu_k - p\|_2
\]

on the noiseless manifold and records \(\delta_{ki}=\|\mu_k-p_{ki}\|_2\). The
projection uses the instance's saved calibration, orthonormal embedding, and
ambient offset; it does not use sampled noisy dataset points. Segment, circle,
flat disk, sphere, torus, cylinder, 10D-hypersphere, and 12D-product-torus projections are
analytic. Mobius, Swiss-roll, and helix projections enumerate coarse local
minima of their one-dimensional objectives and refine every candidate before
choosing the global minimum.

For the two high-dimensional types, the closed-form raw-local projections are:

- hypersphere: normalize a nonzero 11-vector, with its 10D tangent given by the
  orthogonal complement of the projected radius;
- product torus: split a 24-vector into twelve pairs, normalize each nonzero
  pair independently, and use its 90-degree rotation as that circle factor's
  tangent direction.

Evaluation supports two additional 10D dataset types, each with eleven raw
local coordinates:

- `swiss_roll_10d` uses `(theta*cos(theta), theta*sin(theta), h1, ..., h9)`.
  The spiral uses the saved `swiss_theta_min/max` bounds and the existing
  Swiss-roll projection search; each height is clamped independently to the
  saved `swiss_height_min/max` bounds. Its tangent contains the spiral
  derivative and nine height directions. Projection uniqueness is inherited
  from the spiral, including ties between distinct closest points.
- `cylinder_10d` is the lateral product `S^9 × [-2.5, 2.5]`: the first ten
  coordinates lie on the unit sphere and the last coordinate is height.
  Projection normalizes the radial vector and clamps height. Its tangent is
  the nine-dimensional radial orthogonal complement plus the height direction.
  On the cylinder axis, projection returns the first radial unit vector as a
  deterministic representative and marks geometry non-unique.

Both tangents retain all ten directions at height boundaries, matching the
existing boundary convention. Saved calibration, embedding, and offset are
applied exactly as for the other manifold types.

The two twelve-coordinate types use the parameterizations in the
[generator reference](../../reference/toy-manifold-dataset.md#swiss-roll-and-helix-with-twelve-native-coordinates):

- `swiss_roll_12d` has intrinsic dimension 2. Clamp height to its saved bounds
  and minimize the remaining one-dimensional angle objective on the saved
  interval, including both endpoints. Its tangent columns are the analytic
  angle derivative and the height direction, including at boundaries.
- `helix_12d` has intrinsic dimension 1. Search the periodic interval
  `[0, 2*pi)` and use the analytic derivative across its six circular pairs.
  At the raw-local origin, return the angle-zero representative at squared
  distance six and mark geometry non-unique. Distinct tied closest points
  also make geometry non-unique; periodic copies of the same point and tangent
  represent the same geometry.

Both reuse coarse-grid candidate refinement, projection-tie handling, and
saved calibration/embedding/offset transformations. The native dimension
`12` is separate from the dataset dimension, typically `128`.

"Raw-local origin" and "raw-local zero pair" below refer to coordinates after
reversing the saved ambient offset, embedding, and calibration. They are not in
general the zero vector or a zero pair in the model's ambient coordinates.

A component is associated with manifold \(i\) by default when \(i\) is the
unique nearest manifold instance. An optional cutoff can restrict this
association: when `evaluation.max_mean_to_manifold_distance` is a number,
association additionally requires

\[
\delta_{ki} \leq \texttt{evaluation.max_mean_to_manifold_distance}.
\]

The optional cutoff is inclusive. Projection has two distinct uniqueness
questions:

1. **Nearest-instance uniqueness.** A distance tie between separate planted
   manifold instances is marked `ambiguous` and left unassociated.
2. **Within-manifold uniqueness.** The uniquely nearest instance may itself
   have multiple equally near projected points. This occurs at the raw-local
   origin of `hypersphere_10d`, where every sphere point is equally near, or
   when any raw-local pair of `product_torus_6d` or `product_torus_12d` is zero, where every angle on
   that circle factor is equally near.

Noiseless generated samples do not occupy these degenerate locations, but a
learned component mean can. In a within-manifold degeneracy, the projector
returns a deterministic representative point so its distance remains finite
and marks the projection non-unique because neither the point nor its tangent
is identified. If the nearest instance is unique and the representative
distance passes any configured cutoff, the component remains associated and
contributes to rank recovery, but all tangent metrics are undefined.

These three global counts partition all \(K\) components:

- `associated_components`
- `outside_cutoff_components`
- `ambiguous_components`

The same within-manifold rule applies to existing degeneracies such as a mean
at the center of a circle or on the cylinder axis: this is not reported as
cross-manifold ambiguity. Cylinder projection normalizes the radial pair and
clamps height to `[0, 5]`; its tangent spans the circumferential and axial
directions, including at the boundary rings.

## Learned-rank recovery

For HDDC, component \(k\)'s learned rank is the number of active entries in
its saved `rank_mask`: `rank_mask[k].sum()`. Evaluation uses this final rank
directly, without any additional loading-variance or noise-floor filtering.
`evaluation.rank_threshold` has no effect on HDDC. HDDC rank summaries record
`definition: hddc_rank_mask_count` and omit `threshold`.

For vanilla MFA and ARD, component \(k\)'s effective rank remains

\[
\hat r_k = \#\{j : s_{kj}^2 > \tau_{rank}\,\overline{\psi}_k\},
\]

where \(s_{kj}\) is loading-column \(j\)'s scale,
\(\overline{\psi}_k\) is the mean diagonal unique variance, and
\(\tau_{rank}\) is `evaluation.rank_threshold`. These summaries record
`definition: loading_variance_above_noise_threshold` and the threshold value.
The evaluator compares the model's rank against two
targets for the proximity-associated manifold:

- `rank` targets its local intrinsic dimension; and
- `ambient_rank` targets its native ambient dimension, stored by the generator
  as `embedding_dim` (for example, circle 2, helix 3, and `helix_4d` 4).

The native ambient dimension is the dimension of the manifold's coordinate
space before its random embedding. It is not the dataset-wide activation
dimension `D` (typically 128).

Global and per-manifold rank summaries report:

- number of evaluated components;
- mean learned rank;
- exact-match fraction;
- within-one-match fraction; and
- mean absolute rank error.

An empty population reports `null` for every rate or mean.

## Tangent-subspace geometry

For an associated component \(k\) on manifold \(i\), construct the full
covariance

\[
\Sigma_k = W_k W_k^\top + \Psi_k.
\]

Let \(r_i\) be that manifold's intrinsic dimension and \(D\) the ambient
dimension. All tangent metrics use

- \(T_i \in \mathbb{R}^{D \times r_i}\) is an orthonormal basis for the exact
  tangent at the projected mean; and
- leading eigenvectors of \(\Sigma_k\), ordered by descending eigenvalue.

Alignment and full containment require the component's effective rank \(q_k\)
to be at least \(r_i\). Partial containment applies only when \(0<q_k<r_i\).
These three metrics are undefined at rank zero. Adjusted alignment combines
alignment and partial containment across ranks and scores eligible rank-zero
components as zero when the reference tangent is defined. For each metric,
components outside its rank range contribute to `undefined_components` and are
excluded from its score means, while remaining in rank recovery. A summary with
no valid components reports a JSON `null` mean.

### Matched-dimensional alignment

`tangent_alignment` compares \(T_i\) with
\(P_k^{(r_i)} \in \mathbb{R}^{D \times r_i}\), containing exactly the leading
`PC1..PCr_i` eigenvectors. It asks whether the Gaussian's strongest \(r_i\)
covariance directions recover the tangent.

Thus a one-dimensional manifold uses only PC1, while a two-dimensional manifold
uses the plane spanned by PC1 and PC2. The evaluator does not select a more
favorable subset of later PCs. If a two-dimensional tangent is represented by
PC2 and PC3 while PC1 is normal, the leading two-dimensional covariance plane
has only partial overlap and misses one tangent direction.

Let the singular values of \(T_i^\top P_k^{(r_i)}\) be
\(c_j=\cos\theta_j\), the cosines of the principal angles. The component scores
are:

\[
\texttt{subspace_overlap}
  = \frac{1}{r_i}\sum_{j=1}^{r_i} c_j^2,
\qquad
\texttt{worst_direction_cosine}
  = \min_j c_j.
\]

### Learned-rank containment

Let \(q_k\) be component \(k\)'s rank under the same rule used for rank
recovery: the saved component rank for KMeans, the saved mask count for HDDC,
or the loading-variance threshold for vanilla MFA and ARD. For \(q_k\ge r_i\),
`tangent_containment` selects exactly \(r_i\) PCs from the first \(q_k\) to
maximize their subspace overlap with the tangent:

\[
J^* = \operatorname*{argmax}_{J\subseteq\{1,\ldots,q_k\},\,|J|=r_i}
      \left\|P_{k,J}^\top T_i\right\|_F^2,
\qquad
\texttt{subspace_overlap}
  = \frac{1}{r_i}\left\|P_{k,J^*}^\top T_i\right\|_F^2.
\]

The objective separates into individual PC contributions
\(a_j=\|T_i^\top p_{kj}\|^2\). Selecting the \(r_i\) largest contributions
therefore finds the exact maximum without enumerating subsets. Equal
contributions favor the lower PC index. This compares spans; it does not match
individual PCs to individual tangent basis vectors.

Let \(c_j\) be the principal-angle cosines of \(T_i^\top P_{k,J^*}\). The
companion score is

\[
\texttt{worst_direction_cosine}=\min_j c_j.
\]

It describes the same overlap-maximizing subset and is not optimized separately.
Both scores equal one exactly when a subset of \(r_i\) retained PCs spans the
tangent. When \(q_k=r_i\), containment matches leading-PC tangent alignment.
For a curve, overlap reduces to the largest squared tangent cosine among the
retained PCs.

Even \(q_k=D\) does not guarantee a perfect score: tangent information can be
spread across more than \(r_i\) PCs. For example, a tangent line at 45 degrees
to each of two orthogonal retained PCs scores `(0.5, sqrt(0.5))`, despite lying
inside their plane. Extra PCs are not penalized if an exact tangent subset
exists; interpret containment together with rank recovery. Components with
insufficient rank receive no full-containment score.

### Partial containment

`tangent_partial_containment` measures whether the learned subspace lies inside
the tangent when \(0<q_k<r_i\). It uses all leading learned-rank PCs
\(P_k^{(q_k)}\), with the containment direction reversed and no subset selection.
For the \(q_k\) singular values \(c_j\) of \(T_i^\top P_k^{(q_k)}\):

\[
\texttt{subspace_overlap}
  = \frac{1}{q_k}\sum_{j=1}^{q_k}c_j^2
  = \frac{1}{q_k}\left\|T_i^\top P_k^{(q_k)}\right\|_F^2,
\qquad
\texttt{worst_direction_cosine} = \min_{1\le j\le q_k}c_j.
\]

The normalization is the learned rank, not the intrinsic dimension, and no
missing tangent directions are padded with zeros. A learned line lying inside
a 2D tangent scores `(1, 1)`; a line at 45 degrees to that plane scores
`(0.5, sqrt(0.5))`; an orthogonal line scores `(0, 0)`. The worst-direction score
measures the least tangential learned direction. Interpret these scores together
with rank recovery, since they do not measure coverage of the full tangent.

This metric is undefined for \(q_k=0\) and \(q_k\ge r_i\). It uses the same
projection, PCA eligibility, and boundary-eigengap rules as full containment,
checking the boundary between PCs \(q_k\) and \(q_k+1\).

### Adjusted alignment

`tangent_adjusted_alignment` measures tangent recovery across effective ranks
using a single squared-overlap score. For an eligible associated component with
a defined reference tangent, let \(A_k\) be its alignment `subspace_overlap` and
\(P_k\) its partial-containment `subspace_overlap`:

\[
S_k =
\begin{cases}
A_k & q_k \ge r_i,\\
(q_k/r_i)P_k & 0 < q_k < r_i,\\
0 & q_k = 0.
\end{cases}
\]

For positive rank this equals
\(\|T_i^\top P_k^{(\min(q_k,r_i))}\|_F^2/r_i\): it uses the leading
`min(q_k, r_i)` PCs and normalizes by the intrinsic dimension. A perfectly
tangent line on a 2D manifold scores `0.5`; a line at 45 degrees to that tangent
plane scores `0.25`. When `q_k >= r_i`, the score equals matched-dimensional
alignment. Extra dimensions incur no additional penalty; rank recovery remains
a separate diagnostic. There is no worst-direction variant.

Positive-rank scores reuse alignment's boundary-eigengap check at `r_i` or
partial containment's check at `q_k`. An eligible rank-zero component needs no
eigenspace check, but its reference tangent must still be defined. Undefined
tangents and unidentifiable positive-rank subspaces remain undefined. KMeans
components with `pca_valid=False` are excluded, including their sentinel rank
zero; they do not contribute zeros or undefined counts.

Compute the rank adjustment per component before averaging. Global summaries
pool valid components across manifolds; per-manifold summaries use the valid
components associated with that instance. Aggregate mean rank and mean partial
overlap cannot in general reconstruct this score, since ranks and overlaps can
vary together. The evaluator reuses its component geometry without additional
model loading or eigendecomposition.

### Eigenspace identifiability

The matched \(r_i\)-dimensional covariance subspace is evaluated only when its
relative boundary eigengap satisfies

\[
\frac{\lambda_{r_i}-\lambda_{r_i+1}}
     {|\lambda_{r_i}|} > 10^{-6},
\]

with eigenvalues in descending order. Ties among eigenvalues inside the retained
subspace are valid because they do not change the retained subspace. A tie at
the \(r_i/(r_i+1)\) boundary makes matched alignment undefined.

Containment independently applies the same rule at the \(q_k/(q_k+1)\)
boundary before selecting a subset. It needs no boundary check when \(q_k=D\),
because the candidate subspace is the full ambient space. Among components
meeting the rank requirement, one metric can therefore be defined while the
other is undefined.

Containment uses the returned PC basis without additional exclusions for equal
or nearly equal eigenvalues within the candidate subspace. Rotations among
those PCs can change the available subsets and thus the score, even though the
full candidate subspace is unchanged. Near eigenvalue ties, small covariance
perturbations can therefore affect the score.

All tangent metrics are undefined when projection geometry does not determine a unique
tangent or the tangent Jacobian is rank-deficient. A non-unique tangent is never
assigned an artificial zero score. Undefined components remain in the
associated population and are counted explicitly.

All scores lie in \([0,1]\) and are invariant to eigenvector signs. The
principal-angle comparison is invariant to basis rotations within the two
compared subspaces. Containment's subset selection depends on the candidate PC
basis, but not on the choice of tangent basis. For alignment and full containment,
`subspace_overlap` measures average tangent coverage and
`worst_direction_cosine` detects whether any tangent direction is missed.

For example:

| Geometry | `subspace_overlap` | `worst_direction_cosine` |
| --- | ---: | ---: |
| Exact tangent subspace | 1 | 1 |
| Orthogonal 1D PC1 and tangent | 0 | 0 |
| 2D spaces sharing exactly one direction | 0.5 | 0 |

Global and per-manifold summaries are unweighted means over valid associated
components. Each score reports `mean`, `valid_components`, and
`undefined_components`; `mean` is `null` when there are no valid components.
Alignment, full containment, partial containment, and adjusted alignment each
maintain separate validity counts.

## Output schema

The MFA-family report preserves dataset, NLL, clustering, and global assignment-live
fields and adds geometry organized per planted manifold instance. KMeans uses
the same schema version and common fields, replacing `nll` and `bic` with
`quantization` as described above:

```text
schema_version
evaluation
model_kind
K
q_capacity
dataset
heldout_distribution_coverage  # when enabled; full empirical CDF and distance summaries
nll
bic
  value
  standard_bic
  standard_bic_per_sample_reward
  activity_reward
  active_components
  inactive_components
  K
  parameters
  n
  split
  assignment_rule
  formula
  convention
clustering
components
association
  rule
  max_mean_to_manifold_distance
  associated_components
  outside_cutoff_components
  ambiguous_components
rank
  definition
  threshold  # vanilla MFA and ARD only
  population
  components
  mean_learned
  exact_match
  within_one_match
  mean_absolute_error
ambient_rank
  definition
  threshold  # vanilla MFA and ARD only
  population
  components
  mean_learned
  exact_match
  within_one_match
  mean_absolute_error
tangent_alignment
  definition
  rank_requirement: effective_rank_gte_intrinsic_dim
  aggregation
  relative_boundary_eigengap_threshold
  subspace_overlap
    mean
    valid_components
    undefined_components
  worst_direction_cosine
    mean
    valid_components
    undefined_components
tangent_containment
  definition
  rank_requirement: effective_rank_gte_intrinsic_dim
  aggregation
  relative_boundary_eigengap_threshold
  subspace_overlap
    mean
    valid_components
    undefined_components
  worst_direction_cosine
    mean
    valid_components
    undefined_components
tangent_partial_containment
  definition
  rank_requirement: effective_rank_gt_zero_lt_intrinsic_dim
  normalization: effective_rank
  aggregation
  relative_boundary_eigengap_threshold
  subspace_overlap
    mean
    valid_components
    undefined_components
  worst_direction_cosine
    mean
    valid_components
    undefined_components
tangent_adjusted_alignment
  definition
  rank_requirement: effective_rank_gte_zero
  normalization: intrinsic_dim
  zero_rank: zero_if_tangent_defined
  aggregation: unweighted_component_mean
  relative_boundary_eigengap_threshold
  subspace_overlap
    mean
    valid_components
    undefined_components
per_manifold[]
  manifold_id
  type_id
  type_name
  intrinsic_dim
  embedding_dim
  components
    associated
    assignment_live
    assignment_dead
  rank
    target_intrinsic_dim
    components
    mean_learned
    exact_match
    within_one_match
    mean_absolute_error
  ambient_rank
    target_ambient_dim
    components
    mean_learned
    exact_match
    within_one_match
    mean_absolute_error
  tangent_alignment
  tangent_containment
  tangent_partial_containment
  tangent_adjusted_alignment
```

Alignment and full containment record
`rank_requirement: effective_rank_gte_intrinsic_dim`. Partial containment records
`rank_requirement: effective_rank_gt_zero_lt_intrinsic_dim` and
`normalization: effective_rank`. Completed-artifact validation requires these
contracts and the partial-containment definition. Containment records
`definition: best_intrinsic_dim_subset_of_leading_rank_covariance_principal_angles`
for MFA-family models and
`definition: best_intrinsic_dim_subset_of_leading_rank_pca_principal_angles`
for KMeans. Completed-artifact validation requires this definition, so older
reports using the full learned-rank span must be re-evaluated. Metric keys,
aggregation, and schema version remain the same.

Adjusted alignment records
`definition: leading_min_intrinsic_effective_rank_covariance_subspace_overlap`
for MFA-family models and
`definition: leading_min_intrinsic_effective_rank_pca_subspace_overlap` for
KMeans. Completed-artifact validation requires its definition, rank requirement,
intrinsic-dimension normalization, rank-zero rule, and unweighted aggregation.
Reports missing this metric are no longer current evaluations. Existing tangent
metrics keep their definitions, and adjusted alignment is an additive field
within schema version 2.

Generic aggregation includes
`tangent_adjusted_alignment.subspace_overlap.mean`, `.valid_components`, and
`.undefined_components` in both `general_metrics.csv` and
`manifold_metrics.csv`. Existing reports and CSVs gain these fields only after
reevaluation and aggregation; reading them does not recompute the metric.

This is output `schema_version: 2`; MFA-family completed-artifact validation
also requires this version, a finite `bic.value`, `convention: higher_is_better`, and
`formula: -standard_bic / n + active_components`. Version 1 reports contain
standard BIC in `bic.value` and are not accepted as current evaluations.
When coverage is enabled, completed-artifact validation also requires the
`heldout_distribution_coverage` block, with training-only liveness and a valid
full empirical curve. Use `dalg-run-pipeline evaluate --manifest <manifest>` to
replace an obsolete report; normal pipeline resume refuses to overwrite an
invalid existing report. Valid training and assignment artifacts are reused.

`per_manifold` follows the metadata order and contains an entry for every
planted instance, including instances with zero associated components. Global
tangent-metric summaries pool components, not manifold means, so
manifolds with more associated Gaussians receive proportionally more weight.

## Pipeline configuration

The evaluator is enabled after the required assignments stage:

```yaml
assignments:
  enabled: true

evaluation:
  enabled: true
  kind: toy_manifold_tiling
  batch_size: 4096
  device: cuda
  rank_threshold: 1.0
  max_mean_to_manifold_distance: null
  heldout_distribution_coverage: true
```

`null` is the default and disables distance filtering, so every component with
a unique nearest manifold is associated with it. A numeric
`max_mean_to_manifold_distance` enables the old cutoff behavior and must be
finite and positive. `rank_threshold` must be positive when used for vanilla
MFA or ARD; it is ignored for HDDC. Existing manifests may retain the legacy
field, but HDDC metrics explicitly record the mask-based rank definition.
The resolved evaluation
mapping is included in the immutable run identity. Changing the cutoff
therefore creates a new run ID and output artifact rather than overwriting a
completed run with different semantics.

The evaluator requires:

- toy shards created by `save_toy_manifold_shards`, with one activation per row;
- `mfa_model.pt`, `config.json`, and `val_indices.json` in the run directory;
- a complete assignment bundle aligned to the same selected shard stream; and
- the saved `manifold_metadata.pt` referenced by the shard configuration.

Coverage additionally requires a compatible `test/` dataset with at most
100,000 points; disabling coverage removes this requirement. Missing `test/`
raises an exception naming `scripts/temporary/add_toy_manifold_test_split.py`.

## Code organization

- `toy_manifold_geometry.py` implements noiseless projections and orthonormal
  tangent construction for all registered manifold types.
- `toy_manifold_metrics.py` implements proximity association, covariance
  eigenspaces, effective rank, principal-angle scores, and aggregation.
- `toy_manifold_tiling.py` loads artifacts and models, reconstructs the
  train/validation split, computes NLL, augmented BIC, and clustering metrics, and
  assembles the report.
- `toy_manifold_coverage.py` validates independent test splits, streams coverage,
  and checks the coverage report contract; `coverage.py` owns distances and CDF summaries.
- `analysis/bic.py` owns MFA-family parameter counting and the standard BIC formula.
- `analysis/bic_improved.py` owns the active-BIC formula reused by the evaluator
  and standalone helpers for scoring a saved run.

Tests are split along the same boundaries in
`tests/test_toy_manifold_geometry.py`, `tests/test_toy_manifold_metrics.py`, and
`tests/test_toy_manifold_tiling.py`. Pipeline normalization and end-to-end
schema behavior are covered in `tests/test_training_pipeline.py`.
