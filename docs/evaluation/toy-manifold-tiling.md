# Toy-Manifold Tiling Evaluation

> **Kind:** Evaluation contract · **Status:** Current · **Use when:** Interpreting
> or changing toy-manifold association, rank, tangent geometry, or output
> metrics. **Related:** [Dataset generator](../reference/toy-manifold-dataset.md)
> and [YAML training workflow](../workflows/training-pipeline.md)

The toy-manifold tiling evaluator measures model fit and whether an MFA-family
model has placed useful local Gaussian components around each planted manifold
instance. It supports vanilla MFA, ARD, and HDDC checkpoints and writes NLL,
augmented BIC, clustering, rank, and tangent-geometry results to the pipeline run's
`metrics.json`.

The public entry point is:

```python
from dalg.evaluation.toy_manifold_tiling import evaluate_toy_manifold_tiling
```

## Evaluation populations

The evaluator deliberately uses two different component populations:

- Hard MFA assignments define clustering recovery and whether a component is
  assignment-live or assignment-dead.
- Exact mean-to-manifold proximity defines which planted manifold, if any, a
  component represents. Effective-rank and tangent-geometry metrics use this
  population even when an associated component is assignment-dead.

This separation is important while assignment behavior is being investigated.
An assignment-dead Gaussian can still be geometrically close to a planted
manifold, and an assignment-live Gaussian is not assumed to represent the
manifold that supplies most of its assigned points.

## Augmented BIC

The reported `bic.value` is the utilization-adjusted, or active-BIC, score
implemented in [`analysis/bic_improved.py`](../../src/dalg/analysis/bic_improved.py):

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
[research backlog](../research/backlog.md). Compare runs on the same dataset and
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
analytic. Mobius, Swiss-roll, and both helix projections enumerate coarse local
minima of their one-dimensional objectives and refine every candidate before
choosing the global minimum.

For the two high-dimensional types, the closed-form raw-local projections are:

- hypersphere: normalize a nonzero 11-vector, with its 10D tangent given by the
  orthogonal complement of the projected radius;
- product torus: split a 24-vector into twelve pairs, normalize each nonzero
  pair independently, and use its 90-degree rotation as that circle factor's
  tangent direction.

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
   when any raw-local pair of `product_torus_12d` is zero, where every angle on
   that circle factor is equally near.

Noiseless generated samples do not occupy these degenerate locations, but a
learned component mean can. In a within-manifold degeneracy, the projector
returns a deterministic representative point so its distance remains finite
and marks the projection non-unique because neither the point nor its tangent
is identified. If the nearest instance is unique and the representative
distance passes any configured cutoff, the component remains associated and
contributes to rank recovery, but both tangent metrics are undefined.

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
dimension. Both tangent metrics use

- \(T_i \in \mathbb{R}^{D \times r_i}\) is an orthonormal basis for the exact
  tangent at the projected mean; and
- leading eigenvectors of \(\Sigma_k\), ordered by descending eigenvalue.

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

Let \(s_k\) be component \(k\)'s rank under the same rule used for rank
recovery: the saved mask count for HDDC, or the loading-variance threshold for
vanilla MFA and ARD. `tangent_containment`
compares \(T_i\) with
\(P_k^{(s_k)} \in \mathbb{R}^{D \times s_k}\), containing the leading
`PC1..PCs_k` covariance eigenvectors. It asks whether the tangent is contained
anywhere in the component's learned signal subspace, without penalizing extra
Gaussian dimensions.

The singular values of \(T_i^\top P_k^{(s_k)}\) are the unequal-dimensional
principal-angle cosines. Missing cosines are treated as zero when \(s_k<r_i\).
The scores are

\[
\texttt{subspace_overlap}
  = \frac{1}{r_i}\sum_{j=1}^{\min(r_i,s_k)} c_j^2
  = \frac{1}{r_i}\left\|{P_k^{(s_k)}}^\top T_i\right\|_F^2,
\]

\[
\texttt{worst_direction_cosine}
  =
  \begin{cases}
    \min_j c_j, & s_k \ge r_i, \\
    0, & s_k < r_i.
  \end{cases}
\]

When \(s_k\ge r_i\), both scores equal one exactly when the tangent is contained
in the effective-rank PC subspace. When projection supplies a unique tangent,
an effective-rank-zero component has no learned signal subspace and receives
defined zero scores under both `tangent_alignment` and `tangent_containment`;
noise-only covariance axes are not treated as learned tangent directions.

Containment becomes easier as effective rank grows and is trivially perfect
when \(s_k=D\). Interpret it together with the rank-recovery metrics rather than
as a dimension-independent model comparison.

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

Containment independently applies the same rule at the \(s_k/(s_k+1)\)
boundary. It needs no boundary check when \(s_k=0\), because its score is fixed
at zero, or when \(s_k=D\), because the retained subspace is the full ambient
space. One metric can therefore be defined while the other is undefined.

Both metrics are undefined when projection geometry does not determine a unique
tangent or the tangent Jacobian is rank-deficient. This projection rule takes
precedence over the effective-rank-zero and eigengap rules: a non-unique tangent
is never assigned an artificial zero score. Undefined components remain in the
associated population and are counted explicitly.

Both lie in \([0,1]\) and are invariant to eigenvector signs and basis rotations
within either subspace. `subspace_overlap` measures average tangent coverage;
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
Matched alignment and containment maintain separate validity counts.

## Output schema

The evaluator preserves dataset, NLL, clustering, and global assignment-live
fields and adds geometry organized per planted manifold instance:

```text
schema_version
evaluation
model_kind
K
q_capacity
dataset
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
```

This is output `schema_version: 2`; completed-artifact validation requires this
version, a finite `bic.value`, `convention: higher_is_better`, and
`formula: -standard_bic / n + active_components`. Version 1 reports contain
standard BIC in `bic.value` and are not accepted as current evaluations.
For an existing run, archive its old `metrics.json` before resuming the manifest
to regenerate evaluation; the pipeline refuses to overwrite an invalid existing
report. Valid training and assignment artifacts can be reused.

`per_manifold` follows the metadata order and contains an entry for every
planted instance, including instances with zero associated components. Global
alignment and containment summaries pool components, not manifold means, so
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

## Code organization

- `toy_manifold_geometry.py` implements noiseless projections and orthonormal
  tangent construction for all twelve manifold types.
- `toy_manifold_metrics.py` implements proximity association, covariance
  eigenspaces, effective rank, principal-angle scores, and aggregation.
- `toy_manifold_tiling.py` loads artifacts and models, reconstructs the
  train/validation split, computes NLL, augmented BIC, and clustering metrics, and
  assembles the report.
- `analysis/bic.py` owns MFA-family parameter counting and the standard BIC formula.
- `analysis/bic_improved.py` owns the active-BIC formula reused by the evaluator
  and standalone helpers for scoring a saved run.

Tests are split along the same boundaries in
`tests/test_toy_manifold_geometry.py`, `tests/test_toy_manifold_metrics.py`, and
`tests/test_toy_manifold_tiling.py`. Pipeline normalization and end-to-end
schema behavior are covered in `tests/test_training_pipeline.py`.
