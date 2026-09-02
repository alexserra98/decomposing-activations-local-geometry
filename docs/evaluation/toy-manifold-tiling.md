# Toy-Manifold Tiling Evaluation

> **Kind:** Evaluation contract · **Status:** Current · **Use when:** Interpreting
> or changing toy-manifold association, rank, tangent geometry, or output
> metrics. **Related:** [Dataset generator](../reference/toy-manifold-dataset.md)
> and [YAML training workflow](../workflows/training-pipeline.md)

The toy-manifold tiling evaluator measures model fit and whether an MFA-family
model has placed useful local Gaussian components around each planted manifold
instance. It supports vanilla MFA, ARD, and HDDC checkpoints and writes NLL,
BIC, clustering, rank, and tangent-geometry results to the pipeline run's
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

## Bayesian information criterion

BIC uses the saved model's log likelihood on the exact recorded training split:

\[
\operatorname{BIC} = -2\log L + p\log n
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

The output stores the value, `p`, `n`, and split explicitly. This is the
standard minimizing convention, so lower BIC is better. Validation NLL remains
a separate held-out fit metric and is not used in BIC.

## Exact proximity association

For component mean \(\mu_k\) and planted manifold instance \(M_i\), the geometry
module computes the closest point

\[
p_{ki} = \operatorname*{argmin}_{p \in M_i} \|\mu_k - p\|_2
\]

on the noiseless manifold and records \(\delta_{ki}=\|\mu_k-p_{ki}\|_2\). The
projection uses the instance's saved calibration, orthonormal embedding, and
ambient offset; it does not use sampled noisy dataset points. Segment, circle,
flat disk, sphere, and torus projections are analytic. Mobius, Swiss-roll, and
helix projections enumerate coarse local minima of their one-dimensional
objectives and refine every candidate before choosing the global minimum.

A component is associated with manifold \(i\) exactly when:

1. \(i\) is the unique nearest manifold instance; and
2. \(\delta_{ki}\) is at most
   `evaluation.max_mean_to_manifold_distance`.

The cutoff is inclusive. A tie between nearest manifold instances is marked
ambiguous and left unassociated. These three global counts partition all \(K\)
components:

- `associated_components`
- `outside_cutoff_components`
- `ambiguous_components`

The nearest manifold instance can be unique even when projection within that
manifold does not determine a unique tangent, for example for a mean at the
center of a circle. Such a component remains associated and contributes to
rank recovery, but its alignment is undefined.

## Effective-rank recovery

For all model kinds, component \(k\)'s learned effective rank is

\[
\hat r_k = \#\{j : s_{kj}^2 > \tau_{rank}\,\overline{\psi}_k\},
\]

where \(s_{kj}\) is loading-column \(j\)'s scale,
\(\overline{\psi}_k\) is the mean diagonal unique variance, and
\(\tau_{rank}\) is `evaluation.rank_threshold`. HDDC's `rank_mask` is applied
before counting. The target is the intrinsic dimension \(r_i\) of the
proximity-associated manifold.

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

### Effective-rank containment

Let \(s_k\) be component \(k\)'s effective rank under the same loading-scale,
noise-floor, and HDDC-mask rule used for rank recovery. `tangent_containment`
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
in the effective-rank PC subspace. An effective-rank-zero component has no
learned signal subspace and receives defined zero scores under both
`tangent_alignment` and `tangent_containment`; noise-only covariance axes are
not treated as learned tangent directions.

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
tangent or the tangent Jacobian is rank-deficient. Undefined components remain
in the associated population and are counted explicitly rather than assigned a
zero score.

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
  parameters
  n
  split
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
  threshold
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
  tangent_alignment
  tangent_containment
```

This is output `schema_version: 1`; completed-artifact validation requires this
version and a finite `bic.value`.

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
  max_mean_to_manifold_distance: 0.1
```

`rank_threshold` and `max_mean_to_manifold_distance` must be positive; the mean
distance must also be finite. The resolved evaluation mapping is included in
the immutable run identity. Changing the cutoff therefore creates a new run ID
and output artifact rather than overwriting a completed run with different
semantics.

The evaluator requires:

- toy shards created by `save_toy_manifold_shards`, with one activation per row;
- `mfa_model.pt`, `config.json`, and `val_indices.json` in the run directory;
- a complete assignment bundle aligned to the same selected shard stream; and
- the saved `manifold_metadata.pt` referenced by the shard configuration.

## Code organization

- `toy_manifold_geometry.py` implements noiseless projections and orthonormal
  tangent construction for all eight manifold types.
- `toy_manifold_metrics.py` implements proximity association, covariance
  eigenspaces, effective rank, principal-angle scores, and aggregation.
- `toy_manifold_tiling.py` loads artifacts and models, reconstructs the
  train/validation split, computes NLL, BIC, and clustering metrics, and
  assembles the report.
- `analysis/bic.py` owns MFA-family parameter counting and the BIC formula.

Tests are split along the same boundaries in
`tests/test_toy_manifold_geometry.py`, `tests/test_toy_manifold_metrics.py`, and
`tests/test_toy_manifold_tiling.py`. Pipeline normalization and end-to-end
schema behavior are covered in `tests/test_training_pipeline.py`.
