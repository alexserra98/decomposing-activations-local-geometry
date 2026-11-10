# PC1 support coverage: proposed metric

> **Kind:** Experiment · **Status:** Pilot evaluated; median saturation observed · **Use when:**
> Designing a measure of how far a component's PC1 follows observed activation support
> without requiring a known manifold tangent. **Discussed:** 2026-11-09.
> **Related:** [PC1 density experiments](pc1-density-alignment.md),
> [implemented PC1 density contract](../evaluation/unsupervised.md#pc1-density-support),
> and [held-out distribution coverage](../evaluation/heldout-distribution-coverage.md).

## Motivation and scope

Measure the fraction of a straight PC1 segment that remains close enough to real
data, with proximity calibrated against separate real points. This is applicable
when the manifold and its tangent/normal spaces are unknown. Ground-truth tangent
angles would be used only to validate the proposal on synthetic datasets.

The existing density-retention metric averages density relative to center
density. A narrow, high-density peak can dominate that average. The proposed
score instead assigns each query position a binary supported/unsupported label:
every sampled position has equal weight, regardless of density magnitude.

This is a separate diagnostic, implemented as a temporary pilot below. It does not change the existing
metric, pipeline schema, or PAk experiment. It does not establish manifold
membership, tangent accuracy, or an expected advantage for either MFA-EM or
KMeans+PCA. Unlike held-out distribution coverage, it probes learned segments
against data rather than measuring real points' distances to learned centroids.

## Algorithm

### 1. Fix a common coordinate space and evaluation split

Use the same coordinates, Euclidean distance, and evaluation population for all
model comparisons. A first study could reuse the shared PCA75 space: fit the
centered PCA on training activations only, without whitening or standardization.
Do not fit separate projections per component or method in a controlled comparison.

Split the evaluation activations into two disjoint sets, with saved row IDs and
a fixed seed shared by all methods:

- **Reference R:** supplies neighbors for every distance query.
- **Calibration Q:** supplies real queries that determine an acceptance threshold.

Neither set fits model parameters or the PCA. If the evaluation pool is a
validation set previously used for checkpoint selection, label the result as a
descriptive validation diagnostic, not an untouched-test estimate. The sizes of
R and Q, stratification, and split seed remain protocol choices to specify before
evaluation. Require at least k reference points and a nonempty calibration set.

### 2. Calibrate a global support threshold

For any query x in the evaluation space, define

```text
s(x) = d_k(x, R) = Euclidean distance to its k-th nearest point in R
tau  = empirical 95th percentile of {s(q) : q in Q}
supported(x) = 1 if s(x) <= tau, otherwise 0
```

The initial suggestions are k=20 and the 95th percentile; these are not validated
defaults. Specify the empirical-quantile convention before implementation. R and
Q must be separate so calibration points cannot count themselves as neighbors.

Approximately 95% of the calibration scores fall below this threshold, subject
to finite-sample quantiles and ties. This is not a 95% probability that an
accepted query belongs to a manifold, nor a formal conditional-coverage guarantee.
This operation uses neighbor distances directly; it does not estimate density
or raise radii to a density dimension.

### 3. Reuse each component's original PC1 segment

Use the existing covariance-PC1 geometry and eligibility rules. With original
unit PC1 v, covariance eigenvalue lambda_1, and center mu, retain the central 80%
one-dimensional Gaussian marginal interval, with half-length

```text
a_original = Phi^-1(0.9) * sqrt(lambda_1)
```

For a shared row-oriented PCA basis P and training mean m:

```text
c = P(mu - m)
b = P v
u = b / ||b||
a = a_original * ||b||
x(t) = c + t u,  t in [-a, a]
```

Project the original segment; do not recompute the cluster covariance PC1 in
reduced coordinates. Record alpha=||b||^2 and original/projected lengths. Undefined
PC1s and null projections remain explicit exclusions, rather than zero scores.
For ambient evaluation, use c=mu, u=v, and a=a_original.

### 4. Sample uniformly and count accepted positions

For M equally spaced positions spanning both endpoints:

```text
t_j = -a + 2a*j/(M-1),  j = 0, ..., M-1
b_j = supported(c + t_j*u)
C(PC1) = sum_j b_j / M
```

Use an odd M>=3 if the grid must include the center. Increasing M changes spatial
resolution, not segment length. For example, 160 accepted positions out of 200
give an estimated coverage of 0.80; the example does not prescribe M.

The practical score is a count divided by M. Its continuous interpretation is
the length fraction `(1/(2a)) integral[-a,a] supported(c+t*u) dt`; the uniform-grid
fraction approximates this quantity. No density-valued integrand is used.

Save the ordered binary acceptance mask, not just its average. Identical coverage
fractions can represent one continuous supported interval or several disconnected
crossings. A possible additional diagnostic is the supported extent in each
direction contiguous with the center; its discretization remains to be defined.

### 5. Compare with matched random directions

Generate 32 isotropic unit directions directly in the evaluation space, using a
recorded seed such as `42 + component_id`. Evaluate each with the same c, a, M,
R, k, Q, and tau as PC1. Reuse seeds across configurations.

```text
C_random_mean = arithmetic mean_r C(random_r)
Delta C = C(PC1) - C_random_mean
```

C(PC1) is the proposed primary score; Delta C is a secondary directional control.
Both coverage scores lie in [0,1], and Delta C lies in [-1,1]. A PC1 score of
0.80 and a random mean of 0.30 give Delta C=0.50, or 50 percentage points. This
difference remains defined when the random mean is zero; no enrichment ratio is
required. These are illustrative numbers, not measured results.

## Interpretation and unresolved choices

- **Center placement:** each component's own center makes the score depend on
  position as well as direction. Record whether the center itself is supported.
  A future direction-only comparison could transport competing directions to
  common real-data anchors, using a common length; that is a separate control.
- **Spatial scale and curvature:** even an exact tangent line leaves a curved
  manifold over a sufficiently long interval. Original component-specific lengths
  preserve the existing experiment's geometry but do not isolate direction.
  Curves C(u,a) over prespecified common lengths would separate local behavior
  from the extent over which a straight direction remains supported.
- **Population density:** a global kNN-distance threshold can reject legitimate
  sparse regions and accept gaps between densely sampled regions. Local-spacing
  normalization is a possible later extension, not part of this first definition.
- **Dimension and projection:** neighbor distances still depend on dimensionality,
  sample size, and distance concentration. Removing the density exponent does
  not remove these effects. Discarded coordinates remain invisible after PCA.
- **Threshold and resolution:** results depend on k, the calibration percentile,
  the R/Q split, and M. Binary threshold crossings and small unsupported intervals
  require numerical-resolution checks. Hyperparameters must not be chosen to
  favor a particular model.
- **Aggregation:** retain per-component coverage and exclusions. Summaries could
  report equal-component means, medians, and quantiles in ordinary score space;
  zero is a valid coverage value, so historical log-score aggregation should not
  be copied. Repeated rank thresholds are not independent replicates.

## Validation before adoption

1. On synthetic manifolds with known tangents, rotate a direction toward a normal
   direction at fixed anchor and length; examine whether the score declines.
   Repeat at different curvatures, sampling densities, dimensions, and noise levels.
2. Check that isolated high-density patches cannot compensate for a long
   unsupported part of a segment; inspect the binary masks and connected extent.
3. Refine M at fixed geometry and verify score stability. Measure sensitivity to
   prespecified k values, calibration percentiles, and repeated R/Q splits.
4. Compare models with shared references and coordinates; distinguish their own
   center/length comparison from any common-anchor/common-length control.

No tangent or normal estimator is required when subsequently applying the metric
to real activations. The validation above is intended to test what the proposed
proxy captures, not to assume that high coverage already proves good alignment.

## Standard-kNN pilot (2026-11-09)

The pilot runner is `scripts/temporary/run_pc1_support_coverage.py`. It reuses
all four saved PCA75 geometries from the density experiment (Years and
Temperature at layers 4/28), without retraining or refitting PCA. This experiment
uses k=20 neighbor distances directly, not PAk or a density-valued score.

Protocol fixed before evaluating models:

- Restore the replacement validation pool in original row order with the existing
  PAk experiment's prompt/tokenization and data-fingerprint checks. Original
  tensor identity remains unverified. This is a descriptive validation diagnostic.
- Sort original global row IDs, permute with a CPU Torch generator seeded 42,
  assign the first `ceil(0.20*N)` rows to calibration Q and the rest to reference
  R. Years has R=1601/Q=401; Temperature has R=1200/Q=300. The same dataset's
  two layers share row membership. No stratification is selected from results.
- Calibrate one global tau per dataset/layer using the 95th percentile of
  Q-to-R 20th-neighbor distances, NumPy `method="higher"`. Save all calibration
  scores, the threshold's order statistic, row IDs, realized acceptance, and hashes.
- Keep original projected centers, unit PC1s, half-lengths, alpha, exclusions,
  and 32 matched random directions seeded `42+component_id`.
- Evaluate endpoint-inclusive odd grids M=2001 then 4001. Require every direction's
  absolute coverage error against exact continuous support and change from the
  previous grid to be <=0.002 (0.2 percentage points). If needed, refine to 8001
  and 16001 under the same rule; fail if unresolved. Report the finer passing
  grid's **discrete** coverage, preserving the proposal's count/M definition.
- Save the ordered 2001-position and accepted finer-grid masks, bit-packed along
  the position axis (NumPy default big-endian bit order; trim unpacked rows to
  the saved grid length). Save all direction scores, exact coverage diagnostics,
  PC1 supported intervals, center support, and refinement history.
- Summarize ordinary per-component median coverage and median component Delta;
  also save means, quartiles, random mean coverage, and exclusions. Zero remains
  valid. The four overview tables use rank threshold 0.1 and retain the historical
  layout, with columns **PC1 coverage (%)**, **Delta coverage (percentage points)**,
  and eligible/total. Bold values are descriptive comparisons, not significance.

For efficient exact classification, a query is supported iff at least 20
reference balls of radius tau contain it. Intersect each ball with the line,
then count closed overlapping intervals at grid positions. Group coincident
endpoints to compute exact continuous covered length for the resolution check.
Tangencies can contribute a supported grid point but have zero continuous
length. Negative perpendicular squared distances are clipped only within a
64-epsilon relative roundoff bound. Each uniquely scored component is audited
against direct Euclidean neighbor counts at five positions in all 33 directions.

Two independent reviewers checked the method and implementation; their findings
are recorded in `AGENTS_chat.md`. Synthetic tests check a known straight tangent
rotated toward its normal, interval boundaries/tangencies, isolated dense patches,
and equality with direct kNN in dimensions 1/3/28/141. These checks do not replace
the broader curvature, density, noise, threshold, and split-sensitivity studies
listed above. The pilot does not establish tangent accuracy or direction-only
superiority: centers, lengths, density, sample size, and discarded coordinates
still affect the score.

```bash
sbatch scripts/slurm/temporary/sbatch_pc1_support_coverage.sh
# Substitute the returned array job ID:
sbatch --dependency=afterok:<array-job-id> \
  scripts/slurm/temporary/sbatch_collect_pc1_support_coverage.sh \
  outputs/experiments/pc1_support_coverage/job_<array-job-id>
```

Each case writes source manifests, `calibration.pt`, per-configuration component
bundles, `components.csv`, `summary.csv`, `exclusions.csv`, `validation.json`, and
`COMPLETED.json`. Export rechecks all 64 ordinary-score medians and shared layer
splits, then writes PNG/PDF/SVG, the selected 16-row CSV and full 64-row CSV.
Only a completed root `COMPLETED.json` denotes a finished four-case export.

## Explanatory vignette

The accompanying AI-generated illustration is schematic: its distances, point
counts, and 80%/30% coverages are not experimental measurements. Text and formulas
above define the proposal. Generated media remains outside docs, as a local
artifact under outputs:

![PC1 support coverage: calibration, segment queries, and accepted-position fraction](../../outputs/experiments/pc1_support_coverage_proposal/20261109/vignette.png)

The vignette was generated using the built-in image-generation tool. Its visual
brief was to match the original PC1-density vignette's three-panel white/navy/teal
academic layout, replacing density integration with disjoint reference/calibration
sets, a calibrated kNN-distance threshold, uniformly sampled PC1/random segments,
and binary acceptance counts. The output directory is ignored by Git; the proposal
is self-contained and does not require the image to define or reproduce its formulas.

### Pilot execution

Tests `1928980` passed all 10 checks, including the saved-geometry and four-table
export fixture. Smoke array `1928973` completed two real components in every
dataset/layer case. Full evaluation array `1928981` uses all 64 saved configurations;
results are under `outputs/experiments/pc1_support_coverage/job_1928981/`.

### First evaluated result

Full array `1928981` completed all four cases: 64 configurations and 4,786
component entries including repeated geometries/configurations. Final validation
and export `1929046` passed all 11 targeted tests. The independent results review
unpacked every mask and rechecked scores, calibration, splits and summaries.
Maximum absolute grid-to-exact error was 0.0005767 and refinement change 0.0011308,
both below 0.002. Final artifact root is
`outputs/experiments/pc1_support_coverage/job_1928981/`.

At the displayed rank threshold 0.1, **all 16 medians saturate at 100% PC1
coverage, 100% random coverage, and 0 percentage-point Delta**. This protocol
does not distinguish methods at the median. The primary four-table panel keeps
that prespecified aggregation. A supplementary, explicitly labelled mean panel
was added after observing saturation, without changing parameters or component
scores. Means show smaller, mixed differences; for example HDDC Temperature,
layer 28, K=50 has negative mean Delta. These are descriptive results and do not
establish statistical significance or a direction-only advantage.

`02_pannello_tabelle` contains the medians; `03_pannello_medie` contains the
supplementary means (each in PNG/PDF/SVG). `report.md` explains the result and
validation. `runner_at_compute.py` preserves the source version matching the
four manifests; `runner_at_export.py` preserves the final table rendering.
