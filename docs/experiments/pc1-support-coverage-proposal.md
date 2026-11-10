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

## Equal-weight PC1–PC5 extension (2026-11-10)

The user-confirmed extension averages the five **scores**, not the vectors and
not just the endpoints PC1 and PC5. For each component, evaluate original ambient
covariance PCs 1 through 5, each with its own Gaussian central-80% marginal
half-length projected into the saved PCA75 space. Average the five coverage
fractions and the five per-PC Delta values with equal weights. Each PC uses 32
matched random directions at its own projected length, reusing seed
`42+component_id` across PCs. Final medians/means are between component averages,
without variance weighting, logarithms or dropping zero scores.

The temporary runner `scripts/temporary/run_top5_support_coverage.py` reuses
`run_pc1_support_coverage.py` for scoring and all numerical checks. It loads the
exact saved reference tensor, tau, k, R/Q IDs and calibration metadata from
`outputs/experiments/pc1_support_coverage/job_1928981/`; hashes and metadata are
checked. It loads original model checkpoints (or their hash-verified archived
paths), without refitting models, PCA, or calibration. HDDC covariance PCs come
from the SVD of masked loadings plus isotropic noise; KMeans uses saved PCs and
covariance eigenvalues. Reconstructed PC1 geometry is checked against its saved
original segment.

Eligibility is fixed before evaluating scores: both methods require retained
rank >=5, all five positive-variance individually unique axes (adjacent gaps
through the PC5/PC6 boundary >1e-10 times leading variance), and nonnull projected
norms >1e-12. Invalid PCA, dead components, rank deficiency, ties and null
projections are explicit exclusions. The component average always uses five
PCs; it never averages a variable number or invents isotropic-noise directions.
Five ambient PCs can have nonzero, nonorthogonal projections into PCA3D/4D;
projection-space dimension is not a five-PC eligibility criterion, and the
projected directions are not reorthogonalized.

Tables retain the four-panel layout and rank cutoff 0.1. N/D means no eligible
components, and eligible/total counts expose attrition. PC1 coverage and Delta
on the **same five-PC-eligible subset** are also saved, so changes in score can
be separated from population changes. The own-length comparison remains
sensitive to centers, lengths, density and the global threshold. Shorter later
PC intervals may saturate even more readily; this extension is not assumed to
resolve the original saturation.

Independent methodological and implementation reviews are in `AGENTS_chat.md`.
Tests `1931090` passed all 17 checks (11 original plus six top-five tests), and
smoke array `1931091` completed two real components in each case. Full array
`1931164` evaluates all 64 configurations. Its artifacts are under
`outputs/experiments/top5_support_coverage/job_1931164/`; export is dependent on
successful completion. Run via
`scripts/slurm/temporary/sbatch_top5_support_coverage.sh`, then
`scripts/slurm/temporary/sbatch_collect_top5_support_coverage.sh <result-root>`.

The initial full array `1931164` exposed an overly strict 1e-8 orthonormality
check on Years/layer-28 KMeans float32 PCs. The corrected extractor validates
stored Gram matrices at checkpoint precision (10 dtype eps, minimum 1e-10),
then normalizes each original ambient axis individually before projection,
matching the existing PC1 extractor. No directions are rotated or refitted.
Regression job `1931246` passed all 18 tests. All four cases are repeated
consistently in array `1931259`, with export `1931285`, under
`outputs/experiments/top5_support_coverage/job_1931259/`; earlier partial outputs
are not mixed into this result. The source snapshot is `runner_at_compute.py`.

### Evaluated PC1–PC5 result

Array `1931259` and export `1931285` completed all four cases and 64 configurations.
The independent audit `1931305` passed 3,741 component entries and 18,705 per-PC
entries, rechecking masks, all five-score means, calibration identity, exclusion
partitions, eigenvalue gaps and projection rules. Same-subset PC1 scores match
the previous experiment exactly (maximum difference 0). Maximum grid error is
0.0005767 and refinement change 0.0011308, within 0.002.

At rank threshold 0.1, every evaluable median remains **100% coverage and
0 percentage-point Delta**. HDDC K=50 at layer 28 is **N/D** for both Years and
Temperature (0/50 eligible). Other populations are also substantially restricted;
counts range from 3 to 36 eligible components in the displayed nonempty rows.
The five-PC average does not resolve saturation at the original global threshold.
Report its means alongside medians, and compare PC1 only on the same eligible
subset when assessing the effect of changing the metric.

Both four-table panels are available in PNG/PDF/SVG in the corrected result root.
The selected 16-row CSV and full 64-row CSV contain same-subset PC1 statistics;
`report.md` and `method_review_validation.json` record interpretation and checks.

## PC1-only calibration-percentile sensitivity (2026-11-10)

The user requested a controlled percentile comparison on **PC1 only**, explicitly
excluding the PC1–PC5 average. The fixed sweep is q={0.80,0.90,0.95}; all three
outcomes are retained without selecting q from model performance.

Runner: `scripts/temporary/run_pc1_percentile_sweep.py`. Reference tensors, R/Q
membership, calibration distances, PCA75, original projected PC1 geometry,
original eligibility, k=20, original central-80% segment lengths and 32 controls
seeded `42+component_id` come from the completed PC1 run `1928981`. All 64 saved
configurations are evaluated at every percentile (192 summaries). The runner
never applies the Top5 rank>=5 filter and does not retrain or refit PCA. Thresholds
are NumPy `higher` quantiles of the same saved Q-to-R distance scores.

For each component, all percentiles and all 33 directions share one grid. Begin
with 2001/4001 positions; refine every percentile together to 8001/16001 if any
fails the existing <=0.002 exact-coverage or refinement-change criterion. Save
both coarse and accepted finer-grid packed masks with axes (quantile,direction,
position-byte), and all per-direction scores. Pointwise masks must be nested
from q80 through q95. PC1 and random coverage must be nondecreasing; Delta is
not required to be monotone. At five positions along all directions, results
are checked against direct Euclidean 20th-neighbor distances at every percentile.

The q95 continuous direction scores must reproduce the old exact scores within
1e-12; if grid sizes match, all q95 masks must be identical. Otherwise the score
difference must fit the sum of old/new exact-discretization error bounds. Thus
q95 serves as a reproduction control, not a newly selected baseline.

Save ordinary component means, medians, quartiles, fraction of components with
full sampled PC1 coverage, fraction with full random-mean coverage, and the mean
fraction of fully covered random directions. The selected rank-0.1 rows produce
four-table median/mean panels separately for q80/q90/q95; the percentile must
appear in titles and captions. A common sensitivity plot compares mean PC1,
mean random and mean Delta. Full sampled-grid acceptance does not prove the
continuous segment contains no unresolved tiny gap, and global calibration
still does not establish tangent accuracy or a formal membership probability.

Reproduction commands:

```bash
sbatch scripts/slurm/temporary/sbatch_pc1_percentile_sweep.sh
# Substitute the returned array ID:
sbatch --dependency=afterok:<array-job-id> \
  scripts/slurm/temporary/sbatch_collect_pc1_percentile_sweep.sh \
  outputs/experiments/pc1_percentile_sweep/job_<array-job-id>
```

Per case: manifests and source snapshot, geometry/calibration hashes,
per-configuration component bundles, `components.csv`, `summary.csv`, unchanged
exclusions, numerical validation and completion markers. Root export: all 192
summaries, 48 selected rows, q80/q90/q95 panels in PNG/PDF/SVG and a sensitivity
plot. Generated artifacts remain outside docs; this section records the durable
protocol and results.

### Percentile-sweep validation and execution

Independent method and implementation reviews are recorded in `AGENTS_chat.md`.
The implementation reviewer caught and corrected a baseline manifest-key mismatch
before numerical launch; the end-to-end fixture now checks bare-run-ID geometry
hash keys. Job `1932217` passed all 15 tests, including higher quantiles, common
nested grids, q95 reproduction/corruption detection, and all four cases with
192 exported summaries. Smoke array `1932218` completed two real components
per case. Full array `1932231` uses all original PC1 components; artifacts are
under `outputs/experiments/pc1_percentile_sweep/job_1932231/`.

### Evaluated percentile-sweep results

Full array `1932231`, export `1932236` and independent audit `1932241` completed.
All 192 configuration/percentile summaries and 14,358 component/percentile rows
passed the audit, including every saved 3x33-direction mask, common grids,
nesting, calibration identities/order statistics, original PC1 populations,
ordinary summaries, quartiles and full-coverage fractions. Every accepted grid
has 4001 positions. The q95 masks, scores and continuous coverage reproduce
`1928981` exactly (maximum difference 0). Maximum grid error was 0.0006379 and
refinement change 0.0013567, below 0.002.

**All 48 displayed medians remain 100% coverage and 0 pp Delta**, even at q80.
Mean coverage is sensitive, especially Temperature: q95 means span
85.84–90.58%, while q80 means span 63.50–73.54% over the same eight method/K/layer
populations. Temperature/layer 28, HDDC, K=50 changes from mean coverage 86.46%
and Delta -0.58 pp at q95 to 70.23% and +4.21 pp at q80 (q90: 79.39%, +1.84 pp).
At q80, 56.0–64.6% of Temperature components still have full sampled PC1 coverage,
so the median ceiling remains compatible with substantial mean sensitivity.
Delta is not monotone in every condition. These comparisons establish sensitivity
to q, not that q is more influential than the other unvaried parameters.

All three quantiles remain reported; no percentile is selected to maximize a
model's advantage. This is a descriptive, paired sensitivity check, without
significance tests or a known-tangent alignment claim.

Artifacts under `outputs/experiments/pc1_percentile_sweep/job_1932231/`:

- `q80/`, `q90/`, `q95/`: original-layout four-table panels for component medians
  (`02_pannello_tabelle`) and means (`03_pannello_medie`), PNG/PDF/SVG, selected CSV.
- `01_sensibilita_percentile`: mean PC1/random coverage and Delta curves.
- `tutte_le_192_configurazioni.csv` and `tabelle_soglia_01.csv`: all 192 summaries
  and the 48 displayed rows, including quartiles and full-coverage fractions.
- `report.md`, `method_review_validation.json`, per-case source snapshots,
  manifests, numerical validations, component masks and unchanged exclusions.

## PC1-only neighbor sensitivity at the 90th percentile (2026-11-10)

The next user-requested study fixes q=0.90 and varies Euclidean neighbor count
k={10,20,40}, using only PC1. Each k has its own Q-to-R kth-neighbor distances
and its own 90th-percentile `higher` threshold. Reusing the k=20 distance
threshold for other k values would confound the intended calibrated comparison.
All three outcomes are retained; no k is selected from model performance.

Runner: `scripts/temporary/run_pc1_neighbor_sweep.py`. R/Q membership, saved
PCA75, original centers/unit PC1s/central-80% projected segments, original PC1
eligibility, and 32 random controls seeded `42+component_id` remain fixed.
No Top5 rank filter, model fitting or PCA refitting is performed. Since the
original calibration artifact saved R and k20 distances but not Q coordinates,
restore the full replacement validation pool through the existing checked
reconstruction workflow. Require its SHA256 to match the entire original pool
(including Q), restore Q in saved row-ID order and require R to equal the stored
reference tensor exactly. Recomputed k20 calibration distances must also be
bit-identical to those originally saved.

Each k is used consistently for interval-overlap counts, grid masks, calibration
and direct-distance audits. kth calibration distances and tau_k are nondecreasing
with k, but accepted masks, PC1/random coverage and Delta need not be monotone
because tau is recalibrated. All k values share a common final grid per component,
refined together from 2001/4001 to 8001/16001 if necessary under the same <=0.002
exact-error/refinement-change rule. Save packed coarse/final masks, direction
scores, exact coverage, ordinary means/medians/quartiles, sampled full-coverage
fractions, source hashes and unchanged exclusions.

The reproduction anchor is **q90/k20 from percentile run `1932231`**, not the
original q95 run. Its continuous direction scores must match within 1e-12;
on equal grids masks must match exactly, otherwise score differences must fit
the old/new exact-discretization error bound. Tables keep the original four-panel
layout for each k, with fixed q90 explicit in captions, separate means/medians,
and eligible counts. A sensitivity plot shows PC1/random mean coverage and Delta.

Run commands:

```bash
sbatch scripts/slurm/temporary/sbatch_pc1_neighbor_sweep.sh
# Substitute the returned array ID:
sbatch --dependency=afterok:<array-job-id> \
  scripts/slurm/temporary/sbatch_collect_pc1_neighbor_sweep.sh \
  outputs/experiments/pc1_neighbor_sweep/job_<array-job-id>
```

Independent scientific and implementation protocol reviews are in `AGENTS_chat.md`.
The source and full-pool/query provenance are frozen before evaluating outcomes.
Results remain descriptive validation diagnostics in a projected space, with
component center/length confounding and no claimed tangent-angle measurement.

### Neighbor-sweep validation and execution

Job `1932285` passed all 20 tests (five neighbor-sweep checks plus the 15 existing
PC1/percentile checks). Tests cover per-k q90 calibration against directly sorted
distances, correct k usage along all directions, full-pool reconstruction identity,
q90/k20 anchoring and corruption detection, and the four-case/192-summary export.
Smoke array `1932286` completed two real components in each case. Full array
`1932290` evaluates all original PC1 components, with artifacts under
`outputs/experiments/pc1_neighbor_sweep/job_1932290/`.

### Evaluated neighbor-sweep results

Full array `1932290`, export `1932295` and independent audit `1932296` completed.
The audit passed all 192 summaries and 14,358 component/neighbor entries,
including every saved mask, direct kth-neighbor checks for each k at five
positions in all 33 directions for every component, calibration recomputation
from saved R/Q, full-pool identities, original PC1 eligibility/exclusions and
all ordinary summaries. The k20/q90 direction masks, sampled scores and exact
coverage reproduce percentile run `1932231` exactly (maximum difference 0).
Maximum grid error was 0.0005886 and refinement change 0.0011343, below 0.002.

All 48 selected median PC1 coverages remain **100%**. Forty-seven median Deltas
are zero; Temperature/layer 4, HDDC, K=50, k=10 has a small median Delta
**+0.07225 pp**, which does not resolve the broader median saturation.
Means and Delta depend on neighborhood scale: Temperature/layer 4, HDDC, K=50
has mean coverage 86.39%, 82.75%, 82.22% and mean Delta +4.72, +3.33, +1.98 pp
at k=10,20,40. Years/layer 4, HDDC, K=50 has mean Delta +1.88, +0.44, -0.31 pp;
even its sign changes under this controlled calibrated-scale comparison.

Across all 16 fixed dataset/layer/method/K populations, the mean-PC1 coverage
span was larger for q=80/90/95 at k=20 (1.023–22.343 pp) than for k=10/20/40
at q90 (0.187–5.964 pp). This describes the tested ranges only; it is not a
universal ranking of parameter sensitivity, and Delta behaves differently.
All three k values remain reported, without selecting one for model advantage.

Artifacts under `outputs/experiments/pc1_neighbor_sweep/job_1932290/` include
`k10/`, `k20/`, `k40/` four-table median/mean panels in PNG/PDF/SVG; the neighbor
sensitivity plot; full 192-row and selected 48-row CSVs; a paired sensitivity-span
CSV versus the percentile sweep; `report.md`; `method_review_validation.json`;
and per-case calibration tensors, packed masks, source snapshots and manifests.

## PC1 segment-mass × percentile factorial (2026-11-10)

The user requested k=20 fixed, with longer PC1 segments defined by central
Gaussian marginal probability mass m={0.90,0.95,0.99}, each crossed with
calibration percentile q={0.80,0.90,0.95}. This is nine primary variants across
all 64 saved configurations: 576 summaries, 43,074 component/variant rows and
144 selected rank-0.1 table rows. PC1 only; the Top5 average/filter is not used.
All variants remain reported without choosing parameters by model advantage.

**Segment mass is distinct from variance explained.** Neither covariance
variances nor the shared PCA75 projection are changed. For original covariance
PC1 eigenvalue lambda1, the ambient half-length is
`Phi^-1((1+m)/2)*sqrt(lambda1)`. The saved projected central-80% half-length is
therefore multiplied by `Phi^-1((1+m)/2)/Phi^-1(0.9)`: 1.2834861048 for m90,
1.5293680233 for m95, 2.0099302851 for m99. The center, unit projected PC1,
projection alpha, eligibility, R/Q split, reference activations, k=20 and calibrated
q80/q90/q95 distance thresholds remain fixed. Every random control uses the same
new half-length as PC1, with the original 32 directions seeded `42+component_id`.

Runner: `scripts/temporary/run_pc1_length_percentile_sweep.py`, reusing the
existing percentile scorer directly. Within each component and segment mass,
all three q values share a refined grid with nested masks and the existing
<=0.002 exact-error/refinement-change criterion. Grids need not match across
segment masses. Normalized coverage and Delta are **not** assumed monotone with
length: support can be disconnected. Instead, exact supported and unsupported
physical lengths must both be nondecreasing as centered segments expand, and
center-supported flags must remain identical at fixed q. No cross-mass mask
nesting is imposed on grids whose physical positions differ.

The original m80×q80/q90/q95 sweep `1932231` is separately recomputed as a
reproduction control in every component. All continuous direction scores must
match within 1e-12; sampled scores must fit old/new discretization-error bounds;
equal-grid masks must match exactly. Save control masks/scores alongside the
three primary-mass result blocks, but do not include controls in primary counts
or aggregate them together with requested variants.

For each of the nine combinations, export the established four-table median and
mean panels (PNG/PDF/SVG), ordinary means/medians/quartiles, sampled full-coverage
fractions and eligible counts. Save all direction masks, source snapshots,
calibration and geometry fingerprints, unchanged exclusions and reproduction
checks. Longer-segment sensitivity remains conditional on component centers,
lengths, projection and density; it is not a known-tangent angle measurement.

```bash
sbatch scripts/slurm/temporary/sbatch_pc1_length_percentile_sweep.sh
# Substitute the returned array ID:
sbatch --dependency=afterok:<array-job-id> \
  scripts/slurm/temporary/sbatch_collect_pc1_length_percentile_sweep.sh \
  outputs/experiments/pc1_length_percentile_sweep/job_<array-job-id>
```

Method and implementation reviews are recorded in `AGENTS_chat.md`. Synthetic
validation checks actual Gaussian mass, known ball/segment coverage under all
nine conditions, fixed controls, m80 reproduction/corruption, and a four-case
576-summary export fixture using the original PC1 geometry schema.

### Segment-mass factorial validation and execution

Smoke array `1932508` completed two real components in each of the four cases.
The initial synthetic test run exposed an inaccurate hardcoded z99 expectation
and exact floating-point sphere-boundary coincidences in an analytic fixture.
The expected factor was corrected and the fixture's radii moved away from direct
sample boundaries; production scoring and its strict direct-distance audit were
unchanged. Job `1932521` passed all 19 tests (four length-factorial checks plus
15 original PC1/percentile checks). Existing tests still cover closed endpoints
and tangencies. Implementation review approved saving m80 control masks/scores
for subsequent independent inspection, separately from the nine primary variants.

Full array `1932529` runs the four complete cases after successful tests; export
`1932530` is dependent on it. Result root:
`outputs/experiments/pc1_length_percentile_sweep/job_1932529/`.

### Evaluated segment-mass × percentile results

Full array `1932529`, export `1932530` and independent audit `1932567` completed.
All 576 summaries, 43,074 primary component/variant entries and 4,786 separate
m80 reproduction controls passed inspection, including every stored mask and
95,040 representative direct k20-distance classifications. Fixed calibration
and population, Gaussian scaling, q-mask nesting, center support invariance,
exact supported/unsupported physical-length monotonicity and all ordinary
aggregates passed. The m80 controls reproduce all sampled/exact arrays and masks
of `1932231` exactly. Maximum primary grid error is 0.0009574, worst refinement
change including controls 0.0013567, both below 0.002.

Longer segments reduce saturation: **31/144 selected median PC1 coverages are
below 100%**, and **50 median Deltas are positive**. The 48 median PC1 coverages
in the m80 control were all 100%. At m99/q90, Years/layer 4, K=50, HDDC has
median coverage 91.55% and median Delta +12.71 pp, versus KMeans 93.55% and
+3.56 pp. At K=100, HDDC has 85.93% and +10.30 pp, versus KMeans 83.28% and
+3.51 pp. These illustrate stronger directional excess at longer extents;
other variants still saturate. Keep all nine outcomes and the own-center/
variance-derived-length caveat: this is not a direction-only or known-tangent
comparison, and bold entries are not significance tests.

The result root has nine `m90_q80` through `m99_q95` folders, each with the
original-layout four-table mean and median panels in PNG/PDF/SVG and selected
CSV. `report.md` renders all nine mean panels and links their median counterparts.
The root also contains all 576 summaries, 144 selected rows, three length-
sensitivity plots (one per q), machine-readable independent review, and per-case
manifests, source snapshots, unchanged exclusions and component/control masks.

### Presentation export

Slurm job `1932950` generated a 13-slide Italian PDF in
`outputs/experiments/pc1_length_percentile_sweep/job_1932529/slides/PC1_supporto_kNN_sweep.pdf`:
three introductory/method slides, the nine original mean panels (one per page,
ordered by mass then calibration percentile), and a final interpretation slide.
The panels are reused without changing their numbers or layout. The deck
separately identifies the median-based saturation diagnostic in its conclusion.
`slides/manifest.json` records the source panels, aggregation and export job.
Reproduce with `sbatch scripts/slurm/temporary/sbatch_pc1_support_slides.sh`;
the temporary renderer is `scripts/temporary/build_pc1_support_slides.py`.

An English six-step metric cartoon is available alongside the deck as
`slides/PC1_metric_step_by_step.png`, `.pdf` and `.svg`. It illustrates the
PCA/R-Q split, calibration, Gaussian-mass segment, position acceptance, matched
random controls and component aggregation. Geometry and numerical examples are
schematic, not measured results. Renderer:
`scripts/temporary/draw_pc1_metric_cartoon.py`; final export job `1932963`.

The cartoon was regenerated in job `1933190` after an independent image-only
review by a scientist role without method context. Panel 4 now shows a true
circle in equal-aspect schematic coordinates, labels its radius tau_q, and
counts actual schematic reference points inside it. All displayed accepted and
rejected positions use the same radius/count rule. Captions distinguish the
one-dimensional marginal mass, original versus projected geometry, unwhitened
Euclidean distances, validation reuse, random-length matching and equal-weight
component aggregation. The review is recorded in `AGENTS_chat.md`.

### Curves from the validated segment-mass sweep

Slurm export `1933196` reexpresses the existing mean results as six curve
panels (K50/K100 crossed with q80/q90/q95), with four dataset/layer columns.
Upper panels show PC1 and matched-random mean coverage; lower panels show mean
Delta. The x coordinate is the actual segment-length factor relative to m80
(1, 1.2834861, 1.5293680, 2.0099303), with mass labels on a secondary axis.
The m80 points come from the independently reproduced sweep `1932231`; the
m90/m95/m99 points come from `1932529`. Eligibility is checked unchanged along
each curve. These are component-specific length scalings, not common absolute
lengths across models. Connecting lines only join measured settings; no fitted
response, new score calculation or inferential band is introduced.

Artifacts: `outputs/experiments/pc1_length_percentile_sweep/job_1932529/curves/`,
including `PC1_support_curves_all_settings.pdf`, six PNG/PDF/SVG panels and an
export manifest. Renderer: `scripts/temporary/plot_pc1_support_curves.py`.

### Split-cell Delta matrices

Renderer `scripts/temporary/plot_pc1_delta_matrices.py` exports the existing
144 selected mean Delta values as eight 3×3 matrices: one per dataset/layer
and K50/K100. Rows are central Gaussian mass m90/m95/m99; columns are calibration
percentile q80/q90/q95. Each cell places HDDC-EM on the left and KMeans+PCA on
the right, displaying Delta in percentage points; the larger unrounded value
gets bold type and tinted background. This is descriptive, not a significance
claim. Two four-case panels and a two-page PDF are written under
`outputs/experiments/pc1_length_percentile_sweep/job_1932529/delta_matrices/`.
Export job: `1934426`; no metric recomputation.

### Extension to a 4×4 mass/calibration grid

The requested extension adds mass m80 and calibration percentile q75, yielding
m={80,90,95,99}% × q={75,80,90,95}%. k20, original PC1 geometry, unwhitened
PCA75, R/Q populations, eligibility, original variance, seeds and 32 random
controls remain fixed. This is central one-dimensional Gaussian probability
mass, not variance explained. The same 64 saved configurations give 1024
summaries and 256 selected rank-0.1 values.

Reuse the 576 m90/m95/m99 × q80/q90/q95 summaries from `1932529` and the
192 m80 × q80/q90/q95 summaries from `1932231`. Compute only the 256 additional
q75 summaries at the four masses. q75 uses the original calibration distances
with `numpy.quantile(..., method="higher")`; no reference or PCA refit occurs.
Each new component/mass calculation also reproduces q80 on a common refined
grid, checking exact direction scores, discretization-bounded sampled scores,
and every packed mask when grid sizes match. q75 masks must be contained in
these q80 masks. Accepted/unsupported physical length and center acceptance
invariants are checked across all four masses; original direction arrays and
eligibility must match. All masks and reproduction controls are retained.

Runner: `scripts/temporary/run_pc1_extended_mass_percentile.py`;
launcher: `scripts/slurm/temporary/sbatch_pc1_extended_mass_percentile.sh`.
The matrix renderer accepts the four masses/percentiles and preserves original
3×3 defaults. Expanded output has provenance per summary and two four-case
4×4 split-cell panels for K50/K100, plus a two-page PDF.

Analytic test job `1934458` passed six tests, including known ball coverage
at all four masses/two thresholds, q80 exact-score/mask corruption checks and
the existing percentile tests. Four-case smoke array: `1934459`.

All four smoke cases `1934459` passed q80 reproduction and numerical checks.
Full array `1934466` and its dependent validation/matrix export `1934468` use
`outputs/experiments/pc1_extended_mass_percentile/job_1934466/`.
The collector is
`scripts/slurm/temporary/sbatch_collect_pc1_extended_mass_percentile.sh`.

Full array `1934466` and export `1934468` completed successfully. All 1024
summaries and 256 selected values are available in the result root, with
`report.md`, two 4×4 split-cell panels and a combined PDF. The 192 previously
available selected values are exactly unchanged. New q75 work comprises
19144 component/mass entries; all q80 control direction scores and same-grid
masks reproduce exactly (maximum sampled difference zero). Maximum new grid
error is 0.09110 pp and refinement change 0.15739 pp, below 0.2 pp.
Final `matrix_validation.json` verifies every one of the 128 split cells,
all 256 displayed values/highlights, unchanged prior values and both PDF pages.

A layer-4-only view combines both K values in one four-matrix panel, with Years
and Temperature as panel rows and K50/K100 as panel columns. Export job1934481
reuses the expanded CSV without metric recomputation. Outputs are
`delta_matrices/PC1_delta_layer4_K50_K100.png`, `.pdf` and `.svg` in the extended
result root; `layer4_manifest.json` records all64 split cells (128 values).
Reproduce with `plot_pc1_delta_matrices.py --root
outputs/experiments/pc1_extended_mass_percentile/job_1934466 --extended --layer4-only`
via srun inside a Slurm allocation. Previous exports remain available.

### Layer-4 ratio of mean Delta

Requested descriptive comparison for the last combined layer-4 panel: equally
weight its64 dataset/K/mass/percentile settings per model. Job1934509,
`scripts/temporary/summarize_pc1_delta_ratio.py`, gives HDDC mean Delta
5.7038862485pp and KMeans mean Delta2.4445727465pp. Their ratio is2.3332855431,
a difference of3.2593135020pp. This is the ratio of aggregate means, not the
mean of per-cell ratios (which become unstable at zero/negative denominators).
K50 ratio2.3279924; K100 ratio2.3404222. Years ratio3.1849523; Temperature
ratio1.8561408. Summary saved as `layer4_delta_ratio.json` in the extended result
root. These are descriptive averages across correlated settings, not independent
replicates or a significance-tested universal model advantage.

### Neighbor-count ablation of the layer-4 4×4 panel

User requested changing only the neighbor count to k={5,10,40}, keeping the
latest layer-4 panel's two datasets and K50/K100. Retain m={80,90,95,99}% and
q={75,80,90,95}%, PCA75, original geometry/lengths/projection, R/Q rows/order,
eligibility, rank settings and random controls. Recompute calibration kth-neighbor
distances on the identical recovered Q and R, then apply the same `higher`
quantile rule. Holding numerical tau fixed while changing k would change the
calibration protocol, so each k has its own recalibrated tau values.

Runner `scripts/temporary/run_pc1_neighbor_mass_percentile.py` reuses the existing
mass scorer and percentile scorer; the latter now accepts an optional neighbor
argument, default20. Exact interval counts, grid masks and direct-distance audits
all use that same requested k. A full k20 reproduction control is included;
every component/direction/mass/percentile must match the audited original exact
scores and same-grid masks, with numerical bounds for changed grids. Query
recovery verifies original PCA/full-pool fingerprints and exact R coordinates;
recomputed k20 calibration distances must be identical to saved values. No
cross-k coverage or mask monotonicity is assumed. Within each k, q masks must be
nested, and physical supported/unsupported lengths must be nested across m.

Scope:32 saved layer-4 configurations ×16 mass/q settings per k, giving512
summaries and128 selected values per k. Three new k settings give1536 primary
summaries;512 k20 control summaries are separate. Persist full direction scores,
packed masks, calibration/query arrays, manifests and source fingerprints.
Each requested k gets the same four-matrix layer-4 panel, with actual k in its
title/footer, in PNG/PDF/SVG. No training or PAk work.

Tests1934713 passed23 checks, including direct k5/10/20/40 masks across all four
masses and calibration tests, plus previous percentile, mass, extension and
neighbor regressions. Eight-task smoke array1934722 covers both datasets at
all four k values. Launchers:
`scripts/slurm/temporary/sbatch_pc1_neighbor_mass_percentile.sh` and
`scripts/slurm/temporary/sbatch_collect_pc1_neighbor_mass_percentile.sh`.

Full array1934733 is dependent on smoke1934722. Validation/export1934734 is
dependent on full completion. Result root:
`outputs/experiments/pc1_neighbor_mass_percentile/job_1934733/`, with k-specific
subdirectories. The requested panels are
`k5/delta_matrices/PC1_delta_layer4_K50_K100.png` (and corresponding k10/k40
paths); the k20 directory is a separately recorded reproduction control.

All eight smoke cases passed before full execution. Independent table/provenance
audit1934753 depends on export1934734; its script is
`scripts/temporary/audit_pc1_neighbor_mass_tables.py`. It checks every displayed
value/highlight, eligibility, fixed manifest/PCA/reference/query/geometry
fingerprints across k, exact k20 table reproduction, and PDF page count.
It also records descriptive ratios of mean Delta using the same64 equally
weighted settings per model as the preceding layer-4 summary.

Full array1934733, export1934734 and independent audit1934753 completed.
All2048 summaries (1536 requested/512 separate k20 controls), every displayed
value/highlight, fixed-input fingerprints and original eligibility passed.
The k20 selected values, exact direction scores and matching-grid masks reproduce
previous results exactly (maximum sampled difference zero). Primary maximum
grid error is0.11104pp and maximum refinement change0.15614pp, below0.2pp.
Machine-readable independent evidence: `independent_table_validation.json`.

Using the same64 equally weighted layer-4 settings per model, ratios of mean
Delta are1.71414 at k5 (15.99986/9.33403pp),2.20340 at k10
(9.30659/4.22375pp),2.33329 at k20 (5.70389/2.44457pp), and2.06055 at k40
(3.15771/1.53246pp). These are descriptive ratios; the magnitude of both scores
and their ratio depends on k. Some individual configurations favor KMeans, so
the macro-average is not a universal per-condition superiority claim.
