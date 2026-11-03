# PC1 density support on language-model activations

> **Kind:** Experiment · **Status:** Years and temperature pilots evaluated, 2026-11-02 · **Use when:**
> Evaluating saved years or temperature principal directions at layers 4 and 28 against local data density.

This page records the original ambient pilot, the shared global-PCA75 variant
and the paired direction-swap control below. Normalized retention is the
primary score; enrichment is a secondary comparison with matched random
directions.

The repeats below apply the same PCA75 metric to their own saved
activations and models, with a separate training PCA and output directory for
each dataset and layer.

## Question and scope

Does a component's first principal direction pass through more data-supported
space than random directions at the same center and spatial scale?

This pilot uses the existing `years_neutral` Llama-3.1-8B last-token extraction
in its original 4096-dimensional coordinates. Only zero-based layer **4** is
evaluated. The 16 completed configurations combine KMeans+PCA or HDDC-EM,
`K` in `{50, 100}`, and rank thresholds in `{0.0001, 0.001, 0.01, 0.1}`.
No extraction, training, or assignment stage is rerun.

Related contracts: [prompt activations](../reference/prompt-activations.md),
[KMeans+PCA](../models/kmeans.md), and [HDDC](../models/mfa-hddc.md).

## Reference density and split

Use the 2,002 validation activations as a common density reference. Reconstruct
the original 90/10 split with `stratified_split(..., val_frac=0.1, seed=42)` and
require its ordered global row IDs to match every saved `val_indices.json`.
Check the streamed IDs as well. The remaining 18,018 rows fitted the models.
Validation also selected HDDC checkpoints; this is not an untouched test set.
Splitting is by the `years` subset, not by year or prompt-template groups.

For a query `x`, find its 20th nearest reference neighbour at Euclidean radius
`r_20(x)`. The ambient kNN density score is

```text
log p_hat(x) = log(20/n) - log V_D - D log r_20(x)
D = 4096; n = 2002; V_D = pi^(D/2) / Gamma(D/2 + 1)
```

There is no PCA projection, whitening, standardization, or radius floor.
Zero or nonfinite kNN radii fail explicitly. This is a local support surrogate;
ambient density is statistically delicate for data near a low-dimensional
manifold, even when arithmetic is numerically stable.

## Direction and scores

KMeans uses the saved unit PC1 and leading empirical ML covariance eigenvalue.
It does not use the KNN directions saved for MFA initialization. HDDC uses an
SVD of its effective loadings and adds the isotropic noise variance to the
leading squared singular value. Inactive components, isotropic HDDC
components, invalid KMeans PCA, nonpositive variance, and numerically tied
leading eigenvalues are excluded with explicit reasons.

For unit direction `v`, use `a = Phi^-1(0.9) sqrt(lambda_1)` and `x(t)=mu+t v`
for `t` in `[-a,a]`. This is the central 80% **one-dimensional marginal**
interval. Random directions retain the PC1 interval length.

```text
A(v) = (1/(2a)) integral[-a,a] p_hat(mu+t v)/p_hat(mu) dt
E    = A(PC1) / mean_r A(u_r)
```

Generate 32 isotropic unit directions using float64 CPU Gaussian vectors,
normalized to unit length, with seed `42 + component index`. Reuse these seeds
across configurations. A 201-point pilot profile, including the center, locates
candidate density peaks. Bounded searches refine their locations, and adaptive
QUADPACK integration splits around these peaks, with requested relative
tolerance `1e-4`. Failed integration or estimated relative error above `1e-4`
raises explicitly. Density integrands are rescaled before integration to avoid
overflow. Use logsumexp for the arithmetic mean across random directions.
Mean log density is a different quantity.

Trapezoidal scores on nested 101/201-point grids remain diagnostics. The first
real-data smoke showed substantial changes between them, motivating adaptive
integration. Grid estimates do not supply the final scores.

Natural-log retention and enrichment are canonical outputs. Ordinary-space
scores are additionally reported when representable. Center log density and
20th-neighbour radius remain diagnostics. `A` can exceed one if the component
center is less supported than other points on the interval.

## Run and outputs

From the repository root:

```bash
PYTHONPATH=src .venv/bin/python scripts/temporary/run_years_pc1_density.py \
  --output outputs/experiments/years_layer04_pc1_density/my_run
```

The default activation path is the existing readable scratch location under
`/orfeo/scratch/dssc/zenocosini/dalg-cache/`. Override `--activation-dir` when
needed; inputs are never modified. `--smoke` evaluates two components from one
run and marks the output partial. Output directories must be new.

The temporary [Slurm launcher](../../scripts/slurm/temporary/sbatch_years_pc1_density.sh)
runs the full CPU evaluation and a denser-pilot integration check. Create
`logs/jobs` before submitting it. The check recomputes the component with the
largest adaptive-versus-uniform discrepancy per method/K, using 401 pilot
positions, and requires every direction's log score and log enrichment to agree
within `0.001`. Its result is saved in `integration_validation.json`.

Outputs include a manifest with source/checkpoint hashes and reference row IDs,
per-component CSV, equal-component configuration summaries, per-run tensors
with full direction profiles, two diagnostic figures, and a Markdown report.
`COMPLETED.json` distinguishes full completion from a smoke run.

## Interpretation and diagnostics

- Grid flags mark a natural-log change greater than 0.05 between 101/201-point
  retention or enrichment estimates, or between adaptive and 201-point
  retention. They describe the inadequacy of uniform grids, rather than a
  failure of the adaptive integral. Adaptive error estimates are conditional on
  finding all important peaks; a denser-pilot stability check remains useful.
- The center's contribution to each **201-point trapezoidal** result is
  recorded. A large contribution can indicate an unresolved narrow peak in
  those diagnostic profiles.
- Small radius changes are amplified by the ambient exponent 4096. Center
  density, radius, relative leading eigengap, interval length, and profiles
  should accompany scores.
- Random ambient directions are easy controls in high dimensions. Enrichment
  is descriptive, and exceedance fractions are not calibrated p-values.
- KMeans rank thresholds retain the same PC1 geometry at fixed `K`. The runner
  checks centers, eigenvalues, and PC1 agreement; threshold variants are not
  independent replicates.
- The reference mixes year and template variation. Density support alone
  cannot establish alignment specifically with the year variable.
- A straight tangent can leave a curved manifold over a long interval. The
  metric therefore depends on the component's spatial scale even after length
  normalization. Equal-component summaries do not weight by cluster mass.

Numerical checks live in `tests/test_pc1_density.py`: explicit-query distances,
Gaussian integrals, log-space averages, covariance/noise semantics, and invalid
geometry contracts.

## First evaluated pilot

Slurm job `1857030` completed all 16 configurations and 1,200 eligible components
on 2026-10-28. The four denser-pilot checks passed; the maximum change in any
direction's natural-log score was below `9.1e-12`. There were 1,067 uniform-grid
diagnostic flags, reinforcing the need for adaptive integration. Eight numerical
tests and independent code/setup reviews passed.

The [local report](../../outputs/experiments/years_layer04_pc1_density/job_1857030/report.md)
contains configuration summaries and figures, with
[integration validation](../../outputs/experiments/years_layer04_pc1_density/job_1857030/integration_validation.json)
and per-component diagnostics beside it. PC1 generally exceeds the matched
random baseline, but some components have enrichment below one. Higher median
HDDC scores in this pilot do not establish better tangent alignment: component
centers/scales differ, and ambient density amplifies small radius changes.

## Global PCA75 variant

The projection variant retains the same years/layer-4 scope, all 16 saved
models, and the exact original split. Its outputs live under
`outputs/experiments/years_layer04_pc1_density_pca75/`; the ambient artifacts
above are preserved.

Fit **one** global PCA on the 18,018 training activations, centered without
standardization or whitening. Accumulate centered covariance in float64 and
use its full symmetric eigendecomposition. Choose the smallest `d` such that
the cumulative variance is at least 75%; this is not an intrinsic-dimension
estimate. The saved artifact contains the mean, row-oriented orthonormal basis
`P` of shape `(d,4096)`, complete spectrum and cumulative fractions, exact
training/validation row IDs, extraction/source fingerprints, and hashes of
every layer-4 shard. All 16 configurations and the numerical validator reuse
this same artifact.

The first artifact, `global_pca_training_v1.pt`, selects **d=141**. It retains
**75.09623332%** of training variance; 140 dimensions retain 74.98057038%.
Project the 2,002 validation activations with this training mean and basis.
The common kNN estimator keeps `k=20`, now using Euclidean distances, the
ball volume, and the density exponent in **141 dimensions**.

Keep the original unit covariance PC1 `v` and eigenvalue `lambda`, including
HDDC noise. With original mean `mu` and global training mean `m`, define

```text
a = Phi^-1(0.9) sqrt(lambda)
z_center = P(mu-m)
b = P v
alpha = ||b||^2
a_projected = a ||b||
```

Evaluate the projected original segment `z_center + t b` for `t` in `[-a,a]`.
For integration this is equivalently a unit direction `b/||b||` over
`[-a_projected,a_projected]`; the original PC1 and eigenvalue remain saved
separately. No projected cluster covariance PC1 is recomputed. A projected
norm at most `1e-12` is explicitly excluded as `null_pc1_projection`.

Generate 32 isotropic **unit directions directly in d dimensions** and use the
same projected center and `a_projected` for every baseline direction. Projecting
ambient random directions would give different lengths and is not used.
Seed `42 + component index` remains common across configurations.

Normalize the density integral by its projected segment length and by center
density. By change of variables this equals averaging the projected density
over the original `t` interval. Enrichment uses the arithmetic mean random
retention. The original eigenvalue, original/projected half-lengths, `alpha`,
center log density and 20th-neighbour radius are component diagnostics.
The manifest and figures state actual `d` and retained global variance;
`alpha` describes each original direction, not global variance retention.

Run the PCA fitting stage once, then run the metric using that saved artifact:

```bash
sbatch scripts/slurm/temporary/sbatch_fit_years_density_pca.sh
# After PCA completion:
sbatch scripts/slurm/temporary/sbatch_years_pc1_density_pca75.sh
```

The metric runner's `--global-pca <artifact>` selects this variant; omission
retains the ambient experiment. Existing output files/directories are never
overwritten. The adaptive integration and denser-pilot validation are shared;
the latter reloads the saved PCA and original projected segment lengths.
Targeted tests cover minimal variance selection, shifted means, original
endpoint equivalence, projected lengths, alpha/null projections, reduced-space
unit baselines, and density volumes in the correct dimension.

Interpret scores as density support of the **projected original PC1**.
Discarded dimensions are invisible and the retained coordinates still contain
prompt/template variation. Projection can hide deviations from the manifold;
alpha should accompany comparisons. Scores from different density spaces have
different estimators and should not be compared by magnitude as a quality gain.

### Evaluated PCA75 pilot

PCA fitting job `1857063` and evaluation job `1857111` completed on 2026-10-28.
All 16 configurations and 1,200 components were eligible. Four denser-pilot
checks passed, with maximum natural-log score change `1.56e-6`; the 60 grid
flags concern uniform-grid diagnostics. Fourteen numerical tests and independent
code/setup reviews passed. Alpha spans 0.34562–0.95967.

At rank threshold 0.1, the configuration summaries are:

| K | HDDC retention | KMeans retention | HDDC enrichment | KMeans enrichment |
| --- | ---: | ---: | ---: | ---: |
| 50 | 60.33 | 7.12 | 257.73 | 23.54 |
| 100 | 2.40 | 1.40 | 9.83 | 4.25 |

These are exponentiated medians of natural-log component scores, with equal
component weights. HDDC has higher median retention across all tested thresholds
in this projected-density pilot. Centers, scales, and alpha differ across models;
the result does not establish an advantage in the full original geometry.

The [PCA75 report](../../outputs/experiments/years_layer04_pc1_density_pca75/job_1857111/report.md)
includes all configurations, alpha and center diagnostics, and figures; its
[numerical checks](../../outputs/experiments/years_layer04_pc1_density_pca75/job_1857111/integration_validation.json)
are saved beside the component CSV and original/projected geometry tensors.

## Paired direction-swap control

The follow-up isolates the contribution of direction at a fixed center and
length. It reuses all 16 PCA75 geometry bundles, the shared training PCA and
the same 2,002 validation rows. No model or PCA is refitted. Source checkpoint,
configuration, split, PCA and geometry-bundle fingerprints are checked or
recorded; projected geometry is reconstructed and checked against the saved
projection.

For each of the eight `(K, threshold)` comparisons, match HDDC and KMeans
centers bijectively with the Hungarian algorithm, minimizing the sum of
Euclidean distances in the shared PCA space. Matching uses neither density
scores nor directions. All matched pairs enter the primary comparison.

For a matched pair, set `L=min(a_H_projected,a_K_projected)` and evaluate both
projected original unit PC1s at both centers, always on `[-L,L]`. This changes
the interval to a common scale: it is not the original 80% interval for both
components. Record `L/a_H_projected` and `L/a_K_projected`.

```text
delta_H = log A(v_H; center_H, L) - log A(v_K; center_H, L)
delta_K = log A(v_H; center_K, L) - log A(v_K; center_K, L)
paired_delta = (delta_H + delta_K) / 2
```

Positive differences favor the HDDC direction. The center-density denominator
and length normalization cancel exactly within each difference. A hypothetical
shared random baseline at the same center and length would also cancel from
the log-enrichment difference. The original saved enrichment scores have
different baselines/lengths and are not subtracted; no new random directions
are needed. `exp(paired_delta)` is the geometric mean of the two directional
average-density ratios.

Report medians, quartiles and win fractions of the paired differences, both
center-specific results, and all four sign combinations. Hungarian matching
can force nonlocal pairs, so retain distances, distances divided by each
center's `r_20`, nearest-neighbor ranks, and a prespecified mutual-nearest
subset of the Hungarian matches as a sensitivity summary. No score-based
exclusion is applied. Count `abs(paired_delta)<=0.001` as numerically close in
an additional sensitivity diagnostic, keeping the canonical differences.

Use the existing adaptive integration with a 201-position peak-discovery
pilot. Recompute all four integrals with a 401-position pilot for the pair with
the largest adaptive/grid discrepancy and the pair with largest absolute
paired difference in each `(K, threshold)` group, deduplicating selections.
Require changes in every log retention, center-specific difference and paired
difference to remain below `0.001`. Targeted tests check global matching,
common lengths, independent direct-query dense integrals, center cancellation,
direction exchange, eigenvector-sign invariance and identical directions.

```bash
sbatch scripts/slurm/temporary/sbatch_years_pc1_direction_swap.sh
```

The standalone temporary runner is
`scripts/temporary/run_years_pc1_direction_swap.py`; outputs go to a new
directory under `outputs/experiments/years_layer04_pc1_direction_swap/`.
Transporting a PC1 to a different center tests its support there. Even this
control measures projected directional density support, rather than an angle
to a known tangent or alignment specifically with years. Dependent thresholds,
validation checkpoint selection and discarded/template dimensions retain the
limitations above.

### Evaluated direction-swap pilot

Slurm job `1857158` completed all eight model comparisons and 600 matched
pairs on 2026-10-28, using the 16 saved configurations without refitting.
All 19 targeted tests and 16 doubled-pilot checks passed; the maximum
log-retention change was `4.69e-6`. An independent audit verified all paired
formulas, common lengths, bijections and source/bundle fingerprints.

At threshold 0.1:

| K | exp(median paired log ratio), HDDC/KMeans | HDDC wins | Reciprocal-nearest pairs | Ratio in reciprocal subset |
| --- | ---: | ---: | ---: | ---: |
| 50 | 1.850 | 28/50 | 26 | 0.545 |
| 100 | 0.937 | 48/100 | 50 | 0.979 |

At K50, each PC1 wins at its own center in 34/50 pairs. Median center-specific
log ratios are `+2.675` at HDDC centers and `-2.563` at KMeans centers: local
center/direction interaction is substantial. The reciprocal subset reverses
the aggregate K50 advantage. At K100 all four full-set threshold comparisons
have negative median paired differences; reciprocal subsets are closer to a
tie. One K100 pair per threshold is within `0.001` of zero.

This control does not support a robust general advantage of the HDDC
direction alone. The original retention/enrichment results describe each
model's joint center, direction and scale. The control cannot causally
allocate their original gap among these quantities, nor establish tangent
accuracy for either method.

The [direction-swap report](../../outputs/experiments/years_layer04_pc1_direction_swap/job_1857158/report.md)
contains all thresholds, matching diagnostics and the figure; its
[reviewed interpretation](../../outputs/experiments/years_layer04_pc1_direction_swap/job_1857158/interpretation.md)
records the competing readings, numerical checks and limits.

### Final tables and figures

The [final report with formulas](../../outputs/experiments/years_layer04_pc1_density_final/20261028_finale_srun/report.md)
combines the PCA75 and paired-control results across all configurations.
It includes the [combined CSV table](../../outputs/experiments/years_layer04_pc1_density_final/20261028_finale_srun/tabella_finale.csv),
center/absolute-density/length/alpha diagnostics, and four scientific figures
in PNG, PDF and SVG. Distribution bars are component/pair quartiles, not
confidence intervals; threshold variants remain dependent.

The temporary exporter `scripts/temporary/plot_years_pc1_final.py` reads only
the completed CSV/manifest artifacts and verifies summary medians against
component/pair rows. It requires execution on a compute node through `srun`;
the export manifest records the node, job, step and input fingerprints.
No density integral, model or PCA is recomputed for these figures.

```bash
srun --partition=THIN --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=4G \
  --time=00:10:00 .venv/bin/python scripts/temporary/plot_years_pc1_final.py \
  --output outputs/experiments/years_layer04_pc1_density_final/my_export
```

## Temperature at layer 28

The repeat uses the saved `temperature_02d` Llama-3.1-8B last-token extraction,
zero-based layer **28**, and all 16 completed KMeans+PCA/HDDC-EM fits with
`K={50,100}` and thresholds `{0.0001,0.001,0.01,0.1}`. No extraction, model
training or assignment stage is rerun. Every saved validation-row list is
checked against the reconstructed split: **13,500 training and 1,500 validation
rows** from the 15,000-row temperature population.

The shared PCA is fitted anew on these temperature training activations,
centered without standardization or whitening. The minimum dimension reaching
75% variance is **d=3**, retaining **83.20774161%**; two dimensions retain
72.47595175%. The artifact
`outputs/experiments/temperature_layer28_pc1_density_pca75/global_pca_training_v1.pt`
stores the mean/basis/spectrum, exact split IDs and layer-28 shard fingerprints.
It is reused after fitting and never silently overwritten.

Use the same projected original PC1 intervals, covariance/noise conventions,
alpha, k=20 validation density, 32 unit isotropic reduced-space directions,
seed `42+component`, primary normalized retention and arithmetic random
enrichment as above. The volume/exponent use d=3 and the reference size is
1,500. Scores measure support in this projection; their magnitude is not a
quality comparison with years/141D. Baseline difficulty also changes with
dimension and reference population.

The temporary runner is
`scripts/temporary/run_temperature28_pc1_density.py`. The launcher
`scripts/slurm/temporary/sbatch_temperature28_pc1_density_pca75.sh` executes
both tests and numerical analysis through **srun** within a Slurm allocation.
The runner rejects execution outside a compute step. Prior years artifacts
remain unchanged. Exact repeated KMeans evaluation inputs reuse their scores
and profiles across thresholds, with the source run recorded; every source
checkpoint/config/split retains its own fingerprint and component metadata.

### Numerical subdivision refinement

The initial temperature run `1884063` completed the PCA fit, then correctly
stopped when QUADPACK reported roundoff with estimated relative error
`1.5865e-4`, exceeding the required `1e-4`. Direct Euclidean radii agreed with
the polynomial evaluation within relative error `1.47e-15`; the issue was
integration across nonsmooth kNN neighbour switches.

The adaptive integrator now subdivides at **every existing pilot position**,
as well as refined density peaks, instead of 16 initial uniform bins. The
integrand, score, estimator and tolerance are unchanged, and QUADPACK warning
statuses remain errors. A regression fixture checks the failing 3D line against
an independent 50,001-position integral using direct Euclidean distances.
The prior metric source is archived beside the new PCA artifact; the recorded
source hashes and previously stored years scores retain their provenance.

### Evaluated temperature repeat

Job **1884175** completed all 16 configurations on a compute node through
`srun`, reusing two complete bundles from job `1884080`. Resume requires
identical PCA, metric, checkpoint/config/split fingerprints and validation
row order, then checks the projected geometry and complete eligible component
set. Remaining components use six workers with deterministic per-component
seeds. All **15 numerical-core tests and 3 context/resume tests** passed
through `srun`.

There are **1,193 eligible component entries out of 1,200** across the sweep.
Three HDDC entries have isotropic covariance and no unique PC1; four KMeans
entries have invalid cluster PCA. Repeated threshold configurations are
dependent, so these counts are not independent cluster observations.
No null-projection exclusions occurred.

At the **0.1 cutoff for selecting local covariance rank**, the summaries are:

| Method | K | Eligible/total | Retention | Enrichment | Alpha |
| --- | ---: | ---: | ---: | ---: | ---: |
| HDDC-EM | 50 | 50/50 | 0.98593 | 1.17997 | 0.55774 |
| KMeans+PCA | 50 | 50/50 | 1.01819 | 1.05941 | 0.34315 |
| HDDC-EM | 100 | 99/100 | 1.01560 | 1.14778 | 0.53787 |
| KMeans+PCA | 100 | 99/100 | 1.01473 | 1.04198 | 0.27386 |

Score summaries exponentiate the median log score across eligible clusters;
alpha is the ordinary median. HDDC enrichment is greater in all eight
method/K/threshold comparisons. The primary retention score does **not** show
a consistent HDDC advantage: KMeans is higher at K50 for all thresholds;
at K100 the ordering changes with threshold, with nearly equal medians at
0.1. Retention near one means similar average density to the center, not a
uniformly flat density profile. Different centers, interval lengths and
alpha prevent attributing the enrichment gap to direction alone.

All **four 401-position checks** passed; the maximum change in any direction's
log retention was **8.08e-7**. No uniform-grid diagnostic was flagged. The
independent artifact audit checked all source fingerprints, training-only
minimal PCA, original projected lengths, score formulas and summary medians.
Direct Euclidean center queries agreed with all 1,193 saved center log
densities within **2.40e-14**. Both code and experimental-setup reviewers found
no blocking issues.

The [temperature report](../../outputs/experiments/temperature_layer28_pc1_density_pca75/job_1884175/report.md)
contains all 16 summaries, formulas, diagnostics and the
[mean log-density profiles](../../outputs/experiments/temperature_layer28_pc1_density_pca75/job_1884175/mean_profiles.png).
The [artifact audit](../../outputs/experiments/temperature_layer28_pc1_density_pca75/job_1884175/artifact_audit.json)
records the independent checks and compute-node execution provenance.

## Temperature at layer 4

This repeat evaluates the same `temperature_02d` population at zero-based
layer **4**, using all 16 saved fits and the saved 13,500/1,500 training and
validation split. It reuses the metric and numerical integration from the
layer-28 experiment. It fits a new centered, unwhitened training PCA, selecting
the minimum dimension reaching 75% variance; no dimension from layer 28 is
assumed. Original covariance PC1 segments, k=20 density, 32 matched unit
isotropic random directions, seeds and exclusion rules remain as above.

The scoped runner is `scripts/temporary/run_temperature04_pc1_density.py`;
the launcher is
`scripts/slurm/temporary/sbatch_temperature04_pc1_density_pca75.sh`. All tests,
PCA fitting, scoring, validation and plotting execute through **srun**.
Outputs use the new directory
`outputs/experiments/temperature_layer04_pc1_density_pca75/`. Layer-specific
formats, input fingerprints and validation rows prevent reusing the layer-28
PCA or numerical bundles for layer 4.

The training PCA selects **d=4**, retaining **76.36347981%** of variance;
three dimensions retain **72.75358376%**. The artifact
`outputs/experiments/temperature_layer04_pc1_density_pca75/global_pca_training_v1.pt`
records the exact training/validation IDs, mean, basis, spectrum and layer-4
shard fingerprints.

Job **1886682** completed the full pilot and all **19 targeted tests** through
`srun`. There are **1,193 eligible component entries out of 1,200**: three
isotropic HDDC entries and four invalid KMeans PCA entries are excluded.
All four denser-pilot checks passed with maximum log-retention change
**1.81e-6**; no uniform-grid diagnostic was flagged. Code and experimental
setup reviewers found no blocking issues.

At the **0.1 cutoff for selecting local covariance rank**:

| Method | K | Eligible/total | Retention | Enrichment | Alpha |
| --- | ---: | ---: | ---: | ---: | ---: |
| HDDC-EM | 50 | 50/50 | 0.95804 | 1.17468 | 0.14527 |
| KMeans+PCA | 50 | 50/50 | 0.90504 | 1.02699 | 0.10507 |
| HDDC-EM | 100 | 99/100 | 0.98584 | 1.09730 | 0.10627 |
| KMeans+PCA | 100 | 99/100 | 0.95724 | 1.04195 | 0.09916 |

Score summaries exponentiate median log scores across eligible clusters;
alpha is the ordinary median. HDDC has higher retention in **7/8** descriptive
comparisons and higher enrichment in **8/8**. K100 at threshold 0.0001 favors
KMeans on retention; at 0.001 the retention medians are nearly equal. Shared
data and repeated KMeans geometry make these comparisons dependent.

The low alpha values are a material limit: at threshold 0.1 only about
10–15% of the original PC1's squared norm survives projection. Retaining
76.36% of global variance does not preserve the same fraction of every local
direction. Scores concern the projected segments at model-specific centers
and scales, and cannot establish tangent accuracy or superiority over another
layer. Retention below one means average estimated density below the center's
density; it is not a probability of staying on the manifold.

The [layer-4 temperature report](../../outputs/experiments/temperature_layer04_pc1_density_pca75/job_1886682/report.md)
contains all configurations, formulas, diagnostics and the
[mean log-density profiles](../../outputs/experiments/temperature_layer04_pc1_density_pca75/job_1886682/mean_profiles.png).
The independent [artifact audit](../../outputs/experiments/temperature_layer04_pc1_density_pca75/job_1886682/artifact_audit.json)
passed in `srun` job **1886783**, checking all source fingerprints, minimal
training-only PCA, original projected lengths, arithmetic random means and
summary formulas. Validation IDs match layer 28 exactly. Direct Euclidean
queries reproduce all 1,193 center log densities within **2.85e-14**.

## Years at layer 28

This repeat evaluates `years_neutral` at zero-based layer **28**, with all
16 saved KMeans+PCA/HDDC-EM fits. The original 20,020-row last-token population
is split into **18,018 training and 2,002 validation rows**. Saved validation
IDs and stream order are checked for every configuration. No activation
extraction, model training or assignment stage is rerun.

A new global PCA is fitted only on the layer-28 training activations,
centered without standardization or whitening. The minimum dimension reaching
75% variance is **d=28**, retaining **75.67704532%**; 27 dimensions retain
**74.96680586%**. This dimension is selected from training variance, independently
of the layer number. The PCA mean/basis/spectrum, row IDs and source hashes are
stored under `outputs/experiments/years_layer28_pc1_density_pca75/`.

The runner `scripts/temporary/run_years28_pc1_density.py` reuses the existing
metric: original covariance PC1 including HDDC noise, projected original 80%
marginal interval, common k=20 validation density, and 32 isotropic unit random
directions directly in the retained space at the same projected center and
length. Seed `42+component` and adaptive integration remain unchanged. The
volume and density exponent use d=28, with reference size 2,002.

The launcher `scripts/slurm/temporary/sbatch_years28_pc1_density_pca75.sh`
runs tests, PCA fitting, scoring, stability checks and plotting through **srun**
on a CPU compute node. Job **1886802** completed all 16 fits and all **19
targeted tests**. All **1,200 component entries** are eligible, with no exclusions
or uniform-grid flags. The four denser-pilot checks passed with maximum log-score
change **5.39e-7**. Code and experimental setup reviewers found no blocking
issues. Layer-specific formats and fingerprints prevent reusing the
years/layer-4 or temperature PCA and score bundles.

At the **0.1 cutoff for selecting local covariance rank**:

| Method | K | Eligible/total | Retention | Enrichment | Alpha |
| --- | ---: | ---: | ---: | ---: | ---: |
| HDDC-EM | 50 | 50/50 | 0.99685 | 2.25274 | 0.84304 |
| KMeans+PCA | 50 | 50/50 | 0.82588 | 1.51205 | 0.80180 |
| HDDC-EM | 100 | 100/100 | 0.76493 | 1.67557 | 0.82741 |
| KMeans+PCA | 100 | 100/100 | 0.80251 | 1.23350 | 0.70298 |

Score summaries exponentiate median log scores across eligible clusters;
alpha is the ordinary median. The primary retention result at 0.1 is **mixed**:
HDDC is higher at K50, while KMeans is higher at K100. Higher HDDC enrichment
at both cluster counts does not override the K100 retention result; enrichment
compares each PC1 with its own matched random baseline.

Across all cutoffs, HDDC retention is higher in **7/8** descriptive comparisons
and enrichment in **8/8**. The exception on retention is K100 at 0.1. These are
dependent comparisons sharing data and repeated KMeans geometry. No cutoff is
retuned from the density scores, and no calibrated significance claim is made.

At 0.1, median alpha ranges from about 0.70 to 0.84 across groups. Most of the
local PC1 squared norm survives, while discarded coordinates remain invisible.
The scores measure support in this 28D projection at model-specific centers
and scales. They do not establish tangent accuracy, alignment specifically
with years, or a quality ordering against layer 4 based on score magnitude.

The [years/layer-28 report](../../outputs/experiments/years_layer28_pc1_density_pca75/job_1886802/report.md)
contains all configurations, formulas, diagnostics and the
[mean log-density profiles](../../outputs/experiments/years_layer28_pc1_density_pca75/job_1886802/mean_profiles.png).
The independent [artifact audit](../../outputs/experiments/years_layer28_pc1_density_pca75/job_1886802/artifact_audit.json)
passed through `srun` in job **1886889**, checking source fingerprints, minimal
training-only PCA, projected original lengths, arithmetic random means,
summary medians and all eight descriptive comparisons. Validation IDs match
years/layer 4 exactly. Direct Euclidean queries reproduce all 1,200 center
log densities within **1.43e-14**.

## Two-panel overview of the four PCA75 experiments

The [two-panel PDF](../../outputs/experiments/pc1_density_overview/20261102_pca75/riepilogo_due_pannelli.pdf)
summarizes years and temperature at layers 4 and 28 using the established
local-rank cutoff 0.1. The [profile panel](../../outputs/experiments/pc1_density_overview/20261102_pca75/01_pannello_curve.png)
contains eight plots, and the [table panel](../../outputs/experiments/pc1_density_overview/20261102_pca75/02_pannello_tabelle.png)
contains four tables. Their arrangement, PCA dimensions and retained variance
match. Y scales are shared across K50/K100 within an experiment, and independent
between experiments. Bold table entries mark the higher descriptive score
within each method comparison, including the two KMeans retention advantages.

The temporary exporter `scripts/temporary/export_pc1_density_overview.py`
uses the saved profiles and adaptive scores. Job **1886984** ran through
`srun`, verified all 64 configuration medians against 4,800 cluster entries,
and checked all 32 mean curves against previous exports. Both reviewers
approved the code and presentation. The [overview report](../../outputs/experiments/pc1_density_overview/20261102_pca75/report.md)
explains why mean log-profile curves cannot be integrated to recover the
table scores. A [CSV with all 64 configurations](../../outputs/experiments/pc1_density_overview/20261102_pca75/tutte_le_64_configurazioni.csv)
preserves every tested rank cutoff alongside the main table at 0.1.
