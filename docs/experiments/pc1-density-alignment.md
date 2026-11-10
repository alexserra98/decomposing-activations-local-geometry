# PC1 density support on language-model activations

> **Kind:** Experiment · **Status:** Current pipeline workflow; historical pilot results · **Use when:**
> Evaluating PC1 density support or interpreting the years/temperature pilots.
> **Related:** [Unsupervised metric contract](../evaluation/unsupervised.md#pc1-density-support),
> [pipeline workflow](../workflows/training-pipeline.md#evaluate-saved-runs), and
> [configuration reference](../reference/training-pipeline-config.md#evaluation).

## Question and metric

Does a component's first covariance principal direction pass through supported
activation space, and how does its support compare with random directions at the
same center and length? Normalized density retention is the primary score;
enrichment relative to random directions is secondary. Neither measures an angle
to a known manifold tangent.

The supported models are KMeans+PCA and MFA-EM (HDDC with isotropic noise).
The metric defaults to density in a global PCA space retaining at least 75% of
training variance. Ambient-space density remains available. The
[metric contract](../evaluation/unsupervised.md#pc1-density-support) owns the
formulas, exclusions, original-segment projection, validation reference,
integration settings, and report schema.

## Evaluate existing runs

Use the existing immutable pipeline manifest. Training, assignments, and saved
split artifacts must already exist. Inspect the selected evaluations first:

```bash
uv run --locked dalg-run-pipeline evaluate \
  --manifest /path/to/manifest.jsonl \
  --evaluation-kind unsupervised \
  --pc1-density --pc1-density-space pca75 \
  --no-heldout-distribution-coverage \
  --device cpu --submit --dry-run
```

Remove `--dry-run` to submit the evaluation jobs. Omit `--submit` to run within
an existing interactive allocation. `--indices 0 3` selects source manifest rows;
multiple manifests use `--indices 0:0 1:3`. Evaluation replaces each selected
run's `metrics.json` and refreshes experiment CSV summaries. Source manifests,
checkpoints, and assignments retain their existing identities.

The example disables held-out coverage because PC1 density uses the saved
validation split. To compute coverage as well, replace
`--no-heldout-distribution-coverage` with
`--heldout-distribution-coverage --test-shard-dir /path/to/independent_test_shards`.
Neighboring-tile alignment follows the source settings and is enabled by default;
`--no-tile-alignment` disables it for this invocation.

Select the original density space with `--pc1-density-space ambient`. Optional
settings are `--pc1-density-neighbors 20`,
`--pc1-density-random-directions 32`, and `--pc1-density-seed 42`.
Keep these settings and the saved data split identical across model comparisons.
Runs with different training partitions fit different global PCAs.

## Include the metric in a new pipeline

Add this section to a normal training YAML under `configs/experiments/`:

```yaml
assignments:
  enabled: true

evaluation:
  enabled: true
  kind: unsupervised
  device: cpu
  heldout_distribution_coverage: false
  pc1_density: true
  pc1_density_space: pca75
  pc1_density_neighbors: 20
  pc1_density_random_directions: 32
  pc1_density_seed: 42
```

Use `model.kind: kmeans`, or `model.kind: hddc` with
`training.fit_method: em` and exactly one of `model.shared_b: true` and
`model.isotropic_psi: true`. The saved validation split must contain at least
20 retained activation points with the settings above. Training-only PCA needs
at least two training points with nonzero variance.

```bash
uv run --locked dalg-run-pipeline submit configs/experiments/my_experiment.yaml --dry-run
uv run --locked dalg-run-pipeline submit configs/experiments/my_experiment.yaml
uv run --locked dalg-aggregate-metrics /path/to/output_root/experiment_name
```

Run aggregation after evaluation finishes. General CSV columns include
`pc1_density.log_retention.mean`, `.median`, the corresponding
`log_enrichment` summaries, projection information, and eligibility counts.
Detailed component diagnostics and exclusions are serialized as lists.
No per-manifold rows are produced for unsupervised evaluation.

## Provenance and differences from the original pilots

The numerical core, regression fixture, and findings originated in commit
`d6c863f40fa4e58a784faaf9151ab8e44d50f1dc` on
`origin/codex/pc1-density-alignment`. This page replaces its dataset-specific
runner and plotting commands with the current pipeline interface. The historical
scripts and generated reports were not imported.

The pilot runners reconstructed fixed splits, verified them against saved row
IDs, reused one saved PCA per dataset/layer, and saved full density profiles and
figures. The pipeline instead reads each run's saved split directly, recomputes
training PCA for that run, and saves scalar diagnostics in `metrics.json`.
It does not run the paired direction-swap control, export the historical plots,
or automatically perform the pilots' additional 401-position stability checks.
The same adaptive integrator is used; its independent numerical tests include
a 3D kNN neighbor-switch regression checked against a dense direct-query integral.

Historical job numbers and results below are retained from the source document;
they were not recomputed during this port. Validation also selected HDDC
checkpoints, so these results do not describe an untouched test population.

## Historical PCA75 pilots

Each pilot used Llama-3.1-8B last-token activations, KMeans+PCA and HDDC-EM,
`K` in `{50, 100}`, and local-rank cutoffs in `{0.0001, 0.001, 0.01, 0.1}`.
The reference density used k=20 and 32 isotropic random directions with seed
`42 + component_id`. PCA was centered without whitening or standardization.

| Dataset | Layer | Training / validation rows | PCA dimension | Retained variance | Evaluation job |
| --- | ---: | ---: | ---: | ---: | --- |
| Years | 4 | 18,018 / 2,002 | 141 | 75.09623332% | 1857111 |
| Temperature | 28 | 13,500 / 1,500 | 3 | 83.20774161% | 1884175 |
| Temperature | 4 | 13,500 / 1,500 | 4 | 76.36347981% | 1886682 |
| Years | 28 | 18,018 / 2,002 | 28 | 75.67704532% | 1886802 |

At local-rank cutoff 0.1, scores below exponentiate the median natural-log
component score, with equal component weights:

| Dataset / layer | K | HDDC retention | KMeans retention | HDDC enrichment | KMeans enrichment |
| --- | ---: | ---: | ---: | ---: | ---: |
| Years / 4 | 50 | 60.33 | 7.12 | 257.73 | 23.54 |
| Years / 4 | 100 | 2.40 | 1.40 | 9.83 | 4.25 |
| Temperature / 28 | 50 | 0.98593 | 1.01819 | 1.17997 | 1.05941 |
| Temperature / 28 | 100 | 1.01560 | 1.01473 | 1.14778 | 1.04198 |
| Temperature / 4 | 50 | 0.95804 | 0.90504 | 1.17468 | 1.02699 |
| Temperature / 4 | 100 | 0.98584 | 0.95724 | 1.09730 | 1.04195 |
| Years / 28 | 50 | 0.99685 | 0.82588 | 2.25274 | 1.51205 |
| Years / 28 | 100 | 0.76493 | 0.80251 | 1.67557 | 1.23350 |

Temperature/layer-28 retention did not show a consistent HDDC advantage.
Temperature/layer-4 HDDC retention was higher in 7/8 comparisons across
cutoffs, but only about 10–15% of the local PC1 squared norm survived projection
at cutoff 0.1. Years/layer-28 also had a mixed retention result at cutoff 0.1:
HDDC was higher at K50 and KMeans at K100. Higher enrichment does not override
these primary-retention comparisons.

Each years sweep had 1,200 eligible component entries. Each temperature sweep
had 1,193: three isotropic HDDC entries and four invalid KMeans PCA entries
were excluded. These are entries across dependent threshold configurations,
not independent cluster observations.

## Historical paired direction-swap control

The years/layer-4 control (job 1857158) matched HDDC and KMeans centers
bijectively with the Hungarian algorithm in the shared PCA75 space. For each
pair it evaluated both original projected PC1s at both centers using the same
half-length, the minimum of their projected interval half-lengths. This isolated
direction comparisons at fixed center and scale, without random baselines.

At cutoff 0.1, the exponentiated median paired HDDC/KMeans log ratio was 1.850
for K50 and 0.937 for K100. Restricting the Hungarian matches to reciprocal
nearest neighbors changed these to 0.545 and 0.979. At K50, each method's
PC1 won at its own center in 34/50 pairs. The full results did not support a
robust general advantage of the HDDC direction alone.

## Interpretation

Retention above one means average estimated segment density exceeds center
density. Retention below one is not a probability of leaving a manifold.
Different centers, lengths, projection fractions, and density dimensions prevent
attributing raw score gaps to direction alone. Random directions become easier
controls in high dimensions, and discarded directions are invisible after PCA.

Keep alpha, interval lengths, center density, kNN radii, and numerical diagnostics
with the scores. Repeated rank thresholds are dependent, and neither component
quartiles nor win fractions are calibrated significance estimates. The original
ambient pilot (job 1857030) remains historical evidence for the same diagnostic;
its magnitudes are not directly comparable with the PCA75 results.

## Controlled PAk repeat

The follow-up uses DADApy PAk in the same saved PCA75 coordinate spaces,
explicitly fixing its density dimension to the coordinate dimension: years at
layers 4/28 use 141/28, and temperature at layers 4/28 use 4/3. No intrinsic
dimension is estimated. Models, PCA, covariance PC1s, projected interval
lengths, eligibility, 32 random controls and seed `42+component` are reused.
PAk and the kNN k=20 baseline are evaluated on the same replacement references;
historical kNN scores remain separately identified.

### Current reference paths

The scratch root is
`/orfeo/cephfs/scratch/dssc/zenocosini/dalg-cache/sae-manifold/`.
For each `sae_manifold_<dataset>_llama31_8b_last` dataset:

- `pre_holdout/<name>/source/records.jsonl` and its `meta/` preserve the original
  prompt population and order; the archived models and splits remain there.
- `<name>/` holds re-extracted development activations and `<name>/test/` holds
  the complementary test activations. Original prompt IDs, text, labels and
  tokenization allow recovery of the historical validation prompt membership.

The runner restores that validation order in memory, checking the archived
prompt-population fingerprint, extraction settings, complete record identity
including tokenization, and unique coverage of every reference prompt. The
saved historical PCA is applied without refitting. Source shard hashes and
projected-reference hashes are recorded in each output manifest.

These are re-extracted tensors: `original_tensor_identity_verified=false`,
while prompt/tokenization identity is verified. Consequently a comparison to
historical kNN numbers also retains this extraction-provenance difference;
historical scores are not a verified bit-identical estimator-only intervention.
The controlled comparison uses the newly recomputed kNN baseline on exactly
the same replacement reference as PAk. This follows
the reconstruction in `scripts/temporary/rerun_pre_holdout_pc1.py`; no new
activation extraction or training is performed.

### Evaluation and exports

The standalone runner is `scripts/temporary/run_pc1_pak_controlled.py`.
DADApy uses `alpha=1e-6`, `maxk=100`, neither Bonferroni correction, and
an explicit **minimum `k*=4`** for query interpolation. The density dimension
is always fixed; these are neighborhood settings. Query points never enter
the reference population.

The default minimum of three produced unstable near-poles at first-neighbor
switches. Smoke job `1927373` failed even at 12,801 uniform positions, with
log-score changes exceeding one. The diagnostic
`outputs/experiments/pc1_pak_controlled/min_neighbors_check_1927402.json`
compares queries approaching actual neighbor switches: the four-shell fit
has a stable limit while the three-shell estimate continues increasing.
The reported repeat uses the four-shell minimum uniformly for every model,
not a component-specific fallback. It must be labeled as such rather than
presented as unmodified default PAk interpolation.

Batched adaptive Simpson quadrature explicitly splits at every nearest-neighbor
switch along the segment (the lower envelope of affine squared-distance terms),
as well as the 201 pilot positions. Endpoint evaluations resolve switch peaks;
acceptance requires a forced global refinement and summed absolute integration
error at most 1e-4 relative to the integral. Failure at the query budget is fatal.
The worst estimated-error component per method/K is independently recomputed
with 401 pilot positions; all direction log retentions and log enrichment must
agree within 0.001. Diagnostic profiles are saved at 201 positions. Integration
stability is separate from PAk's statistical density uncertainty.

The kNN baseline reuses `KNNLogDensity` and `score_component(adaptive=True)`;
centers, segments and random directions are identical to the PAk comparison.
No intrinsic-dimension PAk or direction-swap experiment is included.

The Slurm launcher is `scripts/slurm/temporary/sbatch_pc1_pak_controlled.sh`.
`--smoke --experiment-index 0` selects two eligible components from the first
years/layer-4 configuration. `--experiment-index 0..3` runs one full sweep;
`--collect-only --output <root>` exports completed sweeps without scoring.
The final `02_pannello_tabelle` PNG/PDF/SVG reuses the original four-table
layout, rows, columns and rank cutoff 0.1, with PAk scores and title.
A companion `tabella_soglia_01.csv` contains the sixteen displayed rows and
the recomputed kNN scores; `knn_baseline_tabelle` exports the same layout for
that baseline. `tutte_le_64_configurazioni.csv` preserves all tested cutoffs.
The root `COMPLETED.json` is written only after all four sweeps and exports.

Smoke job `1927372` found that the shared metadata loader requires activation
shards even for reading the archive. The runner now reads the archive metadata
directly and uses the standard streaming dataset on current shards. Validation
job `1927373` passed all 33 targeted tests, including fixed dimension with ID
estimation forbidden, query-batch independence, analytic Gaussian integrals,
refusal of unverified integrals, and restoration of original validation prompt
order across development/test partitions. Stabilized smoke job `1927423`
passed all 36 numerical tests, including direct-query versus line-polynomial
PAk agreement and an independent narrow-Gaussian integration fixture.
The stabilized smoke completed all eight selected components across the four
experiments. Full array job `1927428` launched the four 16-configuration sweeps;
`1927432` was queued to check component medians and export both table panels
after successful completion. Later failures and retries are recorded below.

```bash
sbatch scripts/slurm/temporary/sbatch_pc1_pak_controlled.sh
# Substitute the returned array job ID:
sbatch --dependency=afterok:<array-job-id> \
  scripts/slurm/temporary/sbatch_collect_pc1_pak_controlled.sh \
  outputs/experiments/pc1_pak_controlled/job_<array-job-id>
```

The original array `1927428` failed after saving 17 complete configurations.
Three sweeps exhausted file descriptors while transferring Torch tensor
storages through the process pool; years/layer 4 also encountered nonfinite
PAk Newton outputs. The obsolete export job `1927432` was canceled.

The worker boundary now sends NumPy arrays rather than shared Torch storages.
PAk's shell likelihood is solved in log space by eliminating its intercept
analytically and bisecting the monotone slope equation. This is the same
likelihood, neighborhood rule and uncertainty formula, without the singular
Newton Hessian. All PAk scores are recomputed; only matched kNN baselines are
reused after checking PCA, reference and protocol fingerprints.

Validation `1927849` passed all 39 tests and the eight-component real-data
smoke. Tests cover 1,500 returned arrays without descriptor accumulation,
analytic exponential shells, public-API agreement to its numerical precision,
and the saved real-query Newton failure fixture. Recovery check `1927854`
reevaluated the complete failing component at 201/401 pilot positions; maximum
direction-log change was `3.03e-5`, below `0.001`.

Array `1927867` completed and validated years/layer 28 and temperature at
layers 4/28 (16 configurations each). Years/layer 4 saved four configurations
before its Simpson integral hit floating-point interval resolution. The export
job `1927871` was canceled because its failed dependency could never succeed.

### Years/layer-4 quadrature retry

Only years/layer 4 was rerun in job `1928218` (array index 0). Its open Gauss
8/16-point quadrature retains the same `1e-4` relative-error criterion, a forced
global refinement, the original pilot positions and exact neighbor switches.
Geometric padding from `1e-2` through `1e-12` around switches resolves narrow
peaks on both sides; open nodes avoid assigning mass to isolated split-point
values. The estimator, four-neighbor minimum, reference population, centers,
segments and random controls are unchanged. All years/layer-4 PAk scores are
recomputed; compatible kNN baseline scores are reused.

Recovery job `1928212` passed all 40 tests, including the narrow-peak and
isolated-split-value regression fixtures. The exact failing K100 component 4
also passed the 201/401-pilot check: maximum direction-log change `5.28e-7`,
enrichment-log change `2.02e-10`, maximum estimated relative error `8.61e-5`.

Result root: `outputs/experiments/pc1_pak_controlled/job_1928218/`. Its three
other dataset/layer directories link to the completed, validated results from
`job_1927867`; they are not recomputed. Export job `1928219` was queued after the retry; it was canceled after
the failed dependency. Final four-case tables require the root `COMPLETED.json`.

### Final PAk status and available tables

Years/layer 4 did not converge. Retry `1928218` exceeded the 200,000-query
budget at component 27, direction 10. Retry `1928494` reused four completed
configurations under the same integration protocol and raised the budget to
one million queries without changing the 1e-4 tolerance. It failed at component
27, direction 21 after 1,066,187 queries, with estimated relative error about
3.27e-4. Export `1928516` remained blocked by that failure. No completed
four-case PAk result is claimed.

The three validated cases remain at
`outputs/experiments/pc1_pak_controlled/job_1927867/`: Years/layer 28 and
Temperature/layers 4/28, each with 16 configurations and a passing independent
401-position integration check. `scripts/temporary/render_pc1_pak_three_tables.py`
renders only these cases in the original layout, at rank threshold 0.1.
PNG/PDF/SVG and the selected CSV are under
`outputs/experiments/pc1_pak_controlled/three_validated_tables/`.

The subsequent [calibrated kNN support-coverage experiment](pc1-support-coverage-proposal.md)
completed all four cases without density integration. It is a different metric,
with its own disjoint reference/calibration split; it does not complete or
replace the unresolved Years/layer-4 PAk measurement.
