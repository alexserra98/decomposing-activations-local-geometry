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
