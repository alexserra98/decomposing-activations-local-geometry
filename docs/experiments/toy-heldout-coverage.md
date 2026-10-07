# Toy Held-Out Coverage Experiment

> **Kind:** Experimental workflow · **Status:** Experimental · **Use when:**
> Running the strict high-dimensional coverage pilot or its paired-noise visual
> control. **Related:**
> [Held-out distribution coverage](../evaluation/heldout-distribution-coverage.md),
> [toy dataset generator](../reference/toy-manifold-dataset.md), and
> [toy-manifold geometry](evaluation/toy-manifold-tiling.md)

This removable experiment tests whether KMeans+PCA and MFA distribute their
learned region centers over unseen samples from the same manifold distribution.
The primary experiment uses the repository's existing high-dimensional toy
manifold shards. A three-dimensional sphere run is retained only as a visual
control. Coverage is the primary result; exact planted-manifold tangent
geometry remains a separate secondary diagnostic.

The implementation is isolated in:

- `scripts/temporary/run_toy_coverage_experiment.py`;
- `scripts/temporary/run_toy_d128_heldout_coverage.py`;
- `scripts/slurm/temporary/sbatch_toy_heldout_coverage.sh`;
- `scripts/slurm/temporary/sbatch_toy_d128_heldout_coverage.sh`; and
- the reusable metric `src/dalg/evaluation/coverage.py`.

## First high-dimensional test

The first strict held-out test reuses the existing noiseless condition at:

```text
/orfeo/scratch/dssc/zenocosini/dalg-cache/assets/
  toy_manifolds_10types_1each_D128_30Keach_noise_sweep_seed0/noiseless/
```

It contains 300,000 observations in the original ambient space, not a reduced
surrogate. There is one instance of each of ten manifold types: segment,
circle, sphere, torus, Swiss roll, helix, 4D helix, 10D hypersphere, 12D product
torus, and cylinder. The data seed is 0, the ambient dimension is 128, the
instance offset radius is 4, and the condition has no observation noise.

The fixed experiment settings are:

| Setting | Value |
| --- | ---: |
| Source observations | 300,000 |
| Manifold instances | 10, one per type |
| Ambient dimension | 128 |
| Train / validation / test | 70% / 15% / 15% |
| Rows per split | 210,000 / 45,000 / 45,000 |
| Split stratification | manifold type |
| Split seed | 42 |
| Model seed | 42 |
| Components, K | 1,000 |
| KMeans local PCA rank | 32 |
| MFA latent rank, q | 32 |
| KMeans iterations / restarts | 100 / 10 |
| MFA batch size / learning rate | 2,048 / 0.001 |
| MFA epoch cap | 1,000 |
| MFA early-stop patience / min delta | 10 / 0.001 |

KMeans, its per-cluster PCA directions, and MFA are fitted only on the 210,000
training rows. MFA checkpoint selection uses only validation NLL. Coverage is
then computed once on the 45,000 untouched test rows in D=128 using distance to
the nearest training-assignment-live centroid. Exact planted-manifold tangent
geometry was added later as a post-hoc model diagnostic from the unchanged
fitted artifacts; it does not use the test rows or alter the coverage result.

Submit it from the repository root with:

```bash
mkdir -p /orfeo/scratch/dssc/zenocosini/dalg-cache/michele/outputs/experiments/toy_heldout_coverage_d128/noiseless_k1000_q32_seed42/logs
sbatch scripts/slurm/temporary/sbatch_toy_d128_heldout_coverage.sh
```

The log directory must exist before `sbatch`: Slurm opens the `#SBATCH
--output` destination before the job script itself can create directories.

The immutable output root is:

```text
/orfeo/scratch/dssc/zenocosini/dalg-cache/michele/outputs/experiments/
  toy_heldout_coverage_d128/noiseless_k1000_q32_seed42/
```

Its layout is:

```text
experiment_config.json
split/
  split.pt
  split.json
  COMPLETED.json
kmeans_pca/
  centroids.pt
  train_assignments.pt
  config.json
  COMPLETED.json
mfa/
  checkpoint.pt
  mfa_model.pt
  training.json
  train_assignments.pt
  COMPLETED.json
coverage/
  metrics.json
  distances.pt
  COMPLETED.json
geometry/
  metrics.json
  kmeans_pca_variances.pt
  COMPLETED.json
figures/
  coverage_curve.png
  embedding_coverage_2d.png
  projection.pt
logs/
README.md
COMPLETED.json
```

`split.pt` is the canonical partition artifact and its SHA-256 fingerprint is
copied into both model-assignment artifacts and the coverage output. A separate
source SHA-256 covers the config, manifold metadata, activation shards, and row
metadata. The source dataset remains read-only and is not duplicated.
`mfa/checkpoint.pt` supports preemption-safe continuation; completion markers
distinguish resumable state from final artifacts. Coverage records SHA-256
digests of both model files and both training-assignment files, preventing stale
metrics from being silently reused after a model replacement.
`coverage/distances.pt` retains one distance per test row so alternative
summaries can be computed without retraining.

The coverage curve reports shared fixed radii from 0.05 through 20, while mean,
RMS, median, `r90`, `r95`, `r99`, and maximum remain scale-independent summary
columns. The 2D figure uses a global PCA basis fitted only on the training rows,
then projects the same test points and both sets of live centroids. Point color
is the original D=128 coverage distance; the projection is explanatory and is
never used to compute coverage.

### Validation status

A 600-row activation-shard smoke run with `K=8`, `q=2`, two MFA epochs, and a
70/15/15 split completed on CPU. It exercised initial fitting, interrupted-stage
continuation, coverage-only reload, split/model fingerprint checks, both PNG
figures, and all completion markers.

The full D=128 fit completed as Slurm job `1725897` on 2026-09-28 with exit
code zero. MFA selected epoch 230 by validation NLL. Coverage and figures were
recomputed from the unchanged fitted artifacts as job `1726005` after a review
identified cancellation in the matrix-multiplication form of float32 Euclidean
distance. The metric now forces the direct `torch.cdist` kernel; a high-offset,
near-centroid regression test protects this case. A subsequent
`--coverage-only` reload reproduced the corrected result after checking the
source, split, model, and assignment fingerprints. The immutable artifacts are
under the output root documented above.

| Method | Live / total components | Mean distance | RMS distance | `r90` | `r95` | `r99` | Maximum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| KMeans+PCA | 1,000 / 1,000 | 0.243412 | 0.351798 | 0.732517 | 0.813919 | 0.862693 | 0.929402 |
| MFA | 322 / 1,000 | 0.810865 | 0.857335 | 1.098778 | 1.329948 | 1.584762 | 1.867132 |

KMeans+PCA has substantially tighter held-out coverage in this run. For
example, at radius 1.0 it covers every test row, while MFA covers 81.61%; MFA
has empirical maximum distance 1.867132 and first reaches 100% at the next
reported grid radius, 2.0. The clearest accompanying diagnostic is component
utilization: MFA has only 322 assignment-live means,
whereas every KMeans centroid is live. This result therefore motivates
investigating MFA component collapse before treating the coverage difference
as a statement about the model families. It is one noiseless dataset and one
training seed, so it is evidence for the pipeline and a concrete failure case,
not a general comparison.

### Post-hoc tangent geometry

The two canonical tangent diagnostics were computed after training from the
unchanged KMeans+PCA and MFA artifacts. Component means are associated with
their uniquely nearest exact planted manifold. Global scores are unweighted
means over proximity-associated components, including assignment-dead MFA
components; they are not means over only the 322 coverage-live components.

The centroid artifact retains KMeans PCA directions but not their eigenvalues.
The geometry stage therefore reconstructs each direction's variance using only
the fixed training rows and saved hard assignments, then evaluates the shared
toy-manifold geometry contract. This supplies the covariance ordering needed
for matched-dimensional alignment without refitting centroids or directions.

| Method | Alignment overlap | Alignment worst cosine | Containment overlap | Containment worst cosine | Valid / associated |
| --- | ---: | ---: | ---: | ---: | ---: |
| KMeans+PCA | 0.860627 | 0.616068 | 1.000000 | 1.000000 | 1,000 / 1,000 |
| MFA | 0.462287 | 0.173090 | 0.676433 | 0.559262 | 983 / 1,000 |

KMeans+PCA recovers the exact tangent much better in its leading
intrinsic-dimensional subspace. Its rank-32 PCA subspace contains the planted
tangent essentially exactly. MFA improves substantially from matched alignment
to rank-32 containment, showing that some tangent information lies outside its
leading intrinsic-dimensional covariance directions, but it remains below
KMeans+PCA. Seventeen MFA components have undefined scores because their
covariance boundary eigenspace is not identifiable; all belong to the 4D-helix
association. The component-weighted global means are dominated by the
high-dimensional product torus and hypersphere, which receive 429 and 267
associated components respectively, so the per-manifold records in
`geometry/metrics.json` should be used when comparing manifold types.

## Low-dimensional visual control

The earlier control uses one sphere embedded directly in three dimensions.
This makes holes, centroid concentration, and off-manifold displacement visible
without a projection artifact. Two paired conditions use `noise_ratio=1000`
and `noise_ratio=10` with all other generator settings fixed.

The generator's independent noise stream means the two conditions retain the
same noiseless samples, calibration, embedding, row order, and labels. Only the
ambient Gaussian noise changes. The production configuration uses:

| Setting | Value |
| --- | ---: |
| Samples per condition | 60,000 |
| Ambient dimension | 3 |
| Manifold | one sphere |
| Offset radius | 0 |
| Train / validation / test | 70% / 15% / 15% |
| Split seed | 1729 |
| Data and training seed | 0 |
| Components | 64 |
| PCA/MFA rank | 2 |

One stratified split is created before fitting either noise condition. Its
indices and SHA-256 fingerprint are saved in `split.pt` and copied into every
model artifact and metrics report. The same indices are reused for both noise
levels.

## Fit and evaluation protocol

For each noise condition:

1. Fit full-dimensional Euclidean KMeans only on training points.
2. Compute rank-two cluster PCA only from those training assignments.
3. Initialize MFA from the same training-only centroids and PCA directions.
4. Optimize MFA NLL on training points and use validation NLL for early
   stopping and best-checkpoint restoration.
5. Define live KMeans and MFA components from hard training assignments.
6. Compute final point-to-nearest-live-centroid coverage only on test points.
7. Compute the existing exact `tangent_alignment` and `tangent_containment`
   diagnostics against the planted sphere.

KMeans has a fixed $K$, PCA rank, and optimizer contract, so validation is
unused. It remains excluded from the KMeans fit to preserve the same training
population used by MFA.

The primary coverage outputs are mean distance, `r95`, `r99`, maximum distance,
and the coverage curve over the shared radii
`0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0`. Alignment scores must not be used
to select a coverage threshold or checkpoint.

## Run the visual control

Submit the visual-control experiment from the repository root:

```bash
sbatch scripts/slurm/temporary/sbatch_toy_heldout_coverage.sh
```

For a small CPU smoke run:

```bash
PYTHONPATH=src .venv/bin/python \
  scripts/temporary/run_toy_coverage_experiment.py \
  --output-dir /tmp/dalg_toy_coverage_smoke \
  --noise-ratios 1000 10 \
  --samples 600 --calibration-size 256 \
  --K 8 --rank 2 \
  --kmeans-iterations 10 --kmeans-restarts 1 \
  --epochs 4 --early-stop-patience 2 \
  --device cpu
```

The output directory must be absent or empty. The script refuses to overwrite
an existing experiment.

## Visual-control artifacts

The output root contains:

```text
split.pt
experiment_config.json
metrics.json
README.md
coverage_curves.html
noise_ratio_<value>/
  dataset.pt
  manifold_metadata.pt
  kmeans_pca.pt
  mfa_model.pt
  mfa_train_assignments.pt
  metrics.json
  coverage_2d.png
  coverage_3d.html
```

`coverage_curves.html` compares the fraction of test points covered at every
shared radius. Each condition's `coverage_3d.html` places KMeans+PCA and MFA
side by side. Test points are colored by their nearest-live-centroid distance,
live centroids are shown as red diamonds, and the panel title reports mean,
`r95`, `r99`, tangent overlap, and worst-direction cosine. The HTML loads the
Plotly browser library from its public CDN; metric artifacts do not depend on
Plotly.

Each `coverage_2d.png` is a static training-PCA projection of the same held-out
points and live centroids. The two method panels share coordinate limits and a
distance color scale clipped at their larger `r99`, so holes and tail-distance
differences remain directly comparable. Re-render these images from a completed
run without retraining:

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/dalg-toy-coverage-matplotlib \
  .venv/bin/python scripts/temporary/run_toy_coverage_experiment.py \
  --output-dir outputs/experiments/toy_heldout_coverage/sphere_seed0 \
  --render-existing-2d
```

The aggregate `README.md` is the compact comparison table. Coverage columns
come first; tangent alignment is intentionally secondary.

## Visual-control validation status

A two-condition CPU smoke run with 600 points, `K=8`, and four MFA epochs
completed successfully. It produced both dataset/model artifact sets, the
aggregate metrics, the coverage curves, and both 3D comparisons. The reusable
coverage and split contracts are tested in `tests/test_coverage.py`.

The control H100 run completed as Slurm job `1725468` on 2026-09-28 with
exit code zero. Its artifacts are under
`outputs/experiments/toy_heldout_coverage/sphere_seed0/`.

| Noise ratio | Method | Mean distance | `r95` | `r99` | Tangent overlap | Worst cosine |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1000 | KMeans+PCA | 0.168772 | 0.256314 | 0.275449 | 0.999984 | 0.999984 |
| 1000 | MFA | 0.170158 | 0.260245 | 0.283852 | 0.999971 | 0.999971 |
| 10 | KMeans+PCA | 0.196434 | 0.294082 | 0.340592 | 0.992171 | 0.992082 |
| 10 | MFA | 0.209115 | 0.329878 | 0.389043 | 0.997118 | 0.997086 |

All 64 components were training-assignment-live in every condition. On the
nearly noiseless condition the methods have similar coverage and alignment. At
`noise_ratio=10`, KMeans+PCA covers the held-out distribution better by the
mean and tail-distance summaries, while MFA recovers the sphere tangent more
accurately. This controlled run therefore demonstrates that centroid coverage
and local-subspace alignment are distinct properties; it does not support
claiming that MFA has better coverage for this setting.

Artifact validation confirmed the disjoint `42,000 / 9,000 / 9,000` split,
the shared split fingerprint across both methods and noise conditions, complete
training-assignment counts, monotone coverage curves, bounded tangent scores,
and paired noise realizations differing only by their configured scale.

## Scope and removal

The strict D=128 path currently evaluates only the first, noiseless dataset.
Noise conditions and repeated seeds should be added only after this pilot's
split, training, coverage, and visualization artifacts have been inspected.

To remove the experiment, delete the four temporary scripts and this page. The
generic held-out coverage metric and its evaluation contract are independent
and can remain for real activation experiments.
