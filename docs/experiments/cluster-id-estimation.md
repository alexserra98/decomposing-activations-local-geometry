# Per-cluster intrinsic dimension across the toy noise sweep

> **Kind:** Experimental workflow · **Status:** Original sweep validated; K=500 repeat in progress · **Use when:**
> Computing separate training/test 2NN and GRIDE estimates for saved toy-noise
> model runs. **Related:** [Toy datasets](../reference/toy-manifold-dataset.md),
> [held-out populations](../evaluation/heldout-distribution-coverage.md), and
> [training pipeline](../workflows/training-pipeline.md).

The experiment lives on `id_est_exp`, starting from `813a7c6`. Its removable
runner is `scripts/temporary/run_cluster_id_experiment.py`; the Slurm entrypoint
is `scripts/slurm/temporary/sbatch_cluster_id_experiment.sh`. Validation uses
`scripts/slurm/temporary/sbatch_cluster_id_validation.sh`.

## Agreed scope

- Include every saved model/configuration run in all four conditions: noiseless
  and noise ratios 1000, 100, and 10.
- Use the readable result root
  `/orfeo/scratch/dssc/zenocosini/dalg-cache/toy_manifolds_10types_1each_D128_30Keach_noise_sweep_seed0`.
- Estimate each cluster separately on training points and independent test
  points. Validation points are excluded from both estimates.
- Keep the fitted model's assignment rule: nearest centroid for KMeans,
  posterior-responsibility argmax for MFA/HDDC. Reuse saved development
  assignments for training and infer test memberships from the saved model.
- The original sweep requires **at least 500 points in that split's cluster**;
  the K=500 repeat below lowers this to 200. Keep every component
  ID, including empty and below-threshold clusters, with its actual population
  and an explicit estimate status.
- Use every assigned point in an eligible cluster, in the original ambient
  coordinates; no subsampling, projection, standardization, or refitting.

The source datasets have 300,000 development points and a separate 30,000-point
supplemental test population. Actual training membership is reconstructed from
each run's `val_indices.json`. With 30,000 test points, at most 60 clusters per
run can satisfy the test population threshold.

## Estimators

Both estimates call [DADApy's public APIs](https://dadapy.readthedocs.io/en/latest/id_estimation.html):

| Estimator | Settings | Stored outputs |
| --- | --- | --- |
| 2NN | `algorithm='base'`, `mu_fraction=0.9`, `data_fraction=1` | ID, reported error, distance scale |
| GRIDE | `range_max=64`; remaining numerical settings use library defaults | Full arrays of IDs, errors, distance scales, and neighbour ranks |

GRIDE ranks are 2, 4, 8, 16, 32, and 64 for eligible production clusters. No
single scale is selected and no mean across scales is substituted for the curve.
With `data_fraction=1`, DADApy reports a 2NN error of zero; this is not a
bootstrap uncertainty estimate.

The two estimators share a DADApy neighbour-distance calculation within each
cluster. Original float32 coordinates are promoted to float64 for distance
arithmetic, without changing their values. Duplicate/zero-neighbour-distance
clusters retain their original populations and receive undefined estimates;
points are not silently deleted. Non-finite outputs and library warnings are
recorded. Unexpected exceptions fail the run rather than producing a completion
marker.

## Planning and output contract

All commands, including planning, tests, and aggregation, must run in a Slurm
allocation on Orfeo. Resource requests, the Python environment, and the output
root are supplied explicitly after review; the launcher has no production
resource defaults.

The runner has three commands:

1. `plan --results-root ROOT --cache-root CACHE --output MANIFEST` records every
   run at `CONDITION/models/FAMILY/RUN`, its input paths, file sizes/timestamps,
   readiness, and the agreed estimator settings. Missing artifacts are reported.
   `--min-population` sets split-specific eligibility (default 500). Supplying
   `--K`, `--surgery-threshold`, and `--models` together restricts the manifest
   to one saved configuration per selected model in all four noise conditions;
   missing requested combinations fail planning. Workers and aggregation read
   the threshold from the manifest, which is also preserved in every result.
2. `run --manifest MANIFEST --index INDEX --output-root OUTPUT --device cpu|cuda`
   processes one recorded run. Slurm arrays pass `INDEX` through the launcher.
   DADApy runs on CPU; `--device` selects the test-assignment device.
3. `aggregate --manifest MANIFEST --output-root OUTPUT` validates completion
   fingerprints and combines all per-cluster rows and per-run eligibility counts.

Each run writes to `OUTPUT/runs/CONDITION/models/FAMILY/RUN/`:

```text
test_assignments.pt
results.json
clusters.csv
COMPLETED.json
```

`results.json` records the run, source fingerprints, train/test populations,
excluded validation count, package versions, source commit, script fingerprint,
Slurm IDs, and one record per component per split. CSV array cells preserve the
full GRIDE arrays as JSON. Aggregate tables go under `OUTPUT/summary/`.
Existing output directories are rejected; original runs and their metrics are
never overwritten. A failed run remains incomplete and its artifacts are
preserved for inspection.

Recorded `/u/dssc/zenocosini/.../dalg-cache/` paths are resolved through the
explicitly supplied readable cache root. Input configs and assignment bundles
are not rewritten.

## Validation status

The user selected the repository's Python 3.12 `.venv` and authorized adding
DADApy there. Installation, validation, and the full experiment ran through
Slurm. Production execution details are recorded below.

Validation job `1747862` failed before installation because pip's home-directory
cache exceeded the user's quota. Retry `1747864` uses scratch for its cache and
the same allocation: EPYC, account `lade`, QoS `normal`, one node/task, four
CPUs, 16 GB RAM, 45 minutes, no GPU. Its output directory is
`outputs/experiments/id_est_exp/validation_20261009_02`.

Job `1747864` completed with exit code zero in 5 minutes 12 seconds and peak
batch memory of 2,423,400 KiB. DADApy installed successfully, all eight tests
passed, and `manifest.json` recorded **740 runs with all required files
present**. This inventory checked file availability; selected checkpoint and
assignment contents were then tested by the smoke runs below. The suite reported one
existing iterable-dataset length warning from its small KMeans fixture.

The validation launcher runs the tests and records a manifest of every source
run. It installs no packages unless `ID_INSTALL_DADAPY=1` is explicitly set in
the reviewed submission. That installation uses DADApy 0.3.4 at Git commit
`c3f7300c9f87fb9e19241f313ae0e023bca687b5`, from the Git dependency declared in
`pyproject.toml`. `ID_EXP_PYTHON` can select an existing alternative environment.
Environment package lists are saved before and after installation. Validation
caches live under the validation output directory; experiment workers use
job-specific temporary caches instead of the home directory.

`tests/test_cluster_id_experiment.py` checks direct DADApy agreement, the
inclusive 500-point threshold, empty/degenerate clusters, saved global-row split
alignment, native KMeans/MFA/HDDC test memberships, and preservation of inputs.

## Completed smoke test

The user approved six runs at both ends of the cluster-count range for each
checkpoint implementation. Array `1747872` used the installed `.venv`, source
commit `44604bd`, and the validation manifest. All six runs used
`noise_ratio_10` and the agreed estimator settings:

| Manifest index | Model | K | Run suffix |
| --- | --- | --- | --- |
| 1 | MFA | 100 | `0fee1d59` |
| 2 | MFA | 2000 | `2042743e` |
| 14 | HDDC, EM | 100 | `04999e1d` |
| 23 | HDDC, EM | 2000 | `0cf61d1c` |
| 149 | KMeans | 100 | `4b90d591` |
| 158 | KMeans | 2000 | `2e55d444` |

Allocation per task: EPYC, account `lade`, QoS `normal`, one node/task, four
CPUs, 16 GB RAM, 45 minutes, no GPU; array concurrency two. All six tasks
completed with exit code zero. Outputs are under
`outputs/experiments/id_est_exp/smoke_20261009_01`.

Executed submission from the repository root:

```bash
sbatch --parsable --job-name=id-est-smoke \
  --partition=EPYC --account=lade --qos=normal \
  --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G --time=00:45:00 \
  --array=1,2,14,23,149,158%2 \
  --output=logs/jobs/id_est_smoke_%A_%a.out \
  --error=logs/jobs/id_est_smoke_%A_%a.err \
  --export=ALL,ID_EXP_PYTHON="$PWD/.venv/bin/python" \
  scripts/slurm/temporary/sbatch_cluster_id_experiment.sh run \
  --manifest "$PWD/outputs/experiments/id_est_exp/validation_20261009_02/manifest.json" \
  --output-root "$PWD/outputs/experiments/id_est_exp/smoke_20261009_01" \
  --device cpu --batch-size 1024
```

| Model | K | Eligible train clusters | Eligible test clusters | Slurm elapsed | MaxRSS, KiB |
| --- | --- | --- | --- | --- | --- |
| MFA | 100 | 40 | 30 | 27 s | 809168 |
| MFA | 2000 | 145 | 4 | 29 s | 1842724 |
| HDDC, EM | 100 | 98 | 6 | 18 s | 680364 |
| HDDC, EM | 2000 | 16 | 0 | 20 s | 1752644 |
| KMeans | 100 | 100 | 13 | 16 s | 685332 |
| KMeans | 2000 | 11 | 0 | 9 s | 757648 |

Follow-up job `1747881`, dependent on `afterok:1747872`, ran
`scripts/slurm/temporary/sbatch_check_cluster_id_smoke.sh` with one CPU, 4 GB,
and ten minutes on EPYC (`lade`, `normal`, no GPU). It verified completion and
source fingerprints, train/validation/test populations, test assignment counts,
the 500-point threshold, every ID and GRIDE scale, and CSV row alignment. It
also exercised aggregation on the six selected runs. All checks passed; every
run retained 270,000 training points, excluded 30,000 validation points, and
assigned 30,000 independent test points. Both estimators succeeded for all
410 eligible training and 53 eligible test clusters, without estimator warnings.

At K=2000 the selected HDDC and KMeans test populations have no cluster with
500 points. Their test rows correctly retain populations and undefined IDs.
The smoke outputs and combined tables occupy about 12.2 MB. Machine-readable
checks and eligibility counts are in `smoke_checks.json` and `summary/runs.csv`.

## Completed production submission

The user approved all 740 manifest entries, including all noise conditions,
five K values, and every saved model/configuration variant. Array `1748327`
and dependent aggregation job `1748336` ran from clean commit `bc12432` on
`id_est_exp`. They used the same numerical runner, estimator settings, CPU
assignment implementation, batch size 1024, and installed `.venv` as the
successful smoke tests. The numerical runner is unchanged from `44604bd`.

Inputs are the immutable recorded paths in
`outputs/experiments/id_est_exp/validation_20261009_02/manifest.json`, rooted at
the agreed scratch results and `assets/` datasets. The output root
is `/orfeo/cephfs/scratch/dssc/malessi/decomposing-activations-local-geometry/outputs/experiments/id_est_exp/full_20261009_01`.
Existing per-run output directories are rejected, and the source artifacts
remain read-only.

| Stage | Allocation | Mapping and dependency |
| --- | --- | --- |
| ID estimates | EPYC, `lade`, `normal`, one node/task, 4 CPUs, 8 GB, 15 minutes, no GPU | Array `0-739%8`; no upstream dependency |
| Aggregate | EPYC, `lade`, `normal`, one node/task, 1 CPU, 4 GB, 15 minutes, no GPU | One job, `afterok` on the entire ID array; cancel on invalid dependency |

Every array task and aggregation completed with exit code zero. The array ran
from 13:04:54 to 13:40:01 on 2026-10-09 (Europe/Rome), about 35 minutes;
aggregation took another 35 seconds. Individual tasks took 9–226 seconds
(mean 22.7 seconds). The largest recorded batch MaxRSS was 1,825,304 KiB,
under 1.8 GiB. Slurm memory measurements are sampled.

Executed commands from the repository root after explicit approval:

```bash
ID_EXP_MANIFEST="$PWD/outputs/experiments/id_est_exp/validation_20261009_02/manifest.json"
ID_EXP_OUTPUT="$PWD/outputs/experiments/id_est_exp/full_20261009_01"
ID_EXP_ARRAY_JOB=$(sbatch --parsable --job-name=id-est-full \
  --partition=EPYC --account=lade --qos=normal \
  --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=8G --time=00:15:00 \
  --array=0-739%8 \
  --output=logs/jobs/id_est_full_%A_%a.out \
  --error=logs/jobs/id_est_full_%A_%a.err \
  --export=ALL,ID_EXP_PYTHON="$PWD/.venv/bin/python" \
  scripts/slurm/temporary/sbatch_cluster_id_experiment.sh run \
  --manifest "$ID_EXP_MANIFEST" --output-root "$ID_EXP_OUTPUT" \
  --device cpu --batch-size 1024)
printf 'ID array: %s\n' "$ID_EXP_ARRAY_JOB"
sbatch --parsable --job-name=id-est-aggregate \
  --partition=EPYC --account=lade --qos=normal \
  --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=4G --time=00:15:00 \
  --dependency="afterok:$ID_EXP_ARRAY_JOB" --kill-on-invalid-dep=yes \
  --output=logs/jobs/id_est_aggregate_%j.out \
  --error=logs/jobs/id_est_aggregate_%j.err \
  --export=ALL,ID_EXP_PYTHON="$PWD/.venv/bin/python" \
  scripts/slurm/temporary/sbatch_cluster_id_experiment.sh aggregate \
  --manifest "$ID_EXP_MANIFEST" --output-root "$ID_EXP_OUTPUT"
```

The output root contains `submission.json`, `slurm_tasks.psv`, and
`slurm_accounting.psv`. These record the source, environment path, exact array
mapping and resources, job IDs, dependency, times, and final task states.

Final validation job `1749225` ran
`scripts/slurm/temporary/sbatch_check_cluster_id_full.sh` on EPYC (`lade`,
`normal`, one node/task, one CPU, 4 GB, 15 minutes, no GPU). It verifies source
and result fingerprints, train/test populations, saved test assignments, every
per-run and aggregated CSV field, eligibility counts, and estimator statuses.
It also reported estimator warnings and unexpected stderr lines without
recomputing the scientific estimates. All checks passed. The report is
`final_checks.json`; the combined tables are `summary/clusters.csv` (335 MiB)
and `summary/runs.csv` (1,480 run/split rows). Total output size is about 1.25 GB.

Every condition has 185 completed model runs: noiseless, noise ratio 1000,
noise ratio 100, and noise ratio 10. The combined cluster table contains
1,139,600 rows, retaining every component ID for both splits.

| Split | Cluster rows | At least 500 points | Valid 2NN and GRIDE | Undefined: zero nearest-neighbour distance | Below 500 points |
| --- | --- | --- | --- | --- | --- |
| Train | 569800 | 98029 | 97848 | 181 | 471771 |
| Test | 569800 | 3637 | 3637 | 0 | 566163 |

All valid GRIDE outputs retain the six neighbour ranks 2, 4, 8, 16, 32, 64.
The 181 undefined training estimates retain their populations and the status
`zero_neighbour_distance`; no points were removed or perturbed. DADApy emitted
its small-distance warning for 1,421 cluster/split records. The library warns
at distances up to `1.01 * float32 epsilon`, so a warning alone does not mean an
exact zero was computed. Warning text is preserved in the result records;
all other eligible estimates were finite. No unexpected stderr issues were
found.

## Notebook: mean estimated ID across noise levels

[toy_manifold_noise_sweep_intrinsic_dimension.ipynb](../../notebooks/toy_manifold_noise_sweep_intrinsic_dimension.ipynb)
reproduces the original noise-sweep notebook's summary table and per-manifold
heatmap using the saved ID estimates. Its first rendering used K=2000,
threshold 0.05, KMeans, HDDC Adam, and HDDC EM with shared noise variance,
across all four noise conditions. The current selectors target the K=500 repeat
below. Other completed configurations can be selected with a matching result
root and manifest; EM with component-specific noise variance has the separate
key `em_bk`. `ID_RESULTS_ROOT` and `ID_MANIFEST_PATH` override the input paths.

Training and independent test appear separately. There is one figure for 2NN
and one for each of the six GRIDE neighbour-rank pairs. The temporary helper
`scripts/temporary/plot_cluster_id_experiment.py` reuses
`_associate_component_means` with the original saved distance cutoff. It projects
only centres eligible in at least one split; no model or ID estimator is refit.
Original per-manifold metrics supply the total associated-component counts.

Within a manifold, means give equal weight to valid clusters meeting the
manifest's minimum population in that split. Summary tables give equal weight to each manifold with
an available mean, and display both contributing manifold and cluster counts.
Missing estimates are undefined, not zero. These populations differ from the
original learned-rank population and can differ between training and test.
Heatmaps print the actual mean ID and its cluster count; colour is the absolute
distance between that mean and the planted noiseless intrinsic dimension, using
the original power scale capped at five. This is a reference comparison, not a
claim that noisy data have that dimension at every neighbourhood scale.

Run the notebook in Slurm using
`scripts/slurm/temporary/sbatch_plot_cluster_id_experiment.sh OUTPUT_DIRECTORY`,
with resources supplied to `sbatch`. The launcher uses the repository `.venv`
and needs `nbclient` and `nbformat`; installation is opt-in via
`ID_PLOT_INSTALL_NOTEBOOK_TOOLS=1`. It runs focused aggregation checks and
saves an executed notebook, two summary-table images, seven train/test figures
in PNG/PDF, numerical CSVs, source fingerprints, source snapshots, and environment
package lists. It rejects an existing output directory. The notebook's original
source remains in the repository branch; generated outputs stay under `outputs/`.

Approved plotting job `1749233` completed with exit code zero in 2 minutes
16 seconds, with batch MaxRSS 932,228 KiB. It used EPYC, account `lade`, QoS
`normal`, one node/task, one CPU, 4 GB RAM, a 30-minute limit, and no GPU.
All four checks passed and all notebook code cells executed successfully.
Outputs are under `outputs/experiments/id_est_exp/figures_20261009_01`: 12
selected runs, 240 manifold/split rows, two summary tables, and seven paired
train/test figures. Visual inspection confirmed labels and annotations are
readable. `nbclient` 0.11.0 and `nbformat` 5.11.1 were added with their schema
dependencies; no previously installed package version changed.

## K=500, threshold=0.2, minimum population 200

On 2026-10-27 the user requested a repeat at K=500, surgery threshold 0.2,
and at least 200 points per split's cluster. HDDC Adam is excluded at the user's
request. The comparison retains KMeans and HDDC EM with shared noise variance,
in all four noise conditions: eight fitted runs. All other estimator, assignment,
train/test isolation, and plotting conventions stay the same.

The result directory is
`outputs/experiments/id_est_exp/k500_tau0p2_min200_20261027_01`;
it contains the new `manifest.json`, per-run results, aggregate tables, and
`figures/`. Original estimates and figures remain in their existing directories.
This reruns ID estimation, including clusters with 200–499 points; changing a
plot filter alone cannot recover these previously skipped estimates.

The existing validation launcher accepts planner options after its output
directory. This repeat uses `--K 500 --surgery-threshold 0.2 --models kmeans em
--min-population 200`. The production checker accepts the old full-sweep output
root as an optional third argument. It checks identical input fingerprints,
populations and test memberships, and verifies previously eligible ID estimates
agree within `rtol=atol=1e-12`, while counting newly eligible clusters.

Validation job `1853543` passed all 19 tests and recorded eight ready runs.
The two warnings are the existing iterable-dataset length warning from the
small KMeans fixture, exercised at both population thresholds. The repeat
uses the existing `.venv`; no packages are installed or updated.
