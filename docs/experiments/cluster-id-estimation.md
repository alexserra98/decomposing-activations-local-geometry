# Per-cluster intrinsic dimension across the toy noise sweep

> **Kind:** Experimental workflow · **Status:** Smoke tests passed; production pending · **Use when:**
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
- Require **at least 500 points in that split's cluster**. Keep every component
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
DADApy there. Installation and validation run only through Slurm. The full
experiment has not been submitted.

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

## Proposed production submission (not authorized or submitted)

Evaluate all 740 manifest entries, including all noise conditions, five K
values, and every saved model/configuration variant. Use the same numerical
runner, estimator settings, CPU assignment implementation, batch size 1024,
and installed `.venv` as the successful smoke tests. The numerical runner is
unchanged from `44604bd`; the subsequent documentation and smoke-check changes
are committed on `id_est_exp` before requesting production approval.

Inputs are the immutable recorded paths in
`outputs/experiments/id_est_exp/validation_20261009_02/manifest.json`, rooted at
the agreed scratch results and `assets/` datasets. The proposed new output root
is `/orfeo/cephfs/scratch/dssc/malessi/decomposing-activations-local-geometry/outputs/experiments/id_est_exp/full_20261009_01`.
Existing per-run output directories are rejected, and the source artifacts
remain read-only.

| Stage | Allocation | Mapping and dependency |
| --- | --- | --- |
| ID estimates | EPYC, `lade`, `normal`, one node/task, 4 CPUs, 8 GB, 15 minutes, no GPU | Array `0-739%8`; no upstream dependency |
| Aggregate | EPYC, `lade`, `normal`, one node/task, 1 CPU, 4 GB, 15 minutes, no GPU | One job, `afterok` on the entire ID array; cancel on invalid dependency |

The six observed task times project to roughly 15–45 minutes for the array at
concurrency eight, excluding queueing and aggregation. Allow roughly one hour
and under 2 GB of output as planning estimates. Other noise conditions and
model variants were not timed; the per-task limits provide headroom. Slurm
memory measurements are sampled; 8 GB exceeds the largest observed 1.8 GiB.

Exact commands, to run from the repository root only after explicit approval:

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

Record both job IDs and the current source commit. Monitor the exact array and
dependent job through `squeue`/`sacct` and their stdout/stderr logs. Validate
completion fingerprints, all 740 runs, and combined train/test totals before
reporting completion. Preserve successful and failed outputs; diagnose failed
indices and propose a bounded retry without overwriting them. The production
submission requires explicit authorization under the user's Orfeo instructions.
