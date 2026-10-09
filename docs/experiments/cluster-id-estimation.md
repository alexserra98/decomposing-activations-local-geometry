# Per-cluster intrinsic dimension across the toy noise sweep

> **Kind:** Experimental workflow · **Status:** Tests passed; smoke test pending · **Use when:**
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
present**. This checks file availability; saved checkpoints and assignment
contents still need an end-to-end smoke test. The test suite reported one
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

## Proposed smoke test (not submitted)

Use the installed `.venv` and the validation manifest to test both ends of the
cluster-count range for each checkpoint implementation. All six selected runs
use `noise_ratio_10` and the agreed estimator settings:

| Manifest index | Model | K | Run suffix |
| --- | --- | --- | --- |
| 1 | MFA | 100 | `0fee1d59` |
| 2 | MFA | 2000 | `2042743e` |
| 14 | HDDC, EM | 100 | `04999e1d` |
| 23 | HDDC, EM | 2000 | `0cf61d1c` |
| 149 | KMeans | 100 | `4b90d591` |
| 158 | KMeans | 2000 | `2e55d444` |

Proposed allocation per task: EPYC, account `lade`, QoS `normal`, one node/task,
four CPUs, 16 GB RAM, 45 minutes, no GPU; array concurrency two. This measures
resource use and checks real saved artifacts before planning the 740-run
production sweep. Expected smoke outputs are under 100 MB, excluding temporary
library caches. Existing result directories are rejected.

From the repository root, after user approval:

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

Inspect every task's exit code, stderr, completion fingerprint, populations,
estimator statuses, elapsed time, and peak memory. Preserve any failed task's
outputs and diagnose it before proposing a retry. The production submission
requires a separate exact plan and explicit authorization under the user's
Orfeo instructions.
