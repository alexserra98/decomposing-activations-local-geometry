# YAML Training Pipeline

> **Kind:** Workflow · **Status:** Current · **Use when:** Planning, submitting,
> resuming, inspecting, or reevaluating saved manifest-based runs. **Related:**
> [YAML configuration reference](../reference/training-pipeline-config.md)

This is an experimental wrapper around the existing training and metric CLIs.
It does not change their implementations. One resolved run executes these
stages in order:

```text
optional KMeans/PCA initialization -> training -> model assignments -> configured evaluation
```

Each stage validates its output and writes a completion marker. Re-running the
same manifest resumes training from the existing checkpoint or skips stages
whose artifacts are already valid.

## First smoke run

Inspect the resolved command and Slurm allocation without submitting:

```bash
uv run --locked dalg-run-pipeline submit \
  configs/archived/toy_manifold_tiling_pipeline_smoke.yaml \
  --dry-run
```

Submit the end-to-end pipeline:

```bash
uv run --locked dalg-run-pipeline submit \
  configs/archived/toy_manifold_tiling_pipeline_smoke.yaml
```

The submit command prints the immutable manifest path. Inspect it later with:

```bash
uv run --locked dalg-run-pipeline status \
  --manifest outputs/experiments/<name>/manifest_<hash>.jsonl
```

For a local or interactive allocation, plan and execute one row directly:

```bash
uv run --locked dalg-run-pipeline plan \
  configs/archived/toy_manifold_tiling_pipeline_smoke.yaml
uv run --locked dalg-run-pipeline run --manifest /path/printed/by/plan --index 0
```

## Evaluate saved runs

`evaluate` recomputes toy-manifold metrics and **overwrites each selected run's
`metrics.json` and `EVALUATION_COMPLETED.json`**. It refreshes the experiment
summary CSVs after all eligible evaluations succeed. Models, assignments,
training configuration, saved splits, and original manifests remain unchanged.

Evaluate one manifest locally, inside an appropriate allocation:

```bash
uv run --locked dalg-run-pipeline evaluate \
  --manifest /path/to/training_manifest.jsonl \
  --device cpu
```

Combine several manifests and submit Slurm arrays:

```bash
uv run --locked dalg-run-pipeline evaluate \
  --manifest /path/to/manifest_a.jsonl /path/to/manifest_b.jsonl \
  --batch-size 4096 --max-mean-to-manifold-distance none \
  --submit --dry-run
```

Remove `--dry-run` to submit. Dry-run lists eligible and skipped runs, effective
settings, original report paths, summary directories, and Slurm commands without
writing files or submitting jobs. Omit `--submit` for sequential local execution.
There is no `--output-dir`: results belong to the original runs.

All rows are selected by default. With one manifest, `--indices 0 2` selects
original rows. With several manifests, `--indices 0:2 1:5` selects row 2 of the
first manifest and row 5 of the second; both indices are zero-based. References
to the same resolved source run directory are evaluated once per invocation,
retaining all source references. Conflicting identities or effective evaluation
settings for the same directory are errors.

Settings default to each manifest row. Override them with `--device`,
`--batch-size`, `--rank-threshold`, or `--max-mean-to-manifold-distance` (`none`
removes the cutoff). An originally disabled evaluation stage is allowed;
unspecified evaluator kinds default to `toy_manifold_tiling`. HDDC and KMeans
continue to use saved ranks regardless of the rank-threshold option. Reports
retain the source `run_id` and `identity_hash`, with effective settings in
`evaluation_config` and manifest references in `evaluation_provenance`.

Held-out coverage defaults to enabled, including for older manifests that omit
the setting. It requires an independent `<dataset>/test/` population of at most
100,000 points. Missing test data raises an error with a command using
`scripts/temporary/add_toy_manifold_test_split.py`. Use
`--no-heldout-distribution-coverage` to reevaluate without coverage, or
`--heldout-distribution-coverage` to override a disabled source setting.
These flags leave the original manifest unchanged. Coverage errors preserve
the previous report. See the
[coverage contract](../evaluation/heldout-distribution-coverage.md#pipeline-integration)
for split isolation, the full empirical curve, and the size guard.

**Every invocation reevaluates**, including runs with valid existing metrics.
Obsolete or malformed metric reports and evaluation markers are replaced after
the new metrics pass validation. If evaluation or validation fails, that run's
previous report and marker stay intact. Updates are atomic per file; earlier
successful runs remain updated if a later run fails. Summary CSVs are refreshed
only after all eligible runs succeed.

The command refreshes `general_metrics.csv` and `manifold_metrics.csv` in every
ancestor directory of an evaluated run that already contains either summary.
If no such directory exists, it creates both summaries in the run's parent
experiment directory. Each refresh aggregates the whole directory, preserving
rows for unselected runs and existing reports from skipped runs. Multiple
manifests can refresh several experiment directories. The dry-run output shows
these destinations before execution.

Missing models, assignments, saved configurations, or split files are skipped
and reported. Shard-only models are unsupported: the evaluator requires a
consolidated checkpoint. Invalid model/assignment artifacts, identity mismatches,
and incompatible datasets cause an error. No eligible runs is also an error;
a non-dry invocation still saves the exclusion report.

Each invocation automatically saves an immutable job plan and `skipped_runs.json`
under `outputs/evaluations/<job-id>/`. Slurm logs also live there. This directory
contains job bookkeeping, not copies of model reports. The printed plan records
sources, effective settings, and summary destinations, so Slurm workers share
one fixed request. Source models and assignments must remain unchanged while
jobs are running; the plan records paths and configuration, not artifact copies.

Slurm arrays are grouped by effective resources. By default each worker uses one
node, one task, and one GPU for CUDA or zero GPUs for CPU. Other resource settings
come from the source manifest. `--resources /path/to/resources.yaml` accepts a
flat mapping of existing resource keys, for example:

```yaml
partition: EPYC
account: LADE
cpus_per_task: 8
memory: 32G
time: '02:00:00'
max_parallel: 4
```

Choose a partition compatible with the requested device. Duplicate references
with different resource settings use the first reference's resources. Nodes and
tasks must remain one. A CPU collection job inherits the first eligible run's
other resources and waits for all evaluation arrays using `afterok` dependencies.
Collection checks that every planned run has a valid report from this invocation
before regenerating summaries; reports from earlier invocations do not satisfy
that check. Local execution performs the same collection.

To retry a failed worker or collection step using its saved plan:

```bash
uv run --locked python -m dalg.evaluation.saved_runs run \
  outputs/evaluations/<job-id>/evaluation_plan.json --index 0
uv run --locked python -m dalg.evaluation.saved_runs collect \
  outputs/evaluations/<job-id>/evaluation_plan.json
```

Worker indices refer to the deduplicated eligible runs in the printed job plan.
A worker retry recomputes that run. Repeating the top-level `evaluate` command
starts a fresh invocation and recomputes all selected eligible runs. Plans from
the former separate-output workflow cannot be used by the overwrite worker.

## Configuration sections

For every supported YAML field, default, and model-specific constraint, see the
[complete configuration reference](../reference/training-pipeline-config.md).

- `experiment`: a name and model output root.
- `dataset`: an existing activation-shard directory, optional subset suffix,
  and layer. The pipeline never starts extraction implicitly.
- `model` and `training`: arguments accepted by the selected trainer.
  `model.kind` selects `mfa`, `ard`, `hddc`, or `kmeans`. HDDC accepts `q_max`
  as an alias for `rank`. KMeans omits `rank` and selects geometry ranks
  through `surgery_threshold`.
- `assignments`: complete model responsibility assignments. KMeans uses hard
  one-hot memberships. Partial `max_batches` output is not a completed stage.
- `evaluation`: `toy_manifold_tiling` requires toy-manifold shards and reports
  clustering, component use, rank, and tangent geometry. MFA-family models also
  report NLL and BIC; KMeans reports split quantization error instead.
- `resources`: Slurm allocation and maximum array concurrency.

Relative paths are resolved against the repository root. The shard subset can
be written either in `shard_dir` (`path#pile_wikipedia_1M`) or as a separate
`dataset.subset`, but not both.

### KMeans as a model

Choose KMeans+PCA through the same plan, submit, run, and status commands:

```yaml
model:
  kind: kmeans
  K: 100
  surgery_threshold: 0.1  # default; effective rank is learned
training:
  device: cuda
  seed: 0
  val_frac: 0.05
assignments:
  enabled: true
evaluation:
  enabled: true
  kind: toy_manifold_tiling
```

Combine this with the normal `experiment`, `dataset`, and `resources` sections.
KMeans supports one CPU or CUDA process. Defaults are Euclidean KMeans++, 100
iterations, 10 restarts, tolerance `1e-6`, and cluster PCA. Clusters with fewer
than two training points are excluded from geometry metrics only. Omit
`model.rank`; Cattell selects the active directions from all PCs. Its fitting stage
writes `kmeans_model.pt`, `config.json`, and `val_indices.json`. The assignment
stage writes `kmeans_model_assignments.pt`; there is no centroid export.
See [KMeans+PCA](../models/kmeans.md) for the model API and Cattell rank masks.

### Reusing KMeans model initialization

For MFA, ARD, and Adam-based HDDC, omitting both `kmeans_model_path` and
`init_model_path` adds automatic KMeans/PCA initialization. The worker uses the
trainer's exact validation split, subset, prefix dropping, and training seed.
Validation activations are excluded. Initialization uses all training points in
their original dimensions with 100 iterations, 10 restarts, tolerance `1e-6`,
and `rank/q_max` PCs. Reservoir settings have no effect on this path.

`direction_init` defaults to `cluster_pca`. Explicit `random` still saves PCA
but uses random MFA loading directions. Initialization always uses overlapping
KNN neighborhoods around each fixed centroid, without rank selection:

```yaml
initialization:
  pca_neighbors: 64
```

The neighbor count must exceed PC capacity and cannot exceed the training
population. KNN covariance uses float64 accumulation centered on the stored
centroid. The worker materializes training activations in host/device memory
and computes KNN covariance in batches. Sparse hard clusters remain usable.

The generated checkpoint is `<run_dir>/initialization/kmeans_model.pt`.
Initialization version 4 records split counts and a SHA-256 training-row
fingerprint in checkpoint metadata. Planning records settings without fitting;
execution validates the checkpoint, writes `INITIALIZATION_COMPLETED.json`, and
reuses valid artifacts on retry. Invalid artifacts are never overwritten.
Older initialization manifests and KMeans v1 checkpoints require a new plan; there is no implicit
conversion of old centroid bundles.

To reuse a fitted model directly:

```yaml
training:
  kmeans_model_path: dalg-cache/path/to/kmeans_model.pt
  direction_init: cluster_pca
```

Planning validates K, D, and sufficient PC capacity. Trainers read `mu` and the
first requested columns of `W_init` directly, without copying or exporting centroids.
Supplied models are reused as provided; the pipeline cannot establish that an
external fitting population excluded the current validation split. With a
supplied checkpoint, `direction_init: random` remains the default. Loading
scales still start at one when using PCs.

HDDC EM preserves its reservoir initialization and full hard-moment pass; its
centroid-only initialization is saved as a KMeans checkpoint without requiring
PCA. It accepts `kmeans_model_path` or a compatible HDDC `init_model_path`;
omit `direction_init` for EM. A full HDDC warm start does not export centroids.
See the [EM contract](../models/mfa-hddc.md#streamed-full-data-em).

`training.centroids_path` is rejected by new pipeline configurations. The old
temporary toy-centroid builder has been removed. Retained external legacy
workflows are deprecated; saved historical artifacts remain untouched.

## HDDC shared noise and sub-epoch surgery

HDDC has two isotropic-noise modes. `isotropic_psi: true` learns one noise floor
`b_k` per component; `shared_b: true` learns one scalar `b` shared by the whole
mixture. The flags are mutually exclusive, and `shared_b` is supported only by
`single_process` HDDC training. When surgery is enabled, select exactly one of
these modes. During shared-b surgery, the new common noise floor pools the
residual covariance of all components that meet `surgery_min_count`.

The shared-b 20K toy configuration uses:

```yaml
model:
  kind: hddc
  K: 200
  q_max: 16
  shared_b: true
  surgery_every_epochs: 1
  surgery_threshold: 0.01
```

See
[`adaptive_q_toy_20k_hddc_shared_b.yaml`](../../configs/archived/adaptive_q_toy_20k_hddc_shared_b.yaml)
for the complete pipeline configuration.

`surgery_every_epochs` also accepts a fraction strictly between 0 and 1 in
`single_process` mode. The fraction is converted to optimizer-step boundaries
using the resolved number of training batches per epoch. With `S` batches, for
example, `0.5` runs surgery after batch `ceil(S / 2)` and again at the epoch
boundary:

```yaml
model:
  kind: hddc
  shared_b: true
  surgery_every_epochs: 0.5
```

The full example is
[`adaptive_q_toy_20k_hddc_shared_b_surgery_half.yaml`](../../configs/archived/adaptive_q_toy_20k_hddc_shared_b_surgery_half.yaml).
For other fractions, surgery runs after the first optimizer step that crosses
each cadence boundary in global epoch progress; the schedule does not reset at
each epoch. Each surgery performs an additional full E-pass over the training
split, so sub-epoch cadences can materially increase runtime. Values greater
than or equal to 1 must be integers; `0` disables surgery.

## Sweeps

The optional `sweep` mapping is a Cartesian product over fields already present
in the YAML:

```yaml
sweep:
  model.K: [50, 100, 200]
  training.seed: [0, 1, 2]
```

This produces nine manifest rows. Runs with identical `resources` are submitted
as one Slurm array. Different resource mappings are placed in separate arrays.

## Run directory

A completed run contains the normal model outputs plus:

```text
run_spec.json
initialization/kmeans_model.pt   # automatic initialization only
initialization/config.json
INITIALIZATION_COMPLETED.json
TRAINING_COMPLETED.json
mfa_model_assignments.pt         # kmeans_model_assignments.pt for KMeans
ASSIGNMENTS_COMPLETED.json
metrics.json
EVALUATION_COMPLETED.json
PIPELINE_COMPLETED.json
```

KMeans model state is stored in `kmeans_model.pt`; MFA-family state retains its
existing model filenames.

The run directory name includes a short hash of the resolved dataset, model,
training, assignment, and evaluation configuration. An existing `run_spec.json`
must match before the pipeline will resume that directory.

For toy-manifold runs, each Gaussian is associated with its unique nearest
planted manifold by default. Setting
`evaluation.max_mean_to_manifold_distance` to a finite positive number adds an
exact mean-to-manifold distance cutoff. Distance ties are ambiguous and remain
unassociated. `metrics.json` records global association counts and one entry
per planted manifold with associated, assignment-live, and assignment-dead
counts. Rank recovery uses the proximity association; assignments define
clustering, the explicit liveness diagnostic, and augmented BIC's training-only
activity reward. `bic.value` uses the
[augmented BIC contract](../experiments/evaluation/toy-manifold-tiling.md#augmented-bic),
with higher values preferred.

The report also includes held-out distribution coverage by default. Training-only
hard assignments define its live centroids; distances are measured on the
separate `test/` stream. Set `evaluation.heldout_distribution_coverage: false`
to omit this metric. Planning checks the test prerequisite before training.

For a manifold of intrinsic dimension `r_i`, tangent alignment compares its
ground-truth tangent basis with the covariance subspace spanned by exactly
`PC1..PCr_i`. `subspace_overlap` is the mean squared cosine of their principal
angles, while `worst_direction_cosine` is the smallest cosine. Both scores are
sign- and basis-invariant and lie in `[0, 1]`. Tangent directions that occur
only in later PCs do not rescue the score.

Tangent containment separately compares the tangent with `PC1..PCs_k`, where
`s_k` is the saved rank-mask count for HDDC, or the component's thresholded
effective rank for vanilla MFA and ARD. It gives full credit when the tangent
is contained in that possibly larger space. Alignment and full containment are undefined
when `s_k < r_i`, including rank zero; those components are excluded from score
means and included in undefined counts.

`tangent_partial_containment` evaluates only `0 < s_k < r_i`. It measures how
fully the tangent contains the learned PCs: squared principal-angle cosines are
averaged over `s_k`, and the worst-direction cosine is the smallest of those
`s_k` cosines. A learned line inside a 2D tangent scores one on both measures.
Rank zero and ranks at least `r_i` are undefined for this metric. Partial
containment uses the same boundary-eigengap rule as full containment.

The leading subspace is defined when the relative boundary eigengap between
eigenvalues `r_i` and `r_i + 1` exceeds `1e-6`; ties within the retained
subspace are valid. Containment applies the same rule at `s_k` and `s_k + 1`,
except at the full ambient dimension. Non-unique tangent
geometry and an undefined leading subspace are counted as undefined. Global
and per-manifold summaries are unweighted component means over
proximity-associated Gaussians, and an empty valid population has a JSON `null`
mean.

See [Toy-Manifold Tiling Evaluation](../experiments/evaluation/toy-manifold-tiling.md) for
the full association, effective-rank, alignment, and output-schema contract.


## Aggregate saved metrics

Collect the saved reports for an experiment or sweep with:

```bash
uv run --locked dalg-aggregate-metrics /path/to/experiment
```

From an existing environment without refreshing installed console scripts:

```bash
PYTHONPATH=src .venv/bin/python -m dalg.analysis.aggregate_metrics /path/to/experiment
```

The command writes `general_metrics.csv` and `manifold_metrics.csv` directly
inside the supplied folder. Each pipeline model run gets one general row, including runs
that differ only in `K`, `surgery_threshold`, or seed. Each planted manifold gets
one row for that run in the manifold table, including manifolds with zero
associated components. Current KMeans+PCA models use the same one-report, one-row
format as MFA and HDDC. No measurements are recomputed or averaged across
configurations, and no model checkpoints are loaded.

Both tables contain `source_path` relative to the experiment folder, `run_id`,
and `evaluation_id`. Use `evaluation_id` to join the tables: it distinguishes
reports with the same saved run name by using the relative report path.
When a report has no saved run ID, the sibling run specification supplies it,
or the relative report directory is used (`.` for a report at the root).
Manifold rows additionally contain `manifold_id` and the saved manifold identity
fields. Rows are ordered by source path, then integer manifold ID. Identifier
columns come first, followed by the remaining columns in alphabetical order.

Nested metrics become dotted columns such as `nll.validation` and
`rank.mean_learned`. Optional sibling `run_spec.json` fields appear as `config.*`
columns in both tables, with convenient `K`, `model_kind`, and
`surgery_threshold` columns as well. Absent or `null` values remain missing;
remaining lists and empty dictionaries are JSON strings. General metric values
stay in the general table, and manifold metric values stay in the manifold table.

```python
from pathlib import Path
import pandas as pd

experiment = Path("/path/to/experiment")
general = pd.read_csv(experiment / "general_metrics.csv")
manifolds = pd.read_csv(experiment / "manifold_metrics.csv")
selected = manifolds.loc[
    (manifolds["K"] == 100) & (manifolds["surgery_threshold"] == 0.1)
]

# Or obtain the DataFrames directly, without writing CSV files:
from dalg.analysis.aggregate_metrics import aggregate_metrics

general, manifolds = aggregate_metrics(experiment)
```

Discovery includes `metrics.json` at the root and in subdirectories, excluding
`centroids`, `archive`, `archived`, `checkpoints`, `snapshots`, and directory
symlinks below the root. The supplied root itself may be a symlink. Deprecated
centroid evaluations are not read or expanded into threshold sweeps. Runs without
reports contribute no rows. Invalid JSON, conflicting fields, invalid manifold
entries, duplicate manifold IDs, and flattened-column collisions cause an error
before either CSV is written. An experiment without reports is also an error. Rerunning replaces
the two summaries with the current reports; original JSON files remain untouched.
CSVs omit the pandas index; an empty manifold table still includes column headers.
The command prints report and row counts and the two output paths.
