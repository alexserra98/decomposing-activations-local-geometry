---
name: dalg-run-pipeline
description: Configure, plan, submit, resume, or inspect YAML-defined DALG training, and reevaluate saved runs from one or more manifests. Use for pipeline-based KMeans+PCA, vanilla fixed-q MFA, or adaptive-rank MFA training, with optional assignments and toy-manifold tiling evaluation.
---

# Run the DALG Training Pipeline

For training, use the supplied YAML as the source of truth. For evaluation-only
requests, use existing manifests and the saved-run evaluation workflow below. If the user asks for pipeline
training without an existing config, create a small config under
`configs/experiments/` from the parameters they supplied. This skill
orchestrates existing stages; it does not replace their trainer or analysis
implementations.

Before changing or launching a config, read the relevant sections of
`docs/workflows/training-pipeline.md` and
`docs/reference/training-pipeline-config.md`. Treat
`src/dalg/cli/run_pipeline.py` and `src/dalg/pipeline.py` as authoritative when
documentation differs from code.

## Requested scope

Apply only the stage-control changes needed for the user's requested scope:

- **Full pipeline:** set `assignments.enabled: true` and
  `evaluation.enabled: true`. Use `evaluation.kind: toy_manifold_tiling`, which
  accepts KMeans+PCA, vanilla MFA, ARD, or HDDC but requires shards created by
  `save_toy_manifold_shards`. If the supplied dataset cannot satisfy that
  contract, report the mismatch instead of inventing another evaluator.
- **Training plus assignments:** set `assignments.enabled: true` and
  `evaluation.enabled: false`.

Do not alter dataset, model, optimization, sweep, or resource settings merely
to make the pipeline launch. Missing inputs or incompatible settings are
prerequisites to report. The pipeline never performs activation extraction.

## Evaluate saved models

Use `dalg-run-pipeline evaluate --manifest <one-or-more-manifests>` to recompute
and overwrite saved-run metrics. Read the
[saved-run evaluation workflow](../../../docs/workflows/training-pipeline.md#evaluate-saved-runs)
for selection, overrides, summary destinations, resources, and recovery commands.

- Use `--submit --dry-run` to inspect a Slurm reevaluation, then omit `--dry-run`
  when submission is requested. Use local execution when requested.
- With one manifest, select rows using `--indices 0 2`; with multiple manifests,
  use zero-based qualified indices such as `--indices 0:2 1:5`.
- Every invocation recomputes metrics and replaces the original `metrics.json`
  and evaluation completion marker after validating the new result. No
  `--output-dir` or revision directory is needed. Preserve models, assignments,
  training configuration, splits, and source manifests.
- The command skips and reports missing prerequisites; it never trains or
  generates assignments. Invalid model/assignment artifacts and incompatible
  datasets are errors; obsolete evaluation reports can be replaced.
- Summary collection refreshes existing ancestor CSV directories, or the run's
  parent when no summaries exist. Inspect the destinations printed by dry-run.
  Full-directory aggregation preserves unselected reports in the summaries.
- Report eligible/skipped counts, automatically saved job plan, summary CSV
  paths, and Slurm IDs. Collection follows successful evaluation automatically.
- A failed run preserves its previous report. Repeat the top-level command to
  reevaluate the selected runs, or use the saved-plan worker to retry individual
  failures before collection. Reports must match that job's plan for collection.

## Model selection

- **KMeans+PCA:** use `model.kind: kmeans` and omit `model.rank`. Model geometry
  computes all hard-cluster PCs and always selects Cattell ranks with
  `model.surgery_threshold` (default `0.1`). `W` masks columns beyond each rank.
  Clusters with fewer than two training points are excluded from geometry
  metrics only; invalid components are zero-filled and marked by `pca_valid`. Supports one CPU or CUDA process;
  reject optimizer or component-sharding options.

- **Vanilla MFA:** use `model.kind: mfa`. `model.rank` is the configured column
  capacity of every component. Use `training.training_mode: vanilla` for one
  full model or `component_shard` to shard K across multiple CUDA processes.
- **HDDC adaptive-rank MFA:** use `model.kind: hddc`. `model.q_max` is the
  maximum component rank and covariance surgery selects effective local ranks.
  Use `training.training_mode: single_process` or `component_shard`.
- **ARD adaptive-rank MFA:** the pipeline still supports `model.kind: ard`, but
  use it only when the config explicitly selects the ARD experiment.

The `toy_manifold_tiling` evaluator uses HDDC's saved `rank_mask.sum(-1)`
directly for rank recovery and tangent containment. `evaluation.rank_threshold`
is ignored for HDDC and KMeans; it applies the loading-variance versus noise-floor rule
only to vanilla MFA and ARD. Existing immutable manifests may retain that
legacy field for HDDC; the metrics record `definition: hddc_rank_mask_count`
without an evaluation threshold.

## Workflow

1. Resolve the config path and inspect the YAML, referenced shard directory,
   model kind, output root, stage flags, sweep, and resources.
2. Make the minimal stage-flag edit when the YAML does not match the requested
   scope. Preserve all unrelated config content and user changes.
3. Validate before execution:
   - For Slurm, run `uv run --locked dalg-run-pipeline submit <config> --dry-run`
     and inspect the manifest and generated `sbatch` command.
   - For local or interactive execution, run
     `uv run --locked dalg-run-pipeline plan <config>`, then inspect the printed
     manifest before running a row.
4. Execute only when the user asks to run or submit:
   - Use `submit <config>` for Slurm. When the user says only “run the
     pipeline,” use the config's declared Slurm resources; use local execution
     only when requested.
   - Use `run --manifest <manifest> --index <index>` for a requested local or
     interactive run. A sweep has multiple rows; do not silently choose one
     unless the user identified it.
5. Report the immutable manifest path, submitted job ID or executed row, stage
   scope, and the command for checking status.

## Resume and stage behavior

- New MFA, ARD, and Adam-based HDDC pipelines without `kmeans_model_path` or
  `init_model_path` first run automatic KMeans/KNN-PCA on the exact training split.
  Validation rows are excluded. Loading directions default to `cluster_pca`;
  explicit `random` is respected while PCA is still saved. Supplied artifacts
  and standalone trainer defaults retain their fitting behavior. Older
  initialization manifests and KMeans v1/v2 checkpoints require a new plan.
- Initialization uses all training activations in their original dimensions,
  with 100 KMeans iterations, 10 restarts, tolerance `1e-6`, and `rank/q_max`
  initialization directions in `W_init`, without rank selection. Configure
  `initialization.pca_neighbors` (default 64), requiring capacity < neighbors <=
  training population. Sparse hard clusters are allowed. Reservoir parameters
  have no effect here. Check memory for training points and batched KNN PCA.
- One manifest row executes in fixed order: optional initialization, training,
  optional assignments, optional evaluation. Planning never fits centroids.
- Generated initialization lives under `<run_dir>/initialization/`, records
  split provenance in `kmeans_model.pt`, and is reused after validation on
  retries. Initialization uses manifest version 4 and `dalg_kmeans_v3`
  checkpoints. Status includes initialization completion; invalid existing
  artifacts must not be overwritten.
- There is no `--stage` or `--only` flag. Existing valid artifacts cause their
  stages to be skipped, so rerunning the same manifest resumes at the first
  incomplete stage.
- Reuse the same manifest for retries. Do not regenerate it after changing the
  YAML and present the new run identity as a continuation of the old run.
- The assignments stage writes standard model assignments to
  `<run_dir>/mfa_model_assignments.pt`, or `kmeans_model_assignments.pt` for
  KMeans. Model state lives separately in `kmeans_model.pt`; there is no
  `centroids.pt` export. KMeans completion validates source/model provenance.
- Supplied `training.kmeans_model_path` is read directly for MFA-family
  initialization; trainers use `mu` and optional `W_init`; geometry `W` cannot
  substitute. Pipeline `centroids_path` is deprecated and rejected. HDDC EM preserves reservoir/hard-moment fitting
  and saves a centroid-only KMeans checkpoint when no model is supplied.
- KMeans tiling evaluation uses saved cluster PCs, spectra, ranks, and validity
  masks, and reports split quantization error instead of NLL/BIC. A zero validation fraction is supported;
  the empty split has a null mean quantization error.
- `toy_manifold_tiling` is the pipeline's toy-data evaluator, not the general
  metrics CLI. Use the dedicated metric
  workflow when the user requests Gaussian overlap, intrinsic dimension,
  description metrics, or another standalone metric.
- Normal pipeline execution refuses invalid existing artifacts. Report the
  validation failure and path. The explicit `evaluate` command replaces metric
  reports and evaluation markers; it preserves models, assignments, manifests,
  and `run_spec.json`.
