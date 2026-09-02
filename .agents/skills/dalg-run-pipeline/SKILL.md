---
name: dalg-run-pipeline
description: Configure, plan, submit, resume, or inspect YAML-defined DALG training through immutable manifests. Use for pipeline-based vanilla fixed-q MFA or adaptive-rank MFA training, with optional assignments and toy-manifold tiling evaluation.
---

# Run the DALG Training Pipeline

Use the supplied YAML as the source of truth. If the user asks for pipeline
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
  accepts vanilla MFA, ARD, or HDDC but requires shards created by
  `save_toy_manifold_shards`. If the supplied dataset cannot satisfy that
  contract, report the mismatch instead of inventing another evaluator.
- **Training plus assignments:** set `assignments.enabled: true` and
  `evaluation.enabled: false`.

Do not alter dataset, model, optimization, sweep, or resource settings merely
to make the pipeline launch. Missing inputs or incompatible settings are
prerequisites to report. The pipeline never performs activation extraction.

## Model selection

- **Vanilla MFA:** use `model.kind: mfa`. `model.rank` is the configured column
  capacity of every component. Use `training.training_mode: vanilla` for one
  full model or `component_shard` to shard K across multiple CUDA processes.
- **HDDC adaptive-rank MFA:** use `model.kind: hddc`. `model.q_max` is the
  maximum component rank and covariance surgery selects effective local ranks.
  Use `training.training_mode: single_process` or `component_shard`.
- **ARD adaptive-rank MFA:** the pipeline still supports `model.kind: ard`, but
  use it only when the config explicitly selects the ARD experiment.

Do not infer effective rank solely from the model kind. The
`toy_manifold_tiling` evaluator applies the same loading-variance versus noise
floor threshold to vanilla MFA, ARD, and HDDC, because any of them can make a
loading column effectively inactive.

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

- One manifest row executes in fixed order: training, optional assignments,
  optional evaluation.
- There is no `--stage` or `--only` flag. Existing valid artifacts cause their
  stages to be skipped, so rerunning the same manifest resumes at the first
  incomplete stage.
- Reuse the same manifest for retries. Do not regenerate it after changing the
  YAML and present the new run identity as a continuation of the old run.
- The assignments stage computes complete MFA-responsibility assignments at
  `<run_dir>/mfa_model_assignments.pt`; it is not the pipeline interface for
  arbitrary centroid assignments.
- `toy_manifold_tiling` is the pipeline's toy-data evaluator, not the general
  metrics CLI. Use the dedicated metric
  workflow when the user requests Gaussian overlap, intrinsic dimension,
  description metrics, or another standalone metric.
- Never overwrite an invalid existing model, assignment, evaluation, manifest,
  or `run_spec.json` artifact. Report the validation failure and its path.
