# Training Pipeline YAML Reference

This file documents every field accepted by the experimental YAML training
pipeline. The source of truth is `src/dalg/pipeline.py` together with the three
existing trainer parsers selected by `model.kind`.

YAML keys use the Python/manifest spelling with underscores, such as
`early_stop_patience`, rather than CLI spelling such as
`--early-stop-patience`. Use only the documented keys; unsupported top-level
keys and trainer arguments are rejected during planning. Relative paths are
resolved against the repository root.

## Top-level structure

The only top-level sections are:

| Section | Required | Purpose |
| --- | --- | --- |
| `experiment` | yes | Experiment identity and output location. |
| `dataset` | yes | Existing activation shards and layer. |
| `model` | yes | Trainer selection and model/method parameters. |
| `training` | no | Optimization, stopping, initialization, and logging. |
| `assignments` | no | Post-training MFA responsibility assignments. |
| `evaluation` | no | Optional evaluation built from those assignments. |
| `resources` | no | Slurm allocation and array concurrency. |
| `sweep` | no | Cartesian sweep axes. |

The planner combines `model` and `training` before invoking the selected
trainer. In practice, put structural and method-specific fields in `model` and
run controls in `training`, as shown below. A field may not appear in both
sections. `kind` and the HDDC `q_max` YAML alias belong in `model`.

The pipeline supplies `shard_dir` and `layer` from `dataset`, and derives
`out_dir` from the experiment and run identity. Do not repeat those three CLI
arguments in `model` or `training`; explicitly setting `out_dir` is rejected.

## `experiment`

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `name` | string | required | Experiment name. It contributes to the run ID, manifest location, Slurm job name, and output subdirectory. |
| `output_root` | path | required | Root for model outputs. Relative paths are resolved from the repository root. Each run gets a derived directory below `<output_root>/<experiment-name>/`. |

## `dataset`

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `shard_dir` | path | required | Existing activation-shard root containing `config.json`, `meta/`, and the selected layer directory. Extraction is never started implicitly. |
| `layer` | integer | required | Activation layer. The planner requires `layerNN/` where `NN` is zero padded. |
| `id` | string | shard directory name | Short dataset identifier used in run names. It does not change the loaded data. |
| `subset` | string or `null` | `null` | Optional subset spec such as `pile_wikipedia_1M`. It may instead be appended to `shard_dir` after `#`, but not supplied in both places. |

Example:

```yaml
dataset:
  id: wikipedia_1m
  shard_dir: dalg-cache/pile_gemma2b_activations
  subset: pile_wikipedia_1M
  layer: 17
```

## `model`

### Fields common to all model kinds

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `kind` | string | required | Selects `mfa`, `ard`, or `hddc`. |
| `K` | integer | required | Number of mixture components. The uppercase spelling is required. |
| `rank` | integer | `10` for MFA/HDDC; `64` for ARD | Latent rank. Its exact meaning depends on `kind`. |

For `mfa`, `rank` is the fixed rank of every component. For `ard`, it is the
maximum available rank before ARD shrinkage and optional pruning. For `hddc`, it
is the fixed rank when surgery is disabled and the maximum per-component rank
when surgery is enabled.

### ARD-only fields (`kind: ard`)

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `alpha0` | float | `1.0` | Gamma shape of the ARD precision prior. |
| `b0` | float | `0.0001` | Gamma rate of the ARD precision prior; must be positive. |
| `ard_lambda` | float | `1.0` | ARD penalty multiplier; must be non-negative. The applied weight is `ard_lambda / n_train_tokens`. Use `0` for an unregularized baseline on the ARD stack. |
| `ard_warmup_frac` | float | `0.15` | Fraction of the schedule horizon trained with zero ARD pressure. Must be in `[0, 1]`. |
| `ard_ramp_frac` | float | `0.20` | Fraction of the schedule horizon over which ARD pressure ramps from zero to one. Must be in `[0, 1]`. Warmup plus ramp may not exceed `1`. |
| `ard_schedule_epochs` | integer or `null` | `null` | Epoch horizon for warmup/ramp. Defaults to `epochs`; must be positive when set. Required when `epochs <= 0` and `ard_lambda > 0`. Preserve the stored value when resuming. |
| `prune_at_end` | boolean | `true` | After best-model rollback, zero loading columns below `rank_threshold`. The unpruned model is retained as `mfa_model_unpruned.pt`. |
| `rank_threshold` | float | `1.0` | A column is active when its variance exceeds this multiple of its component's mean unique variance. Must be positive. |

ARD is single-process only. It does not accept `training_mode` and cannot use
component sharding.

### HDDC-only fields (`kind: hddc`)

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `q_max` | integer | `10` | YAML alias for `rank`. Use one of `q_max` or `rank`, never both. |
| `isotropic_psi` | boolean | `false` | Use one isotropic noise value per component. One isotropic mode is required when surgery is enabled. |
| `shared_b` | boolean | `false` | Use one isotropic noise scalar for the full mixture. Mutually exclusive with `isotropic_psi` and supported only in vanilla training mode. |
| `surgery_every_epochs` | float | `0` | Run covariance surgery every N epochs. Positive integers run at epoch boundaries; fractions below 1 run on the first optimizer step crossing each fractional boundary. `0` disables surgery and provides the fixed-rank baseline. |
| `surgery_threshold` | float | `0.01` | Relative Cattell scree threshold. Must be positive when surgery is enabled. |
| `surgery_min_count` | float | `0.0` | Components with fewer effective points are not rewritten. `0` selects `max(5 * q_max, 50)`. |
| `surgery_warmup_steps` | integer | `0` | Linear learning-rate warmup steps after each surgery; `0` disables it. |

When `surgery_every_epochs > 0`, exactly one of `isotropic_psi` or `shared_b`
must be `true`. HDDC surgery materializes a `(K, D, D)` scatter and is intended
for D=128-scale data. `shared_b` cannot be used with component sharding.

## `training`

These arguments are shared by MFA, ARD, and HDDC unless noted otherwise.

### Data loading, validation, and initialization

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `device` | string | `cuda` | Training device, normally `cuda`, `cpu`, or `mps`. Component sharding requires `cuda`. |
| `seed` | integer or `null` | `null` | Training, data-loader, and centroid-initialization seed. Pipeline run naming and assignment seeding fall back to `0` when omitted. |
| `batch_size` | integer | `128` | Training activation batch size. The activation dataset is already batched internally. |
| `num_workers` | integer | `0` | DataLoader worker count. |
| `val_frac` | float | `0.05` | Fraction of selected rows reserved for validation. Set `0` to disable validation; validation-based early stopping then cannot operate. |
| `split_seed` | integer | `42` | Seed for the deterministic stratified train/validation row split. |
| `val_on_gpu` | boolean | `false` | Materialize validation activations on the selected device in single-process training. |
| `centroids_path` | `.pt` path or `null` | `null` | Reuse a precomputed centroid artifact instead of fitting KMeans. It may be a legacy `(K, D)` tensor or a bundle containing `centroids` and `principal_components`. The pipeline accepts only a direct lowercase `.pt` file, resolves it to an absolute path, and validates `K` and `D`. |
| `direction_init` | `random` or `cluster_pca` | `random` | Initialize each component's loading directions randomly, or from the first `rank/q_max` principal components stored in the centroid artifact. |
| `init_model_path` | `.pt` path or `null` | `null` | HDDC only. Seed an epoch-0 training checkpoint from a saved `MFA_HDDC` whose `K`, `D`, `q`, and isotropic-Psi setting exactly match the YAML model configuration. |

When `centroids_path` is set, the trainer copies that artifact into the run
directory and skips centroid fitting. The initialization arguments below have
no effect in that case. `direction_init: cluster_pca` requires
`principal_components` with shape `(K, D, Q_stored)` and fails during planning
when `Q_stored < rank/q_max`. The trainer slices the first requested directions;
loading scales still initialize to 1. Legacy tensor-only artifacts remain valid
with `direction_init: random`.

`init_model_path` and `centroids_path` are mutually exclusive because the full
model already supplies its means. The initial model must exactly match `K`, `D`,
`q_max`, and the isotropic-Psi setting. Only vanilla HDDC training supports this
option. The saved model has no optimizer history, so the epoch-0 checkpoint
starts with a fresh Adam state; subsequent restarts use the normal local
checkpoint exactly.

| Field | Type | Default | Meaning when fitting centroids |
| --- | --- | --- | --- |
| `pool_size` | integer or `null` | `null` | Reservoir size. When omitted, the trainer derives it from the training-token count and `K`. |
| `max_pool_size` | integer | `2000000` | Upper bound used by the automatic reservoir-size calculation. |
| `proj_dim` | integer | `32` | Projection dimension used by reservoir KMeans. |
| `refine_epochs` | integer | `25` | Extra centroid refinement passes with assignments fixed to the nearest centroid. |
| `vocab_size` | integer | `50257` | Vocabulary-size parameter passed to the centroid initializer. |

### Optimization and stopping

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `epochs` | integer | `10` | Maximum number of epochs. A positive value is sufficient by itself; `max_steps` is not required. |
| `lr` | float | `0.001` | Adam learning rate. The trainers do not use weight decay. |
| `grad_clip` | float or `null` | `null` | Gradient-norm clipping threshold. `null` disables clipping. |
| `steps_per_epoch` | integer or `null` | `null` | Optional cap on batches within each epoch. Intended for debug/smoke runs; when set it must be positive. |
| `max_steps` | integer or `null` | `null` | Optional hard cap on total optimizer steps across epochs. Intended for debug/smoke runs, not required for normal epoch-limited training. |
| `epoch_snapshot_every` | integer | `5` | Save a model snapshot at epoch 1 and every N epochs. Set `0` to disable snapshots. |

If `epochs <= 0`, the current trainers require either `max_steps` or validation
with `early_stop_delta > 0`; patience alone does not satisfy this unbounded-run
guard. ARD also requires `ard_schedule_epochs` when its penalty is active.

There are two independent validation-based early-stopping mechanisms:

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `early_stop_delta` | float | `0.001` | Stop when the absolute change between two consecutive validation NLLs is smaller than this value. Set `0` or a negative value to disable only this mechanism. |
| `early_stop_patience` | integer or `null` | `null` | Stop after this many consecutive epochs without a sufficient improvement over the best validation NLL. `null` or a non-positive value disables patience stopping. |
| `early_stop_min_delta` | float | `0.0` | Improvement required to update the best validation NLL and reset patience: `new_nll < best_nll - early_stop_min_delta`. Smaller changes do not reset patience. |

For patience-only stopping:

```yaml
training:
  epochs: 100
  early_stop_delta: 0.0
  early_stop_patience: 10
  early_stop_min_delta: 0.001
```

To disable all early stopping while retaining an epoch cap, set
`early_stop_delta: 0.0` and omit `early_stop_patience`.

### Execution mode and logging

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `training_mode` | `vanilla` or `component_shard` | `vanilla` | MFA/HDDC only. `vanilla` uses one process. `component_shard` shards K across GPUs and requires `resources.gpus > 1` plus `device: cuda`. |
| `compile` | boolean | `false` | Accepted by the current trainer parsers but not currently consumed by their implementations. |
| `wandb` | boolean | `false` | Enable Weights & Biases logging. Only rank 0 logs in component-sharded mode. |
| `wandb_project` | string or `null` | `null` | W&B project name. |
| `wandb_name` | string or `null` | `null` | W&B run name. When `wandb` is enabled and this is omitted, the pipeline supplies its generated run ID. |

For `component_shard`, `resources.gpus` becomes the `torchrun` process count.
This is component/model parallelism over K, not data parallelism. ARD does not
accept this field.

## `assignments`

This stage computes a complete hard assignment using MFA responsibility argmax.
The pipeline deliberately does not expose partial `max_batches` output or the
nearest-centroid assignment mode.

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `enabled` | boolean | `true` | Run assignments after training. |
| `batch_size` | integer | `1024` | Assignment inference batch size. |
| `device` | string | `cuda` | Assignment inference device. |
| `seed` | integer or `null` | `null` | Assignment data-loader seed. When omitted, uses `training.seed`, falling back to `0`. |
| `use_inference_cache` | boolean | `true` | Use the model's inference cache while scoring responsibilities. |

The output is `<run_dir>/mfa_model_assignments.pt`. It must cover the complete
selected canonical activation stream and have cluster sizes summing to the
assignment count before the stage is marked complete.

## `evaluation`

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `enabled` | boolean | `false` | Run evaluation after assignments. |
| `kind` | string or `null` | `null` | The only current evaluator is `adaptive_q_toy`. |
| `batch_size` | integer | `4096` | Batch size used for evaluation NLL. |
| `device` | string | `cuda` | Evaluation device. |

`evaluation.enabled: true` requires `assignments.enabled: true`.
`adaptive_q_toy` additionally requires `model.kind: ard` or `hddc` and shards
created by the toy-manifold shard writer. It produces NLL, clustering-recovery,
live/dead-component, and learned-rank metrics in `<run_dir>/metrics.json`.

## `resources`

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `partition` | string | `H100` | Slurm partition. Use an empty value only when the cluster should choose. |
| `account` | string | `LADE` | Slurm account. Use an empty value only when no account flag is needed. |
| `nodes` | integer | `1` | Slurm node count; must be positive. The current worker is single-node, so keep this at `1`. |
| `ntasks_per_node` | integer | `1` | Slurm tasks per node; must be positive. Keep this at `1`; `torchrun` creates component-shard processes itself. |
| `cpus_per_task` | integer | `8` | CPUs allocated to each array task; must be positive. |
| `gpus` | integer | `1` | GPUs allocated to each run; must be non-negative. Use `0` for CPU execution and more than `1` only with component sharding. |
| `gpu_type` | string | `H100` | Optional GPU type included in Slurm `--gres`. Ignored when `gpus: 0`. |
| `memory` | string | `80G` | Slurm memory request. |
| `time` | string | `23:00:00` | Slurm wall-time request. |
| `max_parallel` | integer | `4` | Maximum simultaneous array tasks for this resource group; must be positive. |

Runs with identical resolved resource mappings share one Slurm array. Runs with
different mappings are written to separate resource-group manifests and arrays.

## `sweep`

Each key is a dotted path to a field that already exists elsewhere in the YAML.
Each value must be a non-empty list. Axes form a Cartesian product; they are not
zipped.

```yaml
model:
  kind: mfa
  K: 100
  rank: 10

training:
  seed: 0
  lr: 0.001

sweep:
  model.K: [100, 200]
  training.seed: [0, 1, 2]
  training.lr: [0.001, 0.0003]
```

This example creates `2 x 3 x 2 = 12` runs. Duplicate resolved run
configurations are rejected. To sweep a field, include a base value for it in
its normal section first.

## Planner-derived fields

These values are recorded in the immutable JSONL manifest but are not YAML
arguments:

- absolute shard, centroid, output, and run-directory paths;
- trainer module and fully defaulted trainer arguments;
- resource defaults;
- assignment and evaluation defaults;
- stable run ID and full identity hash.

Changing any dataset, model, training, assignment, or evaluation field changes
the run identity. Slurm resources do not change the model run identity.
