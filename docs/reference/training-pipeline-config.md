# Training Pipeline YAML Reference

> **Kind:** Reference · **Status:** Current · **Use when:** Looking up an exact
> pipeline YAML field, default, or model-specific constraint. **Related:**
> [YAML training workflow](../workflows/training-pipeline.md)

This file documents every field accepted by the experimental YAML training
pipeline. The source of truth is `src/dalg/pipeline.py` together with the four
trainer parsers selected by `model.kind`.

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
| `experimental` | no | Opt-in experimental trainer behavior. |
| `initialization` | no | KNN neighborhood size for automatic MFA-family initialization. |
| `assignments` | no | Post-training model responsibility assignments. |
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
| `kind` | string | required | Selects `mfa`, `ard`, `hddc`, or `kmeans`. |
| `K` | integer | required | Number of components or KMeans clusters. The uppercase spelling is required. |
| `rank` | integer | `10` for MFA/HDDC; `64` for ARD; omitted for KMeans | Latent rank or maximum adaptive rank. Its exact meaning depends on `kind`. |

For `mfa`, `rank` is the fixed rank of every component. For `ard`, it is the
maximum available rank before ARD shrinkage and optional pruning. For `hddc`, it
is the fixed rank when surgery is disabled and the maximum per-component rank
when surgery is enabled.

### KMeans-only fields (`kind: kmeans`)

Omit `model.rank`: KMeans computes all cluster PCs and always selects effective
ranks using `model.surgery_threshold` in `[0, 1]` (default `0.1`). `W` is
hard-masked beyond each selected rank. Clusters with fewer than two training
members have zero-filled geometry and are excluded from rank and tangent
metrics. Selected ranks cannot exceed `cluster_counts - 1`.
See the [model contract](../models/kmeans.md).

KMeans accepts these `training` fields instead of optimizer options:

| Field | Default | Meaning |
| --- | --- | --- |
| `max_iter`, `restarts`, `tol` | `100`, `10`, `1e-6` | KMeans++/Lloyd fitting controls. |
| `seed` | `0` | Fitting seed. |
| `device`, `training_mode` | `cuda`, `single_process` | One CPU or CUDA process; multi-process allocations are rejected. |
| `val_frac`, `split_seed` | `0.05`, `42` | Canonical row split; fitting and PCA exclude validation. |
| `drop_prefix` | shard config, then `0` | Number of prefix tokens excluded from all stages. |
| `sample_fraction`, `sample_seed` | `1.0`, `0` | Deterministic training-activation subsampling used by both fitting and PCA. Assignments still cover the entire selected stream. |
| `load_batch_size` | `20000` | Activation loading batch size. |
| `block_x`, `block_c` | `8192`, `8192` | Distance computation block sizes. |
| `pca_chunk_elems`, `pca_eig_batch_size` | `8388608`, `256` | PCA scatter and eigensolver batching. |

The standalone worker's `pca_only` operation is not a pipeline option. KMeans
has no `lr`, optimizer epochs, or component sharding. Its fitting stage writes
`kmeans_model.pt`, `config.json`, and `val_indices.json` directly; there is no
separate automatic initialization stage or centroid export.

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
| `shared_b` | boolean | `false` | Use one trainable isotropic noise scalar `b` for every component, so `Psi_k = b I` throughout the mixture. Mutually exclusive with `isotropic_psi` and supported only in `single_process` training mode. |
| `surgery_every_epochs` | float | `0` | Surgery cadence in epochs. Supported values are `0`, a fraction strictly between 0 and 1, or a positive integer. Fractions run on the first optimizer step crossing each global fractional-epoch boundary and are `single_process`-only. `0` disables surgery and provides the fixed-rank baseline. |
| `surgery_threshold` | float | `0.01` | Relative Cattell scree threshold. Must be positive when surgery is enabled. |
| `surgery_min_count` | non-negative float | `0.0` | Gates mean, mixture-weight, and covariance updates by membership mass `N_k` (soft by default, hard counts with `experimental.hard_assignment_covariance`). Components below the threshold retain their means and weights; eligible weights redistribute only their previous total probability mass. `0` includes every positive-mass component; exact-zero mass is always skipped. |
| `surgery_warmup_steps` | integer | `0` | Linear learning-rate warmup steps after each surgery; `0` disables it. |

When `surgery_every_epochs > 0`, exactly one of `isotropic_psi` or `shared_b`
must be `true`. HDDC surgery materializes a `(K, D, D)` scatter and is intended
for D=128-scale data. `shared_b` cannot be used with component sharding. In
shared-b mode, surgery pools the responsibility-weighted residual covariance of
all components meeting `surgery_min_count` to update the single `b`; components
below the threshold are not otherwise rewritten, but they still inherit the new
global noise floor.

The cadence is resolved against `S`, the number of training batches in an epoch
after applying an optional `training.steps_per_epoch` cap:

| Value | Surgery timing |
| --- | --- |
| `0` | Disabled. |
| `1` | At the end of every epoch, after validation. |
| `3` | At the end of epochs 3, 6, 9, and so on. |
| `0.5` | After optimizer step `ceil(S / 2)` and at the end of each epoch. |
| `0.3` | On the first optimizer step crossing every 0.3 epochs of global progress; boundaries continue across epochs rather than resetting. |

Every surgery runs an additional full E-pass over the training split. A cadence
shorter than one optimizer step is rejected. Fractional cadences are supported
only in `single_process` mode; component-sharded HDDC accepts integer cadences.

The two relevant complete examples are
[`adaptive_q_toy_20k_hddc_shared_b.yaml`](../../configs/archived/adaptive_q_toy_20k_hddc_shared_b.yaml)
and
[`adaptive_q_toy_20k_hddc_shared_b_surgery_half.yaml`](../../configs/archived/adaptive_q_toy_20k_hddc_shared_b_surgery_half.yaml).

### Full-data EM

For `model.kind: hddc`, set `training.fit_method: em` and
`model.shared_b: true`. EM updates all parameters after each complete training
pass, including Cattell rank selection using `model.surgery_threshold`.

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `fit_method` | `adam` or `em` | `adam` | HDDC fitting method; EM supports single-process CPU/CUDA shared scalar noise only. |
| `em_component_chunk_size` | positive integer | `32` | Number of components in likelihood and centered-moment workspaces. |
| `em_eig_batch_size` | positive integer | `128` | Covariances per eigensolver call. |
| `em_tol` | non-negative float | `0.00001` | Relative training-NLL tolerance. Stop after three consecutive small changes with unchanged ranks; zero disables convergence stopping. |

EM requires `1 <= q_max < D` and a positive `epochs` M-step limit. Set
`surgery_every_epochs`, `surgery_min_count`, and `surgery_warmup_steps` to zero
(their defaults). It rejects `steps_per_epoch`, `max_steps`, compilation,
gradient clipping, and learning-rate overrides. Every E-pass includes all
selected training rows and final partial batches. `early_stop_patience` and
`early_stop_min_delta` apply to the selection metric; `early_stop_delta` is an
Adam-only control.

Fresh EM fits recompute all parameters from a full nearest-centroid training
partition, including local covariance directions. Omit `direction_init`;
`cluster_pca` is rejected because stored directions are not used. Centroid-only
and PCA-bearing KMeans checkpoints are accepted via `kmeans_model_path`. Compatible
`init_model_path` models bypass this initialization. Resume uses EM-tagged
`checkpoint.pt`; switching between Adam and EM requires a new run initialized
from `mfa_model.pt`.

The example [hddc_em_D128_1M.yaml](../../configs/archived/hddc_em_D128_1M.yaml)
uses 2048-activation batches and sweeps K=500 and K=5000. The
[model page](../models/mfa-hddc.md#streamed-full-data-em) describes numerical
precision, output files and convergence limits.

## `experimental`

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `hard_assignment_covariance` | boolean | `false` | Use posterior argmax memberships for all periodic HDDC surgery statistics: counts, means, mixture weights, covariance, and noise pooling. |

```yaml
experimental:
  hard_assignment_covariance: true
```

Enabling this requires `model.kind: hddc`, `training.fit_method: adam` (the
default), and positive `model.surgery_every_epochs`. Other model kinds, full-data
EM, and disabled surgery reject the enabled option. Unknown experimental keys
and non-boolean values are rejected. Place this field only in `experimental`,
not `model` or `training`.

The pipeline forwards `--hard-assignment-covariance` to the HDDC CLI and records
its resolved boolean in manifest training arguments, run identity, saved
`config.json`, and W&B configuration. Omitting the section is equivalent to
setting the flag to false. To compare both modes, keep the field explicitly in
the configuration and add:

```yaml
sweep:
  experimental.hard_assignment_covariance: [false, true]
```

See [hard-assignment surgery](../models/mfa-hddc.md#experimental-hard-assignment-surgery)
for centering, empty-cluster handling, and global tie-breaking semantics.

## `initialization`

Automatic MFA-family initialization fits Euclidean KMeans centroids on the exact
training split and computes KNN PCs for `W_init`. This section requires automatic
initialization; it is rejected with supplied `training.kmeans_model_path`,
`training.init_model_path`, or HDDC EM.

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `pca_neighbors` | positive integer | `64` | Nearest training points per centroid; require `rank/q_max < pca_neighbors <= number of training activations`. |

```yaml
initialization:
  pca_neighbors: 64
```

Neighborhoods may overlap and include points assigned to other centroids.
Covariance uses float64 accumulation around the stored centroid. Empty or small
hard clusters do not prevent initialization. Validation points remain excluded.
The neighbor count is part of the immutable run identity and is checked on resume.
Initialization never performs rank selection.

`training.direction_init: cluster_pca` remains the automatic default and now
reads `W_init`. Explicit `random` still saves initialization PCs but uses random
loading directions. `initialization.pca_method` and `training.pca_method` are
removed: model geometry always uses hard members, initialization always uses KNN.
See the [model contract](../models/kmeans.md) for independent state handling.

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
| `kmeans_model_path` | `.pt` path or `null` | `null` | Reuse a fitted KMeans checkpoint directly. If neither this nor `init_model_path` is supplied, MFA, ARD, and Adam-based HDDC pipelines generate training-only KMeans/PCA. Planning validates K, D, and required PC capacity; legacy centroid bundles are rejected. |
| `direction_init` | `random` or `cluster_pca` | `cluster_pca` with automatic initialization; otherwise `random` | Initialize loading directions from KNN initialization PCs or randomly. Explicit `random` is respected even when automatic initialization saves PCA. |
| `init_model_path` | `.pt` path or `null` | `null` | HDDC only. Seed an epoch-0 training checkpoint from a saved `MFA_HDDC` whose `K`, `D`, `q`, and Psi noise mode exactly match the YAML model configuration. |

The remainder of this section describes MFA-family training. KMeans fitting
options are listed [above](#kmeans-only-fields-kind-kmeans).

With `kmeans_model_path`, the trainer loads `mu` and optional `W_init` directly and
skips fitting. It creates no `centroids.pt` copy. `direction_init: cluster_pca`
requires at least `rank/q_max` stored initialization PCs and uses their leading columns;
loading scales still initialize to one. A centroid-only KMeans checkpoint is
valid with `direction_init: random`. `centroids_path` is deprecated and rejected
by the pipeline; standalone legacy interfaces are not migrated.

Fresh MFA, ARD, and Adam-based HDDC training initializes mixture weights from
nearest-Euclidean-centroid counts `n_k` over the selected training split, after
dropping prefix tokens. This requires one full training pass, including when
centroids are supplied; validation rows are excluded and debug step limits do
not truncate the counts. Each empty cluster receives one pseudocount for
initializing its weight:

```text
n_tilde_k = max(n_k, 1)
pi_k = n_tilde_k / sum_j n_tilde_j = max(n_k, 1) / (N + n_empty)
```

Here `N = sum_k n_k` and `n_empty` is the number of zero-count components.
Positive counts are unchanged; when every cluster is populated this reduces
to `pi_k = n_k/N`. For example, counts `[0, 3, 6]` produce weights
`[0.1, 0.3, 0.6]`. The floor keeps initial logits finite so an empty hard
cluster can receive soft responsibility and gradients. It does not guarantee
that the component will acquire substantial membership during training.

The initializer logs how many components receive a pseudocount, previews their
IDs, and reports raw cluster-size extrema. It returns the unmodified raw counts;
pseudocounts do not add observations or change assignments, means, PCA
directions, loading scales, or noise. An empty training stream or a mismatch
between counted and expected training tokens still raises. Component-sharded
runs count on rank zero, broadcast raw counts, and normalize the floored weights
globally across all components, including when an entire shard is empty.

This rule is automatic for both `random` and `cluster_pca` direction modes;
there is no YAML setting. Resumed checkpoints and HDDC `init_model_path` starts
retain their saved weights. The floor applies only at initialization, not as a
weight constraint during optimization or HDDC surgery, and does not relax the
sample requirements for computing cluster PCA. HDDC `fit_method: em` uses its
separate hard-moment initialization and still requires positive membership for
every component to estimate its mean and covariance; see the
[EM contract](../models/mfa-hddc.md#streamed-full-data-em).

Example using a KMeans checkpoint to initialize `W_k`:

```yaml
training:
  kmeans_model_path: dalg-cache/path/to/kmeans_model.pt
  direction_init: cluster_pca
```

This option is available for all three MFA-family model kinds. The trainer reads
PCs from the KMeans model's `W_init`; the pipeline fits it beforehand for automatic initialization.

Automatic initialization uses the exact training split determined by `val_frac`,
`split_seed`, subset selection, and prefix-token dropping. It fits all training
activations with full-dimensional Euclidean KMeans (KMeans++, 100 iterations,
10 restarts, tolerance `1e-6`) and computes `rank/q_max` local PCA directions.
The training seed (or 0) and device are used. No validation activations are
included. Initialization always uses KNN PCA without rank selection; hard
cluster sizes do not limit its capacity. Set the neighborhood size through the
[initialization section](#initialization).

The worker loads training activations into host and device memory. KNN PCA
searches that population in blocks and computes covariance in centroid batches. See the
[initialization workflow](../workflows/training-pipeline.md#reusing-kmeans-model-initialization)
for artifacts, provenance, and resume behavior. The maintained worker is
`dalg-run-training-kmeans`; the temporary toy-centroid builder has been removed.

`init_model_path` and `kmeans_model_path` are mutually exclusive because the full
model already supplies its means. The initial model must exactly match `K`, `D`,
`q_max`, and the Psi noise mode (`diagonal`, component-specific
`isotropic_psi`, or `shared_b`). Only `single_process` HDDC training supports this option.
The saved model has no optimizer history, so the epoch-0 checkpoint starts with
a fresh Adam state; subsequent restarts use the normal local checkpoint exactly.

| Field | Type | Default | Meaning when fitting centroids |
| --- | --- | --- | --- |
| `pool_size` | integer or `null` | `null` | Reservoir size. When omitted, the trainer derives it from the training-token count and `K`. |
| `max_pool_size` | integer | `2000000` | Upper bound used by the automatic reservoir-size calculation. |
| `proj_dim` | integer | `32` | Projection dimension used by reservoir KMeans. |
| `refine_epochs` | integer | `25` | Extra centroid refinement passes with assignments fixed to the nearest centroid. |
| `vocab_size` | integer | `50257` | Vocabulary-size parameter passed to the centroid initializer. |

These reservoir settings apply to standalone trainers and HDDC EM initialization.
They have no effect on the new automatic KMeans/PCA pipeline initialization.

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
| `training_mode` | MFA: `vanilla` or `component_shard`; HDDC: `single_process` or `component_shard` | MFA: `vanilla`; HDDC: `single_process` | `vanilla` names the original fixed-q MFA implementation. `single_process` holds one full HDDC model in one process. `component_shard` shards K across GPUs and requires `resources.gpus > 1` plus `device: cuda`. |
| `compile` | boolean | `false` | Accepted by the current trainer parsers but not currently consumed by their implementations. |
| `wandb` | boolean | `false` | Enable Weights & Biases logging. Only rank 0 logs in component-sharded mode. |
| `wandb_project` | string or `null` | `null` | W&B project name. |
| `wandb_name` | string or `null` | `null` | W&B run name. When `wandb` is enabled and this is omitted, the pipeline supplies its generated run ID. |

For `component_shard`, `resources.gpus` becomes the `torchrun` process count.
This is component/model parallelism over K, not data parallelism. ARD does not
accept this field.

## `assignments`

This stage computes a complete hard assignment using the selected model.
MFA-family models use responsibility argmax; KMeans uses nearest-centroid IDs,
equivalent to the argmax of its one-hot responsibilities. Partial `max_batches`
outputs and legacy centroid-bundle interfaces are not pipeline stages.

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `enabled` | boolean | `true` | Run assignments after training. |
| `batch_size` | integer | `1024` | Assignment inference batch size. |
| `device` | string | `cuda` | Assignment inference device. |
| `seed` | integer or `null` | `null` | Assignment data-loader seed. When omitted, uses `training.seed`, falling back to `0`. |
| `use_inference_cache` | boolean | `true` | Use the model's inference cache while scoring responsibilities; ignored for KMeans. |

The output is `<run_dir>/mfa_model_assignments.pt`, or
`kmeans_model_assignments.pt` for KMeans. It must cover the complete
selected canonical activation stream and have cluster sizes summing to the
assignment count before the stage is marked complete.

## `evaluation`

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `enabled` | boolean | `false` | Run evaluation after assignments. |
| `kind` | string or `null` | `null` | The only current evaluator is `toy_manifold_tiling`. |
| `batch_size` | integer | `4096` | Batch size used for evaluation NLL or quantization error. |
| `device` | string | `cuda` | Evaluation device. |
| `rank_threshold` | float | `1.0` | For vanilla MFA and ARD, a loading column is effectively active when its variance exceeds this multiple of its component's mean unique variance. Ignored for HDDC and KMeans, which use saved component ranks. |
| `max_mean_to_manifold_distance` | float or `null` | `null` | Optional maximum ambient Euclidean distance between a Gaussian mean and its unique nearest exact manifold projection. `null` associates each Gaussian with its unique nearest manifold without distance filtering. |

`evaluation.enabled: true` requires `assignments.enabled: true`.
`toy_manifold_tiling` accepts `model.kind: mfa`, `ard`, `hddc`, or `kmeans` and requires
shards created by the toy-manifold shard writer. For MFA-family models it produces NLL, training-set
augmented BIC, clustering-recovery, live/dead-component, effective-rank,
tangent-alignment, and tangent-containment metrics in
`<run_dir>/metrics.json`. The effective-rank
rule for vanilla MFA and ARD filters columns relative to the learned noise
floor. HDDC instead uses its final `rank_mask.sum(-1)` without further
filtering, for both rank recovery and tangent containment. The output calls the
configured number of available columns `q_capacity`, since it is fixed `q` for
vanilla MFA and an upper bound for adaptive-rank models.

KMeans uses its saved cluster PCs, full spectrum, `pca_valid`, and `component_ranks` for
geometry, without constructing a Gaussian covariance. It omits `nll` and `bic`
and records `quantization.train` and `.validation` with `n`,
`sum_squared_distance`, and `mean_squared_distance` (`null` for an empty split).
KMeans completion checks validate stream and checkpoint provenance and the
`pca_geometry.version: 1` report contract. Sparse clusters (`count < rank + 1`)
are excluded only from rank and tangent metrics; association, clustering, and
quantization retain all components. All-excluded geometry has null summaries.

By default, each component is associated with its unique nearest planted
manifold. Setting `max_mean_to_manifold_distance` to a finite positive number
additionally requires the exact projection distance to be within that cutoff.
Tied nearest distances remain ambiguous and unassociated. Rank recovery and
tangent alignment use all proximity-associated components; assignments
define clustering metrics, the separately reported assignment-live/dead counts,
and augmented BIC's training-only activity reward. See the
[augmented BIC contract](../experiments/evaluation/toy-manifold-tiling.md#augmented-bic) for
the higher-is-better score reported in `bic.value`.

For a manifold with intrinsic dimension `r_i`, tangent alignment compares the
ground-truth tangent space with the full covariance's leading `r_i`-dimensional
eigenspace. It reports the mean squared principal-angle cosine as
`subspace_overlap` and the smallest cosine as `worst_direction_cosine`. The
subspace requires only the relative boundary eigengap between eigenvalues
`r_i` and `r_i + 1` to exceed `1e-6`; internal eigenvalue ties are valid. Each
global and per-manifold summary contains an unweighted mean plus valid and
undefined component counts. A summary with no valid components has a JSON
`null` mean.

`tangent_containment` uses the same scores against the leading effective-rank
covariance subspace and its own effective-rank covariance eigengap boundary.
Alignment and full containment are undefined when the component's effective rank is less
than `r_i`, including rank zero. Such components are counted as undefined and
excluded from score means; an all-undefined summary has a JSON `null` mean.

`tangent_partial_containment` applies only to `0 < q_k < r_i`, where `q_k` is
the same effective rank used for containment. It measures whether the learned
PC subspace lies inside the tangent, averaging squared principal-angle cosines
over `q_k` and reporting the minimum of those cosines. Rank zero and ranks at
least `r_i` are undefined. It appears globally and per manifold, uses the same
eligibility and eigengap rules, and requires no additional configuration.

See [Toy-Manifold Tiling Evaluation](../experiments/evaluation/toy-manifold-tiling.md) for
the exact geometry, metric equations, population definitions, undefined cases,
and `metrics.json` schema.

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

- absolute shard, initialization-model, output, and run-directory paths;
- trainer module and fully defaulted trainer arguments;
- resource defaults;
- assignment and evaluation defaults;
- stable run ID and full identity hash.

New automatic-initialization runs store `method: kmeans_model`, `version: 2`,
and their fitting/PCA settings in `initialization`, included in run identity.
The checkpoint lives at `<run_dir>/initialization/kmeans_model.pt`; existence is
checked during execution, not planning. Version-1 centroid initialization
manifests are rejected with instructions to plan a new run. Historical artifacts
are not converted or rewritten.

Changing any dataset, model, training, assignment, or evaluation field changes
the run identity. Slurm resources do not change the model run identity.

## Evaluation command overrides

`dalg-run-pipeline evaluate` reads existing JSONL manifests and overwrites metrics
in their saved run directories. Every invocation recomputes selected evaluations
and refreshes the experiment summaries. It does not replan training YAML or
change model checkpoints, assignments, or saved run identities. See the
[saved-run evaluation workflow](../workflows/training-pipeline.md#evaluate-saved-runs)
for local and Slurm examples, result layout, and retry behavior.

| Argument | Meaning |
| --- | --- |
| `--manifest PATH [PATH ...]` | Required source manifests, in order. |
| `--indices INDEX [INDEX ...]` | Optional original rows; use `manifest-position:row-index` with multiple manifests. |
| `--device DEVICE` | Override each row's evaluation device. |
| `--batch-size N` | Positive evaluation batch size. |
| `--rank-threshold FLOAT` | Override the MFA/ARD effective-rank threshold; ignored for HDDC/KMeans. |
| `--max-mean-to-manifold-distance FLOAT_OR_NONE` | Positive finite association cutoff, or `none` to remove it. |
| `--resources PATH` | Flat YAML mapping of resource overrides using the existing `resources` keys. |
| `--submit` | Submit evaluation arrays followed by a dependent CPU collection job. |
| `--dry-run` | Inspect without writing outputs, evaluating, or submitting. |

Unspecified evaluation options inherit source values and then pipeline defaults.
Evaluation is explicitly enabled for this command even if disabled in the source;
a missing evaluator kind defaults to `toy_manifold_tiling`. Resource defaults use
one node, one task, and one GPU for CUDA or zero for CPU; other fields inherit
source resources before explicit overrides. A CUDA device requires at least one
allocated GPU. These changes apply only to evaluation, not the source manifest.

The command creates its job plan automatically under `outputs/evaluations/`;
there is no result-directory argument. Reports are validated before atomically
replacing each run's `metrics.json`, followed by its evaluation completion marker.
Existing ancestor summary directories are refreshed, or summaries are created
in each run's parent directory when none exist. Refreshes include unselected
reports in those directories. Missing prerequisites are reported in the job's
`skipped_runs.json`; obsolete metric reports themselves can be overwritten.
