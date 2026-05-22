# AGENTS.md

This file gives future agents the minimum context needed to work effectively in this repository.

## Project Goal

This is a machine learning research codebase for **"From Directions to Regions: Decomposing Activations in Language Models via Local Geometry"**.

The main idea is:
- model LLM activations with a **Mixture of Factor Analyzers (MFA)**
- each component is a **region** with a centroid `mu_k`
- each region also has a **local low-rank subspace** defined by `W_k`

This supports:
- training MFA models on activations
- analyzing regions and overlaps
- estimating local intrinsic dimensionality
- interpreting clusters from top-activating tokens
- steering models with region-level structure

## What The User Cares About

The user is doing research, not building a polished production system.

Priorities:
- keep code **simple**
- keep code **readable**
- keep code **easy to modify**
- prefer direct implementations over abstractions
- do not over-engineer

Do not optimize prematurely. If something is a bit repetitive but clearer, clarity wins.

## Current Repository Layout

The repo uses a `src/` layout.

```text
src/dalg/
  cli/            Main runnable entrypoints
  models/         MFA model and training code
  init/           Initialization / KMeans
  data/           Dataset loaders and sharded activation streaming
  llm/            Activation extraction from TransformerLens models
  analysis/       Overlap, intrinsic dimension, assignments, interpretation helpers
  intervention/   Steering code

scripts/slurm/    Cluster job scripts
outputs/          Generated experiment artifacts and job logs
notebooks/        Exploratory notebooks
tests/            Smoke tests, synthetic-shard fixtures, equivalence checks
```

Important top-level files:
- `pyproject.toml`: package + CLI entrypoints
- `.vscode/launch.json`: useful local debug configs
- `mfa_tutorial.py` and `mfa_tutorial.ipynb`: tutorial material
- `README.md`: short project description

## Main Entry Points

Preferred CLI entrypoints are defined in `pyproject.toml`:
- `dalg-run-extraction`
- `dalg-run-training`
- `dalg-run-metrics`
- `dalg-interpret-mfa`
- `dalg-label-mfa-clusters`
- `dalg-cluster-overlap`
- `dalg-cluster-intrinsic-dim`
- `dalg-build-pile-windows`

The three most important ones are:
- `dalg-run-extraction` — activation extraction into on-disk shards.
- `dalg-run-training` — MFA training (vanilla single-process, or component-sharded model-parallel).
- `dalg-run-metrics` — cluster-level metrics (overlap, intrinsic dim) on a trained MFA.

`dalg-run-extraction` lives in `src/dalg/cli/run_extraction.py` and is a flat
single-command CLI: it reads a pre-tokenized HF windows dataset and writes
per-layer activation shards (resume-safe).

`dalg-run-metrics` lives in `src/dalg/cli/run_metrics.py` and has two
subcommands:
- `overlap`
- `intrinsic-dim`

`dalg-run-training` lives in `src/dalg/cli/run_training.py` and is a single
command selected by `--training-mode {vanilla, component_shard}`. The dataset
is always a directory of pre-extracted activation shards produced by
`dalg-run-extraction`, passed via `--shard-dir`. The older monolithic
`activations.pt` / `tokens.pt` path and the previous `--training-mode ddp`
data-parallel mode have been removed; if you need data parallelism, run more
than one job rather than reintroducing DDP here.

When in doubt, start from `src/dalg/cli/run_extraction.py` for extraction,
`src/dalg/cli/run_metrics.py` for analysis, and
`src/dalg/cli/run_training.py` for training.

## Core Code Map

### `src/dalg/models/mfa.py`

This is the core model.

Key parameters:
- `mu`: component means, shape `(K, D)`
- `dir_raw` and derived loadings: local directions
- `scale_rho`: loading scales
- `psi_rho`: diagonal noise
- `pi_logits`: mixture weights

Important detail:
- likelihood and posterior computations rely on the **Woodbury identity**
- `q` is much smaller than `D`, so many operations are done in the small latent space

Common methods:
- `responsibilities`
- `log_prob`
- `nll`
- `component_posterior`
- `reconstruct`

Serialization helpers:
- `save_mfa`
- `load_mfa`
- `save_component_shard` (per-rank save for component-sharded runs)

### `src/dalg/models/train.py`

Contains `train_nll`, the main optimizer loop.

Important details:
- it is distributed-aware via `torch.distributed`: when initialized, rank 0
  drives validation and metric selection, then broadcasts the selection metric
  so every rank stays in sync.
- only rank 0 prints, drives `tqdm`, runs validation, and logs to W&B.
- validation can be supplied two ways, with `val_tensor` taking priority:
    - `val_tensor`: a pre-materialized tensor holding the whole val split
      (typically built once on rank 0). Iterated in chunks; this is the fast
      path used by the shard-training CLI when the val set fits in memory.
    - `val_loader`: a regular DataLoader fallback for val sets too large to
      materialize.
  When both are `None`, the best-model metric falls back to training NLL.
- it also supports component-sharded training checkpoints, where every rank
  saves and resumes its own model/optimizer shard (`checkpoint_all_ranks=True`).
- after each `loss.backward()` the loop calls
  `raw_model.sync_replicated_grads()` if present, so component-sharded models
  can sum gradients on replicated parameters before `optimizer.step()`.

### `src/dalg/cli/run_training.py`

The training CLI (`dalg-run-training`). Two training modes selected by
`--training-mode`:

- `vanilla` — `cmd_train`: one process, one full MFA model. Must be launched
  with `WORLD_SIZE=1` (i.e. *not* under `torchrun`); `validate_args` rejects
  the combination of `vanilla` and `WORLD_SIZE>1`.
- `component_shard` — `cmd_train_component_shard`: N processes, each holds a
  contiguous `K/N` slice of the MFA components (model-parallel over
  components). Requires `torchrun` with `WORLD_SIZE>1` and `--device cuda`.

The word "shard" is overloaded: `--shard-dir` refers to on-disk *dataset*
shards, while `--training-mode component_shard` refers to splitting the
*model* across processes. The two axes are independent.

Shared helpers in this module:
- `_resolve_activation_data(args, *, log)`: reads shard config / meta and
  builds the stratified train/val row split. Returns a `dict` carrying
  `shard_dir`, `out_dir`, `layer`, `window`, `d_model`, `drop_prefix`,
  `n_train_tokens`, `meta_index`, `train_pos_full`, `val_pos`,
  `train_counts`, `val_counts`, `val_frac`, `split_seed`.
- `_build_train_loader(data, args, *, device)`: builds an
  `ActivationBatchDataset` over `train_pos_full` wrapped in a
  `DataLoader(batch_size=None, ...)`, returning `(loader, steps_per_epoch, train_pos)`.
- `_materialize_val_tensor(...)` and `_build_val_tensor_for_main(data, args, *, device)`:
  rank-0 helper that streams the validation rows into one contiguous tensor
  (on GPU when `--val-on-gpu`, otherwise pinned CPU memory) for use as
  `val_tensor` in `train_nll`.
- `_ensure_centroids(data, args, *, out_dir, is_main, device, barrier)`:
  fits and caches `centroids.pt` on rank 0 via `ReservoirKMeans`, then every
  rank loads the cached centroids (gated by a `dist.barrier()` when relevant).
- `_write_split_info` / `_write_run_config`: persist `val_indices.json` and
  `config.json` next to the model output.
- `_maybe_init_wandb` / `_finish_wandb`: rank-0-only W&B init and teardown,
  driven by `--wandb`, `--wandb-project`, `--wandb-name`.
- `_ComponentShardLoader`: rank-0-broadcast loader that guarantees every rank
  sees the same activation batch in component-shard mode. Rank 0 pulls from
  the underlying `DataLoader` and broadcasts the shape then the tensor;
  non-zero ranks allocate an empty tensor of the broadcast shape and receive
  the data.

### `src/dalg/init/projected_knn.py`

Contains `ReservoirKMeans`, used to initialize MFA centroids at scale.

High-level idea:
- stream activations from a loader
- sample a reservoir
- optionally project to `proj_dim`
- run KMeans
- refine centroids

### `src/dalg/data/shard_activations.py`

Very important for large runs. This is the streaming layer for pre-extracted
activation shards produced by `dalg-run-extraction`.

Expected on-disk layout (`<root>` is `--shard-dir`):

```text
<root>/config.json
<root>/layerNN/shard_NNNNN.pt    # tensors of shape (rows, window, d_model)
<root>/meta/shard_NNNNN.json     # per-shard row metadata: row_indices, rows[i].subset
```

Only this layout is supported now; the older "single tensor" and
auto-synthesized-metadata code paths have been removed. There is no fallback
to a monolithic `activations.pt`.

Public surface:
- `load_meta_index(activation_dir, layer)` returns one entry per row with
  `shard`, `row_in_shard`, `global_row`, `subset`. Subset defaults to
  `"all"` when missing.
- `stratified_split(meta_index, val_frac, seed)` returns sorted
  `(train_positions, val_positions)` stratified by `subset`. Positions index
  into `meta_index`.
- `per_subset_counts(meta_index, positions)` summarizes a row-position list
  by subset.
- `ActivationBatchDataset`: the core `IterableDataset`. Wrap with
  `DataLoader(dataset, batch_size=None, num_workers=...)`. It already yields
  batched activation tensors of shape `(B, d_model)`. With
  `return_metadata=True`, it yields `(x, global_rows, tok_pos)`.
- `ShardActivationBatchDataset`: alias of `ActivationBatchDataset` kept for
  backwards compatibility with callers.

Iteration details worth knowing:
- Each shard is loaded with `torch.load(..., mmap=True, weights_only=True)`,
  the prefix tokens are dropped, and rows × tokens are flattened to
  `(N, d_model)` before being sliced into batches of size `batch_size`.
- Multi-worker sharding: as an `IterableDataset`, every `DataLoader` worker
  runs its own `__iter__`. The dataset partitions the shard list with
  `shards[wid::nworkers]` so workers do not yield duplicate batches.
- `shuffle_shards` and `shuffle_within_shard` are seeded by `seed` and by
  the current `epoch` (set via `set_epoch`); shard-level and within-shard
  shuffles use independent derived seeds.
- Canonical (unshuffled) random access is available via `dataset[i]` for
  tests and interpretation lookups; `dataset.num_items` is the flattened
  token count, while `len(dataset)` is the number of batches.

### `src/dalg/llm/activation_generator.py`

Wraps TransformerLens to extract activations from a model.

Supported activation modes include:
- `residual`
- `residual_pre`
- `mlp`
- `mlp_out`
- `attn_out`

### `src/dalg/analysis/`

Main analysis modules:
- `cluster_overlap.py`: pairwise overlap metrics between MFA components
- `cluster_intrinsic_dim.py`: per-cluster PCA-based intrinsic dimension
- `cluster_assignments.py`: save hard assignments and cluster sizes
- `cluster_labeling.py`: helpers used by `dalg-label-mfa-clusters`
- `subspace_interpretation.py`: top strings / examples per component
- `subspace_visualization.py`: projection and visualization helpers

### `src/dalg/cli/interpret_mfa.py`

Interpretation pipeline for trained MFA models.

Typical flow:
1. stream over shards
2. compute top-responsibility tokens per cluster
3. recover local text context from the HF windows dataset using `global_row` / `tok_pos`
4. optionally label clusters with an LLM

## Main Workflow

For large-scale work, the usual research path is:

1. build token windows dataset (`dalg-build-pile-windows`)
2. extract activations into shards (`dalg-run-extraction`)
3. train MFA from shards (`dalg-run-training --shard-dir ...`)
4. analyze overlaps / intrinsic dimension / assignments (`dalg-run-metrics`
   subcommands, or the standalone `dalg-cluster-*` entrypoints)
5. interpret regions (`dalg-interpret-mfa`, then optionally
   `dalg-label-mfa-clusters`)
6. optionally steer with the learned structure

In practice:
- extraction and training often happen through `scripts/slurm/`
- local debugging often happens through `.vscode/launch.json`

## Cluster / SLURM Notes

The user usually works on a SLURM cluster and often debugs via VS Code remote/tunneling.

Important operational assumptions:
- you are often on a GPU node
- local home storage is limited
- large data usually lives in `/orfeo/scratch/dssc/zenocosini`
- do not delete things from scratch unless explicitly asked

Useful locations:
- job scripts: `scripts/slurm/`
- job logs: `outputs/jobs/`
- experiment artifacts: `outputs/experiments/`
- scratch/cache symlink: `dalg-cache/`
- scratch experiment symlink: `output/`

Important scripts:
- `scripts/slurm/sbatch_train_shards.sh` — single-process vanilla shard
  training (one GPU, full MFA on one rank).
- `scripts/slurm/sbatch_train_component_shards.sh` — component-sharded MFA
  training launched under `torchrun`.

Together these mirror the real production training shape more than small
local runs do.

## Scratch Symlinks

The repo has two top-level symlinks into scratch storage:
- `dalg-cache/` -> `/orfeo/scratch/dssc/zenocosini/dalg-cache/`
- `output/` -> `/orfeo/scratch/dssc/zenocosini/dalg-cache/output/`

Some scratch paths may also appear through older direct aliases such as
`/orfeo/scratch/dssc/zenocosini/pile_gemma2b_activations` or
`/orfeo/scratch/dssc/zenocosini/pile_gemma2b_100M_windows/merged`. When a
scratch path looks missing, check the symlink target and the `dalg-cache/...`
form before assuming data was deleted.

These point to large generated data. Treat them like scratch experiment state:
- do not delete or overwrite large files there unless the user explicitly asks
- prefer writing new large artifacts under these symlinks instead of home storage
- use `output/` for analysis artifacts from recent runs; note that older docs may say `outputs/`

Current `dalg-cache/` contents:
- `pile_gemma2b_100M_windows/`: Hugging Face token-window dataset built from the Pile for Gemma 2B activation extraction.
- `pile_gemma2b_100M_windows/shards/`: intermediate window shards from dataset construction.
- `pile_gemma2b_100M_windows/merged/`: merged HF dataset with Arrow files plus `dataset_info.json` and `state.json`.
- `pile_gemma2b_activations_debug/`: small debug activation extraction output.
- `pile_gemma2b_activations_debug/layer05/`: debug layer 5 activation shard tensors.
- `pile_gemma2b_activations_debug/layer17/`: debug layer 17 activation shard tensors.
- `pile_gemma2b_activations_debug/tokens/`: debug token shard tensors aligned with activation shards.
- `pile_gemma2b_activations_debug/meta/`: debug per-shard JSON metadata.
- `pile_gemma2b_activations/`: main Gemma 2B activation cache and trained MFA run folders.
- `pile_gemma2b_activations/layer05/`: main layer 5 activation shard tensors.
- `pile_gemma2b_activations/layer17/`: main layer 17 activation shard tensors.
- `pile_gemma2b_activations/tokens/`: token shard tensors aligned with the main activations.
- `pile_gemma2b_activations/meta/`: per-shard JSON metadata for the main activations.
- `pile_gemma2b_activations/layer*_????_mfa/` and `layer05_mfa_32000/`: MFA training outputs for specific layers and cluster counts; typical files include `config.json`, `centroids.pt`, `checkpoint.pt`, `mfa_model.pt`, `overlap.pt`, `val_indices.json`, and assignment tensors.
- `output/`: nested output symlink target used for experiment analysis artifacts.
- `pile_gemma2b_build.log`: build/extraction log for the Gemma 2B Pile cache.

Current `output/` contents:
- `output/experiments/`: analysis outputs such as `intrinsic_dims.pt`, `overlap.pt`, PCA plots, heatmaps, histograms, dendrograms, and cluster-size plots.
- `output/experiments/1000_05/`, `1000_17/`, `8000_05/`, `8000_17/`, `32000_05/`, `32000_17/`: experiment folders named by cluster count and layer.
- `output/experiments/*_backup/`: backup copies of earlier experiment analysis outputs.
- `output/experiments/8000_05_assignments_only/`: assignments-only experiment output folder.

## Local Development / Debugging

The repo uses a virtual environment in `.venv`.

Typical setup:

```bash
source .venv/bin/activate
```

When using package imports locally without installing the package, `PYTHONPATH=src` is often needed.

VS Code launch configs already account for this.

On Apple Silicon or mixed backends:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

## Data / Batch Conventions

Some recurring conventions across the codebase:
- shard loaders usually yield activation batches `x`; optional metadata batches are `(x, global_row, tok_pos)`
- `ActivationBatchDataset.__len__` is number of batches; use `dataset.num_items` for flattened token count and `dataset[n]` for canonical random access
- shard-based training uses token windows and drops a prefix (`drop_prefix`) before training
- positive MFA parameters are represented via raw tensors passed through `softplus`
- many analysis utilities stream activations instead of loading everything into memory

Do not casually change these conventions unless the caller chain is checked carefully.

## Docstring
Docstrings and comments need to explain the code, not the changes you have just made to the code. For example, if I ask you to refactor the training loop to be more modular, the docstring should not say "refactored the training loop to be more modular", it should explain how the training loop works and what each part does. The docstring should be written as if the reader has no context about the recent changes, and should provide a clear understanding of the code's functionality and purpose.

## Important Implementation Details

### Sharded training

`dalg-run-training` always reads from on-disk activation shards
(`--shard-dir`). The two training modes differ only in how the model is
distributed across processes:

1. **Vanilla mode** (`--training-mode vanilla`, default)
   - Single process. Must run with `WORLD_SIZE=1`; do not launch under
     `torchrun`.
   - Holds a full MFA model on one device.
   - Saves the usual full-model files: `mfa_model.pt`, `checkpoint.pt`,
     `centroids.pt`, `config.json`, `val_indices.json`.
   - Reference Slurm script: `scripts/slurm/sbatch_train_shards.sh`.

2. **Component-sharded / model-parallel mode** (`--training-mode component_shard`)
   - Launched under `torchrun` with `WORLD_SIZE>1` and `--device cuda`.
   - Reference Slurm script: `scripts/slurm/sbatch_train_component_shards.sh`.
   - Each rank owns a contiguous slice of the MFA components `K`. For
     example, with `K=8000` and 4 ranks, each rank owns about 2000 components.
   - Every rank must see the same activation batches in the same order. This
     is **not** data parallelism. `_ComponentShardLoader` enforces this by
     having rank 0 read the batch and `dist.broadcast` it to other ranks.
   - Increasing GPUs reduces per-rank component memory instead of increasing
     the effective data batch.
   - `BATCH` is the logical batch each component shard sees. It should not be
     multiplied by world size when reasoning about the effective batch.
   - Validation is currently skipped in component-shard mode: `train_nll` is
     called with `val_tensor=None` and the best-epoch metric falls back to
     training NLL. Use `VAL_FRAC=0.0` in Slurm scripts unless a future agent
     implements distributed validation.
   - `load_mfa` can assemble final component-sharded saves from
     `mfa_model_shards.json`. It also supports the historical pattern of
     passing `<run_dir>/mfa_model.pt` when that file is absent but
     `<run_dir>/mfa_model_shards.json` exists. This assembles a full MFA in
     memory, so it can still be too large for some downstream analysis jobs.

There is no longer a `ddp` (data-parallel) mode. If a path in the repo or in
old docs refers to one, treat it as stale.

Shared startup behaviour for both modes:
- rank 0 (or the single process in vanilla mode) computes centroids and saves
  them to `<out_dir>/centroids.pt`; in distributed mode other ranks wait at a
  `dist.barrier()` and then load the cached centroids.
- many debugging issues appear only in the distributed path, not in
  single-process training, so reproduce locally with the small synthetic
  shards under `tests/fixtures/` when possible.

Important component-sharded implementation details:
- `ComponentShardedMFA` lives in `src/dalg/models/mfa.py`.
- Component ownership is assigned by `component_shard_bounds(K, rank, world_size)`.
- The mixture likelihood is assembled with a distributed logsumexp over the
  component dimension. Be careful with autograd here: every rank participates
  in the same logical loss, so a naive autograd-aware all-reduce can
  double-count gradients.
- The shared diagonal noise parameter `psi_rho` is replicated when
  `psi_per_component=False` (the default). Its gradient must be summed across
  ranks before `optimizer.step()`. This is handled by
  `ComponentShardedMFA.sync_replicated_grads()` and called from `train_nll`.
- Component-local parameters such as `mu`, `dir_raw`, `scale_rho`, and local
  `pi_logits` should not be DDP-wrapped or all-reduced.

Component-sharded checkpointing:
- Do **not** gather and save one full model checkpoint on rank 0 for large
  runs. For `K=8000`, `D=2048`, `q=160`, the full `dir_raw` tensor alone is
  about 10 GiB in fp32, before Adam state.
- Component-sharded training saves per-rank checkpoints:
  - `checkpoint_rank0000.pt`
  - `checkpoint_rank0001.pt`
  - ...
  - `checkpoint_shards.json`
- Each rank checkpoint contains that rank's local model shard, local optimizer
  state, epoch, and RNG state.
- Resume assumes the same world size and rank-to-component mapping. If you
  change the number of GPUs, expect shape mismatches or invalid optimizer
  state unless explicit repartitioning support has been added.
- Final component-sharded model saves are also per-rank:
  - `mfa_model_rank0000.pt`
  - `mfa_model_rank0001.pt`
  - ...
  - `mfa_model_shards.json`
- A component-sharded run usually does not have a single `mfa_model.pt`.
  `load_mfa` supports sharded model manifests, so point tools at the
  sharded model path/manifest instead of assuming `mfa_model.pt` exists.

Testing component-sharded changes:
- Use `tests/component_sharded_mfa_equivalence.py` for a small serial-vs-sharded
  equivalence check. It builds a full serial MFA and a two-rank sharded MFA
  from identical weights, trains both on the same batches, gathers the shards,
  and compares parameters.
- A useful local CPU command is:

```bash
PYTHONPATH=src python -m torch.distributed.run --standalone --nproc_per_node=2 \
  tests/component_sharded_mfa_equivalence.py --device cpu --optimizer adam --steps 4
```

- For CUDA, run the same script under a two-GPU Slurm allocation with
  `--device cuda`.
- Also test the actual CLI path with a tiny synthetic shard dataset before
  launching a large job, because the CLI path covers centroid loading,
  checkpoint manifests, and final per-rank shard saves. `tests/fixtures/`
  and `tests/synthetic_shards.py` provide ready-made small shard layouts
  compatible with `ActivationBatchDataset`.

### W&B logging

`train_nll` logs to Weights & Biases on rank 0 only when `wandb.run` is
active. Initialization is opt-in via `--wandb`, `--wandb-project`,
`--wandb-name`, handled by `_maybe_init_wandb` / `_finish_wandb` in
`run_training.py`. When `--wandb` is not passed the logging calls degrade to
no-ops, so the loop has no extra dependency at runtime.

### Outputs

Generated outputs should generally go under:
- `outputs/jobs/`
- `outputs/experiments/`

Avoid scattering logs and generated files across source directories.

### Tutorial files

`mfa_tutorial.py` is not a normal clean Python script; it contains notebook-style magics and is closer to a synced notebook representation.

Be careful when linting or compiling it.

## Guidance For Future Agents

When modifying this repo:
- preserve the user's research-first style
- prefer small, local edits
- avoid heavy abstractions
- do not turn the code into a framework
- keep command paths and SLURM flows aligned with the current package layout

When investigating bugs:
- first determine whether the issue is in the single-process path or the
  distributed component-shard path
- check `scripts/slurm/` and `.vscode/launch.json`
- inspect `outputs/jobs/` logs

When adding new analysis code:
- prefer putting reusable logic under `src/dalg/analysis/`
- expose a CLI only if it is genuinely useful as a standalone workflow

When adding new runnable workflows:
- prefer package entrypoints over ad hoc top-level scripts

## TODO

- Make model-loading CLI arguments more explicit for full vs sharded MFA runs,
  e.g. avoid implying every run has a literal `mfa_model.pt` when
  component-sharded outputs are loaded through `mfa_model_shards.json`.
- Distributed validation for component-sharded training (today it is skipped
  and best-epoch falls back to training NLL).

## Things To Avoid

- do not reintroduce old imports like `from modeling...` or `from experiments...`
- do not add new generated outputs inside source folders
- do not over-abstract simple research code
- do not delete scratch data or large experiment outputs unless the user explicitly asks

## Short Mental Model

If you need a quick picture of the repo:

**windows dataset -> activation shards -> centroid init -> MFA training -> analysis / interpretation / steering**

That is the backbone of the project.
