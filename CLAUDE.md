# CLAUDE.md

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

The repo now uses a `src/` layout.

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
```

Important top-level files:
- `pyproject.toml`: package + CLI entrypoints
- `.vscode/launch.json`: useful local debug configs
- `mfa_tutorial.py` and `mfa_tutorial.ipynb`: tutorial material
- `README.md`: short project description

## Main Entry Points

Preferred CLI entrypoints are defined in `pyproject.toml`:
- `dalg-run-layer`
- `dalg-interpret-mfa`
- `dalg-cluster-overlap`
- `dalg-cluster-intrinsic-dim`
- `dalg-build-pile-windows`

The most important one is:
- `dalg-run-layer`

It lives in `src/dalg/cli/run_layer.py` and orchestrates the main workflow with subcommands:
- `extract`
- `extract-windows`
- `train`
- `overlap`
- `intrinsic-dim`
- `all`

When in doubt, start from `src/dalg/cli/run_layer.py`.

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

### `src/dalg/models/train.py`

Contains `train_nll`, the main optimizer loop.

Important detail:
- it is DDP-aware
- only rank 0 handles some logging / checkpointing decisions

### `src/dalg/init/projected_knn.py`

Contains `ReservoirKMeans`, used to initialize MFA centroids at scale.

High-level idea:
- stream activations from a loader
- sample a reservoir
- optionally project to `proj_dim`
- run KMeans
- refine centroids

### `src/dalg/data/shard_activations.py`

Very important for large runs.

This is the streaming layer for pre-extracted activation shards. It is used heavily by the shard-based training path.

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
- `subspace_interpretation.py`: top strings / examples per component
- `subspace_visualization.py`: projection and visualization helpers

### `src/dalg/cli/interpret_mfa.py`

Interpretation pipeline for trained MFA models.

Typical flow:
1. stream over shards
2. compute top-responsibility tokens per cluster
3. recover local text context
4. optionally label clusters with an LLM

## Main Workflow

For large-scale work, the usual research path is:

1. build token windows dataset
2. extract activations into shards
3. train MFA from shards
4. analyze overlaps / intrinsic dimension / assignments
5. interpret regions
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

Important script:
- `scripts/slurm/sbatch_train_shards.sh`

That script is the reference for distributed shard training and mirrors the real production training shape more than small local runs do.

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
- loaders often yield `(activations, token_ids)` or `(x, ...)`
- shard-based training uses token windows and drops a prefix (`drop_prefix`) before training
- positive MFA parameters are represented via raw tensors passed through `softplus`
- many analysis utilities stream activations instead of loading everything into memory

Do not casually change these conventions unless the caller chain is checked carefully.

## Important Implementation Details

### Sharded training

The shard-based path in `dalg-run-layer train` is different from the simple monolithic path.

Important details:
- it can run under DDP / `torchrun`
- rank 0 may compute centroids and save them
- other ranks wait and then load the saved centroids
- many debugging issues appear only in this path, not in single-process training

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
- first determine whether the issue is in the simple path or the shard/DDP path
- check `scripts/slurm/` and `.vscode/launch.json`
- inspect `outputs/jobs/` logs

When adding new analysis code:
- prefer putting reusable logic under `src/dalg/analysis/`
- expose a CLI only if it is genuinely useful as a standalone workflow

When adding new runnable workflows:
- prefer package entrypoints over ad hoc top-level scripts

## Things To Avoid

- do not reintroduce old imports like `from modeling...` or `from experiments...`
- do not add new generated outputs inside source folders
- do not over-abstract simple research code
- do not delete scratch data or large experiment outputs unless the user explicitly asks

## Short Mental Model

If you need a quick picture of the repo:

**windows dataset -> activation shards -> centroid init -> MFA training -> analysis / interpretation / steering**

That is the backbone of the project.
