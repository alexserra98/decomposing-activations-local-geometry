# AGENTS.md

This file gives future agents the minimum context needed to work effectively in
this repository.

## Project Goal

This is a machine learning research codebase for **"From Directions to Regions:
Decomposing Activations in Language Models via Local Geometry"**.

The core object is a **Mixture of Factor Analyzers (MFA)** over language-model
activations:

- each component is a local activation region with centroid `mu_k`
- each component has a local low-rank subspace `W_k`
- the model supports cluster/region analysis, Gaussian overlap metrics, intrinsic
  dimension estimates, interpretation from top token contexts, and steering

This is research code. Prefer clear, direct implementations over general
frameworks. Keep edits small, readable, and easy to modify.

## Repository Layout

The repo uses a `src/` layout.

```text
src/dalg/
  cli/            Main runnable entrypoints
  models/         MFA model, component-sharded MFA, training loops
  models/adaptive_q/  Per-component-rank variants: ARD prior and HDDC surgery
  cli/adaptive_q/     Their entrypoints
  init/           Reservoir KMeans centroid initialization
  data/           Window builders and activation-shard streaming
  llm/            TransformerLens activation extraction wrapper
  analysis/       Gaussian overlap, intrinsic dim, assignments, labels, description metrics
  intervention/   Region/subspace steering code

scripts/slurm/    Cluster job scripts for extraction, training, metrics, labels
scripts/          Temporary/profiling/synthetic-analysis scripts
scripts/adaptive_q/       Runners for the per-component-rank experiments
scripts/slurm/adaptive_q/ Their cluster job scripts
tests/            Unit/smoke tests and synthetic shard fixtures
notebooks/        Exploratory notebooks
docs/experiments/ Attachable context for temporary experiment workflows
.agents/skills/   Canonical recurring workflow skills for Codex and Claude
.claude/skills/   Symlinks exposing the canonical skills to Claude Code
outputs/          Generated local artifacts; may contain untracked reports
logs/             Slurm logs in current scripts
dalg-cache/       Symlink to scratch cache
```

The main Gemma 2B scratch data is split by responsibility:

- `dalg-cache/pile_gemma2b_activations/` contains activation shards, tokens,
  and metadata.
- `dalg-cache/pile_gemma2b_models/` contains trained MFA run directories,
  centroid collections under `centroids/`, and derived metrics.
- `dalg-cache/output/` contains legacy metric outputs. Leave these existing
  files in place, but do not use this directory for new model metrics.

Important top-level files:

- `pyproject.toml`: package metadata and console scripts
- `README.md`: short project description
- `.vscode/launch.json`: local debug configs when present
- `AGENTS.md` and `CLAUDE.md`: future-agent context

## Main Entrypoints

Preferred CLI entrypoints are defined in `pyproject.toml`:

- `dalg-run-extraction`
- `dalg-run-training`
- `dalg-run-training-ard`
- `dalg-run-training-hddc`
- `dalg-run-metrics`
- `dalg-interpret-mfa`
- `dalg-label-mfa-clusters`
- `dalg-gaussian-overlap`
- `dalg-cluster-intrinsic-dim`
- `dalg-build-pile-windows`
- `dalg-build-newsgroups-windows`

The main workflow is:

```text
token windows dataset
  -> activation shards
  -> centroid init
  -> MFA training
  -> assignments / Gaussian overlap / intrinsic dim
  -> cluster examples and LLM labels
  -> optional description metrics / steering / notebooks
```

There is no current `dalg-run-layer` entrypoint in `pyproject.toml`. If old
docs or logs mention it, treat that path as stale.

## Data Layout

### Toy manifold dataset generator

`src/dalg/data/manifold_dataset.py` provides `ToyManifoldConfig` and
`make_toy_manifold_datasets` for deterministic synthetic local-geometry data.
It defines eight manifold types and creates `manifolds_per_type` independently
embedded instances of each type in `ambient_dim`. The function returns balanced
train and validation `TensorDataset`s containing `(points, manifold_id)`, plus
metadata with type IDs, intrinsic dimensions, embeddings, and offsets.

Set `offset_radius=0` for instances centered at the origin; use a positive
radius for separated/non-centered instances. Generate paired conditions with
the same seed and configuration except for `offset_radius` so they differ only
by the recorded per-instance offsets. There is no dedicated CLI or workflow
skill; import the module directly. Store large generated `.pt` artifacts under
`dalg-cache/assets/`, not in source directories.

Large runs use activation shards produced by `dalg-run-extraction`.

Expected layout for `--shard-dir <root>`:

```text
<root>/config.json
<root>/layerNN/shard_NNNNN.pt
<root>/tokens/shard_NNNNN.pt
<root>/meta/shard_NNNNN.json
```

Layer shard tensors have shape `(rows, window, d_model)`. Metadata maps shard
rows back to global window rows and optional subset labels.

`src/dalg/data/shard_activations.py` is the streaming layer:

- `load_meta_index(activation_dir, layer)` returns one row entry per window
- `stratified_split(meta_index, val_frac, seed)` builds train/val row splits
- `per_subset_counts(meta_index, positions)` summarizes split composition
- `ActivationBatchDataset` streams flattened token activations of shape
  `(batch, d_model)` after dropping prefix tokens
- set `return_metadata=True` when downstream code needs `(x, global_rows, tok_pos)`

`ActivationBatchDataset` is already batched, so wrap it with
`DataLoader(dataset, batch_size=None, ...)`.

### Subset Slice Suffix (`--shard-dir <root>#<spec>`)

`src/dalg/data/subset_spec.py` is a small, deletable helper that lets you re-run
the pipeline on a filtered, randomly-subsampled slice of an existing activation
shard directory **without re-extracting or duplicating activations**. The slice
is encoded as a `#<spec>` suffix on the value you already pass to `--shard-dir`:

```bash
--shard-dir dalg-cache/pile_gemma2b_activations#pile_wikipedia_1M
```

The spec format is `pile_<subset>_<N>[K|M]`:

- `<subset>` is resolved via `_SUBSET_ALIASES` (currently `wikipedia ->
  pile-wikipedia_en`); other names fall back to `pile-<subset>`.
- `<N>[K|M]` is a **token budget**, translated to
  `ceil(N / (window - drop_prefix))` randomly chosen windows of that subset.
- Selection is deterministic (fixed seed) so every pipeline step picks the same
  rows; positions are returned sorted to match canonical stream order.

Two functions form the whole surface:

- `split_shard_dir_spec(value)` -> `(clean_path, spec_or_None)`
- `resolve_spec_positions(meta_index, spec, *, window, drop_prefix)` -> sorted
  positions into the full `meta_index` (all positions when `spec` is `None`)

The suffix is honored end-to-end: training, `dalg-run-metrics assignments` /
`intrinsic-dim`, the standalone `cluster_assignments`, and
`dalg-label-mfa-clusters`. `gaussian-overlap` needs no filter (pure model
geometry).
Assignments save the resolved `subset_spec`, and labeling falls back to that
recorded value when `--shard-dir` carries no suffix, so positions are always
inverted through the same filtered meta. When no `#<spec>` is present behavior is
identical to before. To remove the feature: delete `subset_spec.py`, revert the
`split_shard_dir_spec` call sites to `Path(args.shard_dir)`, and drop the
`positions=` argument added to `stratified_split`.

## Core Model and Training Code

`src/dalg/models/mfa.py` contains:

- `MFA`: full single-process model
- `ComponentShardedMFA`: model-parallel MFA where each rank owns a slice of K
- `save_mfa`, `load_mfa`, `save_component_shard`, `load_component_shards`
- `component_shard_bounds(K, rank, world_size)`

Important model parameters:

- `mu`: component means, `(K, D)`
- `dir_raw` and derived `W`: local directions/loadings
- `scale_rho`: loading scales
- `psi_rho`: diagonal noise
- `pi_logits`: mixture weights

Likelihood and posterior computations use Woodbury-style small latent-space
operations because `q << D`.

`src/dalg/models/train.py` contains `train_nll`, the main optimizer loop. It is
distributed-aware:

- rank 0 handles printing, tqdm, W&B logging, and main checkpoint bookkeeping
- validation can use `val_tensor` or `val_loader`
- component-sharded training saves per-rank checkpoints and synchronizes
  gradients for replicated parameters via `sync_replicated_grads()`

No optimizer in this repo applies weight decay; `Adam` is constructed with its
default `weight_decay=0`. Shrinkage on `W` comes only from the ARD path below.

## Adaptive q

Two deliberately redundant stacks learn a **per-component rank** instead of
fixing it at `--rank`. Both leave `mfa.py`, `train.py`, and `run_training.py`
untouched, so each can evolve independently, and both are collected under
`adaptive_q/` subdirectories:

```text
src/dalg/models/adaptive_q/   mfa_ard, train_ard, mfa_hddc, hddc_surgery, train_hddc
src/dalg/cli/adaptive_q/      run_training_ard, run_training_hddc
scripts/adaptive_q/           toy-manifold runners
scripts/slurm/adaptive_q/     sbatch_train_ard.sh, sbatch_train_hddc.sh
```

These directories have no `__init__.py` and work as implicit namespace packages;
console scripts resolve through the full dotted path
(`dalg.cli.adaptive_q.run_training_hddc:main`). This is a temporary arrangement:
the intent is to converge on a single adaptive-rank model, at which point one of
the two stacks is deleted and the other folds back into `models/` and `cli/`.

### ARD prior path

Learns a per-component rank `q_k` through an ARD prior that shrinks whole
columns of `W_k`:

- `src/dalg/models/adaptive_q/mfa_ard.py`: `MFA_ARD(MFA)`, plus `save_mfa_ard` /
  `load_mfa_ard`
- `src/dalg/models/adaptive_q/train_ard.py`: `train_nll_ard`, `ard_beta_schedule`
- `src/dalg/cli/adaptive_q/run_training_ard.py`: `dalg-run-training-ard`
  (vanilla only)
- `scripts/slurm/adaptive_q/sbatch_train_ard.sh`
- `scripts/adaptive_q/train_ard_toy_manifolds.py`: the same model on toy-manifold
  `.pt` datasets, which the CLI cannot read because it expects activation shards

Invariants that matter when editing this path:

- Because `_dir_hat()` normalizes columns over `D`, `||w_j^k|| == scale_rho`
  exactly, so the ARD penalty is a function of `scale_rho` alone.
- `nu` is eliminated in closed form and detached each forward pass. This adds no
  parameters, so an `MFA_ARD` `state_dict` is identical to an `MFA` one and
  `load_mfa` reads ARD checkpoints unchanged — every downstream analysis works
  on ARD runs with no code changes.
- The ARD penalty is a prior over parameters while the loss is a per-sample
  mean, so the applied weight is `--ard-lambda / n_train_tokens`.
- Selection and early stopping use *validation NLL alone*, never the penalty, so
  ARD runs stay comparable to baseline MFA runs.
- Effective rank counts columns whose variance exceeds `--rank-threshold x
  mean(Psi_k)`. Do not switch this to a peak-relative cutoff: under column
  collapse every scale sits at the same floor, and a relative measure reports
  full rank exactly when the model is degenerate.
- Full ARD pressure from a cold start collapses all columns irrecoverably, so
  `ard_beta` ramps in over epochs. The horizon is stored in the checkpoint and a
  resume that changes it is rejected.
- Pruning is a post-training step only (`MFA_ARD.prune_columns`); the training
  loop never calls it.

### HDDC covariance-surgery path

A second route to a per-component rank, independent of ARD. SGD training is
unchanged; every `T` epochs the closed-form covariance update of the HDDC model
`[a_ij b_i Q_i d_i]` (Bouveyron, Girard & Schmid, arXiv:math/0604064)
re-estimates each component's covariance at an adaptive rank `d_k <= q_max` and
rewrites it in MFA parameters. Three phases: an E-pass accumulating the
responsibility-weighted second moment in float64, then per component an `eigh`
plus a scale-free Cattell scree test that picks `d_k` and the noise level
`b_k = (Tr(S_k) - sum_{j<=d_k} lam_j) / (D - d_k)`, then an Adam-state reset for
the rewritten tensors.

- `src/dalg/models/adaptive_q/mfa_hddc.py`: `MFA_HDDC` /
  `ComponentShardedMFA_HDDC` plus `save_mfa_hddc` / `load_mfa_hddc` and the
  component-shard pair. Unlike `MFA_ARD`, this is a self-contained fork of
  `mfa.py` rather than a subclass, because it changes the parameter shapes: it
  adds `isotropic_psi` (a `(K, 1)` `psi_rho`), a non-trainable `rank_mask`
  buffer `(K, q_max)`, and a `component_ranks` property returning
  `rank_mask.sum(-1)`. `EncodedBatch` / `MFAEncoderDecoder` are deliberately
  *not* forked; they call public methods only, so `mfa.MFAEncoderDecoder`
  accepts an `MFA_HDDC`.
- `src/dalg/models/adaptive_q/hddc_surgery.py`: `SurgeryConfig`, `hddc_surgery`,
  `accumulate_statistics`, `reconstruct_components`, `surgery_params`,
  `reset_optimizer_state`, `parameter_count`
- `src/dalg/models/adaptive_q/train_hddc.py`: `train_nll_hddc`, a copy of
  `train_nll` whose only difference is the `surgery=` argument and the block it
  gates
- `src/dalg/cli/adaptive_q/run_training_hddc.py`: `dalg-run-training-hddc`,
  adding `--isotropic-psi`, `--surgery-every-epochs`, `--surgery-threshold`,
  `--surgery-min-count`, `--surgery-warmup-steps`; `--rank` doubles as `--q-max`
- `scripts/slurm/adaptive_q/sbatch_train_hddc.sh`
- `tests/test_hddc_surgery.py`

Invariants that matter when editing this path:

- Surgery rewrites covariances only (`dir_raw`, `scale_rho`, `psi_rho`,
  `rank_mask`). `mu` and `pi_logits` keep whatever SGD made them, and their Adam
  state is preserved while the rewritten tensors' state is dropped.
- Statistics are centered on the *current* `mu_k`, never on the empirical
  responsibility-weighted mean. Pairing a covariance centered at `mu_hat_k` with
  a retained `mu_k` is inconsistent and leaks the mean shift into the spectrum,
  inflating apparent rank whenever the SGD means lag the data.
- `isotropic_psi` is required: the reconstruction `Sigma_k = W_k W_k^T + b_k I`
  is exact only for isotropic noise. The CLI rejects `--surgery-every-epochs`
  without `--isotropic-psi`.
- Masking is multiplicative, so masked columns get exactly zero gradient and no
  stop-gradient machinery is needed. All `q_max` columns are rewritten at every
  surgery and only the mask records `d_k`, so a rank *increase* needs no revival
  logic.
- Surgery is a partial M-step, so it competes for best-model selection on the
  same validation metric; otherwise a surgery landing on the final epoch would
  be discarded by the end-of-run rollback.
- `rank_mask` is part of the `state_dict`, sharded like the other per-component
  tensors. It and the `(K, 1)` psi_rho make an `MFA_HDDC` checkpoint unreadable
  by `mfa.load_mfa`, so downstream analyses do not consume HDDC runs — unlike the
  ARD path, whose `state_dict` is identical to a plain MFA one.
- Only D=128-scale data is supported: phase A accumulates an explicit
  `(K, D, D)` scatter. There is a TODO for the large-D sketching path.

To remove the feature: delete `models/adaptive_q/{mfa_hddc,hddc_surgery,train_hddc}.py`,
`cli/adaptive_q/run_training_hddc.py`, `scripts/slurm/adaptive_q/sbatch_train_hddc.sh`,
`tests/test_hddc_surgery.py`, and the `dalg-run-training-hddc` entry in
`pyproject.toml`. Nothing outside those files imports them.

`--surgery-every-epochs 0` gives a fixed-q baseline on the same stack, which is
the comparison an adaptive-rank claim needs. Validation data comes from the toy
manifold generator above; see `docs/experiments/hddc-rank-surgery.md`.

## Workflow Skills

Recurring operational procedures live in project skills so they are loaded only
when relevant. Codex discovers the canonical skills under `.agents/skills/`;
Claude Code reaches the same directories through `.claude/skills/` symlinks.

- `dalg-train-mfa`: activation extraction, vanilla training,
  component-sharded training, adaptive-rank ARD training, checkpoint/resume
  behavior, and training launch scripts.
- `dalg-compute-assignments`: source-agnostic hard partitions using either MFA
  responsibility argmax or nearest Euclidean centroids/medoids, plus stream
  alignment and assignment-bundle validation.
- `dalg-compute-metrics`: assignment-source-agnostic intrinsic dimension, MFA
  Gaussian overlap, description metrics, devices, and output policy.
- `dalg-label-mfa-clusters`: top activation contexts, selective or no-LLM
  labeling, cached indexes, and label validation.

Use only the skill relevant to the requested stage. Missing upstream artifacts
are prerequisites to report, not permission to run additional pipeline stages.

## Attachable Experimental Context

Temporary experiment workflows are intentionally not always-loaded guidance or
automatically triggered skills. Attach or read the relevant file only when the
user places that experiment in scope:

- `docs/experiments/wikipedia-kmedoids.md`
- `docs/experiments/synthetic-mfa.md`
- `docs/experiments/hddc-rank-surgery.md`

## Cluster and Scratch Notes

The user usually works on a Slurm cluster and often debugs via VS Code remote or
tunneling.

Common locations:

- `scripts/slurm/`: cluster job scripts
- `logs/jobs/` and `logs/experiments/`: current Slurm log targets
- `outputs/experiments/`: generated local/repo artifacts
- `dalg-cache/`: symlink to `/orfeo/scratch/dssc/zenocosini/dalg-cache/`

Large data generally belongs on scratch, not home storage. Prefer writing new
large artifacts under `dalg-cache/`. Existing legacy outputs under
`dalg-cache/output/` should remain untouched unless the user explicitly asks to
move or delete them.

Useful known scratch contents include:

- `dalg-cache/pile_gemma2b_100M_windows/merged/`
- `dalg-cache/pile_gemma2b_activations/`
- `dalg-cache/pile_gemma2b_models/`
- `dalg-cache/pile_gemma2b_activations_debug/`
- `dalg-cache/qk_sweep_exploration/`

Older direct aliases such as `/orfeo/scratch/dssc/zenocosini/pile_gemma2b_activations`
may still appear in scripts or logs. Check symlink targets before assuming data
is missing.

## Local Development

The repo commonly uses `.venv` or `uv run`.

Typical local setup:

```bash
source .venv/bin/activate
export PYTHONPATH=src
```

For Apple Silicon or mixed backends:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

Useful tests:

```bash
PYTHONPATH=src pytest tests/test_train.py
PYTHONPATH=src pytest tests/test_shard_activations
PYTHONPATH=src pytest tests/test_cluster_labeling.py
PYTHONPATH=src pytest tests/test_description_metrics.py
PYTHONPATH=src python -m torch.distributed.run --standalone --nproc_per_node=2 \
  tests/component_sharded_mfa_equivalence.py --device cpu --optimizer adam --steps 4
```

## Implementation Guidance

- Preserve the research-first style.
- Prefer direct code over abstractions unless the abstraction removes real
  complexity.
- Do not reintroduce stale imports like `from modeling...` or
  `from experiments...`.
- Do not reintroduce the old monolithic training path as the primary path.
  Current training expects activation shards through `--shard-dir`.
- Do not reintroduce DDP data-parallel training into `dalg-run-training`.
  Component sharding is model parallel over K.
- Keep command paths and Slurm scripts aligned with package entrypoints.
- Put reusable analysis logic under `src/dalg/analysis/`; expose a CLI only
  when it is useful as a standalone workflow.
- Comments and docstrings should explain what the code does, not narrate recent
  changes.
- Avoid generated outputs inside source folders.
- When I ask you to implement an experimental module the guiding principle is: **implement something easy to add and easy to remove**. Avoid overengineering or overgeneralizing. I prefere code redundacy rather than chaning the codebase to support a single experiment. 
- If you need to create scripts custom for a execute a specific experiment put under either scripts/temporary of scripts/slurm/temporary. Avoid cluttering the main scripts/ folder with one-off scripts.
- Reuse reuse reuse. I don't want you to reinvent the wheel prefer to reuse existing code and functions in this repo instead of writing new ones.
- This is code for  research not production so if the code halts or fails in some edge case it is ok. I prefer clear and strict contracts rather than loads of fallbacks and error handling. 