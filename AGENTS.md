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
  models/         MFA model, component-sharded MFA, training loop
  init/           Reservoir KMeans centroid initialization
  data/           Window builders and activation-shard streaming
  llm/            TransformerLens activation extraction wrapper
  analysis/       Gaussian overlap, intrinsic dim, assignments, labels, description metrics
  intervention/   Region/subspace steering code

scripts/slurm/    Cluster job scripts for extraction, training, metrics, labels
scripts/          Temporary/profiling/synthetic-analysis scripts
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

## Workflow Skills

Recurring operational procedures live in project skills so they are loaded only
when relevant. Codex discovers the canonical skills under `.agents/skills/`;
Claude Code reaches the same directories through `.claude/skills/` symlinks.

- `dalg-train-mfa`: activation extraction, vanilla training,
  component-sharded training, checkpoint/resume behavior, and training launch
  scripts.
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
- When I ask you to implement an experimental module the guiding principle is: **implement something easy to add and easy to remove**. Avoid overengineering or overgeneralizing.
- If you need to create scripts custom for a execute a specific experiment put under either scripts/temporary of scripts/slurm/temporary. Avoid cluttering the main scripts/ folder with one-off scripts.
