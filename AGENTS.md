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

Use `$dalg-wiki` whenever a task requires navigating, reading, writing, or
reorganizing repository documentation. The skill starts at `docs/README.md` and
routes agents to the relevant research state, model explanation, workflow,
reference, evaluation contract, or attachable experiment context.

## Repository Layout

The repository uses a `src/` layout. The tree below lists maintained paths that
are useful for navigation; local environments, Python caches, W&B run folders,
and package metadata are intentionally omitted.

```text
src/dalg/
  analysis/              Assignments, overlap, intrinsic dimension, labels, and description metrics
  cli/                   Console-script entrypoints
    adaptive_q/          ARD and HDDC training entrypoints
  data/                  Window builders, toy manifolds, and activation-shard streaming
  evaluation/            Toy-manifold geometry, tiling metrics, and pipeline evaluator
  init/                  Reservoir KMeans centroid initialization
  intervention/          Region and local-subspace steering
  llm/                   TransformerLens activation extraction wrapper
  models/                MFA models and training loops
    adaptive_q/          ARD and HDDC per-component-rank variants
  pipeline.py            YAML planning, immutable manifests, and stage orchestration

configs/experiments/     YAML experiment and sweep configurations

scripts/
  slurm/                 Reusable extraction, training, assignment, metric, and labeling jobs
    adaptive_q/          ARD and HDDC training jobs
    experimental/        Launchers for named experimental workflows
    temporary/           One-off or short-lived cluster jobs
  temporary/             One-off experiment and repair scripts
  synthetic_dataset/     Synthetic-MFA experiment scripts
  pca_size_validation/   Scoped PCA-size validation study
  distributed_training/  Standalone distributed-training examples and exercises

tests/                   Unit, smoke, distributed-equivalence, and synthetic-fixture tests
notebooks/               Active exploratory notebooks and visualizations
  archived/              Retained notebooks that are no longer active entrypoints

docs/
  README.md              Canonical task-oriented documentation hub
  research/              Current direction, backlog, and research snapshots
  models/                Durable model explanations and invariants
  workflows/             Task-oriented operational guides
  reference/             Exact configuration, schema, and interface contracts
  evaluation/            Metric definitions and evaluator output contracts
  experiments/           Attachable context for temporary or named experiments

.agents/skills/           Canonical recurring workflow skills for Codex and Claude
.claude/skills/           Symlinks exposing those skills to Claude Code

outputs/                  Generated local reports and experiment artifacts
logs/                     Slurm job and experiment logs
dalg-cache/               Symlink to large scratch datasets, models, and derived artifacts

pyproject.toml            Package metadata and console-script definitions
uv.lock                   Locked Python environment
README.md                 Short public project overview
AGENTS.md                 Canonical instructions for repository agents
CLAUDE.md                 Claude entrypoint that delegates to AGENTS.md
```

Generated artifact boundaries and the important scratch subdirectories are
documented under [Cluster and Scratch Notes](#cluster-and-scratch-notes).

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

### Toy-manifold datasets

Before generating, storing, or changing synthetic local-geometry data, read
`docs/reference/toy-manifold-dataset.md`. It owns the generator configuration,
determinism and paired-condition rules, return metadata, activation-compatible
shard format, storage policy, and links to downstream tiling evaluation.

### Activation shards

Large language-model runs use activation shards produced by
`dalg-run-extraction`.

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

The repository currently has two deliberately independent experimental routes
to per-component rank. Before editing either one, read its canonical wiki page:

- [MFA-ARD](docs/models/mfa-ard.md) owns the shrinkage objective, warmup and
  selection rules, effective-rank definition, checkpoint compatibility, code
  organization, and failure modes.
- [MFA-HDDC](docs/models/mfa-hddc.md) owns isotropic-noise models, covariance
  surgery, rank-mask and optimizer invariants, execution modes, checkpoint
  compatibility, code organization, and failure modes.

Use [HDDC Rank Surgery](docs/experiments/hddc-rank-surgery.md) or the
[Adaptive-q Technical Card](docs/experiments/adaptive-q-technical-card.md) only
when that specific experiment or its measured results are in scope.

## Workflow Skills

Recurring operational procedures live in project skills so they are loaded only
when relevant. Codex discovers the canonical skills under `.agents/skills/`;
Claude Code reaches the same directories through `.claude/skills/` symlinks.

- `dalg-run-pipeline`: configure vanilla or adaptive-rank MFA training, then
  plan, submit, resume, and inspect the manifest-based pipeline with optional
  assignments and toy-manifold tiling evaluation.
- `dalg-wiki`: navigate, read, and maintain the task-routed Markdown wiki under
  `docs/`, including its index, document roles, and cross-links.
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
