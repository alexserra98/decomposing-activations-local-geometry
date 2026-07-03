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
- the model supports cluster/region analysis, overlap metrics, intrinsic
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
  analysis/       Overlap, intrinsic dim, assignments, labels, description metrics
  intervention/   Region/subspace steering code

scripts/slurm/    Cluster job scripts for extraction, training, metrics, labels
scripts/          Temporary/profiling/synthetic-analysis scripts
tests/            Unit/smoke tests and synthetic shard fixtures
notebooks/        Exploratory notebooks
outputs/          Generated local artifacts; may contain untracked reports
logs/             Slurm logs in current scripts
dalg-cache/       Symlink to scratch cache
output/           Symlink to scratch output artifacts
```

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
- `dalg-cluster-overlap`
- `dalg-cluster-intrinsic-dim`
- `dalg-build-pile-windows`
- `dalg-build-newsgroups-windows`

The main workflow is:

```text
token windows dataset
  -> activation shards
  -> centroid init
  -> MFA training
  -> assignments / overlap / intrinsic dim
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
`dalg-label-mfa-clusters`. `overlap` needs no filter (pure model geometry).
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

## Recurring Workflow: Train a New MFA

Most training starts from an existing activation shard directory. If activations
do not exist yet, build token windows first and run extraction:

```bash
uv run dalg-run-extraction \
  --dataset /path/to/windows_dataset \
  --out-dir /path/to/activation_shards \
  --model google/gemma-2b \
  --layers 5 17 \
  --mode residual \
  --extract-batch-size 16 \
  --shard-size 512 \
  --dtype float16 \
  --drop-prefix 32 \
  --device cuda
```

Extraction writes `config.json`, per-layer activation shards, token shards, and
metadata. It is designed to be resume-safe.

### Vanilla Training

Use vanilla mode for one process holding the full MFA on one GPU:

```bash
uv run dalg-run-training \
  --shard-dir /path/to/activation_shards \
  --layer 5 \
  --out-dir /path/to/activation_shards/layer05_1000_10_mfa \
  --K 1000 \
  --rank 10 \
  --epochs 20 \
  --refine-epochs 10 \
  --batch-size 2048 \
  --num-workers 2 \
  --val-frac 0.008 \
  --split-seed 42 \
  --device cuda \
  --seed 42 \
  --training-mode vanilla
```

Relevant Slurm script:

- `scripts/slurm/sbatch_train_shards.sh`

Vanilla outputs usually include:

- `config.json`
- `val_indices.json`
- `centroids.pt`
- `checkpoint.pt`
- `mfa_model.pt`

Notes:

- Do not launch vanilla mode under `torchrun`; `validate_args` rejects
  `WORLD_SIZE > 1`.
- `--centroids-path` can reuse an existing `centroids.pt` or directory
  containing one.
- If `--out-dir` is omitted, the default is under `--shard-dir` with a
  layer/K-based name.
- W&B is opt-in with `--wandb --wandb-project ... --wandb-name ...`.

### Component-Sharded Training

Use component-sharded mode when the full K x D x q model is too large for one
GPU. This is model parallelism over mixture components, not data parallelism:
every rank sees the same activation batch, and each rank owns a contiguous
component slice.

```bash
uv run python -m torch.distributed.run --standalone --nproc_per_node=2 \
  -m dalg.cli.run_training \
  --shard-dir /path/to/activation_shards \
  --layer 5 \
  --out-dir /path/to/activation_shards/layer05_8000_10_component_sharded_mfa \
  --K 8000 \
  --rank 10 \
  --epochs 15 \
  --refine-epochs 10 \
  --batch-size 8192 \
  --num-workers 4 \
  --val-frac 0.05 \
  --split-seed 42 \
  --early-stop-delta 1e-3 \
  --device cuda \
  --seed 42 \
  --training-mode component_shard \
  --compile
```

Relevant Slurm script:

- `scripts/slurm/sbatch_train_component_shards.sh`

Component-sharded outputs include:

- `config.json`
- `val_indices.json`
- `centroids.pt`
- `checkpoint_rank0000.pt`, `checkpoint_rank0001.pt`, ...
- `checkpoint_shards.json`
- `mfa_model_rank0000.pt`, `mfa_model_rank0001.pt`, ...
- `mfa_model_shards.json`

Notes:

- Launch with `torchrun`; `--device cuda` and `WORLD_SIZE > 1` are required.
- Increasing GPUs reduces per-rank component memory. It does not increase the
  logical batch size.
- `_ComponentShardLoader` broadcasts rank-0 training batches so all ranks train
  on identical data.
- Validation currently uses a deterministic `val_loader` on every rank, so each
  rank participates in the same distributed likelihood calls.
- `load_mfa(<run_dir>/mfa_model.pt)` falls back to assembling from
  `<run_dir>/mfa_model_shards.json` when the `.pt` file is absent. This can be
  memory-heavy for large K.
- Do not gather and save a single full checkpoint for large component-sharded
  runs.

Useful smoke test:

```bash
PYTHONPATH=src python -m torch.distributed.run --standalone --nproc_per_node=2 \
  tests/component_sharded_mfa_equivalence.py --device cpu --optimizer adam --steps 4
```

## Recurring Workflow: Compute Metrics

Use `dalg-run-metrics` for current metric workflows. `--data-dir` accepts either
a run directory containing `mfa_model.pt` / `mfa_model_shards.json`, or a direct
path to a `.pt` model file.

### Overlap

```bash
uv run dalg-run-metrics overlap \
  --data-dir /path/to/mfa_run \
  --out-dir /path/to/output_dir \
  --device cuda \
  --batch-pairs 512
```

Output:

- `overlap.pt`

For high-rank models, reduce `--batch-pairs` to avoid GPU OOM. The Slurm metrics
script uses `512` for large `q`.

### Intrinsic Dimension

Intrinsic dimension can stream from activation shards and can optionally reuse
precomputed assignments:

```bash
uv run dalg-run-metrics intrinsic-dim \
  --data-dir /path/to/mfa_run \
  --shard-dir /path/to/activation_shards \
  --layer 5 \
  --out-dir /path/to/output_dir \
  --device cuda \
  --pca-device cpu \
  --pca-workers 8 \
  --assignments-path /path/to/mfa_model_assignments.pt \
  --variance-threshold 0.90 \
  --min-population 100 \
  --max-samples-per-cluster 2000
```

Output:

- `intrinsic_dims.pt`

Relevant Slurm script:

- `scripts/slurm/sbatch_metrics.sh`

### Description Metrics

After clusters have labels, `dalg-run-metrics` can score or compare label text:

```bash
uv run dalg-run-metrics description-fit \
  --labels-path /path/to/cluster_labels.json \
  --out-dir /path/to/cluster_labels \
  --positive-examples 8 \
  --negative-examples 8 \
  --judge-workers 4

uv run dalg-run-metrics description-semantics \
  --labels-path /path/to/cluster_labels.json \
  --out-dir /path/to/cluster_labels \
  --embedding-device cpu \
  --top-k 25 \
  --similarity-threshold 0.70
```

Outputs:

- `description_fit.json`
- `description_semantics.pt` and related JSON summaries

## Recurring Workflow: Compute Assignments

Assignments stream activation shards through a trained MFA, compute
responsibilities, store the hard argmax cluster for each token, and accumulate
cluster sizes plus responsibility-peakedness statistics.

Preferred current command:

```bash
uv run dalg-run-metrics assignments \
  --data-dir /path/to/mfa_run \
  --shard-dir /path/to/activation_shards \
  --layer 5 \
  --batch-size 1024 \
  --device cuda
```

Equivalent direct module:

```bash
uv run python -m dalg.analysis.cluster_assignments \
  --model-path /path/to/mfa_run/mfa_model.pt \
  --shard-dir /path/to/activation_shards \
  --layer 5 \
  --batch-size 1024 \
  --device cuda
```

Default output:

- `<run_dir>/mfa_model_assignments.pt`

Saved fields:

- `cluster_sizes`: `(K,)`
- `assignments`: `(N,)` hard cluster id for each streamed token
- `max_responsibilities`: `(N,)`
- `peakedness`: per-cluster means for entropy, one-minus-max, and top1-minus-top2
- `K`

Relevant Slurm scripts:

- `scripts/slurm/sbatch_assignments.sh`
- `scripts/slurm/sbatch_epoch_assignments.sh`

Notes:

- `--drop-prefix` defaults to the value in `<shard-dir>/config.json`.
- Use `--max-batches` for smoke tests.
- Assignments are useful before intrinsic-dim and required by the preferred
  label workflow.

## Recurring Workflow: Label MFA Gaussians

The preferred labeling path starts from assignments, finds the top activation
examples per cluster, recovers token-window contexts from the HF windows
dataset, and optionally calls the Orfeo-hosted LLM to produce labels.

```bash
uv run dalg-label-mfa-clusters \
  --assignments-path /path/to/mfa_run/mfa_model_assignments.pt \
  --shard-dir /path/to/activation_shards \
  --layer 5 \
  --windows-dataset /path/to/windows_dataset/merged \
  --tokenizer google/gemma-2b \
  --out-dir /path/to/output_dir/cluster_labels \
  --top-n 50 \
  --max-examples-per-cluster 25 \
  --pad 10 \
  --chunk-size 2000000 \
  --llm-workers 4 \
  --llm-temperature 0.0 \
  --llm-max-tokens 512
```

Outputs:

- `top_activations.pt`
- `cluster_examples.json`
- `cluster_labels.json`

Relevant Slurm script:

- `scripts/slurm/sbatch_label_mfa_clusters.sh`

Useful options:

- `--skip-llm`: build top activations and context examples without calling the LLM
- `--clusters 1 2 3`: label specific clusters
- `--max-clusters N`: debug on first N clusters
- `--top-index-path`: reuse or control the cached top-activation index path

`dalg-interpret-mfa` is an older/more integrated interpretation CLI. It can use
an assignments file when present, otherwise it falls back to scanning the model
for top responsibilities. Prefer `dalg-label-mfa-clusters` when assignments have
already been computed.

## Temporary Workflow: Synthetic MFA Analyses

There is an active temporary research workflow under `scripts/` and
`notebooks/` for synthetic MFA experiments. Treat it as analysis code, not core
library API.

Main script:

- `scripts/synthetic_mfa_qk_sweep.py`

Purpose:

- generate data from a known MFA
- fit MFA models over a grid of fitted `K` and `q`
- record responsibility peakiness and label-recovery metrics
- collect results and plots

Typical commands:

```bash
PYTHONPATH=src python scripts/synthetic_mfa_qk_sweep.py generate-dataset \
  --dataset-path /orfeo/scratch/dssc/zenocosini/dalg-cache/assets/synthetic_mfa_Ktrue1000_qtrue20_D500_seed0.pt \
  --D 500 --K-true 1000 --q-true 20 \
  --n-train 500000 --n-test 10000 --seed 0

PYTHONPATH=src python scripts/synthetic_mfa_qk_sweep.py fit-one \
  --dataset-path /path/to/synthetic_dataset.pt \
  --model-root dalg-cache/qk_sweep_exploration \
  --run-name Ktrue1000_qtrue20 \
  --K-fit 1250 --q-fit 20 \
  --device cuda

PYTHONPATH=src python scripts/synthetic_mfa_qk_sweep.py collect-results \
  --model-root dalg-cache/qk_sweep_exploration \
  --run-name Ktrue1000_qtrue20
```

Related scripts:

- `scripts/slurm/sbatch_synthetic_qk_sweep.sh`: Slurm array over fitted K, with
  an inner loop over q values
- `scripts/synthetic_mfa_bhattacharyya_by_q.py`: post-hoc overlap/Bhattacharyya
  summaries across q for a fixed fitted K
- `scripts/synthetic_mfa_feature_splitting.py`: feature-splitting and covariance
  reconstruction analysis over fitted sweep models
- `scripts/slurm/sbatch_feature_splitting.sh`: Slurm wrapper for feature
  splitting
- `notebooks/synthetic_mfa_qk_sweep_results.ipynb`: exploratory result notebook
- `outputs/experiments/synthetic_qk_sweep_report/`: offline report artifacts

Generated synthetic models can be very large and live under `dalg-cache/`.
Do not delete or overwrite them unless the user explicitly asks.

## Cluster and Scratch Notes

The user usually works on a Slurm cluster and often debugs via VS Code remote or
tunneling.

Common locations:

- `scripts/slurm/`: cluster job scripts
- `logs/jobs/` and `logs/experiments/`: current Slurm log targets
- `outputs/experiments/`: generated local/repo artifacts
- `dalg-cache/`: symlink to `/orfeo/scratch/dssc/zenocosini/dalg-cache/`
- `output/`: symlink to scratch output artifacts

Large data generally belongs on scratch, not home storage. Prefer writing new
large artifacts under `dalg-cache/` or `output/`. Do not delete scratch data or
large experiment outputs unless explicitly asked.

Useful known scratch contents include:

- `dalg-cache/pile_gemma2b_100M_windows/merged/`
- `dalg-cache/pile_gemma2b_activations/`
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
