# AGENTS.md

This file gives future agents the minimum context needed to work effectively in
this repository.

## Project Goal

This is a machine learning research codebase for **"From Directions to Regions:
Decomposing Activations in Language Models via Local Geometry"**.

The only supported models are **MFA-EM (HDDC fitted with EM)** and
**KMeans+PCA**.

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
    adaptive_q/          MFA-EM (HDDC) training entrypoint
  data/                  Window builders, toy manifolds, and activation-shard streaming
  evaluation/            Toy-manifold geometry, tiling metrics, and pipeline evaluator
  init/                  Reservoir KMeans centroid initialization
  intervention/          Region and local-subspace steering
  llm/                   TransformerLens activation extraction wrapper
  models/                KMeans+PCA and MFA-EM models and training
    adaptive_q/          HDDC model and full-data EM trainer
  pipeline/              YAML planning, immutable manifests, and stage orchestration
    main.py              Readable planning and execution phases; helpers live beside it

configs/experiments/     YAML experiment and sweep configurations

scripts/
  slurm/                 Reusable extraction, training, assignment, metric, and labeling jobs
    adaptive_q/          HDDC training launcher (select EM explicitly)
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

The main workflow is **write a YAML config -> launch the pipeline -> aggregate
saved metrics**. The entrypoints are `dalg-run-pipeline` and
`dalg-aggregate-metrics`, defined in `pyproject.toml`. Use `$dalg-run-pipeline`
for execution and the [YAML reference](docs/reference/training-pipeline-config.md)
for exact fields.

1. **Write a config** under `configs/experiments/`. Set the experiment name and
   output root, existing activation shards and layer, model, training settings,
   and Slurm resources; add `sweep` axes when needed. Choose `model.kind: kmeans`
   or `model.kind: hddc` with `training.fit_method: em` and the isotropic-noise
   settings described below. Enable `assignments.enabled` and
   `evaluation.enabled` to produce metrics. Choose
   `evaluation.kind: toy_manifold_tiling` for toy geometry or
   `evaluation.kind: unsupervised` for
   ordinary activation shards. The pipeline does not extract activations.

2. **Inspect and launch on Slurm**, replacing the example paths with your config
   and the immutable manifest path printed by the command:

   ```bash
   uv run --locked dalg-run-pipeline submit path/to/config.yaml --dry-run
   uv run --locked dalg-run-pipeline submit path/to/config.yaml
   uv run --locked dalg-run-pipeline status --manifest path/to/manifest.jsonl
   ```

   Each run performs any required initialization, training, assignments, and
   configured evaluation. For a local or interactive allocation, use
   `dalg-run-pipeline plan path/to/config.yaml`, then
   `dalg-run-pipeline run --manifest path/to/manifest.jsonl --index 0` for the
   selected manifest row. Reusing the same manifest resumes incomplete work
   and skips stages with valid completed artifacts.

3. **Aggregate metrics after evaluation finishes.** Pass the experiment folder
   containing the run directories (`<output_root>/<experiment-name>`):

   ```bash
   uv run --locked dalg-aggregate-metrics path/to/experiment
   ```

   This collects saved `metrics.json` reports into `general_metrics.csv` and
   `manifold_metrics.csv` in that folder. Each report contributes one general
   row and, when available, one row per manifold. Aggregation does not recompute
   metrics or average across runs.

See the [pipeline workflow](docs/workflows/training-pipeline.md) for launch,
resume, and saved-run reevaluation details. Lower-level entrypoints remain
`dalg-run-extraction`, `dalg-run-training-kmeans`, and
`dalg-run-training-hddc --fit-method em`.

## Data Layout

### Toy-manifold datasets

Before generating, storing, or changing synthetic local-geometry data, read
`docs/reference/toy-manifold-dataset.md`. It owns the generator configuration,
determinism and paired-condition rules, return metadata, activation-compatible
shard format, storage policy, and links to downstream tiling evaluation.

### SAE-manifold prompts

We use SAE-manifold prompt datasets inspired by Goodfire's paper
"Do Sparse Autoencoders Capture Concept Manifolds?".

## Core Model and Training Code

`src/dalg/models/kmeans.py` provides the pipeline KMeans+PCA model. Read
[the model contract](docs/models/kmeans.md) for its one-hot responsibilities,
full cluster PCA, Cattell rank masks, and checkpoint behavior. New pipeline
initialization uses `kmeans_model.pt` directly; legacy centroid bundles and
external centroid-based KMeans workflows are deprecated.

`src/dalg/models/adaptive_q/mfa_hddc.py` provides `MFA_HDDC`, its per-component
rank masks, and the `save_mfa_hddc` / `load_mfa_hddc` checkpoint helpers.
`src/dalg/models/adaptive_q/train_em_hddc.py` provides `train_em_hddc`, the
full-data EM trainer with KMeans initialization, Cattell rank selection,
convergence checks, and resumable checkpoints.

Before editing MFA-EM, read the
[HDDC EM contract](docs/models/mfa-hddc.md#streamed-full-data-em). EM supports
one CPU or CUDA process with shared or component-specific isotropic noise.

Always select EM explicitly: use `dalg-run-training-hddc --fit-method em`, or
set `model.kind: hddc` and `training.fit_method: em` in pipeline YAML. Select
exactly one noise mode: `model.shared_b: true` or `model.isotropic_psi: true`
(`--shared-b` or `--isotropic-psi` on the CLI).

## Workflow Skills

Recurring operational procedures live in project skills so they are loaded only
when relevant. Codex discovers the canonical skills under `.agents/skills/`;
Claude Code reaches the same directories through `.claude/skills/` symlinks.

- `dalg-run-pipeline`: configure, launch, resume, and inspect KMeans+PCA or
  MFA-EM pipelines, and reevaluate saved runs.
- `dalg-compute-assignments`: compute, inspect, or repair hard assignments using
  MFA responsibility argmax or nearest Euclidean centroids.
- `dalg-wiki`: navigate and maintain the Markdown wiki under `docs/`.

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
PYTHONPATH=src pytest tests/test_hddc_em.py
PYTHONPATH=src pytest tests/test_kmeans.py
PYTHONPATH=src pytest tests/test_shard_activations
PYTHONPATH=src pytest tests/test_aggregate_metrics.py
```

## Implementation Guidance
- Prefer direct code over abstractions unless the abstraction removes real
  complexity.
- Put reusable analysis logic under `src/dalg/analysis/`; expose a CLI only
  when it is useful as a standalone workflow.
- Comments and docstrings should explain what the code does, not narrate recent
  changes.
- Avoid generated outputs inside source folders.
- When I ask you to implement an experimental module the guiding principle is: **implement something easy to add and easy to remove**. Avoid overengineering or overgeneralizing. I prefere code redundacy rather than chaning the codebase to support a single experiment.
- If you need to create scripts custom for a execute a specific experiment put under either scripts/temporary of scripts/slurm/temporary. Avoid cluttering the main scripts/ folder with one-off scripts.
- Reuse reuse reuse. I don't want you to reinvent the wheel prefer to reuse existing code and functions in this repo instead of writing new ones.
- This is code for  research not production so if the code halts or fails in some edge case it is ok. I prefer clear and strict contracts rather than loads of fallbacks and error handling.
- When refactoring code leverage how much ensure backward compatibility and deprecating old code. Since this is a research codebase I am less concerned about backward compatibility and I am ok to let some old code become deprecated. I always in the process of understand what is going to be the right set of experiments, so I prefer adapt the code to the current

# AGENTS CHAT
If you are an agent and want to communicate with other agents use `AGENTS_chat.md`. There you can chat with other AGENTS, or leave messages / suggestions / instructions for future iteration of yourself.
