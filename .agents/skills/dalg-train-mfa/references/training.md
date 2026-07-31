# Training reference

## Select the requested stage

- Build windows only when the user needs a token-window dataset.
- Extract activations only when the requested shard data does not exist or extraction itself is the task.
- Train only the requested model and mode. Do not run metrics afterward unless requested.

## Activation extraction

Use the current `dalg-run-extraction` entrypoint:

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

Expected shard layout:

```text
<root>/config.json
<root>/layerNN/shard_NNNNN.pt
<root>/tokens/shard_NNNNN.pt
<root>/meta/shard_NNNNN.json
```

Extraction is intended to be resume-safe. Inspect `config.json` and progress metadata before restarting it. The main Slurm launcher is `scripts/slurm/sbatch_extract_activations.sh`.

## Vanilla training

Use one process holding the full MFA:

```bash
uv run dalg-run-training \
  --shard-dir /path/to/activation_shards \
  --layer 5 \
  --out-dir /path/to/model_runs/layer05_1000_10_mfa \
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

Use `scripts/slurm/sbatch_train_shards.sh` for the cluster. Do not launch vanilla mode under `torchrun`; argument validation rejects `WORLD_SIZE > 1`.

Typical outputs are `config.json`, `val_indices.json`, `centroids.pt`, `checkpoint.pt`, and `mfa_model.pt`.

## Component-sharded training

Use model parallelism over a contiguous component slice per rank:

```bash
uv run python -m torch.distributed.run --standalone --nproc_per_node=2 \
  -m dalg.cli.run_training \
  --shard-dir /path/to/activation_shards \
  --layer 5 \
  --out-dir /path/to/model_runs/layer05_8000_10_component_sharded_mfa \
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

Use `scripts/slurm/sbatch_train_component_shards.sh`. CUDA and `WORLD_SIZE > 1` are required. More GPUs reduce per-rank component memory; they do not change the logical batch size.

Expected outputs include per-rank checkpoints and models plus `checkpoint_shards.json` and `mfa_model_shards.json`. `load_mfa(<run_dir>/mfa_model.pt)` can fall back to the shard manifest, but assembling a large full model is memory-heavy.

## Adaptive-rank (ARD) training

Use when the per-component rank `q_k` should be learned instead of fixed. The
model is `MFA_ARD`, which adds an ARD prior on the columns of each `W_k`; the
stack is deliberately redundant with the baseline one and lives in
`src/dalg/models/mfa_ard.py`, `src/dalg/models/train_ard.py`, and
`src/dalg/cli/run_training_ard.py`. Single-process only — there is no
component-sharded ARD variant.

```bash
uv run dalg-run-training-ard \
  --shard-dir /path/to/activation_shards \
  --layer 5 \
  --out-dir /path/to/model_runs/layer05_1000_64_mfa_ard \
  --K 1000 \
  --rank 64 \
  --alpha0 1.0 \
  --b0 1e-4 \
  --ard-lambda 1.0 \
  --rank-threshold 1.0 \
  --epochs 20 \
  --batch-size 2048 \
  --val-frac 0.008 \
  --device cuda \
  --seed 42
```

Use `scripts/slurm/sbatch_train_ard.sh` for the cluster.

`--rank` is the **maximum** rank per component; ARD prunes below it, so set it
generously and read the learned rank off the `q_eff` logs.

### Prior and pressure

- `--alpha0` / `--b0` are the Gamma prior on the column precision `nu`. `nu`
  itself is eliminated in closed form each step, so it adds no parameters.
- `--ard-lambda` scales the penalty; the applied weight is
  `lambda / n_train_tokens`, because the penalty is a prior over parameters
  while the loss is a per-sample mean. Expect to sweep lambda over orders of
  magnitude before `q_eff` moves.
- `--b0` sets the payoff for zeroing a column, roughly
  `(D/2 + alpha0 - 1) * log(0.5 * s^2 / b0)` nats. Smaller `b0` means a deeper
  prize and a stiffer trap near `s = 0`; larger `b0` prunes less.

### Beta warmup schedule

Applying full pressure from a cold start collapses every column into the
penalty's `s -> 0` well before the loadings align with any data direction, and
that state is not recoverable. `ard_beta` therefore ramps in:

- `--ard-warmup-frac` (0.15): fraction of epochs at `ard_beta = 0`
- `--ard-ramp-frac` (0.20): fraction over which beta ramps linearly to 1
- `--ard-schedule-epochs`: horizon the fractions are measured against; defaults
  to `--epochs`, and is **required** when `--epochs <= 0` unless
  `--ard-lambda 0`

The horizon is stored in `checkpoint.pt` and a resume that computes a different
one is rejected with an error naming the stored value. To raise the epoch cap on
an existing run, pass `--ard-schedule-epochs <stored value>` so the original ramp
is preserved. The check is skipped when `--ard-lambda 0`.

Known gap: early stopping ends a run before `--epochs`, so the ramp is sized
against the requested budget rather than the realized one.

### Collapse diagnosis

Watch `q_eff`, `ard/psi_mean`, and `ard/dead_components`. Psi inflating by orders
of magnitude while `q_eff` falls to zero is the collapse signature, and Psi moves
first. To tell a recoverable trap from a genuinely excessive lambda, score a
non-collapsed model — from a lower-lambda run or the `--ard-lambda 0` baseline —
under the collapsing run's lambda by setting `model.ard_weight` and comparing
`nll + ard_weight * ard_penalty`:

- collapsed objective worse: the run fell into a trap, lengthen the warmup
- collapsed objective better: lambda is too high for this `b0`

A collapsed run yields no usable comparison point of its own; collapse can occur
within the first epochs, so neither its checkpoint nor its best-by-val model is
pre-collapse. Raising `--ard-warmup-frac` can preserve real structure even where
collapse is the global optimum, landing in a stable local optimum instead — but
too long a warmup leaves no epochs under pressure and nothing is pruned.

### Pruning

Pruning runs **after** training and after the best-epoch rollback, never inside
the loop. `--prune-at-end` (default on, `--no-prune-at-end` to disable) zeros
every column below `--rank-threshold`.

- A column counts toward `q_k` when its variance exceeds
  `rank_threshold * mean(Psi_k)` — measured against the noise floor, not against
  the component's largest column, so a collapsed component reports `q_k = 0`
  rather than full rank.
- Pruning zeros both `scale_rho` and `dir_raw`, so the `W` column is exactly
  zero while the `(K, D, q)` shape is preserved.
- Outputs: `mfa_model.pt` is the pruned model, `mfa_model_unpruned.pt` the
  pre-prune copy. The run prints validation NLL before and after; a material
  jump means `--rank-threshold` was too aggressive.

### Outputs and compatibility

Typical outputs match the vanilla run plus `mfa_model_unpruned.pt`. Because the
closed-form `nu` adds no parameters, an ARD checkpoint has the same `state_dict`
as a plain MFA: `load_mfa` reads it directly and all downstream analysis
(assignments, Gaussian overlap, intrinsic dim, labeling) works unchanged. Use
`load_mfa_ard` when the ARD hyperparameters are needed too. `config.json` and
checkpoint meta record `alpha0`, `b0`, `ard_lambda`, `ard_weight`, the schedule
fractions, `effective_ranks`, and whether the model was pruned.

## Centroids, subsets, and resume behavior

- Reuse centroids with `--centroids-path`; shared centroid collections live under `dalg-cache/pile_gemma2b_models/centroids/`.
- A shard suffix such as `#pile_wikipedia_1M` deterministically selects a token-budgeted subset without copying activations.
- The spec resolves to sorted positions in canonical stream order and must stay consistent downstream.
- Inspect checkpoints and completion artifacts before deciding whether to resume, skip, or start a new output directory.
- The raw CLI fallback may place output below `--shard-dir`; for the main Gemma workflow, always pass a model directory under `dalg-cache/pile_gemma2b_models/`.

## Validation

For component-sharded changes, use:

```bash
PYTHONPATH=src python -m torch.distributed.run --standalone --nproc_per_node=2 \
  tests/component_sharded_mfa_equivalence.py --device cpu --optimizer adam --steps 4
```

Use targeted training and shard-streaming tests for local code changes. Avoid launching a large cluster job merely to validate shell syntax or argument wiring.

For ARD changes:

```bash
PYTHONPATH=src uv run python -m pytest tests/test_mfa_ard.py -q
PYTHONPATH=src uv run python -m pytest tests/test_train.py -q   # baseline path must stay untouched
```

`tests/fixtures/multi_shard` (built by `tests/synthetic_shards.py`) is large
enough for a CPU end-to-end CLI smoke test of the schedule, the horizon guard,
and pruning.
