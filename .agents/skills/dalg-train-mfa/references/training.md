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
