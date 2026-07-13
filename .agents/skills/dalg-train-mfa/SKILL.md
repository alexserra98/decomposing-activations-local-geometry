---
name: dalg-train-mfa
description: Prepare activation shards and train or resume DALG Mixture of Factor Analyzers in vanilla or component-sharded mode. Use when extracting Gemma activations, initializing centroids, launching MFA training, choosing a training mode, resuming checkpoints, editing the training Slurm jobs, or debugging training output and distributed launch behavior.
---

# Train DALG MFA Models

Perform only the preparation or training stage the user requested. Do not launch extraction before training, retrain completed runs, or compute downstream analysis unless explicitly asked.

## Procedure

1. Read `references/training.md` before planning or executing extraction or training.
2. Inspect the current CLI definitions in `pyproject.toml` and the relevant script in `scripts/slurm/`; treat repository code as authoritative when it differs from examples.
3. Identify the activation shard directory, layer, output model directory, `K`, latent rank, training mode, and available device topology.
4. Check existing shard configuration, centroids, checkpoints, and model outputs before writing anything. Preserve existing runs unless the user explicitly authorizes replacement.
5. Use vanilla mode for one process holding the full MFA. Use component-sharded mode only with `torchrun`, CUDA, and more than one process.
6. Keep activation data under `dalg-cache/pile_gemma2b_activations/` and model runs under `dalg-cache/pile_gemma2b_models/`.
7. Run a proportionate smoke test or validation after changing training code or launch scripts.

## Invariants

- Treat component sharding as model parallelism over mixture components, not DDP data parallelism.
- Ensure every component-sharded rank receives the same activation batch.
- Do not gather a large component-sharded run into one full checkpoint.
- Wrap `ActivationBatchDataset` with `DataLoader(batch_size=None)` because the dataset is already batched.
- Preserve subset-suffix selection consistently across training and downstream analysis.
- Prefer existing CLI entrypoints and Slurm scripts over handwritten launch variants.
