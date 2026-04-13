# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research codebase for **"From Directions to Regions: Decomposing Activations in Language Models via Local Geometry"**. The core idea is to model LLM residual-stream activations as a Mixture of Factor Analyzers (MFA) — each component defines a region (centroid) plus a low-rank local subspace (loading matrix W_k). This supports interpretability, visualization, and activation steering.

## Environment Setup

```bash
source .venv/bin/activate
uv pip install -r requirements.txt
```

On Apple Silicon, set before running any model code:
```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```
To add new packages use:
```
uv add ...
```
## Running the Tutorial

```bash
# As a Python script (requires %magic comments stripped or use ipython):
jupyter notebook mfa_tutorial.ipynb

# Or run the .py version directly in an environment with ipykernel:
python mfa_tutorial.py
```

## Architecture

The pipeline flows in one direction: **data → activations → initialization → MFA training → interpretation/steering**.

### Data (`data_utils/concept_dataset.py`)
Two dataset classes share a `get_batches(batch_size) -> List[dict]` interface:
- `ConceptDataset`: unsupported prompts only; batches yield `{"prompt": [...]}`. Accepts CSV, JSON, JSONL.
- `SupervisedConceptDataset`: (prompt, label) pairs; batches yield `{"prompt": [...], "label": [...]}`. Accepts CSV or JSON. JSON can be a list of dicts or a `{label: [prompts]}` dict.

### Activation Extraction (`llm_utils/activation_generator.py`)
`ActivationGenerator` wraps a TransformerLens `HookedTransformer`. Key method:
- `generate_activations(dataset, layers, ...)` → `(List[Tensor(N,D)], freq_tensor)` — one tensor per layer, filtered to non-padding/non-BOS tokens.
- `extract_token_ids(dataset, act_generator, ...)` — tokenizes without a forward pass; returns token IDs, sample IDs, and per-token labels.
- Supported `mode` values: `"residual"`, `"residual_pre"`, `"mlp"`, `"mlp_out"`, `"attn_out"`.

### Initialization (`initializations/projected_knn.py`)
`ReservoirKMeans` scales K-Means to large activation sets:
1. Reservoir-samples a pool from the DataLoader (with optional inverse-frequency weighting).
2. Optionally sketches to `proj_dim` dimensions via a random orthonormal projection.
3. Runs `KMeansTorch` (k-means++ init, multiple restarts) on the sketch.
4. Refines centroids in full-dimensional space via Lloyd iterations (`lloyd_refine_projected`).

### MFA Model (`modeling/mfa.py`)
`MFA(nn.Module)` — parameters:
- `mu` (K, D): component means
- `dir_raw` (K, D, q): raw loading directions (normalized internally)
- `scale_rho` (K, q): softplus-parameterized loading scales
- `psi_rho` (K, D) or (D,): softplus-parameterized diagonal noise variance
- `pi_logits` (K,): mixture log-weights

The core computation (`_core`) uses the **Woodbury identity** with a Cholesky factor of the q×q matrix `M_k = I + W_k^T Ψ^{-1} W_k` (cheap since q ≪ D) to compute log-likelihoods and posterior latents in one batched pass.

Key methods: `responsibilities(x)`, `log_prob(x)`, `nll(x)`, `component_posterior(x)`, `reconstruct(x)`.

Serialization: `save_mfa(model, path)` / `load_mfa(path)` preserve all parameters and optional rotation state.

`MFAEncoderDecoder` builds a shared dictionary `[mu_k | W_k columns]` over all K components and encodes activations as sparse-style coefficients (responsibilities × posterior latent coords).

### Training (`modeling/train.py`)
`train_nll(model, loader, ...)` — Adam optimizer minimizing mixture NLL, with optional validation loader for best-model selection. DataLoader batches must yield `(activations, ...)` where activations are `(B, D)`.

### Interpretation (`analysis/subspace_interpretation.py`)
- `get_top_strings_per_concept(model, loader, tok2str, score=...)` — for each component k, returns the top-scoring token strings by posterior responsibility or per-component log-likelihood.
- `get_top_indices_per_concept(...)` — same but returns sample indices instead of strings.

### Visualization (`analysis/subspace_visualization.py`)
- `project_loader_to_subspace(model, loader, k, ...)` — projects activations assigned to component k onto span(W_k) via least-squares, returning coordinates, energy, and token labels.
- `plot_subspace_scatter(data, dims=(i,j))` — 2D scatter of subspace coordinates.

### Steering (`intervention/mfa_steering.py`)
`MFASteerer(model, mfa)` patches TransformerLens forward hooks to modify residual-stream activations at specified layers. Two strategies:
- **Mean steering** (`intervene` / `generate`): `x' = (1−α)x + α·μ_k`
- **Latent two-stage** (`intervene_latent` / `generate_latent`): centroid pull then subspace displacement `x' = x₁ + W_k @ z`
- Setting `k=None` uses responsibility-weighted mixture centroids/loadings.

## Key Conventions

- DataLoaders throughout the pipeline yield `(activations_tensor, token_ids_tensor)` batches.
- MFA positive parameters (`psi`, `scale`) are stored as raw values and passed through `F.softplus` — never modify them directly.
- `model_device` is for forward passes; `data_device` is where output tensors land (useful when offloading activations to CPU while keeping the LLM on GPU/MPS).
- The tutorial caps at 250k tokens (`MAX_TOKENS`) for speed; remove this cap for full experiments.
