# %% [markdown]
# # MFA Tutorial: Training and Interpreting Mixture of Factor Analyzers
#
# A guide to using **Mixtures of Factor Analyzers (MFA)** to model LM activations
# as **regions + local subspaces**.
#
# ## Workflow
# 1. **Extract** activations from a language model at a chosen layer.
# 2. **Initialize** cluster centroids with (projected) K-Means.
# 3. **Train** the MFA by minimizing the mixture NLL.
# 4. **Interpret** components via top-likelihood tokens.
# 5. **Visualize** the local subspace of a component.
# 6. **Steer** the model using centroid pull or latent-space displacement.

# %%
%load_ext autoreload
%autoreload 2
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # required for MPS

import random
import torch
from torch.utils.data import DataLoader, TensorDataset

from dalg.llm.activation_generator import ActivationGenerator, extract_token_ids
from dalg.data.concept_dataset import SupervisedConceptDataset


# %% [markdown]
# ## Configuration
#
# - `data_path`: JSON dataset with (sentence, concept) pairs.
# - `model_name`: Any TransformerLens-supported model. We use gpt2-small for speed.
# - `layers`: Which layer to factorize.
# - `model_device` / `data_device`: Use "mps" on Apple Silicon, "cuda" on NVIDIA, "cpu" otherwise.
# - `factorization_mode`: Activation stream to factorize. "residual" is a good default.

# %%
data_path = "./data/supervised.json"
model_name = "gpt2-small"
layers = [4]
model_device = "mps"
data_device = "mps"
factorization_mode = "residual"

# %% [markdown]
# ## Loading the model and dataset

# %%
act_generator = ActivationGenerator(
    model_name,
    model_device=model_device,
    data_device=data_device,
    mode=factorization_mode,
)

dataset = SupervisedConceptDataset(data_path)
print(f"Dataset size: {len(dataset)} samples")

# %% [markdown]
# ## Extracting activations
#
# `generate_activations` returns one tensor per layer (shape: num_tokens × d_model)
# and a token-frequency vector used for weighted sampling during initialization.
# We also extract token IDs for later interpretation.

# %%
activations, freq = act_generator.generate_activations(dataset, layers, batch_size=1)
tokens, _, _ = extract_token_ids(dataset, act_generator, batch_size=1)

print(f"Activations: {activations[0].shape}  |  Tokens: {tokens.shape}")

# %% [markdown]
# ## Building data loaders
#
# We cap at 250k activations to keep this tutorial fast on modest hardware.
# Shuffle the training loader so that each epoch sees a different order.

# %%
MAX_TOKENS = 250_000
X = activations[0][:MAX_TOKENS]
tok = tokens[:MAX_TOKENS]

full_ds = TensorDataset(X, tok)
loader = DataLoader(full_ds, batch_size=128, shuffle=True, pin_memory=True)
token_loader = DataLoader(TensorDataset(tok), batch_size=128)

print(f"Training on {len(full_ds):,} token activations")

# %% [markdown]
# ## Initialization: Projected K-Means
#
# `ReservoirKMeans` samples a pool from the loader, sketches it to a lower-dimensional
# space via a random projection, runs K-Means there, then refines centroids on the
# full-dimensional data. This scales to millions of activations.
#
# Alternatively, you can just sample random points as centroids (second cell below).

# %%
from dalg.init.projected_knn import ReservoirKMeans

num_centroids = 500
pool_size = len(full_ds) // 5  # 20 % of the data

knn = ReservoirKMeans(
    num_centroids,
    pool_size=pool_size,
    vocab_size=act_generator.model.cfg.d_vocab,
    device=model_device,
    proj_dim=32,
)
centroids = knn.fit(loader)
# %%
# cache everything
centroids = centroids.cpu()
X = X.cpu()
tok = tok.cpu()
torch.save(centroids, "centroids.pt")
torch.save(X, "activations.pt")
torch.save(tok, "tokens.pt")
# %%
# --- alternative: random point initialization ---
# idx = torch.randperm(len(X))[:num_centroids]
# centroids = X[idx]

# %% [markdown]
# ## Training
#
# We minimize the mixture NLL using Adam. `rank=10` means each component's
# covariance lives in a 10-dimensional subspace. Train until validation loss
# converges; 10 epochs is usually enough to see interpretable structure.

# %%
from dalg.models.mfa import MFA
from dalg.models.train import train_nll

model = MFA(centroids=centroids, rank=10).to(model_device)

# %% [markdown]
# ## Interpretation: top tokens per component
#
# For each component k, we score every activation in the loader by its log-likelihood
# under component k, then surface the tokens with the highest scores.

# %%
from dalg.analysis.subspace_interpretation import get_top_strings_per_concept

def tok_to_str(tok_id):
    return act_generator.model.to_string(tok_id)

results = get_top_strings_per_concept(model, loader, tok_to_str, score="likelihood")

# %% [markdown]
# Inspect the top tokens for the first 25 components.
# We sample from the top-5000 pool to avoid all results looking identical.

# %%
N_COMPONENTS = 25
SAMPLE_SIZE = 10
TOP_POOL = 5000
random.seed(0)

for k, tokens_list in list(results.items())[:N_COMPONENTS]:
    pool = tokens_list[:min(TOP_POOL, len(tokens_list))]
    sample = random.sample(pool, k=min(SAMPLE_SIZE, len(pool)))
    print(f"\n[Component {k}]\n" + "-" * 40)
    for t in sample:
        print(f"  - {str(t).replace(chr(10), '\\n')}")

# %% [markdown]
# ## Visualization: local subspace of a component
#
# Project activations assigned to component k onto its local factor subspace
# (spanned by the loading columns W_k). The coordinates are the least-squares
# coefficients in that basis.

# %%
import dalg.analysis.subspace_visualization as sv

k_to_visualize = 20

subspace_data = sv.project_loader_to_subspace(
    model, loader, k=k_to_visualize, token_to_str=tok_to_str
)

# Plot in 2D using loadings 0 and 8.
# Note: variance is spread across all R=10 dimensions; these two may not be the most
# informative. Try different dim pairs or apply PCA on the coordinates.
sv.plot_subspace_scatter(subspace_data, dims=(0, 8), max_labels=250)

# %% [markdown]
# ## Steering: centroid pull
#
# We interpolate activations towards centroid mu_k:
#
#   x' = (1 - alpha) * x + alpha * mu_k
#
# alpha=1 replaces the activation entirely; lower values give gentler nudges.

# %%
from dalg.intervention.mfa_steering import MFASteerer

steerer = MFASteerer(act_generator.model, model)

prompt = "I think that"
alpha = 0.6
layer = 4
component = 20

base_logits = act_generator.model(act_generator.model.to_tokens(prompt))
steered_logits = steerer.intervene(prompt, layers=[layer], alpha=alpha, k=component)

delta = (steered_logits[0, -1, :] - base_logits[0, -1, :])
top_pos_vals, top_pos_idx = torch.topk(delta, k=15)
print("Top promoted tokens:")
for tid, d in zip(top_pos_idx, top_pos_vals):
    print(f"  {act_generator.model.to_str_tokens([tid])}  Δlogit={d.item():.3f}")

# %% [markdown]
# ## Steering: latent two-stage
#
# We first pull towards the centroid, then move within the local subspace:
#
#   x1 = x + alpha * (mu_k - x)       # centroid pull
#   x' = x1 + W_k @ z                  # subspace displacement
#
# Setting individual z entries moves along the corresponding loading direction.
# Here we push z[0]=20 and z[8]=-10 to land in the "dissertation" region of
# component 20 (visible in the scatter plot above).

# %%
z = torch.zeros(10)
z[0] = 20
z[8] = -10

steered_logits = steerer.intervene_latent(
    prompt, layers=[layer], alpha_centroid=alpha, z=z, k=component
)

delta = (steered_logits[0, -1, :] - base_logits[0, -1, :])
top_pos_vals, top_pos_idx = torch.topk(delta, k=10)
print("Top promoted tokens (latent steering):")
for tid, d in zip(top_pos_idx, top_pos_vals):
    print(f"  {act_generator.model.to_str_tokens([tid])}  Δlogit={d.item():.3f}")
