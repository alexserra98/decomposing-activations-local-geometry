# %%
from pathlib import Path

import pandas as pd
import torch

from dalg.evaluation.tile_alignment import evaluate_tile_alignment
from dalg.models.adaptive_q.mfa_hddc import load_mfa_hddc

# %%
REPO_ROOT = Path(__file__).parent.parent
MODEL_ROOT = REPO_ROOT /  Path("dalg-cache/sae_manifold_temperature_02d_llama31_8b_last/sae_manifold_temperature_02d_hddc_em/hddc__temperature_02d__l28__k50__q16__s42__2af035cd/mfa_model.pt")
model = load_mfa_hddc(MODEL_ROOT, map_location="cpu")
assignments = torch.load(
    MODEL_ROOT.with_name("mfa_model_assignments.pt"), map_location="cpu", weights_only=True,
)

# %%
report = evaluate_tile_alignment(model, assignments["cluster_sizes"] > 0, neighbors=5)
print({name: report[name]["mean"] for name in ("leading_pc_alignment", "best_subset_pc_alignment")})

# %%
per_tile = pd.DataFrame(report["per_tile"])
comparisons = pd.DataFrame(report["comparisons"])
per_tile

# %%
comparisons
