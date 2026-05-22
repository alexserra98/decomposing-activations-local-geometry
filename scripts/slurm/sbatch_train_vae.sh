#!/bin/bash
# Single-GPU VAE training on activation shards.
# Submit with:
#     sbatch scripts/slurm/sbatch_train_vae.sh

#SBATCH --partition=H100
#SBATCH --account=LADE
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=80G
#SBATCH --time=23:00:00
#SBATCH --job-name=vae_layer17
#SBATCH --output=/u/dssc/malessi/scratch/decomposing-activations-local-geometry/logs/jobs/vae_layer17_%j.out

# Data / run config
REPO_ROOT=${REPO_ROOT:-/u/dssc/malessi/scratch/decomposing-activations-local-geometry}
PYTHON=${PYTHON:-"$REPO_ROOT/.venv/bin/python"}
SHARD_DIR=${SHARD_DIR:-/orfeo/scratch/dssc/zenocosini/dalg-cache/pile_gemma2b_activations}
OUT_DIR=${OUT_DIR:-/u/dssc/malessi/scratch/vae_layer17}

LAYER=17
EPOCHS=${EPOCHS:-50}
BATCH=${BATCH:-2048}
NUM_WORKERS=${NUM_WORKERS:-2}
LOG_INTERVAL=${LOG_INTERVAL:-10}
VAL_FRAC=0.008
VAL_ON_GPU=${VAL_ON_GPU:-0}
SPLIT_SEED=42
SEED=42
MAX_STEPS=${MAX_STEPS:-}

LATENT_DIM=${LATENT_DIM:-64}
ENC_HIDDEN_DIMS=${ENC_HIDDEN_DIMS:-1024,512}
DEC_HIDDEN_DIMS=${DEC_HIDDEN_DIMS:-512,1024}
PRIOR=vamp
PRIOR_COMPONENTS=1000
BETA=${BETA:-1.0}
BETA_WARMUP_STEPS=${BETA_WARMUP_STEPS:-0}
LR=${LR:-1e-3}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4}
GRAD_CLIP=${GRAD_CLIP:-1.0}

VAL_ON_GPU_FLAG=""
if [[ "$VAL_ON_GPU" == "1" ]]; then
    VAL_ON_GPU_FLAG="--val-on-gpu"
fi

MAX_STEPS_FLAG=""
[[ -n "$MAX_STEPS" ]] && MAX_STEPS_FLAG="--max-steps $MAX_STEPS"

# Env
mkdir -p "$REPO_ROOT/logs/jobs" "$OUT_DIR"
cd "$REPO_ROOT" || exit 1
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OUT_DIR

LOG_PATH="$OUT_DIR/train.log"

echo "=== $(date) === job ${SLURM_JOB_ID:-local} on $(hostname) ===" | tee "$LOG_PATH"
echo "repo_root: $REPO_ROOT" | tee -a "$LOG_PATH"
echo "python: $PYTHON" | tee -a "$LOG_PATH"
echo "shard_dir: $SHARD_DIR" | tee -a "$LOG_PATH"
echo "layer: $LAYER   out_dir: $OUT_DIR" | tee -a "$LOG_PATH"
echo "training_mode=vae  prior=$PRIOR  prior_components=$PRIOR_COMPONENTS" | tee -a "$LOG_PATH"
echo "epochs=$EPOCHS  batch=$BATCH  num_workers=$NUM_WORKERS  log_interval=$LOG_INTERVAL  val_frac=$VAL_FRAC  val_on_gpu=$VAL_ON_GPU" | tee -a "$LOG_PATH"
echo "latent_dim=$LATENT_DIM  beta=$BETA  beta_warmup_steps=$BETA_WARMUP_STEPS  seed=$SEED" | tee -a "$LOG_PATH"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | tee -a "$LOG_PATH" || true

# Run
"$PYTHON" -u -m dalg.cli.run_training \
    --training-mode vae \
    --shard-dir "$SHARD_DIR" --layer "$LAYER" --out-dir "$OUT_DIR" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH" --num-workers "$NUM_WORKERS" --log-interval "$LOG_INTERVAL" \
    --val-frac "$VAL_FRAC" --split-seed "$SPLIT_SEED" \
    --device cuda --seed "$SEED" \
    --vae-latent-dim "$LATENT_DIM" \
    --vae-enc-hidden-dims "$ENC_HIDDEN_DIMS" \
    --vae-dec-hidden-dims "$DEC_HIDDEN_DIMS" \
    --vae-prior "$PRIOR" --vae-prior-components "$PRIOR_COMPONENTS" \
    --vae-beta "$BETA" --vae-beta-warmup-steps "$BETA_WARMUP_STEPS" \
    --vae-weight-decay "$WEIGHT_DECAY" \
    --lr "$LR" --grad-clip "$GRAD_CLIP" \
    $VAL_ON_GPU_FLAG $MAX_STEPS_FLAG 2>&1 | tee -a "$LOG_PATH"
status=${PIPESTATUS[0]}
if [[ "$status" -ne 0 ]]; then
    echo "=== $(date) === training failed with status $status ===" | tee -a "$LOG_PATH"
    exit "$status"
fi

# Save the learned Vamp prior parameters separately for quick inspection.
"$PYTHON" -u - <<'PY'
import json
import os
from pathlib import Path

import torch

from dalg.models.vae import VAE, build_prior

out_dir = Path(os.environ["OUT_DIR"])
model_path = out_dir / "vae_model.pt"
config_path = out_dir / "config.json"
payload = torch.load(model_path, map_location="cpu", weights_only=False)
config = json.loads(config_path.read_text())

prior = build_prior(
    config["prior"],
    config["latent_dim"],
    config["prior_components"],
    input_dim=config["d_model"],
)
model = VAE(
    input_dim=config["d_model"],
    latent_dim=config["latent_dim"],
    enc_hidden_dims=tuple(config["enc_hidden_dims"]),
    dec_hidden_dims=tuple(config["dec_hidden_dims"]),
    prior=prior,
    beta=config["beta"],
)
model.load_state_dict(payload["state_dict"])
model.eval()

prior_state = {
    key.removeprefix("prior."): value.detach().cpu()
    for key, value in model.state_dict().items()
    if key.startswith("prior.")
}
extra = {}
if hasattr(model.prior, "_component_params"):
    with torch.no_grad():
        mu, logvar = model.prior._component_params()
    extra["component_mu"] = mu.detach().cpu()
    extra["component_logvar"] = logvar.detach().cpu()

torch.save(
    {
        "prior": payload["prior"],
        "prior_state_dict": prior_state,
        **extra,
    },
    out_dir / "prior_params.pt",
)
print(f"saved prior params to {out_dir / 'prior_params.pt'}")
PY

echo "=== $(date) === done ===" | tee -a "$LOG_PATH"
