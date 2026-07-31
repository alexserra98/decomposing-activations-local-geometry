#!/bin/bash
# ── ARD-MFA on the toy-manifold datasets (single H100) ───────────────────
# RANK is the MAXIMUM rank per component; the ARD prior prunes columns below
# it, so read the learned q_k off the q_eff logs / the W&B histogram.
#
#   DATASET=.../toy_manifolds_scattered.pt sbatch scripts/slurm/temporary/sbatch_train_ard_toy.sh
#
#SBATCH --partition=H100
#SBATCH --account=LADE
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --job-name=ard_toy
#SBATCH --output=/u/dssc/zenocosini/decomposing-activations-local-geometry/logs/jobs/ard_toy_%j.out

REPO_ROOT=/u/dssc/zenocosini/decomposing-activations-local-geometry
CACHE_ROOT=${CACHE_ROOT:-/orfeo/scratch/dssc/zenocosini/dalg-cache}

DATASET=${DATASET:-$CACHE_ROOT/assets/toy_manifolds_centered.pt}
TAG=$(basename "$DATASET" .pt)

K=${K:-32000}
RANK=${RANK:-50}                           # maximum q per component
EPOCHS=${EPOCHS:-50}
BATCH=${BATCH:-512}                        # (B, K, q) activations dominate memory
VAL_BATCH=${VAL_BATCH:-256}
LR=${LR:-1e-3}
GRAD_CLIP=${GRAD_CLIP:-5.0}
SEED=${SEED:-42}
KMEANS_ITERS=${KMEANS_ITERS:-30}
MAX_STEPS=${MAX_STEPS:-}

# ── ARD prior on nu ──────────────────────────────────────────────────────
ALPHA0=${ALPHA0:-1.0}                      # Gamma shape
B0=${B0:-1e-4}                             # Gamma rate
ARD_LAMBDA=${ARD_LAMBDA:-1.0}              # applied weight = lambda / n_train
RANK_THRESHOLD=${RANK_THRESHOLD:-1.0}      # column counts in q_k when s^2 > this x mean(Psi)
WARMUP_FRAC=${WARMUP_FRAC:-0.15}           # fraction of epochs at ard_beta=0
RAMP_FRAC=${RAMP_FRAC:-0.20}               # fraction of epochs ramping beta 0 -> 1
PRUNE=${PRUNE:-1}                          # 1 = zero sub-threshold columns after training

PRUNE_FLAG="--prune-at-end"
[[ "$PRUNE" == "0" ]] && PRUNE_FLAG="--no-prune-at-end"

MAX_STEPS_FLAG=""
[[ -n "$MAX_STEPS" ]] && MAX_STEPS_FLAG="--max-steps $MAX_STEPS"

OUT_DIR=${OUT_DIR:-"$CACHE_ROOT/toy_manifold_models/${TAG}_K${K}_q${RANK}_lam${ARD_LAMBDA}_mfa_ard"}
WANDB_NAME=${WANDB_NAME:-"ard_${TAG}_K${K}_q${RANK}_lam${ARD_LAMBDA}"}

mkdir -p "$REPO_ROOT/logs/jobs" "$OUT_DIR"
cd "$REPO_ROOT" || exit 1
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
# Long-lived (B, K, q) buffers fragment the caching allocator badly at K=32k.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

echo "=== $(date) === job $SLURM_JOB_ID on $(hostname) ==="
echo "dataset: $DATASET"
echo "out_dir: $OUT_DIR"
echo "K=$K  max_rank=$RANK  epochs=$EPOCHS  batch=$BATCH  lr=$LR"
echo "ard: alpha0=$ALPHA0  b0=$B0  lambda=$ARD_LAMBDA  rank_threshold=$RANK_THRESHOLD"
echo "ard schedule: warmup=$WARMUP_FRAC  ramp=$RAMP_FRAC  prune_at_end=$PRUNE"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

uv run python scripts/temporary/train_ard_toy_manifolds.py \
    --dataset "$DATASET" --out-dir "$OUT_DIR" \
    --K "$K" --rank "$RANK" --epochs "$EPOCHS" \
    --alpha0 "$ALPHA0" --b0 "$B0" --ard-lambda "$ARD_LAMBDA" \
    --rank-threshold "$RANK_THRESHOLD" \
    --ard-warmup-frac "$WARMUP_FRAC" --ard-ramp-frac "$RAMP_FRAC" \
    $PRUNE_FLAG \
    --batch-size "$BATCH" --val-batch-size "$VAL_BATCH" \
    --lr "$LR" --grad-clip "$GRAD_CLIP" \
    --kmeans-iters "$KMEANS_ITERS" \
    --device cuda --seed "$SEED" \
    --wandb --wandb-project dalg-mfa --wandb-name "$WANDB_NAME" \
    $MAX_STEPS_FLAG

echo "=== $(date) === done ==="
