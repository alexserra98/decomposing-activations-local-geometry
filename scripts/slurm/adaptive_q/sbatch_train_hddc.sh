#!/bin/bash
# ── HDDC covariance-surgery MFA training job (single process) ────────────
# RANK here is q_max, the per-component upper bound. Surgery picks d_k <= q_max
# every SURGERY_EVERY epochs; read the learned ranks off the d_k logs.
# Set SURGERY_EVERY=0 for a fixed-q baseline on the same stack.
#SBATCH --partition=H100
##SBATCH --nodelist=dgx003
#SBATCH --account=LADE
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=80G
#SBATCH --time=23:00:00
#SBATCH --job-name=mfa_train_hddc
#SBATCH --array=5
#SBATCH --output=/u/dssc/zenocosini/decomposing-activations-local-geometry/logs/jobs/mfa_train_hddc_%A_%a.out

set -euo pipefail

# ─────────────────────────────────────────────────

SHARD_DIR=${SHARD_DIR:-/u/dssc/zenocosini/decomposing-activations-local-geometry/dalg-cache/assets/toy_manifolds_8types_2each_D128_150K_shards}
MODELS_DIR=${MODELS_DIR:-/u/dssc/zenocosini/decomposing-activations-local-geometry/dalg-cache/toy_manifold_models}
LAYER=$SLURM_ARRAY_TASK_ID

K=${K:-100}
RANK=${RANK:-32}                           # q_max: upper bound on the adaptive rank
EPOCHS=${EPOCHS:-30}
REFINE_EPOCHS=${REFINE_EPOCHS:-10}
BATCH=${BATCH:-2048}
ASSIGN_BATCH=${ASSIGN_BATCH:-1024}
NUM_WORKERS=${NUM_WORKERS:-2}
POOL_SIZE=${POOL_SIZE:-}                   # default heuristic if empty
VAL_FRAC=${VAL_FRAC:-0.008}
SPLIT_SEED=${SPLIT_SEED:-42}
SEED=${SEED:-42}
VAL_ON_GPU=${VAL_ON_GPU:-0}                # 1 = preload val tensor on GPU
MAX_STEPS=${MAX_STEPS:-}

# ── HDDC surgery ─────────────────────────────────────────────────────────
SURGERY_EVERY=${SURGERY_EVERY:-3}          # epochs between surgeries; 0 = fixed-q baseline
SURGERY_THRESHOLD=${SURGERY_THRESHOLD:-0.01}   # Cattell t, relative to the leading eigenvalue
SURGERY_MIN_COUNT=${SURGERY_MIN_COUNT:-0}      # n_min; 0 => max(5 * q_max, 50)
SURGERY_WARMUP=${SURGERY_WARMUP:-0}            # linear LR warmup steps after each surgery

OUT_DIR=${OUT_DIR:-"$MODELS_DIR/layer$(printf '%02d' "$LAYER")_${K}_${RANK}_mfa_hddc"}
CENTROIDS_FROM=${CENTROIDS_FROM:-}         # optional: reuse centroids from an existing K-matched run

POOL_FLAG=""
if [[ -n "$POOL_SIZE" ]]; then
    POOL_FLAG="--pool-size $POOL_SIZE"
fi

VAL_ON_GPU_FLAG=""
if [[ "$VAL_ON_GPU" == "1" ]]; then
    VAL_ON_GPU_FLAG="--val-on-gpu"
fi

MAX_STEPS_FLAG=""
[[ -n "$MAX_STEPS" ]] && MAX_STEPS_FLAG="--max-steps $MAX_STEPS"

# ── Env ──────────────────────────────────────────────────────────────────
REPO_ROOT=/u/dssc/zenocosini/decomposing-activations-local-geometry

mkdir -p "$REPO_ROOT/logs/jobs" "$OUT_DIR"
cd "$REPO_ROOT" || exit 1
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

if [[ ! -f "$OUT_DIR/centroids.pt" && -n "$CENTROIDS_FROM" && -f "$CENTROIDS_FROM" ]]; then
    cp -n "$CENTROIDS_FROM" "$OUT_DIR/centroids.pt"
fi

echo "=== $(date) === job $SLURM_JOB_ID.$SLURM_ARRAY_TASK_ID on $(hostname) ==="
echo "repo_root: $REPO_ROOT"
echo "shard_dir: $SHARD_DIR   layer: $LAYER   out_dir: $OUT_DIR"
echo "K=$K  q_max=$RANK  epochs=$EPOCHS  refine=$REFINE_EPOCHS  batch=$BATCH  num_workers=$NUM_WORKERS"
echo "surgery: every=$SURGERY_EVERY  t=$SURGERY_THRESHOLD  n_min=$SURGERY_MIN_COUNT  warmup=$SURGERY_WARMUP"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# ── Run ─────────────────────────────────────────────────────────────────
# --isotropic-psi is required whenever surgery is on: the reconstruction
# Sigma_k = W_k W_k^T + b_k I is exact only for isotropic noise.
uv run python -m dalg.cli.adaptive_q.run_training_hddc \
    --shard-dir "$SHARD_DIR" --layer "$LAYER" --out-dir "$OUT_DIR" \
    --K "$K" --q-max "$RANK" --epochs "$EPOCHS" \
    --isotropic-psi \
    --surgery-every-epochs "$SURGERY_EVERY" \
    --surgery-threshold "$SURGERY_THRESHOLD" \
    --surgery-min-count "$SURGERY_MIN_COUNT" \
    --surgery-warmup-steps "$SURGERY_WARMUP" \
    --refine-epochs "$REFINE_EPOCHS" \
    --batch-size "$BATCH" --num-workers "$NUM_WORKERS" \
    --val-frac "$VAL_FRAC" --split-seed "$SPLIT_SEED" \
    --device cuda --seed "$SEED" \
    --wandb --wandb-project dalg-mfa --wandb-name "hddc_L${LAYER}_K${K}_q${RANK}_t${SURGERY_THRESHOLD}_$(date +%H%M%S)" \
    $POOL_FLAG $VAL_ON_GPU_FLAG $MAX_STEPS_FLAG

echo "=== $(date) === training done; computing assignments (batch=$ASSIGN_BATCH) ==="
uv run dalg-run-metrics assignments \
    --data-dir "$OUT_DIR" \
    --model-type hddc \
    --shard-dir "$SHARD_DIR" \
    --layer "$LAYER" \
    --batch-size "$ASSIGN_BATCH" \
    --device cuda \
    --seed "$SEED" \
    --save-path "$OUT_DIR/mfa_model_assignments.pt"

echo "=== $(date) === training + assignments done ==="
