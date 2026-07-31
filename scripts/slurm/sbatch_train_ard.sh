#!/bin/bash
# ── ARD-regularized MFA training job (single process) ────────────────────
# RANK here is the MAXIMUM rank per component; the ARD prior prunes columns
# below it, so set it generously and read the learned q_k off the q_eff logs.
#SBATCH --partition=H100
##SBATCH --nodelist=dgx003
#SBATCH --account=LADE
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=80G
#SBATCH --time=23:00:00
#SBATCH --job-name=mfa_train_ard
#SBATCH --array=5
#SBATCH --output=/u/dssc/zenocosini/decomposing-activations-local-geometry/logs/jobs/mfa_train_ard_%A_%a.out

# ─────────────────────────────────────────────────

SHARD_DIR=${SHARD_DIR:-/orfeo/scratch/dssc/zenocosini/dalg-cache/pile_gemma2b_activations}
MODELS_DIR=${MODELS_DIR:-/orfeo/scratch/dssc/zenocosini/dalg-cache/pile_gemma2b_models}
LAYER=$SLURM_ARRAY_TASK_ID

K=${K:-1000}
RANK=${RANK:-64}                           # maximum q per component
EPOCHS=${EPOCHS:-20}
REFINE_EPOCHS=${REFINE_EPOCHS:-10}
BATCH=${BATCH:-2048}
NUM_WORKERS=${NUM_WORKERS:-2}
POOL_SIZE=${POOL_SIZE:-}                   # default heuristic if empty
VAL_FRAC=${VAL_FRAC:-0.008}
SPLIT_SEED=${SPLIT_SEED:-42}
SEED=${SEED:-42}
VAL_ON_GPU=${VAL_ON_GPU:-0}                # 1 = preload val tensor on GPU
MAX_STEPS=${MAX_STEPS:-}

# ── ARD prior on nu ──────────────────────────────────────────────────────
ALPHA0=${ALPHA0:-1.0}                      # Gamma shape
B0=${B0:-1e-4}                             # Gamma rate
ARD_LAMBDA=${ARD_LAMBDA:-1.0}              # applied weight = lambda / n_train_tokens
RANK_THRESHOLD=${RANK_THRESHOLD:-1.0}      # column variance must exceed this x mean(Psi) to count in q_k
WARMUP_FRAC=${WARMUP_FRAC:-0.15}           # fraction of epochs at ard_beta=0
RAMP_FRAC=${RAMP_FRAC:-0.20}               # fraction of epochs ramping beta 0 -> 1
PRUNE=${PRUNE:-1}                          # 1 = zero sub-threshold columns after training

PRUNE_FLAG="--prune-at-end"
[[ "$PRUNE" == "0" ]] && PRUNE_FLAG="--no-prune-at-end"

OUT_DIR=${OUT_DIR:-"$MODELS_DIR/layer$(printf '%02d' "$LAYER")_${K}_${RANK}_mfa_ard"}
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
echo "K=$K  max_rank=$RANK  epochs=$EPOCHS  refine=$REFINE_EPOCHS  batch=$BATCH  num_workers=$NUM_WORKERS"
echo "ard: alpha0=$ALPHA0  b0=$B0  lambda=$ARD_LAMBDA  rank_threshold=$RANK_THRESHOLD"
echo "ard schedule: warmup=$WARMUP_FRAC  ramp=$RAMP_FRAC  prune_at_end=$PRUNE"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# ── Run ─────────────────────────────────────────────────────────────────
uv run python -m dalg.cli.run_training_ard \
    --shard-dir "$SHARD_DIR" --layer "$LAYER" --out-dir "$OUT_DIR" \
    --K "$K" --rank "$RANK" --epochs "$EPOCHS" \
    --alpha0 "$ALPHA0" --b0 "$B0" --ard-lambda "$ARD_LAMBDA" \
    --rank-threshold "$RANK_THRESHOLD" \
    --ard-warmup-frac "$WARMUP_FRAC" --ard-ramp-frac "$RAMP_FRAC" \
    $PRUNE_FLAG \
    --refine-epochs "$REFINE_EPOCHS" \
    --batch-size "$BATCH" --num-workers "$NUM_WORKERS" \
    --val-frac "$VAL_FRAC" --split-seed "$SPLIT_SEED" \
    --device cuda --seed "$SEED" \
    --wandb --wandb-project dalg-mfa --wandb-name "ard_L${LAYER}_K${K}_q${RANK}_lam${ARD_LAMBDA}_$(date +%H%M%S)" \
    $POOL_FLAG $VAL_ON_GPU_FLAG $MAX_STEPS_FLAG

echo "=== $(date) === done ==="
