#!/bin/bash
# ── Single full-model shard training job ─────────────────────────────────
# Examples:
#     #SBATCH --gres=gpu:H100:1
#     #SBATCH --gres=gpu:A100:1
#SBATCH --partition=H100
##SBATCH --nodelist=dgx003 
#SBATCH --account=LADE
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=80G
#SBATCH --time=23:00:00
#SBATCH --job-name=mfa_train_shards
#SBATCH --array=5,17
#SBATCH --output=/u/dssc/zenocosini/decomposing-activations-local-geometry/logs/jobs/mfa_train_shards_%A_%a.out

# ─────────────────────────────────────────────────

SHARD_DIR=${SHARD_DIR:-/orfeo/scratch/dssc/zenocosini/dalg-cache/pile_gemma2b_activations}
MODELS_DIR=${MODELS_DIR:-/orfeo/scratch/dssc/zenocosini/dalg-cache/pile_gemma2b_models}
LAYER=$SLURM_ARRAY_TASK_ID

K=${K:-1000}
RANK=${RANK:-10}
#337
EPOCHS=${EPOCHS:-20}
REFINE_EPOCHS=${REFINE_EPOCHS:-10}
BATCH=${BATCH:-2048}
# 2048
NUM_WORKERS=${NUM_WORKERS:-2}
POOL_SIZE=${POOL_SIZE:-}                   # default heuristic if empty
VAL_FRAC=${VAL_FRAC:-0.008}
SPLIT_SEED=${SPLIT_SEED:-42}
SEED=${SEED:-42}
VAL_ON_GPU=${VAL_ON_GPU:-0}                # 1 = preload val tensor on GPU (faster eval, more GPU RAM)
MAX_STEPS=${MAX_STEPS:-}

OUT_DIR=${OUT_DIR:-"$MODELS_DIR/layer$(printf '%02d' "$LAYER")_${K}_${RANK}_mfa"}
CENTROIDS_FROM=${CENTROIDS_FROM:-}                # optional: reuse centroids from an existing K-matched run

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
SCRIPT_DIR="/u/dssc/zenocosini/decomposing-activations-local-geometry/src/dalg/cli/run_training.py"
#$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=/u/dssc/zenocosini/decomposing-activations-local-geometry
#$(cd -- "$SCRIPT_DIR/../.." && pwd)

mkdir -p "$REPO_ROOT/logs/jobs" "$OUT_DIR"
cd "$REPO_ROOT" || exit 1
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

if [[ ! -f "$OUT_DIR/centroids.pt" && -n "$CENTROIDS_FROM" && -f "$CENTROIDS_FROM" ]]; then
    cp -n "$CENTROIDS_FROM" "$OUT_DIR/centroids.pt"
fi

echo "=== $(date) === job $SLURM_JOB_ID.$SLURM_ARRAY_TASK_ID on $(hostname) ==="
echo "repo_root: $REPO_ROOT"
echo "shard_dir: $SHARD_DIR   layer: $LAYER   out_dir: $OUT_DIR"
echo "K=$K  rank=$RANK  epochs=$EPOCHS  refine=$REFINE_EPOCHS  batch=$BATCH  num_workers=$NUM_WORKERS"
echo "training_mode=vanilla"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# ── Run ─────────────────────────────────────────────────────────────────
uv run python -m dalg.cli.run_training \
    --shard-dir "$SHARD_DIR" --layer "$LAYER" --out-dir "$OUT_DIR" \
    --K "$K" --rank "$RANK" --epochs "$EPOCHS" \
    --refine-epochs "$REFINE_EPOCHS" \
    --batch-size "$BATCH" --num-workers "$NUM_WORKERS" \
    --val-frac "$VAL_FRAC" --split-seed "$SPLIT_SEED" \
    --device cuda --seed "$SEED" \
    --training-mode vanilla \
    --wandb --wandb-project dalg-mfa --wandb-name "smoketest_L5_K1000_q10_$(date +%H%M%S)" \
    $POOL_FLAG $VAL_ON_GPU_FLAG $MAX_STEPS_FLAG

echo "=== $(date) === done ==="
