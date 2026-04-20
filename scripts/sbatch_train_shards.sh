#!/bin/bash
#SBATCH --partition=H100
#SBATCH --account=LADE
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=80G
#SBATCH --time=23:00:00
#SBATCH --job-name=mfa_train
#SBATCH --array=5,17
#SBATCH --output=output_job/mfa_train_%A_%a.out

# ── Config (edit to taste) ───────────────────────────────────────────────
SHARD_DIR=${SHARD_DIR:-/orfeo/scratch/dssc/zenocosini/pile_gemma2b_activations}
LAYER=$SLURM_ARRAY_TASK_ID
OUT_DIR="$SHARD_DIR/layer$(printf '%02d' "$LAYER")_mfa"

K=${K:-1000}
RANK=${RANK:-10}
EPOCHS=${EPOCHS:-20}
REFINE_EPOCHS=${REFINE_EPOCHS:-10}
BATCH=${BATCH:-4096}
NUM_WORKERS=${NUM_WORKERS:-4}
POOL_SIZE=${POOL_SIZE:-}                   # default heuristic if empty
VAL_FRAC=${VAL_FRAC:-0.05}
SPLIT_SEED=${SPLIT_SEED:-42}
SEED=${SEED:-42}

POOL_FLAG=""
if [[ -n "$POOL_SIZE" ]]; then
    POOL_FLAG="--pool-size $POOL_SIZE"
fi

# ── Env ──────────────────────────────────────────────────────────────────
mkdir -p output_job
cd "$SLURM_SUBMIT_DIR" || exit 1

echo "=== $(date) === job $SLURM_JOB_ID.$SLURM_ARRAY_TASK_ID on $(hostname) ==="
echo "shard_dir: $SHARD_DIR   layer: $LAYER   out_dir: $OUT_DIR"
echo "K=$K  rank=$RANK  epochs=$EPOCHS  refine=$REFINE_EPOCHS  batch=$BATCH  num_workers=$NUM_WORKERS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# ── Run ──────────────────────────────────────────────────────────────────
uv run python experiments/run_layer.py train \
    --shard-dir "$SHARD_DIR" --layer "$LAYER" --out-dir "$OUT_DIR" \
    --K "$K" --rank "$RANK" --epochs "$EPOCHS" \
    --refine-epochs "$REFINE_EPOCHS" \
    --batch-size "$BATCH" --num-workers "$NUM_WORKERS" \
    --val-frac "$VAL_FRAC" --split-seed "$SPLIT_SEED" \
    --device cuda --seed "$SEED" \
    $POOL_FLAG

echo "=== $(date) === done ==="
