#!/bin/bash
#SBATCH --partition=H100
#SBATCH --account=LADE
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:H100:2
#SBATCH --mem=160G
#SBATCH --time=23:00:00
#SBATCH --job-name=mfa_train_ddp
#SBATCH --array=5,17
#SBATCH --output=output_job/mfa_train_ddp_%A_%a.out

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

OUT_DIR="$SHARD_DIR/layer$(printf '%02d' "$LAYER")_$(printf "$K")_mfa"

POOL_FLAG=""
if [[ -n "$POOL_SIZE" ]]; then
    POOL_FLAG="--pool-size $POOL_SIZE"
fi

# ── Env ──────────────────────────────────────────────────────────────────
mkdir -p output_job
cd "$SLURM_SUBMIT_DIR" || exit 1

NPROC=${NPROC:-2}   # GPUs per node = processes per node for torchrun

echo "=== $(date) === job $SLURM_JOB_ID.$SLURM_ARRAY_TASK_ID on $(hostname) ==="
echo "shard_dir: $SHARD_DIR   layer: $LAYER   out_dir: $OUT_DIR"
echo "K=$K  rank=$RANK  epochs=$EPOCHS  refine=$REFINE_EPOCHS  batch=$BATCH  num_workers=$NUM_WORKERS  nproc=$NPROC"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# ── Run (DDP via torchrun) ───────────────────────────────────────────────
uv run torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC" \
    experiments/run_layer.py train \
    --shard-dir "$SHARD_DIR" --layer "$LAYER" --out-dir "$OUT_DIR" \
    --K "$K" --rank "$RANK" --epochs "$EPOCHS" \
    --refine-epochs "$REFINE_EPOCHS" \
    --batch-size "$BATCH" --num-workers "$NUM_WORKERS" \
    --val-frac "$VAL_FRAC" --split-seed "$SPLIT_SEED" \
    --device cuda --seed "$SEED" \
    $POOL_FLAG

echo "=== $(date) === done ==="
