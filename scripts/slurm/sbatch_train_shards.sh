#!/bin/bash
# ── Single knob: edit the --gres line to change #GPUs / GPU type ─────────
# Examples:
#     #SBATCH --gres=gpu:H100:2      (2× H100)
#     #SBATCH --gres=gpu:A100:4      (4× A100)
# NPROC is auto-derived from SLURM_GPUS_ON_NODE below — no other edits needed.
# Rule of thumb: keep --cpus-per-task ≈ 8×GPUs and --mem ≈ 80G×GPUs.
#SBATCH --partition=H100
##SBATCH --nodelist=dgx003 
#SBATCH --account=LADE
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=160G
#SBATCH --time=22:00:00
#SBATCH --job-name=mfa_train_ddp
#SBATCH --array=5,17
#SBATCH --output=/orfeo/cephfs/home/dssc/zenocosini/decomposing-activations-local-geometry/outputs/jobs/mfa_train_ddp_%A_%a.out

# ── Config (edit to taste) ───────────────────────────────────────────────
SHARD_DIR=${SHARD_DIR:-/orfeo/scratch/dssc/zenocosini/pile_gemma2b_activations}
LAYER=$SLURM_ARRAY_TASK_ID
OUT_DIR="$SHARD_DIR/layer$(printf '%02d' "$LAYER")_mfa"

K=${K:-32000}
RANK=${RANK:-10}
EPOCHS=${EPOCHS:-20}
REFINE_EPOCHS=${REFINE_EPOCHS:-10}
BATCH=${BATCH:-4096}
NUM_WORKERS=${NUM_WORKERS:-2}
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
mkdir -p outputs/jobs
cd "$SLURM_SUBMIT_DIR" || exit 1

# Auto-derive from the --gres allocation; env override still wins.
NPROC=${NPROC:-${SLURM_GPUS_ON_NODE:-2}}

echo "=== $(date) === job $SLURM_JOB_ID.$SLURM_ARRAY_TASK_ID on $(hostname) ==="
echo "shard_dir: $SHARD_DIR   layer: $LAYER   out_dir: $OUT_DIR"
echo "K=$K  rank=$RANK  epochs=$EPOCHS  refine=$REFINE_EPOCHS  batch=$BATCH  num_workers=$NUM_WORKERS  nproc=$NPROC"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# ── Run (DDP via torchrun) ───────────────────────────────────────────────
uv run torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC" \
    dalg-run-layer train \
    --shard-dir "$SHARD_DIR" --layer "$LAYER" --out-dir "$OUT_DIR" \
    --K "$K" --rank "$RANK" --epochs "$EPOCHS" \
    --refine-epochs "$REFINE_EPOCHS" \
    --batch-size "$BATCH" --num-workers "$NUM_WORKERS" \
    --val-frac "$VAL_FRAC" --split-seed "$SPLIT_SEED" \
    --device cuda --seed "$SEED" \
    $POOL_FLAG

echo "=== $(date) === done ==="
