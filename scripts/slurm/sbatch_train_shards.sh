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

# ── Config (edit to taste) ───────────────────────────────────────────────

SHARD_DIR=${SHARD_DIR:-/orfeo/scratch/dssc/zenocosini/dalg-cache/pile_gemma2b_activations}
LAYER=$SLURM_ARRAY_TASK_ID

K=${K:-1000}
RANK=${RANK:-337}
#337
EPOCHS=${EPOCHS:-20}
REFINE_EPOCHS=${REFINE_EPOCHS:-10}
BATCH=${BATCH:-1024}
# 2048
NUM_WORKERS=${NUM_WORKERS:-2}
POOL_SIZE=${POOL_SIZE:-}                   # default heuristic if empty
VAL_FRAC=${VAL_FRAC:-0.05}
SPLIT_SEED=${SPLIT_SEED:-42}
SEED=${SEED:-42}
VAL_ON_GPU=${VAL_ON_GPU:-0}                # 1 = preload val tensor on GPU (faster eval, more GPU RAM)
USE_AMP=${USE_AMP:-1} 

OUT_DIR="$SHARD_DIR/layer$(printf '%02d' "$LAYER")_$(printf "$K")_$(printf "$RANK")_mfa"

POOL_FLAG=""
if [[ -n "$POOL_SIZE" ]]; then
    POOL_FLAG="--pool-size $POOL_SIZE"
fi

VAL_ON_GPU_FLAG=""
if [[ "$VAL_ON_GPU" == "1" ]]; then
    VAL_ON_GPU_FLAG="--val-on-gpu"
fi

AMP_FLAG=""
[[ "$USE_AMP" -eq 1 ]] && AMP_FLAG="--use-amp"

# ── Env ──────────────────────────────────────────────────────────────────
SCRIPT_DIR="/u/dssc/zenocosini/decomposing-activations-local-geometry/src/dalg/cli/run_layer.py"
#$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=/u/dssc/zenocosini/decomposing-activations-local-geometry
#$(cd -- "$SCRIPT_DIR/../.." && pwd)

mkdir -p "$REPO_ROOT/logs/jobs"
cd "$REPO_ROOT" || exit 1
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

echo "=== $(date) === job $SLURM_JOB_ID.$SLURM_ARRAY_TASK_ID on $(hostname) ==="
echo "repo_root: $REPO_ROOT"
echo "shard_dir: $SHARD_DIR   layer: $LAYER   out_dir: $OUT_DIR"
echo "K=$K  rank=$RANK  epochs=$EPOCHS  refine=$REFINE_EPOCHS  batch=$BATCH  use_amp=$USE_AMP num_workers=$NUM_WORKERS"
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
    $POOL_FLAG $VAL_ON_GPU_FLAG $AMP_FLAG

echo "=== $(date) === done ==="
