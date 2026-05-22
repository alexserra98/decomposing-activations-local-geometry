#!/bin/bash
# Train MFA with components sharded across GPUs.
#
# This is model-parallel over K:
# every rank sees the same activation batches, and each rank owns a slice of
# the MFA components. Increasing #GPUs reduces per-GPU component memory.
#
# Examples:
#     #SBATCH --gres=gpu:H100:2      (2 component shards)
#     #SBATCH --gres=gpu:H100:4      (4 component shards)
# Rule of thumb: keep --cpus-per-task around 8xGPUs and --mem around 80GxGPUs.
#SBATCH --partition=H100
##SBATCH --nodelist=dgx003
#SBATCH --account=LADE
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:H100:2
#SBATCH --mem=160G
#SBATCH --time=23:00:00
#SBATCH --job-name=mfa_train_component_shards
##SBATCH --begin=now+8hours
#SBATCH --array=5
#SBATCH --output=/u/dssc/zenocosini/decomposing-activations-local-geometry/logs/jobs/mfa_train_component_shards_%A_%a.out

# ── Config (override with sbatch --export=ALL,KEY=value) ─────────────────
SHARD_DIR=${SHARD_DIR:-/orfeo/scratch/dssc/zenocosini/dalg-cache/pile_gemma2b_activations}
LAYER=$SLURM_ARRAY_TASK_ID

K=${K:-1000}
RANK=${RANK:-300}
EPOCHS=${EPOCHS:-3}
REFINE_EPOCHS=${REFINE_EPOCHS:-10}
BATCH=${BATCH:-8192}
NUM_WORKERS=${NUM_WORKERS:-4}
POOL_SIZE=${POOL_SIZE:-}
VAL_FRAC=${VAL_FRAC:-0.05}                   # val runs on every rank via _build_val_loader
SPLIT_SEED=${SPLIT_SEED:-42}
SEED=${SEED:-42}
COMPILE=${COMPILE:-1}                        # torch.compile on by default; set COMPILE=0 to disable
MAX_STEPS=${MAX_STEPS:-1000}                     # optional hard step cap for smoke/bisect runs

LAYER_TAG="layer$(printf '%02d' "$LAYER")"
OUT_DIR=${OUT_DIR:-"$SHARD_DIR/${LAYER_TAG}_${K}_${RANK}_component_sharded_mfa"}
CENTROIDS_FROM=${CENTROIDS_FROM:-"$SHARD_DIR/${LAYER_TAG}_${K}_${RANK}_mfa/centroids.pt"}

POOL_FLAG=""
[[ -n "$POOL_SIZE" ]] && POOL_FLAG="--pool-size $POOL_SIZE"
COMPILE_FLAG=""
[[ "$COMPILE" -eq 1 ]] && COMPILE_FLAG="--compile"
MAX_STEPS_FLAG=""
[[ -n "$MAX_STEPS" ]] && MAX_STEPS_FLAG="--max-steps $MAX_STEPS"

# ── Env ──────────────────────────────────────────────────────────────────
REPO_ROOT=/u/dssc/zenocosini/decomposing-activations-local-geometry

mkdir -p "$REPO_ROOT/logs/jobs" "$OUT_DIR"
cd "$REPO_ROOT" || exit 1
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export NCCL_ASYNC_ERROR_HANDLING=${NCCL_ASYNC_ERROR_HANDLING:-1}

# Auto-derive from the --gres allocation; env override still wins.
NPROC=${NPROC:-${SLURM_GPUS_ON_NODE:-2}}

if [[ ! -f "$OUT_DIR/centroids.pt" && -f "$CENTROIDS_FROM" ]]; then
    cp -n "$CENTROIDS_FROM" "$OUT_DIR/centroids.pt"
fi

echo "=== $(date) === job $SLURM_JOB_ID.$SLURM_ARRAY_TASK_ID on $(hostname) ==="
echo "repo_root: $REPO_ROOT"
echo "shard_dir: $SHARD_DIR   layer: $LAYER   out_dir: $OUT_DIR"
echo "K=$K  rank=$RANK  epochs=$EPOCHS  refine=$REFINE_EPOCHS  batch=$BATCH  num_workers=$NUM_WORKERS  nproc=$NPROC"
echo "training_mode=component_shard  val_frac=$VAL_FRAC  compile=$COMPILE  centroids_from=$CENTROIDS_FROM"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# ── Run (component-sharded MFA via torchrun) ─────────────────────────────
uv run python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$NPROC" \
    -m dalg.cli.run_training \
    --shard-dir "$SHARD_DIR" --layer "$LAYER" --out-dir "$OUT_DIR" \
    --K "$K" --rank "$RANK" --epochs "$EPOCHS" \
    --refine-epochs "$REFINE_EPOCHS" \
    --batch-size "$BATCH" --num-workers "$NUM_WORKERS" \
    --val-frac "$VAL_FRAC" --split-seed "$SPLIT_SEED" \
    --device cuda --seed "$SEED" \
    --training-mode component_shard \
    --wandb --wandb-project dalg-mfa --wandb-name "smoketest_L5_K1000_q337_$(date +%H%M%S)" \
    $POOL_FLAG $COMPILE_FLAG $MAX_STEPS_FLAG

echo "=== $(date) === done ==="
