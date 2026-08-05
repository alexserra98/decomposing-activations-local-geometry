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

set -euo pipefail

# ── Config (override with sbatch --export=ALL,KEY=value) ─────────────────
SHARD_DIR=${SHARD_DIR:-/orfeo/scratch/dssc/zenocosini/dalg-cache/pile_gemma2b_activations}
MODELS_DIR=${MODELS_DIR:-/orfeo/scratch/dssc/zenocosini/dalg-cache/pile_gemma2b_models}
CENTROIDS_DIR=${CENTROIDS_DIR:-"$MODELS_DIR/centroids"}
# SHARD_DIR=${SHARD_DIR:-/orfeo/scratch/dssc/zenocosini/dalg-cache/newsgroups_gemma2b_activations}
LAYER=$SLURM_ARRAY_TASK_ID

SMOKE=${SMOKE:-0}                            # SMOKE=1 runs tiny epochs for quick wiring tests
if [[ "$SMOKE" -eq 1 ]]; then
  K=${K:-64}
  RANK=${RANK:-8}
  EPOCHS=${EPOCHS:-0}                        # unbounded; MAX_STEPS is the smoke safety net
  REFINE_EPOCHS=${REFINE_EPOCHS:-2}
  BATCH=${BATCH:-512}
  NUM_WORKERS=${NUM_WORKERS:-0}
  POOL_SIZE=${POOL_SIZE:-50000}
  VAL_FRAC=${VAL_FRAC:-0.0001}
  STEPS_PER_EPOCH=${STEPS_PER_EPOCH:-2}
  MAX_STEPS=${MAX_STEPS:-6}
  EARLY_STOP_DELTA=${EARLY_STOP_DELTA:-1e9}  # force the early-stop path after two tiny epochs
  COMPILE=${COMPILE:-0}
  WANDB=${WANDB:-0}
  CENTROIDS_PATH=${CENTROIDS_PATH:-}
else
  K=${K:-8000}
  RANK=${RANK:-10}
  EPOCHS=${EPOCHS:-15}
  REFINE_EPOCHS=${REFINE_EPOCHS:-10}
  BATCH=${BATCH:-8192}
  NUM_WORKERS=${NUM_WORKERS:-4}
  POOL_SIZE=${POOL_SIZE:-}
  VAL_FRAC=${VAL_FRAC:-0.05}                 # val runs on every rank via _build_val_loader
  STEPS_PER_EPOCH=${STEPS_PER_EPOCH:-}
  MAX_STEPS=${MAX_STEPS:-}
  EARLY_STOP_DELTA=${EARLY_STOP_DELTA:-1e-3}
  COMPILE=${COMPILE:-1}                      # torch.compile on by default; set COMPILE=0 to disable
  WANDB=${WANDB:-1}
fi
SPLIT_SEED=${SPLIT_SEED:-42}
SEED=${SEED:-42}
ASSIGN_BATCH=${ASSIGN_BATCH:-1024}

LAYER_TAG="layer$(printf '%02d' "$LAYER")"
CENTROID_TAG="k${K}_L$(printf '%02d' "$LAYER")"
if [[ "$SMOKE" -eq 1 ]]; then
  OUT_DIR=${OUT_DIR:-"$MODELS_DIR/${LAYER_TAG}_${K}_${RANK}_smoke_component_sharded_mfa"}
else
  OUT_DIR=${OUT_DIR:-"$MODELS_DIR/${LAYER_TAG}_${K}_${RANK}_component_sharded_mfa"}
  CENTROIDS_PATH=${CENTROIDS_PATH:-"$CENTROIDS_DIR/${CENTROID_TAG}/centroids.pt"}
fi

EXTRA_ARGS=()
[[ -n "$CENTROIDS_PATH" ]] && EXTRA_ARGS+=(--centroids-path "$CENTROIDS_PATH")
[[ -n "$POOL_SIZE" ]] && EXTRA_ARGS+=(--pool-size "$POOL_SIZE")
[[ -n "$STEPS_PER_EPOCH" ]] && EXTRA_ARGS+=(--steps-per-epoch "$STEPS_PER_EPOCH")
[[ -n "$MAX_STEPS" ]] && EXTRA_ARGS+=(--max-steps "$MAX_STEPS")
[[ "$COMPILE" -eq 1 ]] && EXTRA_ARGS+=(--compile)
if [[ "$WANDB" -eq 1 ]]; then
  WANDB_PROJECT=${WANDB_PROJECT:-dalg-mfa}
  WANDB_NAME=${WANDB_NAME:-"run_L$(printf '%02d' "$LAYER")_K${K}_q${RANK}_$(date +%H%M%S)"}
  EXTRA_ARGS+=(--wandb --wandb-project "$WANDB_PROJECT" --wandb-name "$WANDB_NAME")
fi

# ── Env ──────────────────────────────────────────────────────────────────
REPO_ROOT=/u/dssc/zenocosini/decomposing-activations-local-geometry

mkdir -p "$REPO_ROOT/logs/jobs" "$OUT_DIR"
cd "$REPO_ROOT" || exit 1
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export NCCL_ASYNC_ERROR_HANDLING=${NCCL_ASYNC_ERROR_HANDLING:-1}

# Auto-derive from the --gres allocation; env override still wins.
NPROC=${NPROC:-${SLURM_GPUS_ON_NODE:-2}}

echo "=== $(date) === job $SLURM_JOB_ID.$SLURM_ARRAY_TASK_ID on $(hostname) ==="
echo "repo_root: $REPO_ROOT"
echo "shard_dir: $SHARD_DIR   layer: $LAYER   out_dir: $OUT_DIR"
echo "K=$K  rank=$RANK  epochs=$EPOCHS  refine=$REFINE_EPOCHS  batch=$BATCH  num_workers=$NUM_WORKERS  nproc=$NPROC"
echo "training_mode=component_shard  smoke=$SMOKE  val_frac=$VAL_FRAC  early_stop_delta=$EARLY_STOP_DELTA"
echo "steps_per_epoch=${STEPS_PER_EPOCH:-full}  max_steps=${MAX_STEPS:-none}  compile=$COMPILE  wandb=$WANDB"
echo "centroids_path=${CENTROIDS_PATH:-fit}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# ── Run (component-sharded MFA via torchrun) ─────────────────────────────
uv run python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$NPROC" \
    -m dalg.cli.run_training \
    --shard-dir "$SHARD_DIR" --layer "$LAYER" --out-dir "$OUT_DIR" \
    --K "$K" --rank "$RANK" --epochs "$EPOCHS" \
    --refine-epochs "$REFINE_EPOCHS" \
    --batch-size "$BATCH" --num-workers "$NUM_WORKERS" \
    --val-frac "$VAL_FRAC" --split-seed "$SPLIT_SEED" \
    --early-stop-delta "$EARLY_STOP_DELTA" \
    --device cuda --seed "$SEED" \
    --training-mode component_shard \
    "${EXTRA_ARGS[@]}"

echo "=== $(date) === training done; computing assignments (batch=$ASSIGN_BATCH) ==="
uv run dalg-run-metrics assignments \
    --data-dir "$OUT_DIR" \
    --shard-dir "$SHARD_DIR" \
    --layer "$LAYER" \
    --batch-size "$ASSIGN_BATCH" \
    --device cuda \
    --seed "$SEED" \
    --save-path "$OUT_DIR/mfa_model_assignments.pt"

echo "=== $(date) === training + assignments done ==="
