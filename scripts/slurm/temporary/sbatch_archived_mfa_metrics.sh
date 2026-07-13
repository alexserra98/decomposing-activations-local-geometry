#!/bin/bash
#SBATCH --partition=H100
#SBATCH --account=LADE
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=640G
#SBATCH --time=1-02:00:00
#SBATCH --job-name=arch_mfa_metrics
#SBATCH --array=0-5%1
#SBATCH --output=/u/dssc/zenocosini/decomposing-activations-local-geometry/logs/experiments/arch_mfa_metrics_%A_%a.out

set -euo pipefail

REPO_ROOT=/u/dssc/zenocosini/decomposing-activations-local-geometry
SHARD_DIR=/orfeo/scratch/dssc/zenocosini/dalg-cache/pile_gemma2b_activations
MODELS_DIR=/orfeo/scratch/dssc/zenocosini/dalg-cache/pile_gemma2b_models
ARCHIVE_DIR="$MODELS_DIR/archived"
MAX_SAMPLES_PER_CLUSTER=${MAX_SAMPLES_PER_CLUSTER:-2000}
PCA_WORKERS=${PCA_WORKERS:-8}
OVERLAP_BATCH_PAIRS=${OVERLAP_BATCH_PAIRS:-8192}
SKIP_OVERLAP=${SKIP_OVERLAP:-0}

CONFIGS=(
  "layer05_1000_10_mfa:5:1000_05"
  "layer05_8000_mfa:5:8000_05"
  "layer05_32000_mfa:5:32000_05"
  "layer17_1000_mfa:17:1000_17"
  "layer17_8000_mfa:17:8000_17"
  "layer17_32000_mfa:17:32000_17"
)

IFS=":" read -r RUN_NAME LAYER EXPERIMENT_NAME <<< "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
RUN_DIR="$ARCHIVE_DIR/$RUN_NAME"
ASSIGNMENTS_PATH="$RUN_DIR/mfa_model_assignments.pt"
OUT_DIR="$RUN_DIR"

cd "$REPO_ROOT" || exit 1
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUT_DIR"

echo "Archived MFA metrics"
echo "run:                  $RUN_DIR"
echo "layer:                $LAYER"
echo "assignments:          $ASSIGNMENTS_PATH"
echo "out_dir:              $OUT_DIR"
echo "max_samples/cluster:  $MAX_SAMPLES_PER_CLUSTER"
echo "pca_workers:          $PCA_WORKERS"
echo "overlap_batch_pairs:  $OVERLAP_BATCH_PAIRS"

test -f "$ASSIGNMENTS_PATH"

uv run dalg-run-metrics intrinsic-dim \
  --data-dir "$RUN_DIR" \
  --assignments-path "$ASSIGNMENTS_PATH" \
  --shard-dir "$SHARD_DIR" \
  --layer "$LAYER" \
  --out-dir "$OUT_DIR" \
  --device cuda \
  --pca-device cpu \
  --pca-workers "$PCA_WORKERS" \
  --max-samples-per-cluster "$MAX_SAMPLES_PER_CLUSTER"

if [[ "$SKIP_OVERLAP" != "1" ]]; then
  uv run dalg-run-metrics overlap \
    --data-dir "$RUN_DIR" \
    --out-dir "$OUT_DIR" \
    --device cuda \
    --batch-pairs "$OVERLAP_BATCH_PAIRS"
fi
