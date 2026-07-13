#!/bin/bash
#SBATCH --partition=H100
#SBATCH --account=LADE
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=640G
#SBATCH --time=1-02:00:00
#SBATCH --job-name=shuf_intr_dim
#SBATCH --array=0-1%1
#SBATCH --output=/u/dssc/zenocosini/decomposing-activations-local-geometry/logs/experiments/shuf_intr_dim_%A_%a.out

set -euo pipefail

REPO_ROOT=/u/dssc/zenocosini/decomposing-activations-local-geometry
SHARD_DIR=/orfeo/scratch/dssc/zenocosini/dalg-cache/pile_gemma2b_activations
MODELS_DIR=/orfeo/scratch/dssc/zenocosini/dalg-cache/pile_gemma2b_models
ARCHIVE_DIR="$MODELS_DIR/archived"
MAX_SAMPLES_PER_CLUSTER=${MAX_SAMPLES_PER_CLUSTER:-2000}
PCA_WORKERS=${PCA_WORKERS:-8}

CONFIGS=(
  "layer05_1000_10_mfa:5:1000_05_shuffled_assignments"
  "layer05_8000_mfa:5:8000_05_shuffled_assignments"
)

IFS=":" read -r RUN_NAME LAYER EXPERIMENT_NAME <<< "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
ASSIGNMENTS_PATH="$ARCHIVE_DIR/$RUN_NAME/mfa_model_assignments_shuffled_seed0.pt"
OUT_DIR="$ARCHIVE_DIR/$RUN_NAME/shuffled_assignments_metrics"

cd "$REPO_ROOT" || exit 1
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUT_DIR"

echo "Shuffled-assignment intrinsic dimension"
echo "run:                  $RUN_NAME"
echo "layer:                $LAYER"
echo "assignments:          $ASSIGNMENTS_PATH"
echo "out_dir:              $OUT_DIR"
echo "max_samples/cluster:  $MAX_SAMPLES_PER_CLUSTER"
echo "pca_workers:          $PCA_WORKERS"

test -f "$ASSIGNMENTS_PATH"

.venv/bin/python -u -m dalg.cli.run_metrics intrinsic-dim \
  --assignments-path "$ASSIGNMENTS_PATH" \
  --shard-dir "$SHARD_DIR" \
  --layer "$LAYER" \
  --out-dir "$OUT_DIR" \
  --device cuda \
  --pca-device cpu \
  --pca-workers "$PCA_WORKERS" \
  --max-samples-per-cluster "$MAX_SAMPLES_PER_CLUSTER" \
  --seed 0
