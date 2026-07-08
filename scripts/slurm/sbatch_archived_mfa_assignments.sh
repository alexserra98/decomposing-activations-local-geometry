#!/bin/bash
#SBATCH --partition=DGX
#SBATCH --account=LADE
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:A100:1
#SBATCH --mem=160G
#SBATCH --time=1-02:00:00
#SBATCH --job-name=arch_mfa_assign
#SBATCH --array=0-5%2
#SBATCH --output=/u/dssc/zenocosini/decomposing-activations-local-geometry/logs/experiments/arch_mfa_assign_%A_%a.out

set -euo pipefail

REPO_ROOT=/u/dssc/zenocosini/decomposing-activations-local-geometry
SHARD_DIR=/orfeo/scratch/dssc/zenocosini/dalg-cache/pile_gemma2b_activations
ARCHIVE_DIR="$SHARD_DIR/archived"
ASSIGN_BATCH_SIZE=${ASSIGN_BATCH_SIZE:-1024}

CONFIGS=(
  "layer05_1000_10_mfa:5"
  "layer05_8000_mfa:5"
  "layer05_32000_mfa:5"
  "layer17_1000_mfa:17"
  "layer17_8000_mfa:17"
  "layer17_32000_mfa:17"
)

IFS=":" read -r RUN_NAME LAYER <<< "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
RUN_DIR="$ARCHIVE_DIR/$RUN_NAME"
SAVE_PATH="$RUN_DIR/mfa_model_assignments.pt"

cd "$REPO_ROOT" || exit 1
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

echo "Archived MFA assignments"
echo "run:        $RUN_DIR"
echo "layer:      $LAYER"
echo "shard_dir:  $SHARD_DIR"
echo "save_path:  $SAVE_PATH"
echo "batch_size: $ASSIGN_BATCH_SIZE"

uv run dalg-run-metrics assignments \
  --data-dir "$RUN_DIR" \
  --shard-dir "$SHARD_DIR" \
  --layer "$LAYER" \
  --batch-size "$ASSIGN_BATCH_SIZE" \
  --device cuda \
  --save-path "$SAVE_PATH"
