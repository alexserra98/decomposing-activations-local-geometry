#!/bin/bash
#SBATCH --partition=H100
#SBATCH --account=LADE
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=640G
#SBATCH --time=1-02:00:00
#SBATCH --array=0-6%5
#SBATCH --job-name=all_mfa_metrics
#SBATCH --output=/u/dssc/zenocosini/decomposing-activations-local-geometry/logs/experiments/all_mfa_metrics_%x_%A_%a.out

set -euo pipefail

REPO_ROOT=/u/dssc/zenocosini/decomposing-activations-local-geometry
MODELS_DIR=/orfeo/scratch/dssc/zenocosini/dalg-cache/pile_gemma2b_models
SHARD_DIR=/orfeo/scratch/dssc/zenocosini/dalg-cache/pile_gemma2b_activations
STAGE=${STAGE:?Submit with STAGE=assignments, intrinsic_dim, or gaussian_overlap}

# Keep this explicit: the two 32,000-component runs are intentionally excluded.
RUN_SPECS=(
  "layer05_1000_10_component_sharded_mfa:5:1000:10"
  "layer05_1000_10_mfa_1epoch_20260703_1538:5:1000:10"
  "layer05_1000_394_component_sharded_mfa:5:1000:394"
  "layer05_8000_10_component_sharded_mfa:5:8000:10"
  "layer17_1000_10_component_sharded_mfa:17:1000:10"
  "layer17_1000_337_component_sharded_mfa:17:1000:337"
  "layer17_8000_10_component_sharded_mfa:17:8000:10"
)

IFS=: read -r RUN_NAME LAYER K RANK <<< "${RUN_SPECS[$SLURM_ARRAY_TASK_ID]}"
RUN_DIR="$MODELS_DIR/$RUN_NAME"
TMP_DIR="$RUN_DIR/.metrics_tmp_${SLURM_JOB_ID}_${STAGE}"

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$TMP_DIR"
trap 'rm -rf "$TMP_DIR"' EXIT

echo "stage=$STAGE"
echo "run=$RUN_NAME layer=$LAYER K=$K rank=$RANK"
echo "run_dir=$RUN_DIR"

case "$STAGE" in
  assignments)
    BATCH_SIZE=1024
    if (( RANK >= 300 )); then
      BATCH_SIZE=512
    fi
    ASSIGNMENTS_TMP="$TMP_DIR/mfa_model_assignments.pt"
    .venv/bin/python -u -m dalg.cli.run_metrics assignments \
      --data-dir "$RUN_DIR" \
      --shard-dir "$SHARD_DIR" \
      --layer "$LAYER" \
      --batch-size "$BATCH_SIZE" \
      --device cuda \
      --save-path "$ASSIGNMENTS_TMP"
    mv "$ASSIGNMENTS_TMP" "$RUN_DIR/mfa_model_assignments.pt"
    ;;

  intrinsic_dim)
    .venv/bin/python -u -m dalg.cli.run_metrics intrinsic-dim \
      --data-dir "$RUN_DIR" \
      --assignments-path "$RUN_DIR/mfa_model_assignments.pt" \
      --shard-dir "$SHARD_DIR" \
      --layer "$LAYER" \
      --out-dir "$TMP_DIR" \
      --device cuda \
      --pca-device cuda \
      --variance-threshold 0.90 \
      --min-population 100 \
      --max-samples-per-cluster 2000
    mv "$TMP_DIR/intrinsic_dims.pt" "$RUN_DIR/intrinsic_dims.pt"
    ;;

  gaussian_overlap)
    BATCH_PAIRS=4096
    if (( RANK >= 300 )); then
      BATCH_PAIRS=512
    fi
    .venv/bin/python -u -m dalg.cli.run_metrics gaussian-overlap \
      --data-dir "$RUN_DIR" \
      --out-dir "$TMP_DIR" \
      --device cuda \
      --batch-pairs "$BATCH_PAIRS"
    mv "$TMP_DIR/gaussian_overlap.pt" "$RUN_DIR/gaussian_overlap.pt"
    ;;

  *)
    echo "Unknown STAGE=$STAGE; expected assignments, intrinsic_dim, or gaussian_overlap." >&2
    exit 2
    ;;
esac

echo "completed stage=$STAGE run=$RUN_NAME"
