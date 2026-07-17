#!/bin/bash
#SBATCH --partition=H100
#SBATCH --account=LADE
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
##SBATCH --nodelist=dgx002
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=640G
#SBATCH --time=1-02:00:00
#SBATCH --job-name=intrinsic_dim
##SBATCH --begin=now+4hours
#SBATCH --array=5,17
#SBATCH --output=/u/dssc/zenocosini/decomposing-activations-local-geometry/logs/experiments/intrinsic_dim_%x_%A_%a.out

set -euo pipefail

REPO_ROOT=/u/dssc/zenocosini/decomposing-activations-local-geometry
K=${K:-1000}
RANK=${RANK:-10}
METRIC_TARGET=${METRIC_TARGET:-mfa}
OUTPUT_FILENAME=${OUTPUT_FILENAME:-intrinsic_dims.pt}
MAX_SAMPLES=${MAX_SAMPLES:-10000}
MIN_POPULATION=${MIN_POPULATION:-}
TOP_PCS=${TOP_PCS:-}
GRIDE_RANGE_MAX=${GRIDE_RANGE_MAX:-8192}
LAYER=$SLURM_ARRAY_TASK_ID

GRIDE_ARGS=(--gride-range-max "$GRIDE_RANGE_MAX")
if [[ "${COMPUTE_GRIDE:-1}" == "0" ]]; then
  GRIDE_ARGS+=(--no-gride)
fi

SHARD_DIR=/orfeo/scratch/dssc/zenocosini/dalg-cache/pile_gemma2b_activations
MODELS_DIR=${MODELS_DIR:-/orfeo/scratch/dssc/zenocosini/dalg-cache/pile_gemma2b_models}
CENTROIDS_DIR=${CENTROIDS_DIR:-"$MODELS_DIR/centroids"}
LAYER_TAG="layer$(printf '%02d' "$LAYER")"
cd "$REPO_ROOT" || exit 1
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# Step 1: compute cluster assignments (required by intrinsic-dim)
# load_mfa falls back to mfa_model_shards.json when mfa_model.pt is absent
# uv run python -m dalg.analysis.cluster_assignments \
#   --model-path "$DATA_DIR/mfa_model.pt" \
#   --shard-dir "$SHARD_DIR" \
#   --layer "$LAYER" \
#   --device cuda --num-workers 4

case "$METRIC_TARGET" in
  mfa)
    DATA_DIR="$MODELS_DIR/${LAYER_TAG}_${K}_${RANK}_component_sharded_mfa"
    OUT_DIR="$DATA_DIR"
    mkdir -p "$OUT_DIR"
    RUN_OUT_DIR="$OUT_DIR"
    if [[ "$OUTPUT_FILENAME" != "intrinsic_dims.pt" ]]; then
      RUN_OUT_DIR="$OUT_DIR/.tmp_${OUTPUT_FILENAME%.pt}_${SLURM_JOB_ID}_${LAYER}"
      mkdir -p "$RUN_OUT_DIR"
    fi

    uv run dalg-run-metrics intrinsic-dim \
      --data-dir "$DATA_DIR" \
      ${ASSIGNMENTS_PATH:+--assignments-path "$ASSIGNMENTS_PATH"} \
      --shard-dir "$SHARD_DIR" \
      --layer "$LAYER" \
      --out-dir "$RUN_OUT_DIR" \
      --device cuda \
      --max-samples-per-cluster "$MAX_SAMPLES" \
      "${GRIDE_ARGS[@]}" \
      ${MIN_POPULATION:+--min-population "$MIN_POPULATION"} \
      ${TOP_PCS:+--top-pcs "$TOP_PCS"}

    if [[ "$OUTPUT_FILENAME" != "intrinsic_dims.pt" ]]; then
      mv "$RUN_OUT_DIR/intrinsic_dims.pt" "$OUT_DIR/$OUTPUT_FILENAME"
      rmdir "$RUN_OUT_DIR"
    fi
    ;;

  centroids)
    CENTROID_DIR="$CENTROIDS_DIR/k${K}_L$(printf '%02d' "$LAYER")"
    ASSIGNMENTS_PATH="${ASSIGNMENTS_PATH:-$CENTROID_DIR/kmeans_centroid_assignments.pt}"
    OUT_DIR="$CENTROID_DIR"
    mkdir -p "$OUT_DIR"
    RUN_OUT_DIR="$OUT_DIR"
    if [[ "$OUTPUT_FILENAME" != "intrinsic_dims.pt" ]]; then
      RUN_OUT_DIR="$OUT_DIR/.tmp_${OUTPUT_FILENAME%.pt}_${SLURM_JOB_ID}_${LAYER}"
      mkdir -p "$RUN_OUT_DIR"
    fi

    uv run dalg-run-metrics intrinsic-dim \
      --assignments-path "$ASSIGNMENTS_PATH" \
      --shard-dir "$SHARD_DIR" \
      --layer "$LAYER" \
      --out-dir "$RUN_OUT_DIR" \
      --device cuda \
      --max-samples-per-cluster "$MAX_SAMPLES" \
      "${GRIDE_ARGS[@]}" \
      ${MIN_POPULATION:+--min-population "$MIN_POPULATION"} \
      ${TOP_PCS:+--top-pcs "$TOP_PCS"}

    if [[ "$OUTPUT_FILENAME" != "intrinsic_dims.pt" ]]; then
      mv "$RUN_OUT_DIR/intrinsic_dims.pt" "$OUT_DIR/$OUTPUT_FILENAME"
      rmdir "$RUN_OUT_DIR"
    fi
    ;;

  *)
    echo "Unknown METRIC_TARGET=$METRIC_TARGET; expected 'mfa' or 'centroids'." >&2
    exit 2
    ;;
esac

# Step 3: pairwise Gaussian overlap between MFA components
# batch_pairs=512: peak GPU ≈ 10 GB (7 W-type tensors × 512 × D × q × 4 bytes)
# default 4096 OOMs because W-chunk (4096, 2048, 337) alone exceeds H100 memory
# uv run dalg-run-metrics gaussian-overlap \
#   --data-dir "$DATA_DIR" \
#   --out-dir "$OUT_DIR" \
#   --device cuda --batch-pairs 512
