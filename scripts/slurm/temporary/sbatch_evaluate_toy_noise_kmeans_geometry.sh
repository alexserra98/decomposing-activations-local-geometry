#!/bin/bash
#SBATCH --partition=H100
#SBATCH --account=LADE
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --array=0-3
#SBATCH --job-name=toy_noise_geometry
#SBATCH --output=/u/dssc/zenocosini/decomposing-activations-local-geometry/logs/jobs/toy_noise_geometry_%A_%a.out

set -euo pipefail

REPO_ROOT=/u/dssc/zenocosini/decomposing-activations-local-geometry
EXPERIMENT_ROOT=$REPO_ROOT/dalg-cache/toy_manifolds_10types_1each_D128_30Keach_noise_sweep_seed0
NOISE_RATIOS=(10 100 1000 10000)
NOISE_RATIO=${NOISE_RATIOS[$SLURM_ARRAY_TASK_ID]}
CONDITION_DIR=$EXPERIMENT_ROOT/noise_ratio_$NOISE_RATIO
CENTROIDS_PATH=$CONDITION_DIR/centroids/kmeans_k1000_full/centroids.pt
OUT_DIR=$CONDITION_DIR/centroids/kmeans_k1000_full/initialization_evaluation
ASSIGNMENTS_PATH=$OUT_DIR/nearest_centroid_assignments.pt

cd "$REPO_ROOT"
SHARD_DIR=$(.venv/bin/python -c 'import json,sys; from pathlib import Path; print(json.loads(Path(sys.argv[1]).read_text())["source_shard_dir"])' "$CONDITION_DIR/centroids/kmeans_k1000_full/config.json")
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1

mkdir -p "$OUT_DIR"
if [[ -e "$OUT_DIR/metrics.json" || -e "$OUT_DIR/component_metrics.pt" ]]; then
  echo "Refusing to overwrite existing evaluation: $OUT_DIR" >&2
  exit 1
fi

echo "Starting noise_ratio=$NOISE_RATIO at $(date --iso-8601=seconds)"
if [[ ! -e "$ASSIGNMENTS_PATH" ]]; then
  .venv/bin/dalg-run-metrics assignments \
    --centroids-path "$CENTROIDS_PATH" \
    --shard-dir "$SHARD_DIR" \
    --layer 0 --drop-prefix 0 --batch-size 8192 --device cuda \
    --save-path "$ASSIGNMENTS_PATH"
fi

# Pass optional evaluator arguments (for example --cattell-thresholds) through.
.venv/bin/python scripts/temporary/evaluate_toy_kmeans_geometry.py \
  --centroids-path "$CENTROIDS_PATH" \
  --assignments-path "$ASSIGNMENTS_PATH" \
  --shard-dir "$SHARD_DIR" \
  --layer 0 --device cuda --batch-size 10000 \
  --compute-pca --q-max 16 --min-population 26 \
  --relative-boundary-eigengap-threshold 1e-6 \
  --output-path "$OUT_DIR/metrics.json" "$@"

echo "Completed noise_ratio=$NOISE_RATIO at $(date --iso-8601=seconds)"
