#!/bin/bash
#SBATCH --partition=H100
#SBATCH --account=LADE
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --job-name=toy10x1_kmeans_geometry
#SBATCH --output=/u/dssc/zenocosini/decomposing-activations-local-geometry/logs/jobs/toy10x1_kmeans_geometry_%j.out

set -euo pipefail

REPO_ROOT=/u/dssc/zenocosini/decomposing-activations-local-geometry
SHARD_DIR=$REPO_ROOT/dalg-cache/assets/toy_manifolds_10types_1each_D128_15Keach_noise1e4_shards
CENTROID_DIR=$REPO_ROOT/dalg-cache/toy_manifold_models_10types_1each_15Keach/centroids/kmeans_k1000
CENTROIDS_PATH=$CENTROID_DIR/centroids.pt
OUT_DIR=$CENTROID_DIR/initialization_evaluation
ASSIGNMENTS_PATH=$OUT_DIR/nearest_centroid_assignments.pt
METRICS_PATH=$OUT_DIR/metrics.json

mkdir -p "$REPO_ROOT/logs/jobs" "$OUT_DIR"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

echo "=== $(date) === job $SLURM_JOB_ID on $(hostname) ==="
echo "centroids_path=$CENTROIDS_PATH"
echo "shard_dir=$SHARD_DIR"
echo "out_dir=$OUT_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if [[ -e "$METRICS_PATH" ]]; then
  echo "Refusing to overwrite existing metrics: $METRICS_PATH" >&2
  exit 1
fi

# The original K=1000 artifact contains centroids only. Add the same rank-16
# empirical cluster-PCA basis used by the earlier K=300 geometry analysis,
# without refitting or moving the centroids.
.venv/bin/python scripts/temporary/build_toy_kmeans_centroids.py \
  --shard-dir "$SHARD_DIR" \
  --layer 0 \
  --K 1000 \
  --out-dir "$CENTROID_DIR" \
  --device cuda \
  --load-batch-size 20000 \
  --block-x 8192 \
  --block-c 8192 \
  --pca-rank 16 \
  --pca-only

if [[ ! -e "$ASSIGNMENTS_PATH" ]]; then
  .venv/bin/dalg-run-metrics assignments \
    --centroids-path "$CENTROIDS_PATH" \
    --shard-dir "$SHARD_DIR" \
    --layer 0 \
    --batch-size 10000 \
    --device cuda \
    --save-path "$ASSIGNMENTS_PATH"
else
  echo "Reusing existing assignments; the evaluator will validate them."
fi

.venv/bin/python scripts/temporary/evaluate_toy_kmeans_geometry.py \
  --centroids-path "$CENTROIDS_PATH" \
  --assignments-path "$ASSIGNMENTS_PATH" \
  --shard-dir "$SHARD_DIR" \
  --layer 0 \
  --device cuda \
  --batch-size 10000 \
  --max-mean-to-manifold-distance 0.1 \
  --relative-boundary-eigengap-threshold 1e-6 \
  --output-path "$METRICS_PATH"

echo "=== $(date) === KMeans geometry evaluation complete ==="
