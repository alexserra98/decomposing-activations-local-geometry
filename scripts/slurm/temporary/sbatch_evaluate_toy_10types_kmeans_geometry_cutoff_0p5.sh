#!/bin/bash
#SBATCH --partition=H100
#SBATCH --account=LADE
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --job-name=toy10x1_kmeans_c05
#SBATCH --output=/u/dssc/zenocosini/decomposing-activations-local-geometry/logs/jobs/toy10x1_kmeans_geometry_cutoff_0p5_%j.out

set -euo pipefail

REPO_ROOT=/u/dssc/zenocosini/decomposing-activations-local-geometry
SHARD_DIR=$REPO_ROOT/dalg-cache/assets/toy_manifolds_10types_1each_D128_15Keach_noise1e4_shards
CENTROID_DIR=$REPO_ROOT/dalg-cache/toy_manifold_models_10types_1each_15Keach/centroids/kmeans_k1000
EVALUATION_DIR=$CENTROID_DIR/initialization_evaluation
CENTROIDS_PATH=$CENTROID_DIR/centroids.pt
ASSIGNMENTS_PATH=$EVALUATION_DIR/nearest_centroid_assignments.pt
METRICS_PATH=$EVALUATION_DIR/metrics_cutoff_0p5.json

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

echo "=== $(date) === job $SLURM_JOB_ID on $(hostname) ==="
echo "centroids_path=$CENTROIDS_PATH"
echo "assignments_path=$ASSIGNMENTS_PATH"
echo "cutoff=0.5"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if [[ -e "$METRICS_PATH" ]]; then
  echo "Refusing to overwrite existing metrics: $METRICS_PATH" >&2
  exit 1
fi

.venv/bin/python scripts/temporary/evaluate_toy_kmeans_geometry.py \
  --centroids-path "$CENTROIDS_PATH" \
  --assignments-path "$ASSIGNMENTS_PATH" \
  --shard-dir "$SHARD_DIR" \
  --layer 0 \
  --device cuda \
  --batch-size 10000 \
  --max-mean-to-manifold-distance 0.5 \
  --relative-boundary-eigengap-threshold 1e-6 \
  --output-path "$METRICS_PATH"

echo "=== $(date) === cutoff-0.5 KMeans geometry evaluation complete ==="
