#!/bin/bash
#SBATCH --partition=H100
#SBATCH --account=LADE
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=80G
#SBATCH --time=00:30:00
#SBATCH --job-name=toy30k_kmeans_geometry
#SBATCH --output=/u/dssc/zenocosini/decomposing-activations-local-geometry/logs/jobs/toy30k_kmeans_geometry_%j.out

set -euo pipefail

REPO_ROOT=/u/dssc/zenocosini/decomposing-activations-local-geometry
SHARD_DIR=$REPO_ROOT/dalg-cache/assets/toy_manifolds_circle_helix_torus_D128_30K_noise1e4_shards
CENTROID_DIR=$REPO_ROOT/dalg-cache/toy_manifold_models_30k/centroids/kmeans_k300
CENTROIDS_PATH=$CENTROID_DIR/centroids.pt
OUT_DIR=$CENTROID_DIR/initialization_evaluation
ASSIGNMENTS_PATH=$OUT_DIR/nearest_centroid_assignments.pt
METRICS_PATH=$OUT_DIR/metrics.json
DETAILS_PATH=$OUT_DIR/component_metrics.pt

mkdir -p "$REPO_ROOT/logs/jobs" "$OUT_DIR"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

echo "=== $(date) === job $SLURM_JOB_ID on $(hostname) ==="
echo "centroids_path=$CENTROIDS_PATH"
echo "shard_dir=$SHARD_DIR"
echo "out_dir=$OUT_DIR"
echo "overwriting_assignments=$ASSIGNMENTS_PATH"
echo "overwriting_metrics=$METRICS_PATH"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

.venv/bin/dalg-run-metrics assignments \
  --centroids-path "$CENTROIDS_PATH" \
  --shard-dir "$SHARD_DIR" \
  --layer 0 \
  --drop-prefix 0 \
  --batch-size 8192 \
  --device cuda \
  --save-path "$ASSIGNMENTS_PATH"

.venv/bin/python scripts/temporary/evaluate_toy_kmeans_geometry.py \
  --centroids-path "$CENTROIDS_PATH" \
  --assignments-path "$ASSIGNMENTS_PATH" \
  --shard-dir "$SHARD_DIR" \
  --layer 0 \
  --device cuda \
  --batch-size 8192 \
  --q-max 16 \
  --min-population 26 \
  --cattell-thresholds 0.005 0.01 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.5 1.0 \
  --relative-boundary-eigengap-threshold 1e-6 \
  --output-path "$METRICS_PATH" \
  --details-path "$DETAILS_PATH" \
  --overwrite

echo "=== $(date) === KMeans geometry evaluation complete ==="
