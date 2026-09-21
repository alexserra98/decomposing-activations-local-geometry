#!/bin/bash
#SBATCH --partition=H100
#SBATCH --account=LADE
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --array=0-1
#SBATCH --job-name=toy_noise_full_pca32
#SBATCH --output=/u/dssc/zenocosini/decomposing-activations-local-geometry/logs/jobs/toy_noise_full_pca32_%A_%a.out

set -euo pipefail

REPO_ROOT=/u/dssc/zenocosini/decomposing-activations-local-geometry
EXPERIMENT_ROOT=$REPO_ROOT/dalg-cache/toy_manifolds_10types_1each_D128_30Keach_noise_sweep_seed0
NOISE_RATIOS=(100 1000)
NOISE_RATIO=${NOISE_RATIOS[$SLURM_ARRAY_TASK_ID]}
CONDITION_DIR=$EXPERIMENT_ROOT/noise_ratio_$NOISE_RATIO
SHARD_DIR=$CONDITION_DIR/dataset
CENTROID_DIR=$CONDITION_DIR/centroids/kmeans_k1000_full_pca32
CENTROIDS_PATH=$CENTROID_DIR/centroids.pt
EVALUATION_DIR=$CENTROID_DIR/initialization_evaluation
ASSIGNMENTS_PATH=$EVALUATION_DIR/nearest_centroid_assignments.pt

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1

# The builder refuses nonempty output directories. Each condition is a fresh fit.
.venv/bin/python scripts/temporary/build_toy_kmeans_centroids.py \
  --shard-dir "$SHARD_DIR" --layer 0 --K 1000 \
  --out-dir "$CENTROID_DIR" \
  --max-iter 1000 --restarts 10 --tol 1e-6 --seed 0 \
  --sample-fraction 1.0 --sample-seed 0 \
  --device cuda --load-batch-size 20000 --block-x 8192 --block-c 8192 \
  --pca-rank 32

mkdir -p "$EVALUATION_DIR"
.venv/bin/dalg-run-metrics assignments \
  --centroids-path "$CENTROIDS_PATH" --shard-dir "$SHARD_DIR" \
  --layer 0 --drop-prefix 0 --batch-size 8192 --device cuda \
  --save-path "$ASSIGNMENTS_PATH"

# Use the saved hard-cluster PCs and preserve all strict evaluation tolerances.
.venv/bin/python scripts/temporary/evaluate_toy_kmeans_geometry.py \
  --centroids-path "$CENTROIDS_PATH" --assignments-path "$ASSIGNMENTS_PATH" \
  --shard-dir "$SHARD_DIR" --layer 0 --device cuda --batch-size 10000 \
  --q-max 32 --min-population 33 \
  --cattell-thresholds 0.00025 0.005 0.01 0.05 0.1 0.15 0.2 0.5 1.0 \
  --relative-boundary-eigengap-threshold 1e-6 \
  --output-path "$EVALUATION_DIR/metrics.json"
