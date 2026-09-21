#!/bin/bash
#SBATCH --partition=H100
#SBATCH --account=LADE
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --job-name=toy_noise10_knn64
#SBATCH --output=/u/dssc/zenocosini/decomposing-activations-local-geometry/logs/jobs/toy_noise10_knn64_%j.out

set -euo pipefail

REPO_ROOT=/u/dssc/zenocosini/decomposing-activations-local-geometry
CENTROID_ROOT=$REPO_ROOT/dalg-cache/toy_manifolds_10types_1each_D128_30Keach_noise_sweep_seed0/noise_ratio_10/centroids
SHARD_DIR=$REPO_ROOT/dalg-cache/assets/toy_manifolds_10types_1each_D128_30Keach_noise_sweep_seed0/noise_ratio_10

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1

.venv/bin/python scripts/temporary/build_toy_knn_pca_centroids.py \
  --shard-dir "$SHARD_DIR" \
  --layer 0 \
  --centroids-path "$CENTROID_ROOT/kmeans_k1000_full/centroids.pt" \
  --out-dir "$CENTROID_ROOT/kmeans_k1000_full_knn64_pca16_experimental" \
  --sample-fraction 1.0 --sample-seed 0 \
  --neighbors 64 --pca-rank 16 --device cuda \
  --load-batch-size 20000 --point-block-size 8192 --eig-batch-size 128
