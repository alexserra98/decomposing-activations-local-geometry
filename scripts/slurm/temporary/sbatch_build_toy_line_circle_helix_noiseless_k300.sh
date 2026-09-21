#!/bin/bash
#SBATCH --partition=H100
#SBATCH --account=LADE
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --job-name=toy3_noiseless_k300
#SBATCH --output=/u/dssc/zenocosini/decomposing-activations-local-geometry/logs/jobs/toy3_noiseless_k300_%j.out

set -euo pipefail

REPO_ROOT=/u/dssc/zenocosini/decomposing-activations-local-geometry
SHARD_DIR=$REPO_ROOT/dalg-cache/assets/toy_manifolds_line_circle_helix_D128_20Keach_noiseless_shards
OUT_DIR=$REPO_ROOT/dalg-cache/toy_manifold_models_line_circle_helix_20Keach_noiseless/centroids/kmeans_k300_pca16

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1

.venv/bin/python scripts/temporary/build_toy_line_circle_helix_noiseless.py \
  --shard-dir "$SHARD_DIR"

.venv/bin/python scripts/temporary/build_toy_kmeans_centroids.py \
  --shard-dir "$SHARD_DIR" \
  --layer 0 \
  --K 300 \
  --out-dir "$OUT_DIR" \
  --max-iter 100 \
  --restarts 10 \
  --tol 1e-6 \
  --seed 0 \
  --device cuda \
  --load-batch-size 20000 \
  --block-x 8192 \
  --block-c 8192 \
  --pca-rank 16
