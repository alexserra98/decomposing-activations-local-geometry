#!/bin/bash
#SBATCH --partition=H100
#SBATCH --account=LADE
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --job-name=toy_kmeans_k5k
#SBATCH --output=/u/dssc/zenocosini/decomposing-activations-local-geometry/logs/jobs/toy_kmeans_k5000_%j.out

set -euo pipefail

REPO_ROOT=/u/dssc/zenocosini/decomposing-activations-local-geometry
SHARD_DIR=$REPO_ROOT/dalg-cache/assets/toy_manifolds_circle_helix_D128_1M_noise1e4_shards
OUT_DIR=$REPO_ROOT/dalg-cache/toy_manifold_models_1M/centroids/kmeans_k5000

mkdir -p "$REPO_ROOT/logs/jobs"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

echo "=== $(date) === job $SLURM_JOB_ID on $(hostname) ==="
echo "shard_dir=$SHARD_DIR"
echo "out_dir=$OUT_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

.venv/bin/python scripts/temporary/build_toy_kmeans_centroids.py \
  --shard-dir "$SHARD_DIR" \
  --layer 0 \
  --K 5000 \
  --out-dir "$OUT_DIR" \
  --max-iter 100 \
  --restarts 10 \
  --tol 1e-6 \
  --seed 0 \
  --device cuda \
  --load-batch-size 20000 \
  --block-x 8192 \
  --block-c 8192 \
  --pca-rank 32 \
  --pca-only

echo "=== $(date) === centroid PCA enrichment complete ==="
