#!/bin/bash
#SBATCH --partition=H100
#SBATCH --account=LADE
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --job-name=toy4x1_k400_s35
#SBATCH --output=/u/dssc/zenocosini/decomposing-activations-local-geometry/logs/jobs/toy4x1_k400_s35_%j.out

set -euo pipefail

REPO_ROOT=/u/dssc/zenocosini/decomposing-activations-local-geometry
SHARD_DIR=$REPO_ROOT/dalg-cache/assets/toy_manifolds_circle_helix_torus_helix4d_D128_30K_each_noise1e4_shards
OUT_DIR=$REPO_ROOT/dalg-cache/toy_manifold_models_4types_1each_30Keach/centroids/kmeans_k400_sample35pct_pca16

mkdir -p "$REPO_ROOT/logs/jobs"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

echo "=== $(date) === job $SLURM_JOB_ID on $(hostname) ==="
echo "shard_dir=$SHARD_DIR"
echo "out_dir=$OUT_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

for SAMPLE_SEED in $(seq 0 19); do
  echo "Trying deterministic 35% subset with sample_seed=$SAMPLE_SEED"
  if .venv/bin/python scripts/temporary/build_toy_kmeans_centroids.py \
    --shard-dir "$SHARD_DIR" \
    --layer 0 \
    --K 400 \
    --out-dir "$OUT_DIR" \
    --max-iter 100 \
    --restarts 10 \
    --tol 1e-6 \
    --seed 0 \
    --sample-fraction 0.35 \
    --sample-seed "$SAMPLE_SEED" \
    --device cuda \
    --load-batch-size 20000 \
    --block-x 8192 \
    --block-c 8192 \
    --pca-rank 16; then
    echo "=== $(date) === centroid and PCA construction complete ==="
    exit 0
  fi
done

echo "No sample seed in 0..19 satisfied the rank-16 cluster-size contract" >&2
exit 1
