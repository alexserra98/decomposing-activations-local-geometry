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
#SBATCH --job-name=toy_noise_k1000
#SBATCH --output=/u/dssc/zenocosini/decomposing-activations-local-geometry/logs/jobs/toy_noise_k1000_%A_%a.out

set -euo pipefail

REPO_ROOT=/u/dssc/zenocosini/decomposing-activations-local-geometry
EXPERIMENT_ROOT=$REPO_ROOT/dalg-cache/experiments/toy_manifolds_10types_1each_D128_30Keach_noise_sweep_seed0
NOISE_RATIOS=(10 100 1000 10000)
NOISE_RATIO=${NOISE_RATIOS[$SLURM_ARRAY_TASK_ID]}
CONDITION_DIR=$EXPERIMENT_ROOT/noise_ratio_$NOISE_RATIO

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1

for FIT in full sample35pct; do
  SAMPLE_FRACTION=1.0
  if [[ "$FIT" == sample35pct ]]; then
    SAMPLE_FRACTION=0.35
  fi
  echo "Starting noise_ratio=$NOISE_RATIO fit=$FIT at $(date --iso-8601=seconds)"
  .venv/bin/python scripts/temporary/build_toy_kmeans_centroids.py \
    --shard-dir "$CONDITION_DIR/dataset" \
    --layer 0 \
    --K 1000 \
    --out-dir "$CONDITION_DIR/centroids/kmeans_k1000_$FIT" \
    --max-iter 100 \
    --restarts 10 \
    --tol 1e-6 \
    --seed 0 \
    --sample-fraction "$SAMPLE_FRACTION" \
    --sample-seed 0 \
    --device cuda \
    --load-batch-size 20000 \
    --block-x 8192 \
    --block-c 8192 \
    --pca-rank 0
  echo "Completed noise_ratio=$NOISE_RATIO fit=$FIT at $(date --iso-8601=seconds)"
done
