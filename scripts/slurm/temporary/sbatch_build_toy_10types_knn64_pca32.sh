#!/bin/bash
#SBATCH --partition=H100
#SBATCH --account=LADE
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --array=0-1
#SBATCH --job-name=toy10x1_knn64_pca32
#SBATCH --output=/u/dssc/zenocosini/decomposing-activations-local-geometry/logs/jobs/toy10x1_knn64_pca32_%A_%a.out

set -euo pipefail

REPO_ROOT=/u/dssc/zenocosini/decomposing-activations-local-geometry
SHARD_DIR=$REPO_ROOT/dalg-cache/assets/toy_manifolds_10types_1each_D128_15Keach_noise1e4_shards
CENTROID_ROOT=$REPO_ROOT/dalg-cache/toy_manifold_models_10types_1each_15Keach/centroids

if [[ $SLURM_ARRAY_TASK_ID -eq 0 ]]; then
  SAMPLE_FRACTION=1.0
  SOURCE_DIR=$CENTROID_ROOT/kmeans_k1000
  OUTPUT_DIR=$CENTROID_ROOT/kmeans_k1000_full_knn64_pca32_experimental
else
  SAMPLE_FRACTION=0.35
  SOURCE_DIR=$CENTROID_ROOT/kmeans_k1000_sample35pct
  OUTPUT_DIR=$CENTROID_ROOT/kmeans_k1000_sample35pct_knn64_pca32_experimental
fi

mkdir -p "$REPO_ROOT/logs/jobs"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

echo "=== $(date) === job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID on $(hostname) ==="
echo "shard_dir=$SHARD_DIR"
echo "source_dir=$SOURCE_DIR"
echo "output_dir=$OUTPUT_DIR"
echo "sample_fraction=$SAMPLE_FRACTION"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# First fit ordinary KMeans means without PCA. The experimental KNN-PCA step
# below reads these means and writes a separate artifact, leaving this standard
# centroid-only source unchanged.
.venv/bin/python scripts/temporary/build_toy_kmeans_centroids.py \
  --shard-dir "$SHARD_DIR" \
  --layer 0 \
  --K 1000 \
  --out-dir "$SOURCE_DIR" \
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

.venv/bin/python scripts/temporary/build_toy_knn_pca_centroids.py \
  --shard-dir "$SHARD_DIR" \
  --layer 0 \
  --centroids-path "$SOURCE_DIR/centroids.pt" \
  --out-dir "$OUTPUT_DIR" \
  --sample-fraction "$SAMPLE_FRACTION" \
  --sample-seed 0 \
  --neighbors 64 \
  --pca-rank 32 \
  --device cuda \
  --load-batch-size 20000 \
  --point-block-size 8192 \
  --eig-batch-size 128

echo "=== $(date) === KMeans and experimental KNN-PCA construction complete ==="
