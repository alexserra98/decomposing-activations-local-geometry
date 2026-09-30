#!/usr/bin/env bash
#SBATCH --job-name=toy-d128-knn-smoke
#SBATCH --partition=H100
#SBATCH --account=LADE
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/orfeo/scratch/dssc/zenocosini/dalg-cache/michele/outputs/experiments/toy_heldout_coverage_d128/logs/noise10_knn_smoke_%j.out

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
SHARD_DIR="/orfeo/scratch/dssc/zenocosini/dalg-cache/assets/toy_manifolds_10types_1each_D128_30Keach_noise_sweep_seed0/noise_ratio_10"
OUTPUT_ROOT="/orfeo/scratch/dssc/zenocosini/dalg-cache/michele/outputs/experiments/toy_heldout_coverage_d128/smoke_noise10_k8_q2_seed52_knn4_20260929"

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/dalg-toy-d128-knn-smoke-${SLURM_JOB_ID:-local}"
mkdir -p "$MPLCONFIGDIR"

.venv/bin/python -m pytest -q tests/test_neighborhood_pca.py
.venv/bin/python scripts/temporary/run_toy_d128_heldout_coverage.py \
  --shard-dir "$SHARD_DIR" \
  --output-dir "$OUTPUT_ROOT" \
  --layer 0 \
  --train-fraction 0.70 \
  --validation-fraction 0.15 \
  --split-seed 42 \
  --seed 52 \
  --K 8 \
  --rank 2 \
  --pca-method knn \
  --pca-neighbors 4 \
  --kmeans-iterations 10 \
  --kmeans-restarts 1 \
  --epochs 2 \
  --early-stop-patience 2 \
  --batch-size 2048 \
  --device cuda

.venv/bin/python scripts/temporary/run_toy_d128_heldout_coverage.py \
  --shard-dir "$SHARD_DIR" \
  --output-dir "$OUTPUT_ROOT" \
  --layer 0 \
  --train-fraction 0.70 \
  --validation-fraction 0.15 \
  --split-seed 42 \
  --seed 52 \
  --K 8 \
  --rank 2 \
  --pca-method knn \
  --pca-neighbors 4 \
  --kmeans-iterations 10 \
  --kmeans-restarts 1 \
  --epochs 2 \
  --early-stop-patience 2 \
  --batch-size 2048 \
  --geometry-only \
  --device cuda
