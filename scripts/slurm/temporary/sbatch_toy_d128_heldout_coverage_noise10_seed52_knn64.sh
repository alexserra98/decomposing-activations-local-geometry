#!/usr/bin/env bash
#SBATCH --job-name=toy-d128-n10-knn64
#SBATCH --partition=H100
#SBATCH --account=LADE
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=/orfeo/scratch/dssc/zenocosini/dalg-cache/michele/outputs/experiments/toy_heldout_coverage_d128/noise10_k1000_q32_seed52_knn64/logs/slurm_%j.out

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
SHARD_DIR="/orfeo/scratch/dssc/zenocosini/dalg-cache/assets/toy_manifolds_10types_1each_D128_30Keach_noise_sweep_seed0/noise_ratio_10"
OUTPUT_ROOT="/orfeo/scratch/dssc/zenocosini/dalg-cache/michele/outputs/experiments/toy_heldout_coverage_d128/noise10_k1000_q32_seed52_knn64"

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/dalg-toy-d128-noise10-seed52-knn64-${SLURM_JOB_ID:-local}"
mkdir -p "$MPLCONFIGDIR"

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
.venv/bin/python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0)); assert torch.cuda.is_available()'
jq -e '.generator_config.noise_ratio == 10' "$SHARD_DIR/config.json"

.venv/bin/python scripts/temporary/run_toy_d128_heldout_coverage.py \
  --shard-dir "$SHARD_DIR" \
  --output-dir "$OUTPUT_ROOT" \
  --layer 0 \
  --train-fraction 0.70 \
  --validation-fraction 0.15 \
  --split-seed 42 \
  --seed 52 \
  --K 1000 \
  --rank 32 \
  --pca-method knn \
  --pca-neighbors 64 \
  --device cuda

.venv/bin/python scripts/temporary/run_toy_d128_heldout_coverage.py \
  --shard-dir "$SHARD_DIR" \
  --output-dir "$OUTPUT_ROOT" \
  --layer 0 \
  --train-fraction 0.70 \
  --validation-fraction 0.15 \
  --split-seed 42 \
  --seed 52 \
  --K 1000 \
  --rank 32 \
  --pca-method knn \
  --pca-neighbors 64 \
  --geometry-only \
  --device cuda
