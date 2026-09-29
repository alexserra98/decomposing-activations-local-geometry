#!/usr/bin/env bash
#SBATCH --job-name=toy-d128-coverage
#SBATCH --partition=H100
#SBATCH --account=LADE
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=/orfeo/scratch/dssc/zenocosini/dalg-cache/michele/outputs/experiments/toy_heldout_coverage_d128/noiseless_k1000_q32_seed42/logs/slurm_%j.out

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
OUTPUT_ROOT="/orfeo/scratch/dssc/zenocosini/dalg-cache/michele/outputs/experiments/toy_heldout_coverage_d128/noiseless_k1000_q32_seed42"
SHARD_DIR="/orfeo/scratch/dssc/zenocosini/dalg-cache/assets/toy_manifolds_10types_1each_D128_30Keach_noise_sweep_seed0/noiseless"

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/dalg-toy-d128-coverage-matplotlib-${SLURM_JOB_ID:-local}"
mkdir -p "$MPLCONFIGDIR"

.venv/bin/python scripts/temporary/run_toy_d128_heldout_coverage.py \
  --shard-dir "$SHARD_DIR" \
  --output-dir "$OUTPUT_ROOT" \
  --layer 0 \
  --train-fraction 0.70 \
  --validation-fraction 0.15 \
  --split-seed 42 \
  --seed 42 \
  --K 1000 \
  --rank 32 \
  --device cuda
