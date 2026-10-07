#!/usr/bin/env bash
#SBATCH --job-name=toy-heldout-coverage
#SBATCH --partition=H100
#SBATCH --account=LADE
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --output=logs/jobs/toy_heldout_coverage_%j.out

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/dalg-toy-coverage-matplotlib-${SLURM_JOB_ID:-local}"
mkdir -p "$MPLCONFIGDIR"

.venv/bin/python scripts/temporary/run_toy_coverage_experiment.py \
  --output-dir outputs/experiments/toy_heldout_coverage/sphere_seed0 \
  --noise-ratios 1000 10 \
  --samples 60000 \
  --calibration-size 20000 \
  --train-fraction 0.70 \
  --validation-fraction 0.15 \
  --split-seed 1729 \
  --seed 0 \
  --K 64 \
  --rank 2 \
  --device cuda
