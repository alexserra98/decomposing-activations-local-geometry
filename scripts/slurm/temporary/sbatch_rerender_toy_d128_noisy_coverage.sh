#!/usr/bin/env bash
#SBATCH --job-name=toy-d128-rerender
#SBATCH --partition=H100
#SBATCH --account=LADE
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/orfeo/scratch/dssc/zenocosini/dalg-cache/michele/outputs/experiments/toy_heldout_coverage_d128/logs/rerender_noisy_%j.out

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
SOURCE_ROOT="/orfeo/scratch/dssc/zenocosini/dalg-cache/assets/toy_manifolds_10types_1each_D128_30Keach_noise_sweep_seed0"
OUTPUT_BASE="/orfeo/scratch/dssc/zenocosini/dalg-cache/michele/outputs/experiments/toy_heldout_coverage_d128"
ARCHIVE_NAME="figures_invalid_noiseless_title_20260929"

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/dalg-toy-d128-rerender-${SLURM_JOB_ID:-local}"
mkdir -p "$MPLCONFIGDIR"

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
.venv/bin/python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0)); assert torch.cuda.is_available()'

for noise_ratio in 1000 100; do
  shard_dir="${SOURCE_ROOT}/noise_ratio_${noise_ratio}"
  output_root="${OUTPUT_BASE}/noise${noise_ratio}_k1000_q32_seed42"
  archive_dir="${output_root}/${ARCHIVE_NAME}"
  mkdir -p "$archive_dir"
  cp "$output_root/figures/coverage_curve.png" "$archive_dir/coverage_curve.png"
  cp "$output_root/figures/embedding_coverage_2d.png" "$archive_dir/embedding_coverage_2d.png"
  cp "$output_root/README.md" "$archive_dir/README.md"

  .venv/bin/python scripts/temporary/run_toy_d128_heldout_coverage.py \
    --shard-dir "$shard_dir" \
    --output-dir "$output_root" \
    --layer 0 \
    --train-fraction 0.70 \
    --validation-fraction 0.15 \
    --split-seed 42 \
    --seed 42 \
    --K 1000 \
    --rank 32 \
    --coverage-only \
    --render-existing \
    --device cuda
done
