#!/usr/bin/env bash
#SBATCH --job-name=toy-d128-noise
#SBATCH --partition=H100
#SBATCH --account=LADE
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --array=0-2%3
#SBATCH --output=/orfeo/scratch/dssc/zenocosini/dalg-cache/michele/outputs/experiments/toy_heldout_coverage_d128/logs/noise_sweep_%A_%a.out

set -euo pipefail

case "${SLURM_ARRAY_TASK_ID}" in
  0) NOISE_RATIO=1000 ;;
  1) NOISE_RATIO=100 ;;
  2) NOISE_RATIO=10 ;;
  *) echo "unexpected array index: ${SLURM_ARRAY_TASK_ID}" >&2; exit 2 ;;
esac

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
SOURCE_ROOT="/orfeo/scratch/dssc/zenocosini/dalg-cache/assets/toy_manifolds_10types_1each_D128_30Keach_noise_sweep_seed0"
OUTPUT_BASE="/orfeo/scratch/dssc/zenocosini/dalg-cache/michele/outputs/experiments/toy_heldout_coverage_d128"
SHARD_DIR="${SOURCE_ROOT}/noise_ratio_${NOISE_RATIO}"
OUTPUT_ROOT="${OUTPUT_BASE}/noise${NOISE_RATIO}_k1000_q32_seed42"

mkdir -p "${OUTPUT_ROOT}/logs"
TASK_LOG="${OUTPUT_ROOT}/logs/slurm_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"
exec > >(tee -a "$TASK_LOG") 2>&1

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/dalg-toy-d128-noise-${SLURM_JOB_ID:-local}"
mkdir -p "$MPLCONFIGDIR"

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
.venv/bin/python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0)); assert torch.cuda.is_available()'
jq -e --argjson ratio "$NOISE_RATIO" '.generator_config.noise_ratio == $ratio' "$SHARD_DIR/config.json"

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
  --geometry-only \
  --device cuda
