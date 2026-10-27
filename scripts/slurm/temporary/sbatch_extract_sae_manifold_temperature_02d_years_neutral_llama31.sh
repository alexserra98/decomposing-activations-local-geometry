#!/usr/bin/env bash
#SBATCH --job-name=dalg-llama31-temp-years
#SBATCH --partition=H100
#SBATCH --account=LADE
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/jobs/llama31_temperature_02d_years_neutral_%j.out

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export HF_HUB_DISABLE_PROGRESS_BARS=1

.venv/bin/python -u -m dalg.cli.run_prompt_extraction \
  --dataset dalg-cache/sae_manifold_prompts_temperature_02d_years_neutral \
  --out-dir dalg-cache/sae_manifold_temperature_02d_years_neutral_llama31_8b_last \
  --model meta-llama/Llama-3.1-8B \
  --revision d04e592bb4f6aa9cfee91e2e20afa771667e1d4b \
  --device cuda --dtype bfloat16 \
  --extract-batch-size 8 --shard-size 512
