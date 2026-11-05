#!/usr/bin/env bash
#SBATCH --job-name=pc1-valsize-retry
#SBATCH --partition=H100
#SBATCH --account=LADE
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=logs/jobs/pc1_valsize_retry_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export PYTHONPATH="$PWD/outputs/experiments/pc1_validation_size_study/retry_source"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
.venv/bin/python -m dalg.cli.run_pipeline run \
  --manifest outputs/experiments/pc1_validation_size_study/hddc_manifest.jsonl --index 2
