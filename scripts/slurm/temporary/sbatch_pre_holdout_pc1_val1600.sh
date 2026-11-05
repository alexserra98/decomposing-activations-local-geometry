#!/usr/bin/env bash
#SBATCH --job-name=pc1-val1600
#SBATCH --partition=H100
#SBATCH --account=LADE
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/jobs/pre_holdout_pc1_val1600_%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
.venv/bin/python -u scripts/temporary/rerun_pre_holdout_pc1.py evaluate --index "$SLURM_ARRAY_TASK_ID" --reference-seed 43 --reference-size 1600
