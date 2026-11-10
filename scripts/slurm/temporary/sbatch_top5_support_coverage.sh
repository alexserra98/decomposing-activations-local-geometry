#!/usr/bin/env bash
#SBATCH --job-name=top5-support-coverage
#SBATCH --partition=THIN
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --time=02:00:00
#SBATCH --output=logs/jobs/top5_support_coverage_%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/dalg-support-${SLURM_JOB_ID:?}"
srun .venv/bin/python -u scripts/temporary/run_top5_support_coverage.py \
  --output "outputs/experiments/top5_support_coverage/job_${SLURM_ARRAY_JOB_ID:?}" \
  --experiment-index "${SLURM_ARRAY_TASK_ID:?}" --workers 6 "$@"
