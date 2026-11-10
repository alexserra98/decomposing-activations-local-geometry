#!/usr/bin/env bash
#SBATCH --job-name=pc1-pak-controlled
#SBATCH --partition=THIN
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/jobs/pc1_pak_controlled_%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/dalg-pak-${SLURM_JOB_ID:?}"
srun .venv/bin/python -u scripts/temporary/run_pc1_pak_controlled.py \
    --output "outputs/experiments/pc1_pak_controlled/job_${SLURM_ARRAY_JOB_ID:?}" \
    --experiment-index "${SLURM_ARRAY_TASK_ID:?}" --workers 6 "$@"
