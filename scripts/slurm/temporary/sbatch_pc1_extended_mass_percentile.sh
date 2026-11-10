#!/usr/bin/env bash
#SBATCH --job-name=pc1-q75-m80-99
#SBATCH --partition=THIN
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --output=logs/jobs/pc1_extended_%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/dalg-extended-${SLURM_JOB_ID:?}"
srun .venv/bin/python -u scripts/temporary/run_pc1_extended_mass_percentile.py \
 --output "outputs/experiments/pc1_extended_mass_percentile/job_${SLURM_ARRAY_JOB_ID:?}" \
 --experiment-index "${SLURM_ARRAY_TASK_ID:?}" --workers 6 "$@"
