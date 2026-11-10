#!/usr/bin/env bash
#SBATCH --job-name=pc1-knn-mass-percentile
#SBATCH --partition=THIN
#SBATCH --array=0-7%4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=logs/jobs/pc1_neighbor_mass_%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/dalg-neighbor-mass-${SLURM_JOB_ID:?}"
srun .venv/bin/python -u scripts/temporary/run_pc1_neighbor_mass_percentile.py \
 --output "outputs/experiments/pc1_neighbor_mass_percentile/job_${SLURM_ARRAY_JOB_ID:?}" \
 --index "${SLURM_ARRAY_TASK_ID:?}" --workers 6 "$@"
