#!/usr/bin/env bash
#SBATCH --job-name=years-pc1-density-pca75
#SBATCH --partition=THIN
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/jobs/years_pc1_density_pca75_%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/dalg-pc1-pca75-matplotlib-${SLURM_JOB_ID:?}"

.venv/bin/python -u scripts/temporary/run_years_pc1_density.py \
    --global-pca outputs/experiments/years_layer04_pc1_density_pca75/global_pca_training_v1.pt \
    --output "outputs/experiments/years_layer04_pc1_density_pca75/job_${SLURM_JOB_ID:?}" \
    --threads "$OMP_NUM_THREADS"

.venv/bin/python -u scripts/temporary/validate_years_pc1_integration.py \
    --results "outputs/experiments/years_layer04_pc1_density_pca75/job_${SLURM_JOB_ID:?}" \
    --threads "$OMP_NUM_THREADS"
