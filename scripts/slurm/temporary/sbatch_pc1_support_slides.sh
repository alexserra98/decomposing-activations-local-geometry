#!/usr/bin/env bash
#SBATCH --job-name=pc1-support-slides
#SBATCH --partition=THIN
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=logs/jobs/pc1_support_slides_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/dalg-slides-${SLURM_JOB_ID:?}"
srun .venv/bin/python -u scripts/temporary/build_pc1_support_slides.py
