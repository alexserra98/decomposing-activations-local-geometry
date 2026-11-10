#!/usr/bin/env bash
#SBATCH --job-name=pc1-pak-tables
#SBATCH --partition=THIN
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=logs/jobs/pc1_pak_tables_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/dalg-pak-tables-${SLURM_JOB_ID:?}"
srun .venv/bin/python -u scripts/temporary/run_pc1_pak_controlled.py --collect-only --output "${1:?Result root required}"
