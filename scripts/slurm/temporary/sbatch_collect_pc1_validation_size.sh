#!/usr/bin/env bash
#SBATCH --job-name=pc1-valsize-report
#SBATCH --partition=THIN
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/jobs/pc1_valsize_report_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export PYTHONPATH="$PWD/src"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/pc1-valsize-mpl-${SLURM_JOB_ID:?}"
.venv/bin/python -u scripts/temporary/collect_pc1_validation_size.py
