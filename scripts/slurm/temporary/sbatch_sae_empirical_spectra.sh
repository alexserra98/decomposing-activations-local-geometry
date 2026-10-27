#!/usr/bin/env bash
#SBATCH --job-name=sae-empirical-spectra
#SBATCH --partition=H100
#SBATCH --account=LADE
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --array=0-7%2
#SBATCH --output=logs/jobs/sae_empirical_spectra_%A_%a.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"

.venv/bin/python -u scripts/temporary/run_sae_empirical_spectra.py \
    --task-index "${SLURM_ARRAY_TASK_ID:?}" --device cuda \
    --batch-size 256 --eig-batch-size 1
