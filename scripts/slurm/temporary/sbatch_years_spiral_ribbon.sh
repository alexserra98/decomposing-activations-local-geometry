#!/usr/bin/env bash
#SBATCH --job-name=years-spiral-ribbon
#SBATCH --partition=THIN
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --time=01:00:00
#SBATCH --output=logs/jobs/years_spiral_ribbon_%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
srun .venv/bin/python -m pytest -q tests/test_years_spiral_ribbon.py
srun .venv/bin/python -u scripts/temporary/fit_years_spiral_ribbon.py \
    --activation-dir /orfeo/cephfs/scratch/dssc/zenocosini/dalg-cache/sae-manifold/sae_manifold_years_neutral_llama31_8b_last \
    --output "outputs/experiments/years_layer04_spiral_ribbon/job_${SLURM_JOB_ID:?}"
srun .venv/bin/python -u scripts/temporary/audit_years_spiral_ribbon.py \
    "outputs/experiments/years_layer04_spiral_ribbon/job_${SLURM_JOB_ID:?}"
