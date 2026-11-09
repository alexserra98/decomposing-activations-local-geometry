#!/usr/bin/env bash
#SBATCH --job-name=years-ribbon-compare
#SBATCH --partition=EPYC
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=logs/jobs/years_ribbon_comparison_%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
srun .venv/bin/python -m pytest -q tests/test_years_ribbon_comparison.py
srun .venv/bin/python -u scripts/temporary/compare_years_ribbon_tangents.py \
    --activation-dir /orfeo/cephfs/scratch/dssc/zenocosini/dalg-cache/sae-manifold/sae_manifold_years_neutral_llama31_8b_last \
    --ribbon outputs/experiments/years_layer04_spiral_ribbon/job_1905025/final_ribbon_all_development.npz \
    --output "outputs/experiments/years_layer04_ribbon_comparison/job_${SLURM_JOB_ID:?}"
.venv/bin/python scripts/temporary/report_years_ribbon_comparison.py \
    "outputs/experiments/years_layer04_ribbon_comparison/job_${SLURM_JOB_ID:?}"
