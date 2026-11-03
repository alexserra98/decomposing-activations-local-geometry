#!/usr/bin/env bash
#SBATCH --job-name=temperature04-pc1-density
#SBATCH --partition=THIN
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --output=logs/jobs/temperature04_pc1_density_%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/dalg-temperature04-pc1-${SLURM_JOB_ID:?}"

output="outputs/experiments/temperature_layer04_pc1_density_pca75/job_${SLURM_JOB_ID:?}"
resume_args=()
if [[ $# -gt 0 ]]; then
    resume_args=(--resume-from "$1")
fi
srun .venv/bin/python -m pytest -q tests/test_pc1_density.py tests/test_temperature04_pc1_context.py
srun .venv/bin/python -u scripts/temporary/run_temperature04_pc1_density.py \
    --output "$output" --threads "$OMP_NUM_THREADS" --workers 6 "${resume_args[@]}"
