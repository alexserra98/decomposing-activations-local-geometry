#!/usr/bin/env bash
#SBATCH --job-name=years-pc1-direction-swap
#SBATCH --partition=THIN
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/jobs/years_pc1_direction_swap_%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/dalg-pc1-swap-matplotlib-${SLURM_JOB_ID:?}"

output="outputs/experiments/years_layer04_pc1_direction_swap/job_${SLURM_JOB_ID:?}"
test_log=$(mktemp "${TMPDIR:-/tmp}/dalg-pc1-direction-swap-tests.XXXXXX")
.venv/bin/python -m pytest -q tests/test_pc1_density.py tests/test_pc1_direction_swap.py | tee "$test_log"
.venv/bin/python -u scripts/temporary/run_years_pc1_direction_swap.py \
    --output "$output" --threads "$OMP_NUM_THREADS"
cp "$test_log" "$output/tests.txt"
