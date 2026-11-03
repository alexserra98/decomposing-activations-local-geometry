#!/usr/bin/env bash
# Resources, array indices, and dependencies are supplied by sbatch.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONUNBUFFERED=1
if [[ "$1" == assign ]]; then
    .venv/bin/python scripts/temporary/evaluate_sae_saved_runs.py assign --index "${SLURM_ARRAY_TASK_ID:?}"
else
    .venv/bin/python scripts/temporary/evaluate_sae_saved_runs.py "$@"
fi
