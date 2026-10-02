#!/bin/bash
# Resources are supplied by the manifest-based sbatch command.
set -euo pipefail
REPO_ROOT=${SLURM_SUBMIT_DIR:?}
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"$REPO_ROOT/.venv/bin/python" scripts/temporary/evaluate_saved_toy_runs.py \
    "$1" "${SLURM_ARRAY_TASK_ID:?}"
