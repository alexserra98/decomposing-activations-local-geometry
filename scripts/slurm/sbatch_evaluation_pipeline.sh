#!/bin/bash
# Resources and array indices are supplied by dalg-run-pipeline evaluate.
set -euo pipefail

PLAN=$1
MODE=${2:-run}
REPO_ROOT=${SLURM_SUBMIT_DIR:?}
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

if [[ "$MODE" == "collect" ]]; then
    "$REPO_ROOT/.venv/bin/python" -m dalg.evaluation.saved_runs collect "$PLAN"
else
    "$REPO_ROOT/.venv/bin/python" -m dalg.evaluation.saved_runs run "$PLAN" \
        --index "${SLURM_ARRAY_TASK_ID:?}"
fi
