#!/bin/bash
# Resources and array indices are supplied at submission.
set -euo pipefail

MANIFEST=$1
MODE=${2:-run}
REPO_ROOT=${SLURM_SUBMIT_DIR:?}
EVALUATION_ROOT=$(dirname "$MANIFEST")
export PYTHONPATH="$EVALUATION_ROOT/code"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [[ "$MODE" == "collect" ]]; then
    "$REPO_ROOT/.venv/bin/python" "$EVALUATION_ROOT/runner.py" collect "$MANIFEST"
else
    "$REPO_ROOT/.venv/bin/python" "$EVALUATION_ROOT/runner.py" run \
        "$MANIFEST" "${SLURM_ARRAY_TASK_ID:?}"
fi
