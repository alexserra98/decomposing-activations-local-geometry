#!/bin/bash
# Resume an interrupted EM row, then complete its existing pipeline.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
"$PWD/.venv/bin/python" -u scripts/temporary/resume_toy_300k_em.py \
    "$1" "${SLURM_ARRAY_TASK_ID:?}"
