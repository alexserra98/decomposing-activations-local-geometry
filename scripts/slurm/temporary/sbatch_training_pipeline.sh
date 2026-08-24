#!/bin/bash
# Generic worker for one manifest row. Slurm resources are supplied by the
# pipeline submitter because every array contains only one resource profile.

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 /absolute/path/to/manifest.jsonl" >&2
  exit 2
fi

: "${SLURM_ARRAY_TASK_ID:?this worker must run as a Slurm array}"

MANIFEST=$1
REPO_ROOT=${SLURM_SUBMIT_DIR:?Slurm did not record the submission directory}
if [[ ! -f "$REPO_ROOT/pyproject.toml" ]]; then
  echo "submission directory is not the DALG repository: $REPO_ROOT" >&2
  exit 2
fi

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

echo "=== $(date) === job ${SLURM_JOB_ID}.${SLURM_ARRAY_TASK_ID} on $(hostname) ==="
echo "manifest=$MANIFEST index=$SLURM_ARRAY_TASK_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

"$REPO_ROOT/.venv/bin/python" -m dalg.cli.run_pipeline run \
  --manifest "$MANIFEST" \
  --index "$SLURM_ARRAY_TASK_ID"

echo "=== $(date) === pipeline done ==="
