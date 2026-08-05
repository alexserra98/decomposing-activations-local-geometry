#!/bin/bash
# Generic worker for one manifest row. Slurm resources are supplied by the
# pipeline submitter because every array contains only one resource profile.

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 /absolute/path/to/manifest.jsonl" >&2
  exit 2
fi

: "${SLURM_ARRAY_TASK_ID:?this worker must run as a Slurm array}"

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
MANIFEST=$1

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

echo "=== $(date) === job ${SLURM_JOB_ID}.${SLURM_ARRAY_TASK_ID} on $(hostname) ==="
echo "manifest=$MANIFEST index=$SLURM_ARRAY_TASK_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

uv run --locked dalg-run-pipeline run \
  --manifest "$MANIFEST" \
  --index "$SLURM_ARRAY_TASK_ID"

echo "=== $(date) === pipeline done ==="
