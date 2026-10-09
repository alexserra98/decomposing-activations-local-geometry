#!/usr/bin/env bash
# Supply the reviewed Slurm resource requests on the sbatch command line.
set -euo pipefail

: "${SLURM_JOB_ID:?Run this launcher through Slurm}"
REPO_ROOT="${SLURM_SUBMIT_DIR:?Submit from the experiment checkout}"
ID_EXP_PYTHON="${ID_EXP_PYTHON:-$REPO_ROOT/.venv/bin/python}"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export XDG_CACHE_HOME="${SLURM_TMPDIR:-/tmp}/dalg-id-${SLURM_JOB_ID}/cache"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
mkdir -p "$XDG_CACHE_HOME"

hostname
printf 'Slurm job: %s\n' "$SLURM_JOB_ID"
git rev-parse HEAD

if [[ "${1:-}" == run ]]; then
    shift
    exec "$ID_EXP_PYTHON" scripts/temporary/run_cluster_id_experiment.py run \
        --index "${SLURM_ARRAY_TASK_ID:?Provide a run-index job array}" "$@"
fi
exec "$ID_EXP_PYTHON" scripts/temporary/run_cluster_id_experiment.py "$@"
