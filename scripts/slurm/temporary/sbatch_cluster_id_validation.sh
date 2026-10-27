#!/usr/bin/env bash
# Validation only. Supply reviewed Slurm resources on the sbatch command line.
set -euo pipefail

: "${SLURM_JOB_ID:?Run validation through Slurm}"
REPO_ROOT="${SLURM_SUBMIT_DIR:?Submit from the experiment checkout}"
ID_EXP_PYTHON="${ID_EXP_PYTHON:-$REPO_ROOT/.venv/bin/python}"
ID_EXP_VALIDATION_DIR="${1:?Provide a new validation output directory}"
shift
ID_EXP_CACHE_ROOT="/orfeo/scratch/dssc/zenocosini/dalg-cache"
ID_EXP_RESULTS_ROOT="$ID_EXP_CACHE_ROOT/toy_manifolds_10types_1each_D128_30Keach_noise_sweep_seed0"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MAX_JOBS="$OMP_NUM_THREADS"

hostname
printf 'Validation Slurm job: %s\n' "$SLURM_JOB_ID"
git rev-parse HEAD
if [[ -e "$ID_EXP_VALIDATION_DIR" ]]; then
    printf 'Refusing existing validation directory: %s\n' "$ID_EXP_VALIDATION_DIR" >&2
    exit 2
fi
mkdir -p "$ID_EXP_VALIDATION_DIR"
export XDG_CACHE_HOME="$ID_EXP_VALIDATION_DIR/cache"
export PIP_CACHE_DIR="$XDG_CACHE_HOME/pip"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
"$ID_EXP_PYTHON" -m pip freeze > "$ID_EXP_VALIDATION_DIR/environment.before.txt"

# Installation must be explicitly enabled in the reviewed submission.
if [[ "${ID_INSTALL_DADAPY:-0}" == 1 ]]; then
    "$ID_EXP_PYTHON" -m pip install --disable-pip-version-check \
        'dadapy @ git+https://github.com/sissa-data-science/DADApy.git@c3f7300c9f87fb9e19241f313ae0e023bca687b5'
fi

"$ID_EXP_PYTHON" -m pip freeze > "$ID_EXP_VALIDATION_DIR/environment.txt"
bash -n scripts/slurm/temporary/sbatch_cluster_id_experiment.sh
bash -n scripts/slurm/temporary/sbatch_cluster_id_validation.sh
"$ID_EXP_PYTHON" -m pytest tests/test_cluster_id_experiment.py tests/test_cluster_id_plotting.py -q \
    --junitxml="$ID_EXP_VALIDATION_DIR/tests.xml"
"$ID_EXP_PYTHON" scripts/temporary/run_cluster_id_experiment.py plan \
    --results-root "$ID_EXP_RESULTS_ROOT" \
    --cache-root "$ID_EXP_CACHE_ROOT" \
    --output "$ID_EXP_VALIDATION_DIR/manifest.json" "$@"
