#!/bin/bash
#SBATCH --partition=H100
#SBATCH --account=LADE
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --job-name=toy_noise_kmeans_k_sweep
#SBATCH --output=/u/dssc/zenocosini/decomposing-activations-local-geometry/logs/jobs/toy_noise_kmeans_k_sweep_%j.out

set -euo pipefail
REPO_ROOT=/u/dssc/zenocosini/decomposing-activations-local-geometry
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1

.venv/bin/python scripts/temporary/run_toy_noise_kmeans_k_sweep.py run "$@"
