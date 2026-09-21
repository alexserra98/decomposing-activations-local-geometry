#!/bin/bash
#SBATCH --partition=EPYC
#SBATCH --account=LADE
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --job-name=toy10_noise_sweep
#SBATCH --output=/u/dssc/zenocosini/decomposing-activations-local-geometry/logs/jobs/toy10_noise_sweep_%j.out

set -euo pipefail

REPO_ROOT=/u/dssc/zenocosini/decomposing-activations-local-geometry
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

.venv/bin/python scripts/temporary/build_toy_noise_sweep.py \
  --output-root "$REPO_ROOT/dalg-cache/assets/toy_manifolds_10types_1each_D128_30Keach_noise_sweep_seed0" \
  --points-per-type 30000

.venv/bin/python scripts/temporary/validate_toy_noise_sweep.py \
  "$REPO_ROOT/dalg-cache/assets/toy_manifolds_10types_1each_D128_30Keach_noise_sweep_seed0"
