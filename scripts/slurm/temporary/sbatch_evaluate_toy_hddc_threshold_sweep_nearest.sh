#!/bin/bash
# Re-evaluate all existing HDDC threshold-sweep runs without a distance cutoff.
# Submit with: sbatch --array=0-11%12 this_script.sh
#SBATCH --partition=H100
#SBATCH --account=LADE
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --job-name=eval-hddc-nearest
#SBATCH --output=/u/dssc/zenocosini/decomposing-activations-local-geometry/logs/experiments/10types_1each_15keach_hddc_shared_b_active_set_threshold_sweep_nearest/evaluation_%A_%a.out

set -euo pipefail

REPO_ROOT=/u/dssc/zenocosini/decomposing-activations-local-geometry
MANIFEST="$REPO_ROOT/outputs/experiments/10types_1each_15keach_hddc_shared_b_active_set_threshold_sweep/nearest_evaluation_manifest.jsonl"

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "Submit this launcher as a Slurm array." >&2
  exit 1
fi

mkdir -p "$REPO_ROOT/logs/experiments/10types_1each_15keach_hddc_shared_b_active_set_threshold_sweep_nearest"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

"$REPO_ROOT/.venv/bin/python" \
  "$REPO_ROOT/scripts/temporary/evaluate_toy_hddc_threshold_sweep_nearest.py" \
  evaluate \
  --manifest "$MANIFEST" \
  --index "$SLURM_ARRAY_TASK_ID"
