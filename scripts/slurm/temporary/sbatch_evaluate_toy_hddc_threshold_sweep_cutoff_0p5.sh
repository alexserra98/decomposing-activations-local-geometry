#!/bin/bash
# Re-evaluate existing HDDC threshold-sweep runs at association cutoff 0.5.
# Submit a pilot with: sbatch --array=0 this_script.sh
# Submit the remainder with: sbatch --array=1-9%1 this_script.sh
#SBATCH --partition=H100
#SBATCH --account=LADE
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --job-name=eval-hddc-cutoff-0p5
#SBATCH --output=/orfeo/cephfs/home/dssc/zenocosini/decomposing-activations-local-geometry/logs/experiments/10types_1each_15keach_hddc_shared_b_active_set_threshold_sweep_cutoff_0p5/evaluation_%A_%a.out

set -euo pipefail

REPO_ROOT=/orfeo/cephfs/home/dssc/zenocosini/decomposing-activations-local-geometry
MANIFEST="$REPO_ROOT/outputs/experiments/10types_1each_15keach_hddc_shared_b_active_set_threshold_sweep/cutoff_0p5_evaluation_manifest.jsonl"

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "Submit this launcher as a Slurm array." >&2
  exit 1
fi

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

"$REPO_ROOT/.venv/bin/python" \
  "$REPO_ROOT/scripts/temporary/evaluate_toy_hddc_threshold_sweep_cutoff.py" \
  evaluate \
  --manifest "$MANIFEST" \
  --index "$SLURM_ARRAY_TASK_ID"
