#!/bin/bash
#SBATCH --partition=H100
#SBATCH --account=LADE
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=640G
#SBATCH --time=12:00:00
#SBATCH --job-name=arch_mfa_vis
#SBATCH --output=/u/dssc/zenocosini/decomposing-activations-local-geometry/logs/experiments/arch_mfa_vis_%j.out

set -euo pipefail

REPO_ROOT=/u/dssc/zenocosini/decomposing-activations-local-geometry
OUT_ROOT="$REPO_ROOT/output/experiments"
VIS_MAX_COMPONENTS=${VIS_MAX_COMPONENTS:-4000}

EXPERIMENTS=(
  "1000_05"
  "8000_05"
  "32000_05"
  "1000_17"
  "8000_17"
  "32000_17"
)

cd "$REPO_ROOT" || exit 1
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg

for experiment_name in "${EXPERIMENTS[@]}"; do
  out_dir="$OUT_ROOT/$experiment_name"
  echo "Visualizing $experiment_name from $out_dir"
  test -f "$out_dir/intrinsic_dims.pt"
  test -f "$out_dir/overlap.pt"

  DALG_VIS_EXPERIMENT_NAME="$experiment_name" \
  DALG_VIS_OUT="$out_dir" \
  DALG_VIS_MAX_COMPONENTS="$VIS_MAX_COMPONENTS" \
    uv run jupyter nbconvert \
      --to notebook \
      --execute notebooks/visualize.ipynb \
      --output "$out_dir/visualize_executed.ipynb" \
      --ExecutePreprocessor.timeout=-1
done
