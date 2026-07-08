#!/bin/bash
#SBATCH --partition=DGX
#SBATCH --account=LADE
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=dgx002
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:A100:1
#SBATCH --mem=160G
#SBATCH --time=1-02:00:00
#SBATCH --job-name=mfa_mu_assign
#SBATCH --output=/u/dssc/zenocosini/decomposing-activations-local-geometry/logs/experiments/mfa_mu_assign_%j.out

# Nearest-mu assignments: assign each activation token to the nearest trained
# MFA mean (Euclidean), as an alternative to the responsibility-argmax
# assignments already stored in <run_dir>/mfa_model_assignments.pt.
# Writes <run_dir>/mfa_mu.pt and <run_dir>/mfa_model_nearest_centroid_assignments.pt.

# SLURM stages this script into /var/spool/slurm/, so don't rely on
# BASH_SOURCE — hardcode the repo root.
REPO_ROOT=/u/dssc/zenocosini/decomposing-activations-local-geometry
SHARD_DIR=/orfeo/scratch/dssc/zenocosini/dalg-cache/pile_gemma2b_activations

cd "$REPO_ROOT" || exit 1
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

RUNS=(
  "layer05_1000_10_component_sharded_mfa:5"
  "layer05_1000_10_mfa_1epoch_20260703_1538:5"
  "layer05_1000_394_component_sharded_mfa:5"
  "layer17_1000_10_component_sharded_mfa:17"
  "layer17_1000_337_component_sharded_mfa:17"
)

for entry in "${RUNS[@]}"; do
  run_dir="$SHARD_DIR/${entry%%:*}"
  layer="${entry##*:}"
  save_path="$run_dir/mfa_model_nearest_centroid_assignments.pt"

  echo "=== $run_dir (layer $layer) ==="
  if [ -f "$save_path" ]; then
    echo "Output already exists, skipping: $save_path"
    continue
  fi

  uv run python scripts/extract_mfa_mu.py --run-dir "$run_dir" || exit 1

  uv run python -m dalg.analysis.nearest_centroid_assignments \
    --centroids-path "$run_dir/mfa_mu.pt" \
    --shard-dir "$SHARD_DIR" \
    --layer "$layer" \
    --batch-size 8192 \
    --device cuda \
    --save-path "$save_path" || exit 1
done
