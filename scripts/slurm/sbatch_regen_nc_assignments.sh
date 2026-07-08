#!/bin/bash
#SBATCH --partition=DGX
#SBATCH --account=LADE
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=dgx002
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:A100:1
#SBATCH --mem=160G
#SBATCH --time=05:00:00
#SBATCH --job-name=regen_nc_assign
#SBATCH --output=/u/dssc/zenocosini/decomposing-activations-local-geometry/logs/experiments/regen_nc_assign_%j.out

# One-off repair job: regenerate nearest-centroid / kmeans assignment files
# that were saved in worker-interleaved order (computed with --num-workers 2
# before nearest_centroid_assignments.py hardcoded num_workers=0), then
# recompute the intrinsic dims derived from them.
# The scrambled originals are kept next to each output as *.pt.interleaved;
# an existing output file means that step already ran and is skipped.

REPO_ROOT=/u/dssc/zenocosini/decomposing-activations-local-geometry
SHARD_DIR=/orfeo/scratch/dssc/zenocosini/dalg-cache/pile_gemma2b_activations

cd "$REPO_ROOT" || exit 1
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# entry format: centroids_path:layer:save_path
JOBS=(
  "$SHARD_DIR/layer05_1000_10_component_sharded_mfa/mfa_mu.pt:5:$SHARD_DIR/layer05_1000_10_component_sharded_mfa/mfa_model_nearest_centroid_assignments.pt"
  "$SHARD_DIR/layer05_1000_394_component_sharded_mfa/mfa_mu.pt:5:$SHARD_DIR/layer05_1000_394_component_sharded_mfa/mfa_model_nearest_centroid_assignments.pt"
  "$SHARD_DIR/layer17_1000_337_component_sharded_mfa/mfa_mu.pt:17:$SHARD_DIR/layer17_1000_337_component_sharded_mfa/mfa_model_nearest_centroid_assignments.pt"
  "$SHARD_DIR/centroids/k1000_L05/centroids.pt:5:$SHARD_DIR/centroids/k1000_L05/kmeans_centroid_assignments.pt"
  "$SHARD_DIR/centroids/k1000_L17/centroids.pt:17:$SHARD_DIR/centroids/k1000_L17/kmeans_centroid_assignments.pt"
)

for entry in "${JOBS[@]}"; do
  IFS=":" read -r centroids layer save_path <<< "$entry"
  echo "=== REGEN $save_path (layer $layer) ==="
  # A .pt with its .interleaved backup present means this entry was already
  # regenerated; a .pt without backup is the scrambled original to replace.
  if [ -f "$save_path" ] && [ -f "$save_path.interleaved" ]; then
    echo "already regenerated, skipping"
    continue
  fi
  if [ -f "$save_path" ]; then
    mv "$save_path" "$save_path.interleaved"
    echo "backed up scrambled file to $save_path.interleaved"
  fi
  uv run python -m dalg.analysis.nearest_centroid_assignments \
    --centroids-path "$centroids" \
    --shard-dir "$SHARD_DIR" \
    --layer "$layer" \
    --batch-size 8192 \
    --device cuda \
    --save-path "$save_path" || exit 1
  echo "=== OK $save_path ==="
done

# ---------------------------------------------------------------------------
# Intrinsic dims derived from the regenerated partitions
# (same settings as the responsibility-based runs: 90% variance, min pop 100,
# 2000 samples per cluster).

ID_COMMON=(--shard-dir "$SHARD_DIR" --device cuda --pca-device cuda
           --variance-threshold 0.90 --min-population 100
           --max-samples-per-cluster 2000)

echo "=== ID nn layer05_1000_10 ==="
uv run python -m dalg.cli.run_metrics intrinsic-dim \
  --data-dir "$SHARD_DIR/layer05_1000_10_component_sharded_mfa" \
  --assignments-path "$SHARD_DIR/layer05_1000_10_component_sharded_mfa/mfa_model_nearest_centroid_assignments.pt" \
  --layer 5 \
  --out-dir "$REPO_ROOT/output/experiments/1000_05_10_nn" \
  "${ID_COMMON[@]}" || exit 1

echo "=== ID kmeans k1000_L05 ==="
uv run python -m dalg.cli.run_metrics intrinsic-dim \
  --assignments-path "$SHARD_DIR/centroids/k1000_L05/kmeans_centroid_assignments.pt" \
  --layer 5 \
  --out-dir "$REPO_ROOT/output/experiments/centroids_1000_05" \
  "${ID_COMMON[@]}" || exit 1

echo "=== ID kmeans k1000_L17 ==="
uv run python -m dalg.cli.run_metrics intrinsic-dim \
  --assignments-path "$SHARD_DIR/centroids/k1000_L17/kmeans_centroid_assignments.pt" \
  --layer 17 \
  --out-dir "$REPO_ROOT/output/experiments/centroids_1000_17" \
  "${ID_COMMON[@]}" || exit 1

echo "ALL REGENERATIONS AND IDS DONE"
