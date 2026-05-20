#!/bin/bash
#SBATCH --partition=H100
#SBATCH --account=LADE
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
##SBATCH --nodelist=dgx002
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=640G
#SBATCH --time=1-02:00:00
#SBATCH --job-name=mfa_metric_cluster
##SBATCH --begin=now+4hours
#SBATCH --array=5
#SBATCH --output=/u/dssc/zenocosini/decomposing-activations-local-geometry/logs/jobs/mfa_metric_cluster_%A_%a.out

REPO_ROOT=/u/dssc/zenocosini/decomposing-activations-local-geometry
K=1000
RANK=337
LAYER=$SLURM_ARRAY_TASK_ID

SHARD_DIR=/orfeo/scratch/dssc/zenocosini/dalg-cache/pile_gemma2b_activations
LAYER_TAG="layer$(printf '%02d' "$LAYER")"
DATA_DIR="$SHARD_DIR/${LAYER_TAG}_${K}_${RANK}_component_sharded_mfa"
OUT_DIR="$REPO_ROOT/output/experiments/${K}_$(printf '%02d' "$LAYER")_${RANK}"

mkdir -p "$OUT_DIR"
cd "$REPO_ROOT" || exit 1
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# Step 1: compute cluster assignments (required by intrinsic-dim)
# load_mfa falls back to mfa_model_shards.json when mfa_model.pt is absent
# uv run python -m dalg.analysis.cluster_assignments \
#   --model-path "$DATA_DIR/mfa_model.pt" \
#   --shard-dir "$SHARD_DIR" \
#   --layer "$LAYER" \
#   --device cuda --num-workers 4

# # Step 2: intrinsic dimensionality per cluster
# uv run dalg-run-layer intrinsic-dim \
#   --data-dir "$DATA_DIR" \
#   --shard-dir "$SHARD_DIR" \
#   --layer "$LAYER" \
#   --out-dir "$OUT_DIR" \
#   --device cuda --max-samples-per-cluster 2000

# Step 3: pairwise component overlap
# batch_pairs=512: peak GPU ≈ 10 GB (7 W-type tensors × 512 × D × q × 4 bytes)
# default 4096 OOMs because W-chunk (4096, 2048, 337) alone exceeds H100 memory
uv run dalg-run-layer overlap \
  --data-dir "$DATA_DIR" \
  --out-dir "$OUT_DIR" \
  --device cuda --batch-pairs 512
