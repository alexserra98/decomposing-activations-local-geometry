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
#SBATCH --job-name=mfa_assign
#SBATCH --output=/u/dssc/zenocosini/decomposing-activations-local-geometry/logs/experiments/mfa_assign_%j.out

# SLURM stages this script into /var/spool/slurm/, so don't rely on
# BASH_SOURCE — hardcode the repo root.
REPO_ROOT=/u/dssc/zenocosini/decomposing-activations-local-geometry
LAYER=5
LAYER_TAG=$(printf "%02d" "$LAYER")
Q=394
K=1000


# MFA_DIR=/orfeo/scratch/dssc/zenocosini/dalg-cache/pile_gemma2b_models/layer${LAYER_TAG}_${K}_${Q}_component_sharded_mfa
MFA_DIR=dalg-cache/pile_gemma2b_models/layer05_1000_10_mfa_1epoch_20260703_1538
SHARD_DIR=/orfeo/scratch/dssc/zenocosini/dalg-cache/pile_gemma2b_activations
MODELS_DIR=${MODELS_DIR:-/orfeo/scratch/dssc/zenocosini/dalg-cache/pile_gemma2b_models}
CENTROIDS_DIR=${CENTROIDS_DIR:-"$MODELS_DIR/centroids"}

cd "$REPO_ROOT" || exit 1
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

echo "Cluster assignments: K=$K  layer=$LAYER  layer_tag=layer$LAYER_TAG"
echo "MFA dir: $MFA_DIR"

# ---------------------------------------------------------------------------
# MFA responsibility assignments
# Writes <MFA_DIR>/mfa_model_assignments.pt by default.
uv run python -m dalg.analysis.cluster_assignments \
    --model-path "$MFA_DIR/mfa_model.pt" \
    --shard-dir "$SHARD_DIR" \
    --layer "$LAYER" \
    --batch-size 1024 \
    --device cuda

# ---------------------------------------------------------------------------
# KMeans-centroid assignments
# Uncomment this section, and comment the MFA section above, to assign each
# activation to its nearest saved KMeans centroid instead of the trained MFA.
#
# SHARD_DIR=/orfeo/scratch/dssc/zenocosini/dalg-cache/pile_gemma2b_activations
# K=1000
# LAYER=5
# LAYER=$(printf "%02d" "$LAYER")
# KMEANS_CENTROIDS_PATH="$CENTROIDS_DIR/k${K}_L${LAYER}/centroids.pt"
# KMEANS_ASSIGNMENTS_PATH="$CENTROIDS_DIR/k${K}_L${LAYER}/kmeans_centroid_assignments.pt"

# echo "KMeans centroid path: $KMEANS_CENTROIDS_PATH"
# echo "KMeans assignments output: $KMEANS_ASSIGNMENTS_PATH"

# uv run python -m dalg.analysis.nearest_centroid_assignments \
#    --centroids-path "$KMEANS_CENTROIDS_PATH" \
#    --shard-dir "$SHARD_DIR" \
#    --layer "$LAYER" \
#    --batch-size 8192 \
#    --device cuda \
#    --save-path "$KMEANS_ASSIGNMENTS_PATH"
