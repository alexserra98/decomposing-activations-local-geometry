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
#SBATCH --array=0-3
#SBATCH --output=/u/dssc/zenocosini/decomposing-activations-local-geometry/logs/experiments/mfa_assign_%A_%a.out

# SLURM stages this script into /var/spool/slurm/, so don't rely on
# BASH_SOURCE — hardcode the repo root.
REPO_ROOT=/u/dssc/zenocosini/decomposing-activations-local-geometry

# (K, layer) sweep — one entry per array task.
CONFIGS=(
#    "1000:5"
#    "1000:17"
    "8000:5"
    "8000:17"
    "32000:5"
    "32000:17"
)
IFS=":" read -r K LAYER <<< "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
LAYER_TAG=$(printf '%02d' "$LAYER")

MFA_DIR=/orfeo/scratch/dssc/zenocosini/pile_gemma2b_activations/layer${LAYER_TAG}_${K}_mfa
SHARD_DIR=/orfeo/scratch/dssc/zenocosini/pile_gemma2b_activations

cd "$REPO_ROOT" || exit 1
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

echo "Cluster assignments: K=$K  layer=$LAYER"
echo "MFA dir: $MFA_DIR"

# Writes <MFA_DIR>/mfa_model_assignments.pt by default.
uv run python -m dalg.analysis.cluster_assignments \
    --model-path "$MFA_DIR/mfa_model.pt" \
    --shard-dir "$SHARD_DIR" \
    --layer "$LAYER" \
    --batch-size 1024 \
    --device cuda
