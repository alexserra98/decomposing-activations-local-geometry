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
#SBATCH --job-name=mfa_assign_only
#SBATCH --array=5
#SBATCH --output=/u/dssc/zenocosini/decomposing-activations-local-geometry/outputs/experiments/mfa_assign_%A_%a.out

# SLURM stages this script into /var/spool/slurm/ before running it, so
# BASH_SOURCE / dirname tricks resolve to the wrong place. Hardcode the
# repo root instead.
REPO_ROOT=/u/dssc/zenocosini/decomposing-activations-local-geometry
K=8000
LAYER=$SLURM_ARRAY_TASK_ID
LAYER_TAG=$(printf '%02d' "$LAYER")

MFA_DIR=/orfeo/scratch/dssc/zenocosini/dalg-cache/pile_gemma2b_models/layer${LAYER_TAG}_${K}_mfa
SHARD_DIR=/orfeo/scratch/dssc/zenocosini/dalg-cache/pile_gemma2b_activations
PROFILE_DIR="$MFA_DIR/assignment_profile"

mkdir -p "$PROFILE_DIR"
cd "$REPO_ROOT" || exit 1
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# profile_cluster_assignments.py runs only the streaming argmax + peakedness
# loop. Use this to check whether slowdown is coming from assignments or from
# the later intrinsic-dim sampling / PCA work.
#
# --num-rows / --max-batches are set arbitrarily large so the run covers the
# full shard dataset (the script caps num-rows at len(meta_index) internally).
uv run python scripts/profile_cluster_assignments.py \
    --model-path "$MFA_DIR/mfa_model.pt" \
    --shard-dir "$SHARD_DIR" \
    --layer "$LAYER" \
    --batch-size 1024 \
    --num-workers 4 \
    --device cuda \
    --num-rows 1000000000 \
    --max-batches 1000000000 \
    --profile-dir "$PROFILE_DIR"
