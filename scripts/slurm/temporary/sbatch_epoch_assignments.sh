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
#SBATCH --job-name=mfa_epoch_assign
#SBATCH --array=0-4
#SBATCH --output=/u/dssc/zenocosini/decomposing-activations-local-geometry/logs/experiments/mfa_epoch_assign_%A_%a.out

# One array task per epoch snapshot: stream the full dataset through that
# snapshot's MFA and save hard cluster assignments via dalg.analysis.cluster_assignments.
# shuffle is off, so every epoch sees the same token order and the assignment
# vectors are index-aligned for the cross-epoch comparison.

REPO_ROOT=/u/dssc/zenocosini/decomposing-activations-local-geometry
RUN_DIR=/orfeo/scratch/dssc/zenocosini/dalg-cache/pile_gemma2b_models/layer05_1000_10_component_sharded_mfa
SHARD_DIR=/orfeo/scratch/dssc/zenocosini/dalg-cache/pile_gemma2b_activations
LAYER=5

EPOCHS=(epoch_0001 epoch_0005 epoch_0010 epoch_0015 62_epoch)
EPNUM=(1 5 10 15 62)
EPOCH_DIR=${EPOCHS[$SLURM_ARRAY_TASK_ID]}
N=${EPNUM[$SLURM_ARRAY_TASK_ID]}

DATA_DIR=$RUN_DIR/$EPOCH_DIR
EPOCH_OUT=$DATA_DIR

mkdir -p "$EPOCH_OUT"
cd "$REPO_ROOT" || exit 1
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

echo "epoch=$N  data_dir=$DATA_DIR  out=$EPOCH_OUT"

# load_mfa falls back to mfa_model_shards.json when mfa_model.pt is absent.
# --save-path keeps each epoch's assignments in its own output subdir alongside
# gaussian_overlap.pt.
uv run python -m dalg.analysis.cluster_assignments \
    --model-path "$DATA_DIR/mfa_model.pt" \
    --shard-dir "$SHARD_DIR" \
    --layer "$LAYER" \
    --batch-size 1024 \
    --device cuda \
    --save-path "$EPOCH_OUT/mfa_model_assignments.pt"
