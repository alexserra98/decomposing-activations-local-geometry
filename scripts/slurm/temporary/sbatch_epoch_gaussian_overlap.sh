#!/bin/bash
#SBATCH --partition=H100
#SBATCH --account=LADE
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=160G
#SBATCH --time=06:00:00
#SBATCH --job-name=mfa_epoch_gaussian_overlap
#SBATCH --array=0-4
#SBATCH --output=/u/dssc/zenocosini/decomposing-activations-local-geometry/outputs/experiments/mfa_epoch_gaussian_overlap_%A_%a.out

# One array task per epoch snapshot: compute the KxK pairwise Gaussian overlap
# (kl_sym, db, ...) for that snapshot's MFA and save gaussian_overlap.pt. This
# is the per-epoch "distance matrix" the analysis step compares across epochs.

REPO_ROOT=/u/dssc/zenocosini/decomposing-activations-local-geometry
RUN_DIR=/orfeo/scratch/dssc/zenocosini/dalg-cache/pile_gemma2b_models/layer05_1000_10_component_sharded_mfa

EPOCHS=(epoch_0001 epoch_0005 epoch_0010 epoch_0015 62_epoch)
EPNUM=(1 5 10 15 62)
EPOCH_DIR=${EPOCHS[$SLURM_ARRAY_TASK_ID]}
N=${EPNUM[$SLURM_ARRAY_TASK_ID]}

DATA_DIR=$RUN_DIR/$EPOCH_DIR
EPOCH_OUT=$DATA_DIR

mkdir -p "$EPOCH_OUT"
cd "$REPO_ROOT" || exit 1
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

echo "epoch=$N  data_dir=$DATA_DIR  out_dir=$EPOCH_OUT"

# K=1000, q=10 -> Gaussian-overlap tensors are small; batch-pairs=512 is safe.
# load_mfa falls back to mfa_model_shards.json when mfa_model.pt is absent.
uv run dalg-run-metrics gaussian-overlap \
    --data-dir "$DATA_DIR" \
    --out-dir "$EPOCH_OUT" \
    --device cuda --batch-pairs 512
