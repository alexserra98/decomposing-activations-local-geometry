#!/bin/bash
#SBATCH --partition=DGX
#SBATCH --account=LADE
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=dgx002
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:A100:1
#SBATCH --mem=640G
#SBATCH --time=1-02:00:00
#SBATCH --job-name=mfa_id_cluster
##SBATCH --begin=now+4hours
#SBATCH --array=17
#SBATCH --output=/u/dssc/zenocosini/decomposing-activations-local-geometry/outputs/experiments/mfa_metric_cluster_%A_%a.out

REPO_ROOT=/u/dssc/zenocosini/decomposing-activations-local-geometry
K=32000
LAYER=$SLURM_ARRAY_TASK_ID

mkdir -p "/u/dssc/zenocosini/decomposing-activations-local-geometry/outputs/experiments/${K}_$(printf '%02d' "$LAYER")"
cd "$REPO_ROOT" || exit 1
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# uv run dalg-run-layer intrinsic-dim \
#                                 --data-dir /orfeo/scratch/dssc/zenocosini/pile_gemma2b_activations/layer$(printf '%02d' "$LAYER")_${K}_mfa/\
#                                 --shard-dir /orfeo/scratch/dssc/zenocosini/pile_gemma2b_activations\
#                                 --layer $LAYER\
#                                 --out-dir "/u/dssc/zenocosini/decomposing-activations-local-geometry/outputs/experiments/${K}_$(printf '%02d' "$LAYER")"\
#                                 --device cuda\
#                                 --max-samples-per-cluster 2000

uv run dalg-run-layer overlap \
                            --data-dir /orfeo/scratch/dssc/zenocosini/pile_gemma2b_activations/layer$(printf '%02d' "$LAYER")_${K}_mfa/ \
                            --out-dir /u/dssc/zenocosini/decomposing-activations-local-geometry/outputs/experiments/${K}_$(printf '%02d' "$LAYER")

         
