#!/bin/bash
#SBATCH --partition=DGX
#SBATCH --account=LADE
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:A100:1
#SBATCH --mem=160G
#SBATCH --time=15:00:00
#SBATCH --job-name=mfa_id_cluster
##SBATCH --array=5,17
#SBATCH --output=outputs/jobs/mfa_id_cluster_%A_%a.out

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
K=8000

mkdir -p "$REPO_ROOT/outputs/jobs" "$REPO_ROOT/outputs/experiments/8000_05"
cd "$REPO_ROOT" || exit 1
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

uv run dalg-run-layer intrinsic-dim \
                                --data-dir /orfeo/scratch/dssc/zenocosini/pile_gemma2b_activations/layer05_${K}_mfa/\
                                --shard-dir /orfeo/scratch/dssc/zenocosini/pile_gemma2b_activations\
                                --layer 5\
                                --out-dir "outputs/experiments/${K}_05"\
                                --device cuda\
                                --num-workers 4\
                                --max-samples-per-cluster 2000

uv run dalg-run-layer overlap \
                            --data-dir /orfeo/scratch/dssc/zenocosini/pile_gemma2b_activations/layer05_${K}_mfa/ \
                            --out-dir outputs/experiments/${K}_05

         
