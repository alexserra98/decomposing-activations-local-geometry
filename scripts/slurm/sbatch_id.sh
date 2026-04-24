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

uv run dalg-run-layer intrinsic-dim \
                                --data-dir /orfeo/scratch/dssc/zenocosini/pile_gemma2b_activations/layer05_8000mfa/\
                                --shard-dir /orfeo/scratch/dssc/zenocosini/pile_gemma2b_activations\
                                --layer 5\
                                --out-dir outputs/experiments/8000_05\
                                --device cuda\
                                --num-workers 4\
                                --max-samples-per-cluster 2000
