#!/bin/bash
#SBATCH --partition=H100
#SBATCH --account=LADE
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=150G
#SBATCH --time=02:00:00
#SBATCH --job-name=mfa_assign
#SBATCH --output=/u/dssc/zenocosini/decomposing-activations-local-geometry/logs/experiments/tmp_%A.out

cd /u/dssc/zenocosini/decomposing-activations-local-geometry

PYTHONPATH=src /u/dssc/zenocosini/decomposing-activations-local-geometry/.venv/bin/python \
    scripts/synthetic_mfa_qk_sweep.py generate-dataset \
    --dataset-path "/orfeo/scratch/dssc/zenocosini/dalg-cache/assets/synthetic_mfa_Ktrue250_qtrue20_D500_seed0.pt"\
    --D "500"\
    --K-true "250"\
    --q-true "20"\
    --n-train "500000"\
    --n-test "10000"\
    --seed "0"

echo "Dataset generated"