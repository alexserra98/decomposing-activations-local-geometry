#!/bin/bash
#SBATCH --partition=H100
#SBATCH --account=LADE
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --job-name=toy10_hddc_rank_eval
#SBATCH --output=/u/dssc/zenocosini/decomposing-activations-local-geometry/logs/jobs/toy10_hddc_rank_eval_%j.out

set -euo pipefail
cd /u/dssc/zenocosini/decomposing-activations-local-geometry
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1
.venv/bin/python scripts/temporary/reevaluate_toy_noise10_hddc_learned_rank.py
