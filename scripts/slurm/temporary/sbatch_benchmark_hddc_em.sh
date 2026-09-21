#!/bin/bash
#SBATCH --partition=H100
#SBATCH --account=LADE
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --job-name=hddc_em_benchmark
#SBATCH --output=logs/jobs/hddc_em_benchmark_%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
.venv/bin/python scripts/temporary/benchmark_hddc_em.py \
    --out-path "outputs/experiments/hddc_em_benchmark_${SLURM_JOB_ID}.json" "$@"
