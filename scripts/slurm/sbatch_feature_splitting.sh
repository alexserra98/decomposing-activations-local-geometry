#!/bin/bash
# Feature-splitting & covariance-reconstruction analysis over a trained K/q sweep.
#
# Runs scripts/synthetic_mfa_feature_splitting.py over every fitted MFA already
# on disk and writes feature_splitting.csv / feature_splitting.pt into the run
# directory (consumed by notebooks/synthetic_mfa_qk_sweep_results.ipynb).
#
#   sbatch scripts/slurm/sbatch_feature_splitting.sh
#   K_MIN=750 sbatch scripts/slurm/sbatch_feature_splitting.sh
#   ONLY="1500:20,2000:20" sbatch scripts/slurm/sbatch_feature_splitting.sh

#SBATCH --partition=H100
#SBATCH --account=LADE
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=80G
#SBATCH --time=06:00:00
#SBATCH --job-name=feat_split
#SBATCH --output=/u/dssc/zenocosini/decomposing-activations-local-geometry/logs/jobs/feat_split_%j.out

set -euo pipefail

REPO_ROOT=/u/dssc/zenocosini/decomposing-activations-local-geometry
cd "$REPO_ROOT"

mkdir -p "$REPO_ROOT/logs/jobs"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$SLURM_CPUS_PER_TASK}"

RUN_DIR=${RUN_DIR:-"$REPO_ROOT/dalg-cache/qk_sweep_exploration/Ktrue1000_qtrue20"}
K_MIN=${K_MIN:-750}
DEVICE=${DEVICE:-cuda}
RESP_BATCH=${RESP_BATCH:-512}
ONLY=${ONLY:-}

CMD=(python scripts/synthetic_mfa_feature_splitting.py
     --run-dir "$RUN_DIR" --k-min "$K_MIN" --device "$DEVICE" --resp-batch "$RESP_BATCH")
if [[ -n "$ONLY" ]]; then
    CMD+=(--only "$ONLY")
fi

echo "Running: ${CMD[*]}"
srun "${CMD[@]}"
