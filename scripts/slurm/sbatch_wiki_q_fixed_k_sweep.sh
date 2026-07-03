#!/bin/bash
# ── Held-out NLL vs K sweep at fixed q, pile_wikipedia ──
# Trains a vanilla MFA for each K in the grid (fresh KMeans per K on the
# Wikipedia slice). Training only — plotting is a separate dependency job.
#
# Runs as a 2-task job array: the full K grid is split (interleaved, so slow
# large-K runs spread evenly) across the two tasks. Submit via
# scripts/slurm/submit_wiki_q_fixed_k_sweep.sh, which chains the plot job after the
# array finishes. (Plotting is decoupled so a partial array still yields a clean
# figure over whatever trained.)
#SBATCH --partition=H100
#SBATCH --account=LADE
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=80G
#SBATCH --time=1-12:00:00
#SBATCH --array=0-1
#SBATCH --job-name=wiki_q_fixed_k_sweep
#SBATCH --output=/u/dssc/zenocosini/decomposing-activations-local-geometry/logs/jobs/wiki_q_fixed_k_sweep_%A_%a.out

set -euo pipefail

REPO_ROOT=/u/dssc/zenocosini/decomposing-activations-local-geometry
mkdir -p "$REPO_ROOT/logs/jobs"
cd "$REPO_ROOT" || exit 1
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

: "${SUBSET_TOKENS:?set SUBSET_TOKENS via submit_wiki_q_fixed_k_sweep.sh}"
: "${K_LIST:?set K_LIST via submit_wiki_q_fixed_k_sweep.sh}"
: "${LAYER:?set LAYER via submit_wiki_q_fixed_k_sweep.sh}"
: "${RANK:?set RANK via submit_wiki_q_fixed_k_sweep.sh}"

NUM_SHARDS=${SLURM_ARRAY_TASK_COUNT:-1}
SHARD_ID=$(( ${SLURM_ARRAY_TASK_ID:-0} - ${SLURM_ARRAY_TASK_MIN:-0} ))

# Interleave the full grid across shards so large-K (slow) runs spread evenly.
read -ra _ALL_K <<< "$K_LIST"
MY_K=()
for i in "${!_ALL_K[@]}"; do
  (( i % NUM_SHARDS == SHARD_ID )) && MY_K+=("${_ALL_K[$i]}")
done
MY_K_CSV=$(IFS=,; echo "${MY_K[*]}")

echo "=== $(date) === job ${SLURM_ARRAY_JOB_ID}_${SHARD_ID} on $(hostname) ==="
echo "subset_tokens=$SUBSET_TOKENS  layer=$LAYER  rank=$RANK  shard=$SHARD_ID/$NUM_SHARDS  K=$MY_K_CSV"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

uv run python scripts/wiki_q_fixed_k_sweep.py run \
  --subset-tokens "$SUBSET_TOKENS" \
  --k-list "$MY_K_CSV" \
  --layer "$LAYER" \
  --rank "$RANK" \
  --device cuda

echo "=== $(date) === done ==="
