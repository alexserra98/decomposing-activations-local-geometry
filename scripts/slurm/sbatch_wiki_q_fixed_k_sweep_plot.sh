#!/bin/bash
# ── Plot step for the fixed-q K-sweep (held-out NLL vs K) ──
# Collects best_metric from every trained run and writes results.json + the
# nll_vs_k.png figure. Decoupled from training: submit it as a dependency after
# the training array (see scripts/slurm/submit_wiki_q_fixed_k_sweep.sh), or run it
# standalone any time to (re-)plot whatever runs exist on disk.
#SBATCH --partition=EPYC
#SBATCH --account=LADE
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --job-name=wiki_q_fixed_k_sweep_plot
#SBATCH --output=/u/dssc/zenocosini/decomposing-activations-local-geometry/logs/jobs/wiki_q_fixed_k_sweep_plot_%j.out

set -euo pipefail

REPO_ROOT=/u/dssc/zenocosini/decomposing-activations-local-geometry
mkdir -p "$REPO_ROOT/logs/jobs"
cd "$REPO_ROOT" || exit 1
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

: "${SUBSET_TOKENS:?set SUBSET_TOKENS via submit_wiki_q_fixed_k_sweep.sh}"
: "${LAYER:?set LAYER via submit_wiki_q_fixed_k_sweep.sh}"
: "${RANK:?set RANK via submit_wiki_q_fixed_k_sweep.sh}"

echo "=== $(date) === job $SLURM_JOB_ID on $(hostname) ==="
echo "subset_tokens=$SUBSET_TOKENS  layer=$LAYER  rank=$RANK"

uv run python scripts/wiki_q_fixed_k_sweep.py plot \
  --subset-tokens "$SUBSET_TOKENS" \
  --layer "$LAYER" \
  --rank "$RANK"

echo "=== $(date) === done ==="
