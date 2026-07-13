#!/bin/bash
# ── Submit the fixed-q K-sweep: training array + a single plot job after it ──
# Submits the training array, then the plot job with an afterany dependency so
# it runs once the array finishes (afterany, not afterok, so a partial array
# still produces a figure over whatever trained). SUBSET_TOKENS / K_LIST set
# here propagate to both jobs via the exported environment.
#
# To change parallelism, edit `#SBATCH --array=0-1` in sbatch_wiki_q_fixed_k_sweep.sh;
# the grid is split round-robin across however many tasks that range defines.
#
# Usage:
#   scripts/slurm/submit_wiki_q_fixed_k_sweep.sh
#   SUBSET_TOKENS=1M K_LIST="200 400 600 800 1000 1500 2000" scripts/slurm/submit_wiki_q_fixed_k_sweep.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export SUBSET_TOKENS=${SUBSET_TOKENS:-100K}
export K_LIST=${K_LIST:-"10 25 50 75 100 150 200 500 700 1000"}
export LAYER=${LAYER:-17}
export RANK=${RANK:-10}

ARRAY_JID=$(sbatch --parsable "$HERE/sbatch_wiki_q_fixed_k_sweep.sh")
echo "submitted training array: $ARRAY_JID  (subset_tokens=$SUBSET_TOKENS  K_LIST=$K_LIST layer=$LAYER  rank=$RANK)"

PLOT_JID=$(sbatch --parsable --dependency=afterany:"$ARRAY_JID" "$HERE/sbatch_wiki_q_fixed_k_sweep_plot.sh")
echo "submitted plot job: $PLOT_JID  (runs after array $ARRAY_JID)"
