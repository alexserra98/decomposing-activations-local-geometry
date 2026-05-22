#!/bin/bash
# Run the MFA cluster-label Streamlit dashboard on a cluster node.
#
# Important: open the forwarded localhost URL on your laptop, not the raw URL
# printed for the worker node.

set -euo pipefail

REPO_ROOT=/u/dssc/zenocosini/decomposing-activations-local-geometry
PORT=${PORT:-8501}
ADDRESS=${ADDRESS:-127.0.0.1}

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

echo "Starting cluster labels dashboard"
echo "Node: $(hostname)"
echo "Listening on: ${ADDRESS}:${PORT}"
echo
echo "Open the forwarded port on your laptop:"
echo "  http://localhost:${PORT}"
echo

uv run streamlit run scripts/cluster_labels_dashboard.py \
    --server.headless true \
    --server.address "$ADDRESS" \
    --server.port "$PORT" \
    --browser.gatherUsageStats false
