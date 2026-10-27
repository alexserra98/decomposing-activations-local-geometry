#!/usr/bin/env bash
# Pass reviewed resources and an unused output directory to sbatch.
set -euo pipefail
: "${SLURM_JOB_ID:?Run plotting through Slurm}"
REPO_ROOT="${SLURM_SUBMIT_DIR:?Submit from the repository root}"
ID_PLOT_PYTHON="$REPO_ROOT/.venv/bin/python"
export ID_FIGURE_OUTPUT="${1:?Supply the figure output directory}"
mkdir "$ID_FIGURE_OUTPUT"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT/scripts/temporary${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export XDG_CACHE_HOME="${SLURM_TMPDIR:-/tmp}/dalg-id-plot-${SLURM_JOB_ID}/cache"
export PIP_CACHE_DIR="$XDG_CACHE_HOME/pip"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export IPYTHONDIR="$XDG_CACHE_HOME/ipython"
export JUPYTER_PATH="$XDG_CACHE_HOME/jupyter"
export JUPYTER_RUNTIME_DIR="$XDG_CACHE_HOME/jupyter-runtime"
mkdir -p "$XDG_CACHE_HOME" "$JUPYTER_RUNTIME_DIR" "$ID_FIGURE_OUTPUT/source"
hostname
git rev-parse HEAD > "$ID_FIGURE_OUTPUT/source_commit.txt"
git diff --stat > "$ID_FIGURE_OUTPUT/source_diff_stat.txt"
cp scripts/temporary/plot_cluster_id_experiment.py \
   scripts/temporary/run_cluster_id_experiment.py \
   scripts/slurm/temporary/sbatch_plot_cluster_id_experiment.sh \
   tests/test_cluster_id_plotting.py \
   notebooks/toy_manifold_noise_sweep_intrinsic_dimension.ipynb \
   "$ID_FIGURE_OUTPUT/source/"
"$ID_PLOT_PYTHON" -m pip list --format=json > "$ID_FIGURE_OUTPUT/environment_before.json"
if [[ "${ID_PLOT_INSTALL_NOTEBOOK_TOOLS:-0}" == 1 ]]; then
    "$ID_PLOT_PYTHON" -m pip install nbclient nbformat
fi
"$ID_PLOT_PYTHON" -m pip list --format=json > "$ID_FIGURE_OUTPUT/environment_after.json"
"$ID_PLOT_PYTHON" -m pytest -q tests/test_cluster_id_plotting.py \
    --junitxml="$ID_FIGURE_OUTPUT/tests.xml"
"$ID_PLOT_PYTHON" - <<'PY'
import json
import os
from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient
import pandas as pd

output = Path(os.environ["ID_FIGURE_OUTPUT"])
kernel = Path(os.environ["JUPYTER_PATH"]) / "kernels/python3"
kernel.mkdir(parents=True)
(kernel / "kernel.json").write_text(json.dumps({
    "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
    "display_name": "DALG repository .venv", "language": "python",
}))
notebook = nbformat.read("notebooks/toy_manifold_noise_sweep_intrinsic_dimension.ipynb", as_version=4)
nbformat.validate(notebook)
client = NotebookClient(notebook, timeout=1800, kernel_name="python3",
                        resources={"metadata": {"path": str(Path.cwd())}})
try:
    client.execute()
finally:
    nbformat.write(notebook, output / "toy_manifold_noise_sweep_intrinsic_dimension_executed.ipynb")
assert len(list(output.glob("*.png"))) == 9
assert len(list(output.glob("*.pdf"))) == 9
if os.environ.get("ID_BASELINE_FIGURES"):
    baseline = Path(os.environ["ID_BASELINE_FIGURES"])
    keys = ["relative_run", "split", "manifold_id"]
    current = pd.read_csv(output / "manifold_ids.csv").set_index(keys).sort_index().sort_index(axis=1)
    previous = pd.read_csv(baseline / "manifold_ids.csv").set_index(keys).sort_index().sort_index(axis=1)
    pd.testing.assert_index_equal(current.index, previous.index)
    pd.testing.assert_frame_equal(current.xs("train", level="split"),
                                  previous.xs("train", level="split"), rtol=1e-12, atol=1e-12)
    checks = {"training_summaries_unchanged": True, "baseline_figures": str(baseline), "test": {}}
    for estimator in ["two_nn", "gride_2", "gride_4", "gride_8", "gride_16", "gride_32", "gride_64"]:
        before = previous.xs("test", level="split")[f"{estimator}_n"]
        after = current.xs("test", level="split")[f"{estimator}_n"]
        assert after.ge(before).all()
        checks["test"][estimator] = {
            "valid_clusters_before": int(before.sum()), "valid_clusters_after": int(after.sum()),
            "empty_cells_before": int(before.eq(0).sum()), "empty_cells_after": int(after.eq(0).sum()),
            "total_cells": len(after),
        }
    (output / "comparison_checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(json.dumps(checks, indent=2), flush=True)
print(f"Rendered both summary tables and seven train/test figures in {output}", flush=True)
PY
