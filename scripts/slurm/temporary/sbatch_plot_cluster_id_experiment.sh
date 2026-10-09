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
print(f"Rendered both summary tables and seven train/test figures in {output}", flush=True)
PY
