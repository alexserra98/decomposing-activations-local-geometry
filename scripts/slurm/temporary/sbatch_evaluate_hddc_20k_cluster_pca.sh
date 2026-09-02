#!/bin/bash
# Evaluate the existing 20k HDDC cluster-PCA run without retraining it.
#SBATCH --partition=H100
#SBATCH --account=LADE
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH --job-name=eval-hddc-pca-20k
#SBATCH --output=/orfeo/cephfs/home/dssc/zenocosini/decomposing-activations-local-geometry/logs/experiments/adaptive_q_toy_20k_hddc_cluster_pca/evaluation_%j.out

set -euo pipefail

REPO_ROOT=/orfeo/cephfs/home/dssc/zenocosini/decomposing-activations-local-geometry
RUN_DIR=/orfeo/cephfs/scratch/dssc/zenocosini/dalg-cache/toy_manifold_models_20k/adaptive_q_toy_20k_hddc_cluster_pca/hddc__toy_manifolds_d128_20k_noise1e4__l00__k200__q16__s42__afa0fb9e
SHARD_DIR=/orfeo/cephfs/scratch/dssc/zenocosini/dalg-cache/assets/toy_manifolds_circle_helix_D128_20K_noise1e4_shards
ASSIGNMENTS_PATH="$RUN_DIR/mfa_model_assignments.pt"
METRICS_PATH="$RUN_DIR/metrics.json"

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

if [[ -e "$ASSIGNMENTS_PATH" || -e "$METRICS_PATH" ]]; then
  echo "Refusing to overwrite an existing assignment or metrics artifact." >&2
  exit 1
fi

"$REPO_ROOT/.venv/bin/python" -m dalg.cli.run_metrics assignments \
  --data-dir "$RUN_DIR" \
  --shard-dir "$SHARD_DIR" \
  --layer 0 \
  --batch-size 4096 \
  --device cuda \
  --seed 42 \
  --model-type hddc \
  --save-path "$ASSIGNMENTS_PATH"

"$REPO_ROOT/.venv/bin/python" - "$RUN_DIR" "$SHARD_DIR" <<'PY'
import json
import sys
from pathlib import Path

from dalg.evaluation.toy_manifold_tiling import evaluate_toy_manifold_tiling
from dalg.pipeline import _write_json_atomic

run_dir = Path(sys.argv[1])
shard_dir = Path(sys.argv[2])
run_spec = json.loads((run_dir / "run_spec.json").read_text())
metrics = evaluate_toy_manifold_tiling(
    run_dir,
    shard_dir=shard_dir,
    layer=0,
    model_kind="hddc",
    assignments_path=run_dir / "mfa_model_assignments.pt",
    batch_size=4096,
    device="cuda",
)
metrics["run_id"] = run_spec["run_id"]
metrics["identity_hash"] = run_spec["identity_hash"]
metrics_path = run_dir / "metrics.json"
_write_json_atomic(metrics_path, metrics)
print(f"Metrics saved to {metrics_path}")
PY
