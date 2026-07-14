#!/bin/bash
#SBATCH --partition=H100
#SBATCH --account=LADE
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --job-name=pca_size_validation
#SBATCH --output=/u/dssc/zenocosini/decomposing-activations-local-geometry/logs/experiments/pca_size_validation_%j.out

set -euo pipefail

REPO_ROOT=/u/dssc/zenocosini/decomposing-activations-local-geometry
LAYER=${LAYER:-5}
K=${K:-1000}
RANK=${RANK:-10}
NUM_CLUSTERS=${NUM_CLUSTERS:-128}
SEED=${SEED:-0}
SAMPLE_SIZES=${SAMPLE_SIZES:-"2000 5000 10000 20000"}
RUN_NAME=${RUN_NAME:-"layer$(printf '%02d' "$LAYER")_K${K}_q${RANK}_c${NUM_CLUSTERS}_seed${SEED}"}
OVERWRITE=${OVERWRITE:-0}

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

read -r -a SAMPLE_SIZE_ARGS <<< "$SAMPLE_SIZES"
OVERWRITE_ARGS=()
if [[ "$OVERWRITE" == "1" ]]; then
  OVERWRITE_ARGS+=(--overwrite)
fi

uv run python -u scripts/pca_size_validation/run_validation.py \
  --layer "$LAYER" \
  --K "$K" \
  --rank "$RANK" \
  --num-clusters "$NUM_CLUSTERS" \
  --seed "$SEED" \
  --sample-sizes "${SAMPLE_SIZE_ARGS[@]}" \
  --top-pcs 100 \
  --pca-device cuda \
  --output-dir "dalg-cache/pile_gemma2b_models/pca_size_validation/$RUN_NAME" \
  "${OVERWRITE_ARGS[@]}"
