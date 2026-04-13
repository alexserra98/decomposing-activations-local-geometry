#!/bin/bash
#
# SLURM array job: one task per layer.
# Adjust MODEL, DATASET, K, NUM_LAYERS and resource requests to your setup.
#
# Usage:
#   sbatch scripts/sbatch_layer.sh
#
#SBATCH --job-name=mfa-L%a
#SBATCH --array=0-31              # one job per layer (adjust to NUM_LAYERS-1)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=logs/mfa-layer-%a.out
#SBATCH --error=logs/mfa-layer-%a.err

# ── Configuration (edit these) ──────────────────────────────────────────

MODEL="meta-llama/Llama-2-7b"     # TransformerLens model name
DATASET="./data/supervised.json"   # path to dataset
BASE_DIR="results/llama2-7b/supervised"
K=8000
RANK=10
EPOCHS=15
MAX_TOKENS=0                       # 0 = no cap
BATCH_PAIRS=4096                   # overlap chunk size (tune for GPU mem)
SEED=42

# ── Derived ─────────────────────────────────────────────────────────────

LAYER=$SLURM_ARRAY_TASK_ID
OUT_DIR="${BASE_DIR}/layer_$(printf '%02d' $LAYER)"

mkdir -p logs

echo "=== Layer $LAYER === $(date) ==="
echo "Output: $OUT_DIR"

# ── Run ─────────────────────────────────────────────────────────────────

uv run python experiments/run_layer.py all \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --layer "$LAYER" \
    --out-dir "$OUT_DIR" \
    --K "$K" \
    --rank "$RANK" \
    --epochs "$EPOCHS" \
    --max-tokens "$MAX_TOKENS" \
    --device cuda \
    --batch-pairs "$BATCH_PAIRS" \
    --seed "$SEED"

echo "=== Done layer $LAYER === $(date) ==="
