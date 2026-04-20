#!/bin/bash
#SBATCH --partition=DGX
#SBATCH --account=LADE
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:A100:1
#SBATCH --mem=50G
#SBATCH --time=6:00:00
#SBATCH --job-name=gemma_extract
#SBATCH --output=output_job/gemma_extract_%j.out

# ── Config (edit to taste) ───────────────────────────────────────────────
DATASET="/orfeo/scratch/dssc/zenocosini/pile_gemma2b_100M_windows/merged"
OUT_DIR="/orfeo/scratch/dssc/zenocosini/pile_gemma2b_activations"
MODEL="google/gemma-2b"
LAYERS="5 17"
DTYPE="float16"
BATCH=16
SHARD=512
NUM_WORKERS=4

# Pass DEBUG=1 to smoke-test on a few rows: `sbatch --export=ALL,DEBUG=1 scripts/sbatch_extract_activations.sh`
DEBUG_FLAGS=""
if [[ "${DEBUG:-0}" == "1" ]]; then
    DEBUG_FLAGS="--debug --limit ${DEBUG_LIMIT:-64}"
    OUT_DIR="${OUT_DIR}_debug"
fi

# ── Env ──────────────────────────────────────────────────────────────────
mkdir -p output_job
cd "$SLURM_SUBMIT_DIR" || exit 1

echo "=== $(date) === job $SLURM_JOB_ID on $(hostname) ==="
echo "out_dir: $OUT_DIR"
echo "layers: $LAYERS  dtype: $DTYPE  batch: $BATCH  shard: $SHARD"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# ── Run ──────────────────────────────────────────────────────────────────
uv run python experiments/run_layer.py extract-windows \
    --dataset "$DATASET" \
    --out-dir "$OUT_DIR" \
    --model "$MODEL" \
    --layers $LAYERS \
    --dtype "$DTYPE" \
    --extract-batch-size "$BATCH" \
    --shard-size "$SHARD" \
    --num-workers "$NUM_WORKERS" \
    --device cuda \
    $DEBUG_FLAGS

echo "=== $(date) === done ==="
