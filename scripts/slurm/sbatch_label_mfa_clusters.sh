#!/bin/bash
# Label MFA clusters with the Orfeo-hosted LLM API.
#
# CPU-only: activations are already assigned, and the LLM is called remotely.
# Submit the full 2x2 sweep with:
#
#   sbatch scripts/slurm/sbatch_label_mfa_clusters.sh
#
# Useful overrides:
#
#   TOP_N=100 MAX_EXAMPLES=40 LLM_WORKERS=4 sbatch scripts/slurm/sbatch_label_mfa_clusters.sh
#
# Outputs:
#   output/experiments/<K>_<layer>/cluster_labels/
#     top_activations.pt
#     cluster_examples.json
#     cluster_labels.json

#SBATCH --partition=EPYC
#SBATCH --account=LADE
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --job-name=mfa_label
#SBATCH --array=0-3
#SBATCH --output=/u/dssc/zenocosini/decomposing-activations-local-geometry/logs/jobs/mfa_label_%A_%a.out

set -euo pipefail

REPO_ROOT=/u/dssc/zenocosini/decomposing-activations-local-geometry
SHARD_DIR=${SHARD_DIR:-"$REPO_ROOT/dalg-cache/pile_gemma2b_activations"}
WINDOWS_DATASET=${WINDOWS_DATASET:-"$REPO_ROOT/dalg-cache/pile_gemma2b_100M_windows/merged"}
TOKENIZER=${TOKENIZER:-google/gemma-2b}

# Keep these modest by default; override from the sbatch environment if needed.
TOP_N=${TOP_N:-50}
MAX_EXAMPLES=${MAX_EXAMPLES:-25}
PAD=${PAD:-10}
CHUNK_SIZE=${CHUNK_SIZE:-2000000}
LLM_WORKERS=${LLM_WORKERS:-4}
LLM_MAX_TOKENS=${LLM_MAX_TOKENS:-512}
LLM_TEMPERATURE=${LLM_TEMPERATURE:-0.0}

# (K, layer) sweep — one entry per array task.
CONFIGS=(
    "1000:5"
    "1000:17"
    "8000:5"
    "8000:17"
)

IFS=":" read -r K LAYER <<< "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
LAYER_TAG=$(printf "%02d" "$LAYER")
MFA_DIR="$SHARD_DIR/layer${LAYER_TAG}_${K}_mfa"
ASSIGNMENTS_PATH="$MFA_DIR/mfa_model_assignments.pt"
OUT_DIR="$REPO_ROOT/output/experiments/${K}_${LAYER_TAG}/cluster_labels"

mkdir -p "$REPO_ROOT/logs/jobs" "$OUT_DIR"
cd "$REPO_ROOT" || exit 1
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

echo "=== $(date) === job ${SLURM_JOB_ID:-local}.${SLURM_ARRAY_TASK_ID:-0} on $(hostname) ==="
echo "K=$K  layer=$LAYER"
echo "assignments: $ASSIGNMENTS_PATH"
echo "windows dataset: $WINDOWS_DATASET"
echo "out_dir: $OUT_DIR"
echo "top_n=$TOP_N  max_examples=$MAX_EXAMPLES  pad=$PAD  llm_workers=$LLM_WORKERS"

if [[ ! -f "$ASSIGNMENTS_PATH" ]]; then
    echo "Missing assignments file: $ASSIGNMENTS_PATH" >&2
    exit 1
fi

if [[ ! -d "$WINDOWS_DATASET" ]]; then
    echo "Missing windows dataset: $WINDOWS_DATASET" >&2
    exit 1
fi

uv run python -m dalg.cli.label_mfa_clusters \
    --assignments-path "$ASSIGNMENTS_PATH" \
    --shard-dir "$SHARD_DIR" \
    --layer "$LAYER" \
    --windows-dataset "$WINDOWS_DATASET" \
    --tokenizer "$TOKENIZER" \
    --out-dir "$OUT_DIR" \
    --top-n "$TOP_N" \
    --max-examples-per-cluster "$MAX_EXAMPLES" \
    --pad "$PAD" \
    --chunk-size "$CHUNK_SIZE" \
    --llm-workers "$LLM_WORKERS" \
    --llm-max-tokens "$LLM_MAX_TOKENS" \
    --llm-temperature "$LLM_TEMPERATURE"

echo "=== $(date) === done ==="
