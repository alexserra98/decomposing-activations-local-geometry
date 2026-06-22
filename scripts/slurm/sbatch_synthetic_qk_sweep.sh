#!/bin/bash
# Fit all q settings for one fixed K per Slurm array task.
#
# The dataset path is derived from K_TRUE/Q_TRUE by default. Generate it
# manually before launching the sweep; the job fails if the dataset is missing.
#   sbatch scripts/slurm/sbatch_synthetic_qk_sweep.sh
#
# To run one K with all q values:
#   K_GRID="8" Q_GRID="10 50 100 200 500" sbatch --array=0-0 scripts/slurm/sbatch_synthetic_qk_sweep.sh

#SBATCH --partition=H100
#SBATCH --account=LADE
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=80G
#SBATCH --time=23:00:00
#SBATCH --job-name=qk_sweep
#SBATCH --array=0-6%3
#SBATCH --nodelist=dgx003
#SBATCH --output=/u/dssc/zenocosini/decomposing-activations-local-geometry/logs/jobs/qk_sweep_%A_%a.out

set -euo pipefail

REPO_ROOT=/u/dssc/zenocosini/decomposing-activations-local-geometry
SCRATCH_ROOT=/orfeo/scratch/dssc/zenocosini/dalg-cache
cd "$REPO_ROOT"

mkdir -p "$REPO_ROOT/logs/jobs" "$REPO_ROOT/dalg-cache/qk_sweep_exploration"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-$SLURM_CPUS_PER_TASK}"

# K_GRID=${K_GRID:-"10 50 100 250 500 750 1000 1250 1500 2000"}
K_GRID=${K_GRID:-"500 750 1000 1250 1500 2000"}
Q_GRID=${Q_GRID:-"5 10 15 20 25 30 35 40 60 80 100 120 140 160 180 200 220 240 260 280 300 350 400 500"}
# Q_GRID=${Q_GRID:-"300 350 400 500"}
D=${D:-500}
K_TRUE=${K_TRUE:-1000}
Q_TRUE=${Q_TRUE:-20}
N_TRAIN=${N_TRAIN:-500000}
N_TEST=${N_TEST:-10000}
SEED=${SEED:-0}
BATCH_SIZE=${BATCH_SIZE:-512}
EPOCHS=${EPOCHS:-100}
LR=${LR:-1e-3}
GRAD_CLIP=${GRAD_CLIP:-5.0}
EARLY_STOP_DELTA=${EARLY_STOP_DELTA:-1e-4}
EARLY_STOP_PATIENCE=${EARLY_STOP_PATIENCE:-5}
EARLY_STOP_MIN_DELTA=${EARLY_STOP_MIN_DELTA:-1e-3}
MEAN_SCALE=${MEAN_SCALE:-6.0}
FACTOR_SCALE=${FACTOR_SCALE:-1.0}
PSI=${PSI:-0.25}
KMEANS_MAX_ITER=${KMEANS_MAX_ITER:-100}
KMEANS_N_INIT=${KMEANS_N_INIT:-3}
OVERWRITE=${OVERWRITE:-0}

DATASET_PATH=${DATASET_PATH:-"$SCRATCH_ROOT/assets/synthetic_mfa_Ktrue${K_TRUE}_qtrue${Q_TRUE}_D${D}_seed${SEED}.pt"}
STAGE_DATASET=${STAGE_DATASET:-1}
NODE_DATASET_DIR=${NODE_DATASET_DIR:-"${SLURM_TMPDIR:-/tmp}/${USER:-zenocosini}/dalg-cache/assets"}
MODEL_ROOT=${MODEL_ROOT:-"$REPO_ROOT/dalg-cache/qk_sweep_exploration"}
RUN_NAME=${RUN_NAME:-"Ktrue${K_TRUE}_qtrue${Q_TRUE}"}
PYTHON=${PYTHON:-"$REPO_ROOT/.venv/bin/python"}

read -r -a K_VALUES <<< "$K_GRID"
read -r -a Q_VALUES <<< "$Q_GRID"

NUM_K=${#K_VALUES[@]}
NUM_Q=${#Q_VALUES[@]}
TOTAL=$((NUM_K * NUM_Q))
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

if (( TASK_ID >= NUM_K )); then
    echo "Task $TASK_ID is outside K grid size $NUM_K; exiting."
    exit 0
fi

K_FIT=${K_VALUES[$TASK_ID]}

if [[ ! -f "$DATASET_PATH" ]]; then
    echo "Dataset not found: $DATASET_PATH"
    echo "Generate it first with:"
    echo "  PYTHONPATH=src $PYTHON scripts/synthetic_mfa_qk_sweep.py generate-dataset \\"
    echo "    --dataset-path \"$DATASET_PATH\" \\"
    echo "    --D \"$D\" --K-true \"$K_TRUE\" --q-true \"$Q_TRUE\" \\"
    echo "    --n-train \"$N_TRAIN\" --n-test \"$N_TEST\" --seed \"$SEED\""
    exit 1
fi

DATASET_FOR_FIT="$DATASET_PATH"
if [[ "$STAGE_DATASET" == "1" ]]; then
    mkdir -p "$NODE_DATASET_DIR"
    STAGED_DATASET="$NODE_DATASET_DIR/$(basename "$DATASET_PATH")"
    LOCK_FILE="$STAGED_DATASET.lock"
    exec 9>"$LOCK_FILE"
    echo "Staging dataset lock: $LOCK_FILE"
    flock 9
    if [[ ! -f "$STAGED_DATASET" ]]; then
        TMP_STAGED="$STAGED_DATASET.tmp.${SLURM_JOB_ID:-local}.${TASK_ID}"
        echo "Copying dataset to node-local storage: $STAGED_DATASET"
        cp -p "$DATASET_PATH" "$TMP_STAGED"
        mv -f "$TMP_STAGED" "$STAGED_DATASET"
    else
        echo "Using existing node-local dataset: $STAGED_DATASET"
    fi
    DATASET_FOR_FIT="$STAGED_DATASET"
    flock -u 9
    exec 9>&-
fi

echo "=== $(date) === job ${SLURM_JOB_ID:-local}.${TASK_ID} on $(hostname) ==="
echo "repo_root: $REPO_ROOT"
echo "dataset: $DATASET_PATH"
echo "dataset_for_fit: $DATASET_FOR_FIT"
echo "model_root: $MODEL_ROOT   run_name: $RUN_NAME"
echo "grid task: K=$K_FIT  q_values=${Q_VALUES[*]}  (K task $((TASK_ID + 1))/$NUM_K, total fits=$TOTAL)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

for Q_FIT in "${Q_VALUES[@]}"; do
    FIT_DIR="$MODEL_ROOT/$RUN_NAME/K$(printf "%04d" "$K_FIT")_q$(printf "%04d" "$Q_FIT")_seed$(printf "%04d" "$SEED")"
    if [[ "$OVERWRITE" != "1" && -f "$FIT_DIR/metrics.json" && -f "$FIT_DIR/mfa_model.pt" ]]; then
        echo "--- $(date) skipping completed K=$K_FIT q=$Q_FIT at $FIT_DIR ---"
        continue
    fi

    echo "--- $(date) fitting K=$K_FIT q=$Q_FIT ---"
    PYTHONUNBUFFERED=1 "$PYTHON" scripts/synthetic_mfa_qk_sweep.py fit-one \
        --dataset-path "$DATASET_FOR_FIT" \
        --model-root "$MODEL_ROOT" \
        --run-name "$RUN_NAME" \
        --D "$D" \
        --K-true "$K_TRUE" \
        --q-true "$Q_TRUE" \
        --K-grid "$(IFS=,; echo "${K_VALUES[*]}")" \
        --q-grid "$(IFS=,; echo "${Q_VALUES[*]}")" \
        --n-train "$N_TRAIN" \
        --n-test "$N_TEST" \
        --seed "$SEED" \
        --batch-size "$BATCH_SIZE" \
        --epochs "$EPOCHS" \
        --lr "$LR" \
        --grad-clip "$GRAD_CLIP" \
        --early-stop-delta "$EARLY_STOP_DELTA" \
        --early-stop-patience "$EARLY_STOP_PATIENCE" \
        --early-stop-min-delta "$EARLY_STOP_MIN_DELTA" \
        --mean-scale "$MEAN_SCALE" \
        --factor-scale "$FACTOR_SCALE" \
        --psi "$PSI" \
        --kmeans-max-iter "$KMEANS_MAX_ITER" \
        --kmeans-n-init "$KMEANS_N_INIT" \
        --device cuda \
        --K-fit "$K_FIT" \
        --q-fit "$Q_FIT"
done

echo "=== $(date) === done ==="
