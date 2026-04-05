#!/bin/bash
#SBATCH -J synth_graph_grid
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err

# ============================================================
# Unified submit + array-job script for synthetic graph runs.
#
# Usage:
#   1) Edit the lists and training settings below.
#   2) Submit by running:
#        bash synthetic_graph_grid_unified.slurm
#
# The script computes the full Cartesian product of
#   (n, p, d, k, seed)
# and submits itself as a SLURM array with exactly ARRAY_JOBS tasks.
#
# Each array task processes roughly the same number of runs.
#
# Each individual run saves to
#   /home/kirilb/orcd/pool/TopK/SyntheticGraphRuns/
#     n_{n}_p_{p}_d_{d}_k_{k}_seed_{seed}_time_{time}
#
# The Python training script is expected at:
#   /home/kirilb/orcd/pool/TopK/train_siglip_bipartite_npz.py
# ============================================================

# ---------------------------
# Edit these lists
# ---------------------------
N_VALUES=(10 20 40 80)
P_VALUES=(0.25 0.5 1)
D_VALUES=(2 3 4 5 6 7 8 9 10 11 12
K_VALUES=(1 2 3 4)
SEED_VALUES=(0)

# ---------------------------
# Training settings
# ---------------------------
INITIALIZATION="random"
NUM_STEPS=100000
SAVE_EVERY=1000
BATCH_SIZE=""        # leave empty to use full n
LR=0.01
MIN_LR_RATIO=0.01
WARMUP_FRAC=0.05
DEVICE=""            # leave empty to let python decide

# Fixed array size
ARRAY_JOBS=20
MAX_CONCURRENT=20

BASE_SAVE_PATH="/home/kirilb/orcd/pool/TopK/SyntheticGraphRuns"
PYTHON_SCRIPT="embed.py"
LOG_DIR="${BASE_SAVE_PATH}/logs"
mkdir -p "$BASE_SAVE_PATH" "$LOG_DIR"

NUM_N=${#N_VALUES[@]}
NUM_P=${#P_VALUES[@]}
NUM_D=${#D_VALUES[@]}
NUM_K=${#K_VALUES[@]}
NUM_SEED=${#SEED_VALUES[@]}
TOTAL_RUNS=$((NUM_N * NUM_P * NUM_D * NUM_K * NUM_SEED))

# ------------------------------------------------------------
# Submit mode
# ------------------------------------------------------------
if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
  echo "Submitting ${TOTAL_RUNS} total runs"
  echo "Grid sizes: n=${NUM_N}, p=${NUM_P}, d=${NUM_D}, k=${NUM_K}, seed=${NUM_SEED}"
  echo "Array jobs: ${ARRAY_JOBS}"
  echo "Base save path: ${BASE_SAVE_PATH}"

  sbatch --array=0-$((ARRAY_JOBS - 1))%${MAX_CONCURRENT} "$0"
  exit $?
fi

# ------------------------------------------------------------
# Array-task mode
# Each array task gets a balanced chunk of the TOTAL_RUNS runs.
# Task t handles global run ids in:
#   [ floor(t * TOTAL_RUNS / ARRAY_JOBS),
#     floor((t+1) * TOTAL_RUNS / ARRAY_JOBS) - 1 ]
# ------------------------------------------------------------
TASK_ID=${SLURM_ARRAY_TASK_ID}

if [ "$TASK_ID" -ge "$ARRAY_JOBS" ]; then
  echo "Task ${TASK_ID} is outside the array size ${ARRAY_JOBS}. Exiting."
  exit 0
fi

START_RUN=$(( TASK_ID * TOTAL_RUNS / ARRAY_JOBS ))
END_RUN=$(( ( (TASK_ID + 1) * TOTAL_RUNS / ARRAY_JOBS ) - 1 ))

if [ "$START_RUN" -gt "$END_RUN" ]; then
  echo "Task ${TASK_ID} has no assigned runs. Exiting."
  exit 0
fi

if [ ! -f "$PYTHON_SCRIPT" ]; then
  echo "Could not find $PYTHON_SCRIPT"
  echo "Copy train_siglip_bipartite_npz.py there, or edit PYTHON_SCRIPT in this script."
  exit 1
fi

module load miniforge
source activate GPUenv

echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "Assigned global run ids: ${START_RUN} ... ${END_RUN}"

decode_and_run() {
  local GLOBAL_ID=$1

  if [ "$GLOBAL_ID" -ge "$TOTAL_RUNS" ]; then
    echo "Global run id ${GLOBAL_ID} is outside TOTAL_RUNS=${TOTAL_RUNS}. Skipping."
    return
  fi

  local seed_idx rem k_idx d_idx p_idx n_idx
  seed_idx=$(( GLOBAL_ID % NUM_SEED ))
  rem=$(( GLOBAL_ID / NUM_SEED ))

  k_idx=$(( rem % NUM_K ))
  rem=$(( rem / NUM_K ))

  d_idx=$(( rem % NUM_D ))
  rem=$(( rem / NUM_D ))

  p_idx=$(( rem % NUM_P ))
  rem=$(( rem / NUM_P ))

  n_idx=$(( rem % NUM_N ))

  local n p d k seed
  n=${N_VALUES[$n_idx]}
  p=${P_VALUES[$p_idx]}
  d=${D_VALUES[$d_idx]}
  k=${K_VALUES[$k_idx]}
  seed=${SEED_VALUES[$seed_idx]}

  local RUN_TIME RUN_DIR
  RUN_TIME=$(date +"%Y%m%d_%H%M%S")
  RUN_DIR="${BASE_SAVE_PATH}/n_${n}_p_${p}_d_${d}_k_${k}_seed_${seed}_time_${RUN_TIME}"
  mkdir -p "$RUN_DIR"

  echo "--------------------------------------------------"
  echo "Global run id: ${GLOBAL_ID}"
  echo "n=${n} p=${p} d=${d} k=${k} seed=${seed}"
  echo "Saving to ${RUN_DIR}"

  CMD=(
    python "$PYTHON_SCRIPT"
    --n "$n"
    --p "$p"
    --d "$d"
    --k "$k"
    --seed "$seed"
    --initialization "$INITIALIZATION"
    --num_steps "$NUM_STEPS"
    --save_every "$SAVE_EVERY"
    --save_path "$RUN_DIR"
    --lr "$LR"
    --min_lr_ratio "$MIN_LR_RATIO"
    --warmup_frac "$WARMUP_FRAC"
  )

  if [ -n "$BATCH_SIZE" ]; then
    CMD+=(--batch_size "$BATCH_SIZE")
  fi

  if [ -n "$DEVICE" ]; then
    CMD+=(--device "$DEVICE")
  fi

  printf 'Running command:\n'
  printf ' %q' "${CMD[@]}"
  printf '\n'

  "${CMD[@]}"
}

for GLOBAL_ID in $(seq "$START_RUN" "$END_RUN"); do
  decode_and_run "$GLOBAL_ID"
done