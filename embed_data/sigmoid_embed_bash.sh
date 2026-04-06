#!/bin/bash
#SBATCH -J synth_embed_saved
#SBATCH -p mit_preemptable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=05:59:00
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err

# ============================================================
# Run embedding training on pre-generated synthetic graphs.
#
# Usage:
#   bash embed_from_saved_graph_array.sh
#
# This script:
#   1) builds a manifest of valid (n, N, d, k, seed) runs,
#   2) only includes cases with N <= binom(n, k),
#   3) only includes runs whose saved graph file exists,
#   4) splits the manifest as evenly as possible across ARRAY_JOBS tasks,
#   5) each task trains on its assigned chunk.
# ============================================================

# ------------------------
# Grid over saved graphs
# ------------------------
N_LEFT_VALUES=(10 20 40 80 160)
N_RIGHT_VALUES=(10 20 40 80 160 320 640 1280 2560 5120 10240 20480)
D_VALUES=(2 3 4 5 6 7 8 9 10 11 12)
K_VALUES=(1 2 3 4 5)
SEED_VALUES=(0 1 2 3 4 5 6 7 8 9)

# ------------------------
# Training settings
# ------------------------
INITIALIZATION="random"
NUM_STEPS=100000
SAVE_EVERY=1000
BATCH_SIZE=""        # leave empty to use full n
LR=0.01
MIN_LR_RATIO=0.01
WARMUP_FRAC=0.05
DEVICE=""            # leave empty to let python decide

ARRAY_JOBS=20
MAX_CONCURRENT=20

GRAPH_DIR="/home/kirilb/orcd/pool/TopK/SyntheticGraphs"
BASE_SAVE_PATH="/home/kirilb/orcd/pool/TopK/SyntheticEmbeddings/sigmoid_${INITIALIZATION}_initialization"
PYTHON_SCRIPT="sigmoid_embed.py"
MANIFEST_DIR="${BASE_SAVE_PATH}/manifests"
LOG_DIR="${BASE_SAVE_PATH}/logs"

mkdir -p logs
mkdir -p "${BASE_SAVE_PATH}" "${MANIFEST_DIR}" "${LOG_DIR}"

if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    timestamp=$(date +%Y%m%d_%H%M%S)
    manifest="${MANIFEST_DIR}/embed_manifest_${timestamp}.txt"
    : > "${manifest}"

    missing_count=0

    for n in "${N_LEFT_VALUES[@]}"; do
        for N in "${N_RIGHT_VALUES[@]}"; do
            for d in "${D_VALUES[@]}"; do
                for k in "${K_VALUES[@]}"; do
                    max_neighborhoods=$(python3 - <<PY
import math
print(math.comb(${n}, ${k}))
PY
)
                    if [ "${N}" -le "${max_neighborhoods}" ]; then
                        for seed in "${SEED_VALUES[@]}"; do
                            graph_path="${GRAPH_DIR}/graph_n_${n}_k_${k}_N_${N}_seed_${seed}.npy"
                            if [ -f "${graph_path}" ]; then
                                echo "${n} ${N} ${d} ${k} ${seed} ${graph_path}" >> "${manifest}"
                            else
                                echo "Missing graph, skipping: ${graph_path}"
                                missing_count=$((missing_count + 1))
                            fi
                        done
                    fi
                done
            done
        done
    done

    total_runs=$(wc -l < "${manifest}")
    if [ "${total_runs}" -eq 0 ]; then
        echo "No valid runs found."
        echo "Looked for graphs under: ${GRAPH_DIR}"
        exit 1
    fi

    array_count=${ARRAY_JOBS}
    if [ "${total_runs}" -lt "${array_count}" ]; then
        array_count=${total_runs}
    fi

    echo "Graph directory: ${GRAPH_DIR}"
    echo "Result directory: ${BASE_SAVE_PATH}"
    echo "Manifest: ${manifest}"
    echo "Total valid runs with existing graphs: ${total_runs}"
    echo "Missing graph files skipped: ${missing_count}"
    echo "Submitting ${array_count} array tasks."

    sbatch \
        --array=0-$((array_count - 1))%${MAX_CONCURRENT} \
        --export=ALL,MANIFEST="${manifest}",TOTAL_RUNS="${total_runs}",ARRAY_COUNT="${array_count}" \
        "$0"
    exit $?
fi

module load miniforge
source activate GPUenv

if [ -z "${MANIFEST:-}" ] || [ -z "${TOTAL_RUNS:-}" ] || [ -z "${ARRAY_COUNT:-}" ]; then
    echo "Missing MANIFEST, TOTAL_RUNS, or ARRAY_COUNT in environment."
    exit 1
fi

if [ ! -f "${MANIFEST}" ]; then
    echo "Manifest file not found: ${MANIFEST}"
    exit 1
fi

if [ ! -f "${PYTHON_SCRIPT}" ]; then
    echo "Python script not found: ${PYTHON_SCRIPT}"
    exit 1
fi

task_id=${SLURM_ARRAY_TASK_ID}
base=$(( TOTAL_RUNS / ARRAY_COUNT ))
extra=$(( TOTAL_RUNS % ARRAY_COUNT ))

if [ "${task_id}" -lt "${extra}" ]; then
    start=$(( task_id * (base + 1) + 1 ))
    end=$(( start + base ))
else
    start=$(( extra * (base + 1) + (task_id - extra) * base + 1 ))
    end=$(( start + base - 1 ))
fi

if [ "${start}" -gt "${TOTAL_RUNS}" ]; then
    echo "Task ${task_id} has no assigned runs."
    exit 0
fi

if [ "${end}" -gt "${TOTAL_RUNS}" ]; then
    end=${TOTAL_RUNS}
fi

echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "Task ${task_id} processing manifest lines ${start} through ${end}."

for line_no in $(seq "${start}" "${end}"); do
    line=$(sed -n "${line_no}p" "${MANIFEST}")
    n=$(echo "${line}" | awk '{print $1}')
    N=$(echo "${line}" | awk '{print $2}')
    d=$(echo "${line}" | awk '{print $3}')
    k=$(echo "${line}" | awk '{print $4}')
    seed=$(echo "${line}" | awk '{print $5}')
    graph_path=$(echo "${line}" | awk '{print $6}')

    run_time=$(date +%Y%m%d_%H%M%S_%N)
    run_dir="${BASE_SAVE_PATH}/n_${n}_N_${N}_d_${d}_k_${k}_seed_${seed}_time_${run_time}"
    mkdir -p "${run_dir}"

    echo "--------------------------------------------------"
    echo "line=${line_no} n=${n} N=${N} d=${d} k=${k} seed=${seed}"
    echo "graph_path=${graph_path}"
    echo "save_path=${run_dir}"

    cmd=(
        python "${PYTHON_SCRIPT}"
        --graph_path "${graph_path}"
        --n "${n}"
        --N "${N}"
        --d "${d}"
        --k "${k}"
        --seed "${seed}"
        --initialization "${INITIALIZATION}"
        --num_steps "${NUM_STEPS}"
        --save_every "${SAVE_EVERY}"
        --save_path "${run_dir}"
        --lr "${LR}"
        --min_lr_ratio "${MIN_LR_RATIO}"
        --warmup_frac "${WARMUP_FRAC}"
    )

    if [ -n "${BATCH_SIZE}" ]; then
        cmd+=(--batch_size "${BATCH_SIZE}")
    fi

    if [ -n "${DEVICE}" ]; then
        cmd+=(--device "${DEVICE}")
    fi

    printf 'Running command:\n'
    printf ' %q' "${cmd[@]}"
    printf '\n'

    "${cmd[@]}"
done
