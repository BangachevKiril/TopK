#!/bin/bash
#SBATCH -J synth_embed_nce
#SBATCH -p mit_preemptable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=05:59:00
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err

# ============================================================
# Recursively embed all saved synthetic graphs under GRAPH_ROOT
# using a trainable-temperature InfoNCE objective.
#
# Usage:
#   bash infonce_embed_bash.sh /path/to/graph_root /path/to/out_root
#
# This version:
#   1) does NOT use manifests,
#   2) recursively scans GRAPH_ROOT for graph_n_*_k_*_N_*_seed_*.npz
#      (and legacy .npy if no .npz twin exists),
#   3) parses n, k, N, seed from the filename,
#   4) mirrors the same relative subfolder structure under OUT_ROOT,
#   5) runs one embedding job per (graph, d).
# ============================================================

# ------------------------
# Embedding dimensions
# ------------------------
d_values=(2 3 4 5 6 7 8 9 10 11 12 25 50 100)

# ------------------------
# Training settings
# ------------------------
INITIALIZATION="random"
NUM_STEPS=100000
SAVE_EVERY=1000
BATCH_SIZE=""
LR=0.01
MIN_LR_RATIO=0.01
WARMUP_FRAC=0.05
TEMPERATURE=0.1
DEVICE=""

ARRAY_JOBS=40
MAX_CONCURRENT=20

GRAPH_ROOT="${1:-${GRAPH_ROOT:-/home/kirilb/orcd/pool/TopK/SyntheticGraphsSmall}}"
OUT_ROOT="${2:-${OUT_ROOT:-/home/kirilb/orcd/pool/TopK/SyntheticEmbeddingsSmall/infonce_${INITIALIZATION}_initialization}}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT_PATH="$(readlink -f "$0")"
PYTHON_SCRIPT="infonce_embed.py"

list_graphs_unique() {
    local graph_root="$1"

    while IFS= read -r path; do
        echo "$path"
    done < <(find "$graph_root" -type f -name 'graph_n_*_k_*_N_*_seed_*.npz' | sort)

    while IFS= read -r path; do
        if [ ! -f "${path%.npy}.npz" ]; then
            echo "$path"
        fi
    done < <(find "$graph_root" -type f -name 'graph_n_*_k_*_N_*_seed_*.npy' | sort)
}

parse_graph_metadata() {
    local graph_path="$1"
    local base
    base="$(basename "$graph_path")"

    if [[ "$base" =~ ^graph_n_([0-9]+)_k_([0-9]+)_N_([0-9]+)_seed_([0-9]+)\.(npz|npy)$ ]]; then
        echo "${BASH_REMATCH[1]} ${BASH_REMATCH[2]} ${BASH_REMATCH[3]} ${BASH_REMATCH[4]}"
        return 0
    fi

    return 1
}

build_run_dir() {
    local graph_path="$1"
    local d="$2"

    local graph_dir graph_stem rel_dir
    graph_dir="$(dirname "$graph_path")"
    graph_stem="$(basename "${graph_path%.*}")"

    if [ "$graph_dir" = "$GRAPH_ROOT" ]; then
        rel_dir=""
    else
        rel_dir="${graph_dir#${GRAPH_ROOT}/}"
    fi

    if [ -n "$rel_dir" ]; then
        echo "${OUT_ROOT}/${rel_dir}/${graph_stem}/d_${d}"
    else
        echo "${OUT_ROOT}/${graph_stem}/d_${d}"
    fi
}

mkdir -p logs
mkdir -p "${OUT_ROOT}"

mapfile -t GRAPH_PATHS < <(list_graphs_unique "${GRAPH_ROOT}" | sort)

TOTAL_GRAPHS=${#GRAPH_PATHS[@]}
NUM_D_VALUES=${#d_values[@]}
TOTAL_RUNS=$((TOTAL_GRAPHS * NUM_D_VALUES))

if [ "${TOTAL_GRAPHS}" -eq 0 ]; then
    echo "No graph files found under: ${GRAPH_ROOT}"
    echo "Expected filenames like:"
    echo "  graph_n_<n>_k_<k>_N_<N>_seed_<seed>.npz"
    echo "or legacy .npy versions."
    exit 1
fi

if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    array_count=${ARRAY_JOBS}
    if [ "${TOTAL_RUNS}" -lt "${array_count}" ]; then
        array_count=${TOTAL_RUNS}
    fi

    echo "SCRIPT_PATH=${SCRIPT_PATH}"
    echo "PYTHON_SCRIPT=${PYTHON_SCRIPT}"
    echo "GRAPH_ROOT=${GRAPH_ROOT}"
    echo "OUT_ROOT=${OUT_ROOT}"
    echo "TOTAL_GRAPHS=${TOTAL_GRAPHS}"
    echo "NUM_D_VALUES=${NUM_D_VALUES}"
    echo "TOTAL_RUNS=${TOTAL_RUNS}"
    echo "Submitting ${array_count} array tasks."

    sbatch \
        --array=0-$((array_count - 1))%${MAX_CONCURRENT} \
        --export=ALL,GRAPH_ROOT="${GRAPH_ROOT}",OUT_ROOT="${OUT_ROOT}",TOTAL_RUNS="${TOTAL_RUNS}",ARRAY_COUNT="${array_count}" \
        "${SCRIPT_PATH}"
    exit $?
fi

module load miniforge
source activate GPUenv

if [ -z "${TOTAL_RUNS:-}" ] || [ -z "${ARRAY_COUNT:-}" ]; then
    echo "Missing TOTAL_RUNS or ARRAY_COUNT in environment."
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
    start=$(( task_id * (base + 1) ))
    end=$(( start + base ))
else
    start=$(( extra * (base + 1) + (task_id - extra) * base ))
    end=$(( start + base - 1 ))
fi

if [ "${start}" -ge "${TOTAL_RUNS}" ]; then
    echo "Task ${task_id} has no assigned runs."
    exit 0
fi

if [ "${end}" -ge "${TOTAL_RUNS}" ]; then
    end=$((TOTAL_RUNS - 1))
fi

echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "GRAPH_ROOT=${GRAPH_ROOT}"
echo "OUT_ROOT=${OUT_ROOT}"
echo "Task ${task_id} processing run indices ${start} through ${end}."

for global_idx in $(seq "${start}" "${end}"); do
    graph_idx=$(( global_idx / NUM_D_VALUES ))
    d_idx=$(( global_idx % NUM_D_VALUES ))

    graph_path="${GRAPH_PATHS[$graph_idx]}"
    d="${d_values[$d_idx]}"

    metadata="$(parse_graph_metadata "${graph_path}")"
    if [ $? -ne 0 ] || [ -z "${metadata}" ]; then
        echo "Could not parse metadata from graph file: ${graph_path}"
        exit 1
    fi

    read -r n k N seed <<< "${metadata}"

    run_dir="$(build_run_dir "${graph_path}" "${d}")"
    mkdir -p "${run_dir}"

    echo "--------------------------------------------------"
    echo "global_idx=${global_idx} n=${n} k=${k} N=${N} d=${d} seed=${seed}"
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
        --temperature "${TEMPERATURE}"
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