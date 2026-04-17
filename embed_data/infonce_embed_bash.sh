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
# Embed all saved synthetic graphs directly under GRAPH_ROOT (non-recursive).
#
# Usage:
#   bash infonce_embed_bash.sh /path/to/graph_root /path/to/out_root [relative_bias]
#
# GRAPH_ROOT is scanned non-recursively: only files directly inside it are considered.
#
# Optional:
#   relative_bias
#     Kept for interface compatibility with the sigmoid script.
#     For InfoNCE, a global scalar bias is mathematically inert.
#
# This version:
#   1) does NOT use manifests,
#   2) scans only GRAPH_ROOT itself for graph_n_*_k_*_N_*_seed_*.npz
#      (and legacy .npy if no .npz twin exists),
#   3) parses n, k, N, seed from the filename,
#   4) saves outputs directly under OUT_ROOT/<graph_stem>/d_<d>,
#   5) launches exactly one Slurm array task per graph,
#   6) each array task loops over all d values exactly once.
# ============================================================

# ------------------------
# Embedding dimensions
# ------------------------
d_values=(2 3 4 5 6 7 8 9 10 11 12 13 14 25 50 100)

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
RELATIVE_BIAS=0

MAX_CONCURRENT=20

GRAPH_ROOT="${1:-${GRAPH_ROOT:-/home/kirilb/orcd/scratch/TopK/SyntheticGraphsSmall}}"
BASE_OUT_ROOT="${2:-${OUT_ROOT:-/home/kirilb/orcd/scratch/TopK/SyntheticEmbeddingsSmall/infonce_${INITIALIZATION}_initialization}}"
OUT_ROOT="${BASE_OUT_ROOT}"
if [ "${RELATIVE_BIAS+x}" = x ]; then
    OUT_ROOT="${BASE_OUT_ROOT}/b_rel_${RELATIVE_BIAS}"
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT_PATH="$(readlink -f "$0")"
PYTHON_SCRIPT="infonce_embed.py"

list_graphs_unique() {
    local graph_root="$1"

    while IFS= read -r path; do
        echo "$path"
    done < <(find "$graph_root" -maxdepth 1 -type f -name 'graph_n_*_k_*_N_*_seed_*.npz' | sort)

    while IFS= read -r path; do
        if [ ! -f "${path%.npy}.npz" ]; then
            echo "$path"
        fi
    done < <(find "$graph_root" -maxdepth 1 -type f -name 'graph_n_*_k_*_N_*_seed_*.npy' | sort)
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
    echo "SCRIPT_PATH=${SCRIPT_PATH}"
    echo "PYTHON_SCRIPT=${PYTHON_SCRIPT}"
    echo "GRAPH_ROOT=${GRAPH_ROOT}"
    echo "OUT_ROOT=${OUT_ROOT}"
    echo "TEMPERATURE=${TEMPERATURE}"
    if [ -n "${RELATIVE_BIAS}" ]; then
        echo "RELATIVE_BIAS=${RELATIVE_BIAS} (fixed; inert for InfoNCE)"
    else
        echo "RELATIVE_BIAS=<trainable bias; inert for InfoNCE>"
    fi
    echo "TOTAL_GRAPHS=${TOTAL_GRAPHS}"
    echo "NUM_D_VALUES=${NUM_D_VALUES}"
    echo "TOTAL_RUNS=${TOTAL_RUNS}"
    echo "Submitting one array task per graph."

    sbatch \
        --array=0-$((TOTAL_GRAPHS - 1))%${MAX_CONCURRENT} \
        --export=ALL,GRAPH_ROOT="${GRAPH_ROOT}",OUT_ROOT="${OUT_ROOT}",RELATIVE_BIAS="${RELATIVE_BIAS}",TEMPERATURE="${TEMPERATURE}" \
        "${SCRIPT_PATH}"
    exit $?
fi

module load miniforge
CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate GPUenv

if [ ! -f "${PYTHON_SCRIPT}" ]; then
    echo "Python script not found: ${PYTHON_SCRIPT}"
    exit 1
fi

graph_idx=${SLURM_ARRAY_TASK_ID}

if [ "${graph_idx}" -lt 0 ] || [ "${graph_idx}" -ge "${TOTAL_GRAPHS}" ]; then
    echo "Invalid array index ${graph_idx}; TOTAL_GRAPHS=${TOTAL_GRAPHS}."
    exit 1
fi

graph_path="${GRAPH_PATHS[$graph_idx]}"
metadata="$(parse_graph_metadata "${graph_path}")"
if [ $? -ne 0 ] || [ -z "${metadata}" ]; then
    echo "Could not parse metadata from graph file: ${graph_path}"
    exit 1
fi

read -r n k N seed <<< "${metadata}"

echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "GRAPH_ROOT=${GRAPH_ROOT}"
echo "OUT_ROOT=${OUT_ROOT}"
echo "TEMPERATURE=${TEMPERATURE}"
if [ -n "${RELATIVE_BIAS}" ]; then
    echo "RELATIVE_BIAS=${RELATIVE_BIAS} (fixed; inert for InfoNCE)"
else
    echo "RELATIVE_BIAS=<trainable bias; inert for InfoNCE>"
fi
echo "NUM_D_VALUES=${NUM_D_VALUES}"
echo "--------------------------------------------------"
echo "graph_idx=${graph_idx} n=${n} k=${k} N=${N} seed=${seed}"
echo "graph_path=${graph_path}"

for d in "${d_values[@]}"; do
    run_dir="$(build_run_dir "${graph_path}" "${d}")"
    mkdir -p "${run_dir}"

    echo "--------------------------------------------------"
    echo "n=${n} k=${k} N=${N} d=${d} seed=${seed}"
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

    if [ -n "${RELATIVE_BIAS}" ]; then
        cmd+=(--relative_bias "${RELATIVE_BIAS}")
    fi

    printf 'Running command:\n'
    printf ' %q' "${cmd[@]}"
    printf '\n'

    "${cmd[@]}"
done
