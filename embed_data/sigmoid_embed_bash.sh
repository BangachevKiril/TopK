#!/bin/bash
#SBATCH -J synth_embed_saved
#SBATCH -p mit_preemptable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=23:59:00
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err

# ============================================================
# Recursively embed saved synthetic graphs under GRAPH_ROOT,
# but only for graph families matching the requested n, k, p lists.
#
# Usage:
#   bash sigmoid_embed_bash.sh GRAPH_ROOT OUT_ROOT P_VALUES K_VALUES N_VALUES [RELATIVE_BIAS]
#
# Notes:
#   - Graph filenames contain n, k, N, seed but not p.
#   - To filter by p, this script reconstructs the allowed N values using
#       N = floor(p * choose(n, k))
#     exactly as in the graph genseration scripts.
# ============================================================

# ------------------------
# Embedding dimensions
# ------------------------
d_values=(5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30)
p_values=(1.0)
n_values=(20 40 60 80 100 120)
k_values=(3)

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
DEVICE=""
RELATIVE_BIAS="0"

MAX_CONCURRENT=20

GRAPH_ROOT="/home/kirilb/orcd/scratch/TopK_data/SyntheticGraphs"
BASE_OUT_ROOT="/home/kirilb/orcd/scratch/TopK_data/SyntheticEmbeddings/sigmoid_${INITIALIZATION}_initialization"
OUT_ROOT="${BASE_OUT_ROOT}"
if [ "${RELATIVE_BIAS+x}" = x ]; then
    OUT_ROOT="${BASE_OUT_ROOT}/b_rel_${RELATIVE_BIAS}"
fi

P_VALUES_RAW="${p_values[*]}"
K_VALUES_RAW="${k_values[*]}"
N_VALUES_RAW="${n_values[*]}"

SCRIPT_PATH="$(readlink -f "$0")"

choose() {
    local n=$1
    local k=$2
    if [ "$k" -lt 0 ] || [ "$k" -gt "$n" ]; then
        echo 0
        return
    fi
    if [ "$k" -gt $((n - k)) ]; then
        k=$((n - k))
    fi
    local result=1
    local i term
    for ((i=1; i<=k; i++)); do
        term=$(( n - k + i ))
        result=$(( result * term / i ))
    done
    echo "$result"
}

compute_N_from_p() {
    local n=$1
    local k=$2
    local p=$3
    local max_neighborhoods
    max_neighborhoods=$(choose "$n" "$k")

    awk -v p="$p" -v M="$max_neighborhoods" 'BEGIN { printf "%d\n", int(p * M) }'
}

parse_list_to_array() {
    local raw="$1"
    local arr_name="$2"
    local normalized
    local -a parsed
    local -n out_arr="$arr_name"

    normalized="${raw//,/ }"
    read -r -a parsed <<< "$normalized"

    if [ "${#parsed[@]}" -eq 0 ]; then
        echo "Missing required list argument for ${arr_name}." >&2
        return 1
    fi

    out_arr=("${parsed[@]}")
}

contains_value() {
    local needle="$1"
    shift
    local item
    for item in "$@"; do
        if [ "$item" = "$needle" ]; then
            return 0
        fi
    done
    return 1
}

triplet_is_requested() {
    local n="$1"
    local k="$2"
    local N="$3"
    local p computed_N

    if ! contains_value "$n" "${FILTER_N_VALUES[@]}"; then
        return 1
    fi

    if ! contains_value "$k" "${FILTER_K_VALUES[@]}"; then
        return 1
    fi

    for p in "${FILTER_P_VALUES[@]}"; do
        computed_N=$(compute_N_from_p "$n" "$k" "$p")
        if [ "$computed_N" = "$N" ]; then
            return 0
        fi
    done

    return 1
}

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

list_filtered_graphs() {
    local graph_root="$1"
    local graph_path metadata n k N seed

    while IFS= read -r graph_path; do
        metadata="$(parse_graph_metadata "$graph_path")" || continue
        read -r n k N seed <<< "$metadata"

        if triplet_is_requested "$n" "$k" "$N"; then
            echo "$graph_path"
        fi
    done < <(list_graphs_unique "$graph_root" | sort)
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

parse_list_to_array "$P_VALUES_RAW" FILTER_P_VALUES
parse_list_to_array "$K_VALUES_RAW" FILTER_K_VALUES
parse_list_to_array "$N_VALUES_RAW" FILTER_N_VALUES

mkdir -p logs
mkdir -p "${OUT_ROOT}"

mapfile -t ALL_GRAPH_PATHS < <(list_graphs_unique "${GRAPH_ROOT}" | sort)
mapfile -t GRAPH_PATHS < <(list_filtered_graphs "${GRAPH_ROOT}" | sort)

TOTAL_FOUND=${#ALL_GRAPH_PATHS[@]}
TOTAL_GRAPHS=${#GRAPH_PATHS[@]}
NUM_D_VALUES=${#d_values[@]}
TOTAL_RUNS=$((TOTAL_GRAPHS * NUM_D_VALUES))

if [ "${TOTAL_FOUND}" -eq 0 ]; then
    echo "No graph files found under: ${GRAPH_ROOT}"
    echo "Expected filenames like graph_n_<n>_k_<k>_N_<N>_seed_<seed>.npz"
    exit 1
fi

if [ "${TOTAL_GRAPHS}" -eq 0 ]; then
    echo "Found ${TOTAL_FOUND} graph file(s), but none matched the requested n, k, p filters."
    echo "Requested p values: ${FILTER_P_VALUES[*]}"
    echo "Requested k values: ${FILTER_K_VALUES[*]}"
    echo "Requested n values: ${FILTER_N_VALUES[*]}"
    echo "First few graph files actually found:"
    for graph_path in "${ALL_GRAPH_PATHS[@]:0:10}"; do
        echo "  ${graph_path}"
    done
    echo "Recall: filtering by p is done via N = floor(p * choose(n, k))."
    exit 1
fi

if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    echo "SCRIPT_PATH=${SCRIPT_PATH}"
    echo "PYTHON_SCRIPT=sigmoid_embed.py"
    echo "GRAPH_ROOT=${GRAPH_ROOT}"
    echo "OUT_ROOT=${OUT_ROOT}"
    echo "P_VALUES=${FILTER_P_VALUES[*]}"
    echo "K_VALUES=${FILTER_K_VALUES[*]}"
    echo "N_VALUES=${FILTER_N_VALUES[*]}"
    if [ -n "${RELATIVE_BIAS}" ]; then
        echo "RELATIVE_BIAS=${RELATIVE_BIAS} (fixed)"
    else
        echo "RELATIVE_BIAS=<trainable bias>"
    fi
    echo "TOTAL_FOUND=${TOTAL_FOUND}"
    echo "TOTAL_GRAPHS=${TOTAL_GRAPHS}"
    echo "NUM_D_VALUES=${NUM_D_VALUES}"
    echo "TOTAL_RUNS=${TOTAL_RUNS}"
    echo "Submitting one array task per graph."

    sbatch \
        --array=0-$((TOTAL_GRAPHS - 1))%${MAX_CONCURRENT} \
        --export=ALL,GRAPH_ROOT="${GRAPH_ROOT}",OUT_ROOT="${OUT_ROOT}",RELATIVE_BIAS="${RELATIVE_BIAS}",P_VALUES_RAW="${P_VALUES_RAW}",K_VALUES_RAW="${K_VALUES_RAW}",N_VALUES_RAW="${N_VALUES_RAW}" \
        "${SCRIPT_PATH}"
    exit $?
fi

module load miniforge
CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate GPUenv

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
echo "P_VALUES=${FILTER_P_VALUES[*]}"
echo "K_VALUES=${FILTER_K_VALUES[*]}"
echo "N_VALUES=${FILTER_N_VALUES[*]}"
if [ -n "${RELATIVE_BIAS}" ]; then
    echo "RELATIVE_BIAS=${RELATIVE_BIAS} (fixed)"
else
    echo "RELATIVE_BIAS=<trainable bias>"
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
        python "sigmoid_embed.py"
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

    if [ -n "${RELATIVE_BIAS}" ]; then
        cmd+=(--relative_bias "${RELATIVE_BIAS}")
    fi

    printf 'Running command:\n'
    printf ' %q' "${cmd[@]}"
    printf '\n'

    "${cmd[@]}"
done
