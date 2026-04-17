#!/bin/bash
#SBATCH --job-name=spectral_bounds
#SBATCH --output=logs/spectral_bounds_%A_%a.out
#SBATCH --error=logs/spectral_bounds_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --partition=mit_preemptable

# ============================================================
# Solve the spectral-bound CVXPY problem for selected graphs
# stored directly in GRAPH_ROOT, where graph files are named
#
#   graph_n_<n>_k_<k>_N_<N>_seed_<seed>.npz
#
# For each graph file
#
#   GRAPH_ROOT/<graph_filename>.npz
#
# the Python script writes
#
#   GRAPH_ROOT/spectral_bounds/<graph_filename>.npz
#
# This launcher submits ONE Slurm array job with one array index
# per graph, and at most RELEASE_AT_MOST tasks running at once.
#
# Edit GRAPH_ROOT_RAW and WHICH_JOBS_RAW below.
#
# Use:
#   WHICH_JOBS_RAW="-1"
# to run all jobs.
# ============================================================

GRAPH_ROOT_RAW="/home/kirilb/orcd/scratch/TopK/SyntheticGraphsSmall"
WHICH_JOBS_RAW="19,26,27,28,29,30,31,32,51,54,59,61,63,66,67,78,80,81,83,85,87,88,89,90,91,92,93,94,95,96,97,98,100,109,111,113,115,116,117,118,119,120,121,122,123"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT_PATH="$(readlink -f "$0")"
PYTHON_SCRIPT="spectral_bounds.py"
LOG_DIR="logs"

RELEASE_AT_MOST="${RELEASE_AT_MOST:-${release_at_most:-5}}"
OVERWRITE="${OVERWRITE:-0}"
SOLVER="${SOLVER:-auto}"
MAX_ENTRIES="${MAX_ENTRIES:-}"
SCS_EPS="${SCS_EPS:-1e-5}"
SCS_MAX_ITERS="${SCS_MAX_ITERS:-20000}"
VERBOSE="${VERBOSE:-0}"

list_all_graphs() {
    local graph_root="$1"
    find "$graph_root" -maxdepth 1 -type f -name 'graph_n_*_k_*_N_*_seed_*.npz' | sort
}

normalize_job_list() {
    local raw="$1"

    if [ "$raw" = "-1" ]; then
        echo "-1"
        return 0
    fi

    raw="${raw//,/ }"
    raw="$(echo "$raw" | xargs)"

    if [ -z "$raw" ]; then
        echo "Empty WHICH_JOBS list."
        return 1
    fi

    local out=()
    local tok
    for tok in $raw; do
        if ! [[ "$tok" =~ ^[0-9]+$ ]]; then
            echo "Invalid WHICH_JOBS entry: $tok"
            return 1
        fi
        out+=("$tok")
    done

    printf '%s\n' "${out[@]}" | sort -n | uniq | paste -sd, -
}

validate_job_list_against_total() {
    local which_jobs_csv="$1"
    local total_graphs="$2"

    if [ "$which_jobs_csv" = "-1" ]; then
        return 0
    fi

    local token
    IFS=',' read -r -a _which_jobs_arr <<< "$which_jobs_csv"
    for token in "${_which_jobs_arr[@]}"; do
        if [ "$token" -lt 0 ] || [ "$token" -ge "$total_graphs" ]; then
            echo "WHICH_JOBS contains out-of-range index $token for TOTAL_GRAPHS=$total_graphs"
            return 1
        fi
    done

    return 0
}

mkdir -p "$LOG_DIR"

if [ -z "$GRAPH_ROOT_RAW" ]; then
    echo "GRAPH_ROOT_RAW is empty."
    exit 1
fi

if [ ! -d "$GRAPH_ROOT_RAW" ]; then
    echo "Graph directory not found: $GRAPH_ROOT_RAW"
    exit 1
fi

GRAPH_ROOT="$(cd "$GRAPH_ROOT_RAW" && pwd -P)"
OUTPUT_DIR="${GRAPH_ROOT}/spectral_bounds"

if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Python script not found: $PYTHON_SCRIPT"
    exit 1
fi

if ! [[ "$RELEASE_AT_MOST" =~ ^[0-9]+$ ]] || [ "$RELEASE_AT_MOST" -le 0 ]; then
    echo "RELEASE_AT_MOST must be a positive integer, got: $RELEASE_AT_MOST"
    exit 1
fi

if ! [[ "$OVERWRITE" =~ ^[0-9]+$ ]] || { [ "$OVERWRITE" -ne 0 ] && [ "$OVERWRITE" -ne 1 ]; }; then
    echo "OVERWRITE must be 0 or 1, got: $OVERWRITE"
    exit 1
fi

WHICH_JOBS_NORMALIZED="$(normalize_job_list "$WHICH_JOBS_RAW")"
if [ $? -ne 0 ]; then
    exit 1
fi

if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    mkdir -p "$OUTPUT_DIR"

    RUN_TAG="$(date +%Y%m%d_%H%M%S)_$$"
    GRAPH_LIST_FILE="${OUTPUT_DIR}/graph_list_${RUN_TAG}.txt"
    list_all_graphs "$GRAPH_ROOT" > "$GRAPH_LIST_FILE"

    TOTAL_GRAPHS=$(wc -l < "$GRAPH_LIST_FILE")
    TOTAL_GRAPHS="${TOTAL_GRAPHS//[[:space:]]/}"

    if [ "$TOTAL_GRAPHS" -eq 0 ]; then
        echo "No matching graph files found under: $GRAPH_ROOT"
        rm -f "$GRAPH_LIST_FILE"
        exit 1
    fi

    if ! validate_job_list_against_total "$WHICH_JOBS_NORMALIZED" "$TOTAL_GRAPHS"; then
        rm -f "$GRAPH_LIST_FILE"
        exit 1
    fi

    if [ "$WHICH_JOBS_NORMALIZED" = "-1" ]; then
        ARRAY_SPEC="0-$((TOTAL_GRAPHS - 1))%${RELEASE_AT_MOST}"
    else
        ARRAY_SPEC="${WHICH_JOBS_NORMALIZED}%${RELEASE_AT_MOST}"
    fi

    echo "SCRIPT_PATH=$SCRIPT_PATH"
    echo "PYTHON_SCRIPT=$PYTHON_SCRIPT"
    echo "GRAPH_ROOT=$GRAPH_ROOT"
    echo "OUTPUT_DIR=$OUTPUT_DIR"
    echo "GRAPH_LIST_FILE=$GRAPH_LIST_FILE"
    echo "TOTAL_GRAPHS=$TOTAL_GRAPHS"
    echo "RELEASE_AT_MOST=$RELEASE_AT_MOST"
    echo "OVERWRITE=$OVERWRITE"
    echo "SOLVER=$SOLVER"
    echo "WHICH_JOBS=$WHICH_JOBS_NORMALIZED"
    echo "ARRAY_SPEC=$ARRAY_SPEC"
    echo "Submitting one array job."

    sbatch \
        --array="$ARRAY_SPEC" \
        --export=ALL,GRAPH_ROOT="$GRAPH_ROOT",OUTPUT_DIR="$OUTPUT_DIR",GRAPH_LIST_FILE="$GRAPH_LIST_FILE",TOTAL_GRAPHS="$TOTAL_GRAPHS",OVERWRITE="$OVERWRITE",SOLVER="$SOLVER",MAX_ENTRIES="$MAX_ENTRIES",SCS_EPS="$SCS_EPS",SCS_MAX_ITERS="$SCS_MAX_ITERS",VERBOSE="$VERBOSE" \
        "$SCRIPT_PATH"
    exit $?
fi

module load miniforge
CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate GPUenv

if [ -z "${GRAPH_LIST_FILE:-}" ] || [ ! -f "$GRAPH_LIST_FILE" ]; then
    echo "Missing or unreadable GRAPH_LIST_FILE: ${GRAPH_LIST_FILE:-}"
    exit 1
fi

if [ -z "${TOTAL_GRAPHS:-}" ]; then
    echo "Missing TOTAL_GRAPHS in environment."
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

task_id=${SLURM_ARRAY_TASK_ID}
if [ "$task_id" -lt 0 ] || [ "$task_id" -ge "$TOTAL_GRAPHS" ]; then
    echo "Task index out of range: task_id=$task_id TOTAL_GRAPHS=$TOTAL_GRAPHS"
    exit 1
fi

graph_path="$(sed -n "$((task_id + 1))p" "$GRAPH_LIST_FILE")"
if [ -z "$graph_path" ]; then
    echo "Failed to recover graph path for task_id=$task_id from $GRAPH_LIST_FILE"
    exit 1
fi

output_path="${OUTPUT_DIR}/$(basename "$graph_path")"

if [ "$OVERWRITE" -eq 0 ] && [ -f "$output_path" ]; then
    echo "Output already exists, skipping: $output_path"
    exit 0
fi

echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID:-}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "GRAPH_ROOT=${GRAPH_ROOT}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "GRAPH_LIST_FILE=${GRAPH_LIST_FILE}"
echo "GRAPH_PATH=${graph_path}"
echo "OUTPUT_PATH=${output_path}"
echo "SOLVER=${SOLVER}"

cmd=(
    python "$PYTHON_SCRIPT"
    --graph_path "$graph_path"
    --output_path "$output_path"
    --solver "$SOLVER"
    --scs_eps "$SCS_EPS"
    --scs_max_iters "$SCS_MAX_ITERS"
)

if [ "$OVERWRITE" -eq 1 ]; then
    cmd+=(--overwrite)
fi

if [ -n "$MAX_ENTRIES" ]; then
    cmd+=(--max_entries "$MAX_ENTRIES")
fi

if [ "$VERBOSE" -eq 1 ]; then
    cmd+=(--verbose)
fi

printf 'Running command:\n'
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"