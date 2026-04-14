#!/bin/bash
#SBATCH --job-name=spectral_bounds
#SBATCH --output=logs/spectral_bounds_%A_%a.out
#SBATCH --error=logs/spectral_bounds_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=mit_normal

# ============================================================
# Solve the spectral-bound CVXPY problem for every graph stored
# directly in GRAPH_ROOT, where the graph files are named
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
# This launcher submits ONE Slurm ARRAY JOB with one array index
# per graph, and at most RELEASE_AT_MOST tasks running at once.
#
# Usage:
#   bash spectral_bounds_bash.sh /path/to/graph_folder
#
# or:
#   GRAPH_ROOT=/path/to/graph_folder bash spectral_bounds_bash.sh
#
# Optional environment variables:
#   RELEASE_AT_MOST=20
#   OVERWRITE=0
#   SOLVER=auto
#   MAX_ENTRIES=
#   SCS_EPS=1e-5
#   SCS_MAX_ITERS=20000
#   VERBOSE=0
# ============================================================

GRAPH_ROOT_RAW="/home/kirilb/orcd/scratch/TopK/SyntheticGraphsSmall"
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

mkdir -p "$LOG_DIR"

if [ -z "$GRAPH_ROOT_RAW" ]; then
    echo "GRAPH_ROOT was not provided."
    echo "Usage: bash $(basename "$0") /path/to/graph_folder"
    echo "   or: GRAPH_ROOT=/path/to/graph_folder bash $(basename "$0")"
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

    echo "SCRIPT_PATH=$SCRIPT_PATH"
    echo "PYTHON_SCRIPT=$PYTHON_SCRIPT"
    echo "GRAPH_ROOT=$GRAPH_ROOT"
    echo "OUTPUT_DIR=$OUTPUT_DIR"
    echo "GRAPH_LIST_FILE=$GRAPH_LIST_FILE"
    echo "TOTAL_GRAPHS=$TOTAL_GRAPHS"
    echo "RELEASE_AT_MOST=$RELEASE_AT_MOST"
    echo "OVERWRITE=$OVERWRITE"
    echo "SOLVER=$SOLVER"
    echo "Submitting one array job with $TOTAL_GRAPHS tasks."

    sbatch \
        --array=0-$((TOTAL_GRAPHS - 1))%${RELEASE_AT_MOST} \
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
