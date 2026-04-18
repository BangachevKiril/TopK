#!/bin/bash
#SBATCH --job-name=plot_margin_compare_pdfs
#SBATCH --output=logs/plot_margin_compare_pdfs_%A_%a.out
#SBATCH --error=logs/plot_margin_compare_pdfs_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --partition=mit_preemptable

# ============================================================
# Create one side-by-side PDF comparison plot per graph,
# with sigmoid on the left and InfoNCE on the right.
#
# Usage:
#   bash plot_margin_compare_pdfs_bash.sh /path/to/sigmoid_embedding_root /path/to/infonce_embedding_root [plot_root]
#
# Expected embedding layouts:
#
#   <sigmoid_root>/<optional_relative_subdir>/graph_n_<n>_k_<k>_N_<N>_seed_<seed>/d_<dim>
#   <infonce_root>/<optional_relative_subdir>/graph_n_<n>_k_<k>_N_<N>_seed_<seed>/d_<dim>
#
# Matching is done by the graph stem
#
#   graph_n_<n>_k_<k>_N_<N>_seed_<seed>
#
# rather than by identical relative paths.
#
# Optional environment variables:
#   SPECTRAL_ROOT       Explicit location of spectral .npz files.
#   MAX_CONCURRENT      Maximum simultaneously running array tasks. Default: 20
#   WHICH_JOBS_RAW      "-1" for all tasks or comma-separated task indices.
#   OVERWRITE           0 or 1. If 0, skip PDFs that already exist.
#   SPECTRAL_MODE       margin_upper_bound_sqrt_nN | objective | custom_multiplier
#   SPECTRAL_MULTIPLIER Multiplier used only when SPECTRAL_MODE=custom_multiplier
#   LEGEND_OUTSIDE      0 or 1
#   YMIN_FLOOR          Bottom shared y-limit floor. Default: 0.0
# ============================================================

SIGMOID_ROOT_RAW="/home/kirilb/orcd/scratch/TopK/SyntheticEmbeddingsSmall/sigmoid_random_initialization"
INFONCE_ROOT_RAW="/home/kirilb/orcd/scratch/TopK/SyntheticEmbeddingsSmall/infonce_random_initialization"
PLOT_ROOT_RAW="/home/kirilb/orcd/scratch/TopK/SyntheticEmbeddingsSmall/margin_compare_plots"
SPECTRAL_ROOT=""
MAX_CONCURRENT="60"
WHICH_JOBS_RAW="-1"
OVERWRITE="0"
SPECTRAL_MODE="margin_upper_bound_sqrt_nN"
SPECTRAL_MULTIPLIER="1.0"
LEGEND_OUTSIDE="0"
YMIN_FLOOR="0.0"
LOG_DIR="logs"

list_run_roots() {
    local embedding_root="$1"
    find "$embedding_root" -type d -name 'graph_n_*_k_*_N_*_seed_*' | sort | while IFS= read -r path; do
        if find "$path" -maxdepth 1 -mindepth 1 -type d -name 'd_*' | grep -q .; then
            echo "$path"
        fi
    done
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
    local total_jobs="$2"

    if [ "$which_jobs_csv" = "-1" ]; then
        return 0
    fi

    local token
    IFS=',' read -r -a _which_jobs_arr <<< "$which_jobs_csv"
    for token in "${_which_jobs_arr[@]}"; do
        if [ "$token" -lt 0 ] || [ "$token" -ge "$total_jobs" ]; then
            echo "WHICH_JOBS contains out-of-range index $token for TOTAL_JOBS=$total_jobs"
            return 1
        fi
    done

    return 0
}

check_no_duplicate_stems() {
    local map_file="$1"
    local label="$2"
    awk -F'\t' -v label="$label" '
        {
            if (++seen[$1] > 1) {
                printf("Duplicate graph stem in %s: %s\n", label, $1) > "/dev/stderr"
                bad = 1
            }
        }
        END { exit bad }
    ' "$map_file"
}

mkdir -p "$LOG_DIR"

if [ ! -d "$SIGMOID_ROOT_RAW" ]; then
    echo "Sigmoid embedding root not found: $SIGMOID_ROOT_RAW"
    exit 1
fi

if [ ! -d "$INFONCE_ROOT_RAW" ]; then
    echo "InfoNCE embedding root not found: $INFONCE_ROOT_RAW"
    exit 1
fi

SIGMOID_ROOT="$(cd "$SIGMOID_ROOT_RAW" && pwd -P)"
INFONCE_ROOT="$(cd "$INFONCE_ROOT_RAW" && pwd -P)"
mkdir -p "$PLOT_ROOT_RAW"
PLOT_ROOT="$(cd "$PLOT_ROOT_RAW" && pwd -P)"

if ! [[ "$MAX_CONCURRENT" =~ ^[0-9]+$ ]] || [ "$MAX_CONCURRENT" -le 0 ]; then
    echo "MAX_CONCURRENT must be a positive integer, got: $MAX_CONCURRENT"
    exit 1
fi

if ! [[ "$OVERWRITE" =~ ^[0-9]+$ ]] || { [ "$OVERWRITE" -ne 0 ] && [ "$OVERWRITE" -ne 1 ]; }; then
    echo "OVERWRITE must be 0 or 1, got: $OVERWRITE"
    exit 1
fi

if ! [[ "$LEGEND_OUTSIDE" =~ ^[0-9]+$ ]] || { [ "$LEGEND_OUTSIDE" -ne 0 ] && [ "$LEGEND_OUTSIDE" -ne 1 ]; }; then
    echo "LEGEND_OUTSIDE must be 0 or 1, got: $LEGEND_OUTSIDE"
    exit 1
fi

WHICH_JOBS_NORMALIZED="$(normalize_job_list "$WHICH_JOBS_RAW")"
if [ $? -ne 0 ]; then
    exit 1
fi

if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    RUN_TAG="$(date +%Y%m%d_%H%M%S)_$$"
    SIGMOID_MAP_FILE="${PLOT_ROOT}/sigmoid_run_roots_${RUN_TAG}.tsv"
    INFONCE_MAP_FILE="${PLOT_ROOT}/infonce_run_roots_${RUN_TAG}.tsv"
    PAIR_LIST_FILE="${PLOT_ROOT}/compare_run_roots_${RUN_TAG}.tsv"

    list_run_roots "$SIGMOID_ROOT" | while IFS= read -r path; do
        stem="$(basename "$path")"
        rel_path="${path#${SIGMOID_ROOT}/}"
        printf '%s\t%s\t%s\n' "$stem" "$path" "$rel_path"
    done | sort -t $'\t' -k1,1 > "$SIGMOID_MAP_FILE"

    list_run_roots "$INFONCE_ROOT" | while IFS= read -r path; do
        stem="$(basename "$path")"
        printf '%s\t%s\n' "$stem" "$path"
    done | sort -t $'\t' -k1,1 > "$INFONCE_MAP_FILE"

    if ! check_no_duplicate_stems "$SIGMOID_MAP_FILE" "SIGMOID_ROOT=$SIGMOID_ROOT"; then
        rm -f "$SIGMOID_MAP_FILE" "$INFONCE_MAP_FILE" "$PAIR_LIST_FILE"
        exit 1
    fi

    if ! check_no_duplicate_stems "$INFONCE_MAP_FILE" "INFONCE_ROOT=$INFONCE_ROOT"; then
        rm -f "$SIGMOID_MAP_FILE" "$INFONCE_MAP_FILE" "$PAIR_LIST_FILE"
        exit 1
    fi

    join -t $'\t' -1 1 -2 1 "$SIGMOID_MAP_FILE" "$INFONCE_MAP_FILE" | \
        awk -F'\t' '{ printf "%s\t%s\t%s\t%s\n", $1, $2, $3, $4 }' > "$PAIR_LIST_FILE"

    TOTAL_PAIRS=$(wc -l < "$PAIR_LIST_FILE")
    TOTAL_PAIRS="${TOTAL_PAIRS//[[:space:]]/}"

    if [ "$TOTAL_PAIRS" -eq 0 ]; then
        echo "No common graph run roots were found between:"
        echo "  SIGMOID_ROOT=$SIGMOID_ROOT"
        echo "  INFONCE_ROOT=$INFONCE_ROOT"
        echo "Matching is done by graph stem graph_n_<n>_k_<k>_N_<N>_seed_<seed>."
        rm -f "$SIGMOID_MAP_FILE" "$INFONCE_MAP_FILE" "$PAIR_LIST_FILE"
        exit 1
    fi

    if ! validate_job_list_against_total "$WHICH_JOBS_NORMALIZED" "$TOTAL_PAIRS"; then
        rm -f "$SIGMOID_MAP_FILE" "$INFONCE_MAP_FILE" "$PAIR_LIST_FILE"
        exit 1
    fi

    if [ "$WHICH_JOBS_NORMALIZED" = "-1" ]; then
        ARRAY_SPEC="0-$((TOTAL_PAIRS - 1))%${MAX_CONCURRENT}"
    else
        ARRAY_SPEC="${WHICH_JOBS_NORMALIZED}%${MAX_CONCURRENT}"
    fi

    echo "SIGMOID_ROOT=$SIGMOID_ROOT"
    echo "INFONCE_ROOT=$INFONCE_ROOT"
    echo "PLOT_ROOT=$PLOT_ROOT"
    echo "SPECTRAL_ROOT=${SPECTRAL_ROOT:-<infer from config graph_path>}"
    echo "YMIN_FLOOR=$YMIN_FLOOR"
    echo "SIGMOID_MAP_FILE=$SIGMOID_MAP_FILE"
    echo "INFONCE_MAP_FILE=$INFONCE_MAP_FILE"
    echo "PAIR_LIST_FILE=$PAIR_LIST_FILE"
    echo "TOTAL_PAIRS=$TOTAL_PAIRS"
    echo "WHICH_JOBS=$WHICH_JOBS_NORMALIZED"
    echo "ARRAY_SPEC=$ARRAY_SPEC"
    echo "Submitting one array task per common graph stem."

    sbatch \
        --array="$ARRAY_SPEC" \
        --export=ALL,SIGMOID_ROOT="$SIGMOID_ROOT",INFONCE_ROOT="$INFONCE_ROOT",PLOT_ROOT="$PLOT_ROOT",SPECTRAL_ROOT="$SPECTRAL_ROOT",PAIR_LIST_FILE="$PAIR_LIST_FILE",TOTAL_PAIRS="$TOTAL_PAIRS",OVERWRITE="$OVERWRITE",SPECTRAL_MODE="$SPECTRAL_MODE",SPECTRAL_MULTIPLIER="$SPECTRAL_MULTIPLIER",LEGEND_OUTSIDE="$LEGEND_OUTSIDE",YMIN_FLOOR="$YMIN_FLOOR" \
        "$0"
    exit $?
fi

module load miniforge
CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate GPUenv

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
PYTHON_SCRIPT="plot_margins.py"

if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Python script not found: $PYTHON_SCRIPT"
    echo "Place plot_margins.py in the same directory from which you call sbatch."
    exit 1
fi

if [ -z "${PAIR_LIST_FILE:-}" ] || [ ! -f "$PAIR_LIST_FILE" ]; then
    echo "Missing or unreadable PAIR_LIST_FILE: ${PAIR_LIST_FILE:-}"
    exit 1
fi

if [ -z "${TOTAL_PAIRS:-}" ]; then
    echo "Missing TOTAL_PAIRS in environment."
    exit 1
fi

task_id=${SLURM_ARRAY_TASK_ID}
if [ "$task_id" -lt 0 ] || [ "$task_id" -ge "$TOTAL_PAIRS" ]; then
    echo "Task index out of range: task_id=$task_id TOTAL_PAIRS=$TOTAL_PAIRS"
    exit 1
fi

line="$(sed -n "$((task_id + 1))p" "$PAIR_LIST_FILE")"
if [ -z "$line" ]; then
    echo "Failed to recover pair for task_id=$task_id from $PAIR_LIST_FILE"
    exit 1
fi

IFS=$'\t' read -r stem sigmoid_run_root rel_path infonce_run_root <<< "$line"
if [ -z "$stem" ] || [ -z "$sigmoid_run_root" ] || [ -z "$rel_path" ] || [ -z "$infonce_run_root" ]; then
    echo "Malformed pair-list entry: $line"
    exit 1
fi

pdf_path="${PLOT_ROOT}/${rel_path}.pdf"
mkdir -p "$(dirname "$pdf_path")"

if [ "$OVERWRITE" -eq 0 ] && [ -f "$pdf_path" ]; then
    echo "PDF already exists, skipping: $pdf_path"
    exit 0
fi

cmd=(
    python "$PYTHON_SCRIPT"
    --sigmoid_run_root "$sigmoid_run_root"
    --infonce_run_root "$infonce_run_root"
    --output_pdf "$pdf_path"
    --spectral_mode "$SPECTRAL_MODE"
    --spectral_multiplier "$SPECTRAL_MULTIPLIER"
    --ymin_floor "$YMIN_FLOOR"
)

if [ -n "$SPECTRAL_ROOT" ]; then
    cmd+=(--spectral_root "$SPECTRAL_ROOT")
fi

if [ "$LEGEND_OUTSIDE" -eq 1 ]; then
    cmd+=(--legend_outside)
fi

echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID:-}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "STEM=${stem}"
echo "SIGMOID_RUN_ROOT=${sigmoid_run_root}"
echo "INFONCE_RUN_ROOT=${infonce_run_root}"
echo "PDF_PATH=${pdf_path}"
printf 'Running command:\n'
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"
