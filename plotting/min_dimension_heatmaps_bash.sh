#!/bin/bash
#SBATCH --job-name=plot_min_success_dim_heatmaps
#SBATCH --output=logs/plot_min_success_dim_heatmaps_%j.out
#SBATCH --error=logs/plot_min_success_dim_heatmaps_%j.err
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --partition=mit_preemptable

# ============================================================
# Create side-by-side heatmaps comparing sigmoid and InfoNCE.
# For each requested p, write one PDF whose cell (n,k) stores the
# minimal dimension d such that N=floor(p * binom(n,k)) achieved
# positive margin.
#
# You can either edit the *_RAW variables below directly, or pass:
#   bash min_dimension_heatmaps_bash.sh \
#       /path/to/sigmoid_root \
#       /path/to/infonce_root \
#       "0.1,0.25,0.5" \
#       [plot_root]
#
# Optional environment variables:
#   SUCCESS_MODE  best | final      default: best
#   SEED_REDUCE   any  | all        default: any
#   MIN_MARGIN    success threshold default: 0.0
#   FIGWIDTH      default: 14
#   FIGHEIGHT     default: 6
#   ANNOTATE      0 or 1            default: 1
#   OVERWRITE     0 or 1            default: 0
# ============================================================

DEFAULT_SIGMOID_ROOT_RAW="/home/kirilb/orcd/scratch/TopK/SyntheticEmbeddingsSmall/sigmoid_random_initialization/b_rel_0"
DEFAULT_INFONCE_ROOT_RAW="/home/kirilb/orcd/scratch/TopK/SyntheticEmbeddingsSmall/infonce_random_initialization/b_rel_0"
DEFAULT_P_VALUES_RAW=(0.015625 0.03125 0.0625 0.125 0.25 0.5 1.0)
DEFAULT_PLOT_ROOT_RAW="/home/kirilb/orcd/scratch/TopK/SyntheticEmbeddingsSmall/min_success_dim_heatmaps"

SIGMOID_ROOT_RAW="${1:-$DEFAULT_SIGMOID_ROOT_RAW}"
INFONCE_ROOT_RAW="${2:-$DEFAULT_INFONCE_ROOT_RAW}"
PLOT_ROOT_RAW="${4:-$DEFAULT_PLOT_ROOT_RAW}"

if [ $# -ge 3 ] && [ -n "$3" ]; then
    IFS=',' read -r -a P_VALUES_RAW <<< "$3"
else
    P_VALUES_RAW=("${DEFAULT_P_VALUES_RAW[@]}")
fi

SUCCESS_MODE="${SUCCESS_MODE:-best}"
SEED_REDUCE="${SEED_REDUCE:-any}"
MIN_MARGIN="${MIN_MARGIN:-0.0}"
FIGWIDTH="${FIGWIDTH:-14}"
FIGHEIGHT="${FIGHEIGHT:-6}"
ANNOTATE="${ANNOTATE:-1}"
OVERWRITE="${OVERWRITE:-0}"
LOG_DIR="logs"

mkdir -p "$LOG_DIR"

if [ ! -d "$SIGMOID_ROOT_RAW" ]; then
    echo "Sigmoid root not found: $SIGMOID_ROOT_RAW"
    exit 1
fi

if [ ! -d "$INFONCE_ROOT_RAW" ]; then
    echo "InfoNCE root not found: $INFONCE_ROOT_RAW"
    exit 1
fi

if [ "${#P_VALUES_RAW[@]}" -eq 0 ]; then
    echo "P_VALUES_RAW is empty."
    exit 1
fi

if [ "$SUCCESS_MODE" != "best" ] && [ "$SUCCESS_MODE" != "final" ]; then
    echo "SUCCESS_MODE must be 'best' or 'final', got: $SUCCESS_MODE"
    exit 1
fi

if [ "$SEED_REDUCE" != "any" ] && [ "$SEED_REDUCE" != "all" ]; then
    echo "SEED_REDUCE must be 'any' or 'all', got: $SEED_REDUCE"
    exit 1
fi

if ! [[ "$ANNOTATE" =~ ^[0-9]+$ ]] || { [ "$ANNOTATE" -ne 0 ] && [ "$ANNOTATE" -ne 1 ]; }; then
    echo "ANNOTATE must be 0 or 1, got: $ANNOTATE"
    exit 1
fi

if ! [[ "$OVERWRITE" =~ ^[0-9]+$ ]] || { [ "$OVERWRITE" -ne 0 ] && [ "$OVERWRITE" -ne 1 ]; }; then
    echo "OVERWRITE must be 0 or 1, got: $OVERWRITE"
    exit 1
fi

SIGMOID_ROOT="$(cd "$SIGMOID_ROOT_RAW" && pwd -P)"
INFONCE_ROOT="$(cd "$INFONCE_ROOT_RAW" && pwd -P)"
mkdir -p "$PLOT_ROOT_RAW"
PLOT_ROOT="$(cd "$PLOT_ROOT_RAW" && pwd -P)"
P_VALUES_CSV="$(IFS=,; echo "${P_VALUES_RAW[*]}")"

if [ -z "${SLURM_JOB_ID:-}" ]; then
    echo "SIGMOID_ROOT=$SIGMOID_ROOT"
    echo "INFONCE_ROOT=$INFONCE_ROOT"
    echo "P_VALUES_CSV=$P_VALUES_CSV"
    echo "PLOT_ROOT=$PLOT_ROOT"
    echo "SUCCESS_MODE=$SUCCESS_MODE"
    echo "SEED_REDUCE=$SEED_REDUCE"
    echo "MIN_MARGIN=$MIN_MARGIN"
    echo "FIGWIDTH=$FIGWIDTH"
    echo "FIGHEIGHT=$FIGHEIGHT"
    echo "ANNOTATE=$ANNOTATE"
    echo "OVERWRITE=$OVERWRITE"
    echo "Submitting one plotting job."

    sbatch \
        --export=ALL,SIGMOID_ROOT="$SIGMOID_ROOT",INFONCE_ROOT="$INFONCE_ROOT",P_VALUES_CSV="$P_VALUES_CSV",PLOT_ROOT="$PLOT_ROOT",SUCCESS_MODE="$SUCCESS_MODE",SEED_REDUCE="$SEED_REDUCE",MIN_MARGIN="$MIN_MARGIN",FIGWIDTH="$FIGWIDTH",FIGHEIGHT="$FIGHEIGHT",ANNOTATE="$ANNOTATE",OVERWRITE="$OVERWRITE" \
        "$0"
    exit $?
fi

module load miniforge
CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate GPUenv

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    SCRIPT_DIR="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd -P)"
fi
PYTHON_SCRIPT="min_dimension_heatmaps.py"

if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Python script not found: $PYTHON_SCRIPT"
    exit 1
fi

IFS=',' read -r -a P_VALUES_ARR <<< "${P_VALUES_CSV:-}"
if [ "${#P_VALUES_ARR[@]}" -eq 0 ]; then
    echo "Failed to parse any p values from: ${P_VALUES_CSV:-}"
    exit 1
fi

cmd=(
    python "$PYTHON_SCRIPT"
    --sigmoid_root "$SIGMOID_ROOT"
    --infonce_root "$INFONCE_ROOT"
    --output_root "$PLOT_ROOT"
    --success_mode "$SUCCESS_MODE"
    --seed_reduce "$SEED_REDUCE"
    --min_margin "$MIN_MARGIN"
    --figwidth "$FIGWIDTH"
    --figheight "$FIGHEIGHT"
    --p_values
)

for p in "${P_VALUES_ARR[@]}"; do
    p="$(echo "$p" | xargs)"
    if [ -n "$p" ]; then
        cmd+=("$p")
    fi
done

if [ "$ANNOTATE" -eq 1 ]; then
    cmd+=(--annotate)
fi

if [ "$OVERWRITE" -eq 1 ]; then
    cmd+=(--overwrite)
fi

echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SIGMOID_ROOT=$SIGMOID_ROOT"
echo "INFONCE_ROOT=$INFONCE_ROOT"
echo "PLOT_ROOT=$PLOT_ROOT"
echo "P_VALUES_CSV=$P_VALUES_CSV"
printf 'Running command:\n'
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"
