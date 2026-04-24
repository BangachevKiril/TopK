#!/bin/bash
#SBATCH --job-name=plot_min_nonzero_margin
#SBATCH --output=logs/plot_min_nonzero_margin_%j.out
#SBATCH --error=logs/plot_min_nonzero_margin_%j.err
#SBATCH --time=00:05:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --partition=mit_preemptable

# Launch from terminal, for example:
#   bash min_nonzero_margin_bash.sh
#
# Edit the variables below to match the experiment you want.
# The output path is explicit, so it will save exactly where OUTPUT_PREFIX points.
#
# When launched from the terminal, this script submits one Slurm job and writes
# stdout/stderr to logs/plot_min_nonzero_margin_<jobid>.out/.err.

DEFAULT_SIGMOID_ROOT_RAW="/home/kirilb/orcd/scratch/TopK_data/SyntheticEmbeddings/sigmoid_random_initialization/b_rel_0"
DEFAULT_INFONCE_ROOT_RAW="/home/kirilb/orcd/scratch/TopK_data/SyntheticEmbeddings/infonce_random_initialization/b_rel_0"
DEFAULT_P_VALUES_RAW="1.0"
DEFAULT_K_VALUES_RAW="2"
DEFAULT_N_VALUES_RAW="20 40 60 80 100 120 140 160 180 200 220 240"
DEFAULT_SEED_VALUES_RAW="0"
DEFAULT_OUTPUT_PREFIX_RAW="/home/kirilb/data/TopK/diagrams/min_nonzero_margin"
TITLE="Minimal d with positive margin"
DPI="220"

SIGMOID_ROOT_RAW="${1:-$DEFAULT_SIGMOID_ROOT_RAW}"
INFONCE_ROOT_RAW="${2:-$DEFAULT_INFONCE_ROOT_RAW}"
OUTPUT_PREFIX_RAW="${3:-$DEFAULT_OUTPUT_PREFIX_RAW}"

P_VALUES_RAW="${P_VALUES_RAW:-$DEFAULT_P_VALUES_RAW}"
K_VALUES_RAW="${K_VALUES_RAW:-$DEFAULT_K_VALUES_RAW}"
N_VALUES_RAW="${N_VALUES_RAW:-$DEFAULT_N_VALUES_RAW}"
SEED_VALUES_RAW="${SEED_VALUES_RAW:-$DEFAULT_SEED_VALUES_RAW}"
TITLE_RAW="${TITLE:-$DEFAULT_TITLE_RAW}"
DPI_RAW="${DPI:-$DEFAULT_DPI_RAW}"
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

SIGMOID_ROOT="$(cd "$SIGMOID_ROOT_RAW" && pwd -P)"
INFONCE_ROOT="$(cd "$INFONCE_ROOT_RAW" && pwd -P)"
OUTPUT_DIR_RAW="$(dirname "$OUTPUT_PREFIX_RAW")"
mkdir -p "$OUTPUT_DIR_RAW"
OUTPUT_DIR="$(cd "$OUTPUT_DIR_RAW" && pwd -P)"
OUTPUT_PREFIX="${OUTPUT_DIR}/$(basename "$OUTPUT_PREFIX_RAW")"

if [ -z "${SLURM_JOB_ID:-}" ]; then
    echo "SIGMOID_ROOT=$SIGMOID_ROOT"
    echo "INFONCE_ROOT=$INFONCE_ROOT"
    echo "OUTPUT_PREFIX=$OUTPUT_PREFIX"
    echo "P_VALUES_RAW=$P_VALUES_RAW"
    echo "K_VALUES_RAW=$K_VALUES_RAW"
    echo "N_VALUES_RAW=$N_VALUES_RAW"
    echo "SEED_VALUES_RAW=$SEED_VALUES_RAW"
    echo "TITLE=$TITLE_RAW"
    echo "DPI=$DPI_RAW"
    echo "Submitting one plotting job."

    sbatch \
        --export=ALL,SIGMOID_ROOT="$SIGMOID_ROOT",INFONCE_ROOT="$INFONCE_ROOT",OUTPUT_PREFIX="$OUTPUT_PREFIX",P_VALUES_RAW="$P_VALUES_RAW",K_VALUES_RAW="$K_VALUES_RAW",N_VALUES_RAW="$N_VALUES_RAW",SEED_VALUES_RAW="$SEED_VALUES_RAW",TITLE="$TITLE_RAW",DPI="$DPI_RAW" \
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

echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SIGMOID_ROOT=$SIGMOID_ROOT"
echo "INFONCE_ROOT=$INFONCE_ROOT"
echo "OUTPUT_PREFIX=$OUTPUT_PREFIX"
echo "P_VALUES_RAW=$P_VALUES_RAW"
echo "K_VALUES_RAW=$K_VALUES_RAW"
echo "N_VALUES_RAW=$N_VALUES_RAW"
echo "SEED_VALUES_RAW=$SEED_VALUES_RAW"
echo "TITLE=$TITLE"
echo "DPI=$DPI"

cmd=(
  python "min_nonzero_margin.py"
  --sigmoid_root "$SIGMOID_ROOT"
  --infonce_root "$INFONCE_ROOT"
  --output_prefix "$OUTPUT_PREFIX"
  --p_values "$P_VALUES_RAW"
  --k_values "$K_VALUES_RAW"
  --n_values "$N_VALUES_RAW"
  --seed_values "$SEED_VALUES_RAW"
  --title "$TITLE"
  --dpi "$DPI"
)

printf 'Running command:\n'
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"
