#!/bin/bash
#SBATCH --job-name=topk_k2_error_plots
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/tmp/topk_k2_error_plots_%j.out
#SBATCH --error=/tmp/topk_k2_error_plots_%j.err

set -eo pipefail

: "${OUTPUT_ROOT:?OUTPUT_ROOT must be set}"
: "${PLOT_DIR:?PLOT_DIR must be set}"
: "${PLOT_SCRIPT:?PLOT_SCRIPT must be set}"

module load miniforge
conda_base=$(conda info --base)
source "${conda_base}/etc/profile.d/conda.sh"
conda activate GPUenv

python "${PLOT_SCRIPT}" \
    --output-root "${OUTPUT_ROOT}" \
    --plot-dir "${PLOT_DIR}" \
    --n-values "20 40 60 80 100 120 140 160 180 200 220 240" \
    --d-values "5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30" \
    --margin-d-values "6 18 30" \
    --seed 0
