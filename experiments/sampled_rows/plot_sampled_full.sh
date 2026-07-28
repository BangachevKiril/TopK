#!/bin/bash
#SBATCH --job-name=topk_sampled_full_plot
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/home/kirilb/data/TopK/logs/topk_sampled_full_plot_%j.out
#SBATCH --error=/home/kirilb/data/TopK/logs/topk_sampled_full_plot_%j.err

set -eo pipefail

: "${SAMPLED_OUT_ROOT:?SAMPLED_OUT_ROOT must be set}"
: "${MIN_D_PLOT_SCRIPT:?MIN_D_PLOT_SCRIPT must be set}"
: "${MARGIN_PLOT_SCRIPT:?MARGIN_PLOT_SCRIPT must be set}"

module load miniforge
conda_base=$(conda info --base)
source "${conda_base}/etc/profile.d/conda.sh"
conda activate GPUenv

N_VALUES="100 200 400 800 1600 3200 6400 12800 25600"

python "${MIN_D_PLOT_SCRIPT}" \
    --infonce-root "${SAMPLED_OUT_ROOT}/infonce" \
    --sigmoid-root "${SAMPLED_OUT_ROOT}/sigmoid" \
    --output-prefix "${SAMPLED_OUT_ROOT}/sampled_min_d_n100" \
    --n 100 \
    --k-values "4 5 6" \
    --N-values "${N_VALUES}" \
    --seed 0

python "${MARGIN_PLOT_SCRIPT}" \
    --infonce-root "${SAMPLED_OUT_ROOT}/infonce" \
    --sigmoid-root "${SAMPLED_OUT_ROOT}/sigmoid" \
    --output-prefix "${SAMPLED_OUT_ROOT}/sampled_max_margin_n100" \
    --n 100 \
    --k-values "4 5 6" \
    --N-values "${N_VALUES}" \
    --d-values "10 20 30" \
    --seed 0
