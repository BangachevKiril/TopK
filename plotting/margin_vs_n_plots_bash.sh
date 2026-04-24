#!/bin/bash
#SBATCH -J plot_margin
#SBATCH -p mit_normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

# ============================================================
# Plot largest positive saved margin versus log n on one shared plot.
# InfoNCE uses solid lines and SigLIP uses dashed lines.
#
# This script scans both embedding roots recursively for folders like
#   graph_n_<n>_k_<k>_N_<N>_seed_<seed>/d_<d>
# and, for each matching run directory, takes the largest saved
# margin across the .npz checkpoints inside that d_<d> folder.
# ============================================================

INFO_NCE_ROOT="/home/kirilb/orcd/scratch/TopK_data/SyntheticEmbeddings/infonce_random_initialization/b_rel_0"
SIGLIP_ROOT="/home/kirilb/orcd/scratch/TopK_data/SyntheticEmbeddings/sigmoid_random_initialization/b_rel_0"
OUTPUT_DIR="/home/kirilb/orcd/scratch/TopK_data/Plots"
OUTPUT_NAME="max_margin_vs_logn_k2_p1_seed0"

n_values=(20 40 60 80 100 120 140 160 180 200 220 240)
d_values=(5 15 20 25 30)

K_VALUE=2
P_VALUE=1.0
SEED_VALUE=0
LOG_BASE="e"
SIGLIP_LABEL="SigLIP"

mkdir -p logs
mkdir -p "${OUTPUT_DIR}"

module load miniforge
CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate GPUenv

cmd=(
  python margin_vs_n_plots.py
  --infonce-root "${INFO_NCE_ROOT}"
  --siglip-root "${SIGLIP_ROOT}"
  --output-dir "${OUTPUT_DIR}"
  --output-name "${OUTPUT_NAME}"
  --n-values "${n_values[*]}"
  --d-values "${d_values[*]}"
  --k "${K_VALUE}"
  --p "${P_VALUE}"
  --log-base "${LOG_BASE}"
  --siglip-label "${SIGLIP_LABEL}"
)

if [ -n "${SEED_VALUE}" ]; then
  cmd+=(--seed "${SEED_VALUE}")
fi

printf 'Running command:
'
printf ' %q' "${cmd[@]}"
printf '
'

"${cmd[@]}"
