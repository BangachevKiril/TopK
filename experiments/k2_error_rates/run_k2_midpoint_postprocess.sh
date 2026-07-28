#!/bin/bash
#SBATCH --job-name=topk_k2_midpoint_errors
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/tmp/topk_k2_midpoint_errors_%j.out
#SBATCH --error=/tmp/topk_k2_midpoint_errors_%j.err

set -eo pipefail

: "${EMBEDDINGS_ROOT:?EMBEDDINGS_ROOT must be set}"
: "${GRAPH_ROOT:?GRAPH_ROOT must be set}"
: "${PLOT_DIR:?PLOT_DIR must be set}"
: "${MIDPOINT_SCRIPT:?MIDPOINT_SCRIPT must be set}"

module load miniforge
conda_base=$(conda info --base)
source "${conda_base}/etc/profile.d/conda.sh"
conda activate GPUenv

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

threshold_mode="${THRESHOLD_MODE:-class-means}"
command=(
  python "${MIDPOINT_SCRIPT}"
  --embeddings-root "${EMBEDDINGS_ROOT}"
  --graph-root "${GRAPH_ROOT}"
  --plot-dir "${PLOT_DIR}"
  --threshold-mode "${threshold_mode}"
)
if [[ -n "${OUTPUT_PREFIX:-}" ]]; then
  command+=(--output-prefix "${OUTPUT_PREFIX}")
fi

"${command[@]}"
