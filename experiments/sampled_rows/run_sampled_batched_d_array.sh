#!/bin/bash
#SBATCH --job-name=topk_sampled_batchd
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --output=/home/kirilb/data/TopK/logs/topk_sampled_batchd_%A_%a.out
#SBATCH --error=/home/kirilb/data/TopK/logs/topk_sampled_batchd_%A_%a.err

set -eo pipefail

REPO_ROOT="${REPO_ROOT:-/home/kirilb/data/TopK}"
: "${SAMPLED_GRAPH_ROOT:?SAMPLED_GRAPH_ROOT must be set}"
: "${SAMPLED_OUT_ROOT:?SAMPLED_OUT_ROOT must be set}"
: "${BATCHED_TRAINER:?BATCHED_TRAINER must be set}"
: "${N_VALUES_RAW:?N_VALUES_RAW must be set}"

k_values=(4 5 6)
loss_values=(infonce sigmoid)
read -r -a n_values <<< "${N_VALUES_RAW//,/ }"

num_N=${#n_values[@]}
runs_per_loss=$(( ${#k_values[@]} * num_N ))
total_runs=$(( ${#loss_values[@]} * runs_per_loss ))
task_id=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID must be set}

if (( task_id < 0 || task_id >= total_runs )); then
    echo "Invalid task id ${task_id}; expected 0..$((total_runs - 1))."
    exit 2
fi

loss_idx=$((task_id / runs_per_loss))
within_loss=$((task_id % runs_per_loss))
k_idx=$((within_loss / num_N))
N_idx=$((within_loss % num_N))

loss_name=${loss_values[$loss_idx]}
k=${k_values[$k_idx]}
N=${n_values[$N_idx]}
n=100
seed=0
graph_stem="graph_n_${n}_k_${k}_N_${N}_seed_${seed}"
graph_path="${SAMPLED_GRAPH_ROOT}/${graph_stem}.npz"
run_root="${SAMPLED_OUT_ROOT}/${loss_name}/${graph_stem}"

if [[ ! -f "${graph_path}" ]]; then
    echo "Missing graph: ${graph_path}"
    exit 3
fi

if [[ -f "${run_root}/COMPLETE" ]]; then
    echo "Already complete: ${run_root}"
    exit 0
fi

module load miniforge
conda_base=$(conda info --base)
source "${conda_base}/etc/profile.d/conda.sh"
conda activate GPUenv
cd "${REPO_ROOT}"

echo "commit=$(git rev-parse HEAD)"
echo "host=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "loss=${loss_name} n=${n} k=${k} N=${N} seed=${seed}"
echo "graph_path=${graph_path}"
echo "run_root=${run_root}"
echo "dimensions=5..30 (26 independent embeddings batched on one GPU)"

python "${BATCHED_TRAINER}" \
    --loss "${loss_name}" \
    --graph-path "${graph_path}" \
    --n "${n}" \
    --N "${N}" \
    --k "${k}" \
    --d-values "5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30" \
    --num-steps 100000 \
    --save-every 1000 \
    --save-path "${run_root}" \
    --lr 0.01 \
    --min-lr-ratio 0.01 \
    --warmup-frac 0.05 \
    --relative-bias 0 \
    --temperature 0.1 \
    --seed "${seed}" \
    --device cuda \
    --row-chunk-size 2048
