#!/bin/bash
#SBATCH --job-name=topk_k2_errors
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/tmp/topk_k2_errors_%A_%a.out
#SBATCH --error=/tmp/topk_k2_errors_%A_%a.err

set -eo pipefail

: "${GRAPH_ROOT:?GRAPH_ROOT must be set}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT must be set}"
: "${BATCHED_TRAINER:?BATCHED_TRAINER must be set}"

SOURCE_COMMIT="${SOURCE_COMMIT:-81b82ca49eb7714bd435e98e850b1c5afe643a3a}"
N_VALUES_RAW="${N_VALUES_RAW:-240 220 200 180 160 140 120 100 80 60 40 20}"
D_VALUES_RAW="${D_VALUES_RAW:-5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30}"

read -r -a n_values <<< "${N_VALUES_RAW//,/ }"
loss_values=(infonce sigmoid)

num_n=${#n_values[@]}
total_runs=$(( ${#loss_values[@]} * num_n ))
task_id=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID must be set}

if (( task_id < 0 || task_id >= total_runs )); then
    echo "Invalid task id ${task_id}; expected 0..$((total_runs - 1))."
    exit 2
fi

loss_idx=$((task_id / num_n))
n_idx=$((task_id % num_n))
loss_name=${loss_values[$loss_idx]}
n=${n_values[$n_idx]}
k=2
seed=0
N=$((n * (n - 1) / 2))

graph_stem="graph_n_${n}_k_${k}_N_${N}_seed_${seed}"
graph_path="${GRAPH_ROOT}/${graph_stem}.npz"
run_root="${OUTPUT_ROOT}/${loss_name}/${graph_stem}"

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

echo "source_commit=${SOURCE_COMMIT}"
echo "trainer_sha256=$(sha256sum "${BATCHED_TRAINER}" | awk '{print $1}')"
echo "host=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "loss=${loss_name} n=${n} k=${k} N=${N} seed=${seed}"
echo "graph_path=${graph_path}"
echo "run_root=${run_root}"
echo "dimensions=${D_VALUES_RAW} (independent embeddings batched on one GPU)"
echo "classification=positive iff score>0"

python "${BATCHED_TRAINER}" \
    --loss "${loss_name}" \
    --graph-path "${graph_path}" \
    --n "${n}" \
    --N "${N}" \
    --k "${k}" \
    --d-values "${D_VALUES_RAW}" \
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
