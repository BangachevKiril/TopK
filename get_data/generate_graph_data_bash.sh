#!/bin/bash
#SBATCH --job-name=gen_graphs
#SBATCH --output=logs/gen_graphs_%A_%a.out
#SBATCH --error=logs/gen_graphs_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --partition=mit_normal

n_values=(10 20 40 80 160 320 640)
p_values=(0.015625 0.03125 0.0625 0.125 0.25 0.5 1.0)
k_values=(1 2 3 4 5)
seed_values=(0)

NUM_ARRAY_JOBS=20
MAX_CONCURRENT=20
NN_THRESHOLD=100000000

OUTPUT_DIR="/home/kirilb/orcd/pool/TopK/SyntheticGraphsSmall"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT_PATH="$(readlink -f "$0")"
PYTHON_SCRIPT="generate_graph_data.py"

choose() {
    local n=$1
    local k=$2
    if [ "$k" -lt 0 ] || [ "$k" -gt "$n" ]; then
        echo 0
        return
    fi
    if [ "$k" -gt $((n - k)) ]; then
        k=$((n - k))
    fi
    local result=1
    local i term
    for ((i=1; i<=k; i++)); do
        term=$(( n - k + i ))
        result=$(( result * term / i ))
    done
    echo "$result"
}

compute_N_from_p() {
    local n=$1
    local k=$2
    local p=$3
    local max_neighborhoods
    max_neighborhoods=$(choose "$n" "$k")

    python - <<PY
import math
p = float("${p}")
M = int("${max_neighborhoods}")
print(math.floor(p * M))
PY
}

passes_min_size_check() {
    local n=$1
    local k=$2
    local p=$3
    local max_neighborhoods
    max_neighborhoods=$(choose "$n" "$k")

    python - <<PY
n = int("${n}")
p = float("${p}")
M = int("${max_neighborhoods}")
print(1 if p * M >= n  else 0)
PY
}

valid_N_for_triplet() {
    local n=$1
    local k=$2
    local p=$3

    local max_neighborhoods
    local keep_small
    local N

    max_neighborhoods=$(choose "$n" "$k")

    keep_small=$(passes_min_size_check "$n" "$k" "$p")
    if [ "$keep_small" -ne 1 ]; then
        return 1
    fi

    N=$(compute_N_from_p "$n" "$k" "$p")

    if [ "$N" -lt 1 ] || [ "$N" -gt "$max_neighborhoods" ]; then
        return 1
    fi

    if [ $((n * N)) -gt "$NN_THRESHOLD" ]; then
        return 1
    fi

    echo "$N"
    return 0
}

count_total_runs() {
    local total=0
    local n k p N
    for n in "${n_values[@]}"; do
        for k in "${k_values[@]}"; do
            for p in "${p_values[@]}"; do
                N=$(valid_N_for_triplet "$n" "$k" "$p") || continue
                total=$((total + ${#seed_values[@]}))
            done
        done
    done
    echo "$total"
}

get_run_by_index() {
    local target_idx=$1
    local current_idx=0
    local n k p N seed

    for n in "${n_values[@]}"; do
        for k in "${k_values[@]}"; do
            for p in "${p_values[@]}"; do
                N=$(valid_N_for_triplet "$n" "$k" "$p") || continue
                for seed in "${seed_values[@]}"; do
                    if [ "$current_idx" -eq "$target_idx" ]; then
                        echo "${n} ${k} ${p} ${N} ${seed}"
                        return 0
                    fi
                    current_idx=$((current_idx + 1))
                done
            done
        done
    done

    return 1
}

mkdir -p logs
mkdir -p "${OUTPUT_DIR}"

if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    total_runs=$(count_total_runs)

    if [ "$total_runs" -eq 0 ]; then
        echo "No valid runs found."
        exit 1
    fi

    array_count=${NUM_ARRAY_JOBS}
    if [ "$total_runs" -lt "$array_count" ]; then
        array_count=${total_runs}
    fi

    echo "SCRIPT_PATH=${SCRIPT_PATH}"
    echo "PYTHON_SCRIPT=${PYTHON_SCRIPT}"
    echo "OUTPUT_DIR=${OUTPUT_DIR}"
    echo "TOTAL_RUNS=${total_runs}"
    echo "ARRAY_COUNT=${array_count}"
    echo "Submitting ${array_count} array tasks."

    sbatch \
        --array=0-$((array_count - 1))%${MAX_CONCURRENT} \
        --export=ALL,TOTAL_RUNS="${total_runs}",ARRAY_COUNT="${array_count}",NN_THRESHOLD="${NN_THRESHOLD}",OUTPUT_DIR="${OUTPUT_DIR}" \
        "${SCRIPT_PATH}"
    exit $?
fi

module load miniforge
source activate GPUenv

if [ -z "${TOTAL_RUNS:-}" ] || [ -z "${ARRAY_COUNT:-}" ]; then
    echo "Missing TOTAL_RUNS or ARRAY_COUNT in environment."
    exit 1
fi

if [ ! -f "${PYTHON_SCRIPT}" ]; then
    echo "Python script not found: ${PYTHON_SCRIPT}"
    exit 1
fi

task_id=${SLURM_ARRAY_TASK_ID}
base=$(( TOTAL_RUNS / ARRAY_COUNT ))
extra=$(( TOTAL_RUNS % ARRAY_COUNT ))

if [ "${task_id}" -lt "${extra}" ]; then
    start=$(( task_id * (base + 1) ))
    end=$(( start + base ))
else
    start=$(( extra * (base + 1) + (task_id - extra) * base ))
    end=$(( start + base - 1 ))
fi

if [ "${start}" -ge "${TOTAL_RUNS}" ]; then
    echo "Task ${task_id} has no assigned runs."
    exit 0
fi

if [ "${end}" -ge "${TOTAL_RUNS}" ]; then
    end=$((TOTAL_RUNS - 1))
fi

echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "Task ${task_id} processing run indices ${start} through ${end}."

for global_idx in $(seq "${start}" "${end}"); do
    run_spec=$(get_run_by_index "${global_idx}")
    if [ $? -ne 0 ] || [ -z "${run_spec}" ]; then
        echo "Failed to recover run for global_idx=${global_idx}"
        exit 1
    fi

    read -r n k p N seed <<< "${run_spec}"

    echo "Running n=${n}, k=${k}, p=${p}, N=${N}, seed=${seed}"
    python "${PYTHON_SCRIPT}" \
        --n "${n}" \
        --k "${k}" \
        --p "${p}" \
        --seed "${seed}" \
        --nn_threshold "${NN_THRESHOLD}" \
        --output_dir "${OUTPUT_DIR}"
done