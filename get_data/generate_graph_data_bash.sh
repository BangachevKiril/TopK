#!/bin/bash
#SBATCH --job-name=gen_graphs
#SBATCH --output=logs/gen_graphs_%A_%a.out
#SBATCH --error=logs/gen_graphs_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --partition=mit_normal

n_values=(10 20 40 80 160)
N_values=(10 20 40 80 160 320 640 1280 2560 5120 10240 20480)
k_values=(1 2 3 4 5)
seed_values=(0 1 2 3 4 5 6 7 8 9)
NUM_ARRAY_JOBS=20
OUTPUT_DIR="/home/kirilb/orcd/pool/TopK/SyntheticGraphs"
PYTHON_SCRIPT="generate_graph_data.py"
MANIFEST_DIR="${OUTPUT_DIR}/manifests"

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

mkdir -p logs
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${MANIFEST_DIR}"

if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    timestamp=$(date +%Y%m%d_%H%M%S)
    manifest="${MANIFEST_DIR}/graph_manifest_${timestamp}.txt"
    : > "${manifest}"

    for n in "${n_values[@]}"; do
        for N in "${N_values[@]}"; do
            for k in "${k_values[@]}"; do
                max_neighborhoods=$(choose "$n" "$k")
                if [ "$N" -le "$max_neighborhoods" ]; then
                    for seed in "${seed_values[@]}"; do
                        echo "${n} ${k} ${N} ${seed}" >> "${manifest}"
                    done
                fi
            done
        done
    done

    total_runs=$(wc -l < "${manifest}")
    if [ "$total_runs" -eq 0 ]; then
        echo "No valid runs found."
        exit 1
    fi

    array_count=${NUM_ARRAY_JOBS}
    if [ "$total_runs" -lt "$array_count" ]; then
        array_count=${total_runs}
    fi

    echo "Manifest: ${manifest}"
    echo "Total valid runs: ${total_runs}"
    echo "Submitting ${array_count} array tasks."

    sbatch \
        --array=0-$((array_count - 1)) \
        --export=ALL,MANIFEST="${manifest}",TOTAL_RUNS="${total_runs}",ARRAY_COUNT="${array_count}" \
        "$0"
    exit $?
fi

module load miniforge
source activate GPUenv

if [ -z "${MANIFEST:-}" ] || [ -z "${TOTAL_RUNS:-}" ] || [ -z "${ARRAY_COUNT:-}" ]; then
    echo "Missing MANIFEST, TOTAL_RUNS, or ARRAY_COUNT in environment."
    exit 1
fi

if [ ! -f "${MANIFEST}" ]; then
    echo "Manifest file not found: ${MANIFEST}"
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
    start=$(( task_id * (base + 1) + 1 ))
    end=$(( start + base ))
else
    start=$(( extra * (base + 1) + (task_id - extra) * base + 1 ))
    end=$(( start + base - 1 ))
fi

if [ "${start}" -gt "${TOTAL_RUNS}" ]; then
    echo "Task ${task_id} has no assigned runs."
    exit 0
fi

if [ "${end}" -gt "${TOTAL_RUNS}" ]; then
    end=${TOTAL_RUNS}
fi

echo "Task ${task_id} processing manifest lines ${start} through ${end}."

for line_no in $(seq "${start}" "${end}"); do
    line=$(sed -n "${line_no}p" "${MANIFEST}")
    n=$(echo "${line}" | awk '{print $1}')
    k=$(echo "${line}" | awk '{print $2}')
    N=$(echo "${line}" | awk '{print $3}')
    seed=$(echo "${line}" | awk '{print $4}')

    echo "Running n=${n}, k=${k}, N=${N}, seed=${seed}"
    python "${PYTHON_SCRIPT}" \
        --n "${n}" \
        --k "${k}" \
        --N "${N}" \
        --seed "${seed}" \
        --output_dir "${OUTPUT_DIR}"
done
