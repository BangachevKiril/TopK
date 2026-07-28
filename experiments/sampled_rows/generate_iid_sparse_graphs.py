#!/usr/bin/env python3
"""Generate fixed random k-sparse matrices for the sampled-row experiments.

For each (n, k, seed), one iid sequence of uniformly random k-subsets is
generated.  The matrix for each requested N is the first N rows of that
sequence, so the N sweeps are nested.  Sampling is with replacement across
rows and without replacement within each row.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp


def parse_int_list(raw: str) -> list[int]:
    return [int(value) for value in raw.replace(",", " ").split()]


def sample_neighborhoods(n: int, k: int, count: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    neighborhoods = np.empty((count, k), dtype=np.int64)
    for row in range(count):
        neighborhoods[row] = np.sort(rng.choice(n, size=k, replace=False))
    return neighborhoods


def save_graph(neighborhoods: np.ndarray, n: int, k: int, seed: int, output_dir: Path) -> None:
    N = int(neighborhoods.shape[0])
    stem = f"graph_n_{n}_k_{k}_N_{N}_seed_{seed}"
    row_idx = np.repeat(np.arange(N, dtype=np.int64), k)
    col_idx = neighborhoods.reshape(-1)
    data = np.ones(N * k, dtype=np.uint8)
    graph = sp.csr_matrix((data, (row_idx, col_idx)), shape=(N, n), dtype=np.uint8)

    graph_path = output_dir / f"{stem}.npz"
    neighborhoods_path = output_dir / f"{stem}_neighborhoods.npy"
    metadata_path = output_dir / f"{stem}.json"

    sp.save_npz(graph_path, graph, compressed=True)
    np.save(neighborhoods_path, neighborhoods)
    metadata_path.write_text(
        json.dumps(
            {
                "n": n,
                "k": k,
                "N": N,
                "seed": seed,
                "sampling": "iid_uniform_k_subsets_with_replacement_across_rows",
                "nested_prefix_design": True,
                "unique_rows": int(np.unique(neighborhoods, axis=0).shape[0]),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"saved={graph_path} shape={graph.shape} nnz={graph.nnz} "
        f"unique_rows={np.unique(neighborhoods, axis=0).shape[0]}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k-values", type=str, required=True)
    parser.add_argument("--N-values", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    k_values = parse_int_list(args.k_values)
    N_values = sorted(set(parse_int_list(args.N_values)))
    if args.n <= 0 or not k_values or not N_values:
        raise ValueError("n, k-values, and N-values must be non-empty and positive.")
    if min(N_values) <= 0:
        raise ValueError("Every N must be positive.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    max_N = max(N_values)

    for k in k_values:
        if not 1 <= k <= args.n:
            raise ValueError(f"k={k} must satisfy 1 <= k <= n={args.n}.")
        full_sequence = sample_neighborhoods(args.n, k, max_N, args.seed)
        for N in N_values:
            save_graph(full_sequence[:N].copy(), args.n, k, args.seed, args.output_dir)


if __name__ == "__main__":
    main()
