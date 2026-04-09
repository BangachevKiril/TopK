#!/usr/bin/env python3
import argparse
import math
import os
from itertools import combinations

import numpy as np
import scipy.sparse as sp


def n_choose_k(n, k):
    if k < 0 or k > n:
        return 0
    k = min(k, n - k)
    result = 1
    for i in range(1, k + 1):
        result = result * (n - k + i) // i
    return result


def sample_distinct_k_subsets(n, k, N, seed):
    total = n_choose_k(n, k)
    if N > total:
        raise ValueError(
            "Requested N=%d distinct neighborhoods, but C(n,k)=C(%d,%d)=%d."
            % (N, n, k, total)
        )

    rng = np.random.RandomState(seed)

    if N == total:
        neighborhoods = np.array(list(combinations(range(n), k)), dtype=np.int64)
        rng.shuffle(neighborhoods)
        return neighborhoods

    seen = set()
    neighborhoods = []
    while len(neighborhoods) < N:
        subset = tuple(sorted(rng.choice(n, size=k, replace=False).tolist()))
        if subset not in seen:
            seen.add(subset)
            neighborhoods.append(subset)

    return np.array(neighborhoods, dtype=np.int64)


def build_adjacency_csr(neighborhoods, n):
    N, k = neighborhoods.shape
    row_idx = np.repeat(np.arange(N, dtype=np.int64), k)
    col_idx = neighborhoods.reshape(-1)
    data = np.ones(N * k, dtype=np.uint8)
    return sp.csr_matrix((data, (row_idx, col_idx)), shape=(N, n), dtype=np.uint8)


def main():
    parser = argparse.ArgumentParser(
        description="Generate a random bipartite graph with distinct right-side neighborhoods and save A as compressed sparse .npz."
    )
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--p", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--nn_threshold", type=int, default=None)
    parser.add_argument("--save_neighborhoods", action="store_true")
    args = parser.parse_args()

    n = args.n
    k = args.k
    p = args.p
    seed = args.seed

    if n <= 0:
        raise ValueError("n must be positive.")
    if k <= 0 or k > n:
        raise ValueError("k must satisfy 1 <= k <= n.")
    if p <= 0:
        raise ValueError("p must be positive.")

    total = n_choose_k(n, k)
    raw_target = p * total

    if raw_target < n :
        raise ValueError(
            "Skipping because p * C(n,k) = %g * %d = %g < n = %g."
            % (p, total, raw_target, n )
        )

    N = math.floor(raw_target)

    if N <= 0:
        raise ValueError(
            "Computed N=floor(p * C(n,k)) = floor(%g * %d) = %d, which is not positive."
            % (p, total, N)
        )

    if N > total:
        raise ValueError(
            "Computed N=%d exceeds the total number of distinct neighborhoods C(%d,%d)=%d."
            % (N, n, k, total)
        )

    if args.nn_threshold is not None and n * N > args.nn_threshold:
        raise ValueError(
            "Skipping because n * N = %d * %d = %d > nn_threshold = %d."
            % (n, N, n * N, args.nn_threshold)
        )

    os.makedirs(args.output_dir, exist_ok=True)

    base = "graph_n_%d_k_%d_N_%d_seed_%d" % (n, k, N, seed)
    graph_path = os.path.join(args.output_dir, base + ".npz")

    neighborhoods = sample_distinct_k_subsets(n=n, k=k, N=N, seed=seed)
    A = build_adjacency_csr(neighborhoods=neighborhoods, n=n)

    sp.save_npz(graph_path, A, compressed=True)

    if args.save_neighborhoods:
        neigh_path = os.path.join(args.output_dir, base + "_neighborhoods.npy")
        np.save(neigh_path, neighborhoods)

    print("Saved sparse graph to: %s" % graph_path)
    print("Parameters: n=%d, k=%d, p=%g, N=%d, seed=%d" % (n, k, p, N, seed))
    print("Shape: %s" % (A.shape,))
    print("Edges per right vertex: %d" % k)
    print("Number of nonzeros: %d" % A.nnz)
    print("Total possible distinct neighborhoods: %d" % total)


if __name__ == "__main__":
    main()