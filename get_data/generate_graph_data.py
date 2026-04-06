#!/usr/bin/env python3
import argparse
import os
from itertools import combinations

import numpy as np


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
            "Requested N=%d distinct neighborhoods, but C(n,k)=C(%d,%d)=%d." % (N, n, k, total)
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


def build_adjacency(neighborhoods, n):
    N, k = neighborhoods.shape
    A = np.zeros((N, n), dtype=np.uint8)
    rows = np.arange(N)[:, None]
    A[rows, neighborhoods] = 1
    return A


def main():
    parser = argparse.ArgumentParser(
        description="Generate a random bipartite graph with distinct right-side neighborhoods and save A as .npy."
    )
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--save_neighborhoods", action="store_true")
    args = parser.parse_args()

    n = args.n
    k = args.k
    N = args.N
    seed = args.seed

    if n <= 0:
        raise ValueError("n must be positive.")
    if k <= 0 or k > n:
        raise ValueError("k must satisfy 1 <= k <= n.")
    if N <= 0:
        raise ValueError("N must be positive.")

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    neighborhoods = sample_distinct_k_subsets(n=n, k=k, N=N, seed=seed)
    A = build_adjacency(neighborhoods=neighborhoods, n=n)

    base = "graph_n_%d_k_%d_N_%d_seed_%d" % (n, k, N, seed)
    graph_path = os.path.join(args.output_dir, base + ".npy")
    np.save(graph_path, A)

    if args.save_neighborhoods:
        neigh_path = os.path.join(args.output_dir, base + "_neighborhoods.npy")
        np.save(neigh_path, neighborhoods)

    print("Saved graph to: %s" % graph_path)
    print("Shape: %s" % (A.shape,))
    print("Edges per right vertex: %d" % k)
    print("Total possible distinct neighborhoods: %d" % n_choose_k(n, k))


if __name__ == "__main__":
    main()
