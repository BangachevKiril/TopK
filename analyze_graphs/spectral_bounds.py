#!/usr/bin/env python3
r"""
Recursively-computed spectral bounds for sparse bipartite graphs.

For each graph A \in {0,1}^{N x n}, this script solves

    minimize    ||J||_op
    subject to  sum_{i,j} |J_{ij}| = 1,
                J_{ij} >= 0  whenever A_{ij} = 1,
                J_{ij} <= 0  whenever A_{ij} = 0.

The graph is expected to be stored as a SciPy sparse .npz file, as produced by
sp.save_npz(...). The solution is saved to spectral_bounds.npz in the same
folder as the graph unless --output_path is specified.

A convenient reparameterization is used internally:

    J = S \odot X,

where S_{ij} = +1 on edges and S_{ij} = -1 on nonedges, and X >= 0. Then
sum |J_{ij}| = sum X_{ij}, so the problem becomes

    minimize    ||S \odot X||_op
    subject to  X >= 0,
                sum_{i,j} X_{ij} = 1.

This is equivalent to the original problem and is easier for CVXPY to encode.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Iterable, List, Optional

import cvxpy as cp
import numpy as np
import scipy.sparse as sp


DEFAULT_SOLVER_ORDER = [
    "MOSEK",
    "CVXOPT",
    "SCS",
    "SDPA",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph_path",
        type=str,
        required=True,
        help="Path to a graph saved as a SciPy sparse .npz file.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Optional output path. Defaults to <graph_dir>/spectral_bounds.npz.",
    )
    parser.add_argument(
        "--solver",
        type=str,
        default="auto",
        help="CVXPY solver to use. Use 'auto' to try a reasonable fallback order.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite spectral_bounds.npz if it already exists.",
    )
    parser.add_argument(
        "--max_entries",
        type=int,
        default=None,
        help=(
            "Optional safety cap. If N*n exceeds this value, the script exits with "
            "an error before forming the dense sign matrix needed by CVXPY."
        ),
    )
    parser.add_argument(
        "--scs_eps",
        type=float,
        default=1e-5,
        help="Tolerance passed to SCS when SCS is used.",
    )
    parser.add_argument(
        "--scs_max_iters",
        type=int,
        default=20000,
        help="Maximum iterations passed to SCS when SCS is used.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable solver verbosity.",
    )
    return parser.parse_args()


def resolve_output_path(graph_path: Path, output_path_arg: Optional[str]) -> Path:
    if output_path_arg is not None:
        return Path(output_path_arg)
    return graph_path.parent / "spectral_bounds.npz"


def choose_solver_order(solver_arg: str) -> List[str]:
    installed = set(cp.installed_solvers())
    if solver_arg != "auto":
        if solver_arg not in installed:
            raise ValueError(
                f"Requested solver '{solver_arg}' is not installed. "
                f"Installed solvers: {sorted(installed)}"
            )
        return [solver_arg]

    ordered = [solver for solver in DEFAULT_SOLVER_ORDER if solver in installed]
    if not ordered:
        raise ValueError(
            "No suitable CVXPY solver was found. "
            f"Installed solvers: {sorted(installed)}"
        )
    return ordered


def load_graph(graph_path: Path, max_entries: Optional[int]) -> sp.csr_matrix:
    if not graph_path.is_file():
        raise FileNotFoundError(f"Graph file not found: {graph_path}")

    if graph_path.suffix != ".npz":
        raise ValueError(
            f"Expected a sparse .npz graph file, but got: {graph_path}"
        )

    try:
        A = sp.load_npz(graph_path)
    except Exception as exc:
        raise ValueError(f"Failed to load sparse .npz graph from {graph_path}: {exc}") from exc

    if not sp.issparse(A):
        raise ValueError(f"Loaded object is not sparse for file: {graph_path}")

    A = A.tocsr()

    if A.ndim != 2:
        raise ValueError(f"Graph must be a matrix, got ndim={A.ndim} for {graph_path}")

    N, n = A.shape
    if N <= 0 or n <= 0:
        raise ValueError(f"Graph must have positive shape, got {A.shape} for {graph_path}")

    if max_entries is not None and N * n > max_entries:
        raise ValueError(
            f"Refusing to form the dense sign matrix because N*n = {N*n} > max_entries = {max_entries}."
        )

    unique_vals = np.unique(A.data) if A.nnz > 0 else np.array([], dtype=A.dtype)
    if unique_vals.size > 0 and not np.all(np.isin(unique_vals, [0, 1, 1.0, True])):
        raise ValueError(
            f"Expected a 0/1 sparse adjacency matrix. Nonzero values found: {unique_vals[:10]}"
        )

    return A


def build_sign_matrix(A: sp.csr_matrix) -> np.ndarray:
    N, n = A.shape
    S = -np.ones((N, n), dtype=np.float64)

    A = A.tocoo()
    if A.nnz > 0:
        S[A.row, A.col] = 1.0

    return S


def solve_spectral_bound(
    sign_matrix: np.ndarray,
    solver_order: Iterable[str],
    scs_eps: float,
    scs_max_iters: int,
    verbose: bool,
):
    N, n = sign_matrix.shape

    X = cp.Variable((N, n), nonneg=True)
    J_expr = cp.multiply(sign_matrix, X)

    objective = cp.Minimize(cp.norm(J_expr, 2))
    constraints = [cp.sum(X) == 1]
    problem = cp.Problem(objective, constraints)

    last_error = None
    for solver in solver_order:
        try:
            solve_kwargs = {
                "solver": solver,
                "verbose": verbose,
            }
            if solver == "SCS":
                solve_kwargs["eps"] = scs_eps
                solve_kwargs["max_iters"] = scs_max_iters

            start = time.time()
            value = problem.solve(**solve_kwargs)
            solve_time = time.time() - start

            if problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
                X_value = np.asarray(X.value, dtype=np.float64)
                X_value = np.maximum(X_value, 0.0)
                J_value = sign_matrix * X_value
                spectral_norm_from_J = np.linalg.norm(J_value, ord=2)
                abs_sum_from_J = np.abs(J_value).sum()
                signed_sum_from_X = X_value.sum()

                return {
                    "status": problem.status,
                    "solver": solver,
                    "objective_value": float(value),
                    "solve_time_seconds": float(solve_time),
                    "J": J_value,
                    "X": X_value,
                    "spectral_norm_from_J": float(spectral_norm_from_J),
                    "abs_sum_from_J": float(abs_sum_from_J),
                    "signed_sum_from_X": float(signed_sum_from_X),
                }

            last_error = RuntimeError(
                f"Solver {solver} finished with status {problem.status}."
            )
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"All solver attempts failed. Last error: {last_error}")


def save_result(output_path: Path, graph_path: Path, A: sp.csr_matrix, result: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        J=result["J"],
        objective=np.asarray(result["objective_value"], dtype=np.float64),
        spectral_norm_from_J=np.asarray(result["spectral_norm_from_J"], dtype=np.float64),
        abs_sum=np.asarray(result["abs_sum_from_J"], dtype=np.float64),
        magnitude_sum=np.asarray(result["signed_sum_from_X"], dtype=np.float64),
        status=np.asarray(result["status"]),
        solver=np.asarray(result["solver"]),
        solve_time_seconds=np.asarray(result["solve_time_seconds"], dtype=np.float64),
        N=np.asarray(A.shape[0], dtype=np.int64),
        n=np.asarray(A.shape[1], dtype=np.int64),
        nnz=np.asarray(A.nnz, dtype=np.int64),
        graph_path=np.asarray(str(graph_path)),
    )


def main() -> int:
    args = parse_args()

    graph_path = Path(args.graph_path).expanduser().resolve()
    output_path = resolve_output_path(graph_path, args.output_path)

    if output_path.exists() and not args.overwrite:
        print(f"Output already exists, skipping: {output_path}")
        return 0

    A = load_graph(graph_path, max_entries=args.max_entries)
    N, n = A.shape
    density = A.nnz / float(N * n)

    print(f"Loaded graph: {graph_path}")
    print(f"Shape: N={N}, n={n}, nnz={A.nnz}, density={density:.6g}")
    print(f"Output path: {output_path}")

    sign_matrix = build_sign_matrix(A)
    solver_order = choose_solver_order(args.solver)
    print(f"Trying solvers in order: {solver_order}")

    result = solve_spectral_bound(
        sign_matrix=sign_matrix,
        solver_order=solver_order,
        scs_eps=args.scs_eps,
        scs_max_iters=args.scs_max_iters,
        verbose=args.verbose,
    )

    save_result(output_path=output_path, graph_path=graph_path, A=A, result=result)

    print(f"Status: {result['status']}")
    print(f"Solver: {result['solver']}")
    print(f"Objective (solver): {result['objective_value']:.12g}")
    print(f"Objective (recomputed): {result['spectral_norm_from_J']:.12g}")
    print(f"|J|_1: {result['abs_sum_from_J']:.12g}")
    print(f"Solve time [s]: {result['solve_time_seconds']:.6f}")
    print(f"Saved: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
