#!/usr/bin/env python3
"""
Train SigLIP-style embeddings on a random bipartite graph with distinct k-neighborhoods.

This version:
  - loads sparse graph files saved as scipy .npz,
  - keeps the graph in compact neighborhood form in memory,
  - falls back to legacy dense .npy graph files when needed,
  - recursively-friendly: graph_path may live anywhere,
  - saves checkpoints as compressed .npz files,
  - tracks margin, defined as gap / 2,
  - optionally fixes the bias to a user-provided constant via --relative_bias.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph_path", type=str, default=None)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--d", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--p", type=float, default=None)
    parser.add_argument("--N", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--initialization", type=str, default="random", choices=["random", "spectral"])
    parser.add_argument("--num_steps", type=int, required=True)
    parser.add_argument("--save_every", type=int, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--min_lr_ratio", type=float, default=1e-2)
    parser.add_argument("--warmup_frac", type=float, default=0.05)
    parser.add_argument(
        "--relative_bias",
        type=float,
        default=None,
        help="If provided, fix the bias b to this constant and do not optimize it.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: Optional[str]) -> torch.device:
    if device_arg is not None:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def validate_and_resolve_sizes(
    n: int,
    d: int,
    k: int,
    p: Optional[float],
    N: Optional[int],
    batch_size: Optional[int],
) -> Tuple[int, int]:
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}.")
    if d <= 0:
        raise ValueError(f"d must be positive, got {d}.")
    if not (0 <= k <= n):
        raise ValueError(f"k must satisfy 0 <= k <= n, got k={k}, n={n}.")

    total_combinations = math.comb(n, k)
    if total_combinations <= 0:
        raise ValueError("binom(n, k) must be positive.")

    if N is None:
        if p is None:
            raise ValueError("You must provide p when N is None.")
        if p <= 0:
            raise ValueError(f"p must be positive, got {p}.")
        N = math.floor(p * total_combinations)

    if N <= 0:
        raise ValueError(f"Resolved N must be positive, got {N}.")
    if N > total_combinations:
        raise ValueError(
            f"Cannot choose N={N} distinct k-neighborhoods because binom(n, k)={total_combinations}."
        )

    if batch_size is None:
        batch_size = n
    if not (1 <= batch_size <= n):
        raise ValueError(f"batch_size must satisfy 1 <= batch_size <= n, got {batch_size}.")

    return N, batch_size


def sample_unique_ranks(total: int, num_samples: int) -> list[int]:
    if num_samples > total:
        raise ValueError(f"Cannot sample {num_samples} unique ranks from total={total}.")

    if total <= sys.maxsize:
        return random.sample(range(total), num_samples)

    seen = set()
    while len(seen) < num_samples:
        seen.add(random.randrange(total))
    return list(seen)


def unrank_combination(rank: int, n: int, k: int) -> Tuple[int, ...]:
    combo = []
    start = 0
    remaining = rank
    for remaining_k in range(k, 0, -1):
        for value in range(start, n - remaining_k + 1):
            count = math.comb(n - value - 1, remaining_k - 1)
            if remaining < count:
                combo.append(value)
                start = value + 1
                break
            remaining -= count
    return tuple(combo)


def sample_neighborhoods(n: int, k: int, N: int) -> torch.Tensor:
    total_combinations = math.comb(n, k)
    ranks = sample_unique_ranks(total_combinations, N)
    neighborhoods = torch.empty((N, k), dtype=torch.long)
    for idx, rank in enumerate(ranks):
        if k == 0:
            neighborhoods[idx] = torch.empty((0,), dtype=torch.long)
        else:
            neighborhoods[idx] = torch.tensor(unrank_combination(rank, n, k), dtype=torch.long)
    return neighborhoods


def neighborhoods_to_csr(neighborhoods: torch.Tensor, n: int) -> sp.csr_matrix:
    neighborhoods_np = neighborhoods.cpu().numpy()
    N, k = neighborhoods_np.shape
    if k == 0:
        return sp.csr_matrix((N, n), dtype=np.float32)

    row_idx = np.repeat(np.arange(N, dtype=np.int64), k)
    col_idx = neighborhoods_np.reshape(-1)
    data = np.ones(N * k, dtype=np.float32)
    return sp.csr_matrix((data, (row_idx, col_idx)), shape=(N, n), dtype=np.float32)


def dense_adjacency_to_neighborhoods(A_np: np.ndarray, expected_k: int) -> torch.Tensor:
    if A_np.ndim != 2:
        raise ValueError(f"Graph array must be 2D, got shape {A_np.shape}.")

    unique_vals = np.unique(A_np)
    if not np.all(np.isin(unique_vals, [0, 1])):
        raise ValueError(f"Graph must have entries in {{0,1}}, got values {unique_vals}.")

    N = A_np.shape[0]
    row_sums = A_np.sum(axis=1)
    if not np.all(row_sums == expected_k):
        bad_rows = np.where(row_sums != expected_k)[0][:10]
        raise ValueError(
            f"Every row must contain exactly k={expected_k} ones. "
            f"Found violations in rows {bad_rows.tolist()}."
        )

    if expected_k == 0:
        neighborhoods = np.empty((N, 0), dtype=np.int64)
    else:
        neighborhoods = np.empty((N, expected_k), dtype=np.int64)
        for i in range(N):
            neighborhoods[i] = np.flatnonzero(A_np[i]).astype(np.int64, copy=False)

    return torch.from_numpy(neighborhoods).long()


def sparse_csr_to_neighborhoods(A_csr: sp.csr_matrix, expected_k: int) -> torch.Tensor:
    A_csr = A_csr.tocsr()
    A_csr.sort_indices()

    row_counts = np.diff(A_csr.indptr)
    if not np.all(row_counts == expected_k):
        bad_rows = np.where(row_counts != expected_k)[0][:10]
        raise ValueError(
            f"Every row must contain exactly k={expected_k} ones. "
            f"Found violations in rows {bad_rows.tolist()}."
        )

    unique_vals = np.unique(A_csr.data)
    if not np.all(np.isin(unique_vals, [0, 1])):
        raise ValueError(f"Sparse graph must have entries in {{0,1}}, got values {unique_vals}.")

    N = A_csr.shape[0]
    if expected_k == 0:
        neighborhoods = np.empty((N, 0), dtype=np.int64)
    else:
        neighborhoods = A_csr.indices.reshape(N, expected_k).astype(np.int64, copy=True)

    return torch.from_numpy(neighborhoods).long()


def load_graph_neighborhoods(
    graph_path: str,
    expected_N: int,
    expected_n: int,
    expected_k: int,
) -> torch.Tensor:
    path = Path(graph_path)
    if not path.is_file():
        raise FileNotFoundError(f"Graph file not found: {path}")

    if path.suffix == ".npz":
        A_sparse = sp.load_npz(path)
        if A_sparse.shape != (expected_N, expected_n):
            raise ValueError(
                f"Graph shape mismatch: expected {(expected_N, expected_n)}, got {A_sparse.shape}."
            )
        return sparse_csr_to_neighborhoods(A_sparse, expected_k=expected_k).long()

    if path.suffix == ".npy":
        A_np = np.load(path)
        if A_np.shape != (expected_N, expected_n):
            raise ValueError(
                f"Graph shape mismatch: expected {(expected_N, expected_n)}, got {A_np.shape}."
            )
        return dense_adjacency_to_neighborhoods(A_np, expected_k=expected_k).long()

    raise ValueError(f"Unsupported graph file suffix for {path}. Expected .npz or .npy.")


@torch.no_grad()
def normalize_rows(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    norms = x.norm(dim=1, keepdim=True)
    x = x / norms.clamp_min(eps)
    zero_mask = norms.squeeze(1) < eps
    if zero_mask.any():
        refill = torch.randn(int(zero_mask.sum().item()), x.size(1), device=x.device, dtype=x.dtype)
        refill = refill / refill.norm(dim=1, keepdim=True).clamp_min(eps)
        x[zero_mask] = refill
    return x


@torch.no_grad()
def renormalize_rows_inplace(x: torch.Tensor, eps: float = 1e-12) -> None:
    norms = x.norm(dim=1, keepdim=True)
    zero_mask = norms.squeeze(1) < eps
    x.div_(norms.clamp_min(eps))
    if zero_mask.any():
        refill = torch.randn(int(zero_mask.sum().item()), x.size(1), device=x.device, dtype=x.dtype)
        refill = refill / refill.norm(dim=1, keepdim=True).clamp_min(eps)
        x[zero_mask] = refill


def initialize_embeddings(
    neighborhoods_cpu: torch.Tensor,
    N: int,
    n: int,
    d: int,
    initialization: str,
    device: torch.device,
    relative_bias: Optional[float] = None,
) -> Tuple[torch.nn.Parameter, torch.nn.Parameter, torch.nn.Parameter, torch.nn.Parameter]:
    if initialization not in {"random", "spectral"}:
        raise ValueError(f"Unknown initialization: {initialization}")

    if initialization == "random":
        U0 = torch.randn(N, d, device=device)
        V0 = torch.randn(n, d, device=device)
        U0 = normalize_rows(U0)
        V0 = normalize_rows(V0)
    else:
        if d > min(N, n):
            raise ValueError(
                f"Spectral initialization requires d <= min(N, n). Got d={d}, N={N}, n={n}."
            )
        A_csr = neighborhoods_to_csr(neighborhoods_cpu, n=n)
        A_dense = torch.from_numpy(A_csr.toarray()).to(dtype=torch.float32)
        L, _, Rh = torch.linalg.svd(A_dense, full_matrices=False)
        U0 = normalize_rows(L[:, :d].to(device=device))
        V0 = normalize_rows(Rh[:d, :].T.to(device=device))

    U = torch.nn.Parameter(U0)
    V = torch.nn.Parameter(V0)

    if relative_bias is None:
        b = torch.nn.Parameter(torch.zeros(1, 1, device=device))
    else:
        b = torch.nn.Parameter(
            torch.full((1, 1), float(relative_bias), device=device),
            requires_grad=False,
        )

    t = torch.nn.Parameter(torch.ones(1, 1, device=device))
    return U, V, b, t


def make_optimizer_and_scheduler(params, num_steps: int, lr: float, min_lr_ratio: float, warmup_frac: float):
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}.")
    if lr <= 0:
        raise ValueError(f"lr must be positive, got {lr}.")
    if not (0.0 < min_lr_ratio <= 1.0):
        raise ValueError(f"min_lr_ratio must lie in (0, 1], got {min_lr_ratio}.")
    if not (0.0 <= warmup_frac < 1.0):
        raise ValueError(f"warmup_frac must lie in [0, 1), got {warmup_frac}.")

    trainable_params = [param for param in params if getattr(param, "requires_grad", False)]
    if not trainable_params:
        raise ValueError("No trainable parameters were provided to the optimizer.")

    optimizer = torch.optim.Adam(trainable_params, lr=lr)
    warmup_steps = max(1, int(round(num_steps * warmup_frac))) if num_steps > 1 and warmup_frac > 0 else 0
    decay_steps = max(1, num_steps - warmup_steps)

    if warmup_steps > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=decay_steps,
            eta_min=lr * min_lr_ratio,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_steps],
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, num_steps),
            eta_min=lr * min_lr_ratio,
        )

    return optimizer, scheduler


def auto_row_chunk_size(width: int, target_entries: int = 2_000_000) -> int:
    return max(1, target_entries // max(1, width))


def build_subset_position_lookup(n: int, left_subset_cpu: torch.Tensor) -> torch.Tensor:
    subset_pos = torch.full((n,), -1, dtype=torch.long)
    subset_pos[left_subset_cpu] = torch.arange(left_subset_cpu.numel(), dtype=torch.long)
    return subset_pos


def build_chunk_adjacency_from_subset_lookup(
    neighborhoods_chunk_cpu: torch.Tensor,
    subset_pos_cpu: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    neighborhoods_chunk_cpu = neighborhoods_chunk_cpu.long()
    rows, k = neighborhoods_chunk_cpu.shape
    A_chunk = torch.zeros((rows, batch_size), dtype=torch.bool)

    if rows == 0 or k == 0 or batch_size == 0:
        return A_chunk

    selected_pos = subset_pos_cpu[neighborhoods_chunk_cpu]
    valid = selected_pos >= 0
    if valid.any():
        row_ids = torch.arange(rows, dtype=torch.long).unsqueeze(1).expand(-1, k)
        A_chunk[row_ids[valid], selected_pos[valid]] = True

    return A_chunk


def train_one_step(
    U: torch.nn.Parameter,
    V: torch.nn.Parameter,
    b: torch.Tensor,
    t: torch.nn.Parameter,
    neighborhoods_cpu: torch.Tensor,
    batch_size: int,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    N, k = neighborhoods_cpu.shape
    n = V.shape[0]
    dtype = U.dtype
    optimizer.zero_grad(set_to_none=True)

    left_subset_cpu = torch.randperm(n)[:batch_size]
    left_subset = left_subset_cpu.to(device=device)
    subset_pos_cpu = build_subset_position_lookup(n=n, left_subset_cpu=left_subset_cpu)
    row_chunk_size = min(N, auto_row_chunk_size(batch_size))

    total_loss = 0.0
    for row_start in range(0, N, row_chunk_size):
        row_end = min(row_start + row_chunk_size, N)

        U_chunk = U[row_start:row_end]
        V_subset = V.index_select(0, left_subset)
        scores = U_chunk @ V_subset.T

        neighborhoods_chunk_cpu = neighborhoods_cpu[row_start:row_end].long()
        A_chunk = build_chunk_adjacency_from_subset_lookup(
            neighborhoods_chunk_cpu=neighborhoods_chunk_cpu,
            subset_pos_cpu=subset_pos_cpu,
            batch_size=batch_size,
        )

        signs = 1.0 - 2.0 * A_chunk.to(device=device, dtype=dtype)
        chunk_loss = F.softplus(t * (scores - b) * signs).sum()
        chunk_loss.backward()
        total_loss += float(chunk_loss.detach().cpu().item())

    optimizer.step()
    with torch.no_grad():
        renormalize_rows_inplace(U)
        renormalize_rows_inplace(V)

    return total_loss


@torch.no_grad()
def compute_margin(
    U: torch.nn.Parameter,
    V: torch.nn.Parameter,
    neighborhoods_cpu: torch.Tensor,
    device: torch.device,
) -> Tuple[float, float, float]:
    N, k = neighborhoods_cpu.shape
    n = V.shape[0]
    row_chunk_size = min(N, auto_row_chunk_size(n, target_entries=4_000_000))

    pos_min = float("inf")
    neg_max = float("-inf")
    V_detached = V.detach()

    for row_start in range(0, N, row_chunk_size):
        row_end = min(row_start + row_chunk_size, N)

        U_chunk = U[row_start:row_end]
        scores = U_chunk @ V_detached.T
        neighborhoods_chunk = neighborhoods_cpu[row_start:row_end].to(device=device, dtype=torch.long)

        if k > 0:
            pos_scores = scores.gather(1, neighborhoods_chunk)
            pos_min = min(pos_min, float(pos_scores.min().item()))

            masked_scores = scores.clone()
            masked_scores.scatter_(1, neighborhoods_chunk, float("-inf"))
            neg_max = max(neg_max, float(masked_scores.max().item()))
        else:
            neg_max = max(neg_max, float(scores.max().item()))

    margin = float("nan")
    if math.isfinite(pos_min) and math.isfinite(neg_max):
        margin = 0.5 * (pos_min - neg_max)
    return pos_min, neg_max, margin


def save_npz(path: Path, **kwargs) -> None:
    arrays = {}
    for key, value in kwargs.items():
        if isinstance(value, torch.Tensor):
            arrays[key] = value.detach().cpu().numpy()
        elif np.isscalar(value):
            arrays[key] = np.asarray(value)
        else:
            arrays[key] = np.asarray(value)
    np.savez_compressed(path, **arrays)


def save_checkpoint_npz(
    save_dir: Path,
    step: int,
    U: torch.nn.Parameter,
    V: torch.nn.Parameter,
    b: torch.Tensor,
    t: torch.nn.Parameter,
    loss_value: float,
    pos_min: float,
    neg_max: float,
    margin: float,
    lr: float,
) -> None:
    ckpt_path = save_dir / f"checkpoint_step_{step:06d}.npz"
    save_npz(
        ckpt_path,
        step=step,
        U=U,
        V=V,
        b=b,
        t=t,
        loss=loss_value,
        pos_min=pos_min,
        neg_max=neg_max,
        margin=margin,
        lr=lr,
    )
    save_npz(
        save_dir / "latest.npz",
        step=step,
        U=U,
        V=V,
        b=b,
        t=t,
        loss=loss_value,
        pos_min=pos_min,
        neg_max=neg_max,
        margin=margin,
        lr=lr,
    )


def train_siglip_bipartite(
    graph_path: Optional[str] = None,
    n: int = 0,
    d: int = 0,
    k: int = 0,
    p: Optional[float] = None,
    N: Optional[int] = None,
    batch_size: Optional[int] = None,
    initialization: str = "random",
    num_steps: int = 1000,
    save_every: int = 100,
    save_path: str = "./run",
    lr: float = 1e-2,
    min_lr_ratio: float = 1e-2,
    warmup_frac: float = 0.05,
    relative_bias: Optional[float] = None,
    seed: Optional[int] = None,
    device: Optional[str] = None,
) -> Dict:
    if save_every <= 0:
        raise ValueError(f"save_every must be positive, got {save_every}.")

    set_seed(seed)
    resolved_device = resolve_device(device)

    if graph_path is None:
        N, batch_size = validate_and_resolve_sizes(n=n, d=d, k=k, p=p, N=N, batch_size=batch_size)
    else:
        if N is None:
            raise ValueError("When graph_path is provided, you must also provide N.")
        if batch_size is None:
            batch_size = n
        if not (1 <= batch_size <= n):
            raise ValueError(f"batch_size must satisfy 1 <= batch_size <= n, got {batch_size}.")

    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {resolved_device}")
    print(f"Resolved sizes: n={n}, N={N}, d={d}, k={k}, batch_size={batch_size}")
    if relative_bias is None:
        print("Bias mode: trainable")
    else:
        print(f"Bias mode: fixed at relative_bias={relative_bias}")

    if graph_path is None:
        print("Sampling distinct k-neighborhoods...")
        neighborhoods_cpu = sample_neighborhoods(n=n, k=k, N=N).long()
    else:
        print(f"Reading graph from: {graph_path}")
        neighborhoods_cpu = load_graph_neighborhoods(
            graph_path=graph_path,
            expected_N=N,
            expected_n=n,
            expected_k=k,
        ).long()

    config = {
        "graph_path": graph_path,
        "n": n,
        "d": d,
        "k": k,
        "p": p,
        "N": N,
        "batch_size": batch_size,
        "initialization": initialization,
        "num_steps": num_steps,
        "save_every": save_every,
        "save_path": str(save_dir),
        "lr": lr,
        "min_lr_ratio": min_lr_ratio,
        "warmup_frac": warmup_frac,
        "relative_bias": relative_bias,
        "seed": seed,
        "device": str(resolved_device),
    }

    save_npz(
        save_dir / "graph_data.npz",
        neighborhoods=neighborhoods_cpu,
        n=n,
        N=N,
        k=k,
    )
    with open(save_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"Saved graph_data.npz and config.json to {save_dir}")
    print(f"Initializing U and V with '{initialization}' initialization...")

    U, V, b, t = initialize_embeddings(
        neighborhoods_cpu=neighborhoods_cpu,
        N=N,
        n=n,
        d=d,
        initialization=initialization,
        device=resolved_device,
        relative_bias=relative_bias,
    )

    optimizer, scheduler = make_optimizer_and_scheduler(
        params=[U, V, b, t],
        num_steps=num_steps,
        lr=lr,
        min_lr_ratio=min_lr_ratio,
        warmup_frac=warmup_frac,
    )

    last_loss = float("nan")
    pos_min = float("nan")
    neg_max = float("nan")
    margin = float("nan")

    for step in range(1, num_steps + 1):
        last_loss = train_one_step(
            U=U,
            V=V,
            b=b,
            t=t,
            neighborhoods_cpu=neighborhoods_cpu,
            batch_size=batch_size,
            optimizer=optimizer,
            device=resolved_device,
        )
        scheduler.step()

        should_save = (step % save_every == 0) or (step == num_steps)
        if should_save:
            pos_min, neg_max, margin = compute_margin(
                U=U,
                V=V,
                neighborhoods_cpu=neighborhoods_cpu,
                device=resolved_device,
            )
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"step={step:6d} | loss={last_loss:.6f} | lr={current_lr:.6e} | "
                f"t={t.item():.6f} | b={b.item():.6f} | margin={margin:.6f}"
            )
            save_checkpoint_npz(
                save_dir=save_dir,
                step=step,
                U=U,
                V=V,
                b=b,
                t=t,
                loss_value=last_loss,
                pos_min=pos_min,
                neg_max=neg_max,
                margin=margin,
                lr=current_lr,
            )

    save_npz(
        save_dir / "final.npz",
        U=U,
        V=V,
        b=b,
        t=t,
        loss=last_loss,
        pos_min=pos_min,
        neg_max=neg_max,
        margin=margin,
    )
    print(f"Training complete. Final artifacts saved under: {save_dir}")

    return {
        "save_dir": str(save_dir),
        "resolved_N": N,
        "resolved_batch_size": batch_size,
        "relative_bias": relative_bias,
        "device": str(resolved_device),
    }


def main() -> None:
    args = parse_args()
    train_siglip_bipartite(
        graph_path=args.graph_path,
        n=args.n,
        d=args.d,
        k=args.k,
        p=args.p,
        N=args.N,
        batch_size=args.batch_size,
        initialization=args.initialization,
        num_steps=args.num_steps,
        save_every=args.save_every,
        save_path=args.save_path,
        lr=args.lr,
        min_lr_ratio=args.min_lr_ratio,
        warmup_frac=args.warmup_frac,
        relative_bias=args.relative_bias,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()
