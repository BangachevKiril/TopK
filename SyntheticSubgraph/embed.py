#!/usr/bin/env python3
"""
Train SigLIP-style embeddings on a random bipartite graph with distinct k-neighborhoods.

This version saves graph data and checkpoints as .npz files.
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
import torch
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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


def validate_and_resolve_sizes(n: int, d: int, k: int, p: Optional[float], N: Optional[int], batch_size: Optional[int]) -> Tuple[int, int]:
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
        raw_N = p * total_combinations
        rounded_N = int(round(raw_N))
        if not math.isclose(raw_N, rounded_N, rel_tol=1e-12, abs_tol=1e-9):
            raise ValueError(
                "When N is None, p * binom(n, k) must be an integer up to floating-point tolerance. "
                f"Got p * binom(n, k) = {raw_N}."
            )
        N = rounded_N

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


def neighborhoods_to_dense_adjacency(neighborhoods: torch.Tensor, n: int) -> torch.Tensor:
    N, k = neighborhoods.shape
    A = torch.zeros((N, n), dtype=torch.bool)
    if k == 0:
        return A
    row_idx = torch.arange(N, dtype=torch.long).unsqueeze(1).expand(-1, k)
    A[row_idx, neighborhoods] = True
    return A


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
    A_cpu: torch.Tensor,
    N: int,
    n: int,
    d: int,
    initialization: str,
    device: torch.device,
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
        A_float = A_cpu.to(dtype=torch.float32)
        L, _, Rh = torch.linalg.svd(A_float, full_matrices=False)
        U0 = normalize_rows(L[:, :d].to(device=device))
        V0 = normalize_rows(Rh[:d, :].T.to(device=device))

    U = torch.nn.Parameter(U0)
    V = torch.nn.Parameter(V0)
    b = torch.nn.Parameter(torch.zeros(1, 1, device=device))
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

    optimizer = torch.optim.Adam(params, lr=lr)
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


def train_one_step(
    U: torch.nn.Parameter,
    V: torch.nn.Parameter,
    b: torch.nn.Parameter,
    t: torch.nn.Parameter,
    A_cpu: torch.Tensor,
    batch_size: int,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    N, n = A_cpu.shape
    dtype = U.dtype
    optimizer.zero_grad(set_to_none=True)

    left_subset_cpu = torch.randperm(n)[:batch_size]
    left_subset = left_subset_cpu.to(device=device)
    row_chunk_size = min(N, auto_row_chunk_size(batch_size))
    V_subset = V.index_select(0, left_subset)

    total_loss = 0.0
    for row_start in range(0, N, row_chunk_size):
        row_end = min(row_start + row_chunk_size, N)
        U_chunk = U[row_start:row_end]
        scores = U_chunk @ V_subset.T

        A_chunk = A_cpu[row_start:row_end].index_select(1, left_subset_cpu)
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
def compute_gap(U: torch.nn.Parameter, V: torch.nn.Parameter, A_cpu: torch.Tensor, device: torch.device) -> Tuple[float, float, float]:
    N, n = A_cpu.shape
    row_chunk_size = min(N, auto_row_chunk_size(n, target_entries=4_000_000))

    pos_min = float("inf")
    neg_max = float("-inf")
    V_detached = V.detach()

    for row_start in range(0, N, row_chunk_size):
        row_end = min(row_start + row_chunk_size, N)
        U_chunk = U[row_start:row_end]
        scores = U_chunk @ V_detached.T
        A_chunk = A_cpu[row_start:row_end].to(device=device)

        pos_mask = A_chunk
        neg_mask = ~A_chunk

        if pos_mask.any():
            pos_min = min(pos_min, float(scores[pos_mask].min().item()))
        if neg_mask.any():
            neg_max = max(neg_max, float(scores[neg_mask].max().item()))

    gap = float("nan")
    if math.isfinite(pos_min) and math.isfinite(neg_max):
        gap = pos_min - neg_max
    return pos_min, neg_max, gap


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
    b: torch.nn.Parameter,
    t: torch.nn.Parameter,
    loss_value: float,
    pos_min: float,
    neg_max: float,
    gap: float,
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
        gap=gap,
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
        gap=gap,
        lr=lr,
    )


def train_siglip_bipartite(
    n: int,
    d: int,
    k: int,
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
    seed: Optional[int] = None,
    device: Optional[str] = None,
) -> Dict:
    if save_every <= 0:
        raise ValueError(f"save_every must be positive, got {save_every}.")

    set_seed(seed)
    resolved_device = resolve_device(device)
    N, batch_size = validate_and_resolve_sizes(n=n, d=d, k=k, p=p, N=N, batch_size=batch_size)

    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {resolved_device}")
    print(f"Resolved sizes: n={n}, N={N}, d={d}, k={k}, batch_size={batch_size}")
    print("Sampling distinct k-neighborhoods...")
    neighborhoods = sample_neighborhoods(n=n, k=k, N=N)
    print("Building dense adjacency matrix A...")
    A_cpu = neighborhoods_to_dense_adjacency(neighborhoods, n=n)

    config = {
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
        "seed": seed,
        "device": str(resolved_device),
    }

    save_npz(save_dir / "graph_data.npz", A=A_cpu.to(torch.uint8), neighborhoods=neighborhoods)
    with open(save_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"Saved graph_data.npz and config.json to {save_dir}")
    print(f"Initializing U and V with '{initialization}' initialization...")
    U, V, b, t = initialize_embeddings(
        A_cpu=A_cpu,
        N=N,
        n=n,
        d=d,
        initialization=initialization,
        device=resolved_device,
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
    gap = float("nan")

    for step in range(1, num_steps + 1):
        last_loss = train_one_step(
            U=U,
            V=V,
            b=b,
            t=t,
            A_cpu=A_cpu,
            batch_size=batch_size,
            optimizer=optimizer,
            device=resolved_device,
        )
        scheduler.step()

        should_save = (step % save_every == 0) or (step == num_steps)
        if should_save:
            pos_min, neg_max, gap = compute_gap(U=U, V=V, A_cpu=A_cpu, device=resolved_device)
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"step={step:6d} | loss={last_loss:.6f} | lr={current_lr:.6e} | "
                f"t={t.item():.6f} | b={b.item():.6f} | gap={gap:.6f}"
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
                gap=gap,
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
        gap=gap,
    )
    print(f"Training complete. Final artifacts saved under: {save_dir}")

    return {
        "save_dir": str(save_dir),
        "resolved_N": N,
        "resolved_batch_size": batch_size,
        "device": str(resolved_device),
    }


def main() -> None:
    args = parse_args()
    train_siglip_bipartite(
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
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()
