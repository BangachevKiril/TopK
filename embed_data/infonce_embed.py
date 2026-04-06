#!/usr/bin/env python3
"""
Train embeddings on a pre-generated bipartite graph saved as a .npy file,
using the full-batch InfoNCE objective from free_embedding_experiment.py.

The input graph file should contain a dense adjacency matrix A of shape (N, n)
with entries in {0,1}. Each row is expected to have exactly k ones.

This script keeps the same I/O style as the saved-graph SigLIP trainer:
- reads a saved graph from disk
- supports random / spectral initialization
- saves graph_data.npz, config.json, checkpoint_step_XXXXXX.npz, latest.npz, final.npz
- tracks gap statistics and top-k retrieval accuracy
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph_path", type=str, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--d", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--initialization", type=str, default="random", choices=["random", "spectral"])
    parser.add_argument("--num_steps", type=int, required=True)
    parser.add_argument("--save_every", type=int, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--min_lr_ratio", type=float, default=1e-2)
    parser.add_argument("--warmup_frac", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=0.1)
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


def validate_inputs(n: int, d: int, k: int, N: int, batch_size: Optional[int], temperature: float) -> None:
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}.")
    if d <= 0:
        raise ValueError(f"d must be positive, got {d}.")
    if not (0 < k <= n):
        raise ValueError(f"k must satisfy 1 <= k <= n, got k={k}, n={n}.")
    if N <= 0:
        raise ValueError(f"N must be positive, got {N}.")
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}.")

    total_combinations = math.comb(n, k)
    if N > total_combinations:
        raise ValueError(
            f"Cannot have N={N} distinct k-neighborhoods because binom(n, k)={total_combinations}."
        )

    if batch_size is not None and batch_size != n:
        print(
            f"Warning: batch_size={batch_size} was provided, but InfoNCE training here is full-batch "
            f"over all n={n} left vertices, matching free_embedding_experiment.py. Ignoring batch_size."
        )



def load_graph(graph_path: str, expected_N: int, expected_n: int, expected_k: int) -> torch.Tensor:
    path = Path(graph_path)
    if not path.is_file():
        raise FileNotFoundError(f"Graph file not found: {path}")

    A_np = np.load(path)
    if A_np.ndim != 2:
        raise ValueError(f"Graph array must be 2D, got shape {A_np.shape}.")
    if A_np.shape != (expected_N, expected_n):
        raise ValueError(
            f"Graph shape mismatch: expected {(expected_N, expected_n)}, got {A_np.shape}."
        )

    unique_vals = np.unique(A_np)
    if not np.all(np.isin(unique_vals, [0, 1])):
        raise ValueError(f"Graph must have entries in {{0,1}}, got values {unique_vals}.")

    row_sums = A_np.sum(axis=1)
    if not np.all(row_sums == expected_k):
        bad_rows = np.where(row_sums != expected_k)[0][:10]
        raise ValueError(
            f"Every row must contain exactly k={expected_k} ones. "
            f"Found violations in rows {bad_rows.tolist()}."
        )

    return torch.from_numpy(A_np.astype(np.bool_))


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
) -> Tuple[torch.nn.Parameter, torch.nn.Parameter]:
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
    return U, V



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



def infonce_loss(U: torch.Tensor, V: torch.Tensor, A_device: torch.Tensor, temperature: float) -> torch.Tensor:
    logits = (U @ V.T) / temperature
    log_probs = F.log_softmax(logits, dim=1)
    mask = A_device.to(dtype=log_probs.dtype)
    sum_pos_log_probs = (log_probs * mask).sum()
    num_pos = mask.sum().clamp_min(1.0)
    return -sum_pos_log_probs / num_pos



def train_one_step(
    U: torch.nn.Parameter,
    V: torch.nn.Parameter,
    A_device: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    temperature: float,
) -> float:
    optimizer.zero_grad(set_to_none=True)
    loss = infonce_loss(U=U, V=V, A_device=A_device, temperature=temperature)
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        renormalize_rows_inplace(U)
        renormalize_rows_inplace(V)
    return float(loss.detach().cpu().item())


@torch.no_grad()
def compute_gap(U: torch.nn.Parameter, V: torch.nn.Parameter, A_device: torch.Tensor) -> Tuple[float, float, float]:
    scores = U.detach() @ V.detach().T
    pos_mask = A_device
    neg_mask = ~A_device

    pos_min = float(scores[pos_mask].min().item()) if pos_mask.any() else float("inf")
    neg_max = float(scores[neg_mask].max().item()) if neg_mask.any() else float("-inf")

    gap = float("nan")
    if math.isfinite(pos_min) and math.isfinite(neg_max):
        gap = pos_min - neg_max
    return pos_min, neg_max, gap


@torch.no_grad()
def evaluate_top_k_accuracy(U: torch.nn.Parameter, V: torch.nn.Parameter, A_device: torch.Tensor, k: int) -> float:
    scores = U.detach() @ V.detach().T
    topk_idx = torch.topk(scores, k=k, dim=1, largest=True, sorted=False).indices
    hits = A_device.gather(1, topk_idx).to(torch.float32).sum()
    total = float(A_device.sum().item())
    return float((hits / max(total, 1.0)).item())



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
    loss_value: float,
    pos_min: float,
    neg_max: float,
    gap: float,
    accuracy: float,
    best_accuracy: float,
    best_loss: float,
    lr: float,
    temperature: float,
) -> None:
    # Save NaN placeholders for b and t so downstream readers expecting those keys do not break.
    b_placeholder = np.asarray([[np.nan]], dtype=np.float32)
    t_placeholder = np.asarray([[np.nan]], dtype=np.float32)

    ckpt_path = save_dir / f"checkpoint_step_{step:06d}.npz"
    save_npz(
        ckpt_path,
        step=step,
        U=U,
        V=V,
        b=b_placeholder,
        t=t_placeholder,
        loss=loss_value,
        pos_min=pos_min,
        neg_max=neg_max,
        gap=gap,
        accuracy=accuracy,
        best_accuracy=best_accuracy,
        best_loss=best_loss,
        lr=lr,
        temperature=temperature,
    )
    save_npz(
        save_dir / "latest.npz",
        step=step,
        U=U,
        V=V,
        b=b_placeholder,
        t=t_placeholder,
        loss=loss_value,
        pos_min=pos_min,
        neg_max=neg_max,
        gap=gap,
        accuracy=accuracy,
        best_accuracy=best_accuracy,
        best_loss=best_loss,
        lr=lr,
        temperature=temperature,
    )



def train_infonce_on_saved_graph(
    graph_path: str,
    n: int,
    d: int,
    k: int,
    N: int,
    batch_size: Optional[int] = None,
    initialization: str = "random",
    num_steps: int = 1000,
    save_every: int = 100,
    save_path: str = "./run",
    lr: float = 1e-2,
    min_lr_ratio: float = 1e-2,
    warmup_frac: float = 0.05,
    temperature: float = 0.1,
    seed: Optional[int] = None,
    device: Optional[str] = None,
) -> Dict:
    if save_every <= 0:
        raise ValueError(f"save_every must be positive, got {save_every}.")

    set_seed(seed)
    resolved_device = resolve_device(device)
    validate_inputs(n=n, d=d, k=k, N=N, batch_size=batch_size, temperature=temperature)

    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {resolved_device}")
    print(f"Reading graph from: {graph_path}")
    print(f"Resolved sizes: n={n}, N={N}, d={d}, k={k}")
    A_cpu = load_graph(graph_path=graph_path, expected_N=N, expected_n=n, expected_k=k)
    A_device = A_cpu.to(device=resolved_device)

    config = {
        "graph_path": str(Path(graph_path).resolve()),
        "n": n,
        "d": d,
        "k": k,
        "N": N,
        "batch_size_requested": batch_size,
        "batching_mode": "full_batch",
        "initialization": initialization,
        "loss_type": "infonce",
        "num_steps": num_steps,
        "save_every": save_every,
        "save_path": str(save_dir),
        "lr": lr,
        "min_lr_ratio": min_lr_ratio,
        "warmup_frac": warmup_frac,
        "temperature": temperature,
        "seed": seed,
        "device": str(resolved_device),
    }

    save_npz(save_dir / "graph_data.npz", A=A_cpu.to(torch.uint8))
    with open(save_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"Saved graph_data.npz and config.json to {save_dir}")
    print(f"Initializing U and V with '{initialization}' initialization...")
    U, V = initialize_embeddings(
        A_cpu=A_cpu,
        N=N,
        n=n,
        d=d,
        initialization=initialization,
        device=resolved_device,
    )

    optimizer, scheduler = make_optimizer_and_scheduler(
        params=[U, V],
        num_steps=num_steps,
        lr=lr,
        min_lr_ratio=min_lr_ratio,
        warmup_frac=warmup_frac,
    )

    last_loss = float("nan")
    pos_min = float("nan")
    neg_max = float("nan")
    gap = float("nan")
    accuracy = float("nan")
    best_accuracy = 0.0
    best_loss = float("inf")

    for step in range(1, num_steps + 1):
        last_loss = train_one_step(
            U=U,
            V=V,
            A_device=A_device,
            optimizer=optimizer,
            temperature=temperature,
        )
        scheduler.step()

        if last_loss < best_loss:
            best_loss = last_loss

        should_save = (step % save_every == 0) or (step == num_steps)
        if should_save:
            pos_min, neg_max, gap = compute_gap(U=U, V=V, A_device=A_device)
            accuracy = evaluate_top_k_accuracy(U=U, V=V, A_device=A_device, k=k)
            best_accuracy = max(best_accuracy, accuracy)
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"step={step:6d} | loss={last_loss:.6f} | lr={current_lr:.6e} | "
                f"acc={accuracy:.6f} | best_acc={best_accuracy:.6f} | gap={gap:.6f}"
            )
            save_checkpoint_npz(
                save_dir=save_dir,
                step=step,
                U=U,
                V=V,
                loss_value=last_loss,
                pos_min=pos_min,
                neg_max=neg_max,
                gap=gap,
                accuracy=accuracy,
                best_accuracy=best_accuracy,
                best_loss=best_loss,
                lr=current_lr,
                temperature=temperature,
            )

    b_placeholder = np.asarray([[np.nan]], dtype=np.float32)
    t_placeholder = np.asarray([[np.nan]], dtype=np.float32)
    save_npz(
        save_dir / "final.npz",
        U=U,
        V=V,
        b=b_placeholder,
        t=t_placeholder,
        loss=last_loss,
        pos_min=pos_min,
        neg_max=neg_max,
        gap=gap,
        accuracy=accuracy,
        best_accuracy=best_accuracy,
        best_loss=best_loss,
        temperature=temperature,
    )
    print(f"Training complete. Final artifacts saved under: {save_dir}")

    return {
        "save_dir": str(save_dir),
        "device": str(resolved_device),
    }



def main() -> None:
    args = parse_args()
    train_infonce_on_saved_graph(
        graph_path=args.graph_path,
        n=args.n,
        d=args.d,
        k=args.k,
        N=args.N,
        batch_size=args.batch_size,
        initialization=args.initialization,
        num_steps=args.num_steps,
        save_every=args.save_every,
        save_path=args.save_path,
        lr=args.lr,
        min_lr_ratio=args.min_lr_ratio,
        warmup_frac=args.warmup_frac,
        temperature=args.temperature,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()
