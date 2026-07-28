#!/usr/bin/env python3
"""Train independent TopK embeddings for many dimensions in one GPU process.

The dimension axis is batched for throughput, but every d has its own U, V,
temperature, Adam moments, and margin history. Inactive padded coordinates are
kept at zero, so each batch element is mathematically the corresponding
standalone experiment. At the end of training, this variant also records
class-conditional false-positive and false-negative rates at score threshold 0.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F


def parse_int_list(raw: str) -> list[int]:
    return [int(value) for value in raw.replace(",", " ").split()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loss", choices=["infonce", "sigmoid"], required=True)
    parser.add_argument("--graph-path", type=Path, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--d-values", default="5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30")
    parser.add_argument("--num-steps", type=int, default=100_000)
    parser.add_argument("--save-every", type=int, default=1_000)
    parser.add_argument("--save-path", type=Path, required=True)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--min-lr-ratio", type=float, default=0.01)
    parser.add_argument("--warmup-frac", type=float, default=0.05)
    parser.add_argument("--relative-bias", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--row-chunk-size", type=int, default=2048)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_neighborhoods(path: Path, N: int, n: int, k: int) -> torch.Tensor:
    graph = sp.load_npz(path).tocsr()
    graph.sort_indices()
    if graph.shape != (N, n):
        raise ValueError(f"Expected graph shape {(N, n)}, got {graph.shape}.")
    row_counts = np.diff(graph.indptr)
    if not np.all(row_counts == k):
        bad = np.flatnonzero(row_counts != k)[:10].tolist()
        raise ValueError(f"Rows {bad} do not contain exactly k={k} nonzeros.")
    if not np.all(graph.data == 1):
        raise ValueError("Graph entries must all equal one.")
    return torch.from_numpy(graph.indices.reshape(N, k).astype(np.int64, copy=True))


@torch.no_grad()
def normalize_rows(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


@torch.no_grad()
def normalize_rows_inplace(x: torch.Tensor, eps: float = 1e-12) -> None:
    x.div_(x.norm(dim=-1, keepdim=True).clamp_min(eps))


def initialize_batched(
    N: int,
    n: int,
    d_values: list[int],
    loss: str,
    relative_bias: float,
    temperature: float,
    seed: int,
    device: torch.device,
) -> tuple[torch.nn.Parameter, torch.nn.Parameter, torch.Tensor, torch.nn.Parameter]:
    batch = len(d_values)
    d_max = max(d_values)
    U0 = torch.zeros((batch, N, d_max), dtype=torch.float32, device=device)
    V0 = torch.zeros((batch, n, d_max), dtype=torch.float32, device=device)

    # Reset to the same seed for every d, exactly as in separate CLI runs.
    for index, d in enumerate(d_values):
        set_seed(seed)
        U0[index, :, :d] = normalize_rows(torch.randn((N, d), device=device))
        V0[index, :, :d] = normalize_rows(torch.randn((n, d), device=device))

    U = torch.nn.Parameter(U0)
    V = torch.nn.Parameter(V0)
    b = torch.full(
        (batch, 1, 1),
        float(relative_bias),
        dtype=torch.float32,
        device=device,
    )
    initial_t = temperature if loss == "infonce" else 1.0
    t = torch.nn.Parameter(
        torch.full((batch, 1, 1), float(initial_t), dtype=torch.float32, device=device)
    )
    return U, V, b, t


def make_optimizer_and_scheduler(
    parameters: list[torch.Tensor],
    num_steps: int,
    lr: float,
    min_lr_ratio: float,
    warmup_frac: float,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    optimizer = torch.optim.Adam(parameters, lr=lr)
    warmup_steps = max(1, int(round(num_steps * warmup_frac))) if warmup_frac > 0 else 0
    decay_steps = max(1, num_steps - warmup_steps)
    if warmup_steps:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=decay_steps, eta_min=lr * min_lr_ratio
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps]
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_steps, eta_min=lr * min_lr_ratio
        )
    return optimizer, scheduler


def train_step(
    loss_name: str,
    U: torch.nn.Parameter,
    V: torch.nn.Parameter,
    b: torch.Tensor,
    t: torch.nn.Parameter,
    neighborhoods: torch.Tensor,
    adjacency: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    row_chunk_size: int,
) -> np.ndarray:
    batch, N, _ = U.shape
    optimizer.zero_grad(set_to_none=True)
    loss_by_d = torch.zeros(batch, dtype=torch.float64, device=U.device)
    V_transposed = V.transpose(1, 2)
    total_positives = float(N * neighborhoods.shape[1])

    for start in range(0, N, row_chunk_size):
        end = min(start + row_chunk_size, N)
        scores = torch.bmm(U[:, start:end, :], V_transposed)

        if loss_name == "infonce":
            logits = (scores - b) / t
            log_probs = F.log_softmax(logits, dim=2)
            positive_index = neighborhoods[start:end].unsqueeze(0).expand(batch, -1, -1)
            per_d = -log_probs.gather(2, positive_index).sum(dim=(1, 2)) / total_positives
        else:
            signs = 1.0 - 2.0 * adjacency[start:end].to(dtype=U.dtype)
            per_d = F.softplus(t * (scores - b) * signs.unsqueeze(0)).sum(dim=(1, 2))

        per_d.sum().backward()
        loss_by_d += per_d.detach().to(torch.float64)

    optimizer.step()
    with torch.no_grad():
        normalize_rows_inplace(U)
        normalize_rows_inplace(V)
        if loss_name == "infonce":
            t.clamp_(min=1e-4, max=1e2)
    return loss_by_d.cpu().numpy()


@torch.no_grad()
def compute_margins(
    U: torch.Tensor,
    V: torch.Tensor,
    neighborhoods: torch.Tensor,
    row_chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    batch, N, _ = U.shape
    pos_min = torch.full((batch,), float("inf"), device=U.device)
    neg_max = torch.full((batch,), float("-inf"), device=U.device)
    V_transposed = V.transpose(1, 2)

    for start in range(0, N, row_chunk_size):
        end = min(start + row_chunk_size, N)
        scores = torch.bmm(U[:, start:end, :], V_transposed)
        positive_index = neighborhoods[start:end].unsqueeze(0).expand(batch, -1, -1)
        chunk_pos = scores.gather(2, positive_index).amin(dim=(1, 2))
        masked = scores.clone()
        masked.scatter_(2, positive_index, float("-inf"))
        chunk_neg = masked.amax(dim=(1, 2))
        pos_min = torch.minimum(pos_min, chunk_pos)
        neg_max = torch.maximum(neg_max, chunk_neg)

    margin = 0.5 * (pos_min - neg_max)
    return (
        pos_min.cpu().numpy(),
        neg_max.cpu().numpy(),
        margin.cpu().numpy(),
    )


@torch.no_grad()
def compute_error_rates(
    U: torch.Tensor,
    V: torch.Tensor,
    neighborhoods: torch.Tensor,
    row_chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return false-positive/negative counts and class-conditional rates.

    A pair is predicted positive exactly when its score is strictly greater
    than zero. Thus a positive pair with score <= 0 is a false negative.
    """

    batch, N, _ = U.shape
    k = neighborhoods.shape[1]
    n = V.shape[1]
    false_positives = torch.zeros(batch, dtype=torch.int64, device=U.device)
    false_negatives = torch.zeros(batch, dtype=torch.int64, device=U.device)
    V_transposed = V.transpose(1, 2)

    for start in range(0, N, row_chunk_size):
        end = min(start + row_chunk_size, N)
        scores = torch.bmm(U[:, start:end, :], V_transposed)
        positive_index = neighborhoods[start:end].unsqueeze(0).expand(batch, -1, -1)
        positive_scores = scores.gather(2, positive_index)

        predicted_positive = scores > 0
        positive_predicted_positive = positive_scores > 0
        false_positives += (
            predicted_positive.sum(dim=(1, 2))
            - positive_predicted_positive.sum(dim=(1, 2))
        )
        false_negatives += (positive_scores <= 0).sum(dim=(1, 2))

    negative_count = N * (n - k)
    positive_count = N * k
    false_positive_rates = false_positives.to(torch.float64) / negative_count
    false_negative_rates = false_negatives.to(torch.float64) / positive_count
    return (
        false_positives.cpu().numpy(),
        false_negatives.cpu().numpy(),
        false_positive_rates.cpu().numpy(),
        false_negative_rates.cpu().numpy(),
    )


def atomic_save_npz(path: Path, **arrays: Any) -> None:
    temp_path = path.with_name(path.name + ".tmp")
    with temp_path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temp_path, path)


def atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    temp_path = path.with_name(path.name + ".tmp")
    torch.save(payload, temp_path)
    os.replace(temp_path, path)


def save_history(
    path: Path,
    d_values: list[int],
    steps: list[int],
    losses: list[np.ndarray],
    pos_mins: list[np.ndarray],
    neg_maxes: list[np.ndarray],
    margins: list[np.ndarray],
    lrs: list[float],
    temperatures: list[np.ndarray],
) -> None:
    atomic_save_npz(
        path,
        d_values=np.asarray(d_values, dtype=np.int64),
        steps=np.asarray(steps, dtype=np.int64),
        losses=np.asarray(losses, dtype=np.float64),
        pos_mins=np.asarray(pos_mins, dtype=np.float64),
        neg_maxes=np.asarray(neg_maxes, dtype=np.float64),
        margins=np.asarray(margins, dtype=np.float64),
        learning_rates=np.asarray(lrs, dtype=np.float64),
        temperatures=np.asarray(temperatures, dtype=np.float64),
    )


def main() -> None:
    args = parse_args()
    d_values = parse_int_list(args.d_values)
    if not d_values or len(set(d_values)) != len(d_values):
        raise ValueError("d-values must be a nonempty list without duplicates.")
    if min(d_values) <= 0 or args.num_steps <= 0 or args.save_every <= 0:
        raise ValueError("Dimensions, num-steps, and save-every must be positive.")
    if not (1 <= args.k < args.n):
        raise ValueError("Expected 1 <= k < n.")
    if args.row_chunk_size <= 0:
        raise ValueError("row-chunk-size must be positive.")

    device = torch.device(args.device)
    set_seed(args.seed)
    neighborhoods_cpu = load_neighborhoods(args.graph_path, args.N, args.n, args.k)
    neighborhoods = neighborhoods_cpu.to(device=device)
    adjacency = torch.zeros((args.N, args.n), dtype=torch.bool, device=device)
    adjacency.scatter_(1, neighborhoods, True)

    args.save_path.mkdir(parents=True, exist_ok=True)
    history_path = args.save_path / "margin_history.npz"
    resume_path = args.save_path / "resume_state.pt"
    config_path = args.save_path / "batched_config.json"

    config = {
        "loss": args.loss,
        "graph_path": str(args.graph_path),
        "n": args.n,
        "N": args.N,
        "k": args.k,
        "d_values": d_values,
        "num_steps": args.num_steps,
        "save_every": args.save_every,
        "save_path": str(args.save_path),
        "lr": args.lr,
        "min_lr_ratio": args.min_lr_ratio,
        "warmup_frac": args.warmup_frac,
        "relative_bias": args.relative_bias,
        "temperature": args.temperature,
        "seed": args.seed,
        "device": str(device),
        "row_chunk_size": args.row_chunk_size,
        "training_mode": "independent_dimensions_batched_on_one_gpu",
        "classification_threshold": 0.0,
        "false_positive_rate": "count(A_ij=0 and score>0) / count(A_ij=0)",
        "false_negative_rate": "count(A_ij=1 and score<=0) / count(A_ij=1)",
    }
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    U, V, b, t = initialize_batched(
        N=args.N,
        n=args.n,
        d_values=d_values,
        loss=args.loss,
        relative_bias=args.relative_bias,
        temperature=args.temperature,
        seed=args.seed,
        device=device,
    )
    optimizer, scheduler = make_optimizer_and_scheduler(
        [U, V, t],
        num_steps=args.num_steps,
        lr=args.lr,
        min_lr_ratio=args.min_lr_ratio,
        warmup_frac=args.warmup_frac,
    )

    steps: list[int] = []
    losses: list[np.ndarray] = []
    pos_mins: list[np.ndarray] = []
    neg_maxes: list[np.ndarray] = []
    margins: list[np.ndarray] = []
    lrs: list[float] = []
    temperatures: list[np.ndarray] = []
    start_step = 0

    if resume_path.is_file():
        state = torch.load(resume_path, map_location=device, weights_only=False)
        if state["d_values"] != d_values or state["loss"] != args.loss:
            raise ValueError(f"Resume metadata mismatch in {resume_path}.")
        U.data.copy_(state["U"])
        V.data.copy_(state["V"])
        t.data.copy_(state["t"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_step = int(state["step"])
        steps = list(state["history"]["steps"])
        losses = [np.asarray(value) for value in state["history"]["losses"]]
        pos_mins = [np.asarray(value) for value in state["history"]["pos_mins"]]
        neg_maxes = [np.asarray(value) for value in state["history"]["neg_maxes"]]
        margins = [np.asarray(value) for value in state["history"]["margins"]]
        lrs = list(state["history"]["lrs"])
        temperatures = [np.asarray(value) for value in state["history"]["temperatures"]]
        print(f"Resuming from step {start_step} in {resume_path}", flush=True)

    last_loss = (
        losses[-1].copy()
        if losses
        else np.full(len(d_values), np.nan, dtype=np.float64)
    )
    for step in range(start_step + 1, args.num_steps + 1):
        last_loss = train_step(
            loss_name=args.loss,
            U=U,
            V=V,
            b=b,
            t=t,
            neighborhoods=neighborhoods,
            adjacency=adjacency,
            optimizer=optimizer,
            row_chunk_size=args.row_chunk_size,
        )
        scheduler.step()

        if step % args.save_every == 0 or step == args.num_steps:
            pos_min, neg_max, margin = compute_margins(
                U=U,
                V=V,
                neighborhoods=neighborhoods,
                row_chunk_size=args.row_chunk_size,
            )
            current_lr = float(optimizer.param_groups[0]["lr"])
            current_t = t.detach().reshape(-1).cpu().numpy().copy()
            steps.append(step)
            losses.append(last_loss.copy())
            pos_mins.append(pos_min.copy())
            neg_maxes.append(neg_max.copy())
            margins.append(margin.copy())
            lrs.append(current_lr)
            temperatures.append(current_t)
            save_history(
                history_path,
                d_values,
                steps,
                losses,
                pos_mins,
                neg_maxes,
                margins,
                lrs,
                temperatures,
            )
            atomic_torch_save(
                resume_path,
                {
                    "step": step,
                    "loss": args.loss,
                    "d_values": d_values,
                    "U": U.detach(),
                    "V": V.detach(),
                    "t": t.detach(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "history": {
                        "steps": steps,
                        "losses": losses,
                        "pos_mins": pos_mins,
                        "neg_maxes": neg_maxes,
                        "margins": margins,
                        "lrs": lrs,
                        "temperatures": temperatures,
                    },
                },
            )
            margin_text = " ".join(
                f"d={d}:{value:.6f}" for d, value in zip(d_values, margin)
            )
            print(
                f"step={step} lr={current_lr:.6e} margins=[{margin_text}]",
                flush=True,
            )

    margin_array = np.asarray(margins)
    best_indices = np.nanargmax(margin_array, axis=0)
    final_pos = np.asarray(pos_mins)[-1]
    final_neg = np.asarray(neg_maxes)[-1]
    final_margin = margin_array[-1]
    final_t = t.detach().reshape(-1).cpu().numpy()
    (
        false_positive_counts,
        false_negative_counts,
        false_positive_rates,
        false_negative_rates,
    ) = compute_error_rates(
        U=U,
        V=V,
        neighborhoods=neighborhoods,
        row_chunk_size=args.row_chunk_size,
    )
    negative_count = args.N * (args.n - args.k)
    positive_count = args.N * args.k

    for index, d in enumerate(d_values):
        run_dir = args.save_path / f"d_{d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        best_index = int(best_indices[index])
        atomic_save_npz(
            run_dir / "final.npz",
            U=U[index, :, :d].detach().cpu().numpy(),
            V=V[index, :, :d].detach().cpu().numpy(),
            b=np.asarray([[args.relative_bias]], dtype=np.float32),
            t=np.asarray([[final_t[index]]], dtype=np.float32),
            loss=np.asarray(last_loss[index]),
            pos_min=np.asarray(final_pos[index]),
            neg_max=np.asarray(final_neg[index]),
            margin=np.asarray(final_margin[index]),
            best_margin=np.asarray(margin_array[best_index, index]),
            best_step=np.asarray(steps[best_index], dtype=np.int64),
            false_positive_count=np.asarray(
                false_positive_counts[index], dtype=np.int64
            ),
            negative_count=np.asarray(negative_count, dtype=np.int64),
            false_positive_rate=np.asarray(false_positive_rates[index]),
            false_negative_count=np.asarray(
                false_negative_counts[index], dtype=np.int64
            ),
            positive_count=np.asarray(positive_count, dtype=np.int64),
            false_negative_rate=np.asarray(false_negative_rates[index]),
        )
        per_d_config = dict(config)
        per_d_config["d"] = d
        per_d_config["save_path"] = str(run_dir)
        (run_dir / "config.json").write_text(
            json.dumps(per_d_config, indent=2) + "\n", encoding="utf-8"
        )

    (args.save_path / "COMPLETE").write_text(
        f"step={args.num_steps}\ndimensions={' '.join(map(str, d_values))}\n",
        encoding="utf-8",
    )
    print(f"Training complete: {args.save_path}", flush=True)


if __name__ == "__main__":
    main()
