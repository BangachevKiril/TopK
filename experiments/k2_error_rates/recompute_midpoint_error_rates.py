#!/usr/bin/env python3
"""Recompute k=2 FPR/FNR using the midpoint of class-mean scores.

This reads saved final U and V matrices, so it performs no retraining and does
not modify the original threshold-zero result files.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RUN_RE = re.compile(r"graph_n_(\d+)_k_2_N_(\d+)_seed_(\d+)")


def parse_ints(raw: str) -> list[int]:
    return [int(value) for value in raw.split()]


def load_positive_indices(graph_path: Path, expected_n: int, expected_N: int) -> np.ndarray:
    with np.load(graph_path, allow_pickle=False) as graph:
        indices = np.asarray(graph["indices"], dtype=np.int64)
        indptr = np.asarray(graph["indptr"], dtype=np.int64)
        shape = tuple(int(value) for value in graph["shape"])
    if shape != (expected_N, expected_n):
        raise ValueError(f"{graph_path}: expected {(expected_N, expected_n)}, found {shape}")
    row_counts = np.diff(indptr)
    if not np.all(row_counts == 2):
        raise ValueError(f"{graph_path}: not every row has exactly two positives")
    return indices.reshape(expected_N, 2)


def class_statistics(
    U: np.ndarray,
    V: np.ndarray,
    positive_indices: np.ndarray,
    row_chunk_size: int,
) -> tuple[float, float, float, int, int, float, int, int, float]:
    N, d = U.shape
    n, other_d = V.shape
    if d != other_d or positive_indices.shape != (N, 2):
        raise ValueError("incompatible U, V, or positive-index shapes")

    positive_count = N * 2
    negative_count = N * n - positive_count
    all_sum = 0.0
    positive_sum = 0.0

    for start in range(0, N, row_chunk_size):
        end = min(start + row_chunk_size, N)
        scores = U[start:end] @ V.T
        rows = np.arange(end - start)[:, None]
        positive_scores = scores[rows, positive_indices[start:end]]
        all_sum += float(scores.sum(dtype=np.float64))
        positive_sum += float(positive_scores.sum(dtype=np.float64))

    positive_mean = positive_sum / positive_count
    negative_mean = (all_sum - positive_sum) / negative_count
    threshold = 0.5 * (positive_mean + negative_mean)

    predicted_positive_count = 0
    true_positive_count = 0
    for start in range(0, N, row_chunk_size):
        end = min(start + row_chunk_size, N)
        scores = U[start:end] @ V.T
        rows = np.arange(end - start)[:, None]
        positive_scores = scores[rows, positive_indices[start:end]]
        predicted_positive_count += int(np.count_nonzero(scores > threshold))
        true_positive_count += int(np.count_nonzero(positive_scores > threshold))

    false_positive_count = predicted_positive_count - true_positive_count
    false_negative_count = positive_count - true_positive_count
    false_positive_rate = false_positive_count / negative_count
    false_negative_rate = false_negative_count / positive_count
    return (
        positive_mean,
        negative_mean,
        threshold,
        false_positive_count,
        negative_count,
        false_positive_rate,
        false_negative_count,
        positive_count,
        false_negative_rate,
    )


def extrema_statistics(
    U: np.ndarray,
    V: np.ndarray,
    positive_indices: np.ndarray,
    positive_minimum: float,
    negative_maximum: float,
    row_chunk_size: int,
) -> tuple[float, int, int, float, int, int, float]:
    N, d = U.shape
    n, other_d = V.shape
    if d != other_d or positive_indices.shape != (N, 2):
        raise ValueError("incompatible U, V, or positive-index shapes")

    threshold = 0.5 * (positive_minimum + negative_maximum)
    positive_count = N * 2
    negative_count = N * n - positive_count
    predicted_positive_count = 0
    true_positive_count = 0

    for start in range(0, N, row_chunk_size):
        end = min(start + row_chunk_size, N)
        scores = U[start:end] @ V.T
        rows = np.arange(end - start)[:, None]
        positive_scores = scores[rows, positive_indices[start:end]]
        predicted_positive_count += int(np.count_nonzero(scores > threshold))
        true_positive_count += int(np.count_nonzero(positive_scores > threshold))

    false_positive_count = predicted_positive_count - true_positive_count
    false_negative_count = positive_count - true_positive_count
    return (
        threshold,
        false_positive_count,
        negative_count,
        false_positive_count / negative_count,
        false_negative_count,
        positive_count,
        false_negative_count / positive_count,
    )


def compute_rows(
    embeddings_root: Path,
    graph_root: Path,
    n_values: list[int],
    d_values: list[int],
    seed: int,
    row_chunk_size: int,
    threshold_mode: str,
) -> list[dict[str, int | float | str]]:
    rows: list[dict[str, int | float | str]] = []
    graph_cache: dict[tuple[int, int], np.ndarray] = {}

    for loss_dir, loss_label in (("infonce", "InfoNCE"), ("sigmoid", "Sigmoid")):
        for n in n_values:
            N = n * (n - 1) // 2
            run_name = f"graph_n_{n}_k_2_N_{N}_seed_{seed}"
            match = RUN_RE.fullmatch(run_name)
            if match is None:
                raise AssertionError(run_name)
            graph_path = graph_root / f"{run_name}.npz"
            cache_key = (n, N)
            if cache_key not in graph_cache:
                graph_cache[cache_key] = load_positive_indices(graph_path, n, N)
            positive_indices = graph_cache[cache_key]

            for d in d_values:
                final_path = embeddings_root / loss_dir / run_name / f"d_{d}" / "final.npz"
                with np.load(final_path, allow_pickle=False) as final:
                    U = np.asarray(final["U"], dtype=np.float32)
                    V = np.asarray(final["V"], dtype=np.float32)
                    positive_minimum = float(final["pos_min"])
                    negative_maximum = float(final["neg_max"])

                common: dict[str, int | float | str] = {
                        "loss": loss_label,
                        "n": n,
                        "N": N,
                        "d": d,
                }
                if threshold_mode == "class-means":
                    (
                        positive_mean,
                        negative_mean,
                        threshold,
                        false_positive_count,
                        negative_count,
                        false_positive_rate,
                        false_negative_count,
                        positive_count,
                        false_negative_rate,
                    ) = class_statistics(U, V, positive_indices, row_chunk_size)
                    common.update(
                        {
                        "positive_score_mean": positive_mean,
                        "negative_score_mean": negative_mean,
                        "midpoint_threshold": threshold,
                        }
                    )
                elif threshold_mode == "extrema":
                    (
                        threshold,
                        false_positive_count,
                        negative_count,
                        false_positive_rate,
                        false_negative_count,
                        positive_count,
                        false_negative_rate,
                    ) = extrema_statistics(
                        U,
                        V,
                        positive_indices,
                        positive_minimum,
                        negative_maximum,
                        row_chunk_size,
                    )
                    common.update(
                        {
                            "positive_score_minimum": positive_minimum,
                            "negative_score_maximum": negative_maximum,
                            "extrema_midpoint_threshold": threshold,
                            "margin": positive_minimum - negative_maximum,
                        }
                    )
                else:
                    raise ValueError(f"unsupported threshold mode: {threshold_mode}")
                common.update(
                    {
                        "false_positive_count": false_positive_count,
                        "negative_count": negative_count,
                        "false_positive_rate": false_positive_rate,
                        "false_negative_count": false_negative_count,
                        "positive_count": positive_count,
                        "false_negative_rate": false_negative_rate,
                    }
                )
                rows.append(common)
                print(
                    f"{loss_label} n={n} d={d} threshold={threshold:.8g} "
                    f"FPR={false_positive_rate:.8g} FNR={false_negative_rate:.8g}",
                    flush=True,
                )
    return rows


def write_summary(rows: list[dict[str, int | float | str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_rows(
    rows: list[dict[str, int | float | str]],
    n_values: list[int],
    d_values: list[int],
    output_stem: Path,
    threshold_mode: str,
) -> None:
    figure, axes = plt.subplots(
        len(n_values),
        2,
        figsize=(13, 2.65 * len(n_values) + 2.2),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)
    by_key = {(row["loss"], row["n"], row["d"]): row for row in rows}

    for row_index, n in enumerate(n_values):
        for column_index, loss in enumerate(("InfoNCE", "Sigmoid")):
            axis = axes[row_index, column_index]
            fpr = [float(by_key[(loss, n, d)]["false_positive_rate"]) for d in d_values]
            fnr = [float(by_key[(loss, n, d)]["false_negative_rate"]) for d in d_values]
            axis.plot(d_values, fpr, marker="o", markersize=3.5, label="False positive rate")
            axis.plot(d_values, fnr, marker="s", markersize=3.5, label="False negative rate")
            axis.set_yscale("symlog", linthresh=1e-5, linscale=0.8)
            axis.set_ylim(-2e-7, 1.05)
            axis.grid(True, alpha=0.25)
            if row_index == 0:
                axis.set_title(loss, fontsize=16)
            if column_index == 0:
                axis.set_ylabel(f"n={n}\nError probability", fontsize=11)
            if row_index == len(n_values) - 1:
                axis.set_xlabel("Embedding dimension d", fontsize=11)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.968))
    if threshold_mode == "class-means":
        title = "k=2 final error probabilities at midpoint of positive/negative mean scores"
    else:
        title = "k=2 final error probabilities at midpoint of minimum-positive/maximum-negative scores"
    figure.suptitle(title, fontsize=19)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_stem.with_suffix(".png"), dpi=180)
    figure.savefig(output_stem.with_suffix(".pdf"))
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings-root", type=Path, required=True)
    parser.add_argument("--graph-root", type=Path, required=True)
    parser.add_argument("--plot-dir", type=Path, required=True)
    parser.add_argument("--n-values", default="20 40 60 80 100 120 140 160 180 200 220 240")
    parser.add_argument("--d-values", default="5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--row-chunk-size", type=int, default=4096)
    parser.add_argument(
        "--threshold-mode",
        choices=("class-means", "extrema"),
        default="class-means",
    )
    parser.add_argument("--output-prefix")
    args = parser.parse_args()

    n_values = parse_ints(args.n_values)
    d_values = parse_ints(args.d_values)
    rows = compute_rows(
        args.embeddings_root,
        args.graph_root,
        n_values,
        d_values,
        args.seed,
        args.row_chunk_size,
        args.threshold_mode,
    )
    if args.output_prefix is None:
        if args.threshold_mode == "class-means":
            args.output_prefix = "k2_midpoint_false_positive_negative_rates"
        else:
            args.output_prefix = "k2_extrema_midpoint_false_positive_negative_rates"
    output_stem = args.plot_dir / args.output_prefix
    write_summary(rows, output_stem.with_name(f"{args.output_prefix}_summary.csv"))
    plot_rows(rows, n_values, d_values, output_stem, args.threshold_mode)


if __name__ == "__main__":
    main()
