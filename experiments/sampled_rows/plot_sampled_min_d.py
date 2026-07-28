#!/usr/bin/env python3
"""Plot minimal positive-margin dimension versus sampled row count N."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


GRAPH_RE = re.compile(r"graph_n_(\d+)_k_(\d+)_N_(\d+)_seed_(\d+)")
D_RE = re.compile(r"d_(\d+)")


def parse_int_list(raw: str) -> list[int]:
    return [int(value) for value in raw.replace(",", " ").split()]


def load_margin(path: Path) -> float:
    with np.load(path, allow_pickle=False) as data:
        return float(np.asarray(data["margin"]).reshape(-1)[-1])


def discover(root: Path, n: int, seed: int) -> dict[tuple[int, int], dict[int, float]]:
    result: dict[tuple[int, int], dict[int, float]] = {}
    for final_path in root.glob("graph_n_*_k_*_N_*_seed_*/d_*/final.npz"):
        graph_match = GRAPH_RE.fullmatch(final_path.parent.parent.name)
        d_match = D_RE.fullmatch(final_path.parent.name)
        if graph_match is None or d_match is None:
            continue
        run_n, k, N, run_seed = map(int, graph_match.groups())
        if run_n != n or run_seed != seed:
            continue
        result.setdefault((k, N), {})[int(d_match.group(1))] = load_margin(final_path)
    return result


def min_positive(values: dict[int, float]) -> int | None:
    positive = [d for d, margin in values.items() if np.isfinite(margin) and margin > 0]
    return min(positive) if positive else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--infonce-root", type=Path, required=True)
    parser.add_argument("--sigmoid-root", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k-values", type=str, default="4 5 6")
    parser.add_argument("--N-values", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    k_values = parse_int_list(args.k_values)
    N_values = parse_int_list(args.N_values)
    infonce = discover(args.infonce_root, args.n, args.seed)
    sigmoid = discover(args.sigmoid_root, args.n, args.seed)

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_prefix.with_name(args.output_prefix.name + "_summary.csv")
    rows: list[dict[str, object]] = []

    fig, axes = plt.subplots(1, len(k_values), figsize=(5.6 * len(k_values), 5.2), sharey=True)
    if len(k_values) == 1:
        axes = [axes]

    for ax, k in zip(axes, k_values):
        info_y: list[float] = []
        sig_y: list[float] = []
        for N in N_values:
            info_min = min_positive(infonce.get((k, N), {}))
            sig_min = min_positive(sigmoid.get((k, N), {}))
            info_y.append(float(info_min) if info_min is not None else np.nan)
            sig_y.append(float(sig_min) if sig_min is not None else np.nan)
            rows.extend(
                [
                    {
                        "loss": "InfoNCE",
                        "n": args.n,
                        "k": k,
                        "N": N,
                        "min_positive_d": "" if info_min is None else info_min,
                        "dimensions_run": " ".join(map(str, sorted(infonce.get((k, N), {})))),
                    },
                    {
                        "loss": "Sigmoid",
                        "n": args.n,
                        "k": k,
                        "N": N,
                        "min_positive_d": "" if sig_min is None else sig_min,
                        "dimensions_run": " ".join(map(str, sorted(sigmoid.get((k, N), {})))),
                    },
                ]
            )

        ax.plot(N_values, info_y, marker="o", linewidth=2.2, color="#d62728", label="InfoNCE")
        ax.plot(N_values, sig_y, marker="o", linewidth=2.2, color="#1f77b4", label="Sigmoid")
        ax.set_xscale("log", base=2)
        ax.set_xticks(N_values, [str(value) for value in N_values], rotation=45)
        ax.set_title(f"k = {k}", fontsize=15)
        ax.set_xlabel("N sampled rows", fontsize=13)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Minimal d with positive margin", fontsize=13)
    axes[-1].legend(fontsize=12)
    fig.suptitle(
        f"Fixed iid sampled k-sparse matrices, n = {args.n} (seed {args.seed})",
        fontsize=16,
    )
    fig.tight_layout()
    fig.savefig(args.output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(args.output_prefix.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)

    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["loss", "n", "k", "N", "min_positive_d", "dimensions_run"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"saved={args.output_prefix.with_suffix('.pdf')}")
    print(f"saved={args.output_prefix.with_suffix('.png')}")
    print(f"saved={summary_path}")


if __name__ == "__main__":
    main()
