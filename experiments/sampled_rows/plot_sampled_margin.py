#!/usr/bin/env python3
"""Plot largest positive margin achieved during the sampled-row training."""

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


def discover(
    root: Path, n: int, seed: int
) -> dict[tuple[int, int, int], tuple[float, int, float]]:
    result: dict[tuple[int, int, int], tuple[float, int, float]] = {}
    for final_path in root.glob("graph_n_*_k_*_N_*_seed_*/d_*/final.npz"):
        graph_match = GRAPH_RE.fullmatch(final_path.parent.parent.name)
        d_match = D_RE.fullmatch(final_path.parent.name)
        if graph_match is None or d_match is None:
            continue
        run_n, k, N, run_seed = map(int, graph_match.groups())
        if run_n != n or run_seed != seed:
            continue
        d = int(d_match.group(1))
        with np.load(final_path, allow_pickle=False) as data:
            final_margin = float(np.asarray(data["margin"]).reshape(-1)[-1])
            if "best_margin" in data:
                best_margin = float(np.asarray(data["best_margin"]).reshape(-1)[-1])
                best_step = int(np.asarray(data["best_step"]).reshape(-1)[-1])
            else:
                best_margin = final_margin
                best_step = 100_000
        result[(k, N, d)] = (best_margin, best_step, final_margin)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--infonce-root", type=Path, required=True)
    parser.add_argument("--sigmoid-root", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k-values", default="4 5 6")
    parser.add_argument("--N-values", required=True)
    parser.add_argument("--d-values", default="10 20 30")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    k_values = parse_int_list(args.k_values)
    N_values = parse_int_list(args.N_values)
    d_values = parse_int_list(args.d_values)
    datasets = {
        "InfoNCE": discover(args.infonce_root, args.n, args.seed),
        "Sigmoid": discover(args.sigmoid_root, args.n, args.seed),
    }
    colors = {d: plt.cm.viridis(index) for d, index in zip(d_values, np.linspace(0.15, 0.85, len(d_values)))}
    styles = {
        "InfoNCE": {"linestyle": "-", "marker": "o"},
        "Sigmoid": {"linestyle": "--", "marker": "s"},
    }

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_prefix.with_name(args.output_prefix.name + "_summary.csv")
    rows: list[dict[str, object]] = []
    fig, axes = plt.subplots(1, len(k_values), figsize=(5.8 * len(k_values), 5.2), sharey=True)
    if len(k_values) == 1:
        axes = [axes]

    for ax, k in zip(axes, k_values):
        for loss_name, data in datasets.items():
            for d in d_values:
                values: list[float] = []
                for N in N_values:
                    record = data.get((k, N, d))
                    values.append(np.nan if record is None else max(0.0, record[0]))
                    rows.append(
                        {
                            "loss": loss_name,
                            "n": args.n,
                            "k": k,
                            "N": N,
                            "d": d,
                            "largest_positive_margin": "" if record is None else max(0.0, record[0]),
                            "best_step": "" if record is None else record[1],
                            "final_margin": "" if record is None else record[2],
                        }
                    )
                ax.plot(
                    N_values,
                    values,
                    color=colors[d],
                    linewidth=2.0,
                    markersize=5,
                    label=f"{loss_name}, d={d}",
                    **styles[loss_name],
                )
        ax.set_xscale("log", base=2)
        ax.set_xticks(N_values, [str(value) for value in N_values], rotation=45)
        ax.set_title(f"k = {k}", fontsize=15)
        ax.set_xlabel("N sampled rows", fontsize=13)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Largest positive margin", fontsize=13)
    axes[-1].legend(fontsize=9, ncol=2)
    fig.suptitle(
        f"Maximal margin during 100000 steps, fixed iid matrices, n = {args.n}",
        fontsize=16,
    )
    fig.tight_layout()
    fig.savefig(args.output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(args.output_prefix.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)

    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "loss",
                "n",
                "k",
                "N",
                "d",
                "largest_positive_margin",
                "best_step",
                "final_margin",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"saved={args.output_prefix.with_suffix('.pdf')}")
    print(f"saved={args.output_prefix.with_suffix('.png')}")
    print(f"saved={summary_path}")


if __name__ == "__main__":
    main()
