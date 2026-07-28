#!/usr/bin/env python3
"""Plot the k=2 paper reproduction and final sign-error probabilities."""

from __future__ import annotations

import argparse
import csv
import json
import math
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


def scalar(data: np.lib.npyio.NpzFile, key: str) -> float:
    return float(np.asarray(data[key]).reshape(-1)[-1])


def discover(
    root: Path, seed: int
) -> dict[tuple[int, int], dict[str, float | int]]:
    records: dict[tuple[int, int], dict[str, float | int]] = {}
    for final_path in root.glob("graph_n_*_k_2_N_*_seed_*/d_*/final.npz"):
        graph_match = GRAPH_RE.fullmatch(final_path.parent.parent.name)
        d_match = D_RE.fullmatch(final_path.parent.name)
        if graph_match is None or d_match is None:
            continue
        n, k, N, run_seed = map(int, graph_match.groups())
        if k != 2 or run_seed != seed or N != math.comb(n, 2):
            continue
        d = int(d_match.group(1))
        with np.load(final_path, allow_pickle=False) as data:
            records[(n, d)] = {
                "N": N,
                "final_margin": scalar(data, "margin"),
                "best_margin": scalar(data, "best_margin"),
                "best_step": int(scalar(data, "best_step")),
                "false_positive_count": int(scalar(data, "false_positive_count")),
                "negative_count": int(scalar(data, "negative_count")),
                "false_positive_rate": scalar(data, "false_positive_rate"),
                "false_negative_count": int(scalar(data, "false_negative_count")),
                "positive_count": int(scalar(data, "positive_count")),
                "false_negative_rate": scalar(data, "false_negative_rate"),
            }
    return records


def require_complete(
    datasets: dict[str, dict[tuple[int, int], dict[str, float | int]]],
    n_values: list[int],
    d_values: list[int],
) -> None:
    missing = [
        (loss, n, d)
        for loss, records in datasets.items()
        for n in n_values
        for d in d_values
        if (n, d) not in records
    ]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} results; examples: {missing[:10]}")


def save_figure(fig: plt.Figure, prefix: Path) -> None:
    fig.savefig(prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(prefix.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def plot_minimum_dimension(
    datasets: dict[str, dict[tuple[int, int], dict[str, float | int]]],
    n_values: list[int],
    d_values: list[int],
    output_dir: Path,
) -> None:
    styles = {
        "InfoNCE": {"color": "#d62728", "marker": "o"},
        "Sigmoid": {"color": "#1f77b4", "marker": "s"},
    }
    rows: list[dict[str, object]] = []
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    for loss, records in datasets.items():
        minima: list[float] = []
        for n in n_values:
            positive = [
                d
                for d in d_values
                if float(records[(n, d)]["final_margin"]) > 0
            ]
            min_d = min(positive) if positive else None
            minima.append(np.nan if min_d is None else float(min_d))
            rows.append(
                {
                    "loss": loss,
                    "n": n,
                    "N": math.comb(n, 2),
                    "min_positive_d": "" if min_d is None else min_d,
                    "dimensions_run": " ".join(map(str, d_values)),
                }
            )
        ax.plot(n_values, minima, linewidth=2.3, markersize=6, label=loss, **styles[loss])

    ax.set_xticks(n_values)
    ax.set_xlabel("n (number of objects)")
    ax.set_ylabel("Minimum d with positive final margin")
    ax.set_title("k=2: minimum dimension with positive margin after 100000 steps")
    ax.set_ylim(min(d_values) - 0.5, max(d_values) + 0.5)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    save_figure(fig, output_dir / "k2_minimum_positive_dimension")
    write_csv(
        output_dir / "k2_minimum_positive_dimension_summary.csv",
        ["loss", "n", "N", "min_positive_d", "dimensions_run"],
        rows,
    )


def plot_maximal_margin(
    datasets: dict[str, dict[tuple[int, int], dict[str, float | int]]],
    n_values: list[int],
    selected_d: list[int],
    output_dir: Path,
) -> None:
    colors = {
        d: plt.cm.viridis(value)
        for d, value in zip(selected_d, np.linspace(0.15, 0.85, len(selected_d)))
    }
    styles = {
        "InfoNCE": {"linestyle": "-", "marker": "o"},
        "Sigmoid": {"linestyle": "--", "marker": "s"},
    }
    rows: list[dict[str, object]] = []
    fig, ax = plt.subplots(figsize=(8.6, 5.3))
    for loss, records in datasets.items():
        for d in selected_d:
            values: list[float] = []
            for n in n_values:
                record = records[(n, d)]
                best_margin = float(record["best_margin"])
                values.append(max(0.0, best_margin))
                rows.append(
                    {
                        "loss": loss,
                        "n": n,
                        "N": record["N"],
                        "d": d,
                        "largest_positive_margin": max(0.0, best_margin),
                        "best_step": record["best_step"],
                        "final_margin": record["final_margin"],
                    }
                )
            ax.plot(
                n_values,
                values,
                color=colors[d],
                linewidth=2.1,
                markersize=5.5,
                label=f"{loss}, d={d}",
                **styles[loss],
            )

    ax.set_xticks(n_values)
    ax.set_xlabel("n (number of objects)")
    ax.set_ylabel("Largest positive margin")
    ax.set_title("k=2: largest checkpointed margin during 100000 steps")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    save_figure(fig, output_dir / "k2_maximal_margin")
    write_csv(
        output_dir / "k2_maximal_margin_summary.csv",
        [
            "loss",
            "n",
            "N",
            "d",
            "largest_positive_margin",
            "best_step",
            "final_margin",
        ],
        rows,
    )


def plot_error_rates(
    datasets: dict[str, dict[tuple[int, int], dict[str, float | int]]],
    n_values: list[int],
    d_values: list[int],
    output_dir: Path,
) -> None:
    rows: list[dict[str, object]] = []
    fig, axes = plt.subplots(
        len(n_values),
        2,
        figsize=(12.0, 2.25 * len(n_values) + 1.2),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    loss_names = ["InfoNCE", "Sigmoid"]

    for row_index, n in enumerate(n_values):
        for column_index, loss in enumerate(loss_names):
            ax = axes[row_index, column_index]
            records = datasets[loss]
            false_positive_rates = [
                float(records[(n, d)]["false_positive_rate"]) for d in d_values
            ]
            false_negative_rates = [
                float(records[(n, d)]["false_negative_rate"]) for d in d_values
            ]

            ax.plot(
                d_values,
                false_positive_rates,
                color="#1f77b4",
                marker="o",
                markersize=3,
                linewidth=1.7,
                label="False positive rate",
            )
            ax.plot(
                d_values,
                false_negative_rates,
                color="#ff7f0e",
                marker="s",
                markersize=3,
                linewidth=1.7,
                label="False negative rate",
            )
            ax.set_yscale("symlog", base=10, linthresh=1e-4, linscale=0.7)
            ax.set_ylim(0, 1)
            ax.set_yticks([0, 1e-4, 1e-3, 1e-2, 1e-1, 1])
            ax.set_xticks([5, 10, 15, 20, 25, 30])
            ax.grid(True, which="both", alpha=0.25)

            if row_index == 0:
                ax.set_title(loss, fontsize=14)
            if column_index == 0:
                ax.set_ylabel(f"n={n}\nError probability")
            if row_index == len(n_values) - 1:
                ax.set_xlabel("Embedding dimension d")

            for d, false_positive_rate, false_negative_rate in zip(
                d_values, false_positive_rates, false_negative_rates
            ):
                record = records[(n, d)]
                rows.append(
                    {
                        "loss": loss,
                        "n": n,
                        "N": record["N"],
                        "d": d,
                        "false_positive_count": record["false_positive_count"],
                        "negative_count": record["negative_count"],
                        "false_positive_rate": false_positive_rate,
                        "false_negative_count": record["false_negative_count"],
                        "positive_count": record["positive_count"],
                        "false_negative_rate": false_negative_rate,
                    }
                )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.978),
        ncol=2,
        frameon=False,
    )
    fig.suptitle(
        "k=2 final threshold-0 error probabilities (symlog scale)",
        fontsize=16,
        y=0.998,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    save_figure(fig, output_dir / "k2_false_positive_negative_rates")
    write_csv(
        output_dir / "k2_false_positive_negative_rates_summary.csv",
        [
            "loss",
            "n",
            "N",
            "d",
            "false_positive_count",
            "negative_count",
            "false_positive_rate",
            "false_negative_count",
            "positive_count",
            "false_negative_rate",
        ],
        rows,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--plot-dir", type=Path, required=True)
    parser.add_argument("--n-values", default="20 40 60 80 100 120 140 160 180 200 220 240")
    parser.add_argument("--d-values", default="5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30")
    parser.add_argument("--margin-d-values", default="6 18 30")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    n_values = parse_int_list(args.n_values)
    d_values = parse_int_list(args.d_values)
    selected_d = parse_int_list(args.margin_d_values)
    datasets = {
        "InfoNCE": discover(args.output_root / "infonce", args.seed),
        "Sigmoid": discover(args.output_root / "sigmoid", args.seed),
    }
    require_complete(datasets, n_values, d_values)
    args.plot_dir.mkdir(parents=True, exist_ok=True)

    plot_minimum_dimension(datasets, n_values, d_values, args.plot_dir)
    plot_maximal_margin(datasets, n_values, selected_d, args.plot_dir)
    plot_error_rates(datasets, n_values, d_values, args.plot_dir)
    (args.plot_dir / "experiment_metadata.json").write_text(
        json.dumps(
            {
                "k": 2,
                "n_values": n_values,
                "N_values": [math.comb(n, 2) for n in n_values],
                "d_values": d_values,
                "margin_plot_d_values": selected_d,
                "losses": list(datasets),
                "steps": 100000,
                "margin_recording_interval": 1000,
                "classification_threshold": 0.0,
                "false_positive_rate": "count(A_ij=0 and score>0) / count(A_ij=0)",
                "false_negative_rate": "count(A_ij=1 and score<=0) / count(A_ij=1)",
                "seed": args.seed,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"saved plots and summaries under {args.plot_dir}")


if __name__ == "__main__":
    main()
