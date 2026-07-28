#!/usr/bin/env python3
"""Overlay InfoNCE and sigmoid error rates in one panel per n."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--output-stem", type=Path, required=True)
    args = parser.parse_args()

    with args.summary_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    n_values = sorted({int(row["n"]) for row in rows})
    d_values = sorted({int(row["d"]) for row in rows})
    by_key = {(row["loss"], int(row["n"]), int(row["d"])): row for row in rows}

    figure, axes = plt.subplots(
        len(n_values),
        1,
        figsize=(11, 2.55 * len(n_values) + 2.2),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    styles = {
        ("InfoNCE", "false_positive_rate"): {
            "color": "tab:blue",
            "linestyle": "-",
            "marker": "o",
            "label": "InfoNCE false positive",
        },
        ("InfoNCE", "false_negative_rate"): {
            "color": "tab:orange",
            "linestyle": "-",
            "marker": "s",
            "label": "InfoNCE false negative",
        },
        ("Sigmoid", "false_positive_rate"): {
            "color": "tab:blue",
            "linestyle": ":",
            "marker": "o",
            "label": "Sigmoid false positive",
        },
        ("Sigmoid", "false_negative_rate"): {
            "color": "tab:orange",
            "linestyle": ":",
            "marker": "s",
            "label": "Sigmoid false negative",
        },
    }

    for axis, n in zip(axes, n_values):
        for (loss, metric), style in styles.items():
            values = [float(by_key[(loss, n, d)][metric]) for d in d_values]
            axis.plot(
                d_values,
                values,
                linewidth=1.7,
                markersize=3.2,
                **style,
            )
        axis.set_yscale("symlog", linthresh=1e-5, linscale=0.8)
        axis.set_ylim(-2e-7, 1.05)
        axis.set_ylabel(f"n={n}\nError probability", fontsize=10.5)
        axis.grid(True, alpha=0.25)

    axes[-1].set_xlabel("Embedding dimension d", fontsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.972),
    )
    figure.suptitle(
        "k=2 errors at midpoint of minimum-positive/maximum-negative scores",
        fontsize=18,
    )

    args.output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output_stem.with_suffix(".png"), dpi=180)
    figure.savefig(args.output_stem.with_suffix(".pdf"))
    plt.close(figure)


if __name__ == "__main__":
    main()
