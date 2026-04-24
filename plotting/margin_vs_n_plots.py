#!/usr/bin/env python3
import argparse
import csv
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

GRAPH_RE = re.compile(r"^graph_n_(\d+)_k_(\d+)_N_(\d+)_seed_(\d+)$")
D_RE = re.compile(r"^d_(\d+)$")


def parse_int_list(raw: str) -> List[int]:
    values = [int(tok) for tok in raw.replace(",", " ").split()]
    if not values:
        raise ValueError("Expected a non-empty integer list.")
    return values


def choose(n: int, k: int) -> int:
    return math.comb(n, k)


def compute_expected_N(n: int, k: int, p: float) -> int:
    return math.floor(p * choose(n, k))


def scalar_from_npz(path: Path, key: str) -> Optional[float]:
    try:
        with np.load(path, allow_pickle=False) as data:
            if key not in data.files:
                return None
            arr = np.asarray(data[key])
            if arr.size == 0:
                return None
            value = float(arr.reshape(-1)[0])
            if not math.isfinite(value):
                return None
            return value
    except Exception:
        return None


def best_margin_in_run_dir(run_dir: Path) -> Optional[float]:
    best = None
    for npz_path in sorted(run_dir.glob("*.npz")):
        margin = scalar_from_npz(npz_path, "margin")
        if margin is None:
            continue
        if best is None or margin > best:
            best = margin
    return best


def find_graph_ancestor(d_dir: Path, root: Path) -> Optional[Tuple[Path, Tuple[int, int, int, int]]]:
    current = d_dir.parent
    root = root.resolve()
    while True:
        match = GRAPH_RE.match(current.name)
        if match:
            meta = tuple(int(x) for x in match.groups())
            return current, meta
        if current == root or current.parent == current:
            return None
        current = current.parent


def discover_runs(root: Path) -> List[Dict[str, object]]:
    runs: List[Dict[str, object]] = []
    for d_dir in sorted(root.rglob("d_*")):
        if not d_dir.is_dir():
            continue
        d_match = D_RE.match(d_dir.name)
        if d_match is None:
            continue
        found = find_graph_ancestor(d_dir, root)
        if found is None:
            continue
        graph_dir, (n, k, N, seed) = found
        d = int(d_match.group(1))
        runs.append(
            {
                "root": root,
                "graph_dir": graph_dir,
                "run_dir": d_dir,
                "n": n,
                "k": k,
                "N": N,
                "seed": seed,
                "d": d,
            }
        )
    return runs


def log_value(x: int, log_base: str) -> float:
    if log_base == "e":
        return math.log(x)
    if log_base == "2":
        return math.log2(x)
    if log_base == "10":
        return math.log10(x)
    raise ValueError(f"Unsupported log base: {log_base}")


def collect_best_margins(
    root: Path,
    loss_name: str,
    n_values: Iterable[int],
    d_values: Iterable[int],
    k_value: Optional[int],
    p_value: Optional[float],
    seed_value: Optional[int],
) -> Tuple[Dict[Tuple[str, int, int], float], List[Dict[str, object]]]:
    wanted_n = set(n_values)
    wanted_d = set(d_values)
    best: Dict[Tuple[str, int, int], float] = {}
    rows: List[Dict[str, object]] = []

    for run in discover_runs(root):
        n = int(run["n"])
        k = int(run["k"])
        N = int(run["N"])
        seed = int(run["seed"])
        d = int(run["d"])

        if n not in wanted_n or d not in wanted_d:
            continue
        if k_value is not None and k != k_value:
            continue
        if seed_value is not None and seed != seed_value:
            continue
        if p_value is not None:
            expected_N = compute_expected_N(n, k, p_value)
            if N != expected_N:
                continue

        best_margin = best_margin_in_run_dir(Path(run["run_dir"]))
        if best_margin is None:
            continue

        key = (loss_name, d, n)
        previous = best.get(key)
        if previous is None or best_margin > previous:
            best[key] = best_margin

        rows.append(
            {
                "loss": loss_name,
                "n": n,
                "k": k,
                "N": N,
                "seed": seed,
                "d": d,
                "run_dir": str(run["run_dir"]),
                "best_margin_in_run": best_margin,
            }
        )

    return best, rows


def write_summary_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    fieldnames = ["loss", "n", "k", "N", "seed", "d", "run_dir", "best_margin_in_run"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_aggregated_csv(
    path: Path,
    best_infonce: Dict[Tuple[str, int, int], float],
    best_siglip: Dict[Tuple[str, int, int], float],
    n_values: List[int],
    d_values: List[int],
    log_base: str,
) -> None:
    fieldnames = ["loss", "d", "n", "log_n", "best_margin"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for best in [best_infonce, best_siglip]:
            for (loss, d, n), margin in sorted(best.items()):
                if n in n_values and d in d_values:
                    writer.writerow(
                        {
                            "loss": loss,
                            "d": d,
                            "n": n,
                            "log_n": log_value(n, log_base),
                            "best_margin": margin,
                        }
                    )


def make_plot(
    best_infonce: Dict[Tuple[str, int, int], float],
    best_siglip: Dict[Tuple[str, int, int], float],
    n_values: List[int],
    d_values: List[int],
    output_pdf: Path,
    output_png: Path,
    log_base: str,
    p_value: Optional[float],
    k_value: Optional[int],
    siglip_label: str,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(11, 6.5))

    xticks = [log_value(n, log_base) for n in n_values]
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not colors:
        colors = [f"C{i}" for i in range(max(1, len(d_values)))]

    global_max = 0.0
    plotted_any = False
    dimension_handles = []

    for idx, d in enumerate(d_values):
        color = colors[idx % len(colors)]
        x_full = [log_value(n, log_base) for n in n_values]
        y_infonce = []
        y_siglip = []
        d_has_positive = False

        for n in n_values:
            margin_infonce = best_infonce.get(("InfoNCE", d, n))
            margin_siglip = best_siglip.get((siglip_label, d, n))

            if margin_infonce is not None and margin_infonce > 0:
                y_infonce.append(margin_infonce)
                global_max = max(global_max, margin_infonce)
                d_has_positive = True
            else:
                y_infonce.append(float("nan"))

            if margin_siglip is not None and margin_siglip > 0:
                y_siglip.append(margin_siglip)
                global_max = max(global_max, margin_siglip)
                d_has_positive = True
            else:
                y_siglip.append(float("nan"))

        if not d_has_positive:
            continue

        infonce_line, = ax.plot(
            x_full,
            y_infonce,
            color=color,
            linestyle="-",
            marker="o",
            linewidth=2,
            markersize=5,
        )
        ax.plot(
            x_full,
            y_siglip,
            color=color,
            linestyle="--",
            marker="o",
            linewidth=2,
            markersize=5,
        )
        dimension_handles.append((infonce_line, f"d={d}"))
        plotted_any = True

    ax.set_xlabel("log n")
    ax.set_ylabel("m (margin)")
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(n) for n in n_values], rotation=45)
    ax.grid(True, alpha=0.3)
    y_top = 1.0 if global_max <= 0 else 1.05 * global_max
    ax.set_ylim(bottom=0.0, top=y_top)

    title_bits = ["Largest positive margin achieved during training"]
    if k_value is not None:
        title_bits.append(f"k={k_value}")
    if p_value is not None:
        title_bits.append(f"p={p_value}")
    ax.set_title(" | ".join(title_bits), pad=14)

    if plotted_any:
        from matplotlib.lines import Line2D

        style_handles = [
            Line2D([0], [0], color="black", linestyle="-", marker="o", linewidth=2, label="InfoNCE"),
            Line2D([0], [0], color="black", linestyle="--", marker="o", linewidth=2, label=siglip_label),
        ]
        legend_dims = ax.legend(
            [h for h, _ in dimension_handles],
            [lbl for _, lbl in dimension_handles],
            title="dimension",
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            frameon=False,
        )
        ax.add_artist(legend_dims)
        ax.legend(
            handles=style_handles,
            title="loss",
            loc="lower left",
            bbox_to_anchor=(1.02, 0.0),
            borderaxespad=0.0,
            frameon=False,
        )

    fig.tight_layout(rect=[0, 0, 0.82, 0.97])
    fig.savefig(output_pdf, bbox_inches="tight")
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot the largest positive saved margin achieved during training versus log n on one shared plot for InfoNCE and SigLIP."
    )
    parser.add_argument("--infonce-root", type=Path, required=True)
    parser.add_argument("--siglip-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-name", type=str, default="max_margin_vs_logn")
    parser.add_argument("--n-values", type=str, required=True)
    parser.add_argument("--d-values", type=str, required=True)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log-base", choices=["e", "2", "10"], default="e")
    parser.add_argument("--siglip-label", type=str, default="SigLIP")
    args = parser.parse_args()

    n_values = parse_int_list(args.n_values)
    d_values = parse_int_list(args.d_values)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    best_infonce, infonce_rows = collect_best_margins(
        root=args.infonce_root,
        loss_name="InfoNCE",
        n_values=n_values,
        d_values=d_values,
        k_value=args.k,
        p_value=args.p,
        seed_value=args.seed,
    )
    best_siglip, siglip_rows = collect_best_margins(
        root=args.siglip_root,
        loss_name=args.siglip_label,
        n_values=n_values,
        d_values=d_values,
        k_value=args.k,
        p_value=args.p,
        seed_value=args.seed,
    )

    detailed_csv = args.output_dir / f"{args.output_name}_detailed_runs.csv"
    aggregated_csv = args.output_dir / f"{args.output_name}_aggregated.csv"
    output_pdf = args.output_dir / f"{args.output_name}.pdf"
    output_png = args.output_dir / f"{args.output_name}.png"

    write_summary_csv(detailed_csv, infonce_rows + siglip_rows)
    write_aggregated_csv(
        aggregated_csv,
        best_infonce=best_infonce,
        best_siglip=best_siglip,
        n_values=n_values,
        d_values=d_values,
        log_base=args.log_base,
    )
    make_plot(
        best_infonce=best_infonce,
        best_siglip=best_siglip,
        n_values=n_values,
        d_values=d_values,
        output_pdf=output_pdf,
        output_png=output_png,
        log_base=args.log_base,
        p_value=args.p,
        k_value=args.k,
        siglip_label=args.siglip_label,
    )

    print(f"Saved plot to: {output_pdf}")
    print(f"Saved plot to: {output_png}")
    print(f"Saved aggregated summary to: {aggregated_csv}")
    print(f"Saved per-run summary to: {detailed_csv}")
    print(f"Matched InfoNCE entries: {len(infonce_rows)}")
    print(f"Matched {args.siglip_label} entries: {len(siglip_rows)}")


if __name__ == "__main__":
    main()
