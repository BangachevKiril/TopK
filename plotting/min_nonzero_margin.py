#!/usr/bin/env python3
import argparse
import csv
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


GRAPH_RE = re.compile(r"^graph_n_(\d+)_k_(\d+)_N_(\d+)_seed_(\d+)$")
D_RE = re.compile(r"^d_(\d+)$")


@dataclass(frozen=True)
class RunRecord:
    loss_name: str
    graph_stem: str
    n: int
    k: int
    N: int
    seed: int
    d: int
    margin: float
    artifact_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot minimal embedding dimension d achieving strictly positive margin as a function of n, "
            "with one curve for sigmoid and one for InfoNCE."
        )
    )
    parser.add_argument("--sigmoid_root", type=str, required=True)
    parser.add_argument("--infonce_root", type=str, required=True)
    parser.add_argument("--output_prefix", type=str, required=True)
    parser.add_argument("--k_values", type=str, default="")
    parser.add_argument("--n_values", type=str, default="")
    parser.add_argument("--seed_values", type=str, default="")
    parser.add_argument(
        "--p_values",
        type=str,
        default="",
        help=(
            "Optional list of candidate p values. Since graph filenames store N rather than p, "
            "a graph is kept if N = floor(p * binom(n,k)) for at least one requested p."
        ),
    )
    parser.add_argument("--title", type=str, default="")
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def parse_int_list(raw: str) -> Optional[List[int]]:
    raw = raw.strip()
    if not raw:
        return None
    items = raw.replace(",", " ").split()
    return [int(x) for x in items]


def parse_float_list(raw: str) -> Optional[List[float]]:
    raw = raw.strip()
    if not raw:
        return None
    items = raw.replace(",", " ").split()
    return [float(x) for x in items]


def infer_requested_p_match(n: int, k: int, N: int, p_values: Optional[Sequence[float]]) -> bool:
    if not p_values:
        return True
    total = math.comb(n, k)
    for p in p_values:
        if math.floor(p * total) == N:
            return True
    return False


def load_margin_from_npz(npz_path: Path) -> float:
    with np.load(npz_path, allow_pickle=False) as data:
        keys = set(data.files)

        if "margin" in keys:
            arr = np.asarray(data["margin"]).reshape(-1)
            finite = arr[np.isfinite(arr)]
            if finite.size:
                return float(finite[-1])

        for key in ("margin_history", "margins", "margin_values"):
            if key in keys:
                arr = np.asarray(data[key]).reshape(-1)
                finite = arr[np.isfinite(arr)]
                if finite.size:
                    return float(finite[-1])

        if "pos_min" in keys and "neg_max" in keys:
            pos = np.asarray(data["pos_min"]).reshape(-1)
            neg = np.asarray(data["neg_max"]).reshape(-1)
            m = 0.5 * (pos - neg)
            finite = m[np.isfinite(m)]
            if finite.size:
                return float(finite[-1])

    raise ValueError(f"Could not recover a margin from {npz_path}")



def choose_artifact(run_dir: Path) -> Optional[Path]:
    latest = run_dir / "latest.npz"
    if latest.is_file():
        return latest

    final = run_dir / "final.npz"
    if final.is_file():
        return final

    history = run_dir / "margin_history.npz"
    if history.is_file():
        return history

    checkpoints = sorted(run_dir.glob("checkpoint_step_*.npz"))
    if checkpoints:
        return checkpoints[-1]

    return None



def discover_run_records(
    root: Path,
    loss_name: str,
    k_values: Optional[Sequence[int]],
    n_values: Optional[Sequence[int]],
    seed_values: Optional[Sequence[int]],
    p_values: Optional[Sequence[float]],
) -> List[RunRecord]:
    records: List[RunRecord] = []

    for dirpath, dirnames, _filenames in os.walk(root):
        run_dir = Path(dirpath)
        d_match = D_RE.match(run_dir.name)
        if not d_match:
            continue

        graph_dir = run_dir.parent
        graph_match = GRAPH_RE.match(graph_dir.name)
        if not graph_match:
            continue

        n, k, N, seed = map(int, graph_match.groups())
        d = int(d_match.group(1))

        if n_values is not None and n not in n_values:
            continue
        if k_values is not None and k not in k_values:
            continue
        if seed_values is not None and seed not in seed_values:
            continue
        if not infer_requested_p_match(n=n, k=k, N=N, p_values=p_values):
            continue

        artifact = choose_artifact(run_dir)
        if artifact is None:
            continue

        try:
            margin = load_margin_from_npz(artifact)
        except Exception:
            continue

        records.append(
            RunRecord(
                loss_name=loss_name,
                graph_stem=graph_dir.name,
                n=n,
                k=k,
                N=N,
                seed=seed,
                d=d,
                margin=margin,
                artifact_path=str(artifact),
            )
        )

    return records



def minimal_positive_d_by_n(records: Sequence[RunRecord]) -> Dict[int, float]:
    best: Dict[int, int] = {}
    for rec in records:
        if not np.isfinite(rec.margin) or rec.margin <= 0:
            continue
        if rec.n not in best or rec.d < best[rec.n]:
            best[rec.n] = rec.d
    return {n: float(d) for n, d in sorted(best.items())}



def write_summary_csv(path: Path, sigmoid_records: Sequence[RunRecord], infonce_records: Sequence[RunRecord]) -> None:
    rows = []
    for records in (sigmoid_records, infonce_records):
        grouped: Dict[Tuple[str, int], List[RunRecord]] = {}
        for rec in records:
            grouped.setdefault((rec.loss_name, rec.n), []).append(rec)

        for (loss_name, n), group in sorted(grouped.items()):
            positive_ds = sorted({rec.d for rec in group if np.isfinite(rec.margin) and rec.margin > 0})
            rows.append(
                {
                    "loss": loss_name,
                    "n": n,
                    "num_matching_runs": len(group),
                    "num_positive_runs": sum(1 for rec in group if np.isfinite(rec.margin) and rec.margin > 0),
                    "min_positive_d": positive_ds[0] if positive_ds else "",
                    "positive_ds": " ".join(str(x) for x in positive_ds),
                }
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["loss", "n", "num_matching_runs", "num_positive_runs", "min_positive_d", "positive_ds"],
        )
        writer.writeheader()
        writer.writerows(rows)



def make_plot(
    output_prefix: Path,
    sigmoid_min_d: Dict[int, float],
    infonce_min_d: Dict[int, float],
    title: str,
    dpi: int,
) -> None:
    all_ns = sorted(set(sigmoid_min_d) | set(infonce_min_d))
    if not all_ns:
        raise ValueError("No matching runs were found for either sigmoid or InfoNCE.")

    y_sigmoid = [sigmoid_min_d.get(n, np.nan) for n in all_ns]
    y_infonce = [infonce_min_d.get(n, np.nan) for n in all_ns]

    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8.5, 5.4))
    plt.plot(all_ns, y_infonce, marker="o", color="red", linewidth=2.0, label="InfoNCE")
    plt.plot(all_ns, y_sigmoid, marker="o", color="blue", linewidth=2.0, label="Sigmoid")
    plt.xlabel("n")
    plt.ylabel("minimal d with positive margin")
    plt.xticks(all_ns, [str(n) for n in all_ns])
    plt.grid(True, alpha=0.3)
    plt.legend()
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(output_prefix.with_suffix(".png"), dpi=dpi)
    plt.savefig(output_prefix.with_suffix(".pdf"))
    plt.close()



def main() -> None:
    args = parse_args()

    k_values = parse_int_list(args.k_values)
    n_values = parse_int_list(args.n_values)
    seed_values = parse_int_list(args.seed_values)
    p_values = parse_float_list(args.p_values)

    sigmoid_root = Path(args.sigmoid_root).expanduser().resolve()
    infonce_root = Path(args.infonce_root).expanduser().resolve()
    output_prefix = Path(args.output_prefix).expanduser().resolve()

    sigmoid_records = discover_run_records(
        root=sigmoid_root,
        loss_name="sigmoid",
        k_values=k_values,
        n_values=n_values,
        seed_values=seed_values,
        p_values=p_values,
    )
    infonce_records = discover_run_records(
        root=infonce_root,
        loss_name="infonce",
        k_values=k_values,
        n_values=n_values,
        seed_values=seed_values,
        p_values=p_values,
    )

    sigmoid_min_d = minimal_positive_d_by_n(sigmoid_records)
    infonce_min_d = minimal_positive_d_by_n(infonce_records)

    if not sigmoid_records:
        print(f"Warning: found no matching sigmoid runs under {sigmoid_root}")
    if not infonce_records:
        print(f"Warning: found no matching InfoNCE runs under {infonce_root}")

    title = args.title
    if not title:
        title_bits = ["Minimal d achieving positive margin"]
        if p_values:
            title_bits.append("p in {" + ", ".join(str(x) for x in p_values) + "}")
        if k_values:
            title_bits.append("k in {" + ", ".join(str(x) for x in k_values) + "}")
        title = " | ".join(title_bits)

    make_plot(
        output_prefix=output_prefix,
        sigmoid_min_d=sigmoid_min_d,
        infonce_min_d=infonce_min_d,
        title=title,
        dpi=args.dpi,
    )

    summary_csv = output_prefix.parent / f"{output_prefix.name}_summary.csv"
    write_summary_csv(summary_csv, sigmoid_records=sigmoid_records, infonce_records=infonce_records)

    print(f"Saved plot to: {output_prefix.with_suffix('.png')}")
    print(f"Saved plot to: {output_prefix.with_suffix('.pdf')}")
    print(f"Saved summary CSV to: {summary_csv}")
    print(f"Sigmoid minimal d by n: {sigmoid_min_d}")
    print(f"InfoNCE minimal d by n: {infonce_min_d}")


if __name__ == "__main__":
    main()
