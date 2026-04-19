#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import warnings
import zipfile
from dataclasses import dataclass
from decimal import Decimal, ROUND_FLOOR, getcontext
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import numpy as np


GRAPH_STEM_RE = re.compile(r"^graph_n_(\d+)_k_(\d+)_N_(\d+)_seed_(\d+)$")
getcontext().prec = 50


class HeatmapDataError(RuntimeError):
    pass


@dataclass(frozen=True)
class RunRootSummary:
    root: Path
    n: int
    k: int
    N: int
    seed: int
    min_success_d: Optional[int]
    available_dims: Tuple[int, ...]



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create side-by-side heatmaps comparing sigmoid and InfoNCE. For each requested p, "
            "cell (n,k) stores the minimal dimension d such that the run for N=floor(p*binom(n,k)) "
            "achieved positive margin."
        )
    )
    parser.add_argument("--sigmoid_root", type=str, required=True)
    parser.add_argument("--infonce_root", type=str, required=True)
    parser.add_argument(
        "--p_values",
        type=str,
        nargs="+",
        required=True,
        help="One or more p values, e.g. --p_values 0.1 0.25 0.5",
    )
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument(
        "--success_mode",
        type=str,
        default="best",
        choices=["best", "final"],
        help=(
            "A dimension d counts as successful if either its best saved margin (best) or final saved "
            "margin (final) exceeds min_margin. Default: best"
        ),
    )
    parser.add_argument(
        "--seed_reduce",
        type=str,
        default="any",
        choices=["any", "all"],
        help=(
            "How to combine multiple seeds with the same (n,k,N). 'any' uses the easiest seed, while "
            "'all' requires every available seed to succeed and takes the smallest d that works for all."
        ),
    )
    parser.add_argument(
        "--min_margin",
        type=float,
        default=0.0,
        help="A run counts as successful only if its chosen margin statistic is strictly greater than this threshold.",
    )
    parser.add_argument("--figwidth", type=float, default=14.0)
    parser.add_argument("--figheight", type=float, default=6.0)
    parser.add_argument(
        "--annotate",
        action="store_true",
        help="Write the successful dimension into each non-empty heatmap cell.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing PDFs if they already exist.",
    )
    return parser.parse_args()



def parse_p_value(p_raw: str) -> Decimal:
    try:
        return Decimal(p_raw)
    except Exception as exc:
        raise HeatmapDataError(f"Could not parse p='{p_raw}' as a decimal number") from exc



def safe_p_string(p_raw: str) -> str:
    out = p_raw.strip()
    out = out.replace("/", "_over_")
    out = out.replace(".", "p")
    out = out.replace("-", "m")
    out = out.replace("+", "")
    return out



def is_graph_run_root(path: Path) -> bool:
    if not path.is_dir():
        return False
    if GRAPH_STEM_RE.match(path.name) is None:
        return False
    return any(child.is_dir() and child.name.startswith("d_") for child in path.iterdir())



def discover_graph_run_roots(root: Path) -> List[Path]:
    roots: List[Path] = []
    for path in root.rglob("graph_n_*_k_*_N_*_seed_*"):
        if is_graph_run_root(path):
            roots.append(path.resolve())
    return sorted(set(roots))



def discover_dimensions(run_root: Path) -> List[int]:
    dims: List[int] = []
    for path in run_root.glob("d_*"):
        if not path.is_dir():
            continue
        try:
            dims.append(int(path.name.split("_", 1)[1]))
        except Exception:
            continue
    return sorted(set(dims))



def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)



def parse_stem_metadata(stem: str) -> Tuple[int, int, int, int]:
    m = GRAPH_STEM_RE.match(stem)
    if m is None:
        raise HeatmapDataError(f"Could not parse graph metadata from directory name: {stem}")
    return tuple(int(x) for x in m.groups())  # type: ignore[return-value]



def load_reference_metadata(run_root: Path) -> Tuple[int, int, int, int]:
    dims = discover_dimensions(run_root)
    for d in dims:
        config_path = run_root / f"d_{d}" / "config.json"
        if config_path.is_file():
            config = load_json(config_path)
            try:
                return int(config["n"]), int(config["k"]), int(config["N"]), int(config["seed"])
            except Exception:
                break
    return parse_stem_metadata(run_root.name)



def checkpoint_files_for_run_dir(run_dir: Path) -> Tuple[List[Path], List[Path]]:
    checkpoint_steps = sorted(run_dir.glob("checkpoint_step_*.npz"))
    fallback: List[Path] = []
    for name in ("latest.npz", "final.npz"):
        path = run_dir / name
        if path.is_file():
            fallback.append(path)
    return checkpoint_steps, fallback



def _load_margin_rows(paths: Sequence[Path]) -> Tuple[List[Tuple[float, float]], List[str]]:
    rows: List[Tuple[float, float]] = []
    errors: List[str] = []

    for path in paths:
        try:
            if not path.is_file():
                continue
            if path.stat().st_size == 0:
                raise ValueError("empty file")
            with np.load(path, allow_pickle=False) as data:
                if "margin" not in data:
                    raise KeyError("missing 'margin' entry")
                step = float(np.asarray(data["step"]).item()) if "step" in data else math.nan
                margin = float(np.asarray(data["margin"]).item())
        except Exception as exc:
            errors.append(f"{path.name}: {exc}")
            continue
        rows.append((step, margin))

    rows.sort(key=lambda x: (math.inf if math.isnan(x[0]) else x[0]))
    return rows, errors



def load_margin_history(run_dir: Path) -> np.ndarray:
    checkpoint_steps, fallback = checkpoint_files_for_run_dir(run_dir)
    errors: List[str] = []

    rows, step_errors = _load_margin_rows(checkpoint_steps)
    errors.extend(step_errors)

    if not rows:
        fb_rows, fb_errors = _load_margin_rows(fallback)
        rows.extend(fb_rows)
        errors.extend(fb_errors)

    if not rows:
        if checkpoint_steps or fallback:
            joined = "; ".join(errors) if errors else "all candidate files were unreadable"
            raise HeatmapDataError(f"No readable checkpoint files found in {run_dir} ({joined})")
        raise HeatmapDataError(f"No checkpoint files found in {run_dir}")

    if errors:
        warnings.warn(
            f"Ignoring unreadable checkpoint files in {run_dir}: " + "; ".join(errors),
            stacklevel=2,
        )

    return np.asarray([margin for _, margin in rows], dtype=float)



def dimension_success(margins: np.ndarray, success_mode: str, min_margin: float) -> bool:
    finite = margins[np.isfinite(margins)]
    if finite.size == 0:
        return False
    if success_mode == "best":
        value = float(np.max(finite))
    elif success_mode == "final":
        value = float(finite[-1])
    else:
        raise ValueError("success_mode must be 'best' or 'final'")
    return value > float(min_margin)



def summarize_run_root(run_root: Path, success_mode: str, min_margin: float) -> RunRootSummary:
    n, k, N, seed = load_reference_metadata(run_root)
    dims = discover_dimensions(run_root)
    if not dims:
        raise HeatmapDataError(f"No d_<dim> folders found under {run_root}")

    successful_dims: List[int] = []
    readable_dims: List[int] = []
    skipped_dims: List[int] = []

    for d in dims:
        run_dir = run_root / f"d_{d}"
        if not run_dir.is_dir():
            continue
        try:
            margins = load_margin_history(run_dir)
        except HeatmapDataError as exc:
            skipped_dims.append(d)
            warnings.warn(f"Skipping unreadable dimension d={d} under {run_root}: {exc}", stacklevel=2)
            continue
        except Exception as exc:
            skipped_dims.append(d)
            warnings.warn(f"Skipping unreadable dimension d={d} under {run_root}: {exc}", stacklevel=2)
            continue

        readable_dims.append(d)
        if dimension_success(margins=margins, success_mode=success_mode, min_margin=min_margin):
            successful_dims.append(d)

    if not readable_dims:
        warnings.warn(f"No readable dimension folders found under {run_root}", stacklevel=2)

    if skipped_dims:
        warnings.warn(
            f"Skipped unreadable dimensions under {run_root}: {sorted(skipped_dims)}",
            stacklevel=2,
        )

    min_success_d = min(successful_dims) if successful_dims else None
    return RunRootSummary(
        root=run_root,
        n=n,
        k=k,
        N=N,
        seed=seed,
        min_success_d=min_success_d,
        available_dims=tuple(readable_dims if readable_dims else dims),
    )



def _root_preference(path: Path, root: Path) -> Tuple[int, int, str]:
    try:
        rel = path.resolve().relative_to(root.resolve())
        parts = rel.parts
    except Exception:
        parts = path.resolve().parts
        rel = path.resolve()
    repeated_neighbors = sum(1 for a, b in zip(parts, parts[1:]) if a == b)
    return (len(parts), repeated_neighbors, str(rel))



def build_model_index(root: Path, success_mode: str, min_margin: float) -> Dict[Tuple[int, int, int, int], RunRootSummary]:
    run_roots = discover_graph_run_roots(root)
    if not run_roots:
        raise HeatmapDataError(f"No graph-specific run roots were found under {root}")

    index: Dict[Tuple[int, int, int, int], RunRootSummary] = {}
    duplicates: Dict[Tuple[int, int, int, int], List[Path]] = {}

    for run_root in run_roots:
        summary = summarize_run_root(run_root=run_root, success_mode=success_mode, min_margin=min_margin)
        key = (summary.n, summary.k, summary.N, summary.seed)
        if key in index:
            duplicates.setdefault(key, [index[key].root]).append(summary.root)
            prev = index[key]
            prev_d = prev.min_success_d
            curr_d = summary.min_success_d
            if prev_d is None:
                choose_current = curr_d is not None
            elif curr_d is None:
                choose_current = False
            elif curr_d != prev_d:
                choose_current = curr_d < prev_d
            else:
                choose_current = _root_preference(summary.root, root) < _root_preference(prev.root, root)
            if choose_current:
                index[key] = summary
        else:
            index[key] = summary

    if duplicates:
        pieces = []
        for key, roots in sorted(duplicates.items()):
            pieces.append(f"{key}: " + ", ".join(str(r) for r in roots))
        warnings.warn(
            "Found duplicate run roots for the same (n,k,N,seed). Keeping the preferred one.\n" + "\n".join(pieces),
            stacklevel=2,
        )

    return index



def target_N_for_p(n: int, k: int, p: Decimal) -> int:
    total = Decimal(math.comb(n, k))
    return int((p * total).to_integral_value(rounding=ROUND_FLOOR))



def collect_grid_axes(
    sigmoid_index: Dict[Tuple[int, int, int, int], RunRootSummary],
    infonce_index: Dict[Tuple[int, int, int, int], RunRootSummary],
    p: Decimal,
) -> Tuple[List[int], List[int]]:
    pairs = set()
    for index in (sigmoid_index, infonce_index):
        for n, k, N, _seed in index:
            if N == target_N_for_p(n=n, k=k, p=p):
                pairs.add((n, k))
    if not pairs:
        return [], []
    ns = sorted({n for n, _ in pairs})
    ks = sorted({k for _, k in pairs})
    return ns, ks



def build_heatmap_table(
    model_index: Dict[Tuple[int, int, int, int], RunRootSummary],
    ns: Sequence[int],
    ks: Sequence[int],
    p: Decimal,
    seed_reduce: str,
) -> np.ndarray:
    table = np.full((len(ns), len(ks)), np.nan, dtype=float)
    n_to_i = {n: i for i, n in enumerate(ns)}
    k_to_j = {k: j for j, k in enumerate(ks)}

    grouped: Dict[Tuple[int, int], List[Optional[int]]] = {}
    for (n, k, N, _seed), summary in model_index.items():
        if n not in n_to_i or k not in k_to_j:
            continue
        if N != target_N_for_p(n=n, k=k, p=p):
            continue
        grouped.setdefault((n, k), []).append(summary.min_success_d)

    for (n, k), values in grouped.items():
        i = n_to_i[n]
        j = k_to_j[k]
        good = [v for v in values if v is not None]
        if seed_reduce == "any":
            if good:
                table[i, j] = float(min(good))
        elif seed_reduce == "all":
            if values and len(good) == len(values):
                table[i, j] = float(max(good))
        else:
            raise ValueError("seed_reduce must be 'any' or 'all'")

    return table



def build_shared_norm(all_tables: Iterable[np.ndarray]) -> Tuple[Optional[BoundaryNorm], List[float], plt.Colormap]:
    finite_values: List[float] = []
    for table in all_tables:
        finite = table[np.isfinite(table)]
        finite_values.extend(float(x) for x in finite)

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="#e6e6e6")

    if not finite_values:
        return None, [], cmap

    ticks = sorted(set(finite_values))
    if len(ticks) == 1:
        only = ticks[0]
        boundaries = [only - 0.5, only + 0.5]
    else:
        boundaries = [ticks[0] - 0.5 * (ticks[1] - ticks[0])]
        for a, b in zip(ticks[:-1], ticks[1:]):
            boundaries.append(0.5 * (a + b))
        boundaries.append(ticks[-1] + 0.5 * (ticks[-1] - ticks[-2]))

    norm = BoundaryNorm(boundaries=boundaries, ncolors=cmap.N, clip=False)
    return norm, ticks, cmap



def annotate_heatmap(ax, table: np.ndarray, norm: Optional[BoundaryNorm], cmap, fontsize: float = 10.0) -> None:
    nrows, ncols = table.shape
    for i in range(nrows):
        for j in range(ncols):
            value = table[i, j]
            if not np.isfinite(value):
                continue
            if norm is None:
                text_color = "black"
            else:
                rgba = cmap(norm(value))
                luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                text_color = "black" if luminance > 0.55 else "white"
            ax.text(j, i, f"{int(round(value))}", ha="center", va="center", color=text_color, fontsize=fontsize)



def success_description(success_mode: str, min_margin: float) -> str:
    which = "best saved" if success_mode == "best" else "final saved"
    return f"success = {which} margin > {min_margin:g}"



def seed_description(seed_reduce: str) -> str:
    if seed_reduce == "any":
        return "seeds aggregated by any"
    return "seeds aggregated by all"



def write_p_figure(
    p_raw: str,
    output_pdf: Path,
    sigmoid_table: np.ndarray,
    infonce_table: np.ndarray,
    ns: Sequence[int],
    ks: Sequence[int],
    norm: Optional[BoundaryNorm],
    color_ticks: Sequence[float],
    cmap,
    success_mode: str,
    seed_reduce: str,
    min_margin: float,
    figsize: Tuple[float, float],
    annotate: bool,
    overwrite: bool,
) -> Optional[Path]:
    if output_pdf.exists() and not overwrite:
        print(f"PDF already exists, skipping: {output_pdf}")
        return None

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    left_ax, right_ax = axes
    masked_sigmoid = np.ma.masked_invalid(sigmoid_table)
    masked_infonce = np.ma.masked_invalid(infonce_table)

    im_left = left_ax.imshow(masked_sigmoid, cmap=cmap, norm=norm, aspect="auto")
    right_ax.imshow(masked_infonce, cmap=cmap, norm=norm, aspect="auto")

    for ax, title in zip(axes, ["Sigmoid loss", "InfoNCE loss"]):
        ax.set_title(title)
        ax.set_xlabel("k")
        ax.set_xticks(np.arange(len(ks)))
        ax.set_xticklabels([str(k) for k in ks])
        ax.set_yticks(np.arange(len(ns)))
        ax.set_yticklabels([str(n) for n in ns])
        ax.set_xticks(np.arange(-0.5, len(ks), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(ns), 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=0.6, alpha=0.6)
        ax.tick_params(which="minor", bottom=False, left=False)

    left_ax.set_ylabel("n")

    if annotate:
        annotate_heatmap(left_ax, sigmoid_table, norm, cmap)
        annotate_heatmap(right_ax, infonce_table, norm, cmap)

    if norm is None:
        fig.text(0.5, 0.04, "No successful runs for this selection", ha="center", va="center")
    else:
        cbar = fig.colorbar(im_left, ax=axes.ravel().tolist(), shrink=0.92, ticks=list(color_ticks))
        cbar.set_label("Minimal successful dimension")

    fig.suptitle(
        f"Minimal successful dimension for p={p_raw}\n{success_description(success_mode, min_margin)}; {seed_description(seed_reduce)}",
        y=0.98,
    )
    fig.subplots_adjust(left=0.08, right=0.90 if norm is not None else 0.96, bottom=0.12, top=0.84, wspace=0.12)

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_pdf}")
    return output_pdf



def main() -> int:
    args = parse_args()
    sigmoid_root = Path(args.sigmoid_root).expanduser().resolve()
    infonce_root = Path(args.infonce_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()

    if not sigmoid_root.is_dir():
        raise HeatmapDataError(f"sigmoid_root does not exist: {sigmoid_root}")
    if not infonce_root.is_dir():
        raise HeatmapDataError(f"infonce_root does not exist: {infonce_root}")

    print(f"Scanning sigmoid root: {sigmoid_root}")
    sigmoid_index = build_model_index(root=sigmoid_root, success_mode=args.success_mode, min_margin=args.min_margin)
    print(f"Found {len(sigmoid_index)} unique sigmoid (n,k,N,seed) runs")

    print(f"Scanning InfoNCE root: {infonce_root}")
    infonce_index = build_model_index(root=infonce_root, success_mode=args.success_mode, min_margin=args.min_margin)
    print(f"Found {len(infonce_index)} unique InfoNCE (n,k,N,seed) runs")

    p_pairs = [(p_raw, parse_p_value(p_raw)) for p_raw in args.p_values]

    tables_by_p: Dict[str, Tuple[np.ndarray, np.ndarray, List[int], List[int]]] = {}
    for p_raw, p_decimal in p_pairs:
        ns, ks = collect_grid_axes(sigmoid_index=sigmoid_index, infonce_index=infonce_index, p=p_decimal)
        if not ns or not ks:
            warnings.warn(f"No matching runs were found for p={p_raw}; skipping this p.", stacklevel=2)
            continue
        sigmoid_table = build_heatmap_table(
            model_index=sigmoid_index,
            ns=ns,
            ks=ks,
            p=p_decimal,
            seed_reduce=args.seed_reduce,
        )
        infonce_table = build_heatmap_table(
            model_index=infonce_index,
            ns=ns,
            ks=ks,
            p=p_decimal,
            seed_reduce=args.seed_reduce,
        )
        tables_by_p[p_raw] = (sigmoid_table, infonce_table, ns, ks)

    if not tables_by_p:
        raise HeatmapDataError("No plots could be created because none of the requested p values matched the available runs.")

    norm, color_ticks, cmap = build_shared_norm(
        [table for pair in tables_by_p.values() for table in pair[:2]]
    )

    written: List[Path] = []
    for p_raw, _p_decimal in p_pairs:
        if p_raw not in tables_by_p:
            continue
        sigmoid_table, infonce_table, ns, ks = tables_by_p[p_raw]
        output_pdf = output_root / f"min_success_dim_compare_p_{safe_p_string(p_raw)}.pdf"
        maybe = write_p_figure(
            p_raw=p_raw,
            output_pdf=output_pdf,
            sigmoid_table=sigmoid_table,
            infonce_table=infonce_table,
            ns=ns,
            ks=ks,
            norm=norm,
            color_ticks=color_ticks,
            cmap=cmap,
            success_mode=args.success_mode,
            seed_reduce=args.seed_reduce,
            min_margin=args.min_margin,
            figsize=(args.figwidth, args.figheight),
            annotate=args.annotate,
            overwrite=args.overwrite,
        )
        if maybe is not None:
            written.append(maybe)

    print("Done.")
    print(f"Requested p values: {[p for p, _ in p_pairs]}")
    print(f"Generated/updated {len(written)} PDF files under {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
