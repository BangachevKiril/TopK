#!/usr/bin/env python3
from __future__ import annotations

import argparse
import colorsys
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


@dataclass(frozen=True)
class MarginHistory:
    d: int
    steps: np.ndarray
    margins: np.ndarray
    run_dir: Path


class PlotDataError(RuntimeError):
    pass


GRAPH_STEM_PREFIX = "graph_n_"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create one side-by-side PDF comparison plot for one graph, with sigmoid on the left "
            "and InfoNCE on the right. Each run root should contain d_<dim> subfolders."
        )
    )
    parser.add_argument("--sigmoid_run_root", type=str, required=True)
    parser.add_argument("--infonce_run_root", type=str, required=True)
    parser.add_argument("--output_pdf", type=str, required=True)
    parser.add_argument(
        "--spectral_root",
        type=str,
        default=None,
        help=(
            "Optional explicit spectral root. If omitted, the script tries to infer the spectral file "
            "from graph_path saved in config.json by searching ancestor/spectral_bounds/<graph_filename>.npz."
        ),
    )
    parser.add_argument(
        "--spectral_mode",
        type=str,
        default="margin_upper_bound_sqrt_nN",
        choices=["margin_upper_bound_sqrt_nN", "objective", "custom_multiplier"],
    )
    parser.add_argument("--spectral_multiplier", type=float, default=1.0)
    parser.add_argument("--figwidth", type=float, default=15.0)
    parser.add_argument("--figheight", type=float, default=6.0)
    parser.add_argument("--linewidth", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=0.95)
    parser.add_argument(
        "--ymin_floor",
        type=float,
        default=0.0,
        help=(
            "Clamp the bottom of the shared y-axis to be at least this value. "
            "Default 0.0 clips tiny negative margins below the x-axis."
        ),
    )
    parser.add_argument(
        "--legend_outside",
        action="store_true",
        help="Place a single shared legend outside the plot area on the right",
    )
    return parser.parse_args()



def graph_stem(n: int, k: int, N: int, seed: int) -> str:
    return f"graph_n_{n}_k_{k}_N_{N}_seed_{seed}"



def is_graph_run_root(path: Path) -> bool:
    if not path.is_dir():
        return False
    if not path.name.startswith(GRAPH_STEM_PREFIX):
        return False
    for child in path.iterdir():
        if child.is_dir() and child.name.startswith("d_"):
            return True
    return False



def discover_dimensions(run_root: Path) -> List[int]:
    dims: List[int] = []
    for path in run_root.glob("d_*"):
        if not path.is_dir():
            continue
        try:
            dims.append(int(path.name.split("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return sorted(set(dims))



def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)



def load_reference_config(run_root: Path) -> tuple[dict, Path]:
    dims = discover_dimensions(run_root)
    if not dims:
        raise PlotDataError(f"No d_<dim> folders found under {run_root}")

    for d in dims:
        config_path = run_root / f"d_{d}" / "config.json"
        if config_path.is_file():
            return load_json(config_path), config_path

    raise PlotDataError(f"Could not find any config.json under {run_root}/d_*")



def _checkpoint_files_for_run(run_dir: Path) -> List[Path]:
    ckpts = sorted(run_dir.glob("checkpoint_step_*.npz"))
    if ckpts:
        return ckpts

    fallback: List[Path] = []
    for name in ("latest.npz", "final.npz"):
        path = run_dir / name
        if path.is_file():
            fallback.append(path)
    return fallback



def load_margin_history_for_dimension(run_dir: Path, d: int) -> MarginHistory:
    files = _checkpoint_files_for_run(run_dir)
    if not files:
        raise PlotDataError(f"No checkpoint files found in {run_dir}")

    rows: List[Tuple[float, float]] = []
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            if "margin" not in data:
                raise PlotDataError(f"File {path} does not contain a 'margin' entry")
            if "step" in data:
                step = float(np.asarray(data["step"]).item())
            else:
                step = math.nan
            margin = float(np.asarray(data["margin"]).item())
        rows.append((step, margin))

    rows.sort(key=lambda x: (math.inf if math.isnan(x[0]) else x[0]))
    steps = np.asarray([r[0] for r in rows], dtype=float)
    margins = np.asarray([r[1] for r in rows], dtype=float)
    return MarginHistory(d=d, steps=steps, margins=margins, run_dir=run_dir)



def load_all_margin_histories(run_root: Path, dims: Optional[Sequence[int]] = None) -> Dict[int, MarginHistory]:
    if dims is None:
        dims = discover_dimensions(run_root)
    else:
        dims = sorted(set(int(d) for d in dims))

    if not dims:
        raise PlotDataError(f"No d_<dim> folders found under {run_root}")

    histories: Dict[int, MarginHistory] = {}
    missing: List[int] = []
    for d in dims:
        run_dir = run_root / f"d_{d}"
        if not run_dir.is_dir():
            missing.append(d)
            continue
        try:
            histories[d] = load_margin_history_for_dimension(run_dir=run_dir, d=d)
        except PlotDataError:
            raise
        except Exception as exc:
            raise PlotDataError(f"Failed while reading {run_dir}: {exc}") from exc

    if missing:
        warnings.warn(f"Missing run directories for dimensions: {missing}", stacklevel=2)

    if not histories:
        raise PlotDataError(f"No valid margin histories could be loaded from {run_root}")

    return histories



def _load_npz_scalar(path: Path, preferred_keys: Sequence[str]) -> float:
    with np.load(path, allow_pickle=False) as data:
        for key in preferred_keys:
            if key in data:
                return float(np.asarray(data[key]).item())
    raise PlotDataError(f"None of the keys {list(preferred_keys)} were found in {path}")



def infer_spectral_file_from_graph_path(graph_path: Path) -> Path:
    stem = graph_path.stem
    filename = f"{stem}.npz"

    for ancestor in [graph_path.parent, *graph_path.parents]:
        candidate = ancestor / "spectral_bounds" / filename
        if candidate.is_file():
            return candidate.resolve()

    raise PlotDataError(
        f"Could not infer spectral file for {graph_path}. "
        f"Tried ancestor/spectral_bounds/{filename}."
    )



def load_spectral_bound(
    n: int,
    N: int,
    graph_path: Optional[Path],
    graph_stem_str: str,
    spectral_root: Optional[Path],
    mode: str,
    multiplier: float,
) -> tuple[float, Path]:
    if spectral_root is not None:
        spectral_root = spectral_root.expanduser().resolve()
        spectral_path = spectral_root / f"{graph_stem_str}.npz"
        if not spectral_path.is_file():
            candidates = sorted(spectral_root.rglob(f"{graph_stem_str}.npz"))
            if not candidates:
                raise PlotDataError(f"Could not find spectral file {graph_stem_str}.npz under {spectral_root}")
            if len(candidates) > 1:
                pretty = "\n".join(str(c) for c in candidates)
                raise PlotDataError(
                    f"Found multiple spectral candidates for {graph_stem_str}.npz under {spectral_root}:\n{pretty}"
                )
            spectral_path = candidates[0]
    else:
        if graph_path is None:
            raise PlotDataError("Cannot infer spectral file because graph_path is missing from config.json")
        spectral_path = infer_spectral_file_from_graph_path(graph_path)

    objective = _load_npz_scalar(spectral_path, preferred_keys=("objective", "spectral_norm_from_J"))

    if mode == "margin_upper_bound_sqrt_nN":
        value = objective * math.sqrt(n * N)
    elif mode == "objective":
        value = objective
    elif mode == "custom_multiplier":
        value = objective * multiplier
    else:
        raise ValueError(
            "spectral_mode must be one of 'margin_upper_bound_sqrt_nN', 'objective', or 'custom_multiplier'"
        )

    return float(value), spectral_path.resolve()



def build_margin_summary(histories: Dict[int, MarginHistory]) -> List[dict]:
    summary: List[dict] = []
    for d in sorted(histories):
        hist = histories[d]
        final_margin = float(hist.margins[-1]) if hist.margins.size else math.nan
        best_margin = float(np.nanmax(hist.margins)) if hist.margins.size else math.nan
        final_step = float(hist.steps[-1]) if hist.steps.size else math.nan
        summary.append(
            {
                "d": d,
                "num_checkpoints": int(hist.steps.size),
                "final_step": final_step,
                "final_margin": final_margin,
                "best_margin": best_margin,
                "run_dir": str(hist.run_dir),
            }
        )
    return summary



def validate_matching_graphs(sigmoid_config: dict, infonce_config: dict) -> tuple[int, int, int, int, Optional[Path]]:
    keys = ("n", "k", "N", "seed")
    sigmoid_tuple = tuple(int(sigmoid_config[key]) for key in keys)
    infonce_tuple = tuple(int(infonce_config[key]) for key in keys)
    if sigmoid_tuple != infonce_tuple:
        raise PlotDataError(
            "Sigmoid and InfoNCE run roots do not describe the same graph parameters: "
            f"sigmoid={sigmoid_tuple}, infonce={infonce_tuple}"
        )

    sigmoid_graph_path_raw = sigmoid_config.get("graph_path")
    infonce_graph_path_raw = infonce_config.get("graph_path")
    sigmoid_graph_path = Path(sigmoid_graph_path_raw).expanduser().resolve() if sigmoid_graph_path_raw else None
    infonce_graph_path = Path(infonce_graph_path_raw).expanduser().resolve() if infonce_graph_path_raw else None

    graph_path = sigmoid_graph_path or infonce_graph_path
    if sigmoid_graph_path is not None and infonce_graph_path is not None and sigmoid_graph_path != infonce_graph_path:
        warnings.warn(
            f"Sigmoid graph_path ({sigmoid_graph_path}) and InfoNCE graph_path ({infonce_graph_path}) differ. "
            f"Using {graph_path} to infer the spectral file.",
            stacklevel=2,
        )

    return (*sigmoid_tuple, graph_path)



def _reserved_hue_distance(h: float, center: float) -> float:
    return min(abs(h - center), 1.0 - abs(h - center))



def build_dimension_color_map(dims: Sequence[int]) -> Dict[int, tuple[float, float, float]]:
    dims = sorted(set(int(d) for d in dims))
    colors: Dict[int, tuple[float, float, float]] = {}
    golden = 0.618033988749895
    reserved_hues = [0.0, 2.0 / 3.0]  # red and blue, used for reference lines

    hue = 0.11
    for d in dims:
        for _ in range(200):
            bad = any(_reserved_hue_distance(hue, center) < 0.07 for center in reserved_hues)
            if not bad:
                break
            hue = (hue + 0.11) % 1.0

        rgb = colorsys.hls_to_rgb(hue, 0.48, 0.85)
        colors[d] = rgb
        hue = (hue + golden) % 1.0

    return colors



def compute_shared_ylim(
    sigmoid_histories: Dict[int, MarginHistory],
    infonce_histories: Dict[int, MarginHistory],
    spectral_bound: float,
    ymin_floor: float,
) -> tuple[float, float]:
    values: List[float] = [float(spectral_bound), 0.0]

    for histories in (sigmoid_histories, infonce_histories):
        for hist in histories.values():
            if hist.margins.size:
                finite = hist.margins[np.isfinite(hist.margins)]
                values.extend(finite.tolist())

    if not values:
        return ymin_floor, ymin_floor + 1.0

    ymin_data = min(values)
    ymax_data = max(values)

    ymin = max(float(ymin_floor), float(ymin_data))
    ymax = float(ymax_data)

    if ymax <= ymin:
        spread = max(1e-3, 0.05 * max(1.0, abs(ymax)))
        ymax = ymin + spread
    else:
        pad = 0.05 * (ymax - ymin)
        ymax = ymax + pad

    return ymin, ymax



def panel_title(loss_name: str, n: int, k: int, N: int, seed: int) -> str:
    return f"{loss_name}\nn={n}, k={k}, N={N}"



def plot_compare_run_roots_to_pdf(
    sigmoid_run_root: Path,
    infonce_run_root: Path,
    output_pdf: Path,
    spectral_root: Optional[Path] = None,
    spectral_mode: str = "margin_upper_bound_sqrt_nN",
    spectral_multiplier: float = 1.0,
    figsize: tuple[float, float] = (15.0, 6.0),
    linewidth: float = 2.0,
    alpha: float = 0.95,
    legend_outside: bool = False,
    ymin_floor: float = 0.0,
) -> dict:
    sigmoid_run_root = sigmoid_run_root.expanduser().resolve()
    infonce_run_root = infonce_run_root.expanduser().resolve()
    output_pdf = output_pdf.expanduser().resolve()

    if not is_graph_run_root(sigmoid_run_root):
        raise PlotDataError(
            f"sigmoid_run_root does not look like a graph-specific embedding directory with d_<dim> subfolders: {sigmoid_run_root}"
        )
    if not is_graph_run_root(infonce_run_root):
        raise PlotDataError(
            f"infonce_run_root does not look like a graph-specific embedding directory with d_<dim> subfolders: {infonce_run_root}"
        )

    sigmoid_config, sigmoid_config_path = load_reference_config(sigmoid_run_root)
    infonce_config, infonce_config_path = load_reference_config(infonce_run_root)
    n, k, N, seed, graph_path = validate_matching_graphs(sigmoid_config, infonce_config)
    stem = graph_stem(n=n, k=k, N=N, seed=seed)

    sigmoid_histories = load_all_margin_histories(sigmoid_run_root)
    infonce_histories = load_all_margin_histories(infonce_run_root)
    all_dims = sorted(set(sigmoid_histories) | set(infonce_histories))
    color_map = build_dimension_color_map(all_dims)

    spectral_bound, spectral_path = load_spectral_bound(
        n=n,
        N=N,
        graph_path=graph_path,
        graph_stem_str=stem,
        spectral_root=spectral_root,
        mode=spectral_mode,
        multiplier=spectral_multiplier,
    )

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    left_ax, right_ax = axes

    for d in sorted(sigmoid_histories):
        hist = sigmoid_histories[d]
        left_ax.plot(
            hist.steps,
            hist.margins,
            color=color_map[d],
            linewidth=linewidth,
            alpha=alpha,
        )

    for d in sorted(infonce_histories):
        hist = infonce_histories[d]
        right_ax.plot(
            hist.steps,
            hist.margins,
            color=color_map[d],
            linewidth=linewidth,
            alpha=alpha,
        )

    for ax in axes:
        ax.axhline(
            spectral_bound,
            color="blue",
            linestyle="--",
            linewidth=2.0,
        )
        ax.axhline(
            0.0,
            color="red",
            linestyle=":",
            linewidth=2.0,
        )
        ax.set_xlabel("Training step")
        ax.grid(True, alpha=0.25)

    left_ax.set_ylabel("Margin")
    left_ax.set_title(panel_title("Sigmoid loss", n=n, k=k, N=N, seed=seed))
    right_ax.set_title(panel_title("InfoNCE loss", n=n, k=k, N=N, seed=seed))

    ymin, ymax = compute_shared_ylim(
        sigmoid_histories=sigmoid_histories,
        infonce_histories=infonce_histories,
        spectral_bound=spectral_bound,
        ymin_floor=ymin_floor,
    )
    left_ax.set_ylim(ymin, ymax)

    legend_handles: List[Line2D] = []
    legend_labels: List[str] = []
    for d in all_dims:
        legend_handles.append(Line2D([0], [0], color=color_map[d], linewidth=linewidth))
        legend_labels.append(f"d={d}")
    legend_handles.extend(
        [
            Line2D([0], [0], color="blue", linestyle="--", linewidth=2.0),
            Line2D([0], [0], color="red", linestyle=":", linewidth=2.0),
        ]
    )
    legend_labels.extend([f"spectral bound = {spectral_bound:.4g}", "0"])

    if legend_outside:
        ncol = 1 if len(legend_handles) <= 12 else 2
        fig.legend(
            legend_handles,
            legend_labels,
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            frameon=False,
            ncol=ncol,
        )
        fig.tight_layout(rect=(0.0, 0.0, 0.86, 1.0))
    else:
        ncol = min(5, max(2, int(math.ceil(len(legend_handles) / 4))))
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.02),
            frameon=False,
            ncol=ncol,
        )
        fig.tight_layout(rect=(0.0, 0.07, 1.0, 1.0))

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)

    return {
        "sigmoid_run_root": str(sigmoid_run_root),
        "infonce_run_root": str(infonce_run_root),
        "output_pdf": str(output_pdf),
        "sigmoid_config_path": str(sigmoid_config_path),
        "infonce_config_path": str(infonce_config_path),
        "graph_path": str(graph_path) if graph_path is not None else None,
        "spectral_path": str(spectral_path),
        "spectral_bound": spectral_bound,
        "sigmoid_summary": build_margin_summary(sigmoid_histories),
        "infonce_summary": build_margin_summary(infonce_histories),
        "shared_y_limits": [ymin, ymax],
    }



def main() -> None:
    args = parse_args()
    result = plot_compare_run_roots_to_pdf(
        sigmoid_run_root=Path(args.sigmoid_run_root),
        infonce_run_root=Path(args.infonce_run_root),
        output_pdf=Path(args.output_pdf),
        spectral_root=Path(args.spectral_root) if args.spectral_root else None,
        spectral_mode=args.spectral_mode,
        spectral_multiplier=args.spectral_multiplier,
        figsize=(args.figwidth, args.figheight),
        linewidth=args.linewidth,
        alpha=args.alpha,
        legend_outside=args.legend_outside,
        ymin_floor=args.ymin_floor,
    )

    print(f"Saved PDF: {result['output_pdf']}")
    print(f"Sigmoid run root: {result['sigmoid_run_root']}")
    print(f"InfoNCE run root: {result['infonce_run_root']}")
    print(f"Sigmoid config: {result['sigmoid_config_path']}")
    print(f"InfoNCE config: {result['infonce_config_path']}")
    print(f"Graph path: {result['graph_path']}")
    print(f"Spectral path: {result['spectral_path']}")
    print(f"Spectral bound: {result['spectral_bound']:.12g}")
    print(f"Shared y-limits: {tuple(result['shared_y_limits'])}")

    print("[Sigmoid]")
    for row in result["sigmoid_summary"]:
        print(
            f"d={row['d']:>4} | checkpoints={row['num_checkpoints']:>4} | "
            f"final_step={row['final_step']:.0f} | final_margin={row['final_margin']:.6g} | "
            f"best_margin={row['best_margin']:.6g}"
        )

    print("[InfoNCE]")
    for row in result["infonce_summary"]:
        print(
            f"d={row['d']:>4} | checkpoints={row['num_checkpoints']:>4} | "
            f"final_step={row['final_step']:.0f} | final_margin={row['final_margin']:.6g} | "
            f"best_margin={row['best_margin']:.6g}"
        )


if __name__ == "__main__":
    main()
