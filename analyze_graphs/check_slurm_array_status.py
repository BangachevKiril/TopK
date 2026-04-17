#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import List, Tuple


OOM_PATTERNS = [
    re.compile(r"oom[- ]kill", re.IGNORECASE),
    re.compile(r"out of memory", re.IGNORECASE),
    re.compile(r"detected\s+\d+\s+oom-kill event", re.IGNORECASE),
    re.compile(r"memory limit", re.IGNORECASE),
]

SAVED_PATTERN = re.compile(r"^\s*saved\b", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check statuses of Slurm array-job logs named like "
            "spectral_bounds_<jobid>_<taskid>.out/.err and write a summary file."
        )
    )
    parser.add_argument("logs_dir", type=str, help="Path to the logs directory.")
    parser.add_argument("jobid", type=str, help="Array job id, e.g. 11863964.")
    parser.add_argument(
        "total_jobs",
        type=int,
        help="Total number of array tasks. The script checks task ids 0, 1, ..., total_jobs - 1.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="spectral_bounds",
        help="Log filename prefix. Default: spectral_bounds",
    )
    return parser.parse_args()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def has_saved_line(out_text: str) -> bool:
    for line in out_text.splitlines():
        if SAVED_PATTERN.search(line):
            return True
    return False


def is_oom_killed(err_text: str) -> bool:
    return any(pattern.search(err_text) for pattern in OOM_PATTERNS)


def classify_task(out_path: Path, err_path: Path) -> Tuple[str, str]:
    out_exists = out_path.is_file()
    err_exists = err_path.is_file()

    if out_exists:
        out_text = read_text(out_path)
        if has_saved_line(out_text):
            return "Completed", "Found a Saved line in the .out log."

    if err_exists:
        err_text = read_text(err_path)
        if is_oom_killed(err_text):
            return "OOM killed", "Matched an OOM pattern in the .err log."

    if not out_exists or not err_exists:
        missing_parts = []
        if not out_exists:
            missing_parts.append(".out missing")
        if not err_exists:
            missing_parts.append(".err missing")
        return "Missing", ", ".join(missing_parts)

    return "Otherwise failed", "Logs exist, but no Saved line and no OOM pattern were found."


def build_report(logs_dir: Path, jobid: str, total_jobs: int, prefix: str) -> Tuple[str, List[int]]:
    if total_jobs < 0:
        raise ValueError(f"total_jobs must be nonnegative, got {total_jobs}.")

    lines: List[str] = []
    lines.append(f"logs_dir: {logs_dir}")
    lines.append(f"jobid: {jobid}")
    lines.append(f"total_jobs: {total_jobs}")
    lines.append(f"prefix: {prefix}")
    lines.append("")
    lines.append("Per-task status:")

    not_completed: List[int] = []
    counts = {
        "Completed": 0,
        "Missing": 0,
        "OOM killed": 0,
        "Otherwise failed": 0,
    }

    for k in range(total_jobs):
        stem = f"{prefix}_{jobid}_{k}"
        out_path = logs_dir / f"{stem}.out"
        err_path = logs_dir / f"{stem}.err"
        status, reason = classify_task(out_path, err_path)
        counts[status] += 1
        if status != "Completed":
            not_completed.append(k)
        lines.append(f"{k}: {status}")
        lines.append(f"    out: {out_path.name}")
        lines.append(f"    err: {err_path.name}")
        lines.append(f"    reason: {reason}")

    lines.append("")
    lines.append("Summary:")
    lines.append(f"Completed: {counts['Completed']}")
    lines.append(f"Missing: {counts['Missing']}")
    lines.append(f"OOM killed: {counts['OOM killed']}")
    lines.append(f"Otherwise failed: {counts['Otherwise failed']}")
    lines.append("")
    lines.append(f"Not completed ({len(not_completed)} jobs):")
    lines.append(json.dumps(sorted(not_completed)))

    return "\n".join(lines) + "\n", sorted(not_completed)


def main() -> int:
    args = parse_args()

    logs_dir = Path(args.logs_dir).expanduser().resolve()
    if not logs_dir.exists():
        raise FileNotFoundError(f"Logs directory not found: {logs_dir}")
    if not logs_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {logs_dir}")

    report_text, not_completed = build_report(
        logs_dir=logs_dir,
        jobid=args.jobid,
        total_jobs=args.total_jobs,
        prefix=args.prefix,
    )

    output_path = logs_dir / f"{args.jobid}_stats.txt"
    output_path.write_text(report_text, encoding="utf-8")

    print(f"Wrote: {output_path}")
    print(f"Not completed ({len(not_completed)}): {not_completed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
