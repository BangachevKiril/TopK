#!/usr/bin/env python3
"""Validate completeness and exact error-rate identities for the k=2 sweep."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def scalar(data: np.lib.npyio.NpzFile, name: str) -> float:
    return float(np.asarray(data[name]).reshape(()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("embeddings_root", type=Path)
    args = parser.parse_args()

    complete = list(args.embeddings_root.rglob("COMPLETE"))
    finals = list(args.embeddings_root.rglob("final.npz"))
    histories = list(args.embeddings_root.rglob("margin_history.npz"))

    assert len(complete) == 24, f"expected 24 COMPLETE markers, found {len(complete)}"
    assert len(finals) == 624, f"expected 624 final files, found {len(finals)}"
    assert len(histories) == 24, f"expected 24 histories, found {len(histories)}"

    for path in finals:
        with np.load(path, allow_pickle=False) as data:
            margin = scalar(data, "margin")
            fp = int(scalar(data, "false_positive_count"))
            negatives = int(scalar(data, "negative_count"))
            fpr = scalar(data, "false_positive_rate")
            fn = int(scalar(data, "false_negative_count"))
            positives = int(scalar(data, "positive_count"))
            fnr = scalar(data, "false_negative_rate")

        assert np.isfinite(margin), f"nonfinite margin: {path}"
        assert negatives > 0 and positives > 0, f"invalid denominators: {path}"
        assert 0 <= fp <= negatives and 0 <= fn <= positives, f"invalid counts: {path}"
        assert np.isfinite(fpr) and 0 <= fpr <= 1, f"invalid FPR: {path}"
        assert np.isfinite(fnr) and 0 <= fnr <= 1, f"invalid FNR: {path}"
        # Division order can differ by one float64 ULP between torch and NumPy.
        assert np.isclose(fpr, fp / negatives, rtol=4 * np.finfo(float).eps, atol=0), (
            f"FPR identity mismatch: {path}"
        )
        assert np.isclose(fnr, fn / positives, rtol=4 * np.finfo(float).eps, atol=0), (
            f"FNR identity mismatch: {path}"
        )

    for path in histories:
        with np.load(path, allow_pickle=False) as data:
            steps = np.asarray(data["steps"])
            margins = np.asarray(data["margins"])
        assert steps.shape == (100,), f"unexpected steps shape: {path}: {steps.shape}"
        assert int(steps[-1]) == 100000, f"history incomplete: {path}: {steps[-1]}"
        assert margins.shape == (100, 26), f"unexpected margin shape: {path}: {margins.shape}"
        assert np.isfinite(margins).all(), f"nonfinite history: {path}"

    print(
        f"VALID complete={len(complete)} finals={len(finals)} histories={len(histories)} "
        "rates_and_counts=exact histories_end=100000"
    )


if __name__ == "__main__":
    main()
