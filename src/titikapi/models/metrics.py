"""Forecast metrics for a rare binary target."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def average_precision(y: np.ndarray, p: np.ndarray) -> float:
    """PR-AUC (average precision). Unlike ROC-AUC it isn't flattered by the ~98% non-fire cells."""
    y = np.asarray(y).ravel()
    if y.sum() == 0:
        return float("nan")
    return float(average_precision_score(y, np.asarray(p).ravel()))


def brier(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=float).ravel()
    p = np.asarray(p, dtype=float).ravel()
    return float(np.mean((p - y) ** 2))


def topk_coverage(
    y: np.ndarray, p: np.ndarray, frac: float = 0.05, count: np.ndarray | None = None
) -> float:
    """Share of fires that fall in each day's top `frac` riskiest cells.

    The operational question: if patrols could cover only 5% of the region each day, ranked by
    the forecast, what share of fires would they be in the right place for? Cells are ranked
    among *all* cells; `count` restricts which fires are counted (e.g. new ignitions only).
    Ties are broken arbitrarily.
    """
    y = np.asarray(y, bool)
    count = y if count is None else (y & np.asarray(count, bool))
    k = max(1, int(np.ceil(frac * y.shape[1])))
    top = np.argpartition(-np.asarray(p), k - 1, axis=1)[:, :k]
    captured = np.take_along_axis(count, top, axis=1).sum()
    total = count.sum()
    return float(captured / total) if total else float("nan")


def reliability(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Calibration table: mean forecast vs observed frequency in quantile bins of the forecast."""
    y = np.asarray(y, float).ravel()
    p = np.asarray(p, float).ravel()
    edges = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    which = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    df = pd.DataFrame({"bin": which, "p": p, "y": y})
    return df.groupby("bin").agg(mean_forecast=("p", "mean"), observed=("y", "mean"), n=("y", "size"))
