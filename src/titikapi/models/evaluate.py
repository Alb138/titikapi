"""Evaluate a forecast over a set of issue days, overall and on the new-ignition slice."""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable

import numpy as np
import pandas as pd

from titikapi.config import LEAD_DAYS
from titikapi.data.cube import FireMatrix
from titikapi.models.metrics import average_precision, brier, topk_coverage

PredictFn = Callable[[np.ndarray, int], np.ndarray]  # (issue_idx, lead) -> (D, N) probabilities


def issue_days(
    fm: FireMatrix, split: tuple[dt.date, dt.date], max_lead: int = LEAD_DAYS, history: int = 14
) -> np.ndarray:
    """Indices of issue days t in the split whose targets t + max_lead stay inside the split,
    and that have `history` days of fire data before them."""
    start, end = pd.Timestamp(split[0]), pd.Timestamp(split[1])
    t = np.arange(len(fm.time))
    ok = (fm.time >= start) & (fm.time + pd.Timedelta(days=max_lead) <= end) & (t >= history - 1)
    ok &= t + max_lead < len(fm.time)
    return t[ok]


def evaluate(
    name: str,
    predict: PredictFn,
    fm: FireMatrix,
    recent: np.ndarray,
    issue_idx: np.ndarray,
    leads=range(1, LEAD_DAYS + 1),
    topk: float = 0.05,
) -> pd.DataFrame:
    """One row per (lead, slice).

    Slices: `all` cells, and `new_ignition` = cells with no fire in the 14 days up to the issue
    day (known at forecast time). Persistence can't anticipate those fires by construction.
    """
    rows = []
    for lead in leads:
        p = np.asarray(predict(issue_idx, lead), dtype=np.float32)
        y = fm.on[issue_idx + lead]
        quiet = ~recent[issue_idx]
        for slice_name, sel in (("all", None), ("new_ignition", quiet)):
            ys = y if sel is None else y[sel]
            ps = p if sel is None else p[sel]
            rows.append(
                {
                    "model": name,
                    "lead": lead,
                    "slice": slice_name,
                    "ap": average_precision(ys, ps),
                    "brier": brier(ys, ps),
                    f"top{round(topk * 100)}_coverage": topk_coverage(y, p, topk, count=sel),
                    "base_rate": float(ys.mean()),
                    "n_pos": int(ys.sum()),
                }
            )
    return pd.DataFrame(rows)
