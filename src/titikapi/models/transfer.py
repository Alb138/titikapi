"""Can live forecasts switch from S-NPP to NOAA-20 without retraining?

Suomi NPP stopped delivering data on 2 November 2026; the models were trained on S-NPP fire
detections. NOAA-20 carries the same VIIRS instrument in the same orbit, ~50 minutes apart. Over
the years both flew, this module compares their detections cell by cell and re-scores the trained
models with NOAA-20 fire history as input and NOAA-20 fires as the target.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from titikapi.config import LEAD_DAYS
from titikapi.data.firms import daily_fire_grids, load_detections
from titikapi.grid import Grid
from titikapi.models.dataset import Arrays, days_since_fire, grid_inputs
from titikapi.models.metrics import average_precision

NOAA20 = "viirs-jpss1"  # FIRMS archive name for NOAA-20 (JPSS-1); VIIRS data from April 2018
QUIET_DAYS = 14


def sensor_arrays(a: Arrays, years: range, sat: str = NOAA20) -> Arrays:
    """A copy of `a` whose fire channels (on, count, FRP, days since fire) come from `sat`,
    built exactly as build_arrays does for S-NPP. Days outside `years` are left fire-free."""
    grid = Grid()
    det = load_detections(years, grid, sat=sat)
    count = np.zeros(a.fire_on.shape, np.uint8)
    frp = np.zeros(a.fire_on.shape, np.float16)
    for year in years:  # one year at a time keeps the dense daily grids small
        ds = daily_fire_grids(det, pd.Timestamp(year, 1, 1), pd.Timestamp(year, 12, 31), grid)
        pos = a.time.get_indexer(pd.DatetimeIndex(ds["time"].values))
        keep = pos >= 0
        count[pos[keep]] = np.minimum(ds["fire_count"].values[keep][:, a.mask], 255)
        frp[pos[keep]] = np.log1p(ds["frp_sum"].values[keep][:, a.mask])
    on = count > 0
    return dataclasses.replace(
        a, fire_on=on, fire_count=count, frp_log=frp, days_since=days_since_fire(on)
    )


def agreement(snpp: Arrays, other: Arrays, days: np.ndarray) -> pd.DataFrame:
    """Per year: fire cell-days seen by each sensor and how often the other sensor agrees."""
    rows = []
    years = snpp.time[days].year
    for year in sorted(set(years)):
        d = days[years == year]
        s, o = np.asarray(snpp.fire_on[d]), np.asarray(other.fire_on[d])
        both = (s & o).sum()
        rows.append(
            {
                "year": year,
                "snpp_fire_cell_days": int(s.sum()),
                "noaa20_fire_cell_days": int(o.sum()),
                "ratio_noaa20_to_snpp": o.sum() / s.sum(),
                "share_snpp_also_noaa20": both / s.sum(),
                "share_noaa20_also_snpp": both / o.sum(),
            }
        )
    return pd.DataFrame(rows)


def _targets(a: Arrays, idx: np.ndarray) -> np.ndarray:
    return np.stack([[np.asarray(a.fire_on[t + d]) for d in range(1, LEAD_DAYS + 1)] for t in idx])


def predict(a: Arrays, idx: np.ndarray, predictors: dict, batch: int = 16) -> dict[str, np.ndarray]:
    """(D, 7, N) scores per model, from `a`'s fire channels; built in batches to save memory."""
    out = {name: [] for name in predictors}
    for i in range(0, len(idx), batch):
        days = idx[i : i + batch]
        x = np.stack([grid_inputs(a, t) for t in days])
        w = np.asarray(a.weather[days], np.float32)
        for name, f in predictors.items():
            out[name].append(f(x, w))
    return {name: np.concatenate(v) for name, v in out.items()}


def transfer_scores(
    snpp: Arrays, noaa20: Arrays, idx: np.ndarray, predictors: dict
) -> pd.DataFrame:
    """PR-AUC for: A) S-NPP in -> S-NPP out (as trained), B) S-NPP in -> NOAA-20 out,
    C) NOAA-20 in -> NOAA-20 out (the live setup). New ignitions are defined by the target
    sensor's own fire history."""
    p_snpp = predict(snpp, idx, predictors)
    p_n20 = predict(noaa20, idx, predictors)
    y = {"S-NPP": _targets(snpp, idx), "NOAA-20": _targets(noaa20, idx)}
    quiet = {
        "S-NPP": np.asarray(snpp.days_since[idx]) >= QUIET_DAYS,
        "NOAA-20": np.asarray(noaa20.days_since[idx]) >= QUIET_DAYS,
    }
    scenarios = {
        "A: S-NPP in, S-NPP fires": (p_snpp, "S-NPP"),
        "B: S-NPP in, NOAA-20 fires": (p_snpp, "NOAA-20"),
        "C: NOAA-20 in, NOAA-20 fires": (p_n20, "NOAA-20"),
    }
    rows = []
    for scenario, (preds, target) in scenarios.items():
        for model, p in preds.items():
            for lead in (1, LEAD_DAYS):
                yl, pl, q = y[target][:, lead - 1], p[:, lead - 1], quiet[target]
                rows.append(
                    {
                        "scenario": scenario,
                        "model": model,
                        "lead": lead,
                        "ap": average_precision(yl, pl),
                        "new_ap": average_precision(yl[q], pl[q]),
                        "base_rate": float(yl.mean()),
                    }
                )
    return pd.DataFrame(rows)
