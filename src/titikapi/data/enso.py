"""El Niño state: NOAA CPC Oceanic Niño Index (ONI), lagged to what was published by the issue date.

ONI is a 3-month running mean of Niño-3.4 SST anomalies (DJF, JFM, ..., NDJ). A season's value
is published early in the month after it ends, so on any day in month m the newest available
value is the season that ended in month m-1.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import requests

from titikapi.config import RAW_DIR

ONI_URL = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
ONI_PATH = RAW_DIR / "enso" / "oni.ascii.txt"
SEASONS = ["DJF", "JFM", "FMA", "MAM", "AMJ", "MJJ", "JJA", "JAS", "ASO", "SON", "OND", "NDJ"]


def download_oni(dest: Path = ONI_PATH, force: bool = False) -> Path:
    if force or not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(ONI_URL, timeout=60)
        r.raise_for_status()
        dest.write_text(r.text)
    return dest


def parse_oni(text: str) -> pd.Series:
    """ONI anomaly indexed by the first day of the month in which each season ends."""
    rows = []
    for line in text.splitlines()[1:]:
        parts = line.split()
        if len(parts) != 4 or parts[0] not in SEASONS:
            continue
        season, year, anom = parts[0], int(parts[1]), float(parts[3])
        end_month = SEASONS.index(season) + 2  # DJF ends in Feb ... NDJ ends in Jan (next year)
        end_year = year + (end_month - 1) // 12
        rows.append((pd.Timestamp(end_year, (end_month - 1) % 12 + 1, 1), anom))
    return pd.Series(dict(rows), name="oni").sort_index()


def oni_known_at(dates: pd.DatetimeIndex, oni: pd.Series) -> np.ndarray:
    """ONI of the season that ended in the month before each date (NaN if not yet available)."""
    prev_month = (pd.DatetimeIndex(dates).to_period("M") - 1).to_timestamp()
    return oni.reindex(prev_month).to_numpy(dtype=np.float32)


def load_oni() -> pd.Series:
    return parse_oni(download_oni().read_text())
