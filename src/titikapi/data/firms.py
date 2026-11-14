"""NASA FIRMS VIIRS active-fire detections -> daily fire grids.

Training history comes from the yearly per-country archives (science-quality "SP" data, no API
key needed). Live data (2026+) comes from the FIRMS area API, which needs a free MAP_KEY.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xarray as xr
from tqdm import tqdm

from titikapi.config import RAW_DIR
from titikapi.grid import Grid, local_solar_date

COUNTRY_URL = (
    "https://firms.modaps.eosdis.nasa.gov/data/country/{sat}/{year}/{sat}_{year}_{country}.csv"
)
FIRMS_DIR = RAW_DIR / "firms"

# type 0 = presumed vegetation fire; 1 = volcano, 2 = other static land source (e.g. gas flares,
# industry), 3 = offshore. Only vegetation fires are forecast targets.
VEGETATION_FIRE = 0
KEEP_CONFIDENCE = {"n", "h"}  # nominal + high; low-confidence detections are mostly noise


def download_year(
    year: int,
    sat: str = "viirs-snpp",
    country: str = "Indonesia",
    dest_dir: Path = FIRMS_DIR,
    force: bool = False,
) -> Path:
    """Download one yearly country archive. Skips the download if the file already exists."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{sat}_{year}_{country}.csv"
    if dest.exists() and not force:
        return dest
    url = COUNTRY_URL.format(sat=sat, year=year, country=country)
    tmp = dest.with_suffix(".part")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with (
            open(tmp, "wb") as f,
            tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar,
        ):
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    os.replace(tmp, dest)
    return dest


def read_firms_csv(path_or_buf) -> pd.DataFrame:
    return pd.read_csv(
        path_or_buf,
        dtype={"acq_time": str, "confidence": str, "satellite": str, "daynight": str},
    )


def clean_detections(df: pd.DataFrame, grid: Grid) -> pd.DataFrame:
    """Keep confident vegetation-fire detections inside the grid; add local date and cell index.

    Returns columns: lon, lat, frp, local_date, row, col.
    """
    keep = df["confidence"].str.lower().isin(KEEP_CONFIDENCE)
    if "type" in df.columns:  # archive (SP) files have it; NRT API files do not
        keep &= df["type"] == VEGETATION_FIRE
    df = df.loc[keep]

    row, col = grid.point_to_cell(df["longitude"].to_numpy(), df["latitude"].to_numpy())
    inside = row >= 0
    df = df.loc[inside]
    row, col = row[inside], col[inside]

    utc = pd.to_datetime(df["acq_date"] + " " + df["acq_time"].str.zfill(4), format="%Y-%m-%d %H%M")
    return pd.DataFrame(
        {
            "lon": df["longitude"].to_numpy(),
            "lat": df["latitude"].to_numpy(),
            "frp": df["frp"].to_numpy(dtype=np.float32),
            "local_date": local_solar_date(utc, df["longitude"].to_numpy()),
            "row": row,
            "col": col,
        }
    )


def daily_fire_grids(det: pd.DataFrame, start, end, grid: Grid) -> xr.Dataset:
    """Per-day, per-cell detection counts and summed fire radiative power (MW).

    Every day in [start, end] is present, so days without fires are explicit zeros.
    """
    days = pd.date_range(start, end, freq="D")
    ny, nx = grid.shape
    day_idx = days.get_indexer(pd.DatetimeIndex(det["local_date"]))
    ok = day_idx >= 0
    flat = (day_idx[ok] * ny + det["row"].to_numpy()[ok]) * nx + det["col"].to_numpy()[ok]
    size = len(days) * ny * nx
    count = np.bincount(flat, minlength=size).reshape(len(days), ny, nx)
    frp = np.bincount(flat, weights=det["frp"].to_numpy()[ok], minlength=size)
    frp = frp.reshape(len(days), ny, nx)

    coords = {"time": days, **grid.coords()}
    dims = ("time", "lat", "lon")
    return xr.Dataset(
        {
            "fire_count": (dims, np.minimum(count, np.iinfo(np.int16).max).astype(np.int16)),
            "frp_sum": (dims, frp.astype(np.float32)),
        },
        coords=coords,
        attrs={"source": "NASA FIRMS VIIRS S-NPP 375 m, confidence n/h, type 0"},
    )


def load_archive_detections(years: range, grid: Grid, sat: str = "viirs-snpp") -> pd.DataFrame:
    """Cleaned detections from all yearly archives (downloads missing years)."""
    parts = []
    for year in years:
        det = clean_detections(read_firms_csv(download_year(year, sat=sat)), grid)
        parts.append(det.astype({"row": np.int16, "col": np.int16}).drop(columns=["lon", "lat"]))
    return pd.concat(parts, ignore_index=True)


def build_fire_cube(years: range, out: Path, grid: Grid, first_day=None) -> Path:
    """Write daily fire grids for `years` to a Zarr store, one year at a time.

    All detections are loaded first because a night overpass in one year's file can fall on
    1 January of the next year in local time.
    """
    det = load_archive_detections(years, grid)
    for i, year in enumerate(years):
        start = pd.Timestamp(year, 1, 1)
        if i == 0 and first_day is not None:
            start = max(start, pd.Timestamp(first_day))
        ds = daily_fire_grids(det, start, pd.Timestamp(year, 12, 31), grid)
        if i == 0:
            chunks = (64, *grid.shape)
            encoding = {v: {"chunks": chunks} for v in ds.data_vars}
            ds.to_zarr(out, mode="w", consolidated=False, encoding=encoding)
        else:
            ds.to_zarr(out, append_dim="time", consolidated=False)
        n_fire_days = int((ds["fire_count"] > 0).sum())
        print(f"{year}: {n_fire_days:,} fire cell-days", flush=True)
    return out
