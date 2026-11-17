"""NASA FIRMS VIIRS active-fire detections -> daily fire grids.

Training history comes from the yearly per-country archives (science-quality "SP" data, no API
key needed). Live data (2026+) comes from the FIRMS area API, which needs a free MAP_KEY.
"""

from __future__ import annotations

import io
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


ARCHIVE_LAST_YEAR = 2025  # newest yearly archive; later years come from the area API
API_URL = "https://firms.modaps.eosdis.nasa.gov/api"
API_DIR = FIRMS_DIR / "api"
API_MAX_DAYS = 5  # the area endpoint serves at most 5 days per request
STATIC_SOURCE_RADIUS = 0.005  # deg (~550 m): NRT detections this close to a flagged source


def map_key() -> str:
    """FIRMS MAP_KEY from the environment or the project's .env file."""
    from titikapi.config import ROOT

    key = os.environ.get("FIRMS_MAP_KEY")
    env = ROOT / ".env"
    if not key and env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("FIRMS_MAP_KEY="):
                key = line.split("=", 1)[1].strip()
    if not key:
        raise SystemExit("set FIRMS_MAP_KEY (free key: https://firms.modaps.eosdis.nasa.gov/api/)")
    return key


def api_availability(key: str) -> pd.DataFrame:
    r = requests.get(f"{API_URL}/data_availability/csv/{key}/all", timeout=60)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text), parse_dates=["min_date", "max_date"])


def download_api(
    source: str, start, end, grid: Grid, dest_dir: Path = API_DIR, use_cache: bool = True
) -> pd.DataFrame:
    """Detections for [start, end] (UTC dates) from the area API, cached as one CSV per 5-day
    request. Pass use_cache=False for recent near-real-time days, which keep receiving data."""
    key = map_key()
    r = grid.region
    bbox = f"{r.lon_min},{r.lat_min},{r.lon_max},{r.lat_max}"
    dest_dir.mkdir(parents=True, exist_ok=True)
    parts = []
    day, end = pd.Timestamp(start), pd.Timestamp(end)
    while day <= end:
        n = min(API_MAX_DAYS, (end - day).days + 1)
        path = dest_dir / f"{source}_{day.date()}_{n}d.csv"
        if not (use_cache and path.exists()):
            resp = requests.get(
                f"{API_URL}/area/csv/{key}/{source}/{bbox}/{n}/{day.date()}", timeout=300
            )
            resp.raise_for_status()
            if not resp.text.startswith("latitude"):  # errors come back as 200 + a message
                raise RuntimeError(f"FIRMS {source} {day.date()}: {resp.text[:200]}")
            tmp = path.with_suffix(".part")
            tmp.write_text(resp.text)
            os.replace(tmp, path)
        parts.append(read_firms_csv(path))
        day += pd.Timedelta(days=n)
    return pd.concat(parts, ignore_index=True)


def static_source_points(sp: list[pd.DataFrame]) -> np.ndarray:
    """(K, 2) lon/lat of detections FIRMS flagged as volcano/static source/offshore (type > 0)."""
    pts = [df.loc[df["type"] != VEGETATION_FIRE, ["longitude", "latitude"]] for df in sp]
    return np.unique(np.round(pd.concat(pts).to_numpy(), 4), axis=0)


def drop_static_sources(df: pd.DataFrame, points: np.ndarray) -> pd.DataFrame:
    """NRT files have no `type` column: drop detections at locations flagged in the SP record."""
    from scipy.spatial import cKDTree

    dist, _ = cKDTree(points).query(df[["longitude", "latitude"]].to_numpy(), k=1)
    return df.loc[dist > STATIC_SOURCE_RADIUS]


def load_api_detections(year: int, last_day, grid: Grid, sp_archives: list[Path]) -> pd.DataFrame:
    """S-NPP detections for `year` up to `last_day`: science-quality (SP) data where it exists,
    near-real-time (NRT) after that, with static sources removed from the NRT part."""
    avail = api_availability(map_key()).set_index("data_id")
    start, last_day = pd.Timestamp(year, 1, 1), pd.Timestamp(last_day)
    sp_end = min(avail.loc["VIIRS_SNPP_SP", "max_date"], last_day)
    sp = download_api("VIIRS_SNPP_SP", start, sp_end, grid)
    parts = [clean_detections(sp, grid)]
    if sp_end < last_day:
        nrt = download_api("VIIRS_SNPP_NRT", sp_end + pd.Timedelta(days=1), last_day, grid)
        flagged = [sp] + [
            pd.read_csv(p, usecols=["longitude", "latitude", "type"]) for p in sp_archives
        ]
        nrt = drop_static_sources(nrt, static_source_points(flagged))
        parts.append(clean_detections(nrt, grid))
    det = pd.concat(parts, ignore_index=True)
    return det.loc[det["local_date"] <= last_day]


def load_detections(
    years: range, grid: Grid, last_day=None, sat: str = "viirs-snpp"
) -> pd.DataFrame:
    """Cleaned detections: yearly archives up to ARCHIVE_LAST_YEAR (downloads missing years),
    the area API after that."""
    parts, archives = [], []
    for year in years:
        if year <= ARCHIVE_LAST_YEAR:
            archives.append(download_year(year, sat=sat))
            det = clean_detections(read_firms_csv(archives[-1]), grid)
        else:
            det = load_api_detections(year, last_day or pd.Timestamp(year, 12, 31), grid, archives)
        parts.append(det.astype({"row": np.int16, "col": np.int16}).drop(columns=["lon", "lat"]))
    return pd.concat(parts, ignore_index=True)


def build_fire_cube(years: range, out: Path, grid: Grid, first_day=None, last_day=None) -> Path:
    """Write daily fire grids for `years` (up to `last_day`) to a Zarr store, one year at a time.

    All detections are loaded first because a night overpass in one year's file can fall on
    1 January of the next year in local time.
    """
    det = load_detections(years, grid, last_day)
    for i, year in enumerate(years):
        start = pd.Timestamp(year, 1, 1)
        if i == 0 and first_day is not None:
            start = max(start, pd.Timestamp(first_day))
        end = pd.Timestamp(year, 12, 31)
        if last_day is not None:
            end = min(end, pd.Timestamp(last_day))
        ds = daily_fire_grids(det, start, end, grid)
        if i == 0:
            chunks = (64, *grid.shape)
            encoding = {v: {"chunks": chunks} for v in ds.data_vars}
            ds.to_zarr(out, mode="w", consolidated=False, encoding=encoding)
        else:
            ds.to_zarr(out, append_dim="time", consolidated=False)
        n_fire_days = int((ds["fire_count"] > 0).sum())
        print(f"{year}: {n_fire_days:,} fire cell-days", flush=True)
    return out
