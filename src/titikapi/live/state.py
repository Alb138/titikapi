"""Rolling state for the daily job: recent fire history (NOAA-20) and recent weather (IFS).

Fire: the last FIRE_WINDOW local solar days of detection counts and FRP on land cells, enough for
"days since fire" (capped at 255) and 14 days of history.
Weather: the last HISTORY_DAYS days of raw daily fields plus the Fire Weather Index codes carried
from day to day, so each new day's features equal what the training pipeline would compute.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from titikapi.data import firms
from titikapi.data.era5 import AREA
from titikapi.data.features import FWI_STATE_VARS, HISTORY_DAYS, weather_features
from titikapi.grid import Grid

FIRE_WINDOW = 256
LIVE_SENSOR = "VIIRS_NOAA20"  # S-NPP retired 2026-11-02; see titikapi.models.transfer
REFRESH_DAYS = 3  # re-fetch the newest days: near-real-time files keep receiving late granules
RAW_VARS = ("t2m", "d2m", "u10", "v10", "pr")


@dataclass
class FireState:
    dates: pd.DatetimeIndex  # consecutive local solar days, oldest first
    count: np.ndarray  # (T, N land cells) uint8 detections per cell-day
    frp_log: np.ndarray  # (T, N) float16 log1p(summed FRP, MW)

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path,
            dates=self.dates.values.astype("datetime64[D]"),
            count=self.count,
            frp_log=self.frp_log,
        )

    @classmethod
    def load(cls, path: Path) -> FireState:
        z = np.load(path)
        return cls(pd.DatetimeIndex(z["dates"]), z["count"], z["frp_log"])


def fetch_fire_days(start, end, mask: np.ndarray, static_points: np.ndarray) -> FireState:
    """NOAA-20 fire grids for local solar days [start, end]: science-quality (SP) data where
    available, near-real-time (NRT) after that, with known static sources removed from NRT."""
    grid = Grid()
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    avail = firms.api_availability(firms.map_key()).set_index("data_id")
    sp_end = avail.loc[f"{LIVE_SENSOR}_SP", "max_date"]
    # a local solar day spans two UTC dates in this region, so fetch one UTC day either side
    utc0, utc1 = start - pd.Timedelta(days=1), end + pd.Timedelta(days=1)
    parts = []
    if utc0 <= sp_end:
        sp = firms.download_api(f"{LIVE_SENSOR}_SP", utc0, min(utc1, sp_end), grid)
        parts.append(firms.clean_detections(sp, grid))
    if utc1 > sp_end:
        nrt = firms.download_api(
            f"{LIVE_SENSOR}_NRT",
            max(utc0, sp_end + pd.Timedelta(days=1)),
            utc1,
            grid,
            use_cache=False,
        )
        if len(nrt):
            nrt = firms.drop_static_sources(nrt, static_points)
            parts.append(firms.clean_detections(nrt, grid))
    det = pd.concat(parts, ignore_index=True)
    ds = firms.daily_fire_grids(det, start, end, grid)
    count = np.minimum(ds["fire_count"].values[:, mask], 255).astype(np.uint8)
    frp = np.log1p(ds["frp_sum"].values[:, mask]).astype(np.float16)
    return FireState(pd.DatetimeIndex(ds["time"].values), count, frp)


def update_fire(
    state: FireState | None, through, mask: np.ndarray, static_points: np.ndarray
) -> FireState:
    """Extend the fire state to `through`, re-fetching the newest REFRESH_DAYS days."""
    through = pd.Timestamp(through)
    if state is None:
        start = through - pd.Timedelta(days=FIRE_WINDOW - 1)
    else:
        start = min(state.dates[-1] - pd.Timedelta(days=REFRESH_DAYS - 1), through)
    new = fetch_fire_days(start, through, mask, static_points)
    if state is None:
        return new
    keep = state.dates < start
    dates = state.dates[keep].append(new.dates)
    count = np.concatenate([state.count[keep], new.count])[-FIRE_WINDOW:]
    frp = np.concatenate([state.frp_log[keep], new.frp_log])[-FIRE_WINDOW:]
    return FireState(dates[-FIRE_WINDOW:], count, frp)


# ---------------------------------------------------------------- weather


@dataclass
class WeatherState:
    raw: xr.Dataset  # last HISTORY_DAYS days of t2m, d2m (degC), u10, v10 (m/s), pr (mm)
    fwi: dict[str, xr.DataArray]  # ffmc/dmc/dc on raw's last day
    feats: xr.Dataset  # weather features for the most recent days (model inputs before scaling)

    @property
    def last_day(self) -> pd.Timestamp:
        return pd.Timestamp(self.raw["time"].values[-1])

    def save(self, path: Path) -> None:
        ds = self.raw.copy()
        for k, v in self.fwi.items():
            ds[f"state_{k}"] = v.drop_vars("time", errors="ignore")
        for k in self.feats.data_vars:
            ds[f"feat_{k}"] = self.feats[k].rename(time="feat_time")
        ds.to_netcdf(path)

    @classmethod
    def load(cls, path: Path) -> WeatherState:
        ds = xr.load_dataset(path)
        raw = ds[list(RAW_VARS)]
        fwi = {k: ds[f"state_{k}"] for k in FWI_STATE_VARS}
        feats = xr.Dataset(
            {k[5:]: ds[k].rename(feat_time="time") for k in ds.data_vars if k.startswith("feat_")}
        )
        return cls(raw, fwi, feats)


# ECMWF open data is mirrored on several clouds; when one is throttling (HTTP 503), try the next
# instead of waiting through the client's default 500 retries x 2 minutes.
IFS_SOURCES = ("google", "aws", "ecmwf")  # "ecmwf" only keeps the last few days


def _ifs_retrieve(target: str, **request) -> None:
    from ecmwf.opendata import Client

    errors = []
    for source in IFS_SOURCES:
        client = Client(source=source, model="ifs", resol="0p25", maximum_retries=3, retry_after=20)
        try:
            client.retrieve(target=target, **request)
            return
        except Exception as e:  # unavailable, throttled, or (for "ecmwf") too old
            errors.append(f"{source}: {str(e)[:120]}")
    raise RuntimeError("IFS open data unavailable from every mirror: " + " | ".join(errors))


def _read_grib(path: Path) -> xr.Dataset:
    """All fields in a GRIB file as one Dataset (2 m and 10 m fields need separate opening)."""
    import warnings

    import cfgrib

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)  # cfgrib's internal xr.merge default
        parts = cfgrib.open_datasets(str(path), backend_kwargs={"indexpath": ""})
    return xr.merge(
        [p.drop_vars("heightAboveGround", errors="ignore") for p in parts], compat="override"
    )


def _fetch_grib(path: Path, expect: set[str], **request) -> xr.Dataset:
    """Download once to a temporary name and rename when complete, so an interrupted download
    is never mistaken for a finished file. A file on disk is reused only if it has every
    expected field (past runs never change)."""
    if path.exists():
        try:
            ds = _read_grib(path)
            if expect <= set(ds.data_vars):
                return ds
        except Exception:
            pass
        path.unlink()
    tmp = path.with_suffix(".part")
    _ifs_retrieve(str(tmp), **request)
    tmp.replace(path)
    ds = _read_grib(path)
    missing = expect - set(ds.data_vars)
    if missing:
        raise RuntimeError(f"{path.name}: IFS file lacks {sorted(missing)}")
    return ds


def ifs_day(day, lat: np.ndarray, lon: np.ndarray, workdir: Path) -> xr.Dataset:
    """One day of daily fields from ECMWF IFS open data, matching the ERA5 training convention:
    instantaneous fields at 06 UTC (13:00 WIB) and rain over (day-1 06 UTC, day 06 UTC]."""
    day = pd.Timestamp(day)
    prev = day - pd.Timedelta(days=1)
    workdir.mkdir(parents=True, exist_ok=True)
    inst = _fetch_grib(
        workdir / f"inst_{day.date()}.grib2",
        {"t2m", "d2m", "u10", "v10"},
        date=str(day.date()),
        time=0,
        type="fc",
        step=6,
        param=["2t", "2d", "10u", "10v"],
    )
    tp = _fetch_grib(
        workdir / f"tp_{prev.date()}.grib2",
        {"tp"},
        date=str(prev.date()),
        time=0,
        type="fc",
        step=[6, 30],
        param="tp",
    )["tp"]
    north, west, south, east = AREA
    box = {"latitude": slice(north, south), "longitude": slice(west, east)}
    hours = [np.timedelta64(h, "h") for h in (6, 30)]
    rain = (tp.sel(step=hours[1]) - tp.sel(step=hours[0])).sel(**box) * 1000
    inst = inst.sel(**box)
    data = {
        "t2m": inst["t2m"] - 273.15,
        "d2m": inst["d2m"] - 273.15,
        "u10": inst["u10"],
        "v10": inst["v10"],
        "pr": rain.clip(min=0),
    }
    out = xr.Dataset(
        {k: (("lat", "lon"), np.asarray(v.values, np.float32)) for k, v in data.items()},
        coords={"lat": lat, "lon": lon},
    )
    return out.expand_dims(time=[day])


def update_weather(
    state: WeatherState, through, lat, lon, workdir: Path, keep_feats: int = 14
) -> WeatherState:
    """Append IFS days after state.last_day up to `through`, carrying the FWI codes. Keeps the
    features of the last `keep_feats` days (the live job needs only today's)."""
    raw, fwi, feats = state.raw, state.fwi, state.feats
    day = state.last_day + pd.Timedelta(days=1)
    while day <= pd.Timestamp(through):
        daily = ifs_day(day, lat, lon, workdir)
        f = weather_features(daily, state=fwi, history=raw)
        fwi = {k: f[k].isel(time=-1) for k in FWI_STATE_VARS}
        raw = xr.concat([raw, daily], "time").isel(time=slice(-HISTORY_DAYS, None))
        feats = xr.concat([feats, f], "time").isel(time=slice(-keep_feats, None))
        print(f"weather {day.date()}: mean FWI {float(f['fwi'].mean()):.1f}", flush=True)
        day += pd.Timedelta(days=1)
    return WeatherState(raw, fwi, feats)


def bootstrap_weather(last_day: dt.date) -> WeatherState:
    """Initial weather state from the ERA5 training data, ending at `last_day`."""
    from titikapi.config import CUBE_DIR
    from titikapi.data import era5

    last = pd.Timestamp(last_day)
    raw = era5.load_year(last.year).sel(time=slice(None, last))
    if raw.sizes["time"] < HISTORY_DAYS:
        prev = era5.load_year(last.year - 1)
        raw = xr.concat([prev, raw], "time")
    raw = raw[list(RAW_VARS)].isel(time=slice(-HISTORY_DAYS, None)).astype(np.float32)
    w = xr.open_zarr(CUBE_DIR / "weather.zarr", consolidated=False)
    feats = w.sel(time=slice(last - pd.Timedelta(days=13), last)).load()
    fwi = {k: feats[k].isel(time=-1) for k in FWI_STATE_VARS}
    return WeatherState(raw, fwi, feats)
