"""Model inputs: compact on-disk arrays, and per-issue-day samples assembled from them.

`build_arrays()` turns the cubes into memory-mappable .npy files (data/cube/arrays). A sample
for issue day t uses only data known on day t:

    grid inputs  (C, ny, nx)   fire history (14 days), fire intensity, days since last fire,
                               static land layers, forest loss before year(t), climatology for
                               the target months, ONI, season, land mask
    weather      (Cw, 55, 101) day-t weather features at native 0.25 deg (upsampled on GPU)
    targets      (7, ny, nx)   fire on days t+1 .. t+7
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from titikapi.config import CUBE_DIR, FIRE_HISTORY_DAYS, LEAD_DAYS, SPLITS

ARRAYS_DIR = CUBE_DIR / "arrays"
WEATHER_VARS = (
    "t2m", "rh", "wind", "u10", "v10", "pr", "dry_days",
    "rain_7d", "rain_30d", "rain_60d", "rain_90d",
    "ffmc", "dmc", "dc", "isi", "bui", "fwi",
)  # fmt: skip
STATIC_VARS = ("peat_frac", "elev_mean", "elev_std", "treecover2000")
DAYS_SINCE_CAP = 255


def days_since_fire(on: np.ndarray, cap: int = DAYS_SINCE_CAP) -> np.ndarray:
    """(T, N) days since the last fire in each cell, up to and including day t (0 = fire today)."""
    out = np.empty(on.shape, np.uint8)
    run = np.full(on.shape[1], cap, np.int32)
    for t in range(on.shape[0]):
        run = np.where(on[t], 0, np.minimum(run + 1, cap))
        out[t] = run
    return out


def _train_days(time: pd.DatetimeIndex) -> np.ndarray:
    start, end = (pd.Timestamp(d) for d in SPLITS["train"])
    return np.flatnonzero((time >= start) & (time <= end))


def build_arrays(cube_dir: Path = CUBE_DIR, out: Path = ARRAYS_DIR) -> None:
    """Cubes -> data/cube/arrays/*.npy. Normalisation statistics come from training days only."""
    from titikapi.data.cube import load_fire_matrix, load_static
    from titikapi.data.enso import oni_known_at, parse_oni
    from titikapi.models.baselines import Climatology

    out.mkdir(parents=True, exist_ok=True)
    fm = load_fire_matrix(cube_dir)
    static = load_static(cube_dir)
    time = fm.time
    train = _train_days(time)
    np.save(out / "time.npy", time.values.astype("datetime64[D]"))
    np.save(out / "mask.npy", fm.mask)
    np.save(out / "fire_on.npy", fm.on)
    np.save(out / "days_since.npy", days_since_fire(fm.on))

    fire = xr.open_zarr(cube_dir / "fire.zarr", consolidated=False)
    count = np.zeros(fm.on.shape, np.uint8)
    frp = np.zeros(fm.on.shape, np.float16)
    for i in range(0, len(time), 366):
        block = fire.isel(time=slice(i, i + 366))
        count[i : i + 366] = np.minimum(block["fire_count"].values[:, fm.mask], 255)
        frp[i : i + 366] = np.log1p(block["frp_sum"].values[:, fm.mask])
    np.save(out / "fire_count.npy", count)
    np.save(out / "frp_log.npy", frp)

    # static layers, standardised over land cells
    stats = {}
    layers = []
    for v in STATIC_VARS:
        a = static[v].values.astype(np.float32)
        if v in ("elev_mean", "elev_std"):
            a = np.log1p(np.maximum(a, 0))
        mu, sd = float(a[fm.mask].mean()), float(a[fm.mask].std() + 1e-6)
        stats[v] = (mu, sd)
        layers.append((a - mu) / sd)
    np.save(out / "static.npy", np.stack(layers).astype(np.float32))
    np.save(out / "forest_loss.npy", static["forest_loss"].values.astype(np.float32))
    np.save(out / "loss_years.npy", static["loss_year"].values.astype(np.int16))
    np.save(out / "province.npy", fm.province)

    np.save(out / "clim.npy", Climatology().fit(fm, train).rate)
    oni = parse_oni((cube_dir.parent / "raw" / "enso" / "oni.ascii.txt").read_text())
    np.save(out / "oni.npy", np.nan_to_num(oni_known_at(time, oni)))

    # weather at native resolution, aligned to the fire time axis
    w = xr.open_zarr(cube_dir / "weather.zarr", consolidated=False)
    w = w.sel(time=time)
    wx = np.zeros((len(time), len(WEATHER_VARS), w.sizes["lat"], w.sizes["lon"]), np.float16)
    for k, v in enumerate(WEATHER_VARS):
        a = w[v].values.astype(np.float32)
        if v in ("pr", "rain_7d", "rain_30d", "rain_60d", "rain_90d", "dmc", "dc", "bui", "fwi"):
            a = np.log1p(np.maximum(a, 0))  # heavy right tails
        mu, sd = float(a[train].mean()), float(a[train].std() + 1e-6)
        stats[v] = (mu, sd)
        wx[:, k] = (a - mu) / sd
    np.save(out / "weather.npy", wx)
    np.save(out / "weather_lat.npy", w["lat"].values.astype(np.float32))
    np.save(out / "weather_lon.npy", w["lon"].values.astype(np.float32))
    (out / "stats.json").write_text(json.dumps(stats, indent=1))


@dataclass
class Arrays:
    """Memory-mapped view of data/cube/arrays."""

    time: pd.DatetimeIndex
    mask: np.ndarray
    fire_on: np.ndarray
    fire_count: np.ndarray
    frp_log: np.ndarray
    days_since: np.ndarray
    static: np.ndarray
    forest_loss: np.ndarray
    loss_years: np.ndarray
    clim: np.ndarray
    oni: np.ndarray
    weather: np.ndarray
    weather_lat: np.ndarray
    weather_lon: np.ndarray

    @classmethod
    def load(cls, path: Path = ARRAYS_DIR) -> Arrays:
        def ld(name):
            return np.load(path / f"{name}.npy", mmap_mode="r")

        return cls(
            time=pd.DatetimeIndex(np.load(path / "time.npy")),
            **{k: ld(k) for k in cls.__dataclass_fields__ if k != "time"},
        )

    @property
    def n_grid_channels(self) -> int:
        return FIRE_HISTORY_DAYS + 3 + self.static.shape[0] + 2 + 2 + 4


def grid_inputs(a: Arrays, t: int) -> np.ndarray:
    """(C, ny, nx) float32 inputs on the 0.1 deg grid for issue day t (data up to day t only)."""
    ny, nx = a.mask.shape
    m = a.mask

    def to_grid(cells: np.ndarray) -> np.ndarray:
        g = np.zeros((ny, nx), np.float32)
        g[m] = cells
        return g

    ch = [to_grid(a.fire_on[t - k]) for k in range(FIRE_HISTORY_DAYS)]  # today first
    ch.append(to_grid(np.log1p(a.fire_count[t].astype(np.float32))))
    ch.append(to_grid(a.frp_log[t].astype(np.float32)))
    ch.append(to_grid(a.days_since[t].astype(np.float32) / DAYS_SINCE_CAP))
    ch.extend(np.asarray(a.static))

    # forest cleared in the 3 years before year(t), and in all years before that (2001..)
    year = a.time[t].year
    before = a.loss_years < year
    recent = before & (a.loss_years >= year - 3)
    ch.append(np.asarray(a.forest_loss[recent]).sum(0) * 10)  # ~0..1 scale
    ch.append(np.asarray(a.forest_loss[before]).sum(0) * 3)

    # climatology (log rate) for the months of the first and last target day
    for lead in (1, LEAD_DAYS):
        month = (a.time[t] + pd.Timedelta(days=lead)).month
        ch.append(to_grid(np.log(a.clim[month] + 1e-4) / 5 + 1))

    doy = a.time[t].day_of_year / 365.25 * 2 * np.pi
    for value in (np.sin(doy), np.cos(doy), float(a.oni[t]) / 2):
        ch.append(np.full((ny, nx), value, np.float32))
    ch.append(m.astype(np.float32))
    return np.stack(ch).astype(np.float32)


def targets(a: Arrays, t: int) -> np.ndarray:
    """(7, ny, nx) uint8: fire on days t+1 .. t+7."""
    ny, nx = a.mask.shape
    y = np.zeros((LEAD_DAYS, ny, nx), np.uint8)
    for d in range(1, LEAD_DAYS + 1):
        y[d - 1][a.mask] = a.fire_on[t + d]
    return y


def quiet_cells(a: Arrays, t: int, window: int = 14) -> np.ndarray:
    """(ny, nx) bool: land cells with no fire in the `window` days up to t (new-ignition slice)."""
    g = np.zeros(a.mask.shape, bool)
    g[a.mask] = a.days_since[t] >= window
    return g


class FireDataset:
    """torch-style dataset of issue days -> (grid inputs, weather, targets, quiet mask, t).

    Pass a directory instead of an `Arrays` to open the memory maps lazily inside each
    DataLoader worker (pickling an open memmap would copy the whole array to every worker).
    """

    def __init__(self, arrays: Arrays | Path, issue_idx: np.ndarray):
        self._a = arrays if isinstance(arrays, Arrays) else None
        self._path = None if isinstance(arrays, Arrays) else Path(arrays)
        self.idx = np.asarray(issue_idx)

    @property
    def a(self) -> Arrays:
        if self._a is None:
            self._a = Arrays.load(self._path)
        return self._a

    def __getstate__(self):
        state = self.__dict__.copy()
        if self._path is not None:
            state["_a"] = None
        return state

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, i: int):
        t = int(self.idx[i])
        return (
            grid_inputs(self.a, t),
            np.asarray(self.a.weather[t], dtype=np.float32),
            targets(self.a, t),
            quiet_cells(self.a, t),
            t,
        )
