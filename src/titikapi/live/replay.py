"""Replay backtest: the live pipeline run day by day over a past period, using only the inputs the
live system has (NOAA-20 fires, ECMWF IFS weather), scored against NOAA-20 detections.

The model was tested with S-NPP fires and ERA5 reanalysis weather. Live, both inputs differ. Three
scenarios on the same issue days separate the effects:
  evaluation   S-NPP fires + ERA5 weather, scored on S-NPP (the reported test results)
  noaa20_era5  NOAA-20 fires + ERA5 weather, scored on NOAA-20 (sensor change only)
  live         NOAA-20 fires + IFS weather, scored on NOAA-20 (exactly what is deployed)
The IFS weather state is seeded from ERA5 a month before the first issue day and then rolled
forward with IFS only, so the slow drought codes are IFS-driven by the time scoring starts.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from titikapi.config import CUBE_DIR, LEAD_DAYS
from titikapi.live.bundle import FINAL, Bundle
from titikapi.live.forecast import QUIET_DAYS, forecast, live_arrays
from titikapi.live.state import (
    FIRE_WINDOW,
    FireState,
    WeatherState,
    bootstrap_weather,
    fetch_fire_days,
    update_weather,
)
from titikapi.models.metrics import average_precision, brier, topk_coverage

SPINUP_DAYS = 31
EDGES = np.array([0, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0])  # reliability bins


def noaa20_history(start, end, b: Bundle, have: FireState | None) -> FireState:
    """NOAA-20 fire state covering [start, end]: the saved live state where it reaches, the
    FIRMS API for anything earlier."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if have is not None and have.dates[0] <= start:
        return have
    stop = have.dates[0] - pd.Timedelta(days=1) if have is not None else end
    early = fetch_fire_days(start, stop, b.mask, b.static_sources)
    if have is None:
        return early
    return FireState(
        early.dates.append(have.dates),
        np.concatenate([early.count, have.count]),
        np.concatenate([early.frp_log, have.frp_log]),
    )


def _window(fire: FireState, day: pd.Timestamp) -> FireState:
    i = int(fire.dates.get_loc(day))
    s = max(0, i - FIRE_WINDOW + 1)
    return FireState(fire.dates[s : i + 1], fire.count[s : i + 1], fire.frp_log[s : i + 1])


def _score(name: str, p: np.ndarray, y: np.ndarray, quiet: np.ndarray) -> list[dict]:
    """p, y: (D, 7, N); quiet: (D, N) new-ignition cells."""
    rows = []
    for lead in range(1, LEAD_DAYS + 1):
        pl, yl = p[:, lead - 1], y[:, lead - 1]
        rows.append(
            {
                "scenario": name,
                "lead": lead,
                "days": len(p),
                "base_rate": float(yl.mean()),
                "ap": average_precision(yl, pl),
                "new_ap": average_precision(yl[quiet], pl[quiet]),
                "top5": topk_coverage(yl, pl),
                "new_top5": topk_coverage(yl, pl, count=quiet),
                "brier": brier(yl, pl),
            }
        )
    return rows


def reliability(p: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    p, y = p.ravel().astype(np.float64), y.ravel()
    k = np.digitize(p, EDGES[1:-1])
    return pd.DataFrame(
        [
            {
                "bin": f"{EDGES[i]:g}-{EDGES[i + 1]:g}",
                "n": int((k == i).sum()),
                "mean_forecast": float(p[k == i].mean()) if (k == i).any() else np.nan,
                "observed": float(y[k == i].mean()) if (k == i).any() else np.nan,
            }
            for i in range(len(EDGES) - 1)
        ]
    )


def run(start, end, b: Bundle, work: Path, live_fire: FireState | None):
    from titikapi.data import enso
    from titikapi.models.dataset import ARRAYS_DIR, Arrays
    from titikapi.models.train import MODELS_DIR

    days = pd.date_range(start, end, freq="D")
    fire = noaa20_history(days[0] - pd.Timedelta(days=FIRE_WINDOW), days[-1], b, live_fire)
    last_target = days[-1] + pd.Timedelta(days=LEAD_DAYS)
    assert fire.dates[-1] >= last_target, f"NOAA-20 state ends {fire.dates[-1].date()}"

    ws = bootstrap_weather((days[0] - pd.Timedelta(days=SPINUP_DAYS)).date())
    ws = update_weather(
        ws,
        days[-1],
        b.weather_lat,
        b.weather_lon,
        work / "ifs",
        keep_feats=len(days) + SPINUP_DAYS + 1,
    )
    era = xr.open_zarr(CUBE_DIR / "weather.zarr", consolidated=False).sel(time=days).load()
    oni = np.nan_to_num(enso.oni_known_at(days, enso.load_oni()))

    p = {"live": [], "noaa20_era5": [], "persistence_noaa20": []}
    parts = {"ens_unet": [], "lgbm": []}
    for i, day in enumerate(days):
        fw = _window(fire, day)
        for scen, feats in (("live", ws.feats), ("noaa20_era5", era)):
            w = WeatherState(raw=None, fwi={}, feats=feats.sel(time=slice(None, day)))
            out = forecast(b, live_arrays(b, fw, w, float(oni[i])))
            p[scen].append(out[FINAL])
            if scen == "live":
                p["persistence_noaa20"].append(out["persistence"])
                for k in parts:
                    parts[k].append(out[k])
        if i % 10 == 0:
            print(f"replay {day.date()}", flush=True)

    pos = {d: i for i, d in enumerate(fire.dates)}
    y = np.stack(
        [
            [fire.count[pos[d + pd.Timedelta(days=k)]] > 0 for k in range(1, LEAD_DAYS + 1)]
            for d in days
        ]
    )
    quiet = np.stack([(_window(fire, d).count[-QUIET_DAYS:] == 0).all(axis=0) for d in days])
    rows = []
    for name, arr in p.items():
        rows += _score(name, np.stack(arr), y, quiet)
    for name, arr in parts.items():
        rows += _score(f"live_{name}", np.stack(arr), y, quiet)

    # the reported evaluation (S-NPP in and out, ERA5) on the same issue days
    a = Arrays.load(ARRAYS_DIR)
    idx = np.load(MODELS_DIR / FINAL / "idx_test.npy")
    probs = np.load(MODELS_DIR / FINAL / "probs_test.npy", mmap_mode="r")
    sel = np.flatnonzero(np.isin(a.time[idx], days))
    t = idx[sel]
    ys = np.stack([[np.asarray(a.fire_on[tt + k]) for k in range(1, LEAD_DAYS + 1)] for tt in t])
    qs = np.asarray(a.days_since[t]) >= QUIET_DAYS
    rows += _score("evaluation", np.asarray(probs[sel], np.float32), ys, qs)

    rel = pd.concat(
        [
            reliability(np.stack(p["live"])[:, d - 1], y[:, d - 1]).assign(lead=d)
            for d in (1, LEAD_DAYS)
        ]
    )
    return pd.DataFrame(rows), rel
