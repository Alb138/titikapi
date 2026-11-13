"""Weather-derived predictors, computed at the weather's native 0.25 deg resolution.

Every feature on day t uses data from day t or earlier only (tested in test_weather_features).
"""

from __future__ import annotations

import numpy as np
import xarray as xr

RAIN_WINDOWS = (7, 30, 60, 90)
FWI_STATE_VARS = ("ffmc", "dmc", "dc")


def relative_humidity(t_c, td_c):
    """Relative humidity (%) from temperature and dewpoint in degC (Magnus formula)."""

    def es(t):
        return 6.112 * np.exp(17.62 * t / (243.12 + t))

    return np.clip(100.0 * es(td_c) / es(t_c), 0.0, 100.0)


def days_since_rain(pr: xr.DataArray, threshold: float = 1.0, cap: int = 90) -> xr.DataArray:
    """Consecutive days (up to and including today) with less than `threshold` mm of rain."""
    pr = pr.transpose("time", ...)
    values = pr.values
    out = np.zeros(values.shape, dtype=np.int16)
    run = np.zeros(values.shape[1:], dtype=np.int16)
    for i in range(values.shape[0]):
        run = np.where(values[i] < threshold, np.minimum(run + 1, cap), 0).astype(np.int16)
        out[i] = run
    return xr.DataArray(out, dims=pr.dims, coords=pr.coords, name="dry_days")


def rain_sums(pr: xr.DataArray, windows=RAIN_WINDOWS) -> xr.Dataset:
    """Trailing rain totals (mm) over the last `w` days, including today."""
    return xr.Dataset({f"rain_{w}d": pr.rolling(time=w, min_periods=1).sum() for w in windows})


def fire_weather_codes(
    tas_c: xr.DataArray,
    pr_mm: xr.DataArray,
    wind_kmh: xr.DataArray,
    hurs: xr.DataArray,
    state: dict[str, xr.DataArray] | None = None,
) -> xr.Dataset:
    """Canadian Fire Weather Index system codes (the basis of Indonesia's fire danger rating).

    Tropical setting: no fire-season start/stop and no overwintering. Pass `state` (the previous
    day's ffmc/dmc/dc) to continue a run, e.g. in the live pipeline.
    """
    from xclim.compute.fire import cffwis_indices

    tas = tas_c.assign_attrs(units="degC")
    pr = pr_mm.assign_attrs(units="mm/d")
    wind = wind_kmh.assign_attrs(units="km/h")
    rh = hurs.assign_attrs(units="%")
    kwargs = {}
    if state is not None:
        kwargs = dict(
            ffmc0=state["ffmc"].drop_vars("time", errors="ignore"),
            dmc0=state["dmc"].drop_vars("time", errors="ignore"),
            dc0=state["dc"].drop_vars("time", errors="ignore"),
            initial_start_up=False,
        )
    lat = tas["lat"].assign_attrs(units="degrees_north")
    dc, dmc, ffmc, isi, bui, fwi = cffwis_indices(
        tas, pr, wind, rh, lat=lat, season_method=None, overwintering=False, **kwargs
    )
    out = xr.Dataset({"ffmc": ffmc, "dmc": dmc, "dc": dc, "isi": isi, "bui": bui, "fwi": fwi})
    return out.transpose("time", ...)


HISTORY_DAYS = max(RAIN_WINDOWS) + 1  # raw days to carry over between chunks


def weather_features(
    daily: xr.Dataset,
    state: dict[str, xr.DataArray] | None = None,
    history: xr.Dataset | None = None,
) -> xr.Dataset:
    """Daily ERA5/IFS fields (t2m, d2m degC; u10, v10 m/s; pr mm) -> model predictors.

    To process a long record in chunks (or day by day in the live pipeline), pass the previous
    chunk's last `HISTORY_DAYS` raw days as `history` and its final ffmc/dmc/dc as `state`.
    The result then equals processing everything at once.
    """
    rh = relative_humidity(daily["t2m"], daily["d2m"])
    wind_kmh = np.hypot(daily["u10"], daily["v10"]) * 3.6

    pr_ext = daily["pr"] if history is None else xr.concat([history["pr"], daily["pr"]], "time")
    windowed = rain_sums(pr_ext)
    windowed["dry_days"] = days_since_rain(pr_ext)
    windowed = windowed.sel(time=daily["time"])

    feats = xr.Dataset(
        {
            "t2m": daily["t2m"],
            "rh": rh,
            "wind": wind_kmh,
            "u10": daily["u10"],
            "v10": daily["v10"],
            "pr": daily["pr"],
        }
    )
    feats = feats.merge(windowed)
    feats = feats.merge(fire_weather_codes(daily["t2m"], daily["pr"], wind_kmh, rh, state=state))
    return feats.transpose("time", "lat", "lon").astype(np.float32)


def build_weather_cube(years: range, out) -> None:
    """ERA5 years -> data/cube/weather.zarr at 0.25 deg, one year at a time (low memory)."""
    from titikapi.data import era5

    state, history = None, None
    for i, year in enumerate(years):
        daily = era5.load_year(year)
        feats = weather_features(daily, state=state, history=history)
        if i == 0:
            encoding = {v: {"chunks": (64, feats.sizes["lat"], feats.sizes["lon"])} for v in feats}
            feats.to_zarr(out, mode="w", consolidated=False, encoding=encoding)
        else:
            feats.to_zarr(out, append_dim="time", consolidated=False)
        state = {k: feats[k].isel(time=-1) for k in FWI_STATE_VARS}
        history = daily.isel(time=slice(-HISTORY_DAYS, None))
        print(f"{year}: {feats.sizes['time']} days, mean FWI {float(feats['fwi'].mean()):.1f}")
