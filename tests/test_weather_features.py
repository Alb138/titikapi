import numpy as np
import pandas as pd
import pytest
import xarray as xr

from titikapi.data.era5 import daily_precip_noon_to_noon
from titikapi.data.features import (
    days_since_rain,
    fire_weather_codes,
    rain_sums,
    relative_humidity,
)


def _da(values, days):
    values = np.asarray(values, dtype=float)
    return xr.DataArray(
        values[:, None, None] * np.ones((1, 1, 2)),
        dims=("time", "lat", "lon"),
        coords={
            "time": pd.date_range("2019-08-01", periods=days),
            "lat": [0.0],
            "lon": [100.0, 100.25],
        },
    )


def test_relative_humidity():
    assert relative_humidity(30.0, 30.0) == pytest.approx(100.0)
    assert relative_humidity(30.0, 20.0) == pytest.approx(55.0, abs=0.5)
    assert relative_humidity(30.0, 35.0) == pytest.approx(100.0)  # clipped


def test_precip_window_is_previous_24h_ending_06utc():
    times = pd.date_range("2019-09-01 00:00", "2019-09-03 23:00", freq="h")
    tp = np.zeros(len(times))
    tp[times.get_loc(pd.Timestamp("2019-09-02 06:00"))] = 0.001  # belongs to Sep 2
    tp[times.get_loc(pd.Timestamp("2019-09-02 07:00"))] = 0.002  # belongs to Sep 3
    da = xr.DataArray(tp, dims="valid_time", coords={"valid_time": times})
    daily = daily_precip_noon_to_noon(da)  # mm
    assert float(daily.sel(time="2019-09-02")) == pytest.approx(1.0)
    assert float(daily.sel(time="2019-09-03")) == pytest.approx(2.0)


def test_days_since_rain():
    pr = _da([5, 0, 0, 2, 0.5, 0], 6)
    out = days_since_rain(pr, threshold=1.0, cap=90)
    assert out.isel(lat=0, lon=0).values.tolist() == [0, 1, 2, 0, 1, 2]


def test_rain_sums_do_not_look_ahead():
    """Features on day t may only use rain up to and including day t."""
    base = np.arange(40, dtype=float)
    future_changed = base.copy()
    future_changed[30:] = 999.0
    a = rain_sums(_da(base, 40), windows=(7, 30))
    b = rain_sums(_da(future_changed, 40), windows=(7, 30))
    xr.testing.assert_equal(a.isel(time=slice(0, 30)), b.isel(time=slice(0, 30)))
    assert float(a["rain_7d"].isel(time=6, lat=0, lon=0)) == pytest.approx(sum(range(7)))


def test_fire_weather_codes_respond_to_drought_and_rain():
    days = 60
    tas = _da(np.full(days, 33.0), days)
    hurs = _da(np.full(days, 45.0), days)
    wind = _da(np.full(days, 12.0), days)
    dry = fire_weather_codes(tas, _da(np.zeros(days), days), wind, hurs)
    wet = fire_weather_codes(tas, _da(np.full(days, 20.0), days), wind, hurs)
    dc = dry["dc"].isel(lat=0, lon=0).values
    assert np.all(np.diff(dc) > 0)  # drought code keeps rising without rain
    last_dry = float(dry["fwi"].isel(time=-1, lat=0, lon=0))
    last_wet = float(wet["fwi"].isel(time=-1, lat=0, lon=0))
    assert last_dry > last_wet
    assert set(dry.data_vars) == {"ffmc", "dmc", "dc", "isi", "bui", "fwi"}


def test_fire_weather_codes_can_continue_from_state():
    """Running 60 days at once == running 40 days, then 20 more from the saved state."""
    days = 60
    rng = np.random.default_rng(0)
    tas = _da(rng.uniform(26, 34, days), days)
    pr = _da(rng.choice([0.0, 0.0, 0.0, 8.0], days), days)
    wind = _da(rng.uniform(5, 20, days), days)
    hurs = _da(rng.uniform(40, 90, days), days)
    full = fire_weather_codes(tas, pr, wind, hurs)
    first = fire_weather_codes(*(x.isel(time=slice(0, 40)) for x in (tas, pr, wind, hurs)))
    state = {k: first[k].isel(time=-1) for k in ("ffmc", "dmc", "dc")}
    rest = fire_weather_codes(
        *(x.isel(time=slice(40, None)) for x in (tas, pr, wind, hurs)), state=state
    )
    np.testing.assert_allclose(rest["dc"].values, full["dc"].isel(time=slice(40, None)).values)
    np.testing.assert_allclose(rest["fwi"].values, full["fwi"].isel(time=slice(40, None)).values)


def test_chunked_feature_build_equals_all_at_once():
    from titikapi.data.features import FWI_STATE_VARS, HISTORY_DAYS, weather_features

    days = 200
    rng = np.random.default_rng(1)
    daily = xr.Dataset(
        {
            "t2m": _da(rng.uniform(26, 34, days), days),
            "d2m": _da(rng.uniform(18, 25, days), days),
            "u10": _da(rng.uniform(-5, 5, days), days),
            "v10": _da(rng.uniform(-5, 5, days), days),
            "pr": _da(rng.choice([0.0, 0.0, 0.0, 12.0], days), days),
        }
    )
    full = weather_features(daily)
    a = weather_features(daily.isel(time=slice(0, 120)))
    b = weather_features(
        daily.isel(time=slice(120, None)),
        state={k: a[k].isel(time=-1) for k in FWI_STATE_VARS},
        history=daily.isel(time=slice(0, 120)).isel(time=slice(-HISTORY_DAYS, None)),
    )
    xr.testing.assert_allclose(xr.concat([a, b], "time"), full, rtol=1e-5)


def test_year_chunks_respect_cost_limit_and_gaps():
    from titikapi.data.era5 import year_chunks

    assert year_chunks([2011, 2012, 2013, 2014, 2015], cost_per_year=52_704, limit=121_000) == [
        [2011, 2012],
        [2013, 2014],
        [2015],
    ]
    # a gap (2020-2022 skipped) always starts a new request
    assert year_chunks([2018, 2019, 2023, 2024], cost_per_year=8_784, limit=121_000) == [
        [2018, 2019],
        [2023, 2024],
    ]


def test_plan_reuse_keeps_at_most_limit_queued_and_deletes_the_rest():
    from titikapi.data.era5 import plan_reuse

    found = [  # newest first
        ("new_tp", "accepted", "tp", [2019, 2020]),
        ("done", "successful", "inst06", [2024, 2025]),
        ("dup", "accepted", "tp", [2013, 2014]),
        ("mid", "accepted", "tp", [2013, 2014]),
        ("have", "accepted", "tp", [2011, 2012]),  # already downloaded
        ("old_inst", "accepted", "inst06", [2026]),
        ("run", "running", "tp", [2015, 2016]),
    ]
    missing = {"inst06": {2024, 2025, 2026}, "tp": set(range(2013, 2021))}
    claim, surplus = plan_reuse(found, missing, limit=3)
    assert [c[0] for c in claim] == ["done", "run", "old_inst", "mid"]
    assert sum(c[3] for c in claim) == 3  # queued slots used
    assert sorted(surplus) == ["dup", "have", "new_tp"]
