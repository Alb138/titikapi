"""ERA5 reanalysis (Copernicus CDS) -> daily weather at native 0.25 deg.

Needs a free CDS account with the ERA5 licence accepted and an API token in ~/.cdsapirc:
    url: https://cds.climate.copernicus.eu/api
    key: <your-personal-access-token>

Daily convention (matches the Fire Weather Index system and the live IFS forecasts):
- temperature, dewpoint and wind are taken at 06 UTC (13:00 WIB, early afternoon locally)
- precipitation is the 24 h total ending at 06 UTC
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from titikapi.config import RAW_DIR

ERA5_DIR = RAW_DIR / "era5"
DATASET = "reanalysis-era5-single-levels"
AREA = [6.75, 94.75, -6.75, 119.75]  # N, W, S, E: the grid box plus one 0.25 deg cell of padding
INST_VARS = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
]
OBS_HOUR_UTC = 6


def _request(year: int, variables: list[str], times: list[str]) -> dict:
    return {
        "product_type": ["reanalysis"],
        "variable": variables,
        "year": [str(year)],
        "month": [f"{m:02d}" for m in range(1, 13)],
        "day": [f"{d:02d}" for d in range(1, 32)],
        "time": times,
        "area": AREA,
        "data_format": "netcdf",
        "download_format": "unarchived",
    }


HOURS = [f"{h:02d}:00" for h in range(24)]
ERA5_LATENCY_DAYS = 6  # ERA5T appears about 5 days behind real time
# CDS allows only a few queued requests per user and dataset; extra ones are rejected outright.
MAX_IN_FLIGHT = 3
RETRY_WAIT_S = 120
RETRY_HOURS = 12


def _retrieve(client, request: dict, target: Path) -> None:
    """client.retrieve, waiting and retrying while the per-user CDS queue is full."""
    import time

    tmp = target.with_name(target.name + ".part")
    deadline = time.monotonic() + RETRY_HOURS * 3600
    while True:
        try:
            client.retrieve(DATASET, request, str(tmp))
            tmp.replace(target)
            return
        except Exception as e:  # cdsapi raises plain HTTP errors
            queue_full = "Maximum number queued requests" in str(e)
            if not queue_full or time.monotonic() > deadline:
                raise
            time.sleep(RETRY_WAIT_S)


COST_LIMIT = 121_000  # CDS "size" cost limit per request for this dataset
# (file kind, variables, times). CDS cost = variables x times x days x 6.
KINDS = {
    "inst06": (INST_VARS, [f"{OBS_HOUR_UTC:02d}:00"]),
    "tp": (["total_precipitation"], HOURS),
}


def year_path(year: int, kind: str, dest_dir: Path = ERA5_DIR) -> Path:
    return dest_dir / f"era5_{year}_{kind}.nc"


def year_chunks(years: list[int], cost_per_year: int, limit: int = COST_LIMIT) -> list[list[int]]:
    """Group consecutive years into as few requests as fit under the CDS cost limit."""
    chunks: list[list[int]] = []
    for y in sorted(years):
        last = chunks[-1] if chunks else None
        fits = last is not None and (len(last) + 1) * cost_per_year <= limit
        if last is not None and last[-1] == y - 1 and fits:
            last.append(y)
        else:
            chunks.append([y])
    return chunks


def _merge_to(parts: list[Path], target: Path) -> None:
    opened = [xr.open_dataset(p) for p in parts]
    merged = xr.concat(opened, dim=_time_name(opened[0])).load() if len(opened) > 1 else None
    if merged is None:
        merged = opened[0].load()
    for ds in opened:
        ds.close()
    tmp = target.with_name(target.name + ".part")
    merged.to_netcdf(tmp)
    tmp.replace(target)


def _split_years(chunk_file: Path, kind: str, years: list[int], dest_dir: Path) -> None:
    with xr.open_dataset(chunk_file) as ds:
        tname = _time_name(ds)
        for y in years:
            part = ds.sel({tname: slice(f"{y}-01-01", f"{y}-12-31T23:59")}).load()
            tmp = year_path(y, kind, dest_dir).with_suffix(".nc.part")
            part.to_netcdf(tmp)
            tmp.replace(year_path(y, kind, dest_dir))


def _fetch_chunk(kind: str, years: list[int], dest_dir: Path) -> str:
    import cdsapi

    variables, times = KINDS[kind]
    req = _request(years[0], variables, times)
    req["year"] = [str(y) for y in years]
    chunk = dest_dir / f"era5_{years[0]}-{years[-1]}_{kind}.chunk.nc"
    if not chunk.exists():
        _retrieve(cdsapi.Client(quiet=True), req, chunk)
    _split_years(chunk, kind, years, dest_dir)
    chunk.unlink(missing_ok=True)
    return f"{kind} {years[0]}-{years[-1]}"


def _fetch_partial_year(kind: str, last_day, dest_dir: Path, include_current_month=False) -> str:
    """The current year's complete months in one request. The current month up to `last_day`
    needs a second request (days that don't exist yet can't be requested); it is skipped by
    default because the historical record only has to reach the end of the satellite fire
    record, and the live pipeline takes recent weather from ECMWF's forecasts instead."""
    import cdsapi

    variables, times = KINDS[kind]
    year = last_day.year
    blocks = []
    if last_day.month > 1:
        req = _request(year, variables, times)
        req["month"] = [f"{m:02d}" for m in range(1, last_day.month)]
        blocks.append(req)
    if include_current_month or not blocks:
        req = _request(year, variables, times)
        req["month"] = [f"{last_day.month:02d}"]
        req["day"] = [f"{d:02d}" for d in range(1, last_day.day + 1)]
        blocks.append(req)
    client = cdsapi.Client(quiet=True)
    parts = []
    for i, req in enumerate(blocks):
        part = dest_dir / f"era5_{year}_{kind}_block{i}.nc"
        if not part.exists():
            _retrieve(client, req, part)
        parts.append(part)
    _merge_to(parts, year_path(year, kind, dest_dir))
    for p in parts:
        p.unlink(missing_ok=True)
    return f"{kind} {year} ({len(blocks)} request(s), complete months to {last_day:%b})"


def _datastores_client():
    """Low-level CDS client (lists and re-attaches to jobs), using the cdsapi credentials."""
    from ecmwf.datastores import Client

    cfg = {}
    for line in (Path.home() / ".cdsapirc").read_text().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            cfg[k.strip()] = v.strip()
    return Client(url=cfg["url"], key=cfg["key"])


def _kind_of(request: dict) -> str | None:
    for kind, (variables, times) in KINDS.items():
        if sorted(request.get("variable", [])) == sorted(variables) and sorted(
            request.get("time", [])
        ) == sorted(times):
            return kind
    return None


def existing_jobs() -> list[tuple[str, str, list[int]]]:
    """Our full-year requests already queued, running or finished on CDS, e.g. from a run that
    was interrupted. Re-attaching keeps their place in the queue instead of starting over."""
    client = _datastores_client()
    found = []
    for job in client.get_jobs(limit=100).json.get("jobs", []):
        if job.get("status") not in ("accepted", "running", "successful"):
            continue
        req = client.get_remote(job["jobID"]).request
        kind = _kind_of(req)
        full_years = len(req.get("month", [])) == 12 and len(req.get("day", [])) == 31
        same_area = [float(x) for x in req.get("area", [])] == AREA
        if kind and full_years and same_area:
            found.append((job["jobID"], kind, sorted(int(y) for y in req["year"])))
    return found


def _attach(job_id: str, kind: str, years: list[int], dest_dir: Path) -> str:
    chunk = dest_dir / f"era5_{years[0]}-{years[-1]}_{kind}.chunk.nc"
    tmp = chunk.with_name(chunk.name + ".part")
    _datastores_client().get_remote(job_id).download(str(tmp))  # waits for the job to finish
    tmp.replace(chunk)
    _split_years(chunk, kind, years, dest_dir)
    chunk.unlink(missing_ok=True)
    return f"{kind} {years[0]}-{years[-1]} (re-attached to an earlier request)"


def download_years(
    years: list[int], last_day=None, dest_dir: Path = ERA5_DIR, workers: int = MAX_IN_FLIGHT
):
    """Download every missing (year, kind) file using as few CDS requests as possible.

    Full years are batched under the cost limit (13 years of 06 UTC fields, or 2 years of hourly
    precipitation, per request); the current year (if `last_day` is in it) is fetched up to
    `last_day`. Yields a message as each request finishes.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    dest_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    claimed = {kind: set() for kind in KINDS}
    for job_id, kind, job_years in existing_jobs():
        wanted = [y for y in job_years if y in years and not year_path(y, kind, dest_dir).exists()]
        if wanted == job_years and not claimed[kind] & set(job_years):
            jobs.append((_attach, job_id, kind, job_years, dest_dir))
            claimed[kind] |= set(job_years)
    for kind, (variables, times) in KINDS.items():
        missing = [
            y for y in years if not year_path(y, kind, dest_dir).exists() and y not in claimed[kind]
        ]
        full = [y for y in missing if last_day is None or y < last_day.year]
        cost = len(variables) * len(times) * 366 * 6
        jobs += [(_fetch_chunk, kind, chunk, dest_dir) for chunk in year_chunks(full, cost)]
        if last_day is not None and last_day.year in missing:
            jobs.append((_fetch_partial_year, kind, last_day, dest_dir))
    for fn, *args in jobs:
        yield "planned: " + fn.__name__.strip("_") + " " + " ".join(map(str, args[:-1]))
    # re-attached jobs only wait, so they get their own threads on top of `workers`
    with ThreadPoolExecutor(
        max_workers=workers + len(claimed["inst06"]) + len(claimed["tp"])
    ) as pool:
        futures = {pool.submit(fn, *args): (fn.__name__, args[:-1]) for fn, *args in jobs}
        for fut in as_completed(futures):
            name, what = futures[fut]
            try:
                yield f"ok: {fut.result()}"
            except Exception as e:
                yield f"FAILED: {name} {what}: {e}"


def _time_name(ds: xr.Dataset | xr.DataArray) -> str:
    return "valid_time" if "valid_time" in ds.dims else "time"


def daily_precip_noon_to_noon(tp_hourly: xr.DataArray) -> xr.DataArray:
    """Hourly ERA5 precipitation (m, accumulated over the hour ending at valid_time) -> mm/day.

    Day D gets the hours in (D-1 06:00, D 06:00] UTC.
    """
    tname = _time_name(tp_hourly)
    t = pd.DatetimeIndex(tp_hourly[tname].values)
    label = (t - pd.Timedelta(hours=OBS_HOUR_UTC + 1)).floor("D") + pd.Timedelta(days=1)
    da = tp_hourly.assign_coords({tname: label}).rename({tname: "time"})
    return (da.groupby("time").sum() * 1000.0).rename("pr")


def _keep_dim_coords(obj):
    return obj.drop_vars([v for v in obj.coords if v not in ("time", "lat", "lon")])


def load_year(year: int, dest_dir: Path = ERA5_DIR) -> xr.Dataset:
    """Daily dataset for one year at 0.25 deg: t2m, d2m (degC), u10, v10 (m/s), pr (mm).

    Precipitation for 1 January needs the evening of 31 December, so the previous year's
    precipitation file is used when it exists.
    """
    inst = xr.open_dataset(dest_dir / f"era5_{year}_inst06.nc")
    tname = _time_name(inst)
    inst = inst.assign_coords({tname: pd.DatetimeIndex(inst[tname].values).normalize()})
    inst = inst.rename({tname: "time", "latitude": "lat", "longitude": "lon"})

    tp = xr.open_dataset(dest_dir / f"era5_{year}_tp.nc")["tp"]
    prev_path = dest_dir / f"era5_{year - 1}_tp.nc"
    if prev_path.exists():
        prev = xr.open_dataset(prev_path)["tp"]
        ttp = _time_name(prev)
        prev = prev.sel({ttp: slice(f"{year - 1}-12-31", None)})
        tp = xr.concat([prev, tp], dim=ttp)
    pr = daily_precip_noon_to_noon(tp).rename({"latitude": "lat", "longitude": "lon"})

    ds = xr.Dataset(
        {
            "t2m": inst["t2m"] - 273.15,
            "d2m": inst["d2m"] - 273.15,
            "u10": inst["u10"],
            "v10": inst["v10"],
        }
    )
    ds = _keep_dim_coords(ds)
    ds["pr"] = _keep_dim_coords(pr).reindex(time=ds.time)
    return ds.astype(np.float32).load()
