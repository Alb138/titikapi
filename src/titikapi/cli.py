"""Command-line entry point: `uv run titikapi <command>`."""

from __future__ import annotations

import argparse

import numpy as np
import xarray as xr

from titikapi.config import CUBE_DIR, FIRST_DAY, PROVINCES
from titikapi.grid import Grid


def build_fire(args: argparse.Namespace) -> None:
    from titikapi.data.firms import build_fire_cube

    CUBE_DIR.mkdir(parents=True, exist_ok=True)
    years = range(args.start_year, args.end_year + 1)
    out = build_fire_cube(years, CUBE_DIR / "fire.zarr", Grid(), first_day=FIRST_DAY)
    print(f"wrote {out}")


def build_static(args: argparse.Namespace) -> None:
    from titikapi.data.static import province_raster

    grid = Grid()
    prov = province_raster(grid)
    ds = xr.Dataset(
        {
            "province_id": (("lat", "lon"), prov),
            "mask": (("lat", "lon"), prov > 0),
        },
        coords=grid.coords(),
        attrs={"province_names": list(PROVINCES)},
    )
    CUBE_DIR.mkdir(parents=True, exist_ok=True)
    ds.to_zarr(CUBE_DIR / "static.zarr", mode="w", consolidated=False)
    print(f"study region: {int(ds['mask'].sum()):,} land cells of {np.prod(grid.shape):,}")


def download_era5(args: argparse.Namespace) -> None:
    import datetime as dt
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from titikapi.data import era5

    last_day = dt.date.today() - dt.timedelta(days=era5.ERA5_LATENCY_DAYS)

    def fetch(year: int):
        partial = last_day if year == last_day.year else None
        return year, era5.download_year(year, last_day=partial)

    # CDS queues each request; a few in flight at once is much faster than one at a time.
    years = range(args.start_year, args.end_year + 1)
    failed = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch, y): y for y in years}
        for fut in as_completed(futures):
            try:
                year, (inst, tp) = fut.result()
                print(f"{year}: ok ({inst.name}, {tp.name})", flush=True)
            except Exception as e:
                failed.append(futures[fut])
                print(f"{futures[fut]}: FAILED: {e}", flush=True)
    if failed:
        raise SystemExit(f"failed years: {sorted(failed)}; re-run to retry (done files are kept)")


def build_weather(args: argparse.Namespace) -> None:
    from titikapi.data.features import build_weather_cube

    CUBE_DIR.mkdir(parents=True, exist_ok=True)
    years = range(args.start_year, args.end_year + 1)
    build_weather_cube(years, CUBE_DIR / "weather.zarr")


def evaluate_baselines(args: argparse.Namespace) -> None:
    import pandas as pd

    from titikapi.config import ROOT, SPLITS
    from titikapi.data.cube import load_fire_matrix, recent_fire
    from titikapi.models.baselines import Climatology, Persistence
    from titikapi.models.evaluate import evaluate, issue_days

    fm = load_fire_matrix()
    rec = recent_fire(fm.on)
    start, end = (pd.Timestamp(d) for d in SPLITS["train"])
    train_days = np.flatnonzero((fm.time >= start) & (fm.time <= end))
    clim = Climatology().fit(fm, train_days)
    pers = Persistence().fit(fm, rec, issue_days(fm, SPLITS["train"]), climatology=clim)

    out_dir = ROOT / "reports"
    out_dir.mkdir(exist_ok=True)
    for split in args.splits:
        idx = issue_days(fm, SPLITS[split])
        res = pd.concat(
            [
                evaluate("climatology", clim.predict, fm, rec, idx),
                evaluate("persistence", pers.predict, fm, rec, idx),
            ]
        )
        res.insert(0, "split", split)
        res.to_csv(out_dir / f"baselines_{split}.csv", index=False)
        period = f"{fm.time[idx[0]].date()} .. {fm.time[idx[-1]].date()}"
        print(f"\n== {split}: {len(idx)} issue days, {period}")
        table = res.pivot_table(
            index=["slice", "model"], columns="lead", values=["ap", "top5_coverage"]
        )
        with pd.option_context("display.width", 160, "display.precision", 3):
            print(table[[("ap", 1), ("ap", 7), ("top5_coverage", 1), ("top5_coverage", 7)]])


def main() -> None:
    import datetime as dt

    parser = argparse.ArgumentParser(prog="titikapi")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("evaluate-baselines", help="climatology & persistence -> reports/*.csv")
    p.add_argument("--splits", nargs="+", default=["val", "test"])
    p.set_defaults(func=evaluate_baselines)

    # 2011 is a spin-up year: the drought code needs months of history to settle.
    p = sub.add_parser("download-era5", help="ERA5 from Copernicus CDS -> data/raw/era5")
    p.add_argument("--start-year", type=int, default=2011)
    p.add_argument("--end-year", type=int, default=dt.date.today().year)
    p.add_argument("--workers", type=int, default=3, help="CDS allows ~3 queued requests")
    p.set_defaults(func=download_era5)

    p = sub.add_parser("build-weather", help="ERA5 -> weather features -> data/cube/weather.zarr")
    p.add_argument("--start-year", type=int, default=2011)
    p.add_argument("--end-year", type=int, default=dt.date.today().year)
    p.set_defaults(func=build_weather)

    p = sub.add_parser("build-fire", help="FIRMS archives -> data/cube/fire.zarr")
    p.add_argument("--start-year", type=int, default=FIRST_DAY.year)
    p.add_argument("--end-year", type=int, default=2025)
    p.set_defaults(func=build_fire)

    p = sub.add_parser("build-static", help="province raster + land mask -> data/cube/static.zarr")
    p.set_defaults(func=build_static)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
