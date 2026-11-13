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

    from titikapi.data import era5

    last_day = dt.date.today() - dt.timedelta(days=era5.ERA5_LATENCY_DAYS)
    for year in range(args.start_year, args.end_year + 1):
        partial = last_day if year == last_day.year else None
        inst, tp = era5.download_year(year, last_day=partial)
        print(f"{year}: {inst.name}, {tp.name}", flush=True)


def build_weather(args: argparse.Namespace) -> None:
    from titikapi.data.features import build_weather_cube

    CUBE_DIR.mkdir(parents=True, exist_ok=True)
    years = range(args.start_year, args.end_year + 1)
    build_weather_cube(years, CUBE_DIR / "weather.zarr")


def main() -> None:
    import datetime as dt

    parser = argparse.ArgumentParser(prog="titikapi")
    sub = parser.add_subparsers(dest="command", required=True)

    # 2011 is a spin-up year: the drought code needs months of history to settle.
    p = sub.add_parser("download-era5", help="ERA5 from Copernicus CDS -> data/raw/era5")
    p.add_argument("--start-year", type=int, default=2011)
    p.add_argument("--end-year", type=int, default=dt.date.today().year)
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
