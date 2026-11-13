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


def main() -> None:
    parser = argparse.ArgumentParser(prog="titikapi")
    sub = parser.add_subparsers(dest="command", required=True)

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
