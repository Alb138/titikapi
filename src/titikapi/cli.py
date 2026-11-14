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
    from titikapi.data import enso, land
    from titikapi.data.static import province_raster

    grid = Grid()
    prov = province_raster(grid)
    dims = ("lat", "lon")
    ds = xr.Dataset(
        {"province_id": (dims, prov), "mask": (dims, prov > 0)},
        coords=grid.coords(),
        attrs={"province_names": list(PROVINCES)},
    )
    print(f"study region: {int(ds['mask'].sum()):,} land cells of {np.prod(grid.shape):,}")

    if not args.skip_land:
        print("peat (PEATMAP) ...", flush=True)
        ds["peat_frac"] = (dims, land.peat_fraction(grid))
        print("elevation (Copernicus DEM, remote) ...", flush=True)
        elev_mean, elev_std = land.elevation_layers(grid)
        ds["elev_mean"] = (dims, elev_mean)
        ds["elev_std"] = (dims, elev_std)
        print("forest (Hansen GFC) ...", flush=True)
        cover, loss = land.forest_layers(grid)
        ds["treecover2000"] = (dims, cover)
        years = np.arange(land.HANSEN_FIRST_LOSS_YEAR, land.HANSEN_FIRST_LOSS_YEAR + len(loss))
        ds["forest_loss"] = (("loss_year", *dims), loss)
        ds = ds.assign_coords(loss_year=years)

    CUBE_DIR.mkdir(parents=True, exist_ok=True)
    ds.to_zarr(CUBE_DIR / "static.zarr", mode="w", consolidated=False)
    enso.load_oni().to_csv(CUBE_DIR / "oni.csv")
    print(f"wrote {CUBE_DIR / 'static.zarr'} with {list(ds.data_vars)}")


def download_era5(args: argparse.Namespace) -> None:
    import datetime as dt

    from titikapi.data import era5

    last_day = dt.date.today() - dt.timedelta(days=era5.ERA5_LATENCY_DAYS)
    years = [y for y in range(args.start_year, args.end_year + 1) if y not in args.skip]
    failed = 0
    for msg in era5.download_years(
        years, last_day=last_day, workers=args.workers, plan_only=args.plan
    ):
        print(msg, flush=True)
        failed += msg.startswith("FAILED")
    if failed:
        raise SystemExit(f"{failed} request(s) failed; re-run to retry (finished files are kept)")


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


def build_arrays(args: argparse.Namespace) -> None:
    from titikapi.models.dataset import ARRAYS_DIR
    from titikapi.models.dataset import build_arrays as build

    build()
    print(f"wrote {ARRAYS_DIR}")


def _issue_days(split: str):
    from titikapi.config import SPLITS
    from titikapi.models.dataset import ARRAYS_DIR, Arrays

    a = Arrays.load(ARRAYS_DIR)
    start, end = (np.datetime64(d) for d in SPLITS[split])
    t = np.arange(len(a.time))
    ok = (a.time.values >= start) & (a.time.values + np.timedelta64(7, "D") <= end) & (t >= 13)
    return t[ok & (t + 7 < len(a.time))]


def train_model(args: argparse.Namespace) -> None:
    from titikapi.models.train import TrainConfig, train

    cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, base=args.base)
    train(_issue_days("train"), _issue_days("val"), cfg, name=args.name)


def evaluate_model(args: argparse.Namespace) -> None:
    import json

    import pandas as pd
    import torch

    from titikapi.config import ROOT
    from titikapi.data.cube import load_fire_matrix, recent_fire
    from titikapi.models.dataset import ARRAYS_DIR, Arrays
    from titikapi.models.evaluate import evaluate
    from titikapi.models.train import MODELS_DIR, PlattCalibrator, TrainConfig, build_model
    from titikapi.models.train import predict_logits as predict

    run = MODELS_DIR / args.name
    cfg = TrainConfig(**json.loads((run / "config.json").read_text()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a = Arrays.load(ARRAYS_DIR)
    model = build_model(a, cfg).to(device)
    model.load_state_dict(torch.load(run / "best.pt", weights_only=True, map_location=device))

    val_idx = _issue_days("val")
    val_logits = predict(model, ARRAYS_DIR, val_idx, cfg, device)
    y_val = np.stack([a.fire_on[val_idx + d] for d in range(1, 8)], axis=1)
    cal = PlattCalibrator().fit(val_logits, y_val)
    cal.save(run / "calibration.json")

    fm = load_fire_matrix()
    rec = recent_fire(fm.on)
    assert len(fm.time) == len(a.time), "fire cube and arrays are out of sync; rebuild arrays"
    for split in args.splits:
        idx = val_idx if split == "val" else _issue_days(split)
        logits = val_logits if split == "val" else predict(model, ARRAYS_DIR, idx, cfg, device)
        probs = np.stack([cal.transform(logits[:, d], d + 1) for d in range(7)], axis=1)
        np.save(run / f"probs_{split}.npy", probs.astype(np.float16))
        np.save(run / f"idx_{split}.npy", idx)

        def lookup(issue_idx, lead, probs=probs, idx=idx):
            assert np.array_equal(issue_idx, idx)
            return probs[:, lead - 1]

        res = evaluate(args.name, lookup, fm, rec, idx)
        res.insert(0, "split", split)
        res.to_csv(ROOT / "reports" / f"{args.name}_{split}.csv", index=False)
        base = pd.read_csv(ROOT / "reports" / f"baselines_{split}.csv")
        both = pd.concat([base, res])
        table = both.pivot_table(
            index=["slice", "model"], columns="lead", values=["ap", "top5_coverage", "brier"]
        )
        print(f"\n== {split}")
        with pd.option_context("display.width", 180, "display.precision", 4):
            cols = [("ap", 1), ("ap", 7), ("top5_coverage", 1), ("top5_coverage", 7)]
            print(table[cols + [("brier", 1)]])


def main() -> None:
    import datetime as dt

    parser = argparse.ArgumentParser(prog="titikapi")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("build-arrays", help="cubes -> memory-mappable model inputs")
    p.set_defaults(func=build_arrays)

    p = sub.add_parser("train", help="train the U-Net -> models/<name>/best.pt")
    p.add_argument("--name", default="unet")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--base", type=int, default=32)
    p.set_defaults(func=train_model)

    p = sub.add_parser("evaluate-model", help="calibrate on val, score val/test vs baselines")
    p.add_argument("--name", default="unet")
    p.add_argument("--splits", nargs="+", default=["val", "test"])
    p.set_defaults(func=evaluate_model)

    p = sub.add_parser("evaluate-baselines", help="climatology & persistence -> reports/*.csv")
    p.add_argument("--splits", nargs="+", default=["val", "test"])
    p.set_defaults(func=evaluate_baselines)

    # 2011 is a spin-up year: the drought code needs months of history to settle.
    p = sub.add_parser("download-era5", help="ERA5 from Copernicus CDS -> data/raw/era5")
    p.add_argument("--start-year", type=int, default=2011)
    p.add_argument("--end-year", type=int, default=dt.date.today().year)
    p.add_argument("--workers", type=int, default=3, help="CDS allows ~3 queued requests")
    p.add_argument("--skip", type=int, nargs="*", default=[], help="years to leave out")
    p.add_argument("--plan", action="store_true", help="show the requests, submit nothing")
    p.set_defaults(func=download_era5)

    p = sub.add_parser("build-weather", help="ERA5 -> weather features -> data/cube/weather.zarr")
    p.add_argument("--start-year", type=int, default=2011)
    p.add_argument("--end-year", type=int, default=dt.date.today().year)
    p.set_defaults(func=build_weather)

    p = sub.add_parser("build-fire", help="FIRMS archives -> data/cube/fire.zarr")
    p.add_argument("--start-year", type=int, default=FIRST_DAY.year)
    p.add_argument("--end-year", type=int, default=2025)
    p.set_defaults(func=build_fire)

    p = sub.add_parser(
        "build-static", help="provinces, mask, peat, elevation, forest -> data/cube/static.zarr"
    )
    p.add_argument("--skip-land", action="store_true", help="only provinces + mask")
    p.set_defaults(func=build_static)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
