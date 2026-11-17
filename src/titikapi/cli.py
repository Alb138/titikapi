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
    out = build_fire_cube(
        years, CUBE_DIR / "fire.zarr", Grid(), first_day=FIRST_DAY, last_day=args.last_day
    )
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

    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        base=args.base,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        patience=args.patience,
        crop=args.crop,
    )
    train(_issue_days("train"), _issue_days("val"), cfg, name=args.name)


def evaluate_model(args: argparse.Namespace) -> None:
    import json

    import torch

    from titikapi.models.dataset import ARRAYS_DIR, Arrays
    from titikapi.models.train import MODELS_DIR, TrainConfig, build_model
    from titikapi.models.train import predict_logits as predict

    run = MODELS_DIR / args.name
    cfg = TrainConfig(**json.loads((run / "config.json").read_text()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a = Arrays.load(ARRAYS_DIR)
    model = build_model(a, cfg).to(device)
    model.load_state_dict(torch.load(run / "best.pt", weights_only=True, map_location=device))
    _calibrate_and_report(
        args.name, a, args.splits, lambda idx: predict(model, ARRAYS_DIR, idx, cfg, device)
    )


def train_lgbm(args: argparse.Namespace) -> None:
    import pandas as pd

    from titikapi.config import ROOT
    from titikapi.models import lgbm
    from titikapi.models.dataset import ARRAYS_DIR, Arrays
    from titikapi.models.train import MODELS_DIR

    a = Arrays.load(ARRAYS_DIR)
    fb = lgbm.FeatureBuilder(a)
    boosters = lgbm.fit(fb, _issue_days("train"), _issue_days("val"), neg_frac=args.neg_frac)
    run = MODELS_DIR / args.name
    run.mkdir(parents=True, exist_ok=True)
    for d, b in enumerate(boosters, start=1):
        b.save_model(str(run / f"lead{d}.txt"), num_iteration=b.best_iteration)
    gain = {f"lead{d}": b.feature_importance("gain") for d, b in enumerate(boosters, start=1)}
    imp = pd.DataFrame(gain, index=lgbm.FEATURES)
    imp.div(imp.sum()).sort_values("lead1", ascending=False).to_csv(
        ROOT / "reports" / f"{args.name}_importance.csv"
    )
    _calibrate_and_report(
        args.name, a, args.splits, lambda idx: lgbm.predict_raw(boosters, fb, idx)
    )


def evaluate_ensemble(args: argparse.Namespace) -> None:
    """Average the members' calibrated probabilities, then recalibrate and score like a model."""
    from titikapi.models.dataset import ARRAYS_DIR, Arrays
    from titikapi.models.train import MODELS_DIR

    days = {s: _issue_days(s) for s in ("val", *args.splits)}

    def predict(idx):
        split = next(s for s, d in days.items() if np.array_equal(d, idx))
        total = None
        for m in args.members:
            run = MODELS_DIR / m
            assert np.array_equal(np.load(run / f"idx_{split}.npy"), idx), (
                f"{m}: saved {split} predictions are for different days; re-run its evaluation"
            )
            p = np.load(run / f"probs_{split}.npy").astype(np.float32)
            total = p if total is None else total + p
        p = np.clip(total / len(args.members), 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    (MODELS_DIR / args.name).mkdir(parents=True, exist_ok=True)
    (MODELS_DIR / args.name / "members.txt").write_text("\n".join(args.members))
    _calibrate_and_report(args.name, Arrays.load(ARRAYS_DIR), args.splits, predict)


def _calibrate_and_report(name: str, a, splits, predict) -> None:
    """Fit per-lead Platt scaling on validation logits, then score each split against the
    baselines. Saves probs/idx to models/<name>/ and reports/<name>_<split>.csv."""
    import pandas as pd

    from titikapi.config import ROOT
    from titikapi.data.cube import load_fire_matrix, recent_fire
    from titikapi.models.evaluate import evaluate
    from titikapi.models.train import MODELS_DIR, PlattCalibrator

    run = MODELS_DIR / name
    val_idx = _issue_days("val")
    val_logits = predict(val_idx)
    y_val = np.stack([a.fire_on[val_idx + d] for d in range(1, 8)], axis=1)
    cal = PlattCalibrator().fit(val_logits, y_val)
    cal.save(run / "calibration.json")

    fm = load_fire_matrix()
    rec = recent_fire(fm.on)
    assert len(fm.time) == len(a.time), "fire cube and arrays are out of sync; rebuild arrays"
    for split in splits:
        idx = val_idx if split == "val" else _issue_days(split)
        logits = val_logits if split == "val" else predict(idx)
        probs = np.stack([cal.transform(logits[:, d], d + 1) for d in range(7)], axis=1)
        np.save(run / f"probs_{split}.npy", probs.astype(np.float16))
        np.save(run / f"idx_{split}.npy", idx)

        def lookup(issue_idx, lead, probs=probs, idx=idx):
            assert np.array_equal(issue_idx, idx)
            return probs[:, lead - 1]

        res = evaluate(name, lookup, fm, rec, idx)
        res.insert(0, "split", split)
        res.to_csv(ROOT / "reports" / f"{name}_{split}.csv", index=False)
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
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--crop", type=int, default=0, help="train on random crop x crop windows")
    p.set_defaults(func=train_model)

    p = sub.add_parser("evaluate-model", help="calibrate on val, score val/test vs baselines")
    p.add_argument("--name", default="unet")
    p.add_argument("--splits", nargs="+", default=["val", "test"])
    p.set_defaults(func=evaluate_model)

    p = sub.add_parser("evaluate-ensemble", help="average saved model probabilities and score")
    p.add_argument("--name", required=True)
    p.add_argument("--members", nargs="+", required=True, help="model names under models/")
    p.add_argument("--splits", nargs="+", default=["val", "test"])
    p.set_defaults(func=evaluate_ensemble)

    p = sub.add_parser("train-lgbm", help="LightGBM baseline: fit, calibrate, score")
    p.add_argument("--name", default="lgbm")
    p.add_argument("--neg-frac", type=float, default=0.04, help="share of non-fire rows kept")
    p.add_argument("--splits", nargs="+", default=["val", "test"])
    p.set_defaults(func=train_lgbm)

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
    p.add_argument("--last-day", help="YYYY-MM-DD: stop here (e.g. where ERA5 ends)")
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
