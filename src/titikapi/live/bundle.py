"""The inference bundle: everything the daily job needs that is fixed after training.

Trained models and their calibrations, the static map layers, the training statistics used to
scale weather, persistence rates for the scorecard's reference forecast, and the locations of
known gas flares / volcanoes (to clean near-real-time fire data). Built once from local training
outputs and published as a Hugging Face model repo.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

UNET_MEMBERS = ("unet_reg", "unet_crop", "unet_crop16")
ARRAY_FILES = (
    "mask",
    "static",
    "forest_loss",
    "loss_years",
    "clim",
    "province",
    "weather_lat",
    "weather_lon",
)
FINAL = "ens_mix"  # U-Net ensemble + LightGBM, chosen on validation


def make_bundle(out: Path, unets=UNET_MEMBERS, lgbm: str = "lgbm") -> Path:
    from titikapi.config import RAW_DIR, SPLITS
    from titikapi.data import firms
    from titikapi.data.cube import load_fire_matrix, recent_fire
    from titikapi.models.baselines import Climatology, Persistence
    from titikapi.models.dataset import ARRAYS_DIR
    from titikapi.models.evaluate import issue_days
    from titikapi.models.train import MODELS_DIR

    shutil.rmtree(out, ignore_errors=True)
    (out / "arrays").mkdir(parents=True)
    for name in ARRAY_FILES:
        shutil.copy(ARRAYS_DIR / f"{name}.npy", out / "arrays")
    shutil.copy(ARRAYS_DIR / "stats.json", out / "arrays")

    models = out / "models"
    for m in unets:
        (models / m).mkdir(parents=True)
        for f in ("best.pt", "config.json", "calibration.json"):
            shutil.copy(MODELS_DIR / m / f, models / m)
    (models / "lgbm").mkdir()
    for f in [*(f"lead{d}.txt" for d in range(1, 8)), "calibration.json"]:
        shutil.copy(MODELS_DIR / lgbm / f, models / "lgbm")
    for ens, members in (("ens_unet", list(unets)), (FINAL, ["ens_unet", "lgbm"])):
        (models / ens).mkdir()
        shutil.copy(MODELS_DIR / ens / "calibration.json", models / ens)
        (models / ens / "members.txt").write_text("\n".join(members))

    # persistence (the scorecard's reference): P(fire at t+lead | state at t) from training days
    fm = load_fire_matrix()
    rec = recent_fire(fm.on)
    start, end = (pd.Timestamp(d) for d in SPLITS["train"])
    train_days = np.flatnonzero((fm.time >= start) & (fm.time <= end))
    clim = Climatology().fit(fm, train_days)
    pers = Persistence().fit(fm, rec, issue_days(fm, SPLITS["train"]), climatology=clim)
    rates = {str(lead): r.tolist() for lead, r in pers.rate.items()}
    (out / "persistence.json").write_text(json.dumps(rates, indent=1))

    # flagged static sources (type > 0) from every science-quality file on disk
    files = sorted((RAW_DIR / "firms").glob("viirs-*_Indonesia.csv"))
    files += sorted((RAW_DIR / "firms" / "api").glob("VIIRS_*_SP_*.csv"))
    flagged = [pd.read_csv(f, usecols=["longitude", "latitude", "type"]) for f in files]
    np.save(out / "static_sources.npy", firms.static_source_points(flagged))

    meta = {
        "final_model": FINAL,
        "unet_members": list(unets),
        "splits": SPLITS,
        "live_sensor": "VIIRS NOAA-20 (S-NPP, the training sensor, retired 2026-11-02)",
        "static_source_files": [f.name for f in files],
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    return out


class Bundle:
    def __init__(self, path: Path):
        self.path = Path(path)
        arr = self.path / "arrays"
        for name in ARRAY_FILES:
            setattr(self, name, np.load(arr / f"{name}.npy"))
        self.stats = json.loads((arr / "stats.json").read_text())
        rates = json.loads((self.path / "persistence.json").read_text())
        self.persistence = {int(k): np.asarray(v, np.float32) for k, v in rates.items()}
        self.static_sources = np.load(self.path / "static_sources.npy")
        self.meta = json.loads((self.path / "meta.json").read_text())
        self.models = self.path / "models"
