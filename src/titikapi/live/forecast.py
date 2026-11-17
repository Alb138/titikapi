"""Daily forecast from the live state, in exactly the evaluation pipeline's steps:
member models -> per-member calibration -> U-Net ensemble -> + LightGBM -> final calibration."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch

from titikapi.config import LEAD_DAYS
from titikapi.live.bundle import FINAL, Bundle
from titikapi.live.state import FireState, WeatherState
from titikapi.models.dataset import (
    WEATHER_VARS,
    Arrays,
    days_since_fire,
    grid_inputs,
    standardize_weather,
)
from titikapi.models.train import PlattCalibrator, TrainConfig, build_model

QUIET_DAYS = 14


def live_arrays(b: Bundle, fire: FireState, weather: WeatherState, oni: float) -> Arrays:
    """An Arrays view over the fire window whose last day is the issue day."""
    day = fire.dates[-1]
    assert weather.feats["time"].values[-1] == np.datetime64(day), "weather and fire days differ"
    on = fire.count > 0
    T = len(fire.dates)
    w = np.zeros((T, len(WEATHER_VARS), len(b.weather_lat), len(b.weather_lon)), np.float16)
    w[-1] = standardize_weather(weather.feats.isel(time=slice(-1, None)), b.stats)[0]
    oni_arr = np.zeros(T, np.float32)
    oni_arr[-1] = oni
    return Arrays(
        time=fire.dates,
        mask=b.mask,
        fire_on=on,
        fire_count=fire.count,
        frp_log=fire.frp_log,
        days_since=days_since_fire(on),
        static=b.static,
        forest_loss=b.forest_loss,
        loss_years=b.loss_years,
        clim=b.clim,
        oni=oni_arr,
        weather=w,
        weather_lat=b.weather_lat,
        weather_lon=b.weather_lon,
    )


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _calibrated(path, scores: np.ndarray) -> np.ndarray:
    """scores (7, N) -> calibrated probabilities with that model's per-lead Platt scaling."""
    cal = PlattCalibrator.load(path / "calibration.json")
    return np.stack([cal.transform(scores[d], d + 1) for d in range(LEAD_DAYS)])


def _models(b: Bundle, a: Arrays) -> dict:
    """Load the member models once per bundle (the replay backtest calls forecast() daily)."""
    import lightgbm as lgb

    if getattr(b, "_loaded", None) is None:
        nets = {}
        for m in b.meta["unet_members"]:
            cfg = TrainConfig(**json.loads((b.models / m / "config.json").read_text()))
            net = build_model(a, cfg)
            state = torch.load(b.models / m / "best.pt", map_location="cpu", weights_only=True)
            net.load_state_dict(state)
            nets[m] = net.eval()
        boosters = [
            lgb.Booster(model_file=str(b.models / "lgbm" / f"lead{d}.txt"))
            for d in range(1, LEAD_DAYS + 1)
        ]
        b._loaded = {"unets": nets, "boosters": boosters}
    return b._loaded


@torch.no_grad()
def forecast(b: Bundle, a: Arrays) -> dict[str, np.ndarray]:
    """(7, N) probabilities for the last day of `a`: final model, its parts, and persistence."""
    from titikapi.models.lgbm import FeatureBuilder

    models = _models(b, a)
    t = len(a.time) - 1
    x = grid_inputs(a, t)[None]
    w = np.asarray(a.weather[t : t + 1], np.float32)
    mask = torch.from_numpy(np.array(a.mask))
    out = {}
    for m, net in models["unets"].items():
        logits = net(torch.from_numpy(x), torch.from_numpy(w))[0][:, mask].numpy()
        out[m] = _calibrated(b.models / m, logits)
    feats = FeatureBuilder(a).from_inputs(x[0], w[0])
    raw = np.stack([bst.predict(feats, raw_score=True) for bst in models["boosters"]])
    out["lgbm"] = _calibrated(b.models / "lgbm", raw)
    out["ens_unet"] = _calibrated(
        b.models / "ens_unet", _logit(np.mean([out[m] for m in b.meta["unet_members"]], axis=0))
    )
    out[FINAL] = _calibrated(b.models / FINAL, _logit((out["ens_unet"] + out["lgbm"]) / 2))
    out["persistence"] = persistence(b, a)
    return out


def persistence(b: Bundle, a: Arrays) -> np.ndarray:
    """Reference forecast: training-period fire rate for the cell's state today (fire today /
    fire in the past 14 days / none), tie-broken by climatology, as in evaluation."""
    t = len(a.time) - 1
    today = np.asarray(a.fire_on[t])
    recent = np.asarray(a.days_since[t]) < QUIET_DAYS
    state = np.where(today, 0, np.where(recent, 1, 2))
    out = []
    for lead in range(1, LEAD_DAYS + 1):
        month = (a.time[t] + pd.Timedelta(days=lead)).month
        out.append(b.persistence[lead][state] + 1e-6 * b.clim[month])
    return np.stack(out).astype(np.float32)
