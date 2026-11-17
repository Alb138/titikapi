"""Which inputs drive the forecasts? Group permutation importance for the U-Net and LightGBM.

Each input group is scrambled in turn and the forecasts are re-scored. Groups that change from
day to day (fire history, weather, season, ENSO, climatology) are shuffled across days, which
keeps each map intact but pairs it with the wrong day. Fixed map layers (land, forest loss) are
shuffled across land cells instead, since shuffling a fixed map across days changes nothing. The
importance of a group is how much PR-AUC drops. Both models see exactly the same scrambled inputs
(LightGBM's features are rebuilt from them).
"""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import pandas as pd
import torch

from titikapi.config import FIRE_HISTORY_DAYS, LEAD_DAYS
from titikapi.models.dataset import STATIC_VARS, WEATHER_VARS, Arrays, grid_inputs
from titikapi.models.metrics import average_precision

STATIC_GROUPS = ("land (peat, elevation, tree cover)", "forest loss")
QUIET_DAYS = 14  # new-ignition slice: no fire in the 14 days up to the issue day


def input_groups(a: Arrays) -> dict[str, tuple[str, list[int]]]:
    """Group name -> ("grid" | "weather", channel indices). The land mask is never scrambled."""
    h, s = FIRE_HISTORY_DAYS + 3, len(STATIC_VARS)
    groups = {
        "fire history": ("grid", list(range(h))),
        "land (peat, elevation, tree cover)": ("grid", list(range(h, h + s))),
        "forest loss": ("grid", [h + s, h + s + 1]),
        "fire climatology": ("grid", [h + s + 2, h + s + 3]),
        "season": ("grid", [h + s + 4, h + s + 5]),
        "ENSO": ("grid", [h + s + 6]),
    }
    assert h + s + 7 == a.n_grid_channels - 1, "grid channel layout changed"
    wv = list(WEATHER_VARS)
    for name, names in {
        "fire-weather codes": ("ffmc", "dmc", "dc", "isi", "bui", "fwi"),
        "rain memory": ("pr", "dry_days", "rain_7d", "rain_30d", "rain_60d", "rain_90d"),
        "temperature, humidity, wind": ("t2m", "rh", "wind", "u10", "v10"),
    }.items():
        groups[name] = ("weather", [wv.index(v) for v in names])
    assert sum(len(c) for k, c in groups.values() if k == "weather") == len(wv)
    return groups


@contextmanager
def scrambled(x, w, mask, kind, channels, static: bool, rng):
    """Scramble channels of x (D, C, ny, nx) or w (D, Cw, h, w) in place; restore on exit."""
    arr = x if kind == "grid" else w
    saved = arr[:, channels].copy()
    if static:
        land = np.flatnonzero(mask.ravel())
        flat = saved.reshape(len(arr), len(channels), -1).copy()
        flat[:, :, land] = flat[:, :, rng.permutation(land)]
        arr[:, channels] = flat.reshape(saved.shape)
    else:
        arr[:, channels] = saved[rng.permutation(len(arr))]
    try:
        yield
    finally:
        arr[:, channels] = saved


def score(p: np.ndarray, y: np.ndarray, quiet: np.ndarray) -> dict[str, float]:
    """PR-AUC on all land cells and on new-ignition cells, for lead day 1 and the last lead."""
    out = {}
    for lead in (1, LEAD_DAYS):
        pl, yl = p[:, lead - 1], y[:, lead - 1]
        out[f"ap_d{lead}"] = average_precision(yl, pl)
        out[f"new_ap_d{lead}"] = average_precision(yl[quiet], pl[quiet])
    return out


@torch.no_grad()
def unet_scores(model, x, w, mask, batch: int = 8) -> np.ndarray:
    """(D, 7, N) logits for land cells."""
    m = torch.from_numpy(mask)
    out = []
    for i in range(0, len(x), batch):
        logits = model(torch.from_numpy(x[i : i + batch]), torch.from_numpy(w[i : i + batch]))
        out.append(logits[:, :, m].numpy())
    return np.concatenate(out)


def lgbm_scores(boosters, fb, x, w) -> np.ndarray:
    """(D, 7, N) raw scores, from features rebuilt out of (possibly scrambled) inputs."""
    feats = np.concatenate([fb.from_inputs(x[d], w[d]) for d in range(len(x))])
    raw = np.stack([b.predict(feats, raw_score=True) for b in boosters], axis=1)
    return raw.reshape(len(x), -1, LEAD_DAYS).transpose(0, 2, 1)


def permutation_importance(
    a: Arrays, idx: np.ndarray, predictors: dict, repeats: int = 2, seed: int = 0
) -> pd.DataFrame:
    """Rows: model, group, metric, baseline, scrambled (mean over repeats), drop."""
    x = np.stack([grid_inputs(a, t) for t in idx])
    w = np.asarray(a.weather[idx], np.float32)
    y = np.stack([[np.asarray(a.fire_on[t + d]) for d in range(1, LEAD_DAYS + 1)] for t in idx])
    quiet = np.asarray(a.days_since[idx]) >= QUIET_DAYS
    rows = []
    for model_name, predict in predictors.items():
        base = score(predict(x, w), y, quiet)
        print(f"{model_name}: baseline {base}", flush=True)
        for group, (kind, channels) in input_groups(a).items():
            runs = []
            for r in range(repeats):
                rng = np.random.default_rng(seed + r)
                with scrambled(x, w, a.mask, kind, channels, group in STATIC_GROUPS, rng):
                    runs.append(score(predict(x, w), y, quiet))
            for metric, b in base.items():
                s = float(np.mean([run[metric] for run in runs]))
                rows.append(
                    {
                        "model": model_name,
                        "group": group,
                        "metric": metric,
                        "baseline": b,
                        "scrambled": s,
                        "drop": b - s,
                    }
                )
            print(f"  {group}: PR-AUC day 1 drop {rows[-4]['drop']:+.4f}", flush=True)
    return pd.DataFrame(rows)
