"""LightGBM baseline: gradient boosting on per-cell features, one model per lead day.

It sees the same information as the U-Net at issue day t (fire history, weather, static layers,
climatology, season, ENSO). A per-cell model has no view of its surroundings, so it also gets
neighbourhood fire counts: the spatial context a U-Net learns by itself.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import uniform_filter

from titikapi.config import FIRE_HISTORY_DAYS, LEAD_DAYS
from titikapi.grid import Grid
from titikapi.models.dataset import STATIC_VARS, WEATHER_VARS, Arrays, grid_inputs
from titikapi.models.unet import weather_sampling_grid

H = FIRE_HISTORY_DAYS
NEIGHBOURHOODS = (5, 15)  # window widths in cells (~55 km and ~165 km)

FEATURES = [
    *("fire_t0", "fire_t1", "fire_t2", "fire_3d", "fire_7d", "fire_14d"),
    *("log_count", "frp_log", "days_since"),
    *STATIC_VARS,
    *("forest_loss_3y", "forest_loss_all", "clim_lead1", f"clim_lead{LEAD_DAYS}"),
    *("doy_sin", "doy_cos", "oni"),
    *(f"nb{s}_fire_today" for s in NEIGHBOURHOODS),
    *(f"nb{s}_fire_7d" for s in NEIGHBOURHOODS),
    "nb5_frp_today",
    *WEATHER_VARS,
]

# Slow learning rate + strong regularisation. Every model grows the full number of trees and keeps
# the count with the best validation AP (as the U-Net keeps its best epoch). Patience-based early
# stopping cut leads 3-7 at 3-30 trees: the wetter validation years (2021-22) make validation
# loss rise early from the base-rate shift even while the ranking is still improving.
MIN_TREES = 50
PARAMS = {
    "objective": "binary",
    "metric": "average_precision",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_child_samples": 500,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 10.0,
    "verbose": -1,
}


def _box_sum(g: np.ndarray, size: int) -> np.ndarray:
    return uniform_filter(g, size, mode="constant") * size * size


class FeatureBuilder:
    def __init__(self, a: Arrays):
        self.a = a
        grid = Grid()
        self.sg = weather_sampling_grid(a.weather_lat, a.weather_lon, grid.lats, grid.lons)

    def weather(self, t: int) -> np.ndarray:
        """(Cw, ny, nx): weather bilinearly resampled to the grid, exactly as the U-Net sees it."""
        w = torch.from_numpy(np.asarray(self.a.weather[t], np.float32))[None]
        return F.grid_sample(w, self.sg, mode="bilinear", align_corners=True)[0].numpy()

    def day(self, t: int) -> np.ndarray:
        """(N land cells, len(FEATURES)) float32 for issue day t (data up to day t only)."""
        g = grid_inputs(self.a, t)
        fire = g[:H]  # today first
        week = fire[:7].sum(0)
        cols = [fire[0], fire[1], fire[2], fire[:3].sum(0), week, fire.sum(0)]
        cols += list(g[H : H + 3])  # log count, frp, days since fire
        cols += list(g[H + 3 : -1])  # static, forest loss, climatology, season, ENSO (no mask)
        cols += [_box_sum(fire[0], s) for s in NEIGHBOURHOODS]
        cols += [_box_sum(week, s) for s in NEIGHBOURHOODS]
        cols.append(_box_sum(g[H + 1], 5))
        cols += list(self.weather(t))
        assert len(cols) == len(FEATURES)
        m = self.a.mask
        return np.stack([c[m] for c in cols], axis=1).astype(np.float32)


def labels(a: Arrays, t: int) -> np.ndarray:
    """(N, 7) bool: fire on lead days 1..7."""
    return np.stack([np.asarray(a.fire_on[t + d]) for d in range(1, LEAD_DAYS + 1)], axis=1)


def sample_rows(fb: FeatureBuilder, idx, neg_frac: float, seed: int = 0):
    """Every cell that burns on any lead day, plus a random `neg_frac` of the rest.

    Weights are inverse inclusion probabilities, so each lead's model trains on an unbiased
    (weighted) picture of the full population."""
    rng = np.random.default_rng(seed)
    xs, ys, ws = [], [], []
    for t in idx:
        y = labels(fb.a, t)
        pos = y.any(1)
        take = pos | (rng.random(len(pos)) < neg_frac)
        xs.append(fb.day(t)[take])
        ys.append(y[take])
        ws.append(np.where(pos[take], 1.0, 1.0 / neg_frac).astype(np.float32))
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(ws)


def fit(fb: FeatureBuilder, train_idx, val_idx, neg_frac: float = 0.04, rounds: int = 600):
    import lightgbm as lgb

    xt, yt, wt = sample_rows(fb, train_idx, neg_frac, seed=0)
    xv, yv, wv = sample_rows(fb, val_idx, neg_frac, seed=1)
    print(f"lightgbm: {len(xt):,} train rows, {len(xv):,} val rows, {len(FEATURES)} features")
    boosters = []
    for d in range(LEAD_DAYS):
        dtrain = lgb.Dataset(xt, yt[:, d], weight=wt, feature_name=FEATURES, free_raw_data=False)
        dval = lgb.Dataset(xv, yv[:, d], weight=wv, reference=dtrain)
        evals: dict = {}
        b = lgb.train(
            PARAMS,
            dtrain,
            num_boost_round=rounds,
            valid_sets=[dval],
            callbacks=[lgb.record_evaluation(evals)],
        )
        ap = np.asarray(evals["valid_0"]["average_precision"])
        lo = min(MIN_TREES, rounds)
        b.best_iteration = lo + int(np.argmax(ap[lo - 1 :]))
        curve = "  ".join(f"{n}:{ap[n - 1]:.4f}" for n in (lo, rounds // 2, rounds))
        print(f"  lead {d + 1}: best {b.best_iteration} trees ({curve})", flush=True)
        boosters.append(b)
    return boosters


def predict_raw(boosters, fb: FeatureBuilder, idx, batch: int = 32) -> np.ndarray:
    """(D, 7, N) float32 raw scores (log-odds) for every land cell, in `idx` order."""
    out = []
    for i in range(0, len(idx), batch):
        x = np.concatenate([fb.day(t) for t in idx[i : i + batch]])
        raw = np.stack(
            [b.predict(x, raw_score=True, num_iteration=b.best_iteration) for b in boosters], 1
        )
        out.append(raw.reshape(len(idx[i : i + batch]), -1, LEAD_DAYS).transpose(0, 2, 1))
    return np.concatenate(out).astype(np.float32)
