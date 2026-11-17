"""Files the website reads, the forecast archive, and the live scorecard.

latest/forecast.json     issue date, per-lead regional and per-province summaries, riskiest cells
latest/risk_d{1..7}.png  forecast probability on the 0.1 deg grid (transparent outside land)
latest/observed.geojson  fire cells detected in the last 3 days
latest/scorecard.json    how past forecasts did against fires observed since
archive/forecasts/<date>.npz   final model + persistence probabilities, (7, N) each
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

from titikapi.config import LEAD_DAYS, PROVINCES
from titikapi.grid import Grid
from titikapi.live.bundle import FINAL, Bundle
from titikapi.live.state import REFRESH_DAYS, FireState
from titikapi.models.metrics import average_precision, brier, topk_coverage

PNG_NORM = (1e-3, 0.8)  # log colour scale, as in the README figures
TOP_CELLS = 25
REL_EDGES = np.array([0, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0])  # calibration bins


def _cell_coords(b: Bundle) -> tuple[np.ndarray, np.ndarray]:
    grid = Grid()
    rows, cols = np.nonzero(b.mask)
    return grid.lats[rows], grid.lons[cols]


def risk_png(b: Bundle, p: np.ndarray, path: Path) -> None:
    from matplotlib.colors import LogNorm
    from PIL import Image

    from titikapi.viz import FIRE_CMAP

    g = np.zeros(b.mask.shape, np.float32)
    g[b.mask] = p
    rgba = FIRE_CMAP(LogNorm(*PNG_NORM, clip=True)(np.maximum(g, PNG_NORM[0])))
    rgba[..., 3] = np.where(b.mask, 0.85, 0.0)
    Image.fromarray((rgba * 255).astype(np.uint8), "RGBA").save(path, optimize=True)


def forecast_json(b: Bundle, issue: pd.Timestamp, probs: dict, sensor_day: pd.Timestamp) -> dict:
    lat, lon = _cell_coords(b)
    names = list(PROVINCES)
    p_final = probs[FINAL]
    k = int(np.ceil(0.05 * len(p_final[0])))
    leads = []
    for d in range(LEAD_DAYS):
        p = p_final[d]
        top5 = np.argsort(-p)[:k]
        provinces = []
        for i, name in enumerate(names, start=1):
            sel = b.province == i
            provinces.append(
                {
                    "name": name,
                    "expected_fire_cells": round(float(p[sel].sum()), 1),
                    "mean_probability": round(float(p[sel].mean()), 4),
                    "cells_in_top5pct": int(np.isin(np.flatnonzero(sel), top5).sum()),
                }
            )
        top = np.argsort(-p)[:TOP_CELLS]
        leads.append(
            {
                "lead": d + 1,
                "date": str((issue + pd.Timedelta(days=d + 1)).date()),
                "expected_fire_cells": round(float(p.sum()), 1),
                "top5pct_threshold": round(float(p[top5].min()), 4),
                "provinces": sorted(provinces, key=lambda r: -r["expected_fire_cells"]),
                "top_cells": [
                    {
                        "lat": round(float(lat[i]), 2),
                        "lon": round(float(lon[i]), 2),
                        "probability": round(float(p[i]), 3),
                        "province": names[b.province[i] - 1],
                    }
                    for i in top
                ],
            }
        )
    grid = Grid().region
    return {
        "issue_date": str(issue.date()),
        "generated_utc": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%MZ"),
        "fire_data_through": str(sensor_day.date()),
        "model": "U-Net ensemble + LightGBM, calibrated (see github.com/Alb138/titikapi)",
        "fire_sensor": "VIIRS NOAA-20 (NASA FIRMS)",
        "weather": "ECMWF IFS open data (CC BY 4.0)",
        "png_bounds": [
            [grid.lon_min, grid.lat_max],
            [grid.lon_max, grid.lat_max],
            [grid.lon_max, grid.lat_min],
            [grid.lon_min, grid.lat_min],
        ],
        "png_norm": {"type": "log", "min": PNG_NORM[0], "max": PNG_NORM[1]},
        "leads": leads,
    }


def observed_geojson(b: Bundle, fire: FireState, days: int = 3) -> dict:
    lat, lon = _cell_coords(b)
    names = list(PROVINCES)
    feats = []
    for ago in range(days):
        t = len(fire.dates) - 1 - ago
        for i in np.flatnonzero(fire.count[t]):
            feats.append(
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [round(float(lon[i]), 2), round(float(lat[i]), 2)],
                    },
                    "properties": {
                        "date": str(fire.dates[t].date()),
                        "days_ago": ago,
                        "detections": int(fire.count[t, i]),
                        "province": names[b.province[i] - 1],
                    },
                }
            )
    return {"type": "FeatureCollection", "features": feats}


def save_archive(probs: dict, path: Path) -> None:
    np.savez_compressed(
        path,
        final=probs[FINAL].astype(np.float16),
        persistence=probs["persistence"].astype(np.float16),
    )


def score_archive(archive: Path, fire: FireState, scores_csv: Path) -> pd.DataFrame:
    """Score every (forecast, lead) whose target day is now observed and not yet scored.
    One row per issue date x lead: AP and top-5% coverage for the model and persistence."""
    try:
        old = pd.read_csv(scores_csv)
    except (FileNotFoundError, pd.errors.EmptyDataError):  # first runs: nothing scored yet
        old = pd.DataFrame()
    done = set(zip(old.get("issue_date", []), old.get("lead", []), strict=False))
    pos = {d: i for i, d in enumerate(fire.dates.strftime("%Y-%m-%d"))}
    rows = []
    for f in sorted(archive.glob("*.npz")):
        issue = pd.Timestamp(f.stem)
        z = np.load(f)
        for lead in range(1, LEAD_DAYS + 1):
            target = str((issue + pd.Timedelta(days=lead)).date())
            # score only days past the near-real-time refresh window (their data is final)
            late = target in pos and pos[target] >= len(fire.dates) - REFRESH_DAYS
            if (f.stem, lead) in done or target not in pos or late:
                continue
            y = fire.count[pos[target]] > 0
            row = {"issue_date": f.stem, "lead": lead, "target_date": target, "fires": int(y.sum())}
            for name in ("final", "persistence"):
                p = z[name][lead - 1].astype(np.float32)
                row[f"{name}_ap"] = average_precision(y, p)
                row[f"{name}_top5"] = topk_coverage(y[None], p[None]) if y.any() else np.nan
                row[f"{name}_brier"] = brier(y, p)
            # calibration: per probability bin, cells / summed forecast / fires (pooled later)
            p = z["final"][lead - 1].astype(np.float64)
            k = np.digitize(p, REL_EDGES[1:-1])
            for i in range(len(REL_EDGES) - 1):
                row[f"rel{i}_n"] = int((k == i).sum())
                row[f"rel{i}_p"] = float(p[k == i].sum())
                row[f"rel{i}_y"] = int(y[k == i].sum())
            rows.append(row)
    return pd.concat([old, pd.DataFrame(rows)], ignore_index=True)


def scorecard(scores: pd.DataFrame, today: pd.Timestamp) -> dict:
    """Fire-weighted summaries (days with more fires count more) for the last 30 days and since
    launch, per lead; plus the day-1 series for a chart."""
    if scores.empty:
        return {"updated": str(today.date()), "n_scored": 0, "windows": {}, "daily": []}
    s = scores.copy()
    s["target_date"] = pd.to_datetime(s["target_date"])

    def summary(d: pd.DataFrame) -> list[dict]:
        out = []
        for lead, g in d.groupby("lead"):
            g = g[g["fires"] > 0]
            w = g["fires"]
            out.append(
                {
                    "lead": int(lead),
                    "days": int(len(g)),
                    "fires": int(w.sum()),
                    **{
                        f"{m}_{k}": (
                            round(float(np.average(g[f"{m}_{k}"], weights=w)), 4)
                            if len(g)
                            else None
                        )
                        for m in ("final", "persistence")
                        for k in ("ap", "top5")
                    },
                    # Brier is a mean over all cells, so every day counts equally
                    **{
                        f"{m}_brier": round(float(d.loc[d["lead"] == lead, f"{m}_brier"].mean()), 5)
                        for m in ("final", "persistence")
                    },
                }
            )
        return out

    def calibration(d: pd.DataFrame, lead: int) -> list[dict]:
        g = d[d["lead"] == lead]
        out = []
        for i in range(len(REL_EDGES) - 1):
            n = g[f"rel{i}_n"].sum()
            out.append(
                {
                    "bin": f"{REL_EDGES[i]:g}-{REL_EDGES[i + 1]:g}",
                    "cells": int(n),
                    "mean_forecast": round(float(g[f"rel{i}_p"].sum() / n), 4) if n else None,
                    "observed": round(float(g[f"rel{i}_y"].sum() / n), 4) if n else None,
                }
            )
        return out

    recent = s[s["target_date"] > today - pd.Timedelta(days=30)]
    d1 = s[s["lead"] == 1].sort_values("target_date")
    return {
        "updated": str(today.date()),
        "n_scored": int(len(s)),
        "first_issue_date": str(s["issue_date"].min()),
        "note": "Scores are fire-weighted means over target days with at least one fire. "
        "Few fires burn in the wet season (Nov-Apr), so scores are noisy then.",
        "windows": {"last_30_days": summary(recent), "since_launch": summary(s)},
        "calibration_since_launch": {
            f"lead{d}": calibration(s, d) for d in (1, LEAD_DAYS) if "rel0_n" in s
        },
        "daily": [
            {
                "date": str(r.target_date.date()),
                "fires": int(r.fires),
                "final_top5": None if pd.isna(r.final_top5) else round(float(r.final_top5), 3),
                "persistence_top5": None
                if pd.isna(r.persistence_top5)
                else round(float(r.persistence_top5), 3),
            }
            for r in d1.itertuples()
        ],
    }


def write_latest(out: Path, b: Bundle, issue, probs, fire: FireState, scores: pd.DataFrame) -> None:
    latest = out / "latest"
    latest.mkdir(parents=True, exist_ok=True)
    for d in range(LEAD_DAYS):
        risk_png(b, probs[FINAL][d], latest / f"risk_d{d + 1}.png")
    (latest / "forecast.json").write_text(
        json.dumps(forecast_json(b, issue, probs, fire.dates[-1]))
    )
    (latest / "observed.geojson").write_text(json.dumps(observed_geojson(b, fire)))
    (latest / "scorecard.json").write_text(json.dumps(scorecard(scores, issue)))
