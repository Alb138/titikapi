"""The daily job: pull state from Hugging Face, roll it forward to today, forecast, score past
forecasts, publish. Run by .github/workflows/daily.yml; `titikapi live-daily --help` for local use.
"""

from __future__ import annotations

import datetime as dt
import os
from pathlib import Path

import numpy as np
import pandas as pd

from titikapi.config import DATA_DIR

LIVE_REPO = os.environ.get("TITIKAPI_LIVE_REPO", "Alb138/titikapi-live")  # dataset: state + outputs
BUNDLE_REPO = os.environ.get("TITIKAPI_BUNDLE_REPO", "Alb138/titikapi-model")  # models + statics
WORK = DATA_DIR / "live"
KEEP_ARCHIVE_DAYS = 14  # forecasts older than this are fully scored; no need to download them


def pull(work: Path = WORK) -> None:
    from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download

    snapshot_download(BUNDLE_REPO, local_dir=work / "bundle")
    cutoff = str(dt.date.today() - dt.timedelta(days=KEEP_ARCHIVE_DAYS))
    for f in list_repo_files(LIVE_REPO, repo_type="dataset"):
        recent = f.startswith("archive/forecasts/") and Path(f).stem >= cutoff
        if f.startswith("state/") or f == "scores.csv" or recent:
            hf_hub_download(LIVE_REPO, f, repo_type="dataset", local_dir=work / "live")


def push(issue: dt.date, work: Path = WORK) -> None:
    from huggingface_hub import upload_folder

    upload_folder(
        folder_path=work / "live",
        repo_id=LIVE_REPO,
        repo_type="dataset",
        commit_message=f"Forecast issued {issue}",
        allow_patterns=["state/*", "latest/*", "scores.csv", f"archive/forecasts/{issue}.npz"],
    )


def oni_for(day: pd.Timestamp) -> float:
    from titikapi.data import enso

    oni = enso.parse_oni(enso.download_oni(force=True).read_text())
    return float(np.nan_to_num(enso.oni_known_at(pd.DatetimeIndex([day]), oni))[0])


def run(issue: dt.date, work: Path = WORK) -> dict:
    """Roll the state in `work/live/state` forward to `issue`, forecast and publish locally."""
    from titikapi.live import publish
    from titikapi.live.bundle import FINAL, Bundle
    from titikapi.live.forecast import forecast, live_arrays
    from titikapi.live.state import FireState, WeatherState, update_fire, update_weather

    b = Bundle(work / "bundle")
    live = work / "live"
    state = live / "state"
    state.mkdir(parents=True, exist_ok=True)
    issue = pd.Timestamp(issue)

    fire_path = state / "fire.npz"
    fire = FireState.load(fire_path) if fire_path.exists() else None
    fire = update_fire(fire, issue, b.mask, b.static_sources)
    print(
        f"fire state {fire.dates[0].date()} .. {fire.dates[-1].date()}, "
        f"{int((fire.count[-1] > 0).sum())} fire cells on {issue.date()}",
        flush=True,
    )

    weather = WeatherState.load(state / "weather.nc")
    weather = update_weather(weather, issue, b.weather_lat, b.weather_lon, work / "ifs")

    a = live_arrays(b, fire, weather, oni_for(issue))
    probs = forecast(b, a)
    archive = live / "archive" / "forecasts"
    archive.mkdir(parents=True, exist_ok=True)
    publish.save_archive(probs, archive / f"{issue.date()}.npz")
    scores = publish.score_archive(archive, fire, live / "scores.csv")
    scores.to_csv(live / "scores.csv", index=False)
    publish.write_latest(live, b, issue, probs, fire, scores)
    fire.save(fire_path)
    weather.save(state / "weather.nc")
    p = probs[FINAL]
    summary = {
        "issue": str(issue.date()),
        "expected_fire_cells_day1": float(p[0].sum()),
        "expected_fire_cells_day7": float(p[-1].sum()),
        "scored_rows": len(scores),
    }
    print(summary, flush=True)
    return summary


def bootstrap(weather_through: dt.date, work: Path = WORK, push_bundle: bool = False) -> None:
    """One-time setup: build the bundle from local training outputs and seed the weather state
    from ERA5 (the fire state builds itself on the first daily run)."""
    from titikapi.live.bundle import make_bundle
    from titikapi.live.state import bootstrap_weather

    make_bundle(work / "bundle")
    state = work / "live" / "state"
    state.mkdir(parents=True, exist_ok=True)
    bootstrap_weather(weather_through).save(state / "weather.nc")
    print(f"bundle + weather state (ERA5 through {weather_through}) in {work}", flush=True)
    if push_bundle:
        from huggingface_hub import create_repo, upload_folder

        create_repo(BUNDLE_REPO, exist_ok=True)
        upload_folder(
            folder_path=work / "bundle", repo_id=BUNDLE_REPO, commit_message="Inference bundle"
        )
        create_repo(LIVE_REPO, repo_type="dataset", exist_ok=True)
        upload_folder(
            folder_path=work / "live",
            repo_id=LIVE_REPO,
            repo_type="dataset",
            allow_patterns=["state/*"],
            commit_message="Seed weather state from ERA5",
        )
