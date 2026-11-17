"""Live pipeline pieces that don't need the network: scoring bookkeeping and the scorecard."""

import numpy as np
import pandas as pd

from titikapi.live.publish import score_archive, scorecard
from titikapi.live.state import REFRESH_DAYS, FireState


def _fire(days: int, n: int = 50, seed: int = 0) -> FireState:
    rng = np.random.default_rng(seed)
    count = (rng.random((days, n)) < 0.1).astype(np.uint8)
    dates = pd.date_range("2026-09-01", periods=days, freq="D")
    return FireState(dates, count, np.zeros((days, n), np.float16))


def test_score_archive_scores_each_lead_once_and_skips_unsettled_days(tmp_path):
    fire = _fire(20)
    archive = tmp_path / "forecasts"
    archive.mkdir()
    for day in ("2026-09-05", "2026-09-14"):
        p = np.random.default_rng(1).random((7, 50)).astype(np.float16)
        np.savez(archive / f"{day}.npz", final=p, persistence=p)
    csv = tmp_path / "scores.csv"
    s = score_archive(archive, fire, csv)
    last_settled = fire.dates[-1 - REFRESH_DAYS]
    assert (pd.to_datetime(s["target_date"]) <= last_settled).all()
    assert len(s[s.issue_date == "2026-09-05"]) == 7  # all leads settled
    assert len(s[s.issue_date == "2026-09-14"]) == (last_settled - pd.Timestamp("2026-09-14")).days
    s.to_csv(csv, index=False)
    again = score_archive(archive, fire, csv)
    assert len(again) == len(s)  # nothing scored twice


def test_scorecard_weights_days_by_fire_count():
    rows = [
        {
            "issue_date": "2026-09-01",
            "lead": 1,
            "target_date": "2026-09-02",
            "fires": 1,
            "final_ap": 0.0,
            "persistence_ap": 0.0,
            "final_top5": 0.0,
            "persistence_top5": 0.0,
        },
        {
            "issue_date": "2026-09-02",
            "lead": 1,
            "target_date": "2026-09-03",
            "fires": 3,
            "final_ap": 1.0,
            "persistence_ap": 0.5,
            "final_top5": 1.0,
            "persistence_top5": 0.5,
        },
    ]
    for r in rows:
        r.update(final_brier=0.1, persistence_brier=0.2)
    card = scorecard(pd.DataFrame(rows), pd.Timestamp("2026-09-10"))
    lead1 = card["windows"]["since_launch"][0]
    assert lead1["final_top5"] == 0.75 and lead1["persistence_ap"] == 0.375
    assert len(card["daily"]) == 2


def test_scored_rows_carry_calibration_bins(tmp_path):
    fire = _fire(20)
    archive = tmp_path / "forecasts"
    archive.mkdir()
    p = np.random.default_rng(2).random((7, 50)).astype(np.float16)
    np.savez(archive / "2026-09-05.npz", final=p, persistence=p)
    s = score_archive(archive, fire, tmp_path / "scores.csv")
    n = sum(s[f"rel{i}_n"] for i in range(7))
    assert (n == 50).all()  # every cell lands in exactly one bin
    card = scorecard(s, pd.Timestamp("2026-09-30"))
    bins = card["calibration_since_launch"]["lead1"]
    assert sum(b["cells"] for b in bins) == 50


def test_score_archive_accepts_an_empty_scores_file(tmp_path):
    fire = _fire(20)
    (tmp_path / "forecasts").mkdir()
    csv = tmp_path / "scores.csv"
    csv.write_text("
")  # what an empty DataFrame's to_csv leaves behind
    assert score_archive(tmp_path / "forecasts", fire, csv).empty
