import datetime as dt

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

from titikapi.data.cube import FireMatrix, recent_fire
from titikapi.models.baselines import Climatology, Persistence
from titikapi.models.evaluate import evaluate, issue_days
from titikapi.models.metrics import average_precision, brier, topk_coverage


def _fire_matrix(on: np.ndarray, start="2019-01-01") -> FireMatrix:
    t, n = on.shape
    return FireMatrix(
        time=pd.date_range(start, periods=t, freq="D"),
        on=on.astype(bool),
        mask=np.ones((1, n), bool),
        province=np.ones(n, np.uint8),
    )


def test_average_precision_matches_sklearn():
    rng = np.random.default_rng(0)
    y = rng.random(500) < 0.1
    p = rng.random(500) + y * 0.3
    assert average_precision(y, p) == pytest.approx(average_precision_score(y, p))


def test_brier():
    assert brier(np.array([1, 0]), np.array([1.0, 0.0])) == 0.0
    assert brier(np.array([1, 0]), np.array([0.5, 0.5])) == pytest.approx(0.25)


def test_topk_coverage_counts_fires_in_top_cells_each_day():
    y = np.array([[1, 0, 0, 0], [0, 0, 1, 1]], bool)
    p = np.array([[0.9, 0.1, 0.2, 0.3], [0.1, 0.2, 0.8, 0.0]])
    # k = 1 cell per day: day 0 catches 1/1, day 1 catches 1/2 -> 2/3 overall
    assert topk_coverage(y, p, frac=0.25) == pytest.approx(2 / 3)
    # only count fires inside a slice, but rank among all cells
    only = np.array([[False] * 4, [False, False, False, True]])
    assert topk_coverage(y, p, frac=0.25, count=only) == pytest.approx(0.0)


def test_recent_fire_window():
    on = np.array([1, 0, 0, 0, 1, 0], bool)[:, None]
    assert recent_fire(on, window=3)[:, 0].tolist() == [True, True, True, False, True, True]


def test_issue_days_keep_targets_inside_split():
    fm = _fire_matrix(np.zeros((60, 2)))
    split = (dt.date(2019, 1, 20), dt.date(2019, 2, 10))
    idx = issue_days(fm, split, max_lead=7, history=14)
    assert fm.time[idx[0]] == pd.Timestamp("2019-01-20")
    assert fm.time[idx[-1]] + pd.Timedelta(days=7) == pd.Timestamp("2019-02-10")


def test_climatology_uses_training_days_only():
    rng = np.random.default_rng(1)
    on = rng.random((400, 5)) < 0.05
    train = np.arange(0, 300)
    a = Climatology().fit(_fire_matrix(on), train)
    on2 = on.copy()
    on2[300:] = True  # change only the future
    b = Climatology().fit(_fire_matrix(on2), train)
    issue = np.arange(310, 320)
    np.testing.assert_array_equal(a.predict(issue, 1), b.predict(issue, 1))


def test_persistence_learns_conditional_rates():
    # a cell that is on fire every other day: fire today -> no fire tomorrow, and vice versa
    on = np.tile(np.array([1, 0], bool), 100)[:, None]
    fm = _fire_matrix(on)
    rec = recent_fire(on, 14)
    model = Persistence().fit(fm, rec, np.arange(14, 150), leads=(1, 2))
    p1 = model.predict(np.array([20, 21]), 1)  # day 20 has fire, day 21 doesn't
    assert p1[0, 0] == pytest.approx(0.0, abs=1e-3)
    assert p1[1, 0] == pytest.approx(1.0, abs=1e-3)


def test_evaluate_reports_new_ignition_slice():
    rng = np.random.default_rng(2)
    on = rng.random((120, 30)) < 0.05
    fm = _fire_matrix(on)
    rec = recent_fire(on, 14)
    idx = np.arange(20, 100)
    res = evaluate("random", lambda i, lead: rng.random((len(i), 30)), fm, rec, idx, leads=(1, 2))
    assert set(res["slice"]) == {"all", "new_ignition"}
    assert set(res["lead"]) == {1, 2}
    assert {"ap", "brier", "top5_coverage", "base_rate", "n_pos"} <= set(res.columns)
