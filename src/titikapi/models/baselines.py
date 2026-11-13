"""No-learning-required reference forecasts that every model must beat.

Both are fitted on training days only and predict P(fire in cell on day t + lead) from what is
known on issue day t.
"""

from __future__ import annotations

import numpy as np

from titikapi.data.cube import FireMatrix


class Climatology:
    """How often each cell burned in the target's calendar month, in training years.

    Rates are shrunk toward the region-wide rate for that month (`alpha` pseudo-days), so cells
    with few observations don't get extreme values.
    """

    def __init__(self, alpha: float = 30.0):
        self.alpha = alpha

    def fit(self, fm: FireMatrix, train_idx: np.ndarray) -> Climatology:
        self.time = fm.time
        months = fm.time.month[train_idx]
        self.rate = np.zeros((13, fm.n_cells), dtype=np.float32)
        for m in range(1, 13):
            sel = train_idx[months == m]
            k = fm.on[sel].sum(axis=0)
            n = len(sel)
            regional = k.sum() / max(n * fm.n_cells, 1)
            self.rate[m] = (k + self.alpha * regional) / (n + self.alpha)
        return self

    def predict(self, issue_idx: np.ndarray, lead: int) -> np.ndarray:
        target_months = self.time.month[issue_idx + lead]
        return self.rate[target_months]


class Persistence:
    """P(fire at t + lead | the cell's fire state at t), learned from training days.

    States: fire today / fire in the past 14 days but not today / no fire in 14 days.
    Within a state, cells are ordered by climatology (a tiny tie-breaker that leaves
    the probabilities effectively unchanged) so ranking metrics are well defined.
    """

    STATES = ("fire_today", "recent_not_today", "none")

    def fit(
        self,
        fm: FireMatrix,
        recent: np.ndarray,
        train_idx: np.ndarray,
        leads=range(1, 8),
        climatology: Climatology | None = None,
    ) -> Persistence:
        self.on, self.recent = fm.on, recent
        self.climatology = climatology
        self.rate = {}
        for lead in leads:
            t = train_idx[train_idx + lead <= train_idx.max()]
            state = self._state(t)
            nxt = fm.on[t + lead]
            self.rate[lead] = np.array(
                [nxt[state == s].mean() if (state == s).any() else 0.0 for s in range(3)],
                dtype=np.float32,
            )
        return self

    def _state(self, t: np.ndarray) -> np.ndarray:
        today = self.on[t]
        return np.where(today, 0, np.where(self.recent[t], 1, 2)).astype(np.int8)

    def predict(self, issue_idx: np.ndarray, lead: int) -> np.ndarray:
        p = self.rate[lead][self._state(issue_idx)]
        if self.climatology is not None:
            p = p + 1e-6 * self.climatology.predict(issue_idx, lead)
        return p
