"""actionability constraints and
the MAD-scaled L1 distance used by both counterfactual methods.

Recommendations are read as targets for the student's behaviour by the next
checkpoint. Cumulative activity can only go up, the gap since the last login
can only go down, and anything that already happened in the past (clicks
before the course started, late submissions) or that describes the student's
background is held fixed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data_prep import CLICK_COLS, FEATURES

# direction of allowed change for each actionable feature
ACTIONABLE: dict[str, str] = {
    **{c: "increase" for c in CLICK_COLS},
    "active_days": "increase",
    "days_since_last": "decrease",
    "n_submitted": "increase",
    "mean_score": "any",
    "studied_credits": "decrease",
}
ACTIONABLE_COLS = list(ACTIONABLE)
IMMUTABLE_COLS = [c for c in FEATURES if c not in ACTIONABLE]
INTEGER_COLS = [c for c in ACTIONABLE_COLS if c != "mean_score"]

FEATURE_GROUP = {**{c: "engagement" for c in CLICK_COLS + ["active_days", "days_since_last"]},
                 "n_submitted": "assessment", "mean_score": "assessment",
                 "studied_credits": "study_load"}


class Constraints:
    """Holds training-set statistics needed for bounds and distances."""

    def __init__(self, train: pd.DataFrame, upper_percentile: float = 99,
                 min_credits: float = 30):
        self.cap = {c: float(np.percentile(train[c], upper_percentile)) for c in ACTIONABLE_COLS}
        self.min_credits = min_credits
        mad = {}
        for c in ACTIONABLE_COLS:
            v = train[c].to_numpy(dtype=float)
            m = float(np.median(np.abs(v - np.median(v))))
            if m <= 0:
                m = float(np.std(v))
            mad[c] = m if m > 0 else 1.0
        self.mad = pd.Series(mad)[ACTIONABLE_COLS]
        self.mad_vec = self.mad.to_numpy()

    # ---- bounds ------------------------------------------------------------
    def bounds(self, x: pd.Series) -> dict[str, tuple[float, float]]:
        """Permitted [low, high] range for each actionable feature of one student."""
        b = {}
        for c, direction in ACTIONABLE.items():
            v = float(x[c])
            if c == "n_submitted":
                b[c] = (v, max(v, float(x["n_due"])))
            elif c == "mean_score":
                b[c] = (0.0, 100.0)
            elif c == "studied_credits":
                b[c] = (min(v, self.min_credits), v)
            elif direction == "increase":
                b[c] = (v, max(v, self.cap[c]))
            elif direction == "decrease":
                b[c] = (0.0, v)
        return b

    def project(self, x: pd.Series, cand: np.ndarray) -> np.ndarray:
        """Clip candidate actionable values (n x len(ACTIONABLE_COLS)) into the
        student's permitted ranges, and round integer features."""
        b = self.bounds(x)
        lo = np.array([b[c][0] for c in ACTIONABLE_COLS])
        hi = np.array([b[c][1] for c in ACTIONABLE_COLS])
        out = np.clip(cand, lo, hi)
        int_idx = [ACTIONABLE_COLS.index(c) for c in INTEGER_COLS]
        out[:, int_idx] = np.round(out[:, int_idx])
        out = np.clip(out, lo, hi)
        return out

    # ---- distance (Section 3.9.1) -------------------------------------------
    def distance(self, x_act: np.ndarray, cf_act: np.ndarray) -> np.ndarray:
        """MAD-scaled L1 distance over actionable features.
        x_act: (d,) or (n, d); cf_act: (n, d)."""
        return np.abs(cf_act - x_act).__truediv__(self.mad_vec).sum(axis=1)

    def scale(self, act: np.ndarray) -> np.ndarray:
        return act / self.mad_vec

    def is_feasible(self, x: pd.Series, cf: pd.Series, tol: float = 1e-6) -> bool:
        """Check a counterfactual respects every constraint."""
        if any(abs(float(cf[c]) - float(x[c])) > tol for c in IMMUTABLE_COLS):
            return False
        for c, (lo, hi) in self.bounds(x).items():
            if not (lo - tol <= float(cf[c]) <= hi + tol):
                return False
        return True
