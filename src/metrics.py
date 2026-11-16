"""Section 3.10: coverage and quality metrics (RQ1, RQ2)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from .constraints import ACTIONABLE_COLS, FEATURE_GROUP, Constraints
from .data_prep import FEATURES

ACT_IDX = [FEATURES.index(c) for c in ACTIONABLE_COLS]


def per_student_table(results: list[dict], queries: pd.DataFrame, tau: float,
                      cons: Constraints, pool: pd.DataFrame, method: str, week: int):
    """One row per at-risk student, plus the best counterfactual vectors
    (needed later for the robustness test)."""
    nn = NearestNeighbors(n_neighbors=1, metric="manhattan")
    nn.fit(cons.scale(pool[ACTIONABLE_COLS].to_numpy(dtype=float)))

    rows, best_vectors = [], {}
    for r in results:
        x = queries.loc[r["row"]]
        x_act = x[ACTIONABLE_COLS].to_numpy(dtype=float)
        cfs, proba = r["cfs"], r["proba"]
        valid = proba < tau
        rec = {"row": r["row"], "method": method, "week": week,
               "n_generated": int(len(cfs)), "n_valid": int(valid.sum()),
               "found": bool(valid.any()), "burden": np.nan, "sparsity": np.nan,
               "plausibility": np.nan, "diversity": np.nan, "proba_best": np.nan}
        for c in ACTIONABLE_COLS:
            rec[f"chg_{c}"] = np.nan
        if valid.any():
            vcf = cfs[valid]
            act = vcf[:, ACT_IDX]
            d = cons.distance(x_act, act)
            j = int(np.argmin(d))
            best = act[j]
            changed = np.abs(best - x_act) > 1e-6
            rec["burden"] = float(d[j])
            rec["sparsity"] = int(changed.sum())
            rec["plausibility"] = float(nn.kneighbors(cons.scale(best[None, :]))[0][0, 0])
            rec["proba_best"] = float(proba[valid][j])
            for c, ch in zip(ACTIONABLE_COLS, changed):
                rec[f"chg_{c}"] = int(ch)
            if len(act) >= 2:
                sa = cons.scale(act)
                pd_ = [np.abs(sa[a] - sa[b]).sum() for a in range(len(sa)) for b in range(a + 1, len(sa))]
                rec["diversity"] = float(np.mean(pd_))
            best_vectors[r["row"]] = vcf[j]
        rows.append(rec)
    return pd.DataFrame(rows), best_vectors


def quality_summary(st: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (method, week), g in st.groupby(["method", "week"]):
        f = g[g["found"]]
        out.append({
            "method": method, "week": week, "at_risk_students": len(g),
            "coverage": g["found"].mean(),
            "validity": g["n_valid"].sum() / max(g["n_generated"].sum(), 1),
            "proximity_mean": f["burden"].mean(), "proximity_median": f["burden"].median(),
            "sparsity_mean": f["sparsity"].mean(),
            "plausibility_mean": f["plausibility"].mean(),
            "diversity_mean": f["diversity"].mean() if method == "DiCE" else np.nan,
        })
    return pd.DataFrame(out)


def feature_change_frequency(st: pd.DataFrame) -> pd.DataFrame:
    """Percentage of best counterfactuals that change each actionable feature."""
    out = []
    for (method, week), g in st.groupby(["method", "week"]):
        f = g[g["found"]]
        for c in ACTIONABLE_COLS:
            out.append({"method": method, "week": week, "feature": c,
                        "feature_group": FEATURE_GROUP[c],
                        "pct_changed": 100 * f[f"chg_{c}"].mean() if len(f) else np.nan,
                        "n_changed": int(f[f"chg_{c}"].sum()) if len(f) else 0})
    return pd.DataFrame(out)
