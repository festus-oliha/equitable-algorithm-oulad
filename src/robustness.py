"""Section 3.12: robustness of recommendations to retraining (RQ4)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data_prep import FEATURES
from .groups import GROUP_DEFS, assign_groups


def robustness_table(best_vectors: dict[str, dict], test: pd.DataFrame,
                     alt_models: list[tuple[str, object, float]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """best_vectors: {method: {row: full feature vector}} for counterfactuals
    valid under the original model. alt_models: (name, model, threshold).
    Returns per-counterfactual results and a summary by method, model and group."""
    groups = assign_groups(test)
    rows = []
    for method, vecs in best_vectors.items():
        if not vecs:
            continue
        idx = list(vecs)
        X = pd.DataFrame(np.vstack([vecs[i] for i in idx]), columns=FEATURES)
        for name, model, tau in alt_models:
            p = model.predict_proba(X)[:, 1]
            for r, pi in zip(idx, p):
                rows.append({"method": method, "model": name, "row": r,
                             "proba_new": float(pi), "still_valid": bool(pi < tau),
                             **{a: groups.at[r, a] for a in GROUP_DEFS}})
    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail, detail

    summ = []
    for (method, name), g in detail.groupby(["method", "model"]):
        summ.append({"method": method, "model": name, "attribute": "all", "group": "all",
                     "n": len(g), "robustness": g["still_valid"].mean(),
                     "invalidation_rate": 1 - g["still_valid"].mean()})
        for attr in GROUP_DEFS:
            for lab in ("interest", "reference"):
                h = g[g[attr] == lab]
                if len(h):
                    summ.append({"method": method, "model": name, "attribute": attr, "group": lab,
                                 "n": len(h), "robustness": h["still_valid"].mean(),
                                 "invalidation_rate": 1 - h["still_valid"].mean()})
    return detail, pd.DataFrame(summ)
