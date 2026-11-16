"""counterfactual generation with DiCE and Nearest Instance.

Both methods return, for every at-risk student, a list of candidate
counterfactuals (full feature vectors), each checked against the risk model
and its threshold tau. All candidates respect the constraints in
constraints.py; immutable features are always copied from the student.
"""
from __future__ import annotations

import contextlib
import io
import time
import warnings

import numpy as np
import pandas as pd

from .constraints import ACTIONABLE_COLS, Constraints
from .data_prep import FEATURES, TARGET


def _finalise(x: pd.Series, act: np.ndarray, cons: Constraints) -> np.ndarray:
    """Project actionable values into bounds and rebuild full feature rows
    with the student's immutable values."""
    act = cons.project(x, np.atleast_2d(act))
    full = np.tile(x[FEATURES].to_numpy(dtype=float), (len(act), 1))
    idx = [FEATURES.index(c) for c in ACTIONABLE_COLS]
    full[:, idx] = act
    # keep the score consistent with the number of submissions
    sub = FEATURES.index("n_submitted")
    score = FEATURES.index("mean_score")
    full[full[:, sub] == 0, score] = 0.0
    return full


def nearest_instance(model, tau: float, cons: Constraints, pool: pd.DataFrame,
                     queries: pd.DataFrame, batch: int = 256, log=print) -> list[dict]:
    """Section 3.9.3. pool = training students predicted not at risk."""
    pool_act = pool[ACTIONABLE_COLS].to_numpy(dtype=float)
    results = []
    t0 = time.time()
    for n, (i, x) in enumerate(queries.iterrows(), start=1):
        cand = cons.project(x, pool_act)
        cand = np.unique(cand, axis=0)
        x_act = x[ACTIONABLE_COLS].to_numpy(dtype=float)
        dist = cons.distance(x_act, cand)
        # drop the "no change" candidate
        keep = dist > 1e-9
        cand, dist = cand[keep], dist[keep]
        order = np.argsort(dist, kind="stable")
        found = None
        for start in range(0, len(order), batch):
            idx = order[start:start + batch]
            full = _finalise(x, cand[idx], cons)
            p = model.predict_proba(pd.DataFrame(full, columns=FEATURES))[:, 1]
            ok = np.where(p < tau)[0]
            if len(ok):
                j = ok[0]
                found = (full[j], float(p[j]))
                break
        if found is None:
            results.append({"row": i, "cfs": np.empty((0, len(FEATURES))), "proba": np.array([])})
        else:
            results.append({"row": i, "cfs": found[0][None, :], "proba": np.array([found[1]])})
        if n % 250 == 0:
            log(f"    Nearest Instance: {n}/{len(queries)} students ({time.time() - t0:.0f}s)")
    return results


def dice_counterfactuals(model, tau: float, cons: Constraints, train: pd.DataFrame,
                         queries: pd.DataFrame, k: int, seed: int, method: str = "random",
                         log=print) -> list[dict]:
    """Section 3.9.2. DiCE in a model-agnostic mode ("random" sampling by
    default, or "genetic"), one student at a time so that each student's own
    permitted ranges can be applied.

    "random" is the default because, in testing, the genetic mode's default
    initialisation searches near training students with identical immutable
    features and failed to find any counterfactual for most students."""
    import dice_ml

    data = dice_ml.Data(dataframe=train[FEATURES + [TARGET]].astype(float),
                        continuous_features=FEATURES, outcome_name=TARGET)
    m = dice_ml.Model(model=model, backend="sklearn")
    exp = dice_ml.Dice(data, m, method=method)
    extra = {"initialization": "random", "diversity_weight": 0.5} if method == "genetic" else {}
    stop = float(min(0.99, max(0.5, 1.0 - tau + 0.01)))

    results = []
    t0 = time.time()
    errors = 0
    for n, (i, x) in enumerate(queries.iterrows(), start=1):
        b = cons.bounds(x)
        vary = [c for c in ACTIONABLE_COLS if b[c][1] - b[c][0] > 1e-9]
        q = pd.DataFrame([x[FEATURES].astype(float)])
        cfs, proba = np.empty((0, len(FEATURES))), np.array([])
        if vary:
            try:
                np.random.seed(seed + n)
                with warnings.catch_warnings(), contextlib.redirect_stderr(io.StringIO()), \
                        contextlib.redirect_stdout(io.StringIO()):
                    warnings.simplefilter("ignore")
                    r = exp.generate_counterfactuals(
                        q, total_CFs=k, desired_class=0,
                        features_to_vary=vary,
                        permitted_range={c: [b[c][0], b[c][1]] for c in vary},
                        stopping_threshold=stop, verbose=False, **extra)
                cf_df = r.cf_examples_list[0].final_cfs_df
                if cf_df is not None and len(cf_df):
                    act = cf_df[ACTIONABLE_COLS].to_numpy(dtype=float)
                    cfs = _finalise(x, act, cons)
                    cfs = np.unique(cfs, axis=0)
                    proba = model.predict_proba(pd.DataFrame(cfs, columns=FEATURES))[:, 1]
            except Exception as e:  # DiCE raises an error when it finds nothing
                errors += 1
                if "No counterfactuals found" not in str(e) and errors <= 3:
                    log(f"    DiCE error for one student ({type(e).__name__}: {e})")
        results.append({"row": i, "cfs": cfs, "proba": proba})
        if n % 50 == 0:
            log(f"    DiCE: {n}/{len(queries)} students ({time.time() - t0:.0f}s)")
    if errors:
        log(f"    DiCE: {errors} students returned no counterfactuals (recorded as no recourse)")
    return results
