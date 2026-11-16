"""recourse fairness (RQ3). Burden is the distance to the
closest valid counterfactual (Sharma et al., 2020)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu
from statsmodels.stats.multitest import multipletests

from .groups import GROUP_DEFS, assign_groups


def _boot_ratio(a: np.ndarray, b: np.ndarray, n: int, rng: np.random.Generator):
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan
    ra = rng.choice(a, size=(n, len(a)), replace=True).mean(axis=1)
    rb = rng.choice(b, size=(n, len(b)), replace=True).mean(axis=1)
    ratios = ra / np.clip(rb, 1e-12, None)
    return float(np.percentile(ratios, 2.5)), float(np.percentile(ratios, 97.5))


def burden_by_group(st: pd.DataFrame, test: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    """Mean burden per group, burden gap ratio with bootstrap 95% CI,
    Mann-Whitney U tests (Holm-adjusted across all comparisons) and the
    no-recourse rate compared with Fisher's exact test."""
    rng = np.random.default_rng(seed)
    groups = assign_groups(test)
    rows = []
    for (method, week), g in st.groupby(["method", "week"]):
        gg = groups.loc[g["row"].to_numpy()].reset_index(drop=True)
        g = g.reset_index(drop=True)
        for attr, spec in GROUP_DEFS.items():
            lab = gg[attr]
            ia = g[(lab == "interest") & g["found"]]["burden"].to_numpy()
            rb = g[(lab == "reference") & g["found"]]["burden"].to_numpy()
            n_i, n_r = int((lab == "interest").sum()), int((lab == "reference").sum())
            nr_i = int(((lab == "interest") & ~g["found"]).sum())
            nr_r = int(((lab == "reference") & ~g["found"]).sum())
            rec = {"method": method, "week": week, "attribute": attr,
                   "group_interest": spec["labels"][0], "group_reference": spec["labels"][1],
                   "n_interest": n_i, "n_reference": n_r,
                   "burden_interest": ia.mean() if len(ia) else np.nan,
                   "burden_reference": rb.mean() if len(rb) else np.nan}
            rec["gap_ratio"] = rec["burden_interest"] / rec["burden_reference"] \
                if len(ia) and len(rb) and rec["burden_reference"] > 0 else np.nan
            rec["ratio_ci_low"], rec["ratio_ci_high"] = _boot_ratio(ia, rb, n_boot, rng)
            if len(ia) >= 2 and len(rb) >= 2:
                rec["mwu_p_two_sided"] = mannwhitneyu(ia, rb, alternative="two-sided").pvalue
                rec["mwu_p_greater"] = mannwhitneyu(ia, rb, alternative="greater").pvalue
            else:
                rec["mwu_p_two_sided"] = rec["mwu_p_greater"] = np.nan
            rec["no_recourse_rate_interest"] = nr_i / n_i if n_i else np.nan
            rec["no_recourse_rate_reference"] = nr_r / n_r if n_r else np.nan
            if n_i and n_r:
                rec["fisher_p_no_recourse"] = fisher_exact([[nr_i, n_i - nr_i], [nr_r, n_r - nr_r]])[1]
            else:
                rec["fisher_p_no_recourse"] = np.nan
            rows.append(rec)
    out = pd.DataFrame(rows)
    mask = out["mwu_p_two_sided"].notna()
    out["mwu_p_holm"] = np.nan
    if mask.any():
        out.loc[mask, "mwu_p_holm"] = multipletests(out.loc[mask, "mwu_p_two_sided"], method="holm")[1]
    out["mwu_p_greater_holm"] = np.nan
    if mask.any():
        out.loc[mask, "mwu_p_greater_holm"] = multipletests(out.loc[mask, "mwu_p_greater"], method="holm")[1]
    return out


def burden_by_risk_quintile(st: pd.DataFrame, test: pd.DataFrame, proba: pd.Series) -> pd.DataFrame:
    """Section 3.11.3: burden gap within quintiles of predicted risk, to
    separate unfair recourse from genuine differences in risk level."""
    groups = assign_groups(test)
    rows = []
    for (method, week), g in st.groupby(["method", "week"]):
        g = g[g["found"]].copy()
        if len(g) < 10:
            continue
        g["risk"] = proba.loc[g["row"]].to_numpy()
        g["quintile"] = pd.qcut(g["risk"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
        for attr in GROUP_DEFS:
            g["grp"] = groups.loc[g["row"], attr].to_numpy()
            for q, h in g.groupby("quintile", observed=True):
                a = h[h["grp"] == "interest"]["burden"]
                b = h[h["grp"] == "reference"]["burden"]
                rows.append({"method": method, "week": week, "attribute": attr, "risk_quintile": int(q),
                             "n_interest": len(a), "n_reference": len(b),
                             "burden_interest": a.mean() if len(a) else np.nan,
                             "burden_reference": b.mean() if len(b) else np.nan,
                             "gap_ratio": a.mean() / b.mean() if len(a) and len(b) and b.mean() > 0 else np.nan})
    return pd.DataFrame(rows)
