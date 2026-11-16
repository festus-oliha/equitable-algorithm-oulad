"""Section 3.13 (Table 3.5): hypothesis tests H1-H4."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2_contingency, wilcoxon

from .constraints import ACTIONABLE_COLS, FEATURE_GROUP


def h1_feature_groups_by_week(st: pd.DataFrame) -> dict:
    """H1: chi-square test of independence between snapshot week and the
    feature group (engagement vs assessment) of the changes recommended."""
    out = {}
    for method, g in st[st["found"]].groupby("method"):
        table = {}
        for week, h in g.groupby("week"):
            counts = {"engagement": 0, "assessment": 0}
            for c in ACTIONABLE_COLS:
                grp = FEATURE_GROUP[c]
                if grp in counts:
                    counts[grp] += int(h[f"chg_{c}"].sum())
            table[week] = counts
        ct = pd.DataFrame(table).T
        res = {"contingency": ct.to_dict()}
        if ct.shape[0] >= 2 and (ct.values > 0).all():
            chi2, p, dof, _ = chi2_contingency(ct.values)
            share = (ct["assessment"] / ct.sum(axis=1)).to_dict()
            res.update({"chi2": chi2, "p_value": p, "dof": dof,
                        "assessment_share_by_week": share})
        else:
            res["note"] = "Not enough data for a chi-square test"
        out[method] = res
    return out


def h2_plausibility_sparsity(st: pd.DataFrame) -> dict:
    """H2: paired Wilcoxon signed-rank tests on students who have a
    counterfactual from both methods."""
    out = {}
    for week, g in st[st["found"]].groupby("week"):
        p = g.pivot_table(index="row", columns="method", values=["plausibility", "sparsity"])
        p = p.dropna()
        if len(p) < 10 or "DiCE" not in p["plausibility"] or "NearestInstance" not in p["plausibility"]:
            out[week] = {"note": "Fewer than 10 students with both methods"}
            continue
        ni_pl, dc_pl = p["plausibility"]["NearestInstance"], p["plausibility"]["DiCE"]
        ni_sp, dc_sp = p["sparsity"]["NearestInstance"], p["sparsity"]["DiCE"]
        res = {"n_pairs": len(p),
               "plausibility_median_NI": ni_pl.median(), "plausibility_median_DiCE": dc_pl.median(),
               "sparsity_mean_NI": ni_sp.mean(), "sparsity_mean_DiCE": dc_sp.mean()}
        try:
            res["p_NI_more_plausible"] = wilcoxon(ni_pl, dc_pl, alternative="less").pvalue
        except ValueError:
            res["p_NI_more_plausible"] = np.nan
        try:
            res["p_NI_less_sparse"] = wilcoxon(ni_sp, dc_sp, alternative="greater").pvalue
        except ValueError:
            res["p_NI_less_sparse"] = np.nan
        out[week] = res
    return out


def h3_imd(burden: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """H3: most deprived IMD group faces larger burden (one-sided,
    Holm-adjusted) and gap-ratio CI above 1."""
    b = burden[burden["attribute"] == "imd"].copy()
    b["supported"] = (b["mwu_p_greater_holm"] < alpha) & (b["ratio_ci_low"] > 1)
    return b[["method", "week", "n_interest", "n_reference", "burden_interest", "burden_reference",
              "gap_ratio", "ratio_ci_low", "ratio_ci_high", "mwu_p_greater", "mwu_p_greater_holm", "supported"]]


def h4_invalidation(rob_summary: pd.DataFrame, threshold: float, rob_detail: pd.DataFrame) -> pd.DataFrame:
    """H4: one-sided exact binomial test that the invalidation rate after
    retraining exceeds the threshold (default 0.20)."""
    rows = []
    d = rob_detail[rob_detail["model"] == "retrained"]
    keys = ["week", "method"] if "week" in d.columns else ["method"]
    for key, g in d.groupby(keys):
        key = key if isinstance(key, tuple) else (key,)
        k = int((~g["still_valid"]).sum())
        n = len(g)
        rows.append({**dict(zip(keys, key)), "n": n, "invalidated": k, "rate": k / n if n else np.nan,
                     "threshold": threshold,
                     "p_value": binomtest(k, n, threshold, alternative="greater").pvalue if n else np.nan})
    return pd.DataFrame(rows)
