"""load OULAD, build week-k snapshots,
engineer features and split the data by presentation.

Each row of a snapshot is one student registration on one
module-presentation, described only by data dated before the cutoff day.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Feature definitions (Table 3.2). The actionability of each feature is set
# in constraints.py (Table 3.3).
# --------------------------------------------------------------------------
CLICK_TYPES = {
    "forumng": "clicks_forum",
    "quiz": "clicks_quiz",
    "oucontent": "clicks_content",
    "resource": "clicks_resource",
    "homepage": "clicks_homepage",
}
CLICK_COLS = list(CLICK_TYPES.values()) + ["clicks_other"]

ENGAGEMENT = CLICK_COLS + ["active_days", "days_since_last", "pre_start_clicks"]
ASSESSMENT = ["n_submitted", "mean_score", "n_late", "n_due"]
STUDY_LOAD = ["studied_credits"]
DEMOGRAPHIC = ["gender", "age_band", "disability", "imd_band",
               "highest_education", "region", "num_of_prev_attempts", "module"]

FEATURES = ENGAGEMENT + ASSESSMENT + STUDY_LOAD + DEMOGRAPHIC
TARGET = "at_risk"
KEYS = ["code_module", "code_presentation", "id_student"]

# Ordinal encodings for categorical attributes. Tree models do not need
# one-hot encoding, and integer codes keep counterfactual search simple.
AGE_MAP = {"0-35": 0, "35-55": 1, "55<=": 2}
EDU_MAP = {"No Formal quals": 0, "Lower Than A Level": 1,
           "A Level or Equivalent": 2, "HE Qualification": 3,
           "Post Graduate Qualification": 4}


def _imd_code(v) -> int:
    """'0-10%' -> 0, '10-20' -> 1, ..., '90-100%' -> 9, missing -> -1.
    (OULAD writes the 10-20 band without a % sign.)"""
    if pd.isna(v):
        return -1
    try:
        return int(str(v).split("-")[0]) // 10
    except ValueError:
        return -1


def load_raw(data_dir: str | Path, max_day: int, log=print) -> dict[str, pd.DataFrame]:
    """Read the 7 OULAD CSV files. studentVle is large (10.6M rows), so only
    the needed columns are read and rows after the last cutoff are dropped."""
    d = Path(data_dir)
    required = ["courses.csv", "assessments.csv", "vle.csv", "studentInfo.csv",
                "studentRegistration.csv", "studentAssessment.csv", "studentVle.csv"]
    missing = [f for f in required if not (d / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"These OULAD files are missing from {d}: {missing}. "
            "Download OULAD and unzip the 7 CSV files into this folder.")

    raw = {}
    raw["courses"] = pd.read_csv(d / "courses.csv")
    raw["assessments"] = pd.read_csv(d / "assessments.csv")
    raw["vle"] = pd.read_csv(d / "vle.csv", usecols=["id_site", "activity_type"])
    raw["studentInfo"] = pd.read_csv(d / "studentInfo.csv")
    raw["studentRegistration"] = pd.read_csv(d / "studentRegistration.csv")
    raw["studentAssessment"] = pd.read_csv(d / "studentAssessment.csv")

    log("Reading studentVle.csv (this is the large file and may take a minute)...")
    chunks = []
    for chunk in pd.read_csv(
        d / "studentVle.csv",
        usecols=["code_module", "code_presentation", "id_student", "id_site", "date", "sum_click"],
        dtype={"code_module": "category", "code_presentation": "category",
               "id_student": "int64", "id_site": "int64", "date": "int32", "sum_click": "int32"},
        chunksize=2_000_000,
    ):
        chunks.append(chunk[chunk["date"] < max_day])
    svle = pd.concat(chunks, ignore_index=True)
    for c in ("code_module", "code_presentation"):
        svle[c] = svle[c].astype(str)
    raw["studentVle"] = svle
    log(f"Loaded studentVle rows before day {max_day}: {len(svle):,}")
    return raw


def build_snapshot(raw: dict[str, pd.DataFrame], week: int) -> pd.DataFrame:
    """Build the feature table for one snapshot week (Section 3.5.2).
    Uses only events dated before day = week * 7."""
    cutoff = week * 7

    info = raw["studentInfo"].copy()
    reg = raw["studentRegistration"][KEYS + ["date_unregistration"]]
    df = info.merge(reg, on=KEYS, how="left")

    # Exclude students who had already withdrawn before the snapshot.
    left_early = df["date_unregistration"].notna() & (df["date_unregistration"] < cutoff)
    df = df.loc[~left_early].copy()

    # ---- Engagement features -------------------------------------------
    svle = raw["studentVle"]
    svle = svle[svle["date"] < cutoff].merge(raw["vle"], on="id_site", how="left")
    svle["ctype"] = svle["activity_type"].map(CLICK_TYPES).fillna("clicks_other")

    by_type = (svle.pivot_table(index=KEYS, columns="ctype", values="sum_click",
                                aggfunc="sum", fill_value=0)
               .reindex(columns=CLICK_COLS, fill_value=0).reset_index())
    pre = (svle[svle["date"] < 0].groupby(KEYS)["sum_click"].sum()
           .rename("pre_start_clicks").reset_index())
    days = svle.groupby(KEYS)["date"].agg(active_days="nunique", last_day="max").reset_index()

    df = df.merge(by_type, on=KEYS, how="left").merge(pre, on=KEYS, how="left") \
           .merge(days, on=KEYS, how="left")
    df[CLICK_COLS + ["pre_start_clicks", "active_days"]] = \
        df[CLICK_COLS + ["pre_start_clicks", "active_days"]].fillna(0)
    # days since last activity; students with no activity get the full window
    df["days_since_last"] = np.where(df["last_day"].notna(),
                                     (cutoff - 1) - df["last_day"], cutoff)
    df["days_since_last"] = df["days_since_last"].clip(lower=0)
    df = df.drop(columns=["last_day"])

    # ---- Assessment features -------------------------------------------
    asm = raw["assessments"]
    due = asm[asm["date"].notna() & (asm["date"] < cutoff)]
    n_due = due.groupby(["code_module", "code_presentation"]).size() \
               .rename("n_due").reset_index()
    sa = raw["studentAssessment"].merge(
        asm[["id_assessment", "code_module", "code_presentation", "date"]],
        on="id_assessment", how="left")
    sa = sa[sa["date_submitted"] < cutoff]
    sa["late"] = (sa["date"].notna() & (sa["date_submitted"] > sa["date"])).astype(int)
    agg = sa.groupby(KEYS).agg(n_submitted=("id_assessment", "count"),
                              mean_score=("score", "mean"),
                              n_late=("late", "sum")).reset_index()
    df = df.merge(agg, on=KEYS, how="left").merge(
        n_due, on=["code_module", "code_presentation"], how="left")
    df[["n_submitted", "n_late", "n_due"]] = df[["n_submitted", "n_late", "n_due"]].fillna(0)
    df["mean_score"] = df["mean_score"].fillna(0.0)
    # a student cannot have submitted more assessments than were due
    df["n_due"] = np.maximum(df["n_due"], df["n_submitted"])

    # ---- Demographic and history features (immutable) ---------------------
    df["group_gender"] = df["gender"]
    df["group_age"] = df["age_band"]
    df["group_disability"] = df["disability"]
    df["group_imd"] = df["imd_band"].fillna("unknown")

    df["gender"] = (df["gender"] == "F").astype(int)
    df["age_band"] = df["age_band"].map(AGE_MAP).fillna(0).astype(int)
    df["disability"] = (df["disability"] == "Y").astype(int)
    df["imd_band"] = df["imd_band"].map(_imd_code).astype(int)
    df["highest_education"] = df["highest_education"].map(EDU_MAP).fillna(-1).astype(int)
    regions = sorted(raw["studentInfo"]["region"].dropna().unique())
    df["region"] = df["region"].map({r: i for i, r in enumerate(regions)}).fillna(-1).astype(int)
    modules = sorted(raw["courses"]["code_module"].unique())
    df["module"] = df["code_module"].map({m: i for i, m in enumerate(modules)}).astype(int)

    # ---- Target (Section 3.5.1) -----------------------------------------
    df[TARGET] = df["final_result"].isin(["Fail", "Withdrawn"]).astype(int)

    for c in FEATURES:
        df[c] = df[c].astype(float)
    df["week"] = week
    cols = KEYS + ["week", "final_result", "group_gender", "group_age",
                   "group_disability", "group_imd"] + FEATURES + [TARGET]
    return df[cols].reset_index(drop=True)


def split_by_presentation(df: pd.DataFrame, split_cfg: dict) -> dict[str, pd.DataFrame]:
    """Section 3.6: split by presentation. Test registrations of students who
    also appear in the training set are removed."""
    parts = {name: df[df["code_presentation"].isin(split_cfg[name])].reset_index(drop=True)
             for name in ("train", "validation", "test")}
    train_ids = set(parts["train"]["id_student"])
    before = len(parts["test"])
    parts["test"] = parts["test"][~parts["test"]["id_student"].isin(train_ids)].reset_index(drop=True)
    parts["_removed_overlap"] = before - len(parts["test"])
    return parts


def coverage_table(parts: dict[str, pd.DataFrame], week: int) -> pd.DataFrame:
    """Module coverage of each partition (reported per Section 3.6)."""
    rows = []
    for name in ("train", "validation", "test"):
        p = parts[name]
        for (mod, pres), g in p.groupby(["code_module", "code_presentation"]):
            rows.append({"week": week, "partition": name, "module": mod,
                         "presentation": pres, "registrations": len(g),
                         "at_risk_rate": round(g[TARGET].mean(), 3)})
    return pd.DataFrame(rows)
