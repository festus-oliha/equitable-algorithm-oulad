"""Run the full experiment, end to end.
Outputs are written to outputs/ (tables as CSV, figures as PNG, models,
run_log.txt and results_summary.md).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import figures
from .constraints import Constraints
from .counterfactuals import dice_counterfactuals, nearest_instance
from .data_prep import FEATURES, TARGET, build_snapshot, coverage_table, load_raw, split_by_presentation
from .fairness import burden_by_group, burden_by_risk_quintile
from .metrics import feature_change_frequency, per_student_table, quality_summary
from .model import (best_f1_threshold, cv_threshold, evaluate, make_xgb, prediction_fairness,
                    train_logistic, tune_xgb)
from .robustness import robustness_table
from .stats_tests import h1_feature_groups_by_week, h2_plausibility_sparsity, h3_imd, h4_invalidation
from .utils import PROJECT_ROOT, get_logger, load_config, out_dirs, save_json, set_seed


def _md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    if df is None or df.empty:
        return "_No data._\n"
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append("" if pd.isna(v) else floatfmt.format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Equitable algorithmic recourse experiment")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--quick", action="store_true", help="small, fast settings for testing")
    ap.add_argument("--data-dir", default=None, help="override data_dir in config.yaml")
    ap.add_argument("--weeks", type=int, nargs="*", default=None, help="override snapshot weeks, e.g. --weeks 4")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, quick=args.quick)
    if args.data_dir:
        p = Path(args.data_dir)
        cfg["data_dir"] = str(p if p.is_absolute() else PROJECT_ROOT / p)
    if args.weeks:
        cfg["weeks"] = args.weeks
    dirs = out_dirs(cfg)
    log = get_logger(dirs["root"] / "run_log.txt").info
    seed = cfg["seed"]
    set_seed(seed)
    t_start = time.time()
    log(f"Run started. Quick mode: {cfg['quick']}. Data: {cfg['data_dir']}")
    save_json(cfg, dirs["root"] / "config_used.json")

    raw = load_raw(cfg["data_dir"], max_day=max(cfg["weeks"]) * 7, log=log)

    model_rows, cover, pred_fair, students, burdens, quint, rob_sum, rob_det = [], [], [], [], [], [], [], []

    for week in cfg["weeks"]:
        log(f"===== Week {week} snapshot =====")
        snap = build_snapshot(raw, week)
        parts = split_by_presentation(snap, cfg["split"])
        train, val, test = parts["train"], parts["validation"], parts["test"]
        cover.append(coverage_table(parts, week))
        log(f"Registrations - train: {len(train):,}, validation: {len(val):,}, test: {len(test):,} "
            f"(removed {parts['_removed_overlap']} test rows of students also in training)")

        Xtr, ytr = train[FEATURES], train[TARGET]
        Xva, yva = val[FEATURES], val[TARGET]
        Xte, yte = test[FEATURES], test[TARGET]

        # ---- 3.7 Risk model ---------------------------------------------
        log(f"Tuning XGBoost with {cfg['model']['optuna_trials']} Optuna trials...")
        params = tune_xgb(Xtr, ytr, Xva, yva, cfg["model"]["optuna_trials"], seed)
        model = make_xgb(params, ytr, seed).fit(Xtr, ytr)
        tau = best_f1_threshold(yva, model.predict_proba(Xva)[:, 1])
        p_te = model.predict_proba(Xte)[:, 1]
        model_rows.append({"week": week, "model": "XGBoost", **evaluate(yte, p_te, tau)})
        lr = train_logistic(Xtr, ytr)
        tau_lr = best_f1_threshold(yva, lr.predict_proba(Xva)[:, 1])
        model_rows.append({"week": week, "model": "LogisticRegression",
                           **evaluate(yte, lr.predict_proba(Xte)[:, 1], tau_lr)})
        log(f"XGBoost test PR-AUC {model_rows[-2]['pr_auc']:.3f}, F1 {model_rows[-2]['f1']:.3f}, "
            f"threshold {tau:.3f}; flagged {int((p_te >= tau).sum())} of {len(test)} test students")
        model.save_model(dirs["models"] / f"xgb_week{week}.json")
        save_json({"params": params, "threshold": tau}, dirs["models"] / f"xgb_week{week}_params.json")

        pf = prediction_fairness(test, (p_te >= tau).astype(int))
        pf.insert(0, "week", week)
        pred_fair.append(pf)

        # ---- 3.8-3.9 Counterfactuals -------------------------------------
        cons = Constraints(train, cfg["constraints"]["upper_percentile"],
                           cfg["constraints"]["min_studied_credits"])
        p_tr = model.predict_proba(Xtr)[:, 1]
        pool = train[p_tr < tau]
        queries = test[p_te >= tau]
        log(f"Generating Nearest Instance counterfactuals for {len(queries)} at-risk students...")
        res_ni = nearest_instance(model, tau, cons, pool, queries,
                                  cfg["counterfactuals"]["nearest_batch"], log=log)

        max_dice = cfg["counterfactuals"]["dice_max_students"]
        q_dice = queries if (max_dice == 0 or len(queries) <= max_dice) else \
            queries.sample(n=max_dice, random_state=seed)
        log(f"Generating DiCE ({cfg['counterfactuals']['dice_method']}) counterfactuals for "
            f"{len(q_dice)} at-risk students (k={cfg['counterfactuals']['dice_k']}); this is the slowest step...")
        res_dc = dice_counterfactuals(model, tau, cons, train, q_dice, cfg["counterfactuals"]["dice_k"],
                                      seed, cfg["counterfactuals"]["dice_method"], log=log)

        st_ni, vec_ni = per_student_table(res_ni, queries, tau, cons, pool, "NearestInstance", week)
        st_dc, vec_dc = per_student_table(res_dc, q_dice, tau, cons, pool, "DiCE", week)
        st = pd.concat([st_ni, st_dc], ignore_index=True)
        st = st.merge(test[["group_gender", "group_age", "group_disability", "group_imd"]],
                      left_on="row", right_index=True, how="left")
        students.append(st)

        # ---- 3.11 Fairness ------------------------------------------------
        burdens.append(burden_by_group(st, test, cfg["stats"]["bootstrap_resamples"], seed))
        quint.append(burden_by_risk_quintile(st, test, pd.Series(p_te, index=test.index)))

        # ---- 3.12 Robustness ----------------------------------------------
        log("Training retrained and re-seeded models for the robustness test...")
        tv = pd.concat([train, val], ignore_index=True)
        retr = make_xgb(params, tv[TARGET], seed).fit(tv[FEATURES], tv[TARGET])
        tau_r = cv_threshold(params, tv[FEATURES], tv[TARGET], cfg["model"]["cv_folds_retrain"], seed)
        alt = [("retrained", retr, tau_r)]
        for s in cfg["stats"]["robustness_seeds"]:
            ms = make_xgb(params, ytr, s).fit(Xtr, ytr)
            alt.append((f"seed_{s}", ms, best_f1_threshold(yva, ms.predict_proba(Xva)[:, 1])))
        det, summ = robustness_table({"NearestInstance": vec_ni, "DiCE": vec_dc}, test, alt)
        if not det.empty:
            det.insert(0, "week", week)
            summ.insert(0, "week", week)
            rob_det.append(det)
            rob_sum.append(summ)
        log(f"Week {week} done ({time.time() - t_start:.0f}s elapsed)")

    # ---- Combine and save -----------------------------------------------------
    T = dirs["tables"]
    models_df = pd.DataFrame(model_rows)
    st_all = pd.concat(students, ignore_index=True)
    burden_df = pd.concat(burdens, ignore_index=True)
    quint_df = pd.concat(quint, ignore_index=True) if quint else pd.DataFrame()
    rob_sum_df = pd.concat(rob_sum, ignore_index=True) if rob_sum else pd.DataFrame()
    rob_det_df = pd.concat(rob_det, ignore_index=True) if rob_det else pd.DataFrame()
    qual = quality_summary(st_all)
    freq = feature_change_frequency(st_all)

    pd.concat(cover).to_csv(T / "t01_partition_coverage.csv", index=False)
    models_df.to_csv(T / "t02_model_performance.csv", index=False)
    pd.concat(pred_fair).to_csv(T / "t03_prediction_fairness.csv", index=False)
    qual.to_csv(T / "t04_recourse_quality.csv", index=False)
    freq.to_csv(T / "t05_feature_change_frequency.csv", index=False)
    burden_df.to_csv(T / "t06_burden_by_group.csv", index=False)
    quint_df.to_csv(T / "t07_burden_by_risk_quintile.csv", index=False)
    rob_sum_df.to_csv(T / "t08_robustness_summary.csv", index=False)
    st_all.to_csv(T / "per_student_recourse.csv", index=False)
    rob_det_df.to_csv(T / "per_counterfactual_robustness.csv", index=False)

    alpha = cfg["stats"]["alpha"]
    h1 = h1_feature_groups_by_week(st_all)
    h2 = h2_plausibility_sparsity(st_all)
    h3 = h3_imd(burden_df, alpha)
    h4 = h4_invalidation(rob_sum_df, cfg["stats"]["h4_threshold"], rob_det_df) if not rob_det_df.empty else pd.DataFrame()
    h3.to_csv(T / "t09_h3_imd_burden.csv", index=False)
    h4.to_csv(T / "t10_h4_invalidation.csv", index=False)
    save_json({"H1": h1, "H2": h2}, T / "h1_h2_results.json")

    F = dirs["figures"]
    figures.burden_ratio_plot(burden_df, F / "fig_burden_gap_ratio.png")
    figures.feature_change_plot(freq, F / "fig_feature_change_frequency.png")
    if not rob_sum_df.empty:
        figures.robustness_plot(rob_sum_df, F / "fig_robustness.png")

    # ---- Summary for Chapter Four --------------------------------------------
    md = [f"# Results summary\n\nGenerated by src/run_all.py. Quick mode: **{cfg['quick']}**. "
          f"Runtime: {(time.time() - t_start) / 60:.1f} minutes.\n"]
    if cfg["quick"]:
        md.append("> Quick mode uses reduced settings. Do not report these numbers in the thesis.\n")
    md.append("## Risk model performance (Table t02)\n" + _md_table(
        models_df[["week", "model", "pr_auc", "roc_auc", "precision", "recall", "f1", "brier", "threshold", "n", "flagged"]]))
    md.append("## Prediction fairness (Table t03)\n" + _md_table(pd.concat(pred_fair)))
    md.append("## Coverage and quality, RQ1-RQ2 (Table t04)\n" + _md_table(qual))
    md.append("## Burden by group, RQ3 (Table t06)\n" + _md_table(burden_df[[
        "method", "week", "attribute", "n_interest", "n_reference", "burden_interest", "burden_reference",
        "gap_ratio", "ratio_ci_low", "ratio_ci_high", "mwu_p_holm", "no_recourse_rate_interest",
        "no_recourse_rate_reference", "fisher_p_no_recourse"]]))
    if not rob_sum_df.empty:
        md.append("## Robustness, RQ4 (Table t08, overall rows)\n" + _md_table(
            rob_sum_df[rob_sum_df["attribute"] == "all"][["week", "method", "model", "n", "robustness", "invalidation_rate"]]))
    md.append("## Hypotheses\n")
    import json
    _j = lambda o: json.dumps(o, indent=2, default=lambda v: float(v) if hasattr(v, "__float__") else str(v))
    md.append("### H1 (feature groups by week)\n```\n" + _j(h1) + "\n```\n")
    md.append("### H2 (plausibility and sparsity)\n```\n" + _j(h2) + "\n```\n")
    md.append("### H3 (IMD burden)\n" + _md_table(h3))
    md.append("### H4 (invalidation after retraining)\n" + _md_table(h4))
    (dirs["root"] / "results_summary.md").write_text("\n".join(md), encoding="utf-8")

    log(f"All done in {(time.time() - t_start) / 60:.1f} minutes. "
        f"See {dirs['root'] / 'results_summary.md'} and the tables/ and figures/ folders.")


if __name__ == "__main__":
    main()
