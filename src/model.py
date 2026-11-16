"""Section 3.7: the XGBoost risk model, its tuning, decision threshold,
evaluation, a logistic regression reference model and prediction fairness."""
from __future__ import annotations

import numpy as np
import optuna
import pandas as pd
from fairlearn.metrics import equalized_odds_difference
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from xgboost import XGBClassifier

from .data_prep import CLICK_COLS, FEATURES, TARGET
from .groups import GROUP_DEFS, assign_groups

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _pos_weight(y: pd.Series) -> float:
    pos = float((y == 1).sum())
    return float((y == 0).sum()) / max(pos, 1.0)


def make_xgb(params: dict, y_train: pd.Series, seed: int) -> XGBClassifier:
    return XGBClassifier(
        **params,
        scale_pos_weight=_pos_weight(y_train),
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        n_jobs=-1,
        random_state=seed,
    )


def tune_xgb(X_tr, y_tr, X_val, y_val, n_trials: int, seed: int) -> dict:
    """Bayesian optimisation with Optuna, maximising validation PR-AUC."""
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        }
        m = make_xgb(params, y_tr, seed).fit(X_tr, y_tr)
        return average_precision_score(y_val, m.predict_proba(X_val)[:, 1])

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def best_f1_threshold(y_true, proba) -> float:
    """Threshold that maximises F1 for the at-risk class."""
    p, r, t = precision_recall_curve(y_true, proba)
    f1 = 2 * p * r / np.clip(p + r, 1e-12, None)
    if len(t) == 0:
        return 0.5
    return float(t[int(np.nanargmax(f1[:-1]))])


def evaluate(y_true, proba, tau: float) -> dict:
    pred = (proba >= tau).astype(int)
    return {
        "pr_auc": average_precision_score(y_true, proba),
        "roc_auc": roc_auc_score(y_true, proba),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "brier": brier_score_loss(y_true, proba),
        "threshold": tau,
        "n": len(y_true),
        "at_risk_rate": float(np.mean(y_true)),
        "flagged": int(pred.sum()),
    }


def train_logistic(X_tr, y_tr):
    """Reference model. Click counts are log-transformed and all features
    standardised (trees do not need this; logistic regression does)."""
    click_idx = [FEATURES.index(c) for c in CLICK_COLS + ["pre_start_clicks"]]

    def _log_clicks(X):
        X = np.asarray(X, dtype=float).copy()
        X[:, click_idx] = np.log1p(np.clip(X[:, click_idx], 0, None))
        return X

    pipe = make_pipeline(FunctionTransformer(_log_clicks), StandardScaler(),
                         LogisticRegression(max_iter=2000, class_weight="balanced"))
    return pipe.fit(X_tr, y_tr)


def prediction_fairness(test_df: pd.DataFrame, pred: np.ndarray) -> pd.DataFrame:
    """Equalised odds difference of the risk model for each attribute."""
    groups = assign_groups(test_df)
    rows = []
    for attr in GROUP_DEFS:
        mask = groups[attr].notna().to_numpy()
        if mask.sum() == 0 or groups.loc[mask, attr].nunique() < 2:
            continue
        eod = equalized_odds_difference(test_df.loc[mask, TARGET], pred[mask],
                                        sensitive_features=groups.loc[mask, attr])
        y = test_df.loc[mask, TARGET].to_numpy()
        g = groups.loc[mask, attr].to_numpy()
        p = pred[mask]
        rec = {k: recall_score(y[g == k], p[g == k], zero_division=0) for k in ("interest", "reference")}
        rows.append({"attribute": attr, "equalized_odds_difference": eod,
                     "recall_interest": rec["interest"], "recall_reference": rec["reference"],
                     "n_interest": int((g == "interest").sum()),
                     "n_reference": int((g == "reference").sum())})
    return pd.DataFrame(rows)


def cv_threshold(params: dict, X, y, folds: int, seed: int) -> float:
    """Section 3.12: threshold of the retrained model chosen from
    out-of-fold predictions."""
    oof = np.zeros(len(y))
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for tr, va in skf.split(X, y):
        m = make_xgb(params, y.iloc[tr], seed).fit(X.iloc[tr], y.iloc[tr])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
    return best_f1_threshold(y, oof)
