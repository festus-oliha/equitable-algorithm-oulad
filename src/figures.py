"""Figures from the results."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ATTR_LABEL = {"imd": "IMD band", "disability": "Disability", "age": "Age band", "gender": "Gender"}


def burden_ratio_plot(burden: pd.DataFrame, path: Path) -> None:
    """Burden gap ratio with 95% CI for each attribute, method and week.
    A ratio above 1 means the group of interest must change more."""
    b = burden.dropna(subset=["gap_ratio"]).copy()
    if b.empty:
        return
    b["label"] = b["attribute"].map(ATTR_LABEL) + "  (" + b["method"] + ", wk " + b["week"].astype(str) + ")"
    b = b.sort_values(["attribute", "method", "week"])
    fig, ax = plt.subplots(figsize=(8, 0.38 * len(b) + 1.2))
    y = np.arange(len(b))
    err = np.vstack([b["gap_ratio"] - b["ratio_ci_low"], b["ratio_ci_high"] - b["gap_ratio"]])
    ax.errorbar(b["gap_ratio"], y, xerr=np.nan_to_num(err), fmt="o", color="#2a6fbb", ecolor="#8aa9cf", capsize=3)
    ax.axvline(1, color="#888", linestyle="--", linewidth=1)
    ax.set_yticks(y, b["label"])
    ax.invert_yaxis()
    ax.set_xlabel("Burden gap ratio (group of interest / reference), 95% CI")
    ax.set_title("Recourse burden by group")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def feature_change_plot(freq: pd.DataFrame, path: Path) -> None:
    """Share of recommendations that change each feature, by method and week."""
    if freq.empty:
        return
    pv = freq.pivot_table(index="feature", columns=["method", "week"], values="pct_changed")
    pv = pv.loc[pv.mean(axis=1).sort_values().index]
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(pv) + 1.5))
    n = pv.shape[1]
    h = 0.8 / max(n, 1)
    y = np.arange(len(pv))
    for k, col in enumerate(pv.columns):
        ax.barh(y + k * h - 0.4 + h / 2, pv[col].values, height=h, label=f"{col[0]}, week {col[1]}")
    ax.set_yticks(y, pv.index)
    ax.set_xlabel("% of recommendations that change the feature")
    ax.set_title("What the recommendations ask students to change")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def robustness_plot(rob: pd.DataFrame, path: Path) -> None:
    """Share of recommendations still valid: bar = retrained model,
    line = range across the re-seeded models."""
    r = rob[rob["attribute"] == "all"].copy()
    if r.empty:
        return
    r["label"] = r["method"] + ", week " + r["week"].astype(str)
    labels = sorted(r["label"].unique())
    fig, ax = plt.subplots(figsize=(8, 0.75 * len(labels) + 1.4))
    y = np.arange(len(labels))
    retr = [r[(r["label"] == l) & (r["model"] == "retrained")]["robustness"].mean() for l in labels]
    seeds = [r[(r["label"] == l) & r["model"].str.startswith("seed")]["robustness"] for l in labels]
    ax.barh(y, retr, height=0.5, color="#2a6fbb", label="Retrained on newer data")
    for yi, s_ in zip(y, seeds):
        if len(s_):
            ax.plot([s_.min(), s_.max()], [yi, yi], color="#222", linewidth=1.5)
            ax.plot(s_, [yi] * len(s_), "o", color="#222", markersize=3.5)
    ax.plot([], [], "o-", color="#222", markersize=3.5, label="Re-seeded models (range)")
    for yi, v in zip(y, retr):
        ax.text(min(v + 0.01, 0.9), yi + 0.3, f"{v:.0%}", va="center", fontsize=8)
    ax.axvline(0.8, color="#888", linestyle="--", linewidth=1)
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of recommendations still valid (dashed line: 80%, the H4 threshold)")
    ax.set_title("Robustness of recommendations to retraining")
    ax.legend(fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.06), ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
