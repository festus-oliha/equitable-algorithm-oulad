"""Shared helpers: configuration loading, logging, output paths."""
from __future__ import annotations

import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path = "config.yaml", quick: bool = False) -> dict:
    """Load config.yaml. With quick=True, shrink the expensive settings so a
    full run finishes in a few minutes (used for testing the setup)."""
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for key in ("data_dir", "output_dir"):
        p = Path(cfg[key])
        cfg[key] = str(p if p.is_absolute() else PROJECT_ROOT / p)
    if quick:
        cfg["model"]["optuna_trials"] = 5
        cfg["counterfactuals"]["dice_max_students"] = 40
        cfg["counterfactuals"]["dice_k"] = 3
        cfg["stats"]["bootstrap_resamples"] = 200
        cfg["stats"]["robustness_seeds"] = [1, 2]
    cfg["quick"] = quick
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def out_dirs(cfg: dict) -> dict[str, Path]:
    root = Path(cfg["output_dir"])
    dirs = {
        "root": root,
        "tables": root / "tables",
        "figures": root / "figures",
        "models": root / "models",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def get_logger(log_file: Path | None = None) -> logging.Logger:
    logger = logging.getLogger("recourse")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_file is not None:
        fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def save_json(obj, path: Path) -> None:
    def _default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_default)
