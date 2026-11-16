"""demographic groups compared in the fairness
analysis. Each attribute maps a student to 'interest', 'reference' or
missing (students in neither group, e.g. middle IMD bands)."""
from __future__ import annotations

import numpy as np
import pandas as pd

GROUP_DEFS = {
    "imd": {"column": "group_imd",
            "interest": {"0-10%", "10-20", "10-20%", "20-30%"},
            "reference": {"70-80%", "80-90%", "90-100%"},
            "labels": ("Most deprived (0-30%)", "Least deprived (70-100%)")},
    "disability": {"column": "group_disability",
                   "interest": {"Y"}, "reference": {"N"},
                   "labels": ("Declared disability", "No declared disability")},
    "age": {"column": "group_age",
            "interest": {"35-55", "55<="}, "reference": {"0-35"},
            "labels": ("35 and over", "Under 35")},
    "gender": {"column": "group_gender",
               "interest": {"F"}, "reference": {"M"},
               "labels": ("Female", "Male")},
}


def assign_groups(meta: pd.DataFrame) -> pd.DataFrame:
    """Return a frame with one column per attribute holding
    'interest' / 'reference' / NaN."""
    out = pd.DataFrame(index=meta.index)
    for attr, spec in GROUP_DEFS.items():
        col = meta[spec["column"]].astype(str)
        g = pd.Series(np.nan, index=meta.index, dtype=object)
        g[col.isin(spec["interest"])] = "interest"
        g[col.isin(spec["reference"])] = "reference"
        out[attr] = g
    return out
