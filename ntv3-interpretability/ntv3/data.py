"""GUE prom_core_all loading and pilot-cohort sampling.

Same dataset the DNABERT-2 promoter notebooks and the earlier NTv2 port used
(``leannmlindsey/GUE``, config ``prom_core_all``), read directly via
``pandas.read_csv`` against the resolved file URLs -- no ``datasets``
library dependency, matching the convention already used by
``acceptor-interpretability/acceptors/data.py`` elsewhere in this repo.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ntv3.config import DATA_DIR, GUE_FILES, SEED


def _cached_path(split: str) -> Path:
    return DATA_DIR / f"gue_prom_core_all_{split}.csv"


def load_split(split: str) -> pd.DataFrame:
    if split not in GUE_FILES:
        raise ValueError(f"unknown split {split!r}, expected one of {list(GUE_FILES)}")
    cache = _cached_path(split)
    if cache.exists():
        df = pd.read_csv(cache)
    else:
        df = pd.read_csv(GUE_FILES[split])
        df.to_csv(cache, index=False)
    if not {"sequence", "label"}.issubset(df.columns):
        raise ValueError(f"unexpected GUE columns: {list(df.columns)}")
    return df


def load_all_splits() -> dict[str, pd.DataFrame]:
    return {split: load_split(split) for split in GUE_FILES}


def balanced_pilot(df: pd.DataFrame, per_class: int, *, seed: int = SEED) -> pd.DataFrame:
    """Sample a balanced pilot subset, matching the DNABERT-2 notebooks'
    per-class-with-fixed-seed sampling convention."""

    parts = []
    for label, group in df.groupby("label"):
        n = min(per_class, len(group))
        parts.append(group.sample(n=n, replace=False, random_state=seed))
    out = pd.concat(parts, ignore_index=True)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
