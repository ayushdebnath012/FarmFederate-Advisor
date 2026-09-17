#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
tea_weather_mapping.py — assign a station weather day to each tea crop.

WHAT THIS IS, PLAINLY
---------------------
The LEAF corpus carries no capture dates (the manuscript states they are not
documented; the delivered JPEGs are Roboflow exports with EXIF stripped). This
module therefore assigns each crop a weather day from the AMFU Kharagpur record
**synthetically**. The assignment is *not* provenance: it does not recover when a
photograph was taken, and it must never be described as observed pairing.

THE ONE INVARIANT
-----------------
The assignment is **label-independent**. `assign_index()` hashes only
(image_file, box_idx) — the crop's class is not an input and cannot be, by
construction. This matters: choosing a weather day whose agromet rules match the
crop's class would inject a weather->label correlation that does not exist in the
field, and any accuracy gain from it would be an artefact, not a finding. Because
the mapping is label-independent, the attached weather is *noise* with respect to
the label. The expected result of training on it is no change in accuracy within
run-to-run variance, and a fusion gate that stays near zero. A large gain is a
bug to investigate, not a result to report.

For the honest, leakage-free use of this station data, see
`tea_weather_advisory.py`, which applies it at inference where the date is known.

Entry points:
    assign_index(sample_id, n_days, seed)   -> deterministic day index
    build_manifest(sample_ids, ...)         -> DataFrame: crop -> date + features
    save_manifest / load_manifest           -> auditable CSV
    WeatherMap(...)                         -> lookup of feature vectors by sample_id
"""

import hashlib
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from weather_data import (
    load_kgp_weather,
    impute_with_month_medians,
    weather_feature_vector,
    WEATHER_FEATURES,
    WEATHER_DIM,
)

SampleId = Tuple[str, int]

DEFAULT_SEED = 42
_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MANIFEST = os.path.join(_HERE, "data", "weather", "tea_crop_weather_map.csv")


def observed_days(months: Optional[Sequence[int]] = None) -> pd.DataFrame:
    """
    Station days available for assignment, gap-filled from month medians.

    `months` optionally restricts the pool (e.g. a plausible field-work window).
    It is a property of the pool, not of any crop, so it cannot leak label
    information.
    """
    daily = load_kgp_weather()
    filled, _ = impute_with_month_medians(daily)
    pool = filled[filled["observed"]] if "observed" in filled.columns else filled
    if months:
        pool = pool[pool["month"].isin(list(months))]
    if len(pool) == 0:
        raise ValueError("no station days available for the requested window")
    return pool.reset_index(drop=True)


def assign_index(sample_id: SampleId, n_days: int, seed: int = DEFAULT_SEED) -> int:
    """
    Deterministic day index for a crop.

    Hashes only the crop's identity — image filename and box index. The class
    label is deliberately not a parameter, which is what keeps the mapping
    label-independent. Mirrors the blake2b idiom tea_train.py already uses for
    deterministic text-pool selection.
    """
    image_file, box_idx = str(sample_id[0]), int(sample_id[1])
    h = hashlib.blake2b(f"{seed}:{image_file}:{box_idx}".encode("utf-8"), digest_size=8)
    return int(h.hexdigest(), 16) % int(n_days)


def build_manifest(sample_ids: Iterable[SampleId], seed: int = DEFAULT_SEED,
                   months: Optional[Sequence[int]] = None) -> pd.DataFrame:
    """One row per crop: its assigned date and that day's station readings."""
    pool = observed_days(months)
    rows = []
    for sid in sample_ids:
        i = assign_index(sid, len(pool), seed)
        day = pool.iloc[i]
        rows.append({
            "image_file": str(sid[0]),
            "box_idx": int(sid[1]),
            "assigned_date": pd.Timestamp(day["date"]).strftime("%Y-%m-%d"),
            **{k: (None if pd.isna(day.get(k)) else float(day.get(k)))
               for k in ("tmax", "tmin", "tmean", "rh_am", "rh_pm", "rain_mm",
                         "rain_7d", "rainless_days", "sunshine_hours", "vpd_kpa")},
        })
    df = pd.DataFrame(rows)
    df.attrs["synthetic"] = True
    return df


def save_manifest(df: pd.DataFrame, path: str = DEFAULT_MANIFEST) -> str:
    """Persist the mapping so a run can be audited and reproduced exactly."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("# SYNTHETIC weather assignment for LEAF tea crops -- NOT capture provenance.\n")
        f.write("# Dates are hashed from (image_file, box_idx), independently of class label.\n")
        df.to_csv(f, index=False)
    return path


def load_manifest(path: str = DEFAULT_MANIFEST) -> pd.DataFrame:
    return pd.read_csv(path, comment="#")


class WeatherMap:
    """Feature-vector lookup by (image_file, box_idx), for a dataset to consume."""

    def __init__(self, sample_ids: Optional[Iterable[SampleId]] = None,
                 seed: int = DEFAULT_SEED, months: Optional[Sequence[int]] = None):
        self.pool = observed_days(months)
        self.seed = seed
        self._features = np.stack([weather_feature_vector(r) for _, r in self.pool.iterrows()])
        self.sample_ids = list(sample_ids) if sample_ids is not None else []

    @property
    def dim(self) -> int:
        return WEATHER_DIM

    def index_for(self, sample_id: SampleId) -> int:
        return assign_index(sample_id, len(self.pool), self.seed)

    def date_for(self, sample_id: SampleId) -> str:
        return pd.Timestamp(self.pool.iloc[self.index_for(sample_id)]["date"]).strftime("%Y-%m-%d")

    def features_for(self, sample_id: SampleId) -> np.ndarray:
        return self._features[self.index_for(sample_id)]

    def row_for(self, sample_id: SampleId) -> pd.Series:
        return self.pool.iloc[self.index_for(sample_id)]

    def manifest(self) -> pd.DataFrame:
        return build_manifest(self.sample_ids, self.seed)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Build the synthetic tea-crop -> weather-day map.")
    ap.add_argument("--annotations", default=os.path.join(os.path.dirname(_HERE), "tea_results",
                                                          "annotation", "annotations.csv"))
    ap.add_argument("--out", default=DEFAULT_MANIFEST)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--months", default="", help="comma-separated months to draw from (default: all)")
    args = ap.parse_args()

    ann = pd.read_csv(args.annotations)
    sids = list(zip(ann["image_file"].astype(str), ann["box_idx"].astype(int)))
    months = [int(m) for m in args.months.split(",") if m.strip()] or None
    man = build_manifest(sids, seed=args.seed, months=months)
    path = save_manifest(man, args.out)
    print(f"[map] {len(man)} crops -> {man['assigned_date'].nunique()} distinct station days")
    print(f"[map] wrote {path}")
    print(man.head(5).to_string(index=False))
