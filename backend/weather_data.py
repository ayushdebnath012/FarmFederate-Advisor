#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
weather_data.py — real agro-meteorological observations as a modality for FarmFederate.

Source: IMD-style "Daily Met data" workbook of the AMFU Kharagpur observatory
(station 42893, 22°34'N 87°23'E, 53 m). The sheet is a sequence of monthly
blocks, one per observation time (08:30 IST and 17:30 IST). By IMD convention
the morning observation carries the past-24h rainfall and the night minimum,
the evening observation carries the day maximum.

This module:
    - parses the workbook into one tidy record per calendar day (cached as CSV),
    - derives agronomic indices (rain accumulations, rainless spells, VPD,
      heat-day counts, a bucket-model soil-moisture proxy),
    - renders a day as a machine-readable "WEATHER: ..." line for the text encoder
      and as a "SENSORS: ..." line that replaces the old Gaussian sensor simulator,
    - encodes a day as a fixed-length feature vector for the model's weather branch,
    - assigns agromet-rule risk labels (hand-crafted priors, see weather_risk_labels).

Entry points:
    load_kgp_weather(xlsx_path=None, cache_csv=None) -> pd.DataFrame
    WeatherSampler(df)                                -> label-aware day sampler
    weather_summary_line(row) / sensor_line_from_weather(row, rng)
    weather_feature_vector(row) / weather_risk_labels(row)
"""

import os
import re
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

ISSUE_LABELS = ["water_stress", "nutrient_def", "pest_risk", "disease_risk", "heat_stress"]
LABEL_TO_ID = {k: i for i, k in enumerate(ISSUE_LABELS)}

STATION_NAME = "AMFU Kharagpur"
STATION_ID = "42893"

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
DEFAULT_XLSX = os.path.join(
    _REPO, "Weather data", "@ Daily Met data_KGP (42893) Obs Station 2024.xlsx"
)
DEFAULT_CACHE = os.path.join(_HERE, "data", "weather", "kgp_42893_daily.csv")

# ----------------------------------------------------------------------------
# Workbook parsing
# ----------------------------------------------------------------------------
_MONTHS = {
    "JAN": 1, "FEB": 2, "MAR": 3, "MARCH": 3, "APR": 4, "APRIL": 4, "MAY": 5,
    "JUN": 6, "JUNE": 6, "JUL": 7, "JULY": 7, "AUG": 8, "AUGUST": 8,
    "SEP": 9, "SEPT": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}
# e.g. "JAN 2024(05.30 P.M)", "FEB-2024(8.30A.M)", "JULY -2025 (5.30 P.M)"
_TITLE_RE = re.compile(
    r"^\s*([A-Z]+)\s*-?\s*(\d{4})\s*\(\s*(\d{1,2})[.:]?(\d{2})\s*([AP])\.?\s*M", re.I
)

# Fixed column layout of every monthly block (0 = day of month).
_COLS = {
    1: "db_temp", 2: "wb_temp", 3: "tmax", 4: "tmin", 5: "rain_mm",
    6: "dew_point", 7: "vapour_pressure", 8: "rh", 9: "wind_dir", 10: "wind_speed",
    11: "evaporation", 12: "soil_temp_5cm", 13: "soil_temp_10cm", 14: "soil_temp_20cm",
    15: "sunshine_hours", 16: "solar_rad_kj",
}
# Plausible physical ranges; anything outside (typos like "26.625.1") becomes NaN.
_RANGES = {
    "db_temp": (3, 50), "wb_temp": (3, 45), "tmax": (5, 50), "tmin": (3, 40),
    "rain_mm": (0, 500), "dew_point": (-5, 35), "vapour_pressure": (0, 60), "rh": (0, 100),
    "wind_speed": (0, 80), "evaporation": (0, 25), "soil_temp_5cm": (0, 60),
    "soil_temp_10cm": (0, 60), "soil_temp_20cm": (0, 60), "sunshine_hours": (0, 14),
    "solar_rad_kj": (0, 40000),
}
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
_DIR_RE = re.compile(r"^(N|NNE|NE|ENE|E|ESE|SE|SSE|S|SSW|SW|WSW|W|WNW|NW|NNW|CALM)$", re.I)


def _num(v, col: str) -> float:
    """Coerce a cell to float, tolerating hand-typed values like '23..0' or '29./83'."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    if isinstance(v, (int, float, np.integer, np.floating)):
        x = float(v)
    else:
        m = _NUM_RE.search(str(v))
        if not m:
            return np.nan
        x = float(m.group(0))
    lo, hi = _RANGES.get(col, (-np.inf, np.inf))
    return x if lo <= x <= hi else np.nan


def _wind_dir(v) -> Optional[str]:
    if isinstance(v, str) and _DIR_RE.match(v.strip()):
        return v.strip().upper()
    return None


def _parse_title(row: Sequence) -> Optional[tuple]:
    for v in row:
        if isinstance(v, str):
            m = _TITLE_RE.match(v)
            if m and m.group(1).upper() in _MONTHS:
                obs = "am" if m.group(5).upper() == "A" else "pm"
                return _MONTHS[m.group(1).upper()], int(m.group(2)), obs
    return None


def parse_met_workbook(xlsx_path: str) -> pd.DataFrame:
    """Return long-format observations: one row per (date, obs) with raw fields."""
    raw = pd.read_excel(xlsx_path, sheet_name=0, header=None)
    records = []
    block = None  # (month, year, obs)
    for _, row in raw.iterrows():
        vals = row.tolist()
        title = _parse_title(vals)
        if title:
            block = title
            continue
        if block is None:
            continue  # the untitled template rows at the top of the sheet
        c0 = vals[0]
        if isinstance(c0, str) and c0.strip().upper().startswith("MEAN"):
            block = None
            continue
        if not isinstance(c0, (int, float, np.integer, np.floating)) or np.isnan(float(c0)):
            continue
        day = int(c0)
        if not 1 <= day <= 31:
            continue
        month, year, obs = block
        try:
            date = pd.Timestamp(year=year, month=month, day=day)
        except ValueError:
            continue  # e.g. day 31 in a 30-day month (template rows)
        rec = {"date": date, "obs": obs}
        for j, name in _COLS.items():
            v = vals[j] if j < len(vals) else None
            rec[name] = _wind_dir(v) if name == "wind_dir" else _num(v, name)
        records.append(rec)
    if not records:
        raise ValueError(f"No monthly observation blocks found in {xlsx_path}")
    df = pd.DataFrame(records)
    # keep the first entry if a day was typed twice in a block
    df = df.drop_duplicates(subset=["date", "obs"], keep="first")
    return df.sort_values(["date", "obs"]).reset_index(drop=True)


def _first(a: pd.Series, b: pd.Series) -> pd.Series:
    return a.where(a.notna(), b)


def _merge_am_pm(long: pd.DataFrame) -> pd.DataFrame:
    am = long[long.obs == "am"].set_index("date")
    pm = long[long.obs == "pm"].set_index("date")
    idx = am.index.union(pm.index)
    am = am.reindex(idx)
    pm = pm.reindex(idx)

    d = pd.DataFrame(index=idx)
    d["tmax"] = _first(pm["tmax"], am["tmax"])
    d["tmax"] = _first(d["tmax"], pd.concat([am["db_temp"], pm["db_temp"]], axis=1).max(axis=1))
    d["tmin"] = _first(am["tmin"], pm["tmin"])
    d["rain_mm"] = _first(am["rain_mm"], pm["rain_mm"])
    d["db_am"], d["db_pm"] = am["db_temp"], pm["db_temp"]
    d["rh_am"], d["rh_pm"] = am["rh"], pm["rh"]
    d["dew_point"] = pd.concat([am["dew_point"], pm["dew_point"]], axis=1).mean(axis=1)
    d["vapour_pressure"] = pd.concat([am["vapour_pressure"], pm["vapour_pressure"]], axis=1).mean(axis=1)
    d["wind_speed"] = pd.concat([am["wind_speed"], pm["wind_speed"]], axis=1).mean(axis=1)
    d["wind_dir"] = _first(am["wind_dir"], pm["wind_dir"])
    # the PM sheet records evaporation as 0 (it is a 24h morning reading)
    d["evaporation"] = _first(am["evaporation"].where(am["evaporation"] > 0), pm["evaporation"].where(pm["evaporation"] > 0))
    d["soil_temp_5cm"] = _first(am["soil_temp_5cm"], pm["soil_temp_5cm"])
    d["soil_temp_10cm"] = _first(am["soil_temp_10cm"], pm["soil_temp_10cm"])
    d["sunshine_hours"] = _first(am["sunshine_hours"], pm["sunshine_hours"])
    d.index.name = "date"
    return d


def _vpd_kpa(t: pd.Series, rh: pd.Series) -> pd.Series:
    es = 0.6108 * np.exp(17.27 * t / (t + 237.3))
    return (es * (1.0 - rh / 100.0)).clip(lower=0.0)


def _rainless_run(rain: np.ndarray, wet_threshold: float = 2.5) -> np.ndarray:
    """Consecutive days (inclusive) with < wet_threshold mm; IMD 'rainy day' is >= 2.5 mm."""
    out = np.zeros(len(rain), dtype=np.int32)
    run = 0
    for i, r in enumerate(rain):
        run = 0 if r >= wet_threshold else run + 1
        out[i] = run
    return out


def _bucket_soil_moisture(rain: np.ndarray, et: np.ndarray, capacity: float = 100.0) -> np.ndarray:
    """Single-layer water balance; returns storage as a 0-100 fraction of capacity."""
    s = 0.5 * capacity
    out = np.zeros(len(rain), dtype=np.float32)
    for i in range(len(rain)):
        s = float(np.clip(s + rain[i] - et[i], 0.0, capacity))
        out[i] = 100.0 * s / capacity
    return out


def _recompute_means(d: pd.DataFrame) -> None:
    """tmean / rh_mean need both readings (a lone Tmax would inflate VPD); VPD follows."""
    d["tmean"] = 0.5 * (d["tmax"] + d["tmin"])
    d["tmean"] = _first(d["tmean"], 0.5 * (d["db_am"] + d["db_pm"]))
    d["rh_mean"] = 0.5 * (d["rh_am"] + d["rh_pm"])
    d["vpd_kpa"] = _vpd_kpa(d["tmean"], d["rh_mean"])


def add_derived_indices(d: pd.DataFrame) -> pd.DataFrame:
    """Fill the calendar, then add accumulations, spells, VPD and a soil-moisture proxy."""
    full = pd.date_range(d.index.min(), d.index.max(), freq="D")
    d = d.reindex(full)
    d.index.name = "date"

    _recompute_means(d)
    d["tmax_delta"] = d["tmax"].diff()

    rain = d["rain_mm"].fillna(0.0)
    d["rain_7d"] = rain.rolling(7, min_periods=1).sum()
    d["rain_30d"] = rain.rolling(30, min_periods=1).sum()
    d["rainless_days"] = _rainless_run(rain.to_numpy())
    d["heat_days_3"] = (d["tmax"] >= 36).astype(int).rolling(3, min_periods=1).sum()
    d["tmax_3d"] = d["tmax"].rolling(3, min_periods=1).mean()

    # crop ET ~ 0.7 x pan evaporation; fall back to a temperature proxy when the pan was not read
    et = (0.7 * d["evaporation"]).where(d["evaporation"].notna(), 0.15 * d["tmean"].fillna(27.0))
    d["soil_moisture_pct"] = _bucket_soil_moisture(rain.to_numpy(), et.to_numpy())

    doy = d.index.dayofyear.to_numpy()
    d["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    d["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    d["month"] = d.index.month
    d["observed"] = d[["tmax", "tmin", "rh_am", "rh_pm", "rain_mm"]].notna().any(axis=1)
    return d.reset_index()


def load_kgp_weather(xlsx_path: Optional[str] = None, cache_csv: Optional[str] = None,
                     use_cache: bool = True) -> pd.DataFrame:
    """Daily weather table. Parses the workbook once and caches a CSV next to the backend."""
    xlsx_path = xlsx_path or os.environ.get("FARMFED_WEATHER_XLSX") or DEFAULT_XLSX
    cache_csv = cache_csv or DEFAULT_CACHE
    if use_cache and os.path.exists(cache_csv) and (
        not os.path.exists(xlsx_path) or os.path.getmtime(cache_csv) >= os.path.getmtime(xlsx_path)
    ):
        return pd.read_csv(cache_csv, parse_dates=["date"])
    if not os.path.exists(xlsx_path):
        raise FileNotFoundError(f"weather workbook not found: {xlsx_path}")
    long = parse_met_workbook(xlsx_path)
    daily = add_derived_indices(_merge_am_pm(long))
    os.makedirs(os.path.dirname(cache_csv), exist_ok=True)
    daily.to_csv(cache_csv, index=False)
    return daily


# ----------------------------------------------------------------------------
# Model features
# ----------------------------------------------------------------------------
# (feature, climatological mean, scale) — fixed so train/serve encode identically.
WEATHER_FEATURE_SCALE = [
    ("tmax", 32.0, 5.0), ("tmin", 22.0, 5.0), ("tmean", 27.0, 5.0),
    ("rh_am", 80.0, 12.0), ("rh_pm", 60.0, 15.0), ("rain_mm", 4.0, 12.0),
    ("rain_7d", 30.0, 45.0), ("rain_30d", 120.0, 130.0), ("rainless_days", 10.0, 15.0),
    ("vpd_kpa", 1.2, 0.8), ("dew_point", 20.0, 6.0), ("wind_speed", 5.0, 3.0),
    ("evaporation", 4.0, 2.5), ("soil_temp_5cm", 27.0, 6.0), ("sunshine_hours", 5.0, 3.0),
    ("heat_days_3", 0.5, 1.0), ("soil_moisture_pct", 40.0, 25.0),
    ("doy_sin", 0.0, 0.7), ("doy_cos", 0.0, 0.7),
]
WEATHER_FEATURES = [f for f, _, _ in WEATHER_FEATURE_SCALE]
WEATHER_DIM = len(WEATHER_FEATURES)


IMPUTABLE = [
    "tmax", "tmin", "rh_am", "rh_pm", "rain_mm", "dew_point",
    "wind_speed", "evaporation", "soil_temp_5cm", "sunshine_hours",
]


def month_medians(daily: pd.DataFrame) -> pd.DataFrame:
    obs = daily[daily["observed"]] if "observed" in daily.columns else daily
    cols = [c for c in IMPUTABLE if c in obs.columns]
    return obs.groupby("month")[cols].median()


def impute_with_month_medians(daily: pd.DataFrame, medians: Optional[pd.DataFrame] = None):
    """
    The station stopped logging humidity/evaporation for long stretches of 2025-26.
    Fill those from the same-month climatology so rules and features stay usable, and
    return a boolean mask of which cells were imputed (the text renderer flags them with '~').
    """
    medians = month_medians(daily) if medians is None else medians
    filled = daily.copy()
    mask = pd.DataFrame(False, index=daily.index, columns=IMPUTABLE)
    for c in IMPUTABLE:
        if c not in filled.columns:
            continue
        fill = filled["month"].map(medians[c]) if c in medians.columns else np.nan
        mask[c] = filled[c].isna() & pd.notna(fill)
        filled[c] = filled[c].fillna(fill)
    # means and VPD follow from the (possibly imputed) component readings
    _recompute_means(filled)
    return filled, mask


def weather_feature_vector(row) -> np.ndarray:
    """z-scored feature vector for the model's weather branch; NaN -> 0 (climatological mean)."""
    out = np.zeros(WEATHER_DIM, dtype=np.float32)
    for i, (name, mu, sd) in enumerate(WEATHER_FEATURE_SCALE):
        x = row.get(name, np.nan)
        out[i] = 0.0 if pd.isna(x) else float(np.clip((float(x) - mu) / sd, -4.0, 4.0))
    return out


# ----------------------------------------------------------------------------
# Agromet-rule risk labels (hand-crafted priors in the spirit of IMD/AAS bulletins)
# ----------------------------------------------------------------------------
def _f(row, key, default=np.nan) -> float:
    v = row.get(key, default)
    return default if pd.isna(v) else float(v)


def weather_risk_labels(row) -> List[int]:
    """Label ids whose risk the day's weather alone supports (used as a prior for the corpus)."""
    tmax, tmean = _f(row, "tmax"), _f(row, "tmean")
    rh_am, rh_pm, rh_mean = _f(row, "rh_am"), _f(row, "rh_pm"), _f(row, "rh_mean")
    rain, rain7, rain30 = _f(row, "rain_mm", 0.0), _f(row, "rain_7d", 0.0), _f(row, "rain_30d", 0.0)
    dry, heat3 = _f(row, "rainless_days", 0.0), _f(row, "heat_days_3", 0.0)
    vpd, evap = _f(row, "vpd_kpa"), _f(row, "evaporation")

    labs = set()
    # heat: >= 38 C is the IMD heat-wave departure for plains; 3 consecutive >= 36 C days; hot-dry afternoons
    if tmax >= 38 or heat3 >= 3 or (tmax >= 36 and rh_pm <= 35):
        labs.add("heat_stress")
    # water: a rainless spell with high atmospheric demand during the warm season
    if dry >= 7 and rain7 < 2.5 and tmean >= 24 and (vpd >= 1.5 or evap >= 4.5):
        labs.add("water_stress")
    # fungal/bacterial: prolonged leaf wetness (humid mornings after rain) at moderate temperatures
    if (rh_am >= 85 and rain7 >= 10 and 18 <= tmean <= 32) or (rain >= 25 and rh_pm >= 75):
        labs.add("disease_risk")
    # sucking pests: warm, humid but settled weather a few days after rain
    if 24 <= tmean <= 34 and 55 <= rh_mean <= 85 and rain7 < 10 and dry >= 3:
        labs.add("pest_risk")
    # nutrient: nitrate leaching / waterlogging after very heavy rain
    if rain7 >= 150 or rain30 >= 450:
        labs.add("nutrient_def")
    return sorted(LABEL_TO_ID[x] for x in labs)


# ----------------------------------------------------------------------------
# Text rendering
# ----------------------------------------------------------------------------
def _fmt(x, nd=1, unit="", approx=False) -> str:
    if pd.isna(x):
        return "n/a"
    return f"{'~' if approx else ''}{float(x):.{nd}f}{unit}"


def weather_summary_line(row, imputed: Optional[Sequence[str]] = None) -> str:
    """
    Machine-readable daily summary for the text encoder. Values filled from the month
    climatology are prefixed with '~'. Deliberately avoids keyword-label trigger words
    ("dry", "hot", "moisture", ...) so weak labelling only reads the farmer's log.
    """
    imp = set(imputed or ())
    date = pd.Timestamp(row["date"]).strftime("%Y-%m-%d") if not pd.isna(row.get("date", np.nan)) else "unknown"
    wind = f"{row['wind_dir']} " if isinstance(row.get("wind_dir"), str) else ""

    def v(name, nd=1, unit=""):
        return _fmt(row.get(name), nd, unit, approx=name in imp)

    return (
        f"WEATHER ({STATION_NAME}/IMD {STATION_ID}, {date}): "
        f"Tmax={v('tmax', 1, '°C')}, Tmin={v('tmin', 1, '°C')}, "
        f"RH {v('rh_am', 0, '%')} morning / {v('rh_pm', 0, '%')} evening, "
        f"rain_24h={v('rain_mm', 1, 'mm')}, rain_7d={v('rain_7d', 1, 'mm')}, "
        f"rainless_days={int(_f(row, 'rainless_days', 0))}, dew_point={v('dew_point', 1, '°C')}, "
        f"VPD={v('vpd_kpa', 2, 'kPa')}, wind={wind}{v('wind_speed', 0, 'kmph')}, "
        f"pan_evap={v('evaporation', 1, 'mm')}, soil_temp_5cm={v('soil_temp_5cm', 1, '°C')}, "
        f"sunshine={v('sunshine_hours', 1, 'h')}."
    )


def sensor_line_from_weather(row, rng: Optional[np.random.Generator] = None,
                             imputed: Optional[Sequence[str]] = None) -> str:
    """
    A field-sensor summary consistent with the day's station weather, in the exact
    `SENSORS: ...` format the rest of the pipeline expects, followed by the WEATHER line.
    Soil pH is not observed by the station and stays synthetic.
    """
    rng = rng or np.random.default_rng()
    sm = 10.0 + 0.4 * _f(row, "soil_moisture_pct", 50.0) + rng.normal(0, 1.5)
    temp = _f(row, "db_pm", _f(row, "tmean", 29.0)) + rng.normal(0, 0.4)
    hum = _f(row, "rh_mean", 60.0) + rng.normal(0, 2.0)
    vpd = _f(row, "vpd_kpa", 1.4) + rng.normal(0, 0.05)
    rain = _f(row, "rain_mm", 0.0)
    delta = _f(row, "tmax_delta", 0.0)
    trend = "↑" if delta > 1.0 else ("↓" if delta < -1.0 else "→")
    sensors = (
        f"SENSORS: soil_moisture={np.clip(sm, 5, 60):.1f}%, soil_pH={np.clip(rng.normal(6.5, 0.4), 5.5, 7.5):.1f}, "
        f"temp={np.clip(temp, 5, 48):.1f}°C, humidity={np.clip(hum, 15, 100):.0f}%, "
        f"VPD={np.clip(vpd, 0.05, 5):.1f} kPa, rainfall_24h={rain:.1f}mm (trend: {trend})."
    )
    return f"{sensors}\n{weather_summary_line(row, imputed)}"


# ----------------------------------------------------------------------------
# Sampling
# ----------------------------------------------------------------------------
class WeatherSampler:
    """Draws observed days, optionally preferring days whose rule labels overlap a target set."""

    def __init__(self, daily: pd.DataFrame, seed: int = 123, observed_only: bool = True):
        df = daily[daily["observed"]] if observed_only and "observed" in daily.columns else daily
        self.df = df.reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError("weather table has no observed days")
        self.rng = np.random.default_rng(seed)
        self.month_medians = month_medians(self.df)
        self.df, mask = impute_with_month_medians(self.df, self.month_medians)
        self.imputed: List[List[str]] = [list(mask.columns[m]) for m in mask.to_numpy()]
        self.rule_labels: List[List[int]] = [weather_risk_labels(r) for _, r in self.df.iterrows()]
        self.by_label: Dict[int, np.ndarray] = {
            k: np.array([i for i, labs in enumerate(self.rule_labels) if k in labs], dtype=np.int64)
            for k in range(len(ISSUE_LABELS))
        }
        self.features = np.stack([weather_feature_vector(r) for _, r in self.df.iterrows()])

    def __len__(self) -> int:
        return len(self.df)

    def sample_index(self, prefer_labels: Optional[Sequence[int]] = None, p_match: float = 0.7) -> int:
        if prefer_labels and self.rng.random() < p_match:
            k = int(self.rng.choice(list(prefer_labels)))
            pool = self.by_label.get(k)
            if pool is not None and len(pool) > 0:
                return int(self.rng.choice(pool))
        return int(self.rng.integers(0, len(self.df)))

    def sample(self, prefer_labels: Optional[Sequence[int]] = None, p_match: float = 0.7):
        """Returns (row, feature_vector, rule_label_ids, imputed_field_names)."""
        i = self.sample_index(prefer_labels, p_match)
        return self.df.iloc[i], self.features[i], self.rule_labels[i], self.imputed[i]

    def sensor_text(self, prefer_labels: Optional[Sequence[int]] = None, p_match: float = 0.7):
        """Convenience for the corpus builders: (sensor+weather text, features, rule_label_ids)."""
        i = self.sample_index(prefer_labels, p_match)
        return self.sensor_text_for(i), self.features[i], self.rule_labels[i]

    def sensor_text_for(self, i: int) -> str:
        return sensor_line_from_weather(self.df.iloc[i], self.rng, self.imputed[i])

    def label_support(self) -> Dict[str, int]:
        return {ISSUE_LABELS[k]: int(len(v)) for k, v in self.by_label.items()}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Parse the KGP met workbook and print a summary.")
    ap.add_argument("--xlsx", default=None)
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()
    daily = load_kgp_weather(args.xlsx, use_cache=not args.no_cache)
    obs = daily[daily["observed"]]
    print(f"days={len(daily)} observed={len(obs)} span={daily['date'].min().date()}..{daily['date'].max().date()}")
    print(obs[["tmax", "tmin", "rh_am", "rh_pm", "rain_mm", "evaporation", "soil_temp_5cm"]].describe().round(1).to_string())
    s = WeatherSampler(daily)
    print("rule-label support (days):", s.label_support())
    row, feat, labs, imp = s.sample(prefer_labels=[LABEL_TO_ID["heat_stress"]])
    print(sensor_line_from_weather(row, s.rng, imp))
    print("labels:", [ISSUE_LABELS[k] for k in labs], "features:", np.round(feat, 2).tolist())
