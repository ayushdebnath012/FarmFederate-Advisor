#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
tea_weather_advisory.py — real station weather as *advisory context* for the tea
(LEAF) system.

WHY THIS IS NOT A TRAINING FEATURE
----------------------------------
The LEAF corpus (200 photographs -> 371 crops from the IIT Kharagpur tea garden)
carries no capture dates: the manuscript states "Capture dates, camera details,
and independent expert adjudication are not documented in the supplied bundle,"
and the delivered JPEGs are Roboflow exports whose EXIF has been stripped. There
is therefore no key on which observed weather could honestly be joined to a crop.

Attaching weather to those crops anyway — in particular sampling a weather day
whose agromet rules *match* the crop's label, as datasets_loader does when it
generates the Advisor corpus — would manufacture a weather->label correlation
that does not exist in the field. The model would learn it and the reported
metrics would improve for a fabricated reason, defeating the leakage audit
(source-grouped splits, frozen train-fitted vocabulary, zero crop overlap).

So this module never touches training or evaluation of the 371 crops. It runs at
*inference* time, where the date is genuinely known: a scout photographs a leaf
today, the classifier returns a class from image/note evidence alone, and this
module adds the real observation for today's date as independent context. That
also upgrades the app's sensor cards, which the manuscript flags as
"demonstration content", to measured station data.

The risk rules below are hand-crafted agronomic priors for tea in eastern India,
not fitted parameters. They are reported as such and never combine with the
model's logits — `advise()` returns the two side by side and leaves the
weighting to the reader.

Entry points:
    weather_for_date(date)              -> the observed day (or None)
    class_risk(tea_class, row)          -> ("high"|"moderate"|"low", [drivers])
    advise(tea_class, date, confidence) -> full advisory payload
    risk_table(date)                    -> risk for all five classes
"""

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from weather_data import load_kgp_weather, impute_with_month_medians, STATION_NAME, STATION_ID

TEA_CLASSES = ["LEAF_BLIGHT", "LEAF_HOPPERS", "LEAF_RUST", "LOOPER_CATERPILLARS", "MOSQUITO_BUG"]

# Plain-language names for advisory text.
PEST_NAMES = {
    "LEAF_BLIGHT": "leaf blight",
    "LEAF_HOPPERS": "tea leaf hopper (Empoasca spp.)",
    "LEAF_RUST": "blister/rust-type leaf disease (Exobasidium vexans)",
    "LOOPER_CATERPILLARS": "looper caterpillar (Hyposidra talaca)",
    "MOSQUITO_BUG": "tea mosquito bug (Helopeltis theivora)",
}

_CACHE: Dict[str, pd.DataFrame] = {}


def _table() -> pd.DataFrame:
    """Daily station table with month-median gap filling, cached per process."""
    if "daily" not in _CACHE:
        daily = load_kgp_weather()
        filled, _ = impute_with_month_medians(daily)
        _CACHE["daily"] = filled
    return _CACHE["daily"]


def weather_for_date(date) -> Optional[pd.Series]:
    """The observed day, or None when the date falls outside the station record."""
    df = _table()
    ts = pd.Timestamp(date).normalize()
    hit = df[df["date"] == ts]
    if len(hit) == 0:
        return None
    return hit.iloc[0]


def latest_observation() -> pd.Series:
    """Most recent day the station actually recorded."""
    df = _table()
    obs = df[df["observed"]] if "observed" in df.columns else df
    return obs.iloc[-1]


def _f(row, key, default=np.nan) -> float:
    v = row.get(key, default)
    return default if pd.isna(v) else float(v)


# ---------------------------------------------------------------------------
# Agronomic risk rules (hand-crafted priors, tea in eastern India)
# ---------------------------------------------------------------------------
def class_risk(tea_class: str, row) -> Tuple[str, List[str]]:
    """
    Weather-only risk for one tea class on one observed day.

    Returns (level, drivers) where level is "high" | "moderate" | "low" and
    drivers name the observed quantities that triggered it. Weather alone never
    diagnoses — it says whether conditions favour the organism.
    """
    if tea_class not in TEA_CLASSES:
        raise ValueError(f"unknown tea class {tea_class!r}; expected one of {TEA_CLASSES}")

    tmean = _f(row, "tmean", 27.0)
    rh_am, rh_pm, rh_mean = _f(row, "rh_am", 80.0), _f(row, "rh_pm", 60.0), _f(row, "rh_mean", 70.0)
    rain, rain7 = _f(row, "rain_mm", 0.0), _f(row, "rain_7d", 0.0)
    dry = _f(row, "rainless_days", 0.0)
    sun = _f(row, "sunshine_hours", 5.0)

    drivers: List[str] = []

    # Blister blight: the classic tea rule — persistent leaf wetness with little
    # sunshine at mild temperatures. Sunshine hours are the dominant control.
    if tea_class == "LEAF_RUST":
        if rh_mean >= 83 and sun <= 3.5 and 15 <= tmean <= 26:
            drivers = [f"RH {rh_mean:.0f}%", f"sunshine {sun:.1f} h", f"mean temp {tmean:.1f}°C"]
            return "high", drivers
        if rh_mean >= 78 and sun <= 5.0 and tmean <= 29:
            drivers = [f"RH {rh_mean:.0f}%", f"sunshine {sun:.1f} h"]
            return "moderate", drivers
        return "low", [f"sunshine {sun:.1f} h", f"RH {rh_mean:.0f}%"]

    # Brown/grey blight: wet foliage plus warmth; follows sustained rainfall.
    if tea_class == "LEAF_BLIGHT":
        if rain7 >= 50 and rh_am >= 85 and 20 <= tmean <= 30:
            drivers = [f"7-day rain {rain7:.0f} mm", f"morning RH {rh_am:.0f}%", f"mean temp {tmean:.1f}°C"]
            return "high", drivers
        if rain7 >= 20 and rh_mean >= 75:
            drivers = [f"7-day rain {rain7:.0f} mm", f"RH {rh_mean:.0f}%"]
            return "moderate", drivers
        return "low", [f"7-day rain {rain7:.0f} mm"]

    # Tea jassid: warm and humid, but heavy rain washes nymphs off the flush.
    if tea_class == "LEAF_HOPPERS":
        if 25 <= tmean <= 32 and 70 <= rh_mean <= 92 and rain < 10 and dry >= 2:
            drivers = [f"mean temp {tmean:.1f}°C", f"RH {rh_mean:.0f}%", f"{int(dry)} d since rain"]
            return "high", drivers
        if 23 <= tmean <= 34 and rh_mean >= 60 and rain < 25:
            drivers = [f"mean temp {tmean:.1f}°C", f"RH {rh_mean:.0f}%"]
            return "moderate", drivers
        return "low", [f"24 h rain {rain:.1f} mm"]

    # Looper: warm settled spells with good sunshine favour larval development.
    if tea_class == "LOOPER_CATERPILLARS":
        if 25 <= tmean <= 32 and dry >= 3 and sun >= 5:
            drivers = [f"mean temp {tmean:.1f}°C", f"{int(dry)} d since rain", f"sunshine {sun:.1f} h"]
            return "high", drivers
        if tmean >= 23 and dry >= 2:
            drivers = [f"mean temp {tmean:.1f}°C", f"{int(dry)} d since rain"]
            return "moderate", drivers
        return "low", [f"{int(dry)} d since rain"]

    # Helopeltis: humid, mild-warm, overcast; suppressed by hard rain and glare.
    if 24 <= tmean <= 30 and rh_mean >= 78 and rain < 25 and sun <= 6.5:
        drivers = [f"mean temp {tmean:.1f}°C", f"RH {rh_mean:.0f}%", f"sunshine {sun:.1f} h"]
        return "high", drivers
    if 22 <= tmean <= 32 and rh_mean >= 68:
        drivers = [f"mean temp {tmean:.1f}°C", f"RH {rh_mean:.0f}%"]
        return "moderate", drivers
    return "low", [f"RH {rh_mean:.0f}%"]


def risk_table(date=None) -> Dict[str, Dict]:
    """Weather-only risk for all five classes on a date (default: latest observation)."""
    row = latest_observation() if date is None else weather_for_date(date)
    if row is None:
        return {}
    out = {}
    for k in TEA_CLASSES:
        level, drivers = class_risk(k, row)
        out[k] = {"risk": level, "drivers": drivers}
    return out


# ---------------------------------------------------------------------------
# Advisory text
# ---------------------------------------------------------------------------
_ACTIONS = {
    "LEAF_RUST": {
        "high": "Conditions strongly favour blister blight. Inspect young flush now; "
                "prioritise a protectant spray round and avoid prolonging leaf wetness.",
        "moderate": "Watch young flush on the shaded side of the section; keep the spray interval tight.",
        "low": "Blister blight pressure is low today; routine scouting is enough.",
    },
    "LEAF_BLIGHT": {
        "high": "Wet foliage and warmth favour blight spread. Remove badly affected leaves, "
                "improve drainage, and avoid overhead wetting.",
        "moderate": "Keep monitoring lower canopy; improve airflow where bushes are dense.",
        "low": "Blight pressure is low today; routine scouting is enough.",
    },
    "LEAF_HOPPERS": {
        "high": "Warm humid settled weather favours jassid build-up. Check flush undersides "
                "and consider an intervention if counts exceed threshold.",
        "moderate": "Sample the flush for nymphs; no action needed unless counts rise.",
        "low": "Recent rain suppresses hopper numbers; routine scouting is enough.",
    },
    "LOOPER_CATERPILLARS": {
        "high": "Warm settled weather favours looper development. Scout for early instars "
                "and treat patches before defoliation spreads.",
        "moderate": "Spot-check the section edges for feeding damage.",
        "low": "Looper pressure is low today; routine scouting is enough.",
    },
    "MOSQUITO_BUG": {
        "high": "Humid overcast warmth favours Helopeltis. Inspect tender shoots for feeding "
                "lesions and treat hotspots promptly.",
        "moderate": "Check tender shoots on the shaded side of the section.",
        "low": "Helopeltis pressure is low today; routine scouting is enough.",
    },
}


def observation_line(row) -> str:
    """One readable line of what the station actually measured that day."""
    def v(k, nd=1, unit=""):
        x = row.get(k)
        return "n/a" if pd.isna(x) else f"{float(x):.{nd}f}{unit}"
    date = pd.Timestamp(row["date"]).strftime("%Y-%m-%d")
    return (f"{STATION_NAME} (IMD {STATION_ID}) on {date}: "
            f"Tmax {v('tmax',1,'°C')}, Tmin {v('tmin',1,'°C')}, "
            f"RH {v('rh_am',0,'%')}/{v('rh_pm',0,'%')}, "
            f"rain 24h {v('rain_mm',1,' mm')}, 7-day rain {v('rain_7d',0,' mm')}, "
            f"sunshine {v('sunshine_hours',1,' h')}.")


def advise(tea_class: str, date=None, model_confidence: Optional[float] = None) -> Dict:
    """
    Advisory payload for a prediction made on `date`.

    The model's class (from image/note evidence) and the weather-only risk are
    returned side by side and deliberately NOT merged into one score: the weather
    never saw this bush, and the classifier never saw the weather.
    """
    row = latest_observation() if date is None else weather_for_date(date)
    if row is None:
        return {
            "tea_class": tea_class,
            "weather_available": False,
            "note": "No station observation for that date; advisory is based on the image/note only.",
        }
    level, drivers = class_risk(tea_class, row)
    agrees = level in ("high", "moderate")
    return {
        "tea_class": tea_class,
        "pest_name": PEST_NAMES.get(tea_class, tea_class),
        "model_confidence": model_confidence,
        "weather_available": True,
        "observation_date": pd.Timestamp(row["date"]).strftime("%Y-%m-%d"),
        "observation": observation_line(row),
        "weather_risk": level,
        "risk_drivers": drivers,
        "advice": _ACTIONS[tea_class][level],
        "context": (
            f"Station conditions are {'consistent with' if agrees else 'not particularly favourable for'} "
            f"{PEST_NAMES.get(tea_class, tea_class)} ({', '.join(drivers)})."
        ),
        "caveat": ("Weather risk is an independent agronomic prior from station observations; "
                   "it did not influence the model's prediction."),
    }


if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser(description="Weather-conditioned tea advisory (inference-time only).")
    ap.add_argument("--class", dest="tea_class", default=None, choices=TEA_CLASSES)
    ap.add_argument("--date", default=None, help="YYYY-MM-DD (default: latest station observation)")
    args = ap.parse_args()
    if args.tea_class:
        print(json.dumps(advise(args.tea_class, args.date), indent=2))
    else:
        row = latest_observation() if args.date is None else weather_for_date(args.date)
        print(observation_line(row))
        for k, v in risk_table(args.date).items():
            print(f"  {k:22s} {v['risk']:9s} ({', '.join(v['drivers'])})")
