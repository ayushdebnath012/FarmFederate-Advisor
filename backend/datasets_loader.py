#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
datasets_loader.py — text + image dataset builders for FarmFederate.

Text sources (downloaded via Hugging Face `datasets`):
    - CGIAR/gardian-ai-ready-docs
    - argilla/farming
    - ag_news (agri-filtered)
    - synthetic local "LocalMini" agri log-style data
    - "weather": logs grounded in real AMFU Kharagpur (IMD 42893) daily observations

Weather (see weather_data.py): when the KGP workbook is available, every fused
sample carries a real observed day (SENSORS line derived from it + a WEATHER line),
a numeric feature vector for the model's weather branch (column "weather"), and
agromet-rule risk labels merged into the weak labels. Control with
FARMFED_WEATHER_MODE = full | text-only | none and FARMFED_WEATHER_XLSX = <path>.

Image sources (Hugging Face datasets, all auto-downloaded if available):
    - BrandonFors/Plant-Diseases-PlantVillage-Dataset  (PlantVillage mirror)
    - Saon110/bd-crop-vegetable-plant-disease-dataset
    - timm/plant-pathology-2021
    - uqtwei2/PlantWild

If an image dataset fails to load (timeout, 404, permission), it is skipped.
If *all* fail, training falls back to a dummy gray image in memory.

Main entry points:

    build_text_corpus_mix(...)
    load_stress_image_datasets_hf(...)
"""

import os
import re
import time
import random
import hashlib
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd

from PIL import Image

try:
    from datasets import load_dataset, DownloadConfig, concatenate_datasets, DatasetDict
    HAS_DATASETS = True
except Exception:
    HAS_DATASETS = False

from weather_data import (
    WeatherSampler,
    load_kgp_weather,
    weather_risk_labels,
    WEATHER_DIM,
    WEATHER_FEATURES,
)

# ----------------- core labels -----------------
ISSUE_LABELS = ["water_stress", "nutrient_def", "pest_risk", "disease_risk", "heat_stress"]
LABEL_TO_ID = {k: i for i, k in enumerate(ISSUE_LABELS)}
NUM_LABELS = len(ISSUE_LABELS)

SEED = 123
random.seed(SEED)
np.random.seed(SEED)

# ----------------- text utils -----------------
def _norm(txt: str) -> str:
    return re.sub(r"\s+", " ", str(txt)).strip()


def _ascii_ratio(s: str) -> float:
    if not s:
        return 0.0
    return sum(1 for ch in s if ord(ch) < 128) / max(1, len(s))


def _lang_ok(s: str) -> bool:
    return _ascii_ratio(s) >= 0.6


# ------------- weak labels & ag context -------------
KW = {
    "water": [
        "dry", "wilting", "wilt", "parched", "drought", "moisture", "irrigation",
        "canopy stress", "water stress", "droop", "cracking soil", "hard crust",
        "soil moisture low"
    ],
    "nutrient": [
        "nitrogen", "phosphorus", "potassium", "npk", "fertilizer", "fertiliser",
        "chlorosis", "chlorotic", "interveinal", "leaf color chart", "lcc", "spad",
        "low spad", "older leaves yellowing", "old leaves yellowing",
        "necrotic margin", "micronutrient", "deficiency"
    ],
    "pest": [
        "pest", "aphid", "whitefly", "borer", "hopper", "weevil", "caterpillar",
        "larvae", "thrips", "mites", "trap", "sticky residue", "honeydew",
        "chewed", "webbing", "frass", "insect"
    ],
    "disease": [
        "blight", "rust", "mildew", "smut", "rot", "leaf spot", "necrosis",
        "pathogen", "fungal", "bacterial", "viral", "lesion", "mosaic",
        "wilt disease", "canker", "powdery mildew", "downy"
    ],
    "heat": [
        "heatwave", "hot", "scorch", "sunburn", "thermal stress",
        "high temperature", "blistering", "desiccation", "sun scorch", "leaf burn",
        "heat stress"
    ],
}

AG_CONTEXT = re.compile(
    r"\b(agri|agricultur|farm|farmer|field|crop|soil|irrigat|harvest|yield|"
    r"paddy|rice|wheat|maize|corn|cotton|soy|orchard|greenhouse|seedling|"
    r"fertiliz|manure|compost|pest|fung|blight|leaf|canopy|mulch|drip|sprinkler|"
    r"nursery|plantation|horticul)\b",
    re.I,
)


def is_ag_context(s: str) -> bool:
    return bool(AG_CONTEXT.search(s))


_MACHINE_LINE = re.compile(r"^\s*(SENSORS|WEATHER)\b[^\n]*$", re.I | re.M)


def strip_machine_lines(text: str) -> str:
    """Drop SENSORS:/WEATHER(...) lines so keyword labelling only reads the human log."""
    return _MACHINE_LINE.sub("", str(text)).strip()


def weak_labels(text: str) -> List[int]:
    # keyword rules must not fire on the sensor/weather lines ("soil_moisture=..." used to
    # match the "moisture" keyword and tag every fused row as water_stress)
    t = strip_machine_lines(text).lower()
    if not is_ag_context(t):
        return []
    labs = set()
    if any(k in t for k in KW["water"]):
        labs.add("water_stress")
    if any(k in t for k in KW["nutrient"]):
        strong_n = any(
            x in t
            for x in [
                "chlorosis", "chlorotic", "interveinal", "npk", "nitrogen",
                "potassium", "leaf color chart", "lcc", "low spad", "spad"
            ]
        )
        qualified_yellow = (
            "yellowing" in t
            and ("older leaves" in t or "old leaves" in t)
        )
        if strong_n or qualified_yellow:
            labs.add("nutrient_def")
    if any(k in t for k in KW["pest"]):
        labs.add("pest_risk")
    if any(k in t for k in KW["disease"]):
        labs.add("disease_risk")
    if any(k in t for k in KW["heat"]):
        labs.add("heat_stress")
    return [LABEL_TO_ID[x] for x in sorted(labs)]


# ----------------- weather modality (real station data) -----------------
WEATHER_MODES = ("full", "text-only", "none")
_WEATHER: Dict[str, object] = {"mode": None, "sampler": None}


def configure_weather(mode: Optional[str] = None, xlsx_path: Optional[str] = None) -> Optional[WeatherSampler]:
    """
    mode: "full" (WEATHER text + numeric features + rule labels), "text-only"
    (no numeric features), or "none" (legacy Gaussian sensors). Defaults to
    FARMFED_WEATHER_MODE, else "full" when the workbook/cache is found.
    """
    mode = (mode or os.environ.get("FARMFED_WEATHER_MODE") or "full").strip().lower()
    if mode not in WEATHER_MODES:
        raise ValueError(f"FARMFED_WEATHER_MODE must be one of {WEATHER_MODES}, got {mode!r}")
    sampler = None
    if mode != "none":
        try:
            sampler = WeatherSampler(load_kgp_weather(xlsx_path), seed=SEED)
            print(f"[Weather] {len(sampler)} observed days loaded (mode={mode}); "
                  f"rule-label support: {sampler.label_support()}")
        except Exception as e:  # keep the legacy path usable without the workbook
            print(f"[Weather] unavailable ({e}); falling back to synthetic sensors.")
            mode = "none"
    _WEATHER["mode"], _WEATHER["sampler"] = mode, sampler
    return sampler


def weather_mode() -> str:
    if _WEATHER["mode"] is None:
        configure_weather()
    return _WEATHER["mode"]


def weather_sampler() -> Optional[WeatherSampler]:
    if _WEATHER["mode"] is None:
        configure_weather()
    return _WEATHER["sampler"]


def sample_sensor_context(prefer_labels: Optional[List[int]] = None):
    """
    (sensor_text, weather_features | None, rule_label_ids). With weather data the
    sensor line is derived from a real day (label-aware 70% of the time so a heat
    log tends to be paired with a hot day); otherwise the legacy Gaussian simulator.
    """
    sampler = weather_sampler()
    if sampler is None:
        return simulate_sensor_summary(), None, []
    text, feat, labs = sampler.sensor_text(prefer_labels)
    if weather_mode() == "text-only":
        feat = None
    return text, feat, labs


def merge_labels(*label_lists) -> List[int]:
    out = set()
    for labs in label_lists:
        out.update(int(k) for k in (labs or []))
    return sorted(out)


# ----------------- synthetic sensors + fuse -----------------
def simulate_sensor_summary() -> str:
    """Legacy Gaussian sensor line; grounded in a real observed day when weather data is on."""
    sampler = weather_sampler()
    if sampler is not None:
        return sampler.sensor_text()[0]
    return _simulate_sensor_summary_synthetic()


def _simulate_sensor_summary_synthetic() -> str:
    soil_m = round(np.clip(np.random.normal(30, 6), 10, 50), 1)
    soil_ph = round(np.clip(np.random.normal(6.5, 0.4), 5.5, 7.5), 1)
    temp = round(np.clip(np.random.normal(29, 4), 18, 40), 1)
    hum = round(np.clip(np.random.normal(60, 12), 30, 90), 0)
    vpd = round(np.clip(np.random.normal(1.4, 0.4), 0.6, 2.4), 1)
    rain = round(np.clip(np.random.normal(1.0, 1.0), 0.0, 6.0), 1)
    trend = np.random.choice(["↑", "↓", "→"], p=[0.3, 0.3, 0.4])
    return (
        f"SENSORS: soil_moisture={soil_m}%, soil_pH={soil_ph}, temp={temp}°C, "
        f"humidity={hum}%, VPD={vpd} kPa, rainfall_24h={rain}mm (trend: {trend})."
    )


def fuse_text(sensor_txt: str, main_txt: str, mqtt_msg: str = "") -> str:
    main_txt = _norm(main_txt)
    if main_txt.strip().startswith("SENSORS:"):
        base = main_txt
        if "LOG:" not in base:
            base = f"{base}\nLOG: (no additional log)"
    else:
        base = f"{sensor_txt}\nLOG: {main_txt}"
    if mqtt_msg:
        base = f"{base}\nMQTT: {_norm(mqtt_msg)}"
    return base


# ----------------- synthetic local corpus -----------------
LOCAL_BASE = [
    "Maize leaves show interveinal chlorosis and older leaves are yellowing after light rains.",
    "Tomato plants have whiteflies; sticky residue under leaves; some curling.",
    "Rice field shows cracked, dry soil; seedlings drooping under midday sun.",
    "Wheat leaves with orange pustules; reduced tillering; humid mornings reported.",
    "Chili plants show sun scorch on exposed fruits during heatwave; leaf edges crisping.",
]

SYMPTOMS = [
    "leaf curling and pale yellowing on older leaves",
    "sticky residue and presence of aphids",
    "powdery mildew patches on lower canopy",
    "lesions with yellow halos and necrotic centers",
    "wilting during afternoon; recovery at night",
    "chewed margins and frass on leaves",
    "sun scorch on exposed leaves",
    "brown rust pustules along veins",
    "dry topsoil with hard crusting",
    "interveinal chlorosis indicating nutrient stress",
]

CROPS = ["rice", "wheat", "maize", "soybean", "cotton", "tomato", "chili", "potato", "banana", "cabbage"]
WEATHERS = ["a hot, dry wind", "sudden heavy rain", "two cloudy days", "a heatwave", "no rainfall for a week"]

TEMPLATES = [
    "Farmer noted {symptom} while sensors read temp {temp}°C and humidity {hum}%.",
    "{crop} field observed {symptom}; irrigation last 48h minimal; VPD around {vpd} kPa.",
    "After {weather}, plants show {symptom}. Soil moisture near {sm}% and pH {ph}.",
]


def make_balanced_local(n_per: int = 300, n_per_nutrient: int = 600) -> List[str]:
    seeds = {
        "water_stress": [
            "Topsoil is cracking and leaves droop at midday; irrigation uneven.",
            "Canopy stress at noon; mulch missing; dry beds observed.",
        ],
        "nutrient_def": [
            "Interveinal chlorosis on older leaves suggests nitrogen deficiency.",
            "Marginal necrosis indicates potassium shortfall.",
            "Leaf Color Chart shows low score; possible N deficiency.",
        ],
        "pest_risk": [
            "Aphids and honeydew on undersides; sticky traps catching many.",
            "Chewed margins and frass; small caterpillars on leaves.",
        ],
        "disease_risk": [
            "Orange pustules indicate rust; humid mornings; leaf spots spreading.",
            "Powdery mildew on lower canopy; poor airflow in dense rows.",
        ],
        "heat_stress": [
            "Sun scorch on exposed leaves during heatwave; leaf edges crisping.",
            "High temperature window causing thermal stress around midday.",
        ],
    }
    texts = []
    for k, lst in seeds.items():
        reps = n_per_nutrient if k == "nutrient_def" else n_per
        for _ in range(reps):
            texts.append(random.choice(lst))
    random.shuffle(texts)
    return texts


def _maybe_read_mqtt(mqtt_csv: str) -> List[str]:
    if mqtt_csv and os.path.exists(mqtt_csv):
        df = pd.read_csv(mqtt_csv)
        if "message" in df.columns:
            return df["message"].astype(str).tolist()
    return []


def _weather_phrase(r) -> str:
    """Pick the WEATHERS phrase that matches an observed day (falls back to random)."""
    rain, dry = float(r.get("rain_mm", 0) or 0), int(r.get("rainless_days", 0) or 0)
    tmax, rh_pm, sun = float(r.get("tmax", 30) or 30), float(r.get("rh_pm", 60) or 60), float(r.get("sunshine_hours", 6) or 6)
    if rain >= 20:
        return "sudden heavy rain"
    if tmax >= 38:
        return "a heatwave"
    if dry >= 7:
        return "no rainfall for a week"
    if sun < 3:
        return "two cloudy days"
    if tmax >= 35 and rh_pm < 40:
        return "a hot, dry wind"
    return random.choice(WEATHERS)


def build_localmini(max_samples: int = 0, mqtt_csv: str = "", extra_csv: str = "") -> pd.DataFrame:
    """
    Raw (unfused) local logs with columns ["text", "day"]; build_text_corpus_mix attaches
    the SENSORS/WEATHER lines and MQTT. When station data is on, the templated logs read
    their in-text numbers from a real day and pin that day (column "day") so the fused
    sample is internally consistent.
    """
    sampler = weather_sampler()
    rows: List[Tuple[str, Optional[int]]] = [(t, None) for t in list(LOCAL_BASE) + make_balanced_local(300, 600)]

    # templated symptom logs with in-text sensor readings
    for _ in range(2000):
        day = None
        if sampler is not None:
            day = sampler.sample_index()
            r = sampler.df.iloc[day]
            fields = dict(
                temp=round(float(r.get("db_pm") if not pd.isna(r.get("db_pm")) else r["tmean"]) + np.random.normal(0, 0.5), 1),
                hum=int(np.clip(float(r["rh_mean"]) + np.random.normal(0, 3), 15, 100)),
                vpd=round(float(np.clip(r["vpd_kpa"] + np.random.normal(0, 0.05), 0.1, 5)), 1),
                sm=round(float(np.clip(10 + 0.4 * r["soil_moisture_pct"] + np.random.normal(0, 1.5), 2, 60)), 1),
                weather=_weather_phrase(r),
            )
        else:
            fields = dict(
                temp=round(np.clip(np.random.normal(32, 4), 15, 45), 1),
                hum=int(np.clip(np.random.normal(55, 15), 15, 95)),
                vpd=round(np.clip(np.random.normal(1.8, 0.7), 0.2, 4.0), 1),
                sm=round(np.clip(np.random.normal(20, 7), 2, 60), 1),
                weather=random.choice(WEATHERS),
            )
        s = random.choice(TEMPLATES).format(
            symptom=random.choice(SYMPTOMS),
            crop=random.choice(CROPS),
            ph=round(np.clip(np.random.normal(6.5, 0.6), 4.5, 8.5), 1),
            **fields,
        )
        rows.append((s, day))

    # extra CSV of farmer queries (optional)
    if extra_csv and os.path.exists(extra_csv):
        df_extra = pd.read_csv(extra_csv)
        rows.extend((t, None) for t in df_extra.get("text", pd.Series(dtype=str)).astype(str).tolist())

    df = pd.DataFrame([(_norm(t), d) for t, d in rows if weak_labels(t)], columns=["text", "day"])
    if max_samples and len(df) > max_samples:
        df = df.sample(max_samples, random_state=SEED).reset_index(drop=True)
    return df


# ----------------- weather-grounded logs -----------------
# one template family per rule label; wording carries the matching keyword so the
# weak labeller and the agromet rule agree on these rows
WEATHER_LOG_TEMPLATES = {
    "heat_stress": [
        "Afternoon temperature reached {tmax}°C (third day above 36°C); {crop} canopy shows sun scorch and leaf burn by 2 pm.",
        "Heatwave conditions at Kharagpur, max {tmax}°C with evening humidity {rh_pm}%; {crop} flowers dropping, leaf edges crisping.",
        "Hot westerly wind, {tmax}°C at the station; {crop} seedlings show thermal stress and blistering on exposed leaves.",
    ],
    "water_stress": [
        "No effective rain for {dry} days and pan evaporation {evap} mm/day; topsoil dry and cracking, {crop} wilting at midday.",
        "Rainless spell of {dry} days with VPD {vpd} kPa; {crop} leaves droop after noon and recover only at night, irrigation overdue.",
        "Soil moisture low after {dry} days without rain, Tmax {tmax}°C; {crop} field shows drought symptoms and hard crust.",
    ],
    "disease_risk": [
        "{rain} mm rain overnight with morning humidity {rh_am}%; leaf spot lesions and blight spreading in the lower {crop} canopy.",
        "Persistent leaf wetness, {rain7} mm rain this week; rust pustules and powdery mildew noticed on {crop} rows.",
        "Humid mornings ({rh_am}% RH) after showers, temperatures {tmean}°C; fungal lesions with yellow halos on {crop} leaves.",
    ],
    "pest_risk": [
        "Warm settled weather after the rains ({tmean}°C, RH {rh_mean}%); aphid and whitefly counts rising on sticky traps in {crop}.",
        "{dry} rainless days at {tmean}°C; thrips and mites building up on {crop}, sticky residue and honeydew under leaves.",
        "Sunny spell ({sun} h sunshine) and mild nights; stem borer moths in the light trap, chewed margins on {crop} leaves.",
    ],
    "nutrient_def": [
        "{rain7} mm rain this week has leached nitrogen from the {crop} beds; older leaves show interveinal chlorosis.",
        "Waterlogging after {rain30} mm rain in 30 days; {crop} leaves pale yellow, leaf color chart score low, likely N deficiency.",
        "Heavy monsoon rain ({rain7} mm/week) washed out top-dressed fertilizer; {crop} older leaves yellowing from the tips.",
    ],
}


def build_weather_logs(max_per: int = 2000) -> List[Tuple[str, List[float], List[int]]]:
    """
    Farmer-style logs written from real observed days, one per active rule label per day.
    Returns (raw_log, weather_features, rule_label_ids); the mixer fuses the matching
    SENSORS/WEATHER lines itself so text, features and labels describe the same day.
    """
    sampler = weather_sampler()
    if sampler is None:
        raise RuntimeError("weather data not available")
    rows = []
    for i in range(len(sampler)):
        r = sampler.df.iloc[i]
        f = lambda k, nd=0: ("n/a" if pd.isna(r.get(k)) else f"{float(r[k]):.{nd}f}")
        fields = dict(
            crop=random.choice(CROPS), tmax=f("tmax", 1), tmean=f("tmean", 1), rh_am=f("rh_am"),
            rh_pm=f("rh_pm"), rh_mean=f("rh_mean"), rain=f("rain_mm", 1), rain7=f("rain_7d"),
            rain30=f("rain_30d"), dry=int(r.get("rainless_days", 0) or 0), evap=f("evaporation", 1),
            vpd=f("vpd_kpa", 1), sun=f("sunshine_hours", 1),
        )
        for k in sampler.rule_labels[i]:
            tpl = random.choice(WEATHER_LOG_TEMPLATES[ISSUE_LABELS[k]])
            rows.append((i, tpl.format(**fields)))
    random.shuffle(rows)
    if max_per and len(rows) > max_per:
        rows = rows[:max_per]
    out = []
    for i, log in rows:
        feat = None if weather_mode() == "text-only" else sampler.features[i]
        out.append((log, i, feat))
    return out


# ----------------- HF loading helpers (TEXT) -----------------
def _load_ds(name, split=None, streaming=False):
    """
    Load a dataset, accepting a list of candidate repo ids. Hugging Face has moved
    several canonical datasets under namespaces (ag_news -> fancyzhx/ag_news), and a
    bare id now raises "Repository id must be 'namespace/name'", so callers pass every
    id the dataset has been published under and the first that resolves wins.
    """
    if not HAS_DATASETS:
        raise RuntimeError("The `datasets` library is not installed.")
    if isinstance(name, (list, tuple)):
        last = None
        for cand in name:
            try:
                return _load_ds(cand, split=split, streaming=streaming)
            except Exception as e:
                last = e
                print(f"[Mix] {cand} unavailable: {str(e)[:90]}")
        raise last if last else RuntimeError("no candidate dataset id resolved")
    dlconf = DownloadConfig(max_retries=3)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    kw = {"streaming": streaming, "download_config": dlconf}
    if token:
        kw["token"] = token
        kw["use_auth_token"] = token
    for attempt in range(4):
        try:
            if split:
                return load_dataset(name, split=split, **kw)
            return load_dataset(name, **kw)
        except Exception as e:
            if any(x in str(e) for x in ["429", "Read timed out", "504", "Temporary failure", "Connection"]):
                time.sleep(min(60, 1.5 * (2 ** attempt)))
                kw["streaming"] = True
                continue
            raise
    kw["streaming"] = True
    return load_dataset(name, split=split, **kw)


AGRI_RE = re.compile(
    r"\b(agri|agriculture|farm|farmer|farming|crop|soil|harvest|irrigat|pest|blight|"
    r"drought|yield|wheat|rice|paddy|maize|soy|cotton|fertiliz|orchard|greenhouse|horticul)\b",
    re.I,
)
NON_AG_NOISE = re.compile(
    r"\b(NFL|NBA|MLB|NHL|tennis|golf|soccer|stocks?|Nasdaq|Dow Jones|Hollywood|movie|concert)\b",
    re.I,
)


def build_gardian_stream(max_per: int = 2000) -> List[str]:
    # Try multiple agricultural datasets as fallback
    dataset_options = [
        "CGIAR/gardian-ai-ready-docs",
        "maharshipandya/agricultural-datasets",
        "turing-motors/agricultural-qa"
    ]
    
    texts = []
    for dataset_name in dataset_options:
        try:
            ds = _load_ds(dataset_name, streaming=True)
            seen = 0
            if isinstance(ds, dict):
                splits = ds.keys()
            else:
                splits = [None]
            for sp in splits:
                subset = ds[sp] if sp is not None else ds
                for r in subset:
                    raw = (r.get("text") or r.get("content") or r.get("question") or "").strip()
                    if raw and _lang_ok(raw):
                        texts.append(_norm(raw))
                        seen += 1
                        if seen >= max_per:
                            break
                if seen >= max_per:
                    break
            if texts:  # If we got any texts, return them
                return texts
        except Exception as e:
            print(f"[WARN] {dataset_name} failed: {e}, trying next...")
            continue
    return texts


# field pairs argilla/farming has used across revisions, newest first
_ARGILLA_FIELDS = [
    ("instruction", "response"),
    ("evolved_questions", "domain_expert_answer"),
    ("question", "answer"),
    ("prompt", "completion"),
]


def _argilla_text(rec: dict) -> str:
    """Question+answer text from a record, whichever field names this revision uses."""
    for q_key, a_key in _ARGILLA_FIELDS:
        if q_key in rec or a_key in rec:
            q = str(rec.get(q_key, "") or "").strip()
            a = str(rec.get(a_key, "") or "").strip()
            if q or a:
                return f"{q} {a}".strip()
    # last resort: the longest free-text fields on the record
    vals = [str(v).strip() for k, v in rec.items()
            if isinstance(v, str) and k != "id" and len(str(v).strip()) > 40]
    return " ".join(vals[:2]).strip()


def build_argilla_stream(max_per: int = 2000) -> List[str]:
    ds = _load_ds("argilla/farming")
    splits = ds if isinstance(ds, dict) else {"train": ds}
    texts = []
    for sp in splits:
        for r in splits[sp]:
            raw = _argilla_text(dict(r))
            if raw and _lang_ok(raw):
                texts.append(_norm(raw))
                if len(texts) >= max_per:
                    return texts
    return texts


def build_agnews_agri(max_per: int = 2000) -> List[str]:
    train = _load_ds(["fancyzhx/ag_news", "ag_news"], split="train", streaming=True)
    texts = []
    seen = 0
    for r in train:
        raw = (r.get("text") or "").strip()
        if (
            raw
            and AGRI_RE.search(raw)
            and not NON_AG_NOISE.search(raw)
            and _lang_ok(raw)
        ):
            texts.append(_norm(raw))
            seen += 1
            if seen >= max_per:
                break
    return texts


# ----------------- MIX builder for text -----------------
def build_text_corpus_mix(
    mix_sources: str = "gardian,argilla,agnews,localmini,weather",
    max_per_source: int = 2000,
    max_samples: int = 0,
    mqtt_csv: str = "",
    extra_csv: str = "",
) -> pd.DataFrame:
    """
    Returns columns ["text", "labels"] plus "weather" (list[float], length WEATHER_DIM)
    when the weather modality is on in "full" mode. Labels = keyword weak labels of the
    log  U  agromet-rule labels of the paired observed day.
    """
    sources = [s.strip().lower() for s in mix_sources.split(",") if s.strip()]
    sampler = weather_sampler()
    mqtt_msgs = _maybe_read_mqtt(mqtt_csv)
    # (source, raw_text, fixed_weather_day_index | None)
    pool: List[Tuple[str, str, Optional[int]]] = []

    def _try(name: str, fn):
        print(f"[Mix] loading {name} (<= {max_per_source}) ...")
        try:
            texts = fn(max_per_source)
            pool.extend([(name, t, None) for t in texts])
            print(f"[Mix] {name} added {len(texts)} rows")
        except Exception as e:
            print(f"[Mix] {name} skipped: {e}")

    if "gardian" in sources:
        _try("gardian", build_gardian_stream)
    if "argilla" in sources:
        _try("argilla", build_argilla_stream)
    if "agnews" in sources:
        _try("agnews", build_agnews_agri)
    if "localmini" in sources:
        df_local = build_localmini(max_per_source, mqtt_csv, extra_csv)
        for t, day in df_local[["text", "day"]].itertuples(index=False):
            pool.append(("localmini", t, None if pd.isna(day) else int(day)))
    if "weather" in sources:
        if sampler is None:
            print("[Mix] weather skipped: no station data")
        else:
            wl = build_weather_logs(max_per_source)
            pool.extend([("weather", log, day) for log, day, _ in wl])
            print(f"[Mix] weather added {len(wl)} rows")

    # deduplicate by text hash
    seen = set()
    dedup: List[Tuple[str, str, Optional[int]]] = []
    for src, txt, day in pool:
        h = hashlib.sha1(_norm(txt).encode("utf-8", "ignore")).hexdigest()
        if h not in seen:
            seen.add(h)
            dedup.append((src, _norm(txt), day))

    use_features = sampler is not None and weather_mode() == "full"
    rows = []
    for src, raw, day in dedup:
        kw_labs = weak_labels(raw)
        if sampler is None:
            sensor, feat, w_labs = simulate_sensor_summary(), None, []
        elif day is not None:  # weather-grounded log: keep its own day
            sensor = sampler.sensor_text_for(day)
            feat, w_labs = sampler.features[day], sampler.rule_labels[day]
        else:
            sensor, feat, w_labs = sample_sensor_context(prefer_labels=kw_labs)
        mqtt = random.choice(mqtt_msgs) if (mqtt_msgs and src == "localmini" and random.random() < 0.4) else ""
        text = fuse_text(sensor, raw, mqtt)
        labs = merge_labels(kw_labs, w_labs)
        if labs:
            rows.append((text, labs, src, None if feat is None else [float(x) for x in feat]))
    df = pd.DataFrame(rows, columns=["text", "labels", "source", "weather"])
    print("[Mix] source breakdown:")
    print(df["source"].value_counts())

    if max_samples and len(df) > max_samples:
        df = df.sample(max_samples, random_state=SEED).reset_index(drop=True)
    cols = ["text", "labels"] + (["weather"] if use_features else [])
    return df[cols]


# ----------------- Image datasets (PlantVillage + others via HF) -----------------
def load_plant_images_hf(max_images: int = 4000):
    """
    Legacy helper: approximate PlantVillage via a HF mirror.

    Uses: BrandonFors/Plant-Diseases-PlantVillage-Dataset
    Returns a dataset with an `image` column, or None if it fails.
    """
    if not HAS_DATASETS:
        print("[Images] datasets not installed; no HF images.")
        return None

    name = "BrandonFors/Plant-Diseases-PlantVillage-Dataset"
    try:
        print(f"[Images] loading {name} from HF (this will download once)...")
        ds = load_dataset(name, split="train")
        if "image" not in ds.column_names:
            print(f"[Images] {name} has no 'image' column; skipping.")
            return None
        if max_images and len(ds) > max_images:
            ds = ds.shuffle(seed=SEED).select(range(max_images))
        print(f"[Images] {name} loaded: {len(ds)} samples")
        return ds
    except Exception as e:
        print(f"[Images] failed to load {name}: {e}")
        return None


def load_stress_image_datasets_hf(
    max_total_images: int = 20000,
    max_per_dataset: int = 6000,
):
    """
    Try to load multiple plant stress / disease datasets from Hugging Face.

    Datasets attempted (train split):
        - BrandonFors/Plant-Diseases-PlantVillage-Dataset
        - Saon110/bd-crop-vegetable-plant-disease-dataset
        - timm/plant-pathology-2021
        - uqtwei2/PlantWild

    Returns:
        - a concatenated HF dataset with an `image` column, or
        - None (if all fail)

    Any dataset failing to load or lacking an `image` column is skipped.
    """
    if not HAS_DATASETS:
        print("[Images] datasets not installed; no HF images.")
        return None

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    common_kwargs = {}
    if token:
        common_kwargs["token"] = token
        common_kwargs["use_auth_token"] = token

    specs = [
        ("BrandonFors/Plant-Diseases-PlantVillage-Dataset", {"split": "train"}),
        ("Saon110/bd-crop-vegetable-plant-disease-dataset", {"split": "train"}),
        ("timm/plant-pathology-2021", {"split": "train"}),
        ("uqtwei2/PlantWild", {"split": "train"}),
    ]

    ds_list = []

    for name, kw in specs:
        full_kw = dict(kw)
        full_kw.update(common_kwargs)
        try:
            print(f"[Images] trying to load {name} ({kw}) ...")
            ds = load_dataset(name, **full_kw)

            # Handle DatasetDict vs Dataset
            if isinstance(ds, DatasetDict):
                if "train" in ds:
                    ds = ds["train"]
                else:
                    first_split = list(ds.keys())[0]
                    ds = ds[first_split]

            if "image" not in ds.column_names:
                print(f"[Images] {name} has no 'image' column; skipping.")
                continue

            if max_per_dataset and len(ds) > max_per_dataset:
                ds = ds.shuffle(seed=SEED).select(range(max_per_dataset))

            print(f"[Images] {name} loaded: {len(ds)} samples")
            ds_list.append(ds)
        except Exception as e:
            print(f"[Images] failed to load {name}: {e}")

    if not ds_list:
        print("[Images] no HF image datasets could be loaded; using dummy images in training.")
        return None

    merged = concatenate_datasets(ds_list)
    if max_total_images and len(merged) > max_total_images:
        merged = merged.shuffle(seed=SEED).select(range(max_total_images))

    print(f"[Images] merged image dataset size: {len(merged)} samples "
          f"(max_total_images={max_total_images})")
    return merged


# ----------------- Convenience for summaries -----------------
def summarize_labels(df: pd.DataFrame, tag: str = "set"):
    counts = np.zeros(NUM_LABELS, int)
    for labs in df["labels"]:
        for k in labs:
            counts[k] += 1
    print(f"[{tag}] label counts:", {ISSUE_LABELS[i]: int(c) for i, c in enumerate(counts)})
