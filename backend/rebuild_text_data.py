"""
Replace data/{class}/text.csv with properly labeled real agricultural text.

Sources:
  1. PlantVillageVQA CSVs (GitHub, text-only, no images)
     -> disease_risk: Comprehensive Description + Causal Reasoning answers
        for known plant diseases (Late Blight, Rust, Mildew, etc.)
  2. argilla/farming (HuggingFace)
     -> all 5 classes via keyword scoring

Run:
    pip install datasets pandas
    python rebuild_text_data.py
"""

import random, re, csv, io
from pathlib import Path
from collections import defaultdict
import urllib.request

random.seed(42)

DATA_DIR          = Path(__file__).parent.parent / "data"
STRESS_LABELS     = ["water_stress", "nutrient_def", "pest_risk", "disease_risk", "heat_stress"]
SAMPLES_PER_CLASS = 500

# ── Disease → stress class mapping ────────────────────────────────────────────
DISEASE_MAP = {
    # disease_risk — fungal/bacterial/viral
    "late blight":                    "disease_risk",
    "early blight":                   "disease_risk",
    "bacterial spot":                 "disease_risk",
    "septoria leaf spot":             "disease_risk",
    "target spot":                    "disease_risk",
    "leaf mold":                      "disease_risk",
    "powdery mildew":                 "disease_risk",
    "tomato yellow leaf curl virus":  "disease_risk",
    "tomato mosaic virus":            "disease_risk",
    "black rot":                      "disease_risk",
    "haunglongbing citrus greening":  "disease_risk",
    "esca black measles":             "disease_risk",
    "common rust":                    "disease_risk",
    "northern leaf blight":           "disease_risk",
    "apple scab":                     "disease_risk",
    "cercospora leaf spot gray leaf spot": "disease_risk",
    "cedar apple rust":               "disease_risk",
    "leaf blight isariopsis leaf spot":   "disease_risk",
    # pest_risk
    "spider mites":                   "pest_risk",
    "aphid":                          "pest_risk",
    "mite":                           "pest_risk",
    # water_stress
    "leaf scorch":                    "water_stress",
}

GOOD_QTYPES = {
    "Comprehensive Description",
    "Causal Reasoning",
    "Specific Disease Identification",
    "Visual Attribute Grounding",
}

# ── Keyword scoring (fallback for argilla/farming) ────────────────────────────
CROP_VOCAB = re.compile(
    r"\b(crop|crops|harvest|yield|paddy|rice|wheat|maize|cotton|soybean|potato|"
    r"tomato|cassava|sorghum|sugarcane|irrigation|fertilizer|pesticide|fungicide|"
    r"soil moisture|soil fertility|agronomy|agricultur|farm|plant disease|"
    r"crop disease|leaf spot|stem borer|aphid|armyworm|whitefly|blight|"
    r"rust disease|drought stress|water stress|heat stress|nutrient deficien|"
    r"chlorosis|mildew|powdery|downy|lesion|necrosis|pathogen)\b",
    re.I,
)

KW = {
    "water_stress": {
        "high":   ["drought", "water stress", "wilting", "water deficit",
                   "dry spell", "soil moisture", "irrigation failure", "desiccation"],
        "medium": ["irrigation", "moisture", "rainfall deficit", "stomatal", "turgor"],
        "low":    ["dry", "precipitation", "arid"],
    },
    "nutrient_def": {
        "high":   ["nutrient deficiency", "nitrogen deficiency", "phosphorus deficiency",
                   "potassium deficiency", "chlorosis", "interveinal chlorosis",
                   "micronutrient", "soil fertility"],
        "medium": ["fertilizer", "nutrient", "npk", "nitrogen", "phosphorus",
                   "potassium", "zinc deficiency", "iron deficiency"],
        "low":    ["soil", "spad", "fertiliser"],
    },
    "pest_risk": {
        "high":   ["pest infestation", "insect damage", "stem borer", "aphid infestation",
                   "armyworm", "whitefly", "thrips", "mite infestation", "leaf miner"],
        "medium": ["aphid", "caterpillar", "mite", "larvae", "weevil", "nematode", "hopper"],
        "low":    ["pest", "insect pest", "crop pest"],
    },
    "disease_risk": {
        "high":   ["plant disease", "crop disease", "fungal infection", "blast disease",
                   "late blight", "rust disease", "powdery mildew", "downy mildew",
                   "leaf spot", "lesion", "pathogen", "blight"],
        "medium": ["mildew", "rust", "rot", "necrosis", "wilt disease", "sporulation",
                   "fungicide", "phytophthora"],
        "low":    ["disease", "fungus", "pustule"],
    },
    "heat_stress": {
        "high":   ["heat stress", "thermal stress", "heat wave", "high temperature stress",
                   "pollen sterility", "canopy temperature", "heat injury"],
        "medium": ["temperature stress", "scorch", "grain fill", "heat tolerance"],
        "low":    ["heat", "hot weather", "high temperature crop"],
    },
}


def score_text(text: str):
    t = text.lower()
    if not CROP_VOCAB.search(t):
        return None, 0
    scores = {}
    for stress, groups in KW.items():
        s = 0
        for kw in groups["high"]:
            if kw in t: s += 3
        for kw in groups["medium"]:
            if kw in t: s += 2
        for kw in groups["low"]:
            if kw in t: s += 1
        scores[stress] = s
    best = max(scores, key=scores.get)
    return (best, scores[best]) if scores[best] > 0 else (None, 0)


try:
    import pandas as pd
    from datasets import load_dataset
except ImportError:
    raise SystemExit("Install: pip install datasets pandas")

labelled = defaultdict(list)


# ═══════════════════════════════════════════════════════════════════════════════
# Source 1: PlantVillageVQA — annotation CSVs from GitHub (no images)
# ═══════════════════════════════════════════════════════════════════════════════
print("[1/2] PlantVillageVQA annotation CSVs (GitHub, text only) ...")

PVVQA_BASE = (
    "https://raw.githubusercontent.com/SyedNazmusSakib-SNS/"
    "PlantVillageVQA/main/QA_Dataset/"
)
# Use the final full-quality version (v4_5 has 210k rows)
PVVQA_FILES = [
    "vqa_dataset_v4_5.csv",
    "vqa_dataset_v4_4.csv",
    "vqa_dataset_v4_3.csv",
]

pvvqa_added = 0
for fname in PVVQA_FILES:
    # Stop early if disease_risk already well-covered
    if len(labelled.get("disease_risk", [])) >= SAMPLES_PER_CLASS * 4:
        print(f"   disease_risk saturated, skipping {fname}")
        break
    try:
        url = PVVQA_BASE + fname
        print(f"   Downloading {fname} ...")
        with urllib.request.urlopen(url, timeout=30) as r:
            content = r.read().decode("utf-8")
        df = pd.read_csv(io.StringIO(content))

        # Filter to descriptive question types only
        df = df[df["question_type"].isin(GOOD_QTYPES)].copy()
        # Exclude Healthy and Unknown
        df = df[~df["disease"].str.lower().isin(["healthy", "unknown/other"])].copy()

        for _, row in df.iterrows():
            disease = str(row.get("disease", "")).strip().lower()
            answer  = str(row.get("answer", "")).strip()
            question = str(row.get("question", "")).strip()
            crop    = str(row.get("crop", "")).strip()

            if len(answer) < 30:
                continue

            # Direct mapping via disease name
            stress = DISEASE_MAP.get(disease)
            if stress is None:
                # Try partial match
                for d_key, s_val in DISEASE_MAP.items():
                    if d_key in disease:
                        stress = s_val
                        break

            if stress:
                text = f"{question} {answer}".strip() if question else answer
                labelled[stress].append(text[:600])
                pvvqa_added += 1

        print(f"   {fname}: {pvvqa_added} labeled so far")
    except Exception as e:
        print(f"   WARN {fname}: {e}")

print(f"   -> PlantVillageVQA total: {pvvqa_added} labeled texts")


# ═══════════════════════════════════════════════════════════════════════════════
# Source 2: argilla/farming — genuine crop Q&A (for all 5 classes)
# ═══════════════════════════════════════════════════════════════════════════════
print("[2/2] argilla/farming ...")
try:
    ds = load_dataset("argilla/farming")
    texts = []
    for split in (ds if isinstance(ds, dict) else {"train": ds}):
        for row in (ds[split] if isinstance(ds, dict) else ds):
            a = str(row.get("response",
                   row.get("domain_expert_answer", ""))).strip()
            q = str(row.get("instruction",
                   row.get("evolved_questions", ""))).strip()
            combined = (q + " " + a).strip()
            if combined: texts.append(combined)
            if a:         texts.append(a)
    added = 0
    for t in texts:
        t = t.strip()
        if len(t) < 40: continue
        label, sc = score_text(t)
        if label:
            labelled[label].append(t[:600])
            added += 1
    print(f"   -> {added} labeled from {len(texts)} texts")
except Exception as e:
    print(f"   WARN: {e}")


# ── Report ────────────────────────────────────────────────────────────────────
print("\nCollected per class:")
for s in STRESS_LABELS:
    print(f"  {s}: {len(labelled[s])} texts")

min_avail = min(len(labelled[s]) for s in STRESS_LABELS)
if min_avail < 5:
    raise SystemExit(f"\nInsufficient data (min={min_avail}). Check connectivity.")


# ── Write corrected text.csv files ───────────────────────────────────────────
print("\nWriting text.csv files ...")
for label_idx, stress in enumerate(STRESS_LABELS):
    pool = labelled[stress]
    random.shuffle(pool)

    if len(pool) >= SAMPLES_PER_CLASS:
        samples = pool[:SAMPLES_PER_CLASS]
    else:
        factor  = (SAMPLES_PER_CLASS // len(pool)) + 1
        samples = (pool * factor)[:SAMPLES_PER_CLASS]
        print(f"  NOTE: {stress} only {len(pool)} unique -> oversampled to {SAMPLES_PER_CLASS}")

    out_path = DATA_DIR / stress / "text.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text","labels","label_name","source"])
        writer.writeheader()
        for t in samples:
            writer.writerow({"text": t, "labels": f"[{label_idx}]",
                             "label_name": stress, "source": "pvvqa_argilla"})
    print(f"  {stress}: {len(samples)} rows -> {out_path}")


# ── Merged dataset ────────────────────────────────────────────────────────────
all_rows = []
for label_idx, stress in enumerate(STRESS_LABELS):
    df = pd.read_csv(DATA_DIR / stress / "text.csv")
    all_rows.append(df)

df_all = pd.concat(all_rows, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
merged_path = DATA_DIR / "crop_stress_text_dataset.csv"
df_all.to_csv(merged_path, index=False)
print(f"\nMerged dataset: {len(df_all)} rows -> {merged_path}")
print(df_all["label_name"].value_counts().to_string())

# ── Spot-check ────────────────────────────────────────────────────────────────
print("\n--- Spot-check (2 samples per class) ---")
for stress in STRESS_LABELS:
    sub = df_all[df_all["label_name"] == stress]["text"].sample(2, random_state=1).tolist()
    print(f"\n[{stress}]")
    for s in sub:
        print(f"  * {s[:180]}")

print("\nDone.")
