"""
Generate a labeled crop stress text dataset using the Claude API.
Produces a CSV with columns: text, label, label_name, difficulty, crop, region

Usage:
    pip install anthropic pandas
    export ANTHROPIC_API_KEY=your_key
    python generate_crop_stress_dataset.py

Output: data/crop_stress_text_dataset.csv  (~2500 samples, 500/class)
"""

import os
import json
import time
import random
import pandas as pd
import anthropic
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────
SAMPLES_PER_CLASS = 500        # total per stress class
BATCH_SIZE        = 10         # samples per API call (keep prompts short)
OUTPUT_PATH       = Path(__file__).parent.parent / "data" / "crop_stress_text_dataset.csv"
MODEL             = "claude-haiku-4-5-20251001"   # fast + cheap

STRESS_LABELS = ["water_stress", "nutrient_def", "pest_risk", "disease_risk", "heat_stress"]

CROPS = [
    "rice", "wheat", "maize", "potato", "tomato", "cotton", "soybean",
    "sugarcane", "cassava", "groundnut", "sunflower", "barley", "sorghum",
    "chickpea", "lentil", "onion", "chili", "banana", "mango", "coffee",
]

REGIONS = [
    "South Asia", "Sub-Saharan Africa", "Southeast Asia", "East Africa",
    "Latin America", "Mediterranean", "West Africa", "South America",
]

DIFFICULTIES = ["easy", "medium", "hard"]

# Per-class generation instructions
CLASS_INSTRUCTIONS = {
    "water_stress": """
Generate realistic crop field observation texts describing WATER STRESS symptoms only.
Symptoms to mention (vary across samples): wilting, leaf rolling, stomatal closure,
reduced turgor, drooping, grey-green discolouration, tip burn, reduced growth,
soil cracking, low tensiometer readings, reduced canopy cover.
Do NOT mention: disease lesions, insects, pests, heat, nutrient yellowing.
""",
    "nutrient_def": """
Generate realistic crop field observation texts describing NUTRIENT DEFICIENCY symptoms only.
Symptoms to mention (vary across samples): interveinal chlorosis, yellowing from older leaves,
purple stem discolouration, slow growth, pale canopy, leaf margin browning, stunted roots,
low SPAD readings, tissue test deficits. Specify which nutrient (N, P, K, Fe, Zn, B, Ca, S).
Do NOT mention: insects, disease spots, water shortage, high temperature.
""",
    "pest_risk": """
Generate realistic crop field observation texts describing PEST / INSECT INFESTATION symptoms only.
Symptoms to mention (vary across samples): leaf mining, skeletonisation, frass, honeydew,
defoliation, stem boring, dead heart, whitefly colonies, aphid clusters, mite webbing,
thrip scarring, caterpillar feeding, sweep net counts above threshold.
Do NOT mention: disease lesions, yellowing from nutrients, drought, heat.
""",
    "disease_risk": """
Generate realistic crop field observation texts describing PLANT DISEASE symptoms only.
Symptoms to mention (vary across samples): lesions, pustules, blight, rust, powdery mildew,
downy mildew, angular spots, crown rot, damping off, sheath blight, blast, late blight,
sporulation, water-soaked spots, necrotic halos, pathogen confirmed by lab.
Do NOT mention: insects, wilting from drought, nutrient yellowing, heat scorch.
""",
    "heat_stress": """
Generate realistic crop field observation texts describing HEAT STRESS symptoms only.
Symptoms to mention (vary across samples): pollen sterility, forced maturity, leaf rolling
from heat (not drought), bleaching, tip burn in extreme heat, reduced grain fill,
canopy temperature above 38°C, accelerated senescence, flower drop, wax bloom.
Mention specific high temperature readings (35-44°C). Do NOT mention drought, insects, disease.
""",
}

DIFFICULTY_INSTRUCTIONS = {
    "easy":   "Use clear, unambiguous symptom language. A single dominant symptom should identify the class.",
    "medium": "Use realistic field notes mixing 2-3 symptoms. One general term (e.g. 'yellowing', 'wilting') that could apply to other classes is OK.",
    "hard":   "Write an ambiguous observation mentioning symptoms that partially overlap with one other stress class. The primary class should still be identifiable from context.",
}

# ── API call ────────────────────────────────────────────────────────────────
def generate_batch(client: anthropic.Anthropic, stress_class: str,
                   n: int, difficulty: str, crop: str, region: str) -> list[str]:
    prompt = f"""You are an experienced agronomist writing field observation notes.

Class: {stress_class.upper()}
{CLASS_INSTRUCTIONS[stress_class]}

Difficulty: {difficulty.upper()}
{DIFFICULTY_INSTRUCTIONS[difficulty]}

Crop: {crop}
Region: {region}

Write exactly {n} different field observation texts (1-3 sentences each).
Each text must be a realistic farm scouting note or farmer report.
Vary sentence structure, symptom combination, and severity across texts.

Return ONLY a JSON array of strings, no commentary:
["text1", "text2", ...]"""

    for attempt in range(3):
        try:
            msg = client.messages.create(
                model=MODEL,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = msg.content[0].text.strip()
            # extract JSON array
            start = raw.find("[")
            end   = raw.rfind("]") + 1
            texts = json.loads(raw[start:end])
            return [t for t in texts if isinstance(t, str) and len(t) > 20]
        except Exception as e:
            print(f"  Retry {attempt+1}/3 ({e})")
            time.sleep(2 ** attempt)
    return []


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("Set ANTHROPIC_API_KEY environment variable.")

    client = anthropic.Anthropic(api_key=api_key)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for label_idx, stress in enumerate(STRESS_LABELS):
        print(f"\n[{label_idx+1}/5] Generating {SAMPLES_PER_CLASS} samples for: {stress}")
        collected = []

        # Distribute across difficulties: 30% easy, 40% medium, 30% hard
        targets = {
            "easy":   int(SAMPLES_PER_CLASS * 0.30),
            "medium": int(SAMPLES_PER_CLASS * 0.40),
            "hard":   SAMPLES_PER_CLASS - int(SAMPLES_PER_CLASS * 0.30) - int(SAMPLES_PER_CLASS * 0.40),
        }

        for difficulty, target in targets.items():
            diff_collected = []
            while len(diff_collected) < target:
                crop   = random.choice(CROPS)
                region = random.choice(REGIONS)
                n_req  = min(BATCH_SIZE, target - len(diff_collected))
                texts  = generate_batch(client, stress, n_req, difficulty, crop, region)
                for t in texts:
                    diff_collected.append({"text": t, "crop": crop, "region": region,
                                           "difficulty": difficulty})
                    if len(diff_collected) >= target:
                        break
                print(f"  {difficulty}: {len(diff_collected)}/{target}", end="\r")
            collected.extend(diff_collected)
            print(f"  {difficulty}: {len(diff_collected)}/{target} ✓")

        for item in collected[:SAMPLES_PER_CLASS]:
            rows.append({
                "text":       item["text"],
                "label":      label_idx,
                "label_name": stress,
                "difficulty": item["difficulty"],
                "crop":       item["crop"],
                "region":     item["region"],
            })

    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(df)} samples → {OUTPUT_PATH}")
    print(df["label_name"].value_counts())
    print(df["difficulty"].value_counts())


if __name__ == "__main__":
    main()
