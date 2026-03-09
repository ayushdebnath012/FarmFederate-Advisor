"""
Generate labeled text dataset from local crop stress images using BLIP captioning.

Reads images from  data/{class_name}/images/
Writes to          data/crop_stress_text_dataset.csv  (appends if file exists)

Run on Colab (free T4):
    !pip install -q transformers pillow pandas torch torchvision
    %run generate_image_captions.py

Or locally:
    python generate_image_captions.py
"""

import os, random, sys
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# ── Config ──────────────────────────────────────────────────────────────────
MAX_PER_CLASS  = 400          # captions to generate per stress class
IMG_BATCH_SIZE = 8            # images processed per GPU batch
MODEL_ID       = "Salesforce/blip-image-captioning-large"
SEED           = 42

# ★ SET THIS if auto-detection fails on Colab ★
# e.g. MANUAL_DATA_DIR = "/content/drive/MyDrive/FarmFederate/data"
MANUAL_DATA_DIR = None

STRESS_LABELS  = ["water_stress", "nutrient_def", "pest_risk", "disease_risk", "heat_stress"]

# Multiple symptom-specific prompts per class.
# BLIP continues each prefix — two prompts per class doubles captions
# and captures different aspects of the stress condition.
CLASS_PROMPTS = {
    "water_stress": [
        "a wilting crop plant with drooping leaves showing",
        "a drought stressed plant with dry curling leaves and",
    ],
    "nutrient_def": [
        "a plant leaf with yellowing chlorosis showing",
        "a pale discoloured leaf with nutrient deficiency and",
    ],
    "pest_risk": [
        "a crop leaf with insect damage and feeding holes showing",
        "a plant damaged by pests with chewed edges and",
    ],
    "disease_risk": [
        "a diseased plant leaf with fungal spots showing",
        "an infected crop leaf with blight and lesions and",
    ],
    "heat_stress": [
        "a heat stressed plant with scorched brown leaf tips showing",
        "a crop plant with heat damage wilted tips and",
    ],
}

# ── Paths ────────────────────────────────────────────────────────────────────
def _find_data_dir() -> Path:
    if MANUAL_DATA_DIR:
        p = Path(MANUAL_DATA_DIR)
        print(f"  Using MANUAL_DATA_DIR: {p}")
        p.mkdir(parents=True, exist_ok=True)
        return p

    candidates = [
        Path("/content/drive/MyDrive/FarmFederate/data"),
        Path("/content/drive/My Drive/FarmFederate/data"),
        Path("/content/FarmFederate/data"),
    ]
    try:
        candidates.append(Path(__file__).parent.parent / "data")
    except NameError:
        pass

    for p in candidates:
        try:
            if p.exists():
                # Check at least one class image folder exists
                has_images = any(
                    (p / cls / "images").exists()
                    for cls in ["disease_risk", "water_stress", "pest_risk"]
                )
                if has_images:
                    print(f"  Auto-detected DATA_DIR: {p}")
                    return p
        except Exception:
            pass

    # Fallback: use Drive root if mounted, else local
    for drive_root in [Path("/content/drive/MyDrive"), Path("/content/drive/My Drive")]:
        if drive_root.exists():
            p = drive_root / "FarmFederate" / "data"
            p.mkdir(parents=True, exist_ok=True)
            print(f"  Drive mounted, using: {p}")
            return p

    p = Path("/content/data")
    p.mkdir(parents=True, exist_ok=True)
    print(f"  WARNING: No FarmFederate data folder found. Using: {p}")
    print("  Set MANUAL_DATA_DIR at the top of this script to fix this.")
    return p

DATA_DIR   = _find_data_dir()
OUTPUT_CSV = DATA_DIR / "crop_stress_text_dataset.csv"
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)  # ensure directory exists

# ── Device ───────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}  |  DATA_DIR: {DATA_DIR}")

# ── Load BLIP ────────────────────────────────────────────────────────────────
print(f"Loading BLIP model: {MODEL_ID} ...")
processor = BlipProcessor.from_pretrained(MODEL_ID)
model     = BlipForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.float16 if DEVICE=="cuda" else torch.float32)
model     = model.to(DEVICE).eval()
print("BLIP ready.")


def caption_batch(image_paths: list[str], prompt: str) -> list[str]:
    """Caption a batch of images with the given conditional prompt."""
    images = []
    valid_paths = []
    for p in image_paths:
        try:
            img = Image.open(p).convert("RGB")
            images.append(img)
            valid_paths.append(p)
        except Exception:
            pass
    if not images:
        return []

    inputs = processor(
        images=images,
        text=[prompt] * len(images),
        return_tensors="pt",
        padding=True,
    ).to(DEVICE)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=120,
            num_beams=4,
            repetition_penalty=1.5,
            length_penalty=1.2,
        )
    captions = processor.batch_decode(out, skip_special_tokens=True)
    return captions


# ── Main ─────────────────────────────────────────────────────────────────────
rows = []

for label_idx, stress in enumerate(STRESS_LABELS):
    img_dir = DATA_DIR / stress / "images"
    if not img_dir.exists():
        print(f"  [SKIP] {img_dir} not found")
        continue

    all_paths = [str(p) for p in img_dir.glob("*.jpg")] + \
                [str(p) for p in img_dir.glob("*.png")]
    random.seed(SEED + label_idx)
    random.shuffle(all_paths)
    paths = all_paths[:MAX_PER_CLASS]

    prompts  = CLASS_PROMPTS[stress]   # list of 2 prompts
    captions = []

    for prompt in prompts:
        for i in range(0, len(paths), IMG_BATCH_SIZE):
            batch   = paths[i : i + IMG_BATCH_SIZE]
            batch_c = caption_batch(batch, prompt)
            captions.extend(batch_c)
            done = min(i + IMG_BATCH_SIZE, len(paths))
            print(f"  {stress} [{prompts.index(prompt)+1}/2]: {done}/{len(paths)}", end="\r")

    print(f"  {stress}: {len(captions)} captions generated ({len(paths)} images x 2 prompts)")

    for cap in captions:
        cap = cap.strip()
        if len(cap) < 30:
            continue
        rows.append({
            "text":       cap,
            "label":      label_idx,
            "label_name": stress,
            "difficulty": "medium",      # real images = medium difficulty
            "crop":       "unknown",
            "region":     "unknown",
            "source":     "blip_caption",
        })

df_new = pd.DataFrame(rows)

# Merge with existing CSV if present
if OUTPUT_CSV.exists():
    df_existing = pd.read_csv(OUTPUT_CSV)
    # Keep existing rows that are NOT from blip_caption (don't duplicate)
    if "source" in df_existing.columns:
        df_existing = df_existing[df_existing["source"] != "blip_caption"]
    df_out = pd.concat([df_existing, df_new], ignore_index=True)
else:
    df_out = df_new

df_out = df_out.sample(frac=1, random_state=SEED).reset_index(drop=True)
df_out.to_csv(OUTPUT_CSV, index=False)

print(f"\nSaved {len(df_out)} total rows → {OUTPUT_CSV}")
print(df_out["label_name"].value_counts().to_string())
