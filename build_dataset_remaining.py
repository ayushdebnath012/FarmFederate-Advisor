"""
Complete the farmfederate_dataset/ folder:
1. Generate text corpus (2000 synthetic samples, 400/class)
2. Copy field-collected images from Classified_Crops/crops/
3. Write per-class label CSVs
4. Write dataset_info.json
5. Write README.md
"""

import csv
import json
import os
import random
import re
import shutil
import pathlib

random.seed(42)

BASE = pathlib.Path("C:/Users/USER_HP/Desktop/FarmFederate")
DS   = BASE / "farmfederate_dataset"
CROPS_DIR = BASE / "data" / "Classified_Crops" / "crops"

CLASSES = ["gray_blight", "helopeltis", "algal_leaf_spot", "brown_blight", "red_leaf_spot"]
CID     = {c: i for i, c in enumerate(CLASSES)}

# ── 1. TEXT CORPUS ────────────────────────────────────────────────────────────
print("Generating text corpus...")

# Class-specific symptom templates (40% usage)
class_specific = {
    "gray_blight": [
        "The tea leaf shows distinct {adj} gray patches with irregular margins spreading across the {part}.",
        "Gray blight infection causes {adj} grayish-brown lesions on the {part} of tea plants.",
        "Affected tea shoots display {adj} necrotic tissue with a characteristic gray discoloration.",
        "The {part} of the tea bush exhibits gray blight symptoms including {adj} sunken lesions.",
        "Tea growers reported {adj} gray blight damage with necrotic spots covering the {part}.",
    ],
    "helopeltis": [
        "Helopeltis feeding damage causes {adj} brown scorch marks on the {part} of tea shoots.",
        "The tea flush shows {adj} helopeltis bug puncture wounds with dark halos on {part}.",
        "Tea tip borer helopeltis has caused {adj} characteristic shot-hole patterns on the {part}.",
        "Young tea shoots suffer {adj} helopeltis damage with angular {part} lesions and necrosis.",
        "The {part} displays classic helopeltis infestation marks with {adj} brown stippling patterns.",
    ],
    "algal_leaf_spot": [
        "Circular {adj} algal leaf spots with raised orange-red centers appear on the {part}.",
        "Tea leaves show {adj} Cephaleuros virescens spots creating rust-orange patches on {part}.",
        "The {part} has {adj} algal spot colonies with characteristic powdery orange surface texture.",
        "Algal leaf spot disease presents {adj} circular lesions with orange spore masses on {part}.",
        "Tea plant {part} affected by {adj} algal spots showing concentric rings and orange sporulation.",
    ],
    "brown_blight": [
        "Brown blight causes {adj} water-soaked lesions that turn necrotic on the tea {part}.",
        "The {part} shows {adj} brown blight symptoms with irregular dark brown spots and margins.",
        "Tea leaves display {adj} brown blight patches caused by Colletotrichum species on {part}.",
        "Severe {adj} brown blight infection results in premature defoliation of tea {part}.",
        "The tea {part} has {adj} brown blight lesions with concentric rings and yellow halos.",
    ],
    "red_leaf_spot": [
        "Red leaf spot creates {adj} reddish-brown circular lesions on the upper tea {part} surface.",
        "The {part} shows {adj} red spot disease caused by Phyllosticta with characteristic red margins.",
        "Tea leaves exhibit {adj} red leaf spot lesions with distinct reddish centers on {part}.",
        "The {part} displays {adj} red spot symptoms with sunken centers and dark red borders.",
        "Farmers noted {adj} red leaf spot outbreaks causing {part} damage and reduced yield.",
    ],
}

# Cross-class general templates (60% usage)
cross_class = [
    "The tea plantation in {region} reports significant crop stress affecting the {part} of plants.",
    "Field scouts observed {adj} leaf disease symptoms requiring immediate treatment in the {part}.",
    "Tea crop health assessment shows {adj} fungal infection spreading through {part} tissue.",
    "The {part} of tea plants shows {adj} signs of stress possibly related to environmental factors.",
    "Agronomists identified {adj} disease patterns on tea {part} consistent with pathogen infection.",
    "Tea bushes in the {region} garden show {adj} leaf symptoms requiring diagnostic testing.",
    "Crop monitoring detected {adj} anomalies in tea {part} coloration indicating possible disease.",
    "The {part} exhibits {adj} discoloration and tissue damage consistent with tea leaf disease.",
    "Disease pressure on tea {part} has been {adj} in the current growing season.",
    "Tea cultivation report: {adj} stress indicators found on {part} across multiple plots.",
    "Leaf biopsy from the {part} shows {adj} cellular damage consistent with pathogen attack.",
    "Tea estate management flagged {adj} infection on the {part} requiring fungicide application.",
    "Remote sensing detected {adj} spectral anomalies in tea canopy at {part} level.",
    "Pest and disease survey recorded {adj} damage to tea {part} in highland gardens.",
    "Weather-related stress compounded by {adj} fungal infection on tea {part} tissue.",
    "The {part} shows mixed {adj} symptoms suggesting complex tea leaf disease interaction.",
    "Extension officer identified {adj} disease outbreak affecting tea {part} in the cooperative.",
    "Tea quality tests show degraded leaf integrity with {adj} damage to {part} structures.",
]

ADJS  = ["severe", "moderate", "mild", "extensive", "localized", "widespread", "early-stage", "advanced", "chronic", "acute"]
PARTS = ["leaf lamina", "leaf tip", "leaf margin", "upper leaf surface", "lower leaf surface", "young shoot", "mature leaf", "leaf midrib", "petiole"]
REGS  = ["Assam", "Darjeeling", "Nilgiri", "Sri Lanka", "Kenya", "Yunnan", "Sylhet", "Munnar", "Kangra"]

text_dir = DS / "text"
text_dir.mkdir(exist_ok=True)

rows = []
for cls in CLASSES:
    cid = CID[cls]
    per_class = 400
    n_specific = int(per_class * 0.40)
    n_cross    = per_class - n_specific
    templates_s = class_specific[cls]

    for i in range(n_specific):
        tmpl = templates_s[i % len(templates_s)]
        text = tmpl.format(
            adj=random.choice(ADJS),
            part=random.choice(PARTS),
            region=random.choice(REGS),
        )
        split = "train" if i < int(n_specific * 0.8) else "val"
        rows.append({"text": text, "class_id": cid, "class_name": cls,
                     "split": split, "template_type": "class_specific"})

    for i in range(n_cross):
        tmpl = random.choice(cross_class)
        text = tmpl.format(
            adj=random.choice(ADJS),
            part=random.choice(PARTS),
            region=random.choice(REGS),
        )
        split = "train" if i < int(n_cross * 0.8) else "val"
        rows.append({"text": text, "class_id": cid, "class_name": cls,
                     "split": split, "template_type": "cross_class"})

random.shuffle(rows)

corpus_path = text_dir / "text_corpus.csv"
with open(corpus_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["text", "class_id", "class_name", "split", "template_type"])
    w.writeheader()
    w.writerows(rows)

print(f"  Written {len(rows)} rows to {corpus_path}")

# ── 2. FIELD-COLLECTED IMAGES ─────────────────────────────────────────────────
print("\nCopying field-collected images...")

field_out = DS / "field_collected"
field_labels_rows = []

if not CROPS_DIR.exists():
    print(f"  WARNING: {CROPS_DIR} not found, skipping field-collected copy")
else:
    CLASS_MAP = {
        "Algal Leaf":   "algal_leaf_spot",
        "Algal_Leaf":   "algal_leaf_spot",
        "AlgalLeaf":    "algal_leaf_spot",
        "algal_leaf":   "algal_leaf_spot",
        "algal leaf":   "algal_leaf_spot",
        "Gray Blight":  "gray_blight",
        "Gray_Blight":  "gray_blight",
        "GrayBlight":   "gray_blight",
        "gray_blight":  "gray_blight",
        "gray blight":  "gray_blight",
        "Helopeltis":   "helopeltis",
        "helopeltis":   "helopeltis",
        "Brown Blight": "brown_blight",
        "Brown_Blight": "brown_blight",
        "BrownBlight":  "brown_blight",
        "brown_blight": "brown_blight",
        "brown blight": "brown_blight",
        "Red Leaf Spot":"red_leaf_spot",
        "Red_Leaf_Spot":"red_leaf_spot",
        "RedLeafSpot":  "red_leaf_spot",
        "red_leaf_spot":"red_leaf_spot",
        "red leaf spot":"red_leaf_spot",
    }

    for cls_dir in sorted(CROPS_DIR.iterdir()):
        if not cls_dir.is_dir():
            continue
        canonical = CLASS_MAP.get(cls_dir.name)
        if canonical is None:
            # Try partial match
            for k, v in CLASS_MAP.items():
                if k.lower() in cls_dir.name.lower():
                    canonical = v
                    break
        if canonical is None:
            print(f"  SKIP unknown dir: {cls_dir.name}")
            continue

        out_dir = field_out / canonical
        out_dir.mkdir(parents=True, exist_ok=True)

        imgs = [f for f in cls_dir.iterdir()
                if f.suffix.lower() in {".jpg", ".jpeg", ".png"}]

        for img in imgs:
            dest = out_dir / img.name
            shutil.copy2(img, dest)
            field_labels_rows.append({
                "filename": img.name,
                "rel_path": f"field_collected/{canonical}/{img.name}",
                "class_name": canonical,
                "class_id": CID[canonical],
                "source": "field_collected",
            })

    print(f"  Copied {len(field_labels_rows)} field images")

    # Write field labels CSV
    field_csv = DS / "labels" / "field_labels.csv"
    with open(field_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["filename", "rel_path", "class_name", "class_id", "source"])
        w.writeheader()
        w.writerows(field_labels_rows)
    print(f"  Written {field_csv}")

# ── 3. PER-CLASS LABEL CSVs ───────────────────────────────────────────────────
print("\nWriting per-class label CSVs...")

master_csv = DS / "labels" / "master_labels.csv"
per_class_data = {cls: [] for cls in CLASSES}

with open(master_csv, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        per_class_data[row["class_name"]].append(row)

for cls in CLASSES:
    cls_csv = DS / "labels" / f"{cls}_labels.csv"
    rows_cls = per_class_data[cls]
    if rows_cls:
        with open(cls_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_cls[0].keys()))
            w.writeheader()
            w.writerows(rows_cls)
        print(f"  {cls}: {len(rows_cls)} rows -> {cls_csv.name}")

# ── 4. DATASET INFO JSON ──────────────────────────────────────────────────────
print("\nWriting dataset_info.json...")

# Count images
img_counts = {}
for split in ["train", "val"]:
    img_counts[split] = {}
    for cls in CLASSES:
        p = DS / "images" / split / cls
        img_counts[split][cls] = len(list(p.glob("*"))) if p.exists() else 0

total_imgs = sum(sum(v.values()) for v in img_counts.values())

# Count text
text_train = sum(1 for r in rows if r["split"] == "train")
text_val   = sum(1 for r in rows if r["split"] == "val")

# Source breakdown from master CSV
source_counts = {"field": 0, "kaggle": 0}
with open(master_csv, "r", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if row["source"] == "field":
            source_counts["field"] += 1
        else:
            source_counts["kaggle"] += 1

info = {
    "name": "FarmFederate Tea Leaf Disease Dataset",
    "version": "1.0",
    "description": (
        "Hybrid dataset for multimodal federated learning on tea leaf stress detection. "
        "Combines 200 field-collected images (669 labeled instances) from tea gardens "
        "with images sourced from the Kaggle Tea Leaf Disease dataset (Mamun et al., 2023). "
        "Training uses a balanced 600-samples/class cap (3,000 total)."
    ),
    "classes": {cls: {"id": CID[cls]} for cls in CLASSES},
    "splits": {
        "train": {"total": sum(img_counts["train"].values()), "per_class": img_counts["train"]},
        "val":   {"total": sum(img_counts["val"].values()),   "per_class": img_counts["val"]},
    },
    "total_images": total_imgs,
    "source_breakdown": {
        "field_collected": source_counts["field"],
        "kaggle": source_counts["kaggle"],
        "field_collected_raw": 200,
        "field_labeled_instances": 669,
    },
    "text_corpus": {
        "total_samples": len(rows),
        "per_class": 400,
        "train": text_train,
        "val": text_val,
        "template_types": {"class_specific": "40%", "cross_class": "60%"},
    },
    "field_collected_raw": {
        "total_images": 200,
        "labeled_instances": 669,
        "annotation": "multi-label",
        "location": "field_collected/",
    },
    "federated_setup": {
        "clients": 5,
        "rounds": 8,
        "local_epochs": 3,
        "aggregation": "FedAvg",
        "data_distribution": "non-IID Dirichlet alpha=1.0",
    },
    "citations": [
        "Mamun et al. (2023). Tea Leaf Disease Dataset. Kaggle.",
        "FarmFederate: Multimodal Federated Learning for Tea Crop Stress Detection. 2026.",
    ],
}

info_path = DS / "dataset_info.json"
with open(info_path, "w", encoding="utf-8") as f:
    json.dump(info, f, indent=2)
print(f"  Written {info_path}")

# ── 5. README ─────────────────────────────────────────────────────────────────
print("\nWriting README.md...")

readme = f"""# FarmFederate Tea Leaf Disease Dataset

**Version 1.0** — April 2026

Hybrid multimodal dataset for tea crop stress detection, used in the FarmFederate
federated learning study.

## Overview

| Attribute | Value |
|-----------|-------|
| Classes | 5 tea stress categories |
| Training images | 2,400 (480/class) |
| Validation images | 600 (120/class) |
| Text samples | 2,000 (400/class) |
| Field-collected images | 200 (669 labeled instances) |
| Total unique images | 5,067+ |

## Classes

| ID | Name | Description |
|----|------|-------------|
| 0 | gray_blight | Pestalotiopsis-caused gray necrotic lesions |
| 1 | helopeltis | Tea mosquito bug puncture damage |
| 2 | algal_leaf_spot | Cephaleuros virescens orange algal spots |
| 3 | brown_blight | Colletotrichum brown water-soaked lesions |
| 4 | red_leaf_spot | Phyllosticta reddish circular lesions |

## Folder Structure

```
farmfederate_dataset/
├── images/
│   ├── train/{{class}}/          # 480 images per class
│   └── val/{{class}}/            # 120 images per class
├── labels/
│   ├── master_labels.csv         # All 3,000 image labels
│   ├── train.csv                 # (see splits/)
│   ├── {{class}}_labels.csv      # Per-class label files
│   └── field_labels.csv          # Field-collected multi-label annotations
├── splits/
│   ├── train.csv                 # 2,400 training image paths
│   └── val.csv                   # 600 validation image paths
├── text/
│   └── text_corpus.csv           # 2,000 synthetic text samples
├── field_collected/{{class}}/    # Original 200 field images (669 instances)
├── metadata/                     # (reserved for future use)
└── dataset_info.json             # Full statistics

```

## Data Sources

### Training Images (3,000 total)
- **Kaggle Tea Leaf Disease Dataset** (Mamun et al., 2023): Primary source.
  4,867 images across 5 classes. Balanced to 600/class for training.
- **Field-collected** (200 images, 669 labeled instances): Tea garden surveys
  in Northeast India. Multi-label annotations; small subset included per class.

### Text Corpus (2,000 samples)
Synthetic agricultural text generated from domain templates:
- 40% class-specific symptom descriptions
- 60% cross-class general agricultural text
- 80/20 train/val split per class

## CSV Columns

**master_labels.csv / splits/train.csv / splits/val.csv:**
```
filename, rel_path, class_name, class_id, source, split
```

**text/text_corpus.csv:**
```
text, class_id, class_name, split, template_type
```

**labels/field_labels.csv:**
```
filename, rel_path, class_name, class_id, source
```

## Usage

```python
import pandas as pd
from PIL import Image

df = pd.read_csv("farmfederate_dataset/labels/master_labels.csv")
train_df = df[df["split"] == "train"]

# Load an image
img_path = "farmfederate_dataset/" + train_df.iloc[0]["rel_path"]
img = Image.open(img_path)
```

## Citation

If you use this dataset, please cite:
```
FarmFederate: Multimodal Federated Learning for Tea Crop Stress Detection (2026)
Mamun et al. Tea Leaf Disease Dataset. Kaggle (2023).
```

## Statistics

- **Source breakdown**: {source_counts['field']} field-collected + {source_counts['kaggle']} Kaggle (in 3,000-image training set)
- **Class balance**: Perfectly balanced at 600/class (train+val)
- **Original field imbalance**: 26:1 (helopeltis under-represented)
- **Text templates**: {len(cross_class)} cross-class + {sum(len(v) for v in class_specific.values())} class-specific
"""

readme_path = DS / "README.md"
with open(readme_path, "w", encoding="utf-8") as f:
    f.write(readme)
print(f"  Written {readme_path}")

# ── SUMMARY ───────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("DATASET COMPLETE")
print("="*60)
print(f"Images:       {total_imgs} ({sum(img_counts['train'].values())} train + {sum(img_counts['val'].values())} val)")
print(f"Text corpus:  {len(rows)} samples ({text_train} train + {text_val} val)")
print(f"Field images: {len(field_labels_rows)}")
print(f"Location:     {DS}")
