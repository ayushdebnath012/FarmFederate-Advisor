"""
Classify images in Real Dataset using Ref-Dataset as reference.

Ref-Dataset has 5 named disease classes (2 images each).
Real Dataset has 200 images with YOLO OBB labels (class IDs 0-4).

This script:
1. Extracts ResNet-50 embeddings from reference images
2. Classifies each Real Dataset image by cosine similarity to references
3. Determines class ID → disease name mapping
4. Outputs results CSV + classes.txt for YOLO
"""

import os
import csv
import json
from pathlib import Path
from collections import Counter, defaultdict

import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────
REF_DIR   = Path(r"C:\Users\USER_HP\Desktop\FarmFederate\Ref-Dataset")
REAL_IMGS = Path(r"C:\Users\USER_HP\Desktop\FarmFederate\Real Dataset\images")
REAL_LABS = Path(r"C:\Users\USER_HP\Desktop\FarmFederate\Real Dataset\labels")
OUT_DIR   = Path(r"C:\Users\USER_HP\Desktop\FarmFederate\Real Dataset")

# ── Model setup ────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
model.fc = torch.nn.Identity()          # remove classifier head → 2048-d embedding
model.eval().to(device)

transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]),
])


def embed(img_path: Path) -> torch.Tensor:
    img = Image.open(img_path).convert("RGB")
    x   = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model(x)
    return F.normalize(feat.squeeze(0), dim=0).cpu()


def crop_obb(img: Image.Image, line: str) -> Image.Image | None:
    """Crop the OBB region from image given a YOLO OBB label line."""
    parts = line.strip().split()
    if len(parts) < 9:
        return None
    coords = list(map(float, parts[1:9]))   # x1,y1,x2,y2,x3,y3,x4,y4 (normalised)
    W, H = img.size
    xs = [coords[i] * W for i in range(0, 8, 2)]
    ys = [coords[i] * H for i in range(1, 8, 2)]
    x_min, x_max = max(0, int(min(xs))), min(W, int(max(xs)) + 1)
    y_min, y_max = max(0, int(min(ys))), min(H, int(max(ys)) + 1)
    if x_max <= x_min or y_max <= y_min:
        return None
    return img.crop((x_min, y_min, x_max, y_max))


# ── Step 1: Build reference embeddings ────────────────────────────────────
print("\n=== Building reference embeddings ===")
ref_embeddings: dict[str, list[torch.Tensor]] = {}

for class_dir in sorted(REF_DIR.iterdir()):
    if not class_dir.is_dir():
        continue
    class_name = class_dir.name
    embs = []
    for img_file in class_dir.iterdir():
        if img_file.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
            embs.append(embed(img_file))
            print(f"  {class_name}: {img_file.name}")
    if embs:
        ref_embeddings[class_name] = embs

class_names = sorted(ref_embeddings.keys())
print(f"\nReference classes ({len(class_names)}): {class_names}")

# Mean embedding per class
ref_mean: dict[str, torch.Tensor] = {
    c: torch.stack(embs).mean(0)
    for c, embs in ref_embeddings.items()
}
ref_mean = {c: F.normalize(v, dim=0) for c, v in ref_mean.items()}


def classify_embedding(feat: torch.Tensor) -> tuple[str, float]:
    """Return (class_name, cosine_similarity) for the best match."""
    best_cls, best_sim = "", -1.0
    for cls, mean_emb in ref_mean.items():
        sim = float(torch.dot(feat, mean_emb))
        if sim > best_sim:
            best_cls, best_sim = cls, sim
    return best_cls, best_sim


# ── Step 2: Classify Real Dataset images ──────────────────────────────────
print("\n=== Classifying Real Dataset images ===")

results = []          # list of dicts per image
yolo_id_votes = defaultdict(list)  # yolo_class_id → [predicted_class_name, ...]

img_files = sorted(REAL_IMGS.glob("*.jpg")) + sorted(REAL_IMGS.glob("*.png"))
print(f"Found {len(img_files)} images")

for img_path in img_files:
    label_path = REAL_LABS / (img_path.stem + ".txt")
    img = Image.open(img_path).convert("RGB")

    # --- Whole-image classification ---
    whole_feat = embed(img_path)
    whole_cls, whole_sim = classify_embedding(whole_feat)

    # --- Per-crop classification (from OBB boxes) ---
    crop_preds: list[dict] = []
    if label_path.exists():
        with open(label_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yolo_id = int(line.split()[0])
                crop = crop_obb(img, line)
                if crop is None:
                    crop_preds.append({"yolo_id": yolo_id, "cls": whole_cls, "sim": whole_sim, "source": "whole"})
                    continue
                # embed crop
                crop_t = transform(crop.resize((224, 224))).unsqueeze(0).to(device)
                with torch.no_grad():
                    feat = model(crop_t)
                feat = F.normalize(feat.squeeze(0), dim=0).cpu()
                cls, sim = classify_embedding(feat)
                crop_preds.append({"yolo_id": yolo_id, "cls": cls, "sim": sim, "source": "crop"})
                yolo_id_votes[yolo_id].append(cls)

    results.append({
        "image": img_path.name,
        "whole_class": whole_cls,
        "whole_sim": round(whole_sim, 4),
        "crops": crop_preds,
    })

    if len(results) % 20 == 0:
        print(f"  Processed {len(results)}/{len(img_files)}")

print(f"  Processed {len(img_files)}/{len(img_files)} — done")


# ── Step 3: Determine YOLO class ID → disease name mapping ────────────────
print("\n=== YOLO class ID to disease name mapping ===")

yolo_id_to_class: dict[int, str] = {}
for yolo_id, votes in sorted(yolo_id_votes.items()):
    counter = Counter(votes)
    best_name, best_count = counter.most_common(1)[0]
    pct = 100 * best_count / len(votes)
    yolo_id_to_class[yolo_id] = best_name
    print(f"  ID {yolo_id}: {best_name}  ({best_count}/{len(votes)} = {pct:.0f}%)  distribution: {dict(counter)}")


# ── Step 4: Write outputs ──────────────────────────────────────────────────

# classes.txt (ordered by YOLO ID; fill gaps with Unknown)
max_id = max(yolo_id_to_class.keys()) if yolo_id_to_class else 4
classes_txt = OUT_DIR / "classes.txt"
with open(classes_txt, "w") as f:
    for i in range(max_id + 1):
        f.write(yolo_id_to_class.get(i, f"Unknown_{i}") + "\n")
print(f"\nWrote {classes_txt}")

# Full results CSV
csv_path = OUT_DIR / "classification_results.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image", "whole_image_class", "whole_sim",
                     "num_crops", "crop_summary"])
    for r in results:
        crop_summary = "; ".join(
            f"id{c['yolo_id']}->{c['cls']}({c['sim']:.3f})"
            for c in r["crops"]
        )
        writer.writerow([
            r["image"],
            r["whole_class"],
            r["whole_sim"],
            len(r["crops"]),
            crop_summary,
        ])
print(f"Wrote {csv_path}")

# Summary JSON
summary = {
    "yolo_id_to_disease": yolo_id_to_class,
    "class_distribution_in_real_dataset": {
        k: v for k, v in Counter(
            cls
            for r in results
            for c in r["crops"]
            for cls in [c["cls"]]
        ).most_common()
    },
    "num_images": len(img_files),
    "reference_classes": class_names,
}
json_path = OUT_DIR / "classification_summary.json"
with open(json_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"Wrote {json_path}")

print("\n=== Summary ===")
print(json.dumps(summary, indent=2))
