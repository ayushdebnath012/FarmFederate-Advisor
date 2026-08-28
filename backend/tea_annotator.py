# -*- coding: utf-8 -*-
"""
Tea Leaf Disease Annotator
==========================
Three modes:

  1. DEFAULT  (no flags)
     Saves contact-sheet PNGs per class + template-based annotations.csv
     from tea literature symptom vocabulary. Fast, no extra downloads.

  2. --auto_caption
     Uses Salesforce/blip-image-captioning-base (downloaded once, ~450 MB)
     to visually describe every crop, then fuses the visual description with
     disease-specific vocabulary from literature.
     Result: richer, image-grounded text prompts saved to annotations.csv.
     Runtime: ~15-20 min on CPU for 371 crops.

  3. --save_crops
     Extracts every OBB bounding-box crop and saves it as a JPEG under
     tea_results/annotation/crops/<disease>/<image>_box<idx>.jpg
     Useful for manual review or training a separate classifier.

Usage
-----
  # Template annotations (instant):
  python backend/tea_annotator.py

  # Auto-caption with BLIP (recommended, ~15 min):
  python backend/tea_annotator.py --auto_caption

  # Also save individual crop images:
  python backend/tea_annotator.py --auto_caption --save_crops

  # Fix wrong class mapping then re-caption:
  python backend/tea_annotator.py --auto_caption --remap 0:algal_leaf_spot 3:brown_blight

  # Train with the generated annotations:
  python backend/tea_disease_federated.py --annotations tea_results/annotation/annotations.csv
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# DEFAULT CLASS MAPPING  (edit here or via --remap on the command line)
# ---------------------------------------------------------------------------

CLASS_MAP: Dict[int, str] = {
    0: "gray_blight",
    1: "helopeltis",
    2: "algal_leaf_spot",
    3: "brown_blight",
    4: "red_leaf_spot",
}

# ---------------------------------------------------------------------------
# DISEASE SYMPTOM VOCABULARY  (from tea literature)
# ---------------------------------------------------------------------------

_SYMPTOMS: Dict[str, List[str]] = {
    "brown_blight": [
        "brown necrotic patches on the lamina with water-soaked margins",
        "irregular brown lesions spreading along leaf veins with dark borders",
        "necrotic brown areas covering the blade, edges curling inward",
        "light brown to tan discolouration with a darker peripheral ring",
        "extensive browning of the leaf tissue, tip and margin blight visible",
        "brown patches with concentric zonation and chlorotic halos around lesions",
        "progressive brown necrosis from leaf margin toward the midrib",
        "large coalescing brown spots with distinct darker edges",
        "tan-coloured dead tissue surrounded by a yellow-green transition zone",
    ],
    "helopeltis": [
        "angular brown feeding scars surrounded by a red-brown halo from Helopeltis puncture",
        "scattered circular necrotic spots from insect stylet penetration, edges dark brown",
        "multiple pinhole-sized lesions with radiating brown discolouration typical of tea mosquito bug",
        "raised corky spots on young shoots caused by Helopeltis theivora feeding",
        "dark depressed feeding punctures with secondary yellowing on young flush",
        "wilting and die-back of shoot tips accompanied by brown puncture marks on stipules",
        "numerous small round lesions with a depressed centre and reddish margin",
        "insect feeding damage visible as scattered dark punctures on soft leaf tissue",
    ],
    "gray_blight": [
        "pale grey to ashen lesions with a thin dark margin caused by Pestalotiopsis infection",
        "greyish-white blotches expanding from the leaf tip, concentric rings faintly visible",
        "bleached grey patches on older leaves, acervuli visible as dark specks on the surface",
        "grey necrotic areas with a straw-coloured centre and brown border",
        "diffuse ash-coloured discolouration of the blade with scattered black fruiting bodies",
        "pale silvery lesions surrounded by a narrow brown band on mature leaves",
        "light grey irregular patches coalescing across the blade surface",
        "grey-white dead tissue with small dark pycnidia scattered throughout",
    ],
    "algal_leaf_spot": [
        "circular raised green-to-orange felty spots caused by Cephaleuros virescens algal growth",
        "velvety orange-red algal tufts scattered across the upper leaf surface",
        "thallus-covered circular patches turning rusty brown with rough surface texture",
        "greenish algal encrustations on both leaf surfaces reducing photosynthetic area",
        "orange felt-like algal spots with scalloped margins on shaded lower canopy leaves",
        "rough circular lesions with superficial orange sporulation typical of algal leaf spot",
        "raised rust-coloured circular patches with a powdery or velvety surface texture",
        "orange-brown crusty spots adhering to the leaf blade with fibrous filaments",
    ],
    "red_leaf_spot": [
        "distinct red-brown circular spots with a pale tan centre and dark red border",
        "small red to purple spots coalescing into larger necrotic areas on the blade",
        "round red lesions with concentric rings and a chlorotic yellow halo",
        "reddish-brown blotches on the upper surface, darker on the underside",
        "numerous tiny red flecks enlarging to irregular red-brown patches under wet conditions",
        "red necrotic spots of 2-5 mm diameter scattered across the lamina surface",
        "dark red circular lesions with a slightly raised margin on the leaf surface",
        "red-purple spots with a necrotic tan centre, characteristic of Colletotrichum infection",
    ],
}

_CONTEXT: Dict[str, List[str]] = {
    "brown_blight": [
        "high humidity and warm temperatures favoured disease spread",
        "prevalent during the monsoon flush when canopy is dense",
        "observed on shaded bushes in the low-grown region",
        "associated with poor drainage and waterlogged field conditions",
        "infection enhanced by prolonged leaf wetness and warm nights",
    ],
    "helopeltis": [
        "damage concentrated on tender young flush in the early plucking round",
        "infestation higher at forest margins where shade trees harbour the pest",
        "accompanying secondary fungal infection through entry wounds",
        "field scouting revealed 3-5 Helopeltis nymphs per bush on affected plots",
        "pest population peaked during dry spells with minimal predator activity",
    ],
    "gray_blight": [
        "infection prevalent on senescent leaves during the dry inter-monsoon period",
        "found mostly in the inner canopy where air circulation is restricted",
        "associated with nitrogen-deficient and potassium-deficient tea",
        "sporadic occurrence on neglected bushes with dense overhead shade",
        "disease spread accelerated by mechanical damage during pruning operations",
    ],
    "algal_leaf_spot": [
        "found predominantly on lower canopy leaves under heavy shade trees",
        "incidence higher in gardens with high atmospheric humidity year-round",
        "more prevalent in poorly maintained gardens with excess shade",
        "algal growth reduced after copper-based fungicide application",
        "common in old tea gardens with thick canopy cover and slow air movement",
    ],
    "red_leaf_spot": [
        "spots appeared after extended leaf wetness during overcast weather",
        "incidence increased in poorly drained plots with heavy clay soils",
        "young mature leaves most susceptible during the late monsoon season",
        "field observation recorded following high rainfall and low temperature",
        "disease pressure increased during cool humid weather in the hill regions",
    ],
}

_TEMPLATES_BLIP = [
    "Field image shows {blip_caption}. Disease diagnosis: {symptom}. Field context: {context}.",
    "Visual observation: {blip_caption}. Agronomic note: {symptom}. Background: {context}.",
    "Crop inspection — image depicts {blip_caption}. Symptom assessment: {symptom}. Site condition: {context}.",
    "Scout report: photograph shows {blip_caption}. Clinical sign: {symptom}. Observation: {context}.",
    "Diagnostic record: leaf image — {blip_caption}. Confirmed symptom: {symptom}. Context: {context}.",
]

_TEMPLATES_PLAIN = [
    "Field observation: Leaf exhibits {symptom}. {context}.",
    "Tea bush inspection report: Disease symptom — {symptom}. Environmental note: {context}.",
    "Scout record: {symptom}. Agronomic context: {context}.",
    "Diagnostic note: The sampled leaf shows {symptom}. Background: {context}.",
    "Plantation report: {symptom} observed on tea leaves. Field condition: {context}.",
    "Crop health log: Visible symptom — {symptom}. Observation context: {context}.",
    "Advisory record: Leaf condition consistent with {symptom}. Site context: {context}.",
]


def generate_text_plain(disease: str, rng: random.Random) -> str:
    sym  = rng.choice(_SYMPTOMS[disease])
    ctx  = rng.choice(_CONTEXT[disease])
    tmpl = rng.choice(_TEMPLATES_PLAIN)
    return tmpl.format(symptom=sym, context=ctx)


def generate_text_blip(blip_caption: str, disease: str, rng: random.Random) -> str:
    sym  = rng.choice(_SYMPTOMS[disease])
    ctx  = rng.choice(_CONTEXT[disease])
    tmpl = rng.choice(_TEMPLATES_BLIP)
    # Clean up BLIP caption: strip "a photo of", lowercase, remove trailing period
    cap = blip_caption.lower().strip().rstrip(".")
    for prefix in ("a photo of ", "an image of ", "a picture of ", "a close up of "):
        if cap.startswith(prefix):
            cap = cap[len(prefix):]
            break
    return tmpl.format(blip_caption=cap, symptom=sym, context=ctx)


# ---------------------------------------------------------------------------
# OBB CROP HELPER
# ---------------------------------------------------------------------------

CROP_PADDING = 0.10
TARGET_SIZE  = 224


def obb_to_rect(corners: List[float], W: int, H: int,
                padding: float = CROP_PADDING) -> Tuple[float, float, float, float]:
    xs = [corners[i] * W for i in range(0, 8, 2)]
    ys = [corners[i] * H for i in range(1, 8, 2)]
    px, py = padding * W, padding * H
    x0 = max(0.0, min(xs) - px);  y0 = max(0.0, min(ys) - py)
    x1 = min(float(W), max(xs) + px); y1 = min(float(H), max(ys) + py)
    if x1 - x0 < 8: x0, x1 = max(0.0, x0 - 8), min(float(W), x1 + 8)
    if y1 - y0 < 8: y0, y1 = max(0.0, y0 - 8), min(float(H), y1 + 8)
    return x0, y0, x1, y1


def crop_image(img: Image.Image, corners: List[float],
               size: int = TARGET_SIZE) -> Image.Image:
    W, H = img.size
    x0, y0, x1, y1 = obb_to_rect(corners, W, H)
    return img.crop((x0, y0, x1, y1)).resize(
        (size, size), Image.LANCZOS
    ).convert("RGB")


# ---------------------------------------------------------------------------
# LOAD ALL CROPS
# ---------------------------------------------------------------------------

def load_all_crops(data_dir: Path, class_map: Dict[int, str]) -> List[Dict]:
    records = []
    img_dir = data_dir / "images"
    lbl_dir = data_dir / "labels"
    for lf in sorted(lbl_dir.glob("*.txt")):
        img_path = img_dir / (lf.stem + ".jpg")
        if not img_path.exists():
            img_path = img_dir / (lf.stem + ".png")
        if not img_path.exists():
            continue
        with open(lf) as f:
            for box_idx, line in enumerate(f):
                parts = line.strip().split()
                if len(parts) < 9:
                    continue
                cls_id = int(parts[0])
                if cls_id not in class_map:
                    continue
                records.append({
                    "image_file": img_path.name,
                    "box_idx":    box_idx,
                    "class_id":   cls_id,
                    "disease":    class_map[cls_id],
                    "corners":    [float(v) for v in parts[1:9]],
                    "img_path":   img_path,
                })
    return records


# ---------------------------------------------------------------------------
# BLIP AUTO-CAPTIONER
# ---------------------------------------------------------------------------

class BLIPCaptioner:
    """Lightweight wrapper around BLIP image-captioning-base."""

    MODEL_ID = "Salesforce/blip-image-captioning-base"

    def __init__(self):
        print("  Loading BLIP model (Salesforce/blip-image-captioning-base) ...")
        print("  First run: ~450 MB download to HuggingFace cache.")
        from transformers import BlipProcessor, BlipForConditionalGeneration
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = BlipProcessor.from_pretrained(self.MODEL_ID)
        self.model = BlipForConditionalGeneration.from_pretrained(
            self.MODEL_ID, torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)
        self.model.eval()
        print(f"  BLIP ready on {self.device}.")

    def caption(self, img: Image.Image) -> str:
        import torch
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            ids = self.model.generate(**inputs, max_new_tokens=50)
        return self.processor.decode(ids[0], skip_special_tokens=True)


# ---------------------------------------------------------------------------
# CONTACT SHEETS
# ---------------------------------------------------------------------------

def save_contact_sheet(
    records: List[Dict],
    class_id: int,
    disease: str,
    out_path: Path,
    n_samples: int = 20,
    cols: int = 5,
    rng: Optional[random.Random] = None,
):
    if rng is None:
        rng = random.Random(42)
    class_recs = [r for r in records if r["class_id"] == class_id]
    if not class_recs:
        print(f"  [class {class_id}] No crops found.")
        return
    sample = rng.sample(class_recs, min(n_samples, len(class_recs)))
    imgs = []
    for r in sample:
        try:
            img = Image.open(r["img_path"]).convert("RGB")
            imgs.append(crop_image(img, r["corners"]))
        except Exception:
            pass
    if not imgs:
        return

    rows   = (len(imgs) + cols - 1) // cols
    pad    = 4
    header = 28
    W      = cols * (TARGET_SIZE + pad) + pad
    H      = rows * (TARGET_SIZE + pad) + pad + header
    sheet  = Image.new("RGB", (W, H), (40, 40, 40))
    draw   = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
    draw.text((pad, 4),
              f"Class {class_id} -> {disease}  ({len(class_recs)} total boxes)",
              fill=(255, 220, 80), font=font)
    for i, img in enumerate(imgs):
        row = i // cols; col = i % cols
        sheet.paste(img, (pad + col * (TARGET_SIZE + pad),
                          header + pad + row * (TARGET_SIZE + pad)))
    sheet.save(out_path)
    print(f"  Saved {out_path.name}  ({len(imgs)} crops shown)")


# ---------------------------------------------------------------------------
# SAVE INDIVIDUAL CROPS
# ---------------------------------------------------------------------------

def save_all_crops(records: List[Dict], crops_dir: Path):
    """Save every bounding-box crop as a JPEG organised by disease class."""
    crops_dir.mkdir(parents=True, exist_ok=True)
    counts: Dict[str, int] = {}
    for r in records:
        out_dir = crops_dir / r["disease"]
        out_dir.mkdir(exist_ok=True)
        stem = Path(r["image_file"]).stem
        out_path = out_dir / f"{stem}_box{r['box_idx']:02d}.jpg"
        if out_path.exists():
            counts[r["disease"]] = counts.get(r["disease"], 0) + 1
            continue
        try:
            img  = Image.open(r["img_path"]).convert("RGB")
            crop = crop_image(img, r["corners"])
            crop.save(out_path, "JPEG", quality=92)
            counts[r["disease"]] = counts.get(r["disease"], 0) + 1
        except Exception as e:
            print(f"    WARN {out_path.name}: {e}")
    print(f"\n  Crops saved to {crops_dir}")
    for disease, n in sorted(counts.items()):
        print(f"    {disease:<22}: {n:>4} crops")


# ---------------------------------------------------------------------------
# MAIN ANNOTATION  (template or BLIP)
# ---------------------------------------------------------------------------

def annotate(
    records: List[Dict],
    out_path: Path,
    use_blip: bool = False,
    seed: int = 42,
):
    rng = random.Random(seed)

    captioner: Optional[BLIPCaptioner] = None
    if use_blip:
        try:
            captioner = BLIPCaptioner()
        except Exception as e:
            print(f"  WARNING: Could not load BLIP ({e}). Falling back to templates.")

    try:
        from tqdm import tqdm
        iterator = tqdm(records, desc="  Annotating", unit="crop",
                        file=sys.stdout, dynamic_ncols=True)
    except ImportError:
        iterator = records

    rows = []
    for r in iterator:
        disease = r["disease"]
        if captioner is not None:
            try:
                img  = Image.open(r["img_path"]).convert("RGB")
                crop = crop_image(img, r["corners"])
                blip_cap = captioner.caption(crop)
                text = generate_text_blip(blip_cap, disease, rng)
            except Exception:
                text = generate_text_plain(disease, rng)
        else:
            text = generate_text_plain(disease, rng)

        rows.append([
            r["image_file"],
            r["box_idx"],
            r["class_id"],
            disease,
            text,
        ])

    # Write to a temp file first, then rename — avoids PermissionError if
    # the target CSV is open in Excel or another program.
    tmp_path = out_path.with_suffix(".tmp")
    with open(tmp_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_file", "box_idx", "class_id", "disease", "text"])
        writer.writerows(rows)
    try:
        if out_path.exists():
            out_path.unlink()
        tmp_path.rename(out_path)
    except PermissionError:
        print(f"\n  WARNING: Could not overwrite {out_path.name} — it may be open.")
        print(f"  Annotations saved to: {tmp_path}")
        print(f"  Close the file and rename {tmp_path.name} -> {out_path.name} manually.")
        out_path = tmp_path

    mode = "BLIP + literature" if (captioner is not None) else "literature templates"
    print(f"\n  Saved {len(rows)} annotations ({mode}) -> {out_path}")


# ---------------------------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------------------------

def print_summary(records: List[Dict], class_map: Dict[int, str]):
    counts = Counter(r["disease"] for r in records)
    print("\n" + "=" * 58)
    print("DATASET SUMMARY")
    print("=" * 58)
    print(f"  {'Class':>5}  {'Disease':<22}  {'Boxes':>5}")
    print(f"  {'-'*48}")
    for cls_id in sorted(class_map):
        name = class_map[cls_id]
        print(f"  {cls_id:>5}  {name:<22}  {counts.get(name, 0):>5}")
    print(f"  {'-'*48}")
    print(f"  {'TOTAL':<29}  {sum(counts.values()):>5}")
    print("=" * 58)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Tea Leaf Disease Annotator — generate text prompts for every OBB crop"
    )
    parser.add_argument("--data_dir",     default="C:/Users/USER_HP/Desktop/FarmFederate/Real Dataset")
    parser.add_argument("--out_dir",      default="C:/Users/USER_HP/Desktop/FarmFederate/tea_results/annotation")
    parser.add_argument("--auto_caption", action="store_true",
                        help="Use BLIP to visually describe each crop (~15 min on CPU)")
    parser.add_argument("--save_crops",   action="store_true",
                        help="Save every crop as an individual JPEG under out_dir/crops/")
    parser.add_argument("--samples",      type=int, default=20,
                        help="Crops per contact sheet (default 20)")
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--remap",        nargs="*", metavar="ID:DISEASE",
                        help="Override class mapping, e.g. --remap 0:algal_leaf_spot 3:brown_blight")
    args = parser.parse_args()

    # Build class map
    class_map = dict(CLASS_MAP)
    if args.remap:
        for item in args.remap:
            cls_id, disease = item.split(":", 1)
            class_map[int(cls_id)] = disease.strip()

    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mode_str = "BLIP auto-caption + literature" if args.auto_caption else "literature templates"
    print("=" * 58)
    print("TEA LEAF DISEASE ANNOTATOR")
    print("=" * 58)
    print(f"  Mode     : {mode_str}")
    print(f"  Data dir : {data_dir}")
    print(f"  Output   : {out_dir}")
    print(f"\n  Class mapping:")
    for k, v in sorted(class_map.items()):
        print(f"    {k} -> {v}")

    # ── Load records
    print("\n[1] Loading OBB crops ...")
    records = load_all_crops(data_dir, class_map)
    print(f"  Found {len(records)} boxes across "
          f"{len(set(r['image_file'] for r in records))} images")
    print_summary(records, class_map)

    # ── Contact sheets
    print("\n[2] Saving contact sheets ...")
    rng = random.Random(args.seed)
    for cls_id, disease in sorted(class_map.items()):
        save_contact_sheet(records, cls_id, disease,
                           out_dir / f"contact_class_{cls_id}_{disease}.png",
                           n_samples=args.samples, rng=rng)

    # ── Save individual crops
    if args.save_crops:
        print("\n[3] Saving individual crop images ...")
        save_all_crops(records, out_dir / "crops")

    # ── Generate text annotations
    step = 4 if args.save_crops else 3
    print(f"\n[{step}] Generating text annotations ...")
    if args.auto_caption:
        print("  This will take ~15-20 min on CPU. Progress shown below.")
        print("  First run downloads BLIP (~450 MB) to HuggingFace cache.\n")
    annotate(records, out_dir / "annotations.csv",
             use_blip=args.auto_caption, seed=args.seed)

    print("\n" + "=" * 58)
    print("DONE")
    print("=" * 58)
    print(f"\n  annotations.csv  -> {out_dir / 'annotations.csv'}")
    if args.save_crops:
        print(f"  crops/           -> {out_dir / 'crops'}")
    print(f"\n  To train with these annotations:")
    print(f"    python backend/tea_disease_federated.py \\")
    print(f"      --annotations \"{out_dir / 'annotations.csv'}\"")


if __name__ == "__main__":
    main()
