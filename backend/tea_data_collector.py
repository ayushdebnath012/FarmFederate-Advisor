# -*- coding: utf-8 -*-
"""
Tea Leaf Disease - Combined Dataset Collector
=============================================
Downloads and merges three real sources into one 8-class dataset:

  Source 1 (TeaLeafNet)   harjindersinghdibru/tealeafnet
      4 classes  x 1250 images = 5000 images (balanced, classification format)
      BB  -> brown_blight
      GL  -> gray_blight
      RR  -> red_rust
      RSM -> red_spot_margin

  Source 2 (Fahmida)      fahmidahossainmaria/tea-leaf-disease-dataset
      6 classes, YOLO bbox format  (cropped to patches here)
      Algal Leaf Spot -> algal_leaf_spot
      Brown Blight    -> brown_blight     (merged with TeaLeafNet BB)
      Gray Blight     -> gray_blight      (merged with TeaLeafNet GL)
      Healthy         -> healthy
      Helopeltis      -> helopeltis
      Red Leaf Spot   -> red_leaf_spot

  Source 3 (Real OBB)     user's C:/Users/USER_HP/Desktop/FarmFederate/Real Dataset
      5 classes, YOLO OBB format (cropped to patches here)
      Class 0 (129) -> brown_blight      (most common tea disease)
      Class 1 (  9) -> helopeltis        (rare insect damage)
      Class 2 ( 67) -> gray_blight
      Class 3 (102) -> algal_leaf_spot
      Class 4 ( 64) -> red_leaf_spot
      NOTE: mapping is a best-guess. Edit REAL_OBB_CLASS_MAP to override.

Final combined dataset:
    tea_combined/
        brown_blight/        algal_leaf_spot/
        gray_blight/         healthy/
        red_rust/            helopeltis/
        red_spot_margin/     red_leaf_spot/

Usage:
    python tea_data_collector.py
    python tea_data_collector.py --skip_download   # if already downloaded
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

COMBINED_DIR  = Path("C:/Users/USER_HP/Desktop/FarmFederate/tea_combined")
DOWNLOAD_DIR  = Path("C:/Users/USER_HP/Desktop/FarmFederate/tea_downloads")
REAL_OBB_DIR  = Path("C:/Users/USER_HP/Desktop/FarmFederate/Real Dataset")

# 8 unified disease classes
COMBINED_CLASSES = [
    "brown_blight",      # 0
    "gray_blight",       # 1
    "red_rust",          # 2
    "red_spot_margin",   # 3
    "algal_leaf_spot",   # 4
    "healthy",           # 5
    "helopeltis",        # 6
    "red_leaf_spot",     # 7
]

# TeaLeafNet folder abbreviation -> unified class
TEALEAFNET_MAP: Dict[str, str] = {
    "BB":  "brown_blight",
    "GL":  "gray_blight",
    "RR":  "red_rust",
    "RSM": "red_spot_margin",
}

# Fahmida dataset class name -> unified class
FAHMIDA_MAP: Dict[str, str] = {
    "Algal Leaf Spot": "algal_leaf_spot",
    "Brown Blight":    "brown_blight",
    "Gray Blight":     "gray_blight",
    "Healthy":         "healthy",
    "Helopeltis":      "helopeltis",
    "Red Leaf Spot":   "red_leaf_spot",
}

# Real OBB class id -> unified class
# Corrected via visual comparison with reference disease images (2026-04-14):
#   Class 0: large gray-white necrotic blotches  -> gray_blight
#   Class 1: young curled leaves, puncture marks -> helopeltis
#   Class 2: orange-brown circular spots         -> algal_leaf_spot
#   Class 3: large scattered necrotic lesions    -> brown_blight
#   Class 4: small bird's-eye circular spots     -> red_leaf_spot
REAL_OBB_CLASS_MAP: Dict[int, str] = {
    0: "gray_blight",
    1: "helopeltis",
    2: "algal_leaf_spot",
    3: "brown_blight",
    4: "red_leaf_spot",
}

CROP_PADDING = 0.10
TARGET_SIZE  = 224

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def ensure_dirs():
    for cls in COMBINED_CLASSES:
        (COMBINED_DIR / cls).mkdir(parents=True, exist_ok=True)
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)


def save_crop(img: Image.Image, out_path: Path):
    img.convert("RGB").resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS).save(out_path, "JPEG", quality=90)


def _obb_to_rect(corners: List[float], W: int, H: int) -> Tuple[float, float, float, float]:
    """OBB 8-coord (normalised) -> axis-aligned bbox with padding."""
    xs = [corners[i] * W for i in range(0, 8, 2)]
    ys = [corners[i] * H for i in range(1, 8, 2)]
    px, py = CROP_PADDING * W, CROP_PADDING * H
    x0 = max(0, min(xs) - px)
    y0 = max(0, min(ys) - py)
    x1 = min(W, max(xs) + px)
    y1 = min(H, max(ys) + py)
    if x1 - x0 < 8: x0, x1 = max(0, x0-8), min(W, x1+8)
    if y1 - y0 < 8: y0, y1 = max(0, y0-8), min(H, y1+8)
    return x0, y0, x1, y1


def _yolo_bbox_to_rect(parts: List[float], W: int, H: int) -> Tuple[float, float, float, float]:
    """Standard YOLO bbox (cx cy w h normalised) -> pixel rect with padding."""
    cx, cy, bw, bh = parts
    px, py = CROP_PADDING * W, CROP_PADDING * H
    x0 = max(0, (cx - bw/2) * W - px)
    y0 = max(0, (cy - bh/2) * H - py)
    x1 = min(W, (cx + bw/2) * W + px)
    y1 = min(H, (cy + bh/2) * H + py)
    return x0, y0, x1, y1


def _unique_name(out_dir: Path, prefix: str) -> Path:
    i = 0
    while True:
        p = out_dir / f"{prefix}_{i:05d}.jpg"
        if not p.exists():
            return p
        i += 1

# ---------------------------------------------------------------------------
# SOURCE 1 — TeaLeafNet  (already downloaded to DOWNLOAD_DIR)
# ---------------------------------------------------------------------------

def process_tealeafnet(tealeafnet_root: Path) -> Dict[str, int]:
    """Copy TeaLeafNet full-image jpegs into COMBINED_DIR."""
    counts: Dict[str, int] = {}
    root = tealeafnet_root / "5000_tea_leaf_with_blackbg_geotagged"
    if not root.exists():
        print(f"  [TeaLeafNet] Not found at {root}. Run with --download first.")
        return counts

    for folder, unified_cls in TEALEAFNET_MAP.items():
        src_dir = root / folder
        if not src_dir.exists():
            print(f"  [TeaLeafNet] Missing folder {src_dir}")
            continue
        out_dir = COMBINED_DIR / unified_cls
        n = 0
        for img_path in src_dir.glob("*.jpg"):
            dst = _unique_name(out_dir, f"tln_{folder}")
            try:
                img = Image.open(img_path)
                save_crop(img, dst)
                n += 1
            except Exception as e:
                print(f"    WARN: {img_path.name}: {e}")
        print(f"  [TeaLeafNet] {folder} ({unified_cls}): {n} images")
        counts[unified_cls] = counts.get(unified_cls, 0) + n
    return counts


# ---------------------------------------------------------------------------
# SOURCE 2 — Fahmida YOLO bbox dataset
# ---------------------------------------------------------------------------

def download_fahmida(out_dir: Path):
    import kaggle
    api = kaggle.KaggleApi()
    api.authenticate()
    print("  Downloading fahmidahossainmaria/tea-leaf-disease-dataset ...")
    api.dataset_download_files(
        "fahmidahossainmaria/tea-leaf-disease-dataset",
        path=str(out_dir / "fahmida"),
        unzip=True,
        quiet=False,
    )
    print("  Download complete.")


def process_fahmida(fahmida_root: Path) -> Dict[str, int]:
    """
    Parse YOLO bbox labels, crop each bounding box, save to COMBINED_DIR.
    Expects structure:  fahmida_root/Tea Leaf Diseases/{train,valid,test}/images + labels
    """
    counts: Dict[str, int] = {}
    dataset_root = fahmida_root / "Tea Leaf Diseases"
    if not dataset_root.exists():
        # try flat
        dataset_root = fahmida_root
    if not dataset_root.exists():
        print(f"  [Fahmida] Not found at {fahmida_root}. Run with --download first.")
        return counts

    # Read class names from data.yaml
    yaml_path = dataset_root / "data.yaml"
    class_names: List[str] = list(FAHMIDA_MAP.keys())
    if yaml_path.exists():
        with open(yaml_path) as f:
            for line in f:
                if line.strip().startswith("names:"):
                    import ast
                    raw = line.split(":", 1)[1].strip()
                    try:
                        class_names = ast.literal_eval(raw)
                    except Exception:
                        pass
                    break

    for split in ("train", "valid", "test"):
        img_dir = dataset_root / split / "images"
        lbl_dir = dataset_root / split / "labels"
        if not img_dir.exists():
            continue

        for lbl_file in lbl_dir.glob("*.txt"):
            img_path = img_dir / (lbl_file.stem + ".jpg")
            if not img_path.exists():
                img_path = img_dir / (lbl_file.stem + ".jpeg")
            if not img_path.exists():
                continue

            try:
                img = Image.open(img_path).convert("RGB")
                W, H = img.size
            except Exception:
                continue

            with open(lbl_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls_id = int(parts[0])
                    if cls_id >= len(class_names):
                        continue
                    src_name = class_names[cls_id]
                    unified_cls = FAHMIDA_MAP.get(src_name)
                    if unified_cls is None:
                        continue

                    x0, y0, x1, y1 = _yolo_bbox_to_rect(
                        [float(v) for v in parts[1:5]], W, H
                    )
                    crop = img.crop((x0, y0, x1, y1))
                    out_dir = COMBINED_DIR / unified_cls
                    dst = _unique_name(out_dir, f"fhm_{split}")
                    save_crop(crop, dst)
                    counts[unified_cls] = counts.get(unified_cls, 0) + 1

    for cls, n in sorted(counts.items()):
        print(f"  [Fahmida] {cls}: {n} crops")
    return counts


# ---------------------------------------------------------------------------
# SOURCE 3 — Real OBB dataset (user's data)
# ---------------------------------------------------------------------------

def process_real_obb(real_dir: Path) -> Dict[str, int]:
    """Crop OBB instances from the user's Real Dataset."""
    counts: Dict[str, int] = {}
    img_dir = real_dir / "images"
    lbl_dir = real_dir / "labels"
    if not img_dir.exists():
        print(f"  [RealOBB] images dir not found at {img_dir}")
        return counts

    for lbl_file in lbl_dir.glob("*.txt"):
        img_path = img_dir / (lbl_file.stem + ".jpg")
        if not img_path.exists():
            continue
        try:
            img = Image.open(img_path).convert("RGB")
            W, H = img.size
        except Exception:
            continue

        with open(lbl_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 9:
                    continue
                cls_id = int(parts[0])
                unified_cls = REAL_OBB_CLASS_MAP.get(cls_id)
                if unified_cls is None:
                    continue
                corners = [float(v) for v in parts[1:9]]
                x0, y0, x1, y1 = _obb_to_rect(corners, W, H)
                crop = img.crop((x0, y0, x1, y1))
                out_dir = COMBINED_DIR / unified_cls
                dst = _unique_name(out_dir, f"real_c{cls_id}")
                save_crop(crop, dst)
                counts[unified_cls] = counts.get(unified_cls, 0) + 1

    for cls, n in sorted(counts.items()):
        print(f"  [RealOBB] {cls}: {n} crops")
    return counts


# ---------------------------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------------------------

def print_summary():
    from collections import Counter
    print("\n" + "="*60)
    print("COMBINED DATASET SUMMARY")
    print("="*60)
    total = 0
    for cls in COMBINED_CLASSES:
        n = len(list((COMBINED_DIR / cls).glob("*.jpg")))
        print(f"  {cls:<22}: {n:>5} images")
        total += n
    print(f"  {'TOTAL':<22}: {total:>5} images")
    print(f"  Classes: {len(COMBINED_CLASSES)}")
    print(f"  Output : {COMBINED_DIR}")
    print("="*60)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    global COMBINED_DIR, DOWNLOAD_DIR, REAL_OBB_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_download", action="store_true",
                        help="Skip Kaggle downloads (use if already downloaded)")
    parser.add_argument("--skip_tealeafnet", action="store_true")
    parser.add_argument("--skip_fahmida",    action="store_true")
    parser.add_argument("--skip_real_obb",   action="store_true")
    parser.add_argument("--combined_dir",    default=str(COMBINED_DIR))
    parser.add_argument("--download_dir",    default=str(DOWNLOAD_DIR))
    parser.add_argument("--real_dir",        default=str(REAL_OBB_DIR))
    args = parser.parse_args()

    COMBINED_DIR  = Path(args.combined_dir)
    DOWNLOAD_DIR  = Path(args.download_dir)
    REAL_OBB_DIR  = Path(args.real_dir)

    # Recreate combined dir (clean build)
    if COMBINED_DIR.exists():
        print(f"Clearing existing combined dir: {COMBINED_DIR}")
        shutil.rmtree(COMBINED_DIR)
    ensure_dirs()

    print("\n" + "="*60)
    print("TEA LEAF DISEASE - DATASET COLLECTOR")
    print("="*60)

    # ── Source 1: TeaLeafNet
    if not args.skip_tealeafnet:
        tealeafnet_dir = DOWNLOAD_DIR / "tealeafnet"
        if not (tealeafnet_dir / "5000_tea_leaf_with_blackbg_geotagged").exists() and not args.skip_download:
            print("\n[1/3] Downloading TeaLeafNet from Kaggle ...")
            import kaggle
            api = kaggle.KaggleApi()
            api.authenticate()
            api.dataset_download_files(
                "harjindersinghdibru/tealeafnet",
                path=str(tealeafnet_dir),
                unzip=True, quiet=False,
            )
        elif (DOWNLOAD_DIR / "5000_tea_leaf_with_blackbg_geotagged").exists():
            tealeafnet_dir = DOWNLOAD_DIR  # already extracted here
        print("\n[1/3] Processing TeaLeafNet ...")
        process_tealeafnet(tealeafnet_dir)

    # ── Source 2: Fahmida
    if not args.skip_fahmida:
        fahmida_dir = DOWNLOAD_DIR / "fahmida"
        if not fahmida_dir.exists() and not args.skip_download:
            print("\n[2/3] Downloading fahmida tea-leaf-disease-dataset from Kaggle ...")
            download_fahmida(DOWNLOAD_DIR)
        print("\n[2/3] Processing Fahmida dataset ...")
        process_fahmida(fahmida_dir)

    # ── Source 3: Real OBB
    if not args.skip_real_obb:
        print("\n[3/3] Processing Real OBB dataset ...")
        process_real_obb(REAL_OBB_DIR)

    print_summary()
    print(f"\nDone! Run tea_disease_federated.py with:")
    print(f"  python backend/tea_disease_federated.py --data_dir \"{COMBINED_DIR}\"")


if __name__ == "__main__":
    main()
