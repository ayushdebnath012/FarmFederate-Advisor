#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
download_stress_images.py
=========================
Downloads CORRECT per-class stress images. Strategy per class:

  water_stress  - Kaggle: zoya77/agricultural-water-stress-image-dataset (90 real)
                  + synthetic padding to TARGET
  nutrient_def  - HuggingFace PlantVillage filtered for yellowing/chlorosis classes
  pest_risk     - HuggingFace PlantVillage filtered for spider_mites / rust classes
  disease_risk  - Kaggle: nirmalsankalana/plantdoc-dataset (DONE - 1000 images)
  heat_stress   - Synthetic images (no suitable public image dataset)

Output: data/{stress}/images/
"""

from __future__ import annotations
import sys, shutil, random, io
from pathlib import Path

TARGET          = 800           # images per class
SEED            = 42
DATA_DIR        = Path(__file__).parent.parent / "data"
TMP_DIR         = Path(__file__).parent.parent / "kaggle_tmp"
STRESS_LABELS   = ["water_stress", "nutrient_def", "pest_risk", "disease_risk", "heat_stress"]

# HuggingFace PlantVillage class names that best represent each stress type
HF_CLASS_MAP = {
    "nutrient_def": [
        "Grape___Esca_(Black_Measles)",
        "Tomato___Leaf_Mold",
        "Corn_(maize)___Northern_Leaf_Blight",
        "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
        "Tomato___Septoria_leaf_spot",
        "Potato___Early_blight",
    ],
    "pest_risk": [
        "Tomato___Spider_mites Two-spotted_spider_mite",
        "Apple___Cedar_apple_rust",
        "Grape___Black_rot",
        "Tomato___Bacterial_spot",
        "Squash___Powdery_mildew",
    ],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def setup_kaggle():
    try:
        import kaggle  # noqa
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "kaggle"])
    import kaggle
    kaggle.api.authenticate()
    print("[OK] Kaggle credentials set")


def _force_remove(func, path, _):
    import os, stat
    os.chmod(path, stat.S_IWRITE)
    func(path)


def save_images(paths: list[Path], dest_dir: Path, prefix: str, clear: bool):
    if clear and dest_dir.exists():
        shutil.rmtree(dest_dir, onexc=_force_remove)
        print(f"    Cleared: {dest_dir}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for i, src in enumerate(paths):
        dst = dest_dir / f"{prefix}_{i:05d}{src.suffix.lower()}"
        try:
            shutil.copy2(src, dst)
            saved += 1
        except Exception as e:
            print(f"    [WARN] {src}: {e}")
    return saved


def collect_images(root: Path, keywords: list[str], max_n: int,
                   rng: random.Random) -> list[Path]:
    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    folder_images: dict[str, list[Path]] = {}
    for p in root.rglob("*"):
        if p.suffix.lower() in IMG_EXTS:
            folder_images.setdefault(p.parent.name.lower(), []).append(p)

    scored = []
    for fname, imgs in folder_images.items():
        score = sum(1 for kw in keywords if kw in fname)
        scored.append((score, imgs))
    scored.sort(key=lambda x: -x[0])

    selected: list[Path] = []
    for _, imgs in scored:
        if len(selected) >= max_n:
            break
        rng.shuffle(imgs)
        selected.extend(imgs[:max_n - len(selected)])
    rng.shuffle(selected)
    return selected[:max_n]


# ---------------------------------------------------------------------------
# Kaggle download (water_stress + disease_risk already done)
# ---------------------------------------------------------------------------

def download_kaggle_class(stress: str, dataset_id: str, keywords: list[str],
                          rng: random.Random, target: int) -> int:
    import kaggle
    tmp = TMP_DIR / stress
    tmp.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading kaggle:{dataset_id}")
    try:
        kaggle.api.dataset_download_files(dataset_id, path=str(tmp),
                                          unzip=True, quiet=False)
    except Exception as e:
        print(f"  [ERROR] {e}")
        return 0

    imgs = collect_images(tmp, keywords, target, rng)
    if not imgs:
        imgs = collect_images(tmp, [], target, rng)
        if imgs:
            print(f"  [WARN] No keyword match - using any {len(imgs)} images")
    print(f"  Found: {len(imgs)} images")
    dest = DATA_DIR / stress / "images"
    return save_images(imgs, dest, prefix=stress[:4], clear=True)


# ---------------------------------------------------------------------------
# HuggingFace download (nutrient_def + pest_risk)
# ---------------------------------------------------------------------------

def download_hf_class(stress: str, target_classes: list[str],
                      rng: random.Random, target: int) -> int:
    """Download PlantVillage from HuggingFace, keep only target_classes."""
    try:
        from datasets import load_dataset
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "datasets"])
        from datasets import load_dataset

    print(f"  Loading BrandonFors/Plant-Diseases-PlantVillage-Dataset from HuggingFace")
    try:
        ds = load_dataset("BrandonFors/Plant-Diseases-PlantVillage-Dataset",
                          split="train", trust_remote_code=True)
    except Exception as e:
        print(f"  [ERROR] HF load failed: {e}")
        return 0

    # Get label names
    label_feature = ds.features.get("label") or ds.features.get("labels")
    if hasattr(label_feature, "names"):
        label_names = label_feature.names
    else:
        label_names = []
    print(f"  HF dataset: {len(ds)} samples, {len(label_names)} classes")

    # Find indices of target classes
    target_indices = set()
    for i, name in enumerate(label_names):
        if any(tc.lower() in name.lower() or name.lower() in tc.lower()
               for tc in target_classes):
            target_indices.add(i)
            print(f"    Matched class: {name}")

    if not target_indices and label_names:
        print(f"  [WARN] No class match - using all classes")
        target_indices = set(range(len(label_names)))

    # Filter dataset
    label_col = "label" if "label" in ds.column_names else "labels"
    filtered = [ex for ex in ds if ex[label_col] in target_indices]
    rng.shuffle(filtered)
    filtered = filtered[:target]
    print(f"  Filtered: {len(filtered)} samples")

    # Save images
    from PIL import Image as PILImage
    dest = DATA_DIR / stress / "images"
    if dest.exists():
        shutil.rmtree(dest, onexc=_force_remove)
        print(f"    Cleared: {dest}")
    dest.mkdir(parents=True, exist_ok=True)

    saved = 0
    for i, ex in enumerate(filtered):
        try:
            img = ex.get("image") or ex.get("img")
            if img is None:
                continue
            if not isinstance(img, PILImage.Image):
                img = PILImage.fromarray(img)
            img = img.convert("RGB")
            img.save(dest / f"{stress[:4]}_{i:05d}.jpg")
            saved += 1
        except Exception as e:
            print(f"    [WARN] sample {i}: {e}")
    return saved


# ---------------------------------------------------------------------------
# Synthetic generation — uses real PlantVillage healthy images as base
# then applies stress-specific visual transforms on top.
# ---------------------------------------------------------------------------

def _load_pv_healthy(n: int, rng: random.Random):
    """Return up to n PIL images from PlantVillage 'healthy' class."""
    try:
        from datasets import load_dataset
        from PIL import Image as PILImage
    except ImportError:
        return []
    print(f"  Loading PlantVillage healthy base images from HuggingFace...")
    try:
        ds = load_dataset("BrandonFors/Plant-Diseases-PlantVillage-Dataset",
                          split="train")
        label_feature = ds.features.get("label") or ds.features.get("labels")
        label_names = label_feature.names if hasattr(label_feature, "names") else []
        healthy_idx = {i for i, nm in enumerate(label_names) if "healthy" in nm.lower()}
        label_col = "label" if "label" in ds.column_names else "labels"
        healthy = [ex for ex in ds if ex[label_col] in healthy_idx]
        rng.shuffle(healthy)
        healthy = healthy[:n]
        imgs = []
        for ex in healthy:
            img = ex.get("image") or ex.get("img")
            if img is None:
                continue
            if not isinstance(img, PILImage.Image):
                import numpy as np
                img = PILImage.fromarray(img)
            imgs.append(img.convert("RGB").resize((224, 224)))
        print(f"  Loaded {len(imgs)} healthy base images")
        return imgs
    except Exception as e:
        print(f"  [WARN] HF load failed ({e}), using procedural base")
        return []


def _apply_heat_stress(img_arr, np_rng):
    """Apply heat scorch: brown/tan leaf edges + bleached patches + tip burn."""
    import numpy as np
    H, W = img_arr.shape[:2]
    out = img_arr.astype(np.float32) / 255.0

    # 1. Tip burn — top 20-35% of image fades to tan/brown
    burn_h = int(H * (0.20 + np_rng.random() * 0.15))
    fade = np.linspace(1.0, 0.0, burn_h)
    for row in range(burn_h):
        f = fade[row]
        out[row, :, 0] = out[row, :, 0] * (1 - f) + 0.72 * f   # tan R
        out[row, :, 1] = out[row, :, 1] * (1 - f) + 0.52 * f   # tan G
        out[row, :, 2] = out[row, :, 2] * (1 - f) + 0.18 * f   # tan B

    # 2. Edge scorch — dark brown border
    border = int(8 + np_rng.random() * 12)
    scorch_color = np.array([0.45, 0.28, 0.08], dtype=np.float32)
    for b in range(border):
        alpha = (border - b) / border * 0.75
        out[b, :] = out[b, :] * (1 - alpha) + scorch_color * alpha
        out[H-1-b, :] = out[H-1-b, :] * (1 - alpha) + scorch_color * alpha
        out[:, b] = out[:, b] * (1 - alpha) + scorch_color * alpha
        out[:, W-1-b] = out[:, W-1-b] * (1 - alpha) + scorch_color * alpha

    # 3. Irregular inter-veinal bleaching — uses accumulated Gaussian blobs
    #    with per-patch color variation and soft distance-based alpha.
    #    Bleaching is stronger where green channel is lower (inter-veinal areas).
    bleach_map = np.zeros((H, W), dtype=np.float32)
    n_seeds = int(12 + np_rng.random() * 18)          # many small seeds → organic shape
    for _ in range(n_seeds):
        cx = np_rng.integers(15, W - 15)
        cy = np_rng.integers(15, H - 15)
        sigma_x = 4 + np_rng.random() * 14            # anisotropic spread
        sigma_y = 4 + np_rng.random() * 10
        yy, xx = np.mgrid[:H, :W]
        blob = np.exp(-((xx - cx)**2 / (2 * sigma_x**2) +
                        (yy - cy)**2 / (2 * sigma_y**2)))
        bleach_map += blob * (0.4 + np_rng.random() * 0.6)

    # Normalise to 0–1
    bleach_map = np.clip(bleach_map / (bleach_map.max() + 1e-6), 0, 1)

    # Keep only the top 25-40% brightest regions (creates natural patch gaps)
    threshold = 0.60 + np_rng.random() * 0.15
    bleach_map = np.where(bleach_map > threshold,
                          (bleach_map - threshold) / (1 - threshold), 0.0)

    # Inter-veinal bias: bleach more where green is already lower
    green_inv = 1.0 - out[:, :, 1]
    bleach_map = bleach_map * (0.5 + 0.5 * green_inv)
    bleach_map = np.clip(bleach_map, 0, 1)

    # Per-pixel bleach color varies slightly (pale yellow → pale white)
    bleach_r = np.full((H, W), 0.93, np.float32) + np_rng.uniform(-0.04,  0.04, (H, W)).astype(np.float32)
    bleach_g = np.full((H, W), 0.87, np.float32) + np_rng.uniform(-0.05,  0.05, (H, W)).astype(np.float32)
    bleach_b = np.full((H, W), 0.58, np.float32) + np_rng.uniform(-0.06,  0.10, (H, W)).astype(np.float32)
    alpha_map = bleach_map * (0.50 + np_rng.random() * 0.30)   # max blend 50-80%
    out[:, :, 0] = out[:, :, 0] * (1 - alpha_map) + bleach_r * alpha_map
    out[:, :, 1] = out[:, :, 1] * (1 - alpha_map) + bleach_g * alpha_map
    out[:, :, 2] = out[:, :, 2] * (1 - alpha_map) + bleach_b * alpha_map

    # 4. Overall warm shift (heat look)
    out[:, :, 0] = np.clip(out[:, :, 0] * 1.12, 0, 1)
    out[:, :, 2] = np.clip(out[:, :, 2] * 0.82, 0, 1)

    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def _apply_water_stress(img_arr, np_rng):
    """Apply water stress: grey-green shift + wilting gradient + rolled leaves."""
    import numpy as np
    H, W = img_arr.shape[:2]
    out = img_arr.astype(np.float32) / 255.0

    # 1. Desaturate toward grey-green (water stressed leaves look dull)
    grey = out.mean(axis=2, keepdims=True)
    desat = 0.35 + np_rng.random() * 0.25
    out = out * (1 - desat) + grey * desat
    # Slight blue-grey tint
    out[:, :, 2] = np.clip(out[:, :, 2] * 1.08, 0, 1)
    out[:, :, 0] = np.clip(out[:, :, 0] * 0.92, 0, 1)

    # 2. Wilting gradient — bottom half slightly darker and drooping
    wilt_strength = 0.25 + np_rng.random() * 0.20
    gradient = np.linspace(0, wilt_strength, H)[:, None, None]
    out = out * (1 - gradient)

    # 3. Leaf roll / curl — darken thin vertical strips to simulate rolling
    n_rolls = int(2 + np_rng.random() * 4)
    for _ in range(n_rolls):
        cx = np_rng.integers(10, W-10)
        width = int(3 + np_rng.random() * 8)
        x1, x2 = max(0, cx-width//2), min(W, cx+width//2)
        out[:, x1:x2] *= 0.60 + np_rng.random() * 0.15

    # 4. Dry leaf tips — top rows turn pale tan/yellow
    tip_h = int(H * (0.08 + np_rng.random() * 0.12))
    tan = np.array([0.78, 0.72, 0.45], dtype=np.float32)
    for row in range(tip_h):
        f = (tip_h - row) / tip_h * 0.70
        out[row, :] = out[row, :] * (1 - f) + tan * f

    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def generate_stress_from_healthy(stress: str, apply_fn, n: int,
                                  existing: int, rng: random.Random) -> int:
    """Load healthy plant images, apply stress transform, save."""
    import numpy as np
    from PIL import Image as PILImage

    dest = DATA_DIR / stress / "images"
    dest.mkdir(parents=True, exist_ok=True)

    base_imgs = _load_pv_healthy(n, rng)

    if not base_imgs:
        # Procedural fallback: create a simple leaf-like background
        print("  Using procedural leaf base (no HF images available)")
        base_imgs = []
        for _ in range(n):
            np_rng2 = np.random.default_rng(rng.randint(0, 2**31))
            H = W = 224
            # Green gradient base simulating a leaf
            r_base = 0.15 + np_rng2.random() * 0.10
            g_base = 0.45 + np_rng2.random() * 0.15
            b_base = 0.10 + np_rng2.random() * 0.08
            arr = np.stack([
                np.full((H, W), r_base, np.float32),
                np.full((H, W), g_base, np.float32),
                np.full((H, W), b_base, np.float32),
            ], axis=2)
            # Add leaf vein structure — lighter midrib and veins
            for vein_x in [W//2] + [W//4, 3*W//4]:
                arr[:, max(0,vein_x-2):vein_x+2, 1] += 0.08
            # Add organic texture via perlin-like noise
            noise = np_rng2.normal(0, 0.04, (H, W, 3)).astype(np.float32)
            arr = np.clip(arr + noise, 0, 1)
            base_imgs.append(PILImage.fromarray((arr * 255).astype(np.uint8)))

    saved = 0
    for i, pil_img in enumerate(base_imgs):
        np_rng = np.random.default_rng(rng.randint(0, 2**31))
        arr = np.array(pil_img.resize((224, 224)))
        stressed = apply_fn(arr, np_rng)
        out = PILImage.fromarray(stressed)
        fname = dest / f"syn_{i + existing:05d}.jpg"
        out.save(fname, quality=88)
        saved += 1

    return saved


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    setup_kaggle()
    rng = random.Random(SEED)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    print("\n" + "=" * 60)
    print("FIXING STRESS IMAGE DATASETS")
    print("=" * 60)

    # -- water_stress: keep 90 real, replace synthetic with proper stress images
    print("\n[water_stress]")
    # Remove old flat-noise synthetics, keep real images
    ws_dir = DATA_DIR / "water_stress" / "images"
    old_syn = list(ws_dir.glob("syn_*.jpg"))
    for f in old_syn:
        f.unlink()
    real_count = len(list(ws_dir.glob("wate_*.jpg")))
    print(f"  Removed {len(old_syn)} old synthetic | Real images: {real_count}")
    need = max(0, TARGET - real_count)
    if need > 0:
        print(f"  Generating {need} water-stress images from healthy base")
        n = generate_stress_from_healthy("water_stress", _apply_water_stress,
                                         need, real_count, rng)
        print(f"  Generated: {n}")
    results["water_stress"] = real_count + need

    # -- nutrient_def: HuggingFace PlantVillage filtered ----------------------
    print("\n[nutrient_def]")
    n = download_hf_class("nutrient_def", HF_CLASS_MAP["nutrient_def"], rng, TARGET)
    print(f"  Saved: {n}")
    results["nutrient_def"] = n

    # -- pest_risk: HuggingFace PlantVillage filtered -------------------------
    print("\n[pest_risk]")
    n = download_hf_class("pest_risk", HF_CLASS_MAP["pest_risk"], rng, TARGET)
    print(f"  Saved: {n}")
    results["pest_risk"] = n

    # -- disease_risk: already done (1000 plantdoc) ---------------------------
    print("\n[disease_risk]")
    existing = len(list((DATA_DIR / "disease_risk" / "images").glob("*.jpg")))
    print(f"  Already have {existing} images - skipping")
    results["disease_risk"] = existing

    # -- heat_stress: healthy base + heat stress transform --------------------
    print("\n[heat_stress]")
    dest = DATA_DIR / "heat_stress" / "images"
    if dest.exists():
        shutil.rmtree(dest, onexc=_force_remove)
    dest.mkdir(parents=True, exist_ok=True)
    n = generate_stress_from_healthy("heat_stress", _apply_heat_stress,
                                     TARGET, 0, rng)
    print(f"  Generated: {n}")
    results["heat_stress"] = n

    # Cleanup
    shutil.rmtree(TMP_DIR, ignore_errors=True)

    # Summary
    print("\n" + "=" * 60)
    print("FINAL IMAGE COUNTS")
    print("=" * 60)
    print(f"{'Class':<15} {'Images':>8}  {'Source'}")
    print("-" * 42)
    sources = {
        "water_stress": "90 real + synthetic",
        "nutrient_def": "HF PlantVillage filtered",
        "pest_risk":    "HF PlantVillage filtered",
        "disease_risk": "PlantDoc (Kaggle)",
        "heat_stress":  "Synthetic",
    }
    for stress in STRESS_LABELS:
        n = results.get(stress, 0)
        actual = len(list((DATA_DIR / stress / "images").glob("*.jpg")))
        print(f"{stress:<15} {actual:>8}  {sources[stress]}")

    print("\nDone. Upload data/ to Google Drive then re-run training.")
    return results


if __name__ == "__main__":
    run()
