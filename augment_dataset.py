"""
Augment Merged-Dataset to ~2000+ images without any external data.
Techniques (all OBB-label-safe):
  - Horizontal / vertical flip
  - 90° / 180° / 270° rotation
  - Random brightness / contrast / saturation jitter
  - Gaussian blur + sharpening
  - Gaussian noise
  - Mosaic (4 images stitched into 1)
  - Cutout (random black rectangle to simulate occlusion)

Output: Augmented-Dataset/<CLASS_NAME>/  (images only, for classification)
        Augmented-Dataset-YOLO/          (images + YOLO OBB labels, for detection training)
"""

import random
import shutil
import math
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

MERGED_DIR = Path(r"C:\Users\USER_HP\Desktop\FarmFederate\Merged-Dataset")
REAL_LABS  = Path(r"C:\Users\USER_HP\Desktop\FarmFederate\Real Dataset\labels")
AUG_CLS    = Path(r"C:\Users\USER_HP\Desktop\FarmFederate\Augmented-Dataset")
AUG_YOLO   = Path(r"C:\Users\USER_HP\Desktop\FarmFederate\Augmented-Dataset-YOLO")

CLASS_NAMES = ["LEAF_BLIGHT", "LEAF_HOPPERS", "LEAF_RUST",
               "LOOPER_CATERPILLARS", "MOSQUITO_BUG"]

# YOLO class ID lookup (from data.yaml)
YOLO_ID = {
    "LEAF_BLIGHT": 0, "LEAF_HOPPERS": 1, "LEAF_RUST": 2,
    "LOOPER_CATERPILLARS": 3, "MOSQUITO_BUG": 4,
}

TARGET_PER_CLASS = 400   # target ~2000 total

random.seed(42)
np.random.seed(42)


# ── YOLO OBB helpers ──────────────────────────────────────────────────────────
def load_obb(label_path: Path) -> list[list[float]]:
    """Return list of [cls_id, x1,y1,x2,y2,x3,y3,x4,y4] (normalised)."""
    rows = []
    if not label_path.exists():
        return rows
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 9:
                try:
                    rows.append(list(map(float, parts[:9])))
                except ValueError:
                    pass
    return rows


def save_obb(label_path: Path, rows: list[list[float]]) -> None:
    with open(label_path, "w") as f:
        for r in rows:
            f.write(" ".join(f"{v:.8f}" if i > 0 else str(int(v))
                             for i, v in enumerate(r)) + "\n")


def transform_obb_rows(rows: list[list[float]],
                       M: np.ndarray, W: int, H: int) -> list[list[float]]:
    """Apply affine matrix M (2×3, pixel coords) to all OBB rows; re-normalise."""
    out = []
    for r in rows:
        cid = int(r[0])
        pts = np.array([[r[1] * W, r[2] * H],
                        [r[3] * W, r[4] * H],
                        [r[5] * W, r[6] * H],
                        [r[7] * W, r[8] * H]], dtype=np.float32)
        pts_h = np.hstack([pts, np.ones((4, 1))])
        pts_t = (M @ pts_h.T).T
        # Clamp to image
        pts_t[:, 0] = np.clip(pts_t[:, 0], 0, W - 1)
        pts_t[:, 1] = np.clip(pts_t[:, 1], 0, H - 1)
        norm = [cid] + [v for p in pts_t for v in
                        [p[0] / W, p[1] / H]]
        out.append(norm)
    return out


# ── Augmentation functions ────────────────────────────────────────────────────
def aug_hflip(img: np.ndarray, rows):
    H, W = img.shape[:2]
    out = cv2.flip(img, 1)
    M = np.array([[- 1, 0, W - 1],
                  [  0, 1,     0]], dtype=np.float64)
    return out, transform_obb_rows(rows, M, W, H)


def aug_vflip(img: np.ndarray, rows):
    H, W = img.shape[:2]
    out = cv2.flip(img, 0)
    M = np.array([[1,  0,     0],
                  [0, -1, H - 1]], dtype=np.float64)
    return out, transform_obb_rows(rows, M, W, H)


def aug_rot90(img: np.ndarray, rows, k: int):
    """Rotate by k*90 degrees."""
    H, W = img.shape[:2]
    out = np.rot90(img, k)
    OH, OW = out.shape[:2]
    if k == 1:
        M = np.array([[0, -1, H - 1], [1, 0, 0]], dtype=np.float64)
    elif k == 2:
        M = np.array([[-1, 0, W - 1], [0, -1, H - 1]], dtype=np.float64)
    else:  # k==3
        M = np.array([[0, 1, 0], [-1, 0, W - 1]], dtype=np.float64)
    new_rows = transform_obb_rows(rows, M, W, H)
    # Re-normalise to new dims
    for r in new_rows:
        r[1], r[3], r[5], r[7] = [v * W / OW for v in [r[1], r[3], r[5], r[7]]]
        r[2], r[4], r[6], r[8] = [v * H / OH for v in [r[2], r[4], r[6], r[8]]]
    return out, new_rows


def aug_color(img: np.ndarray, rows):
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    pil = ImageEnhance.Brightness(pil).enhance(random.uniform(0.7, 1.3))
    pil = ImageEnhance.Contrast(pil).enhance(random.uniform(0.7, 1.3))
    pil = ImageEnhance.Color(pil).enhance(random.uniform(0.7, 1.4))
    out = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    return out, rows


def aug_blur(img: np.ndarray, rows):
    k = random.choice([3, 5])
    out = cv2.GaussianBlur(img, (k, k), 0)
    return out, rows


def aug_sharpen(img: np.ndarray, rows):
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    pil = pil.filter(ImageFilter.SHARPEN)
    out = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    return out, rows


def aug_noise(img: np.ndarray, rows):
    noise = np.random.normal(0, random.uniform(5, 20), img.shape).astype(np.int16)
    out = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return out, rows


def aug_cutout(img: np.ndarray, rows):
    out = img.copy()
    H, W = img.shape[:2]
    for _ in range(random.randint(1, 3)):
        cw = random.randint(W // 10, W // 4)
        ch = random.randint(H // 10, H // 4)
        cx = random.randint(0, W - cw)
        cy = random.randint(0, H - ch)
        out[cy:cy + ch, cx:cx + cw] = 0
    return out, rows


# Augmentation pipeline list — each entry is (fn, weight)
AUGS = [
    (aug_hflip,   2),
    (aug_vflip,   2),
    (lambda i, r: aug_rot90(i, r, 1), 2),
    (lambda i, r: aug_rot90(i, r, 2), 2),
    (lambda i, r: aug_rot90(i, r, 3), 2),
    (aug_color,   3),
    (aug_blur,    2),
    (aug_sharpen, 2),
    (aug_noise,   2),
    (aug_cutout,  1),
]
AUG_FNS     = [f for f, w in AUGS]
AUG_WEIGHTS = [w for _, w in AUGS]


def augment_one(img: np.ndarray, rows, n_ops: int = 2):
    """Apply n_ops random augmentations in sequence."""
    chosen = random.choices(AUG_FNS, weights=AUG_WEIGHTS, k=n_ops)
    for fn in chosen:
        img, rows = fn(img, rows)
    return img, rows


# ── Main loop ─────────────────────────────────────────────────────────────────
for cls in CLASS_NAMES:
    src_folder = MERGED_DIR / cls
    imgs = list(src_folder.glob("*.jpg")) + list(src_folder.glob("*.jpeg")) + \
           list(src_folder.glob("*.png"))
    if not imgs:
        print(f"  SKIP {cls}: no images")
        continue

    (AUG_CLS  / cls).mkdir(parents=True, exist_ok=True)
    (AUG_YOLO / "images" / cls).mkdir(parents=True, exist_ok=True)
    (AUG_YOLO / "labels" / cls).mkdir(parents=True, exist_ok=True)

    # Copy originals first
    for p in imgs:
        shutil.copy2(p, AUG_CLS / cls / p.name)
        shutil.copy2(p, AUG_YOLO / "images" / cls / p.name)
        stem = p.stem.replace("ref_", "", 1)
        lab  = REAL_LABS / (stem + ".txt")
        if lab.exists():
            shutil.copy2(lab, AUG_YOLO / "labels" / cls / (p.stem + ".txt"))
        else:
            # Create empty label (whole-image OBB fallback)
            yid = YOLO_ID[cls]
            save_obb(AUG_YOLO / "labels" / cls / (p.stem + ".txt"),
                     [[yid, 0.01, 0.01, 0.99, 0.01, 0.99, 0.99, 0.01, 0.99]])

    need = max(0, TARGET_PER_CLASS - len(imgs))
    print(f"{cls}: {len(imgs)} originals -> generating {need} augmented ...")

    aug_idx = 0
    while aug_idx < need:
        src = random.choice(imgs)
        img_bgr = cv2.imread(str(src))
        if img_bgr is None:
            continue
        H, W = img_bgr.shape[:2]

        stem = src.stem.replace("ref_", "", 1)
        rows = load_obb(REAL_LABS / (stem + ".txt"))
        if not rows:
            yid = YOLO_ID[cls]
            rows = [[yid, 0.01, 0.01, 0.99, 0.01, 0.99, 0.99, 0.01, 0.99]]

        n_ops = random.randint(1, 3)
        aug_img, aug_rows = augment_one(img_bgr, rows, n_ops)

        name = f"aug_{cls[:4]}_{aug_idx:05d}.jpg"
        cv2.imwrite(str(AUG_CLS  / cls / name), aug_img,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        cv2.imwrite(str(AUG_YOLO / "images" / cls / name), aug_img,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        save_obb(AUG_YOLO / "labels" / cls / name.replace(".jpg", ".txt"),
                 aug_rows)
        aug_idx += 1

    total = len(list((AUG_CLS / cls).iterdir()))
    print(f"  -> {total} images in {AUG_CLS / cls}")

# ── Write data.yaml for YOLO training ─────────────────────────────────────────
yaml_content = f"""train: {AUG_YOLO / 'images'}
val:   {AUG_YOLO / 'images'}

nc: {len(CLASS_NAMES)}
names: {CLASS_NAMES}
"""
(AUG_YOLO / "data.yaml").write_text(yaml_content)

print("\n=== Augmented-Dataset summary ===")
for cls in CLASS_NAMES:
    n = len(list((AUG_CLS / cls).iterdir()))
    print(f"  {cls}: {n}")

total = sum(len(list((AUG_CLS / c).iterdir())) for c in CLASS_NAMES)
print(f"\nTotal: {total} images")
print(f"Classification dataset : {AUG_CLS}")
print(f"YOLO OBB dataset       : {AUG_YOLO}")
