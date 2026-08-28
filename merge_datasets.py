"""
Merge Ref-Dataset and Real Dataset into one organized folder structure.

Class mapping (from data.yaml):
  0: LEAF_BLIGHT      <-> Ref-Dataset/Leaf-blight
  1: LEAF_HOPPERS     <-> Ref-Dataset/Leaf-Hopper
  2: LEAF_RUST        <-> Ref-Dataset/Leaf Rust
  3: LOOPER_CATERPILLARS <-> Ref-Dataset/Looper Caterpillar
  4: MOSQUITO_BUG     <-> Ref-Dataset/Mosquito bug

Output: Merged-Dataset/<CLASS_NAME>/
"""

import shutil
from pathlib import Path
from collections import Counter

REF_DIR    = Path(r"C:\Users\USER_HP\Desktop\FarmFederate\Ref-Dataset")
REAL_IMGS  = Path(r"C:\Users\USER_HP\Desktop\FarmFederate\Real Dataset\images")
REAL_LABS  = Path(r"C:\Users\USER_HP\Desktop\FarmFederate\Real Dataset\labels")
OUT_DIR    = Path(r"C:\Users\USER_HP\Desktop\FarmFederate\Merged-Dataset")

# YOLO class ID <-> name and Ref-Dataset folder
CLASS_MAP = {
    0: ("LEAF_BLIGHT",          "Leaf-blight"),
    1: ("LEAF_HOPPERS",         "Leaf-Hopper"),
    2: ("LEAF_RUST",            "Leaf Rust"),
    3: ("LOOPER_CATERPILLARS",  "Looper Caterpillar"),
    4: ("MOSQUITO_BUG",         "Mosquito bug"),
}

# Create output class folders
for class_id, (class_name, _) in CLASS_MAP.items():
    (OUT_DIR / class_name).mkdir(parents=True, exist_ok=True)

# ── Step 1: Copy Ref-Dataset images ───────────────────────────────────────
print("=== Copying Ref-Dataset images ===")
ref_count = 0
for class_id, (class_name, ref_folder) in CLASS_MAP.items():
    src_dir = REF_DIR / ref_folder
    if not src_dir.exists():
        print(f"  WARNING: {src_dir} not found")
        continue
    for img_file in src_dir.iterdir():
        if img_file.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
            dest = OUT_DIR / class_name / f"ref_{img_file.name}"
            shutil.copy2(img_file, dest)
            print(f"  {class_name} <- {img_file.name}")
            ref_count += 1

print(f"  Copied {ref_count} reference images\n")

# ── Step 2: Sort Real Dataset images by dominant YOLO class ───────────────
print("=== Sorting Real Dataset images ===")
real_count = Counter()
unclassified = []

for img_path in sorted(REAL_IMGS.glob("*.jpg")) + sorted(REAL_IMGS.glob("*.png")):
    label_path = REAL_LABS / (img_path.stem + ".txt")
    if not label_path.exists():
        unclassified.append(img_path.name)
        continue

    # Count class IDs in this image's label file
    class_ids = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                try:
                    class_ids.append(int(parts[0]))
                except ValueError:
                    pass

    if not class_ids:
        unclassified.append(img_path.name)
        continue

    dominant_id = Counter(class_ids).most_common(1)[0][0]
    if dominant_id not in CLASS_MAP:
        unclassified.append(img_path.name)
        continue

    class_name = CLASS_MAP[dominant_id][0]
    dest = OUT_DIR / class_name / img_path.name
    shutil.copy2(img_path, dest)
    real_count[class_name] += 1

print("  Real Dataset images per class:")
total_real = 0
for class_id, (class_name, _) in CLASS_MAP.items():
    n = real_count[class_name]
    total_real += n
    print(f"    {class_name}: {n}")
if unclassified:
    print(f"  Unclassified (no label): {len(unclassified)}")
print(f"  Total copied: {total_real}\n")

# ── Summary ────────────────────────────────────────────────────────────────
print("=== Merged-Dataset summary ===")
for class_id, (class_name, _) in CLASS_MAP.items():
    folder = OUT_DIR / class_name
    n = len(list(folder.iterdir()))
    print(f"  {class_name}: {n} images")

print(f"\nMerged dataset at: {OUT_DIR}")
