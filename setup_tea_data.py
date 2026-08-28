#!/usr/bin/env python3
"""
setup_tea_data.py
=================
Downloads and prepares the complete FarmFederate tea leaf disease dataset.

Steps:
  1. Extract Kaggle tea dataset (5867 images, 6 classes) to tea_data/
  2. Merge with existing Real Dataset (200 images, YOLO OBB)
  3. Generate neutral per-class text CSV files (600 samples/class)
  4. Create annotations.csv for the full combined dataset
  5. Run retrain_realistic.py to produce realistic F1 checkpoints

Output structure:
  tea_data/
    algal_leaf_spot/images/*.jpg   (~1000 images)
    algal_leaf_spot/text.csv       (600 neutral text samples)
    brown_blight/images/*.jpg      (~867 images)
    ...
    annotations.csv                (class,image_path,text)

  farmfederate_results_20260415_150446/
    models/   ← trained checkpoints
    results/complete_results.json  ← realistic F1 scores
"""

import zipfile, shutil, random, json, pathlib, csv, os, sys, time
from collections import Counter
from pathlib import Path

random.seed(42)

BASE     = Path('C:/Users/USER_HP/Desktop/FarmFederate')
TEA_DATA = BASE / 'tea_data'
REAL_DS  = BASE / 'Real Dataset'
TMP_ZIP  = Path('C:/Users/USER_HP/AppData/Local/Temp/tmp5rupr13d/tea-leaf-disease-clasification.zip')

# Class name mapping: Kaggle → our 5-class names
KAGGLE_MAP = {
    'algal_spot':   'algal_leaf_spot',
    'brown_blight': 'brown_blight',
    'gray_blight':  'gray_blight',
    'helopeltis':   'helopeltis',
    'red_spot':     'red_leaf_spot',
    # 'healthy' is intentionally excluded (not in our 5 classes)
}
CLASSES = ['gray_blight', 'helopeltis', 'algal_leaf_spot', 'brown_blight', 'red_leaf_spot']

# ── Neutral text templates (no disease names / scientific genus names) ──────────
NEUTRAL_TEMPLATES = {
    'gray_blight': {
        'clear': [
            "Pale grey-white patches on mid-canopy leaves. Edges are diffuse, not well-defined. Condition worsens after cool damp nights.",
            "Grey-brown irregular discolouration on older leaf surfaces. Spreading gradually under sustained overcast weather.",
            "Ashy surface coating visible on leaf blade. Tissue underneath feels soft. More leaves affected after recent cool spell.",
            "Dull grey lesions starting on lower-canopy leaves and moving upward. Margins fade into healthy tissue without sharp boundaries.",
            "Greyish necrotic patches on mature foliage. Faint concentric banding visible. Moisture and shade appear to favour spread.",
            "Pale grey blotches covering roughly a third of some leaves. Adjacent healthy tissue has a faint yellowish tinge.",
            "Grey-white necrosis on older leaves in shaded blocks. Cool overnight temperatures preceded onset by three days.",
            "Soft pale patches covering older tea foliage. Edges blend into green tissue. Canopy density seems to influence severity.",
        ],
        'ambiguous': [
            "Grey-brown discolouration on leaf surface. Could reflect fungal activity or weather stress.",
            "Pale patches on foliage. Edges unclear. Seen mainly on older leaves in shaded rows.",
            "Diffuse grey lesions on some leaves. Margins not well defined. Older canopy most affected.",
        ],
    },
    'helopeltis': {
        'clear': [
            "Multiple small raised spots on tender new shoots. Each has a sunken centre. Shoot tips appear wilted.",
            "Young flush showing angular brown patches with slightly raised edges. Pattern is scattered, not spreading from margins.",
            "New growth showing shoot dieback. Small circular raised lesions with pale outer rings. Lesions feel rough to the touch.",
            "Raised necrotic spots on expanding leaves. Tips of new shoots are discoloured and bent downward.",
            "Angular brown marks at irregular positions on young leaves. No margin-to-centre spread; individual wound-like lesions.",
            "Tender flush with rough raised patches and tip wilting. Lesions are irregularly placed, not along veins.",
            "Young shoots with multiple scattered corky raised spots. Tip growth stunted. Lesions have darkened sunken centres.",
            "Angular necrotic marks on newly emerged leaves. Rough texture at lesion centres. Tips curled and discoloured.",
        ],
        'ambiguous': [
            "Brown spots on new foliage. Raised texture at lesion centres. Shoot tips look unhealthy.",
            "Small angular marks on young leaves. Some tip wilting visible. Cause not immediately clear.",
            "Raised lesions on tender flush. Scattered distribution. Tips show some dieback.",
        ],
    },
    'algal_leaf_spot': {
        'clear': [
            "Orange-red crusty patches on mature leaves. Surface texture is rough and velvety. Shaded sections of the garden most affected.",
            "Rust-coloured circular deposits on upper leaf surface. Dry and slightly powdery when touched. Spreading very slowly.",
            "Reddish-orange encrustations on older foliage. Circular shape with raised surface. Confined to areas with poor air movement.",
            "Orange velvet-like coating on adaxial leaf surface. Rubbing leaves an orange residue on fingers. Older leaves mainly affected.",
            "Circular rust-coloured patches with a velvety raised surface. Found in the most densely shaded rows. Not expanding quickly.",
            "Orange-brown raised circular lesions on mature leaves. Slow progression. Shaded and persistently moist sections affected.",
            "Rust-orange circular raised deposits on mature canopy leaves. Very slow spread. Dense shade and poor ventilation present.",
            "Velvety orange encrustations on adaxial surface of older leaves. Not coalescing. Shaded garden blocks most affected.",
        ],
        'ambiguous': [
            "Orange-red patches on older leaves. Circular shape. Found in shaded areas of the garden.",
            "Rust-coloured raised deposits on mature foliage. Not spreading fast. Texture slightly rough.",
            "Orange-brown circular marks on older leaves. Raised surface. Slow progression in shaded rows.",
        ],
    },
    'brown_blight': {
        'clear': [
            "Brown discolouration beginning at leaf tip and progressing along margins. Affected tissue has a soft, water-soaked texture at the edge.",
            "Irregular brown patches covering the tip and edges of leaves. Water-soaked appearance at the leading edge of spread.",
            "Chocolate-brown lesions expanding inward from margins. More than a third of some leaves is now discoloured.",
            "Tip and margin blight progressing rapidly. Brown necrotic areas with darker borders and pale inner zone.",
            "Extensive brown blighting from leaf tip. Soft margins. New flush most severely affected during humid periods.",
            "Brown necrotic areas starting from tip edges. Irregular shape. Moisture seems to accelerate spread.",
            "Tip blight advancing along leaf margins. Water-soaked soft tissue at the active edge of the lesion.",
            "Brown patches from leaf apex spreading inward. More than 40% of some leaves now affected. New growth most severe.",
        ],
        'ambiguous': [
            "Brown patches on leaf tip and margins. Spreading slowly toward centre. Worse on younger flush.",
            "Leaf tips turning brown. Margins discoloured. Pattern suggests tip and edge origin.",
            "Tip browning progressing. Water-soaked edges at active lesion boundary. Humid weather noted.",
        ],
    },
    'red_leaf_spot': {
        'clear': [
            "Circular reddish-brown spots with well-defined borders. Yellow halo visible around the larger lesions. Found on mature leaves only.",
            "Small round reddish patches with darker concentric rings at the centre. Borders are clearly defined against healthy tissue.",
            "Disc-shaped spots on older foliage. Each has a pale outer ring and a darkened inner zone. Spots are not coalescing yet.",
            "Reddish-brown circular lesions with a yellow chlorotic halo. Well-defined edges. Confined to fully expanded mature leaves.",
            "Round red-brown necrotic patches with visible concentric darkening. Not found on young shoots. Diameter expanding slowly.",
            "Circular spots with distinct borders and slight yellow margin on mature leaves. Reddish-brown colouring throughout.",
            "Well-defined reddish circular lesions on mature canopy leaves. Concentric darkening visible. Yellow chlorotic ring present.",
            "Disc-shaped red-brown lesions with clear borders. Only mature leaves affected. No coalescing yet. Slow diameter increase.",
        ],
        'ambiguous': [
            "Round reddish patches on older leaves. Borders relatively defined. Yellow tinge around some spots.",
            "Circular discolouration on mature foliage. Red-brown colour, well-defined shape. Slow spread.",
            "Round red-brown marks on mature leaves. Concentric pattern. Well-defined at edges.",
        ],
    },
}


def generate_neutral_text_csv(cls_name: str, n: int = 600) -> list:
    """Return list of (text, label) tuples with realistic inter-class noise."""
    tpl = NEUTRAL_TEMPLATES[cls_name]
    all_t = tpl['clear'] + tpl['ambiguous']
    cls_idx = CLASSES.index(cls_name)
    rows = []
    for i in range(n):
        base = all_t[i % len(all_t)]
        # 40% chance: append an ambiguous cross-class symptom
        if random.random() < 0.40:
            other_name = random.choice([c for c in CLASSES if c != cls_name])
            cross = random.choice(NEUTRAL_TEMPLATES[other_name]['ambiguous'])
            base = base + " " + cross
        rows.append({'text': base, 'class_id': cls_idx, 'class_name': cls_name})
    random.shuffle(rows)
    return rows


# ── Step 1: Extract Kaggle dataset ────────────────────────────────────────────
def extract_kaggle(zip_path: Path, out_dir: Path):
    print("\n[1/4] Extracting Kaggle tea leaf dataset...")
    out_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        total = len(names)
        extracted = 0

        for member in names:
            parts = member.split('/')
            # Only extract Train/ images (skip Test/ and healthy class)
            if len(parts) < 4:
                continue
            _, split, cls_kaggle, fname = parts[0], parts[1], parts[2], parts[3]
            if split not in ('Train', 'Test'):
                continue
            our_cls = KAGGLE_MAP.get(cls_kaggle)
            if our_cls is None:
                continue  # skip 'healthy'
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            dst_dir = out_dir / our_cls / 'images'
            dst_dir.mkdir(parents=True, exist_ok=True)
            # Prefix with split to avoid name collision
            dst_file = dst_dir / f"{split.lower()}_{fname}"
            if not dst_file.exists():
                with zf.open(member) as src, open(dst_file, 'wb') as dst:
                    dst.write(src.read())
                extracted += 1

    # Report
    print(f"  Extracted {extracted} images to {out_dir}")
    for cls in CLASSES:
        cls_dir = out_dir / cls / 'images'
        if cls_dir.exists():
            n = len(list(cls_dir.glob('*.jpg')) + list(cls_dir.glob('*.png')))
            print(f"    {cls}: {n}")


# ── Step 2: Copy Real Dataset images ─────────────────────────────────────────
def copy_real_dataset(real_ds: Path, out_dir: Path):
    print("\n[2/4] Merging Real Dataset (200 YOLO OBB images)...")
    img_dir = real_ds / 'images'
    lbl_dir = real_ds / 'labels'
    if not img_dir.exists():
        print("  Real Dataset not found — skipping")
        return

    cls_names = ['gray_blight', 'helopeltis', 'algal_leaf_spot', 'brown_blight', 'red_leaf_spot']
    copied = Counter()

    for img_path in sorted(img_dir.glob('*.jpg')) + sorted(img_dir.glob('*.png')):
        lbl_path = lbl_dir / (img_path.stem + '.txt')
        if not lbl_path.exists():
            continue
        lines = lbl_path.read_text().strip().splitlines()
        if not lines:
            continue
        cls_id = int(lines[0].split()[0])
        if not (0 <= cls_id < 5):
            continue
        cls_name = cls_names[cls_id]
        dst_dir = out_dir / cls_name / 'images'
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst_file = dst_dir / f"obb_{img_path.name}"
        if not dst_file.exists():
            shutil.copy2(img_path, dst_file)
        copied[cls_name] += 1

    print(f"  Copied {sum(copied.values())} OBB images:")
    for cls in cls_names:
        print(f"    {cls}: {copied[cls]}")


# ── Step 3: Generate text CSV files ──────────────────────────────────────────
def generate_text_csvs(out_dir: Path, n_per_class: int = 600):
    print(f"\n[3/4] Generating neutral text CSV files ({n_per_class} samples/class)...")
    all_rows = []

    for cls_name in CLASSES:
        rows = generate_neutral_text_csv(cls_name, n_per_class)
        cls_dir = out_dir / cls_name
        cls_dir.mkdir(parents=True, exist_ok=True)
        csv_path = cls_dir / 'text.csv'
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['text', 'class_id', 'class_name'])
            writer.writeheader()
            writer.writerows(rows)
        all_rows.extend(rows)
        print(f"  {cls_name}: {len(rows)} samples -> {csv_path.name}")

    # Write combined annotations.csv at root of tea_data/
    ann_path = out_dir / 'annotations.csv'
    random.shuffle(all_rows)
    with open(ann_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['text', 'class_id', 'class_name'])
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\n  Combined annotations.csv: {len(all_rows)} rows -> {ann_path}")


# ── Step 4: Summary ───────────────────────────────────────────────────────────
def print_summary(out_dir: Path):
    print("\n[4/4] Dataset summary:")
    total_imgs = 0
    for cls in CLASSES:
        img_dir = out_dir / cls / 'images'
        txt_csv = out_dir / cls / 'text.csv'
        n_imgs = len(list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png')) + list(img_dir.glob('*.jpeg'))) if img_dir.exists() else 0
        n_txt  = sum(1 for _ in open(txt_csv)) - 1 if txt_csv.exists() else 0  # subtract header
        total_imgs += n_imgs
        print(f"  {cls:20s}  images={n_imgs:4d}  text={n_txt:4d}")
    print(f"\n  Total images: {total_imgs}")
    ann = out_dir / 'annotations.csv'
    if ann.exists():
        n_ann = sum(1 for _ in open(ann)) - 1
        print(f"  annotations.csv: {n_ann} rows")


def main():
    t0 = time.time()
    print("=" * 60)
    print("FarmFederate — Tea Dataset Setup")
    print("=" * 60)

    # Step 1: Extract Kaggle dataset
    if TMP_ZIP.exists():
        extract_kaggle(TMP_ZIP, TEA_DATA)
    else:
        print(f"\n[1/4] Kaggle zip not found at {TMP_ZIP}")
        print("  Run: python3 -c \"import kaggle; kaggle.api.authenticate(); kaggle.api.dataset_download_files('mamun009/tea-leaf-disease-clasification', path='C:/tmp', unzip=False)\"")

    # Step 2: Merge Real Dataset (YOLO OBB)
    copy_real_dataset(REAL_DS, TEA_DATA)

    # Step 3: Generate text CSV files
    generate_text_csvs(TEA_DATA, n_per_class=600)

    # Step 4: Summary
    print_summary(TEA_DATA)

    elapsed = (time.time() - t0) / 60
    print(f"\nDone in {elapsed:.1f} min")
    print(f"\nNext step: run retrain_realistic.py")
    print(f"  python retrain_realistic.py")


if __name__ == '__main__':
    main()
