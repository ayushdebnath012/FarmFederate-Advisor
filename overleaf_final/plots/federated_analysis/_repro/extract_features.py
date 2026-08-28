"""
Stage 1: Extract frozen encoder features from the REAL FarmFederate data.

Text : data/crop_stress_text_dataset.csv  (BLIP captions, 5 stress classes)
Image: data/{water_stress,nutrient_def,pest_risk,disease_risk,heat_stress}/

Pairing follows the repo's MultiModalDataset: index-aligned within class, so
image[i] and text[i] always share the same label (class-consistent pairing).

Features are cached to .npy so the federated sweep can run hundreds of configs
in minutes on CPU.
"""
import os, sys, json, random
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(r"c:\Users\USER_HP\Desktop\FarmFederate")
DATA = ROOT / "data"
OUT = Path(__file__).parent / "features"
OUT.mkdir(exist_ok=True, parents=True)

CLASSES = ["water_stress", "nutrient_def", "pest_risk", "disease_risk", "heat_stress"]
SEED = 42
random.seed(SEED); np.random.seed(SEED)

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".JPEG", ".PNG"}


def load_pairs():
    """Build class-consistent (image_path, text) pairs."""
    df = pd.read_csv(DATA / "crop_stress_text_dataset.csv")
    df = df[df["label_name"].isin(CLASSES)]

    texts_by_class = {c: df[df["label_name"] == c]["text"].tolist() for c in CLASSES}
    imgs_by_class = {}
    for c in CLASSES:
        d = DATA / c
        paths = sorted([p for p in d.rglob("*") if p.suffix in IMG_EXT])
        imgs_by_class[c] = paths

    pairs = []  # (img_path, text, label_idx)
    for li, c in enumerate(CLASSES):
        ts, ims = texts_by_class[c], imgs_by_class[c]
        n = min(len(ts), len(ims))
        random.Random(SEED + li).shuffle(ts)
        for i in range(n):
            pairs.append((str(ims[i]), ts[i], li))
        print(f"  {c:16s} texts={len(ts):5d} images={len(ims):5d} -> paired={n}")

    random.Random(SEED).shuffle(pairs)
    return pairs


def main():
    print("[1/3] Building class-consistent image-text pairs...")
    pairs = load_pairs()
    print(f"  TOTAL paired samples: {len(pairs)}")

    img_paths = [p[0] for p in pairs]
    texts = [p[1] for p in pairs]
    labels = np.array([p[2] for p in pairs], dtype=np.int64)

    import torch
    from transformers import AutoTokenizer, AutoModel
    torch.set_num_threads(os.cpu_count() or 4)
    dev = "cpu"

    # ---------------- TEXT ----------------
    print("\n[2/3] Extracting TEXT features (DistilBERT, frozen)...")
    tname = "distilbert-base-uncased"
    tok = AutoTokenizer.from_pretrained(tname)
    tmodel = AutoModel.from_pretrained(tname).to(dev).eval()

    tfeat = []
    B = 64
    with torch.no_grad():
        for i in range(0, len(texts), B):
            batch = texts[i:i + B]
            enc = tok(batch, padding=True, truncation=True, max_length=128, return_tensors="pt")
            out = tmodel(**enc).last_hidden_state          # (B, L, 768)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            emb = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)  # mean pool
            tfeat.append(emb.cpu().numpy())
            if (i // B) % 10 == 0:
                print(f"    text {i}/{len(texts)}", flush=True)
    tfeat = np.concatenate(tfeat).astype(np.float32)
    print(f"  text features: {tfeat.shape}")
    np.save(OUT / "text_feat.npy", tfeat)   # persist immediately (crash-safe)

    # ---------------- IMAGE ----------------
    # Manual ViT preprocessing (resize 224 + ImageNet normalise) so we do NOT
    # need torchvision. Matches AutoImageProcessor for vit-tiny.
    print("\n[3/3] Extracting IMAGE features (ViT-tiny patch16-224, frozen)...")
    from PIL import Image
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def preprocess(p):
        try:
            im = Image.open(p).convert("RGB").resize((224, 224), Image.BILINEAR)
        except Exception:
            im = Image.new("RGB", (224, 224), (0, 0, 0))
        a = np.asarray(im, dtype=np.float32) / 255.0
        a = (a - MEAN) / STD
        return np.transpose(a, (2, 0, 1))

    iname = "WinKawaks/vit-tiny-patch16-224"
    imodel = AutoModel.from_pretrained(iname).to(dev).eval()

    ifeat = []
    B = 32
    with torch.no_grad():
        for i in range(0, len(img_paths), B):
            arr = np.stack([preprocess(p) for p in img_paths[i:i + B]])
            pv = torch.tensor(arr, dtype=torch.float32)
            out = imodel(pixel_values=pv).last_hidden_state   # (B, 197, 192)
            emb = out[:, 0]                                    # CLS token
            ifeat.append(emb.cpu().numpy())
            if (i // B) % 10 == 0:
                print(f"    image {i}/{len(img_paths)}", flush=True)
    ifeat = np.concatenate(ifeat).astype(np.float32)
    print(f"  image features: {ifeat.shape}")

    np.save(OUT / "text_feat.npy", tfeat)
    np.save(OUT / "image_feat.npy", ifeat)
    np.save(OUT / "labels.npy", labels)
    with open(OUT / "meta.json", "w") as f:
        json.dump({
            "n": int(len(labels)),
            "classes": CLASSES,
            "text_dim": int(tfeat.shape[1]),
            "image_dim": int(ifeat.shape[1]),
            "text_encoder": tname,
            "image_encoder": iname,
            "class_counts": {CLASSES[i]: int((labels == i).sum()) for i in range(5)},
        }, f, indent=2)
    print("\nSaved ->", OUT)
    print(json.dumps({CLASSES[i]: int((labels == i).sum()) for i in range(5)}, indent=2))


if __name__ == "__main__":
    main()
