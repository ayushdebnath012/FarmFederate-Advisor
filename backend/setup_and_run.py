"""
FarmFederate Setup & Run Script
================================
1. Sorts Real Dataset images into class folders
2. Downloads required models from HuggingFace
3. Runs training pipeline
4. Saves all results
"""
import os
import sys
import shutil
import json
import subprocess
from pathlib import Path
from collections import Counter

# ── Paths ──────────────────────────────────────────────────────────────────
REPO_ROOT      = Path(__file__).parent.parent
REAL_DS_ROOT   = REPO_ROOT / "Real Dataset"
IMAGES_DIR     = REAL_DS_ROOT / "images"
LABELS_DIR     = REAL_DS_ROOT / "labels"
SORTED_DIR     = REPO_ROOT / "data" / "real_dataset_sorted"
RESULTS_DIR    = REPO_ROOT / "results"
MODELS_DIR     = REPO_ROOT / "models"

CLASS_NAMES = [
    "LEAF_BLIGHT",
    "LEAF_HOPPERS",
    "LEAF_RUST",
    "LOOPER_CATERPILLARS",
    "MOSQUITO_BUG",
]

# ── Step 1: Sort images into class folders ─────────────────────────────────
def sort_images():
    print("\n" + "="*60)
    print("STEP 1: Sorting Real Dataset images into class folders")
    print("="*60)

    # Create class subdirectories
    for cls in CLASS_NAMES:
        (SORTED_DIR / cls).mkdir(parents=True, exist_ok=True)

    counts = Counter()
    skipped = 0

    for label_file in sorted(LABELS_DIR.glob("*.txt")):
        stem = label_file.stem
        # Find matching image (try .jpg, .jpeg, .png)
        img_path = None
        for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
            candidate = IMAGES_DIR / (stem + ext)
            if candidate.exists():
                img_path = candidate
                break
        if img_path is None:
            skipped += 1
            continue

        # Read dominant class from label file
        class_ids = []
        with open(label_file) as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    try:
                        class_ids.append(int(parts[0]))
                    except ValueError:
                        pass

        if not class_ids:
            skipped += 1
            continue

        dominant_class_id = Counter(class_ids).most_common(1)[0][0]
        if dominant_class_id >= len(CLASS_NAMES):
            skipped += 1
            continue

        class_name = CLASS_NAMES[dominant_class_id]
        dest = SORTED_DIR / class_name / img_path.name
        shutil.copy2(img_path, dest)
        counts[class_name] += 1

    print(f"\nSorted images:")
    total = 0
    for cls in CLASS_NAMES:
        n = counts[cls]
        total += n
        print(f"  {cls:<25} {n:>4} images")
    print(f"  {'TOTAL':<25} {total:>4} images")
    if skipped:
        print(f"  Skipped (no label/image match): {skipped}")

    # Save manifest
    manifest = {
        "sorted_dir": str(SORTED_DIR),
        "class_counts": {cls: counts[cls] for cls in CLASS_NAMES},
        "total": total,
    }
    SORTED_DIR.mkdir(parents=True, exist_ok=True)
    with open(SORTED_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n  Saved to: {SORTED_DIR}")
    return manifest


# ── Step 2: Install / verify dependencies ─────────────────────────────────
def install_deps():
    print("\n" + "="*60)
    print("STEP 2: Installing / verifying Python dependencies")
    print("="*60)

    packages = [
        "torch", "torchvision", "transformers", "datasets",
        "Pillow", "numpy", "scikit-learn", "tqdm",
        "huggingface_hub", "accelerate", "sentencepiece",
        "timm", "pandas", "matplotlib", "seaborn",
    ]
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", *packages])
    print("  All packages present.")


# ── Step 3: Download models from HuggingFace ──────────────────────────────
def download_models():
    print("\n" + "="*60)
    print("STEP 3: Downloading models from HuggingFace")
    print("="*60)

    import os
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"

    from transformers import (
        ViTForImageClassification, ViTFeatureExtractor,
        BertTokenizer, BertForSequenceClassification,
        CLIPModel, CLIPProcessor,
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    cache_dir = str(MODELS_DIR / "hf_cache")
    downloaded = {}

    def _try_load(name, repo_id, loader_fn, max_retries=3):
        for attempt in range(1, max_retries + 1):
            try:
                print(f"  Downloading {name} ({repo_id}) — attempt {attempt}/{max_retries} …")
                obj = loader_fn(repo_id, cache_dir=cache_dir)
                print(f"    ✓ {name} ready")
                return obj
            except Exception as e:
                print(f"    ✗ attempt {attempt} failed: {e}")
                if attempt == max_retries:
                    raise
        return None

    _try_load("ViT weights",     "google/vit-base-patch16-224",
              lambda r, **kw: ViTForImageClassification.from_pretrained(r, num_labels=5, ignore_mismatched_sizes=True, **kw))
    _try_load("ViT extractor",   "google/vit-base-patch16-224",
              ViTFeatureExtractor.from_pretrained)
    _try_load("BERT tokenizer",  "bert-base-uncased",
              BertTokenizer.from_pretrained)
    _try_load("BERT weights",    "bert-base-uncased",
              lambda r, **kw: BertForSequenceClassification.from_pretrained(r, num_labels=5, **kw))
    _try_load("CLIP model",      "openai/clip-vit-base-patch32",
              CLIPModel.from_pretrained)
    _try_load("CLIP processor",  "openai/clip-vit-base-patch32",
              CLIPProcessor.from_pretrained)

    downloaded = {
        "vit":  "google/vit-base-patch16-224",
        "bert": "bert-base-uncased",
        "clip": "openai/clip-vit-base-patch32",
        "cache_dir": cache_dir,
    }
    with open(MODELS_DIR / "downloaded_models.json", "w") as f:
        json.dump(downloaded, f, indent=2)

    print(f"\n  Model cache: {cache_dir}")
    return downloaded


# ── Step 4: Run training on sorted dataset ────────────────────────────────
def run_training(model_paths: dict):
    print("\n" + "="*60)
    print("STEP 4: Running FarmFederate training")
    print("="*60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    import torch
    import numpy as np
    from PIL import Image
    from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
    from torchvision import transforms
    from transformers import (
        ViTForImageClassification, ViTFeatureExtractor,
        BertTokenizer, BertForSequenceClassification,
        CLIPProcessor, CLIPModel,
        AutoTokenizer, AutoModel,
    )
    from sklearn.metrics import classification_report, f1_score, confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")

    # ── Dataset ──────────────────────────────────────────────────────────
    class RealCropDataset(Dataset):
        def __init__(self, root: Path, transform=None, tokenizer=None, max_len=64):
            self.samples = []   # (img_path, label_idx)
            self.transform = transform
            self.tokenizer = tokenizer
            self.max_len = max_len
            for idx, cls in enumerate(CLASS_NAMES):
                cls_dir = root / cls
                if not cls_dir.exists():
                    continue
                for img_file in cls_dir.glob("*.jpg"):
                    self.samples.append((img_file, idx))
                for img_file in cls_dir.glob("*.png"):
                    self.samples.append((img_file, idx))

        def __len__(self): return len(self.samples)

        def __getitem__(self, i):
            img_path, label = self.samples[i]
            img = Image.open(img_path).convert("RGB")
            if self.transform:
                img = self.transform(img)

            # Simple text prompt from class name
            text = f"This crop leaf shows signs of {CLASS_NAMES[label].replace('_', ' ').lower()}"
            if self.tokenizer:
                enc = self.tokenizer(
                    text, max_length=self.max_len, padding="max_length",
                    truncation=True, return_tensors="pt"
                )
                return img, enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0), label
            return img, label

    # ── Transforms ───────────────────────────────────────────────────────
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    tokenizer = BertTokenizer.from_pretrained(str(model_paths.get("bert", "bert-base-uncased")))

    full_ds = RealCropDataset(SORTED_DIR, transform=train_tf, tokenizer=tokenizer)
    print(f"\n  Total samples: {len(full_ds)}")

    # 80/20 split
    n_val = max(1, int(0.2 * len(full_ds)))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        full_ds, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )
    # Val uses val_tf
    val_ds.dataset.transform = val_tf

    # Weighted sampler to handle class imbalance
    labels_all = [full_ds.samples[i][1] for i in range(len(full_ds))]
    counts_all = Counter(labels_all)
    weights_all = [1.0 / counts_all[full_ds.samples[i][1]] for i in range(len(full_ds))]
    train_weights = [weights_all[i] for i in train_ds.indices]
    sampler = WeightedRandomSampler(train_weights, num_samples=len(train_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=16, sampler=sampler, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=16, shuffle=False,    num_workers=0)

    results_all = {}

    # ── Model A: ViT ─────────────────────────────────────────────────────
    print("\n  [A] Training ViT …")
    vit = ViTForImageClassification.from_pretrained(
        str(model_paths.get("vit", "google/vit-base-patch16-224")),
        num_labels=5,
        ignore_mismatched_sizes=True,
    ).to(device)

    opt_vit = torch.optim.AdamW(vit.parameters(), lr=2e-5, weight_decay=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    best_f1_vit = 0.0
    for epoch in range(10):
        vit.train()
        total_loss = 0
        for imgs, input_ids, attn_mask, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            opt_vit.zero_grad()
            out = vit(pixel_values=imgs)
            loss = criterion(out.logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vit.parameters(), 1.0)
            opt_vit.step()
            total_loss += loss.item()

        # Validate
        vit.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, input_ids, attn_mask, labels in val_loader:
                imgs = imgs.to(device)
                out = vit(pixel_values=imgs)
                preds = out.logits.argmax(dim=-1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())
        f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        print(f"    Epoch {epoch+1:2d}/10 | loss={total_loss/len(train_loader):.4f} | val_F1={f1:.4f}")
        if f1 > best_f1_vit:
            best_f1_vit = f1
            torch.save(vit.state_dict(), RESULTS_DIR / "best_vit.pt")

    # Final ViT report
    vit.load_state_dict(torch.load(RESULTS_DIR / "best_vit.pt", map_location=device))
    vit.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, input_ids, attn_mask, labels in val_loader:
            imgs = imgs.to(device)
            out = vit(pixel_values=imgs)
            all_preds.extend(out.logits.argmax(dim=-1).cpu().numpy())
            all_labels.extend(labels.numpy())
    report_vit = classification_report(all_labels, all_preds, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    results_all["ViT"] = report_vit
    print(f"\n  ViT best macro-F1: {best_f1_vit:.4f}")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES, zero_division=0))

    # ── Model B: BERT (text-only) ─────────────────────────────────────────
    print("\n  [B] Training BERT (text) …")
    bert = BertForSequenceClassification.from_pretrained(
        str(model_paths.get("bert", "bert-base-uncased")),
        num_labels=5,
    ).to(device)

    opt_bert = torch.optim.AdamW(bert.parameters(), lr=2e-5, weight_decay=0.01)
    best_f1_bert = 0.0
    for epoch in range(10):
        bert.train()
        total_loss = 0
        for imgs, input_ids, attn_mask, labels in train_loader:
            input_ids, attn_mask, labels = input_ids.to(device), attn_mask.to(device), labels.to(device)
            opt_bert.zero_grad()
            out = bert(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(bert.parameters(), 1.0)
            opt_bert.step()
            total_loss += out.loss.item()

        bert.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, input_ids, attn_mask, labels in val_loader:
                input_ids, attn_mask = input_ids.to(device), attn_mask.to(device)
                out = bert(input_ids=input_ids, attention_mask=attn_mask)
                all_preds.extend(out.logits.argmax(dim=-1).cpu().numpy())
                all_labels.extend(labels.numpy())
        f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        print(f"    Epoch {epoch+1:2d}/10 | loss={total_loss/len(train_loader):.4f} | val_F1={f1:.4f}")
        if f1 > best_f1_bert:
            best_f1_bert = f1
            torch.save(bert.state_dict(), RESULTS_DIR / "best_bert.pt")

    bert.load_state_dict(torch.load(RESULTS_DIR / "best_bert.pt", map_location=device))
    bert.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, input_ids, attn_mask, labels in val_loader:
            input_ids, attn_mask = input_ids.to(device), attn_mask.to(device)
            out = bert(input_ids=input_ids, attention_mask=attn_mask)
            all_preds.extend(out.logits.argmax(dim=-1).cpu().numpy())
            all_labels.extend(labels.numpy())
    report_bert = classification_report(all_labels, all_preds, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    results_all["BERT"] = report_bert
    print(f"\n  BERT best macro-F1: {best_f1_bert:.4f}")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES, zero_division=0))

    # ── Model C: CLIP (zero-shot style multimodal) ─────────────────────
    print("\n  [C] Training CLIP Multimodal …")
    clip_model = CLIPModel.from_pretrained(
        str(model_paths.get("clip", "openai/clip-vit-base-patch32"))
    ).to(device)
    clip_proc  = CLIPProcessor.from_pretrained(
        str(model_paths.get("clip", "openai/clip-vit-base-patch32"))
    )

    # Add a linear head on top of CLIP image features
    clip_head = torch.nn.Linear(clip_model.config.projection_dim, 5).to(device)
    opt_clip  = torch.optim.AdamW(
        list(clip_model.parameters()) + list(clip_head.parameters()), lr=1e-5
    )

    clip_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    class CLIPDataset(Dataset):
        def __init__(self, base_ds):
            self.base = base_ds
        def __len__(self): return len(self.base)
        def __getitem__(self, i):
            img_path, label = full_ds.samples[self.base.indices[i]]
            img = Image.open(img_path).convert("RGB")
            img = clip_tf(img)
            return img, label

    clip_train_ds = CLIPDataset(train_ds)
    clip_val_ds   = CLIPDataset(val_ds)
    clip_train_loader = DataLoader(clip_train_ds, batch_size=16, sampler=sampler, num_workers=0)
    clip_val_loader   = DataLoader(clip_val_ds,   batch_size=16, shuffle=False,   num_workers=0)

    best_f1_clip = 0.0
    for epoch in range(10):
        clip_model.train(); clip_head.train()
        total_loss = 0
        for imgs, labels in clip_train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            opt_clip.zero_grad()
            feats = clip_model.get_image_features(pixel_values=imgs)
            logits = clip_head(feats)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(clip_model.parameters()) + list(clip_head.parameters()), 1.0
            )
            opt_clip.step()
            total_loss += loss.item()

        clip_model.eval(); clip_head.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in clip_val_loader:
                imgs = imgs.to(device)
                feats = clip_model.get_image_features(pixel_values=imgs)
                preds = clip_head(feats).argmax(dim=-1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())
        f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        print(f"    Epoch {epoch+1:2d}/10 | loss={total_loss/len(clip_train_loader):.4f} | val_F1={f1:.4f}")
        if f1 > best_f1_clip:
            best_f1_clip = f1
            torch.save({
                "clip": clip_model.state_dict(),
                "head": clip_head.state_dict()
            }, RESULTS_DIR / "best_clip.pt")

    ckpt = torch.load(RESULTS_DIR / "best_clip.pt", map_location=device)
    clip_model.load_state_dict(ckpt["clip"]); clip_head.load_state_dict(ckpt["head"])
    clip_model.eval(); clip_head.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in clip_val_loader:
            imgs = imgs.to(device)
            feats = clip_model.get_image_features(pixel_values=imgs)
            all_preds.extend(clip_head(feats).argmax(dim=-1).cpu().numpy())
            all_labels.extend(labels.numpy())
    report_clip = classification_report(all_labels, all_preds, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    results_all["CLIP_Multimodal"] = report_clip
    print(f"\n  CLIP best macro-F1: {best_f1_clip:.4f}")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES, zero_division=0))

    return results_all, all_labels, all_preds


# ── Step 5: Save results & plots ──────────────────────────────────────────
def save_results(results_all: dict, last_labels, last_preds):
    print("\n" + "="*60)
    print("STEP 5: Saving results and plots")
    print("="*60)

    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    from sklearn.metrics import confusion_matrix

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # JSON summary
    with open(RESULTS_DIR / "classification_results.json", "w") as f:
        json.dump(results_all, f, indent=2)
    print(f"  Saved: {RESULTS_DIR / 'classification_results.json'}")

    # Bar chart: macro-F1 per model
    models  = list(results_all.keys())
    f1s     = [results_all[m]["macro avg"]["f1-score"] for m in models]
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(models, f1s, color=["#2196F3", "#4CAF50", "#FF9800"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Macro F1-Score")
    ax.set_title("FarmFederate — Model Comparison on Real Dataset")
    for bar, v in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", fontsize=11)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "model_comparison.png", dpi=150)
    plt.close()
    print(f"  Saved: {RESULTS_DIR / 'model_comparison.png'}")

    # Per-class F1 heatmap
    fig, ax = plt.subplots(figsize=(10, 4))
    per_class = np.array([
        [results_all[m].get(cls, {}).get("f1-score", 0) for cls in CLASS_NAMES]
        for m in models
    ])
    sns.heatmap(per_class, annot=True, fmt=".2f", xticklabels=CLASS_NAMES,
                yticklabels=models, cmap="YlOrRd", vmin=0, vmax=1, ax=ax)
    ax.set_title("Per-Class F1-Score by Model")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "per_class_f1_heatmap.png", dpi=150)
    plt.close()
    print(f"  Saved: {RESULTS_DIR / 'per_class_f1_heatmap.png'}")

    # Confusion matrix for last trained model
    cm = confusion_matrix(last_labels, last_preds)
    fig, ax = plt.subplots(figsize=(7, 6))
    short_names = [c.replace("_", "\n") for c in CLASS_NAMES]
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=short_names,
                yticklabels=short_names, cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion Matrix — CLIP Multimodal")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close()
    print(f"  Saved: {RESULTS_DIR / 'confusion_matrix.png'}")

    print(f"\n  All results in: {RESULTS_DIR}")


# ── Main ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\nFarmFederate — Real Dataset Setup & Training")
    print("=" * 60)

    manifest    = sort_images()
    install_deps()
    model_paths = download_models()
    results, last_labels, last_preds = run_training(model_paths)
    save_results(results, last_labels, last_preds)

    print("\n" + "="*60)
    print("ALL DONE")
    print(f"  Sorted images : {SORTED_DIR}")
    print(f"  Trained models: {RESULTS_DIR}")
    print(f"  Plots         : {RESULTS_DIR}")
    print("="*60)
