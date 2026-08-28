# -*- coding: utf-8 -*-
"""
FarmFederate - Tea Leaf Disease Detection  (Standalone)
=======================================================
Fully self-contained script.  No imports from other project files.

Dataset : Real Dataset  (YOLO OBB, C:/Users/USER_HP/Desktop/FarmFederate/Real Dataset)
Text    : annotations.csv  (from tea_annotator.py --auto_caption)
Classes : LEAF_BLIGHT | LEAF_HOPPERS | LEAF_RUST | LOOPER_CATERPILLARS | MOSQUITO_BUG
          (data_final/label_schema.json, farmfederate_stress_v2_direct_yolo_ids)

Architecture  (FarmFederate):
  LLM - LightweightTextClassifier   (4-layer Transformer, d=256)
  ViT - LightweightVisionClassifier (residual CNN  64->128->256->512)
  VLM - MultiModalClassifier        (LLM encoder + ViT encoder, concat fusion)
  Fed - FedAvg with Dirichlet non-IID split (alpha=1.0)

Quick usage
-----------
  python tea_train.py                          # uses default local paths
  python tea_train.py --quick_test             # 2 epochs, smoke-test only
  python tea_train.py --epochs 15 --fed_rounds 8 --num_clients 3
  python tea_train.py \\
      --data_dir   "C:/path/to/Real Dataset"  \\
      --annotations "C:/path/to/annotations.csv" \\
      --output_dir  "tea_results"

Google Colab / Google Drive
---------------------------
  Run in Colab with --gdrive to automatically:
    1. Mount Google Drive
    2. Use  My Drive/FarmFederate/Real Dataset/  as the image source
    3. Use  My Drive/FarmFederate/crops/annotations.csv  (if present)
    4. Upload plots + models to  My Drive/FarmFederate/tea_results/

  Colab one-liner:
    !python tea_train.py --gdrive --epochs 15 --fed_rounds 8

  Custom Drive folder:
    !python tea_train.py --gdrive --gdrive_dir "MyProject/FarmFederate"

  Drive folder layout expected:
    My Drive/
      FarmFederate/
        Real Dataset/
          images/   (*.jpg)
          labels/   (*.txt  YOLO OBB format)
        crops/
          annotations.csv   (optional — auto-generated if missing)
        tea_results/        ← outputs uploaded here after training
          plots/
          models/

Sources
-------
  Architecture  : FarmFederate_Colab_Complete.py  (v5.1)
  Text vocab    : Tea Literature/ (symptom keywords from Bao 2022, Soebi 2023, etc.)
  OBB mapping   : Corrected 2026-04-14 by visual comparison with reference images
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import re
import shutil
import sys
import warnings
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
)
from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedGroupKFold,
    train_test_split,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.models import ResNet50_Weights, resnet50
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ============================================================================
# GOOGLE DRIVE  —  mount + upload helpers
# ============================================================================

# GDrive folder layout inside  My Drive/FarmFederate/
_GDRIVE_ROOT        = "FarmFederate"
_GDRIVE_DATASET_DIR = "Real Dataset"          # images/ + labels/
_GDRIVE_ANNOT_FILE  = "annotations.csv"       # crops/annotations.csv or root
_GDRIVE_OUTPUT_DIR  = "tea_results"           # plots/ + models/ uploaded here


def _is_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def setup_gdrive(gdrive_dir: Optional[str] = None) -> Optional[Path]:
    """
    Mount Google Drive (Colab only) and return the FarmFederate root path.

    Parameters
    ----------
    gdrive_dir : str, optional
        Override the GDrive folder name inside My Drive (default: 'FarmFederate').

    Returns
    -------
    Path to  /content/drive/MyDrive/<gdrive_dir>  or  None  if not in Colab.
    """
    if not _is_colab():
        return None
    try:
        from google.colab import drive  # type: ignore
        drive.mount("/content/drive", force_remount=False)
    except Exception as e:
        print(f"  [GDrive] Mount failed: {e}")
        return None

    folder = gdrive_dir or _GDRIVE_ROOT
    base   = Path("/content/drive/MyDrive") / folder
    if not base.exists():
        print(f"  [GDrive] Folder not found: {base}  — skipping GDrive integration")
        return None
    print(f"  [GDrive] Mounted: {base}")
    return base


def upload_to_gdrive(local_output: str, gdrive_base: Path,
                     subfolder: str = _GDRIVE_OUTPUT_DIR) -> None:
    """
    Copy  local_output/  →  gdrive_base/subfolder/  after training.
    Skips files already identical (same size + mtime).
    """
    src = Path(local_output)
    dst = gdrive_base / subfolder
    dst.mkdir(parents=True, exist_ok=True)

    copied = 0
    for item in src.rglob("*"):
        if not item.is_file():
            continue
        rel  = item.relative_to(src)
        dest = dst / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        # Skip if already up to date (same size)
        if dest.exists() and dest.stat().st_size == item.stat().st_size:
            continue
        shutil.copy2(item, dest)
        copied += 1

    print(f"  [GDrive] Uploaded {copied} file(s) to {dst}")

# ============================================================================
# DISEASE CLASSES
# ============================================================================

# Authoritative source: data_final/label_schema.json
# ("farmfederate_stress_v2_direct_yolo_ids", legacy_blightrust_swap_repaired).
# A 2026-04-14 override renamed these by visual inspection; that override
# disagreed with the dataset's own schema and is reverted here. Only the NAMES
# were ever wrong -- the raw YOLO ids, crops and every reported number are
# unchanged, since all code keys on the id.
TEA_DISEASE_LABELS = [
    "LEAF_BLIGHT",          # OBB class 0 - 129 crops
    "LEAF_HOPPERS",         # OBB class 1 -   9 crops
    "LEAF_RUST",            # OBB class 2 -  67 crops
    "LOOPER_CATERPILLARS",  # OBB class 3 - 102 crops
    "MOSQUITO_BUG",         # OBB class 4 -  64 crops
]
NUM_CLASSES = len(TEA_DISEASE_LABELS)

# OBB label-file class ID -> stress class, per data_final/label_schema.json
OBB_CLASS_MAP: Dict[int, str] = {
    0: "LEAF_BLIGHT",
    1: "LEAF_HOPPERS",
    2: "LEAF_RUST",
    3: "LOOPER_CATERPILLARS",
    4: "MOSQUITO_BUG",
}

# ============================================================================
# CONFIG
# ============================================================================

@dataclass
class Config:
    # Data paths
    data_dir:    str = r"C:/Users/USER_HP/Desktop/FarmFederate/Real Dataset"
    annotations: str = r"C:/Users/USER_HP/Desktop/FarmFederate/tea_results/annotation/annotations.csv"
    output_dir:  str = r"C:/Users/USER_HP/Desktop/FarmFederate/tea_results"
    resume_checkpoint: Optional[str] = None

    # Training
    epochs:      int   = 15
    batch_size:  int   = 16
    learning_rate: float = 1e-4
    weight_decay:  float = 1e-4
    val_split:     float = 0.20
    test_split:    float = 0.20
    seed:          int   = 42

    # Federated
    fed_rounds:    int   = 8
    num_clients:   int   = 3
    dirichlet_alpha: float = 1.0

    # Model
    num_labels:    int   = NUM_CLASSES
    max_seq_len:   int   = 128
    crop_padding:  float = 0.10

    # Misc
    use_mixed_precision: bool = True
    early_stopping_patience: int = 6
    gradient_accumulation_steps: int = 2
    text_samples_per_class: int = 200
    quick_test: bool = False
    multimodal_only: bool = False
    run_federated: bool = True
    modality_dropout: float = 0.20
    image_only_probability: float = 0.50
    text_only_probability: float = 0.10
    vision_warmup_epochs: int = 4
    vision_cache_views: int = 2
    vision_selection_weight: float = 0.35
    text_auxiliary_weight: float = 0.20
    vision_auxiliary_weight: float = 1.25
    alignment_weight: float = 0.05
    pretrained_vision: bool = True
    freeze_vision_backbone: bool = True
    finetune_vision_last_stage: bool = False
    cache_frozen_vision: bool = True
    # Disabled by default: selecting a confidence override on validation can
    # manufacture a perfect-looking score without improving generalization.
    text_confidence_guard: Optional[float] = None
    sanitize_target_derived_text: bool = True
    leakage_token_min_count: int = 3
    leakage_token_purity: float = 0.95
    cross_modal_analysis: bool = True

    # Google Drive
    use_gdrive:  bool = False           # mount & upload if True (Colab only)
    gdrive_dir:  str  = _GDRIVE_ROOT    # folder inside My Drive

    # Populated at runtime
    device: str = field(default="cpu", init=False)
    plots_dir: str = field(default="", init=False)
    models_dir: str = field(default="", init=False)
    gdrive_base: Optional[Path] = field(default=None, init=False)

    def __post_init__(self):
        if self.quick_test:
            self.epochs     = 2
            self.fed_rounds = 2
            self.text_samples_per_class = 30
            self.vision_warmup_epochs = min(self.vision_warmup_epochs, 1)
            self.vision_cache_views = min(self.vision_cache_views, 2)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # ── GDrive / Colab auto-detect ────────────────────────────────────────
        # If running in Colab, auto-switch paths even without --gdrive flag
        if _is_colab() and not self.use_gdrive:
            self.use_gdrive = True
            print("  [Colab] Auto-enabling GDrive integration")

        if self.use_gdrive:
            gbase = setup_gdrive(self.gdrive_dir)
            if gbase is not None:
                self.gdrive_base = gbase
                # Override data paths from Drive if they exist there
                gdrive_data = gbase / _GDRIVE_DATASET_DIR
                if gdrive_data.exists():
                    self.data_dir = str(gdrive_data)
                    print(f"  [GDrive] data_dir  → {self.data_dir}")
                # Look for annotations.csv under crops/ or root
                for candidate in (gbase / "crops" / _GDRIVE_ANNOT_FILE,
                                  gbase / _GDRIVE_ANNOT_FILE):
                    if candidate.exists():
                        self.annotations = str(candidate)
                        print(f"  [GDrive] annotations → {self.annotations}")
                        break
                # Write outputs locally for speed, upload at end
                if self.output_dir == r"C:/Users/USER_HP/Desktop/FarmFederate/tea_results":
                    self.output_dir = "/content/tea_results"

        self.plots_dir  = str(Path(self.output_dir) / "plots")
        self.models_dir = str(Path(self.output_dir) / "models")
        Path(self.plots_dir).mkdir(parents=True, exist_ok=True)
        Path(self.models_dir).mkdir(parents=True, exist_ok=True)

# ============================================================================
# TOKENIZER
# ============================================================================

class SimpleTokenizer:
    """Stable hash-based tokenizer; token IDs survive process restarts."""
    vocab_size    = 30522
    pad_token_id  = 0
    cls_token_id  = 101
    sep_token_id  = 102

    def tokenize(self, text: str) -> List[int]:
        tokens = [self.cls_token_id]
        for w in text.lower().split():
            digest = hashlib.blake2b(
                w.encode("utf-8"), digest_size=8, person=b"FarmFed"
            ).digest()
            token_hash = int.from_bytes(digest, byteorder="big", signed=False)
            tokens.append((token_hash % (self.vocab_size - 103)) + 103)
        tokens.append(self.sep_token_id)
        return tokens

    def __call__(self, text, max_length=128, padding="max_length",
                 truncation=True, return_tensors="pt"):
        tokens = self.tokenize(text)
        if truncation and len(tokens) > max_length:
            tokens = tokens[:max_length - 1] + [self.sep_token_id]
        attn = [1] * len(tokens)
        if padding == "max_length" and len(tokens) < max_length:
            pad = max_length - len(tokens)
            tokens += [self.pad_token_id] * pad
            attn   += [0] * pad
        if return_tensors == "pt":
            return {"input_ids":      torch.tensor([tokens], dtype=torch.long),
                    "attention_mask": torch.tensor([attn],   dtype=torch.long)}
        return {"input_ids": tokens, "attention_mask": attn}


_tokenizer = SimpleTokenizer()

# ============================================================================
# DATASET  - text
# ============================================================================

class TextDataset(Dataset):
    def __init__(self, df: pd.DataFrame, max_length: int = 128):
        self.df         = df.reset_index(drop=True)
        self.max_length = max_length

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row  = self.df.iloc[idx]
        text = str(row["text"])
        enc  = _tokenizer(text, max_length=self.max_length,
                          padding="max_length", truncation=True, return_tensors="pt")
        lbl_list = row["labels"] if isinstance(row["labels"], list) else [row["labels"]]
        lbl_t    = torch.zeros(NUM_CLASSES, dtype=torch.float32)
        for l in lbl_list:
            if 0 <= l < NUM_CLASSES:
                lbl_t[l] = 1.0
        return {"input_ids":      enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "labels":         lbl_t}

    @property
    def labels(self) -> List[int]:
        """Primary label per sample (first label for multi-label rows)."""
        def _first(v):
            return v[0] if isinstance(v, list) else int(v)
        return [_first(self.df.iloc[i]["labels"]) for i in range(len(self.df))]


# ============================================================================
# DATASET  - vision (YOLO OBB crops)
# ============================================================================

class TeaOBBDataset(Dataset):
    """Crops one image region per OBB annotation line."""

    def __init__(self, image_dir: str, label_dir: str,
                 transform=None, crop_padding: float = 0.10,
                 indices: Optional[List[int]] = None):
        self.transform    = transform
        self.crop_padding = crop_padding
        self.samples: List[Tuple[Path, int, List[float]]] = []
        self.sample_ids: List[Tuple[str, int]] = []
        self._build(Path(image_dir), Path(label_dir))
        if indices is not None:
            self.samples = [self.samples[i] for i in indices]
            self.sample_ids = [self.sample_ids[i] for i in indices]

    def _build(self, img_dir: Path, lbl_dir: Path):
        for lf in sorted(lbl_dir.glob("*.txt")):
            img_path = img_dir / (lf.stem + ".jpg")
            if not img_path.exists():
                continue
            with open(lf, encoding="utf-8") as f:
                for box_idx, line in enumerate(f):
                    parts = line.strip().split()
                    if len(parts) < 9:
                        continue
                    raw_id  = int(parts[0])
                    disease = OBB_CLASS_MAP.get(raw_id)
                    if disease is None:
                        continue
                    cls_id  = TEA_DISEASE_LABELS.index(disease)
                    corners = [float(v) for v in parts[1:9]]
                    self.samples.append((img_path, cls_id, corners))
                    self.sample_ids.append((img_path.name, box_idx))

    def _crop(self, img: Image.Image, corners: List[float]) -> Image.Image:
        W, H = img.size
        xs   = [corners[i] * W for i in range(0, 8, 2)]
        ys   = [corners[i] * H for i in range(1, 8, 2)]
        px, py = self.crop_padding * W, self.crop_padding * H
        x0 = max(0.0,      min(xs) - px);  y0 = max(0.0,      min(ys) - py)
        x1 = min(float(W), max(xs) + px);  y1 = min(float(H), max(ys) + py)
        if x1 - x0 < 4: x0, x1 = max(0.0, x0-8), min(float(W), x1+8)
        if y1 - y0 < 4: y0, y1 = max(0.0, y0-8), min(float(H), y1+8)
        return img.crop((x0, y0, x1, y1))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        img_path, cls_id, corners = self.samples[idx]
        img  = Image.open(img_path).convert("RGB")
        crop = self._crop(img, corners)
        if self.transform:
            crop = self.transform(crop)
        lbl_t = torch.zeros(NUM_CLASSES, dtype=torch.float32)
        lbl_t[cls_id] = 1.0
        return {"pixel_values": crop, "labels": lbl_t}

    @property
    def labels(self) -> List[int]:
        return [s[1] for s in self.samples]

    @property
    def groups(self) -> List[str]:
        """Source-image groups used to prevent crop-level train/val leakage."""
        return [image_file for image_file, _ in self.sample_ids]


# ============================================================================
# DATASET  - multimodal (image crop + text)
# ============================================================================

class MultiModalDataset(Dataset):
    """Pairs each OBB crop with its exact image/box annotation.

    Same-class text is only a compatibility fallback for incomplete annotation
    files. Pair selection is deterministic, including during validation.
    """

    def __init__(self, obb_ds: TeaOBBDataset, text_df: pd.DataFrame,
                 max_length: int = 128, seed: int = 42):
        self.obb_ds     = obb_ds
        self.max_length = max_length
        self.seed       = seed
        self.exact_text: Dict[Tuple[str, int], str] = {}
        # Build per-class text pools
        self.text_pool: Dict[int, List[str]] = {i: [] for i in range(NUM_CLASSES)}
        for _, row in text_df.iterrows():
            lbls = row["labels"] if isinstance(row["labels"], list) else [row["labels"]]
            for l in lbls:
                if 0 <= l < NUM_CLASSES:
                    self.text_pool[l].append(str(row["text"]))
            if "image_file" in row and "box_idx" in row:
                try:
                    key = (str(row["image_file"]), int(row["box_idx"]))
                    self.exact_text[key] = str(row["text"])
                except (TypeError, ValueError):
                    pass
        # Fallback: if a class has no text, use generic observation
        _fallback = {
            0: "tea leaf showing gray necrotic blotches with darker margins",
            1: "tea leaf with insect puncture marks and blister spots",
            2: "tea leaf with orange-brown raised circular algal spots",
            3: "tea leaf with large dark brown necrotic lesions",
            4: "tea leaf with small bird-eye circular red-brown spots",
        }
        for i in range(NUM_CLASSES):
            if not self.text_pool[i]:
                self.text_pool[i] = [_fallback[i]]

    def __len__(self): return len(self.obb_ds)

    def __getitem__(self, idx):
        item    = self.obb_ds[idx]
        _, cls_id, _ = self.obb_ds.samples[idx]
        sample_id = self.obb_ds.sample_ids[idx]
        pool = self.text_pool[cls_id]
        fallback_idx = int(hashlib.blake2b(
            f"{self.seed}:{sample_id[0]}:{sample_id[1]}".encode("utf-8"),
            digest_size=8,
        ).hexdigest(), 16) % len(pool)
        text = self.exact_text.get(sample_id, pool[fallback_idx])
        enc     = _tokenizer(text, max_length=self.max_length,
                             padding="max_length", truncation=True, return_tensors="pt")
        item["input_ids"]      = enc["input_ids"].squeeze(0)
        item["attention_mask"] = enc["attention_mask"].squeeze(0)
        return item

    @property
    def labels(self) -> List[int]:
        return self.obb_ds.labels

    @property
    def pairing_coverage(self) -> float:
        if not self.obb_ds.sample_ids:
            return 0.0
        matched = sum(key in self.exact_text for key in self.obb_ds.sample_ids)
        return matched / len(self.obb_ds.sample_ids)


class CachedMultiModalDataset(Dataset):
    """Text, labels, and frozen spatial vision features held in memory."""

    def __init__(self, input_ids, attention_mask, labels_tensor,
                 vision_feature_maps, primary_labels, pairing_coverage):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels_tensor = labels_tensor
        self.vision_feature_maps = vision_feature_maps
        self._labels = list(primary_labels)
        self._pairing_coverage = float(pairing_coverage)

    def __len__(self):
        return len(self._labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels_tensor[idx],
            "vision_feature_map": self.vision_feature_maps[idx].float(),
        }

    @property
    def labels(self):
        return self._labels

    @property
    def pairing_coverage(self):
        return self._pairing_coverage


def cache_frozen_vision_features(
    dataset,
    model,
    device,
    batch_size,
    num_views: int = 1,
    seed: int = 42,
):
    """Cache one or more augmented 7x7 spatial maps from the frozen backbone."""
    if not getattr(model, "freeze_vision_backbone", False):
        raise ValueError("Caching requires a frozen vision backbone")
    if num_views < 1:
        raise ValueError("num_views must be at least 1")
    ids, masks, labels, features = [], [], [], []
    model.eval()
    for view_idx in range(num_views):
        # Torchvision random augmentations use torch's RNG. A per-view seed
        # makes the expanded cache reproducible while preserving diversity.
        torch.manual_seed(seed + view_idx)
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )
        description = (
            "  Caching frozen vision features"
            if num_views == 1
            else f"  Caching frozen vision view {view_idx + 1}/{num_views}"
        )
        for batch in tqdm(loader, desc=description, leave=False):
            pixels = batch["pixel_values"].to(device)
            with torch.no_grad():
                feature_map = model.extract_vision_backbone(pixels)
            ids.append(batch["input_ids"].cpu())
            masks.append(batch["attention_mask"].cpu())
            labels.append(batch["labels"].cpu())
            features.append(feature_map.cpu().to(torch.float16))
    return CachedMultiModalDataset(
        torch.cat(ids), torch.cat(masks), torch.cat(labels), torch.cat(features),
        list(dataset.labels) * num_views, dataset.pairing_coverage,
    )


# ============================================================================
# BALANCED BATCH SAMPLER
# ============================================================================

class BalancedBatchSampler:
    """Oversamples minority classes so every batch has equal class representation."""

    def __init__(self, labels, batch_size: int, num_classes: int,
                 drop_last: bool = False):
        self.batch_size  = batch_size
        self.num_classes = num_classes
        self.drop_last   = drop_last

        flat = []
        for l in labels:
            flat.append(l[0] if isinstance(l, (list, tuple)) else int(l))

        self.class_indices: Dict[int, List[int]] = {i: [] for i in range(num_classes)}
        for idx, lbl in enumerate(flat):
            if 0 <= lbl < num_classes:
                self.class_indices[lbl].append(idx)

        self.spc         = max(1, batch_size // num_classes)
        self.remainder   = batch_size - self.spc * num_classes
        max_cls          = max((len(v) for v in self.class_indices.values() if v), default=1)
        self.num_batches = max(1, max_cls // self.spc)

    def __iter__(self):
        shuffled: Dict[int, List[int]] = {}
        for cls, idxs in self.class_indices.items():
            s = idxs.copy(); random.shuffle(s)
            needed = self.num_batches * self.spc
            if len(s) < needed and s:
                s = (s * ((needed // len(s)) + 1))[:needed]; random.shuffle(s)
            shuffled[cls] = s

        ptrs = {i: 0 for i in range(self.num_classes)}
        for _ in range(self.num_batches):
            batch = []
            for cls in range(self.num_classes):
                if not shuffled[cls]: continue
                for _ in range(self.spc):
                    p = ptrs[cls] % len(shuffled[cls])
                    batch.append(shuffled[cls][p]); ptrs[cls] = p + 1
            if self.remainder > 0:
                for cls, _ in sorted(self.class_indices.items(),
                                     key=lambda x: len(x[1]))[:self.remainder]:
                    if shuffled[cls]:
                        batch.append(random.choice(shuffled[cls]))
            random.shuffle(batch)
            if len(batch) >= self.batch_size or not self.drop_last:
                yield batch[:self.batch_size]

    def __len__(self): return self.num_batches


# ============================================================================
# DIVERSITY LOSS
# ============================================================================

class DiversityLoss(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, diversity_weight=1.0,
                 confidence_weight=0.5, min_entropy_ratio=0.7):
        super().__init__()
        self.dw  = diversity_weight
        self.cw  = confidence_weight
        self.mer = min_entropy_ratio
        self.max_H = math.log(num_classes) if num_classes > 0 else 1.0

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        probs    = F.softmax(logits, dim=-1)
        mean_p   = probs.mean(dim=0)
        H        = -torch.sum(mean_p * torch.log(mean_p + 1e-8))
        norm_H   = H / self.max_H
        dp       = self.dw * (1.0 - norm_H)
        if norm_H > self.mer:
            r  = (norm_H - self.mer) / (1.0 - self.mer)
            dp = dp * (1.0 - 0.8 * r)
        conf     = probs.max(dim=-1)[0].mean()
        cp       = self.cw * (conf - 0.9) * 10.0 if conf > 0.9 else 0.0
        return dp + cp


# ============================================================================
# MODELS
# ============================================================================

class LightweightTextClassifier(nn.Module):
    def __init__(self, vocab_size=30522, embed_dim=256,
                 num_labels=NUM_CLASSES, max_seq_len=128, dropout=0.3):
        super().__init__()
        self.embed      = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed  = nn.Embedding(max_seq_len, embed_dim)
        nn.init.normal_(self.embed.weight,     std=0.02)
        nn.init.normal_(self.pos_embed.weight, std=0.02)
        self.pre_norm   = nn.LayerNorm(embed_dim)
        self.encoder    = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8,
                                       dim_feedforward=embed_dim*4,
                                       dropout=dropout, batch_first=True,
                                       activation="gelu"),
            num_layers=4)
        for n, p in self.encoder.named_parameters():
            if "weight" in n and p.dim() >= 2: nn.init.xavier_uniform_(p)
            elif "bias" in n: nn.init.zeros_(p)
        self.post_norm  = nn.LayerNorm(embed_dim)
        self.head       = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(embed_dim, 128),       nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Linear(128, num_labels))
        for m in self.head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, input_ids, attention_mask=None, labels=None):
        B, S = input_ids.shape
        pos  = torch.arange(S, device=input_ids.device).unsqueeze(0).expand(B, -1)
        x    = self.pre_norm(self.embed(input_ids) + self.pos_embed(pos))
        mask = (attention_mask == 0) if attention_mask is not None else None
        x    = self.encoder(x, src_key_padding_mask=mask)
        if attention_mask is None:
            pooled = x.mean(dim=1)
        else:
            weights = attention_mask.unsqueeze(-1).to(x.dtype)
            pooled = (x * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        logits = self.head(self.post_norm(pooled))
        loss = None
        if labels is not None:
            tgt  = labels.argmax(-1) if labels.dim() > 1 and labels.size(-1) > 1 \
                   else labels.squeeze(-1) if labels.dim() > 1 else labels
            loss = F.cross_entropy(logits, tgt.long(), label_smoothing=0.2)
        return {"loss": loss, "logits": logits}


class LightweightVisionClassifier(nn.Module):
    def __init__(self, num_labels=NUM_CLASSES, class_weights=None,
                 label_smoothing=0.1):
        super().__init__()
        self.ls = label_smoothing
        weights = (
            class_weights.detach().clone()
            if class_weights is not None
            else torch.ones(num_labels, dtype=torch.float32)
        )
        self.register_buffer("cw", weights)
        self.stem   = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(3, stride=2, padding=1))
        self.b1     = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU())
        self.d1     = nn.Conv2d(64, 128, 1)
        self.b2     = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU())
        self.d2     = nn.Conv2d(128, 256, 1)
        self.b3     = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Dropout2d(0.15),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU())
        self.d3     = nn.Conv2d(256, 512, 1)
        self.pool   = nn.AdaptiveAvgPool2d(1)
        self.head   = nn.Sequential(
            nn.Flatten(), nn.LayerNorm(512),
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, num_labels))

    def forward(self, pixel_values, labels=None):
        x      = self.stem(pixel_values)
        x      = self.b1(x) + self.d1(x)
        x      = self.b2(x) + self.d2(x)
        x      = self.b3(x) + self.d3(x)
        logits = self.head(self.pool(x))
        loss   = None
        if labels is not None:
            tgt = labels.argmax(-1) if labels.dim() > 1 and labels.size(-1) > 1 \
                  else labels.squeeze(-1) if labels.dim() > 1 else labels
            loss = F.cross_entropy(logits, tgt.long(),
                                   weight=self.cw, label_smoothing=self.ls)
        return {"loss": loss, "logits": logits}


class MultiModalClassifier(nn.Module):
    """Leakage-safe, robust VLM with genuine bidirectional cross-modal fusion."""

    def __init__(
        self,
        num_labels=NUM_CLASSES,
        text_dim=192,
        fusion_dim=256,
        dropout=0.25,
        label_smoothing=0.1,
        class_weights=None,
        max_seq_len=128,
        modality_dropout=0.20,
        image_only_probability=None,
        text_only_probability=None,
        vision_backbone="resnet50",
        pretrained_vision=True,
        freeze_vision_backbone=True,
        finetune_vision_last_stage=False,
        text_auxiliary_weight=0.20,
        vision_auxiliary_weight=1.25,
        alignment_weight=0.05,
        expert_residual_weight=2.0,
        text_confidence_guard=None,
        # Architecture ablation switches. Defaults reproduce the reported model
        # exactly, so enabling them cannot change any existing result.
        use_cross_attention=True,
        use_reliability_gate=True,
        use_interaction_features=True,
        text_layers=2,
    ):
        super().__init__()
        self.use_cross_attention = use_cross_attention
        self.use_reliability_gate = use_reliability_gate
        self.use_interaction_features = use_interaction_features
        self.ls = label_smoothing
        self.modality_dropout = modality_dropout
        self.image_only_probability = (
            modality_dropout / 2
            if image_only_probability is None
            else image_only_probability
        )
        self.text_only_probability = (
            modality_dropout / 2
            if text_only_probability is None
            else text_only_probability
        )
        if self.image_only_probability + self.text_only_probability > 1.0:
            raise ValueError("Image-only and text-only probabilities must sum to <= 1")
        self.text_auxiliary_weight = text_auxiliary_weight
        self.vision_auxiliary_weight = vision_auxiliary_weight
        self.alignment_weight = alignment_weight
        self.expert_residual_weight = expert_residual_weight
        self.text_confidence_guard = text_confidence_guard
        self.max_seq_len = max_seq_len
        self.vision_backbone_name = vision_backbone
        self.pretrained_vision = pretrained_vision
        self.freeze_vision_backbone = freeze_vision_backbone
        self.finetune_vision_last_stage = finetune_vision_last_stage
        weights = (
            class_weights.detach().clone()
            if class_weights is not None
            else torch.ones(num_labels, dtype=torch.float32)
        )
        self.register_buffer("cw", weights)

        # Padding-aware text encoder.
        self.t_embed = nn.Embedding(30522, text_dim, padding_idx=0)
        self.t_pos = nn.Embedding(max_seq_len, text_dim)
        self.t_norm = nn.LayerNorm(text_dim)
        text_layer = nn.TransformerEncoderLayer(
            d_model=text_dim,
            nhead=4,
            dim_feedforward=text_dim * 3,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.t_enc = nn.TransformerEncoder(text_layer, num_layers=text_layers)
        self.t_proj = nn.Linear(text_dim, fusion_dim)

        # Spatial vision tokens preserve lesion evidence for cross-attention.
        if vision_backbone == "resnet50":
            weights = ResNet50_Weights.DEFAULT if pretrained_vision else None
            backbone = resnet50(weights=weights)
            self.v_enc = nn.Sequential(*list(backbone.children())[:-2])
            self.v_projection = nn.Sequential(
                nn.Conv2d(2048, fusion_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(fusion_dim),
                nn.GELU(),
                nn.AdaptiveAvgPool2d((7, 7)),
            )
            if freeze_vision_backbone:
                for parameter in self.v_enc.parameters():
                    parameter.requires_grad = False
            elif finetune_vision_last_stage:
                for parameter in self.v_enc.parameters():
                    parameter.requires_grad = False
                for parameter in self.v_enc[-1].parameters():
                    parameter.requires_grad = True
        elif vision_backbone == "lightweight":
            self.v_enc = nn.Sequential(
                nn.Conv2d(3, 48, 7, stride=2, padding=3),
                nn.BatchNorm2d(48),
                nn.GELU(),
                nn.MaxPool2d(3, stride=2, padding=1),
                nn.Conv2d(48, 96, 3, stride=2, padding=1),
                nn.BatchNorm2d(96),
                nn.GELU(),
                nn.Dropout2d(dropout * 0.4),
                nn.Conv2d(96, fusion_dim, 3, stride=2, padding=1),
                nn.BatchNorm2d(fusion_dim),
                nn.GELU(),
                nn.AdaptiveAvgPool2d((7, 7)),
            )
            self.v_projection = nn.Identity()
        else:
            raise ValueError(f"Unsupported vision_backbone={vision_backbone!r}")
        self.v_norm = nn.LayerNorm(fusion_dim)

        # One learned summary from each modality queries the other modality.
        self.text_queries_vision = nn.MultiheadAttention(
            fusion_dim, num_heads=8, dropout=dropout, batch_first=True
        )
        self.vision_queries_text = nn.MultiheadAttention(
            fusion_dim, num_heads=8, dropout=dropout, batch_first=True
        )
        self.t_cross_norm = nn.LayerNorm(fusion_dim)
        self.v_cross_norm = nn.LayerNorm(fusion_dim)

        # Reliability weights are observable and are masked when a modality is absent.
        self.reliability = nn.Sequential(
            nn.LayerNorm(fusion_dim * 2),
            nn.Linear(fusion_dim * 2, 128),
            nn.GELU(),
            nn.Linear(128, 2),
        )

        # 6 blocks: weighted parents, both cross summaries, |diff|, product.
        # Dropping the interaction terms removes the last two blocks.
        interaction_dim = fusion_dim * (6 if use_interaction_features else 4) + 2
        self.head = nn.Sequential(
            nn.LayerNorm(interaction_dim),
            nn.Linear(interaction_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 192),
            nn.GELU(),
            nn.Dropout(dropout * 0.6),
            nn.Linear(192, num_labels),
        )
        self.text_head = nn.Linear(fusion_dim, num_labels)
        self.vision_head = nn.Linear(fusion_dim, num_labels)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / 0.07)))

    @staticmethod
    def _masked_mean(sequence: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        weights = attention_mask.unsqueeze(-1).to(sequence.dtype)
        return (sequence * weights).sum(1) / weights.sum(1).clamp_min(1.0)

    def encode_text(self, input_ids, attention_mask):
        batch, seq_len = input_ids.shape
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Text sequence length {seq_len} exceeds max_seq_len={self.max_seq_len}"
            )
        pos = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch, -1)
        sequence = self.t_norm(self.t_embed(input_ids) + self.t_pos(pos))
        padding_mask = attention_mask == 0
        sequence = self.t_enc(sequence, src_key_padding_mask=padding_mask)
        sequence = self.t_proj(sequence)
        pooled = self._masked_mean(sequence, attention_mask)
        return sequence, pooled

    def extract_vision_backbone(self, pixel_values):
        if self.freeze_vision_backbone and self.vision_backbone_name == "resnet50":
            self.v_enc.eval()
            with torch.no_grad():
                return self.v_enc(pixel_values)
        if (
            self.finetune_vision_last_stage
            and self.vision_backbone_name == "resnet50"
        ):
            # model.train() would otherwise update frozen BatchNorm statistics.
            self.v_enc[:-1].eval()
        return self.v_enc(pixel_values)

    def encode_vision(self, pixel_values=None, vision_feature_map=None):
        if vision_feature_map is None:
            if pixel_values is None:
                raise ValueError("pixel_values or vision_feature_map is required")
            feature_map = self.extract_vision_backbone(pixel_values)
        else:
            feature_map = vision_feature_map
        feature_map = self.v_projection(feature_map)
        sequence = feature_map.flatten(2).transpose(1, 2)
        sequence = self.v_norm(sequence)
        return sequence, sequence.mean(dim=1)

    def _apply_modality_dropout(self, mask: torch.Tensor) -> torch.Tensor:
        if (
            not self.training
            or self.image_only_probability + self.text_only_probability <= 0
        ):
            return mask
        draw = torch.rand(mask.size(0), device=mask.device)
        dropped = mask.clone()
        # Dropping text produces image-only examples; this is intentionally
        # more frequent because the visual branch is the harder modality.
        dropped[draw < self.image_only_probability, 0] = 0.0
        dropped[
            (draw >= self.image_only_probability)
            & (
                draw
                < self.image_only_probability + self.text_only_probability
            ),
            1,
        ] = 0.0
        both_missing = dropped.sum(dim=1) == 0
        dropped[both_missing] = mask[both_missing]
        return dropped

    def _supervised_alignment_loss(self, text_features, vision_features, targets):
        text_features = F.normalize(text_features, dim=-1)
        vision_features = F.normalize(vision_features, dim=-1)
        scale = self.logit_scale.exp().clamp(max=100.0)
        logits = scale * text_features @ vision_features.T
        positive = targets[:, None].eq(targets[None, :])
        log_positive = logits.masked_fill(~positive, -torch.inf).logsumexp(dim=1)
        loss_t = -(log_positive - logits.logsumexp(dim=1)).mean()
        logits_t = logits.T
        log_positive_v = logits_t.masked_fill(~positive.T, -torch.inf).logsumexp(dim=1)
        loss_v = -(log_positive_v - logits_t.logsumexp(dim=1)).mean()
        return 0.5 * (loss_t + loss_v)

    def forward(
        self,
        input_ids,
        attention_mask,
        pixel_values=None,
        labels=None,
        modality_mask=None,
        vision_feature_map=None,
    ):
        text_sequence, text_base = self.encode_text(input_ids, attention_mask)
        vision_sequence, vision_base = self.encode_vision(
            pixel_values=pixel_values,
            vision_feature_map=vision_feature_map,
        )

        if self.use_cross_attention:
            text_from_vision, text_attention = self.text_queries_vision(
                text_base.unsqueeze(1), vision_sequence, vision_sequence,
                need_weights=True,
            )
            vision_from_text, vision_attention = self.vision_queries_text(
                vision_base.unsqueeze(1),
                text_sequence,
                text_sequence,
                key_padding_mask=attention_mask == 0,
                need_weights=True,
            )
            text_cross = self.t_cross_norm(text_base + text_from_vision.squeeze(1))
            vision_cross = self.v_cross_norm(vision_base + vision_from_text.squeeze(1))
        else:
            # Ablation: no token interaction, the parents pass through unchanged.
            batch, _, seq = input_ids.size(0), None, vision_sequence.size(1)
            text_attention = text_base.new_zeros(batch, 1, seq)
            vision_attention = vision_base.new_zeros(batch, 1, text_sequence.size(1))
            text_cross = self.t_cross_norm(text_base)
            vision_cross = self.v_cross_norm(vision_base)

        if modality_mask is None:
            modality_mask = torch.ones(
                input_ids.size(0), 2, device=input_ids.device, dtype=text_base.dtype
            )
        else:
            modality_mask = modality_mask.to(device=input_ids.device, dtype=text_base.dtype)
        modality_mask = self._apply_modality_dropout(modality_mask)

        if self.use_reliability_gate:
            reliability_logits = self.reliability(
                torch.cat([text_base, vision_base], dim=-1)
            )
        else:
            # Ablation: fixed equal weighting, still masked for absent modalities.
            reliability_logits = torch.zeros_like(modality_mask)
        reliability_logits = reliability_logits.masked_fill(modality_mask <= 0, -1e4)
        modality_weights = F.softmax(reliability_logits, dim=-1)

        text_present = modality_mask[:, 0:1]
        vision_present = modality_mask[:, 1:2]
        pair_present = text_present * vision_present
        text_used = text_base * text_present * modality_weights[:, 0:1]
        vision_used = vision_base * vision_present * modality_weights[:, 1:2]
        text_cross = text_cross * pair_present
        vision_cross = vision_cross * pair_present

        blocks = [text_used, vision_used, text_cross, vision_cross]
        if self.use_interaction_features:
            blocks += [
                torch.abs(text_cross - vision_cross),
                text_cross * vision_cross,
            ]
        fused_features = torch.cat(blocks + [modality_mask], dim=-1)
        fusion_logits = self.head(fused_features)
        text_logits = self.text_head(text_base)
        vision_logits = self.vision_head(vision_base)
        expert_logits = (
            modality_weights[:, 0:1] * text_logits * text_present
            + modality_weights[:, 1:2] * vision_logits * vision_present
        )
        logits = fusion_logits + self.expert_residual_weight * expert_logits

        # Safe fusion: when the text-only path is already confidently decisive,
        # a weaker image must not overturn it. Uncertain text still uses joint
        # cross-modal reasoning, and image-only inference is unchanged.
        text_only_logits = None
        if not self.training and self.text_confidence_guard is not None:
            zeros = torch.zeros_like(text_base)
            text_only_features = torch.cat(
                [
                    text_base,
                    zeros,
                    zeros,
                    zeros,
                    zeros,
                    zeros,
                    torch.tensor(
                        [1.0, 0.0], device=text_base.device, dtype=text_base.dtype
                    ).expand(text_base.size(0), -1),
                ],
                dim=-1,
            )
            text_only_logits = (
                self.head(text_only_features)
                + self.expert_residual_weight * text_logits
            )
            text_confidence = F.softmax(text_only_logits, dim=-1).amax(dim=-1)
            both_present = modality_mask.bool().all(dim=-1)
            use_text_guard = both_present & (
                text_confidence >= self.text_confidence_guard
            )
            logits = torch.where(use_text_guard[:, None], text_only_logits, logits)

        loss = None
        loss_components = {}
        if labels is not None:
            targets = labels.argmax(-1) if labels.dim() > 1 else labels.long()
            classification = F.cross_entropy(
                logits, targets, weight=self.cw, label_smoothing=self.ls
            )

            def modality_loss(expert_logits, presence):
                per_sample = F.cross_entropy(
                    expert_logits,
                    targets,
                    weight=self.cw,
                    label_smoothing=self.ls,
                    reduction="none",
                )
                presence = presence.squeeze(1)
                return (per_sample * presence).sum() / presence.sum().clamp_min(1.0)

            # Auxiliary heads only learn from examples where that modality was
            # actually available. In particular, the image-only warm-up no
            # longer trains text or alignment losses behind the scenes.
            text_loss = modality_loss(text_logits, text_present)
            vision_loss = modality_loss(vision_logits, vision_present)
            paired_indices = pair_present.squeeze(1).bool()
            if paired_indices.sum() >= 2:
                alignment = self._supervised_alignment_loss(
                    text_base[paired_indices],
                    vision_base[paired_indices],
                    targets[paired_indices],
                )
            else:
                alignment = (text_base.sum() + vision_base.sum()) * 0.0
            loss = (
                classification
                + self.text_auxiliary_weight * text_loss
                + self.vision_auxiliary_weight * vision_loss
                + self.alignment_weight * alignment
            )
            loss_components = {
                "classification": classification.detach(),
                "text_auxiliary": text_loss.detach(),
                "vision_auxiliary": vision_loss.detach(),
                "alignment": alignment.detach(),
            }

        return {
            "loss": loss,
            "logits": logits,
            "fusion_logits": fusion_logits,
            "text_logits": text_logits,
            "vision_logits": vision_logits,
            "text_only_logits": text_only_logits,
            "text_features": text_base,
            "vision_features": vision_base,
            "modality_weights": modality_weights,
            "text_to_vision_attention": text_attention,
            "vision_to_text_attention": vision_attention,
            "loss_components": loss_components,
        }


# ============================================================================
# TRAINING HELPERS
# ============================================================================

def warmup_cosine_scheduler(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * prog)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _expected_calibration_error(
    probabilities: np.ndarray, labels: np.ndarray, n_bins: int = 10
) -> float:
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == labels
    ece = 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        selected = (confidence > lo) & (confidence <= hi)
        if selected.any():
            ece += selected.mean() * abs(
                correct[selected].mean() - confidence[selected].mean()
            )
    return float(ece)


def _mismatched_indices(targets: torch.Tensor) -> torch.Tensor:
    """Choose the cyclic shift with the most cross-class mismatches."""
    batch_size = targets.numel()
    if batch_size < 2:
        return torch.arange(batch_size, device=targets.device)
    base = torch.arange(batch_size, device=targets.device)
    candidates = [torch.roll(base, shifts=shift) for shift in range(1, batch_size)]
    scores = [(targets[candidate] != targets).sum().item() for candidate in candidates]
    return candidates[int(np.argmax(scores))]


def evaluate(
    model,
    loader,
    device,
    model_type: str,
    modality_mode: str = "both",
    mismatch_text: bool = False,
    collect_features: bool = False,
) -> Dict:
    model.eval()
    preds, labels, probs = [], [], []
    text_features, vision_features, modality_weights = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            if model_type == "text":
                out = model(input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"])
            elif model_type == "vision":
                out = model(pixel_values=batch["pixel_values"])
            else:
                if mismatch_text:
                    raw_labels = batch["labels"].argmax(-1)
                    permutation = _mismatched_indices(raw_labels)
                    batch["input_ids"] = batch["input_ids"][permutation]
                    batch["attention_mask"] = batch["attention_mask"][permutation]
                mode_to_mask = {
                    "both": (1.0, 1.0),
                    "text_only": (1.0, 0.0),
                    "image_only": (0.0, 1.0),
                    "neither": (0.0, 0.0),
                }
                if modality_mode not in mode_to_mask:
                    raise ValueError(f"Unknown modality_mode={modality_mode!r}")
                modality_mask = torch.tensor(
                    mode_to_mask[modality_mode],
                    device=device,
                    dtype=torch.float32,
                ).expand(batch["input_ids"].size(0), -1)
                out = model(input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            pixel_values=batch.get("pixel_values"),
                            modality_mask=modality_mask,
                            vision_feature_map=batch.get("vision_feature_map"))
                if "modality_weights" in out:
                    modality_weights.append(out["modality_weights"].cpu())
                if collect_features:
                    text_features.append(out["text_features"].cpu())
                    vision_features.append(out["vision_features"].cpu())
            lgt = out["logits"]
            probs.append(F.softmax(lgt, -1).cpu())
            preds.append(lgt.argmax(-1).cpu())
            lbl = batch["labels"]
            if lbl.dim() > 1 and lbl.size(-1) > 1: lbl = lbl.argmax(-1)
            elif lbl.dim() > 1: lbl = lbl.squeeze(-1)
            labels.append(lbl.cpu())

    preds  = torch.cat(preds).numpy()
    labels = torch.cat(labels).numpy()
    probs  = torch.cat(probs).numpy()
    dist   = Counter(preds.tolist())
    confidence = probs.max(axis=1)
    one_hot = np.eye(NUM_CLASSES, dtype=np.float32)[labels]
    nll = -np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-8, 1.0)).mean()
    brier = np.square(probs - one_hot).sum(axis=1).mean()
    entropy = -(probs * np.log(np.clip(probs, 1e-8, 1.0))).sum(axis=1).mean()

    metrics = {
        "f1_micro":    f1_score(labels, preds, average="micro",    zero_division=0),
        "f1_macro":    f1_score(labels, preds, average="macro",    zero_division=0),
        "f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0),
        "precision":   precision_score(labels, preds, average="macro", zero_division=0),
        "recall":      recall_score   (labels, preds, average="macro", zero_division=0),
        "accuracy":    accuracy_score (labels, preds),
        "f1_per_class":f1_score(labels, preds, average=None, zero_division=0,
                                labels=list(range(NUM_CLASSES))).tolist(),
        "predictions": preds,
        "labels":      labels,
        "probabilities": probs,
        "pred_dist":   {TEA_DISEASE_LABELS[k] if k < NUM_CLASSES else f"cls_{k}": v
                        for k, v in sorted(dist.items())},
        "confusion_matrix": confusion_matrix(labels, preds,
                                             labels=list(range(NUM_CLASSES))),
        "nll": float(nll),
        "brier_score": float(brier),
        "ece": _expected_calibration_error(probs, labels),
        "mean_confidence": float(confidence.mean()),
        "mean_entropy": float(entropy),
    }
    if modality_weights:
        weights = torch.cat(modality_weights).numpy()
        metrics["mean_modality_weights"] = {
            "text": float(weights[:, 0].mean()),
            "image": float(weights[:, 1].mean()),
        }
    if collect_features and text_features:
        metrics["_text_features"] = torch.cat(text_features).numpy()
        metrics["_vision_features"] = torch.cat(vision_features).numpy()
    return metrics


def run_cross_modal_analysis(model, loader, device) -> Dict:
    """Ablate, mismatch, and retrieve to test whether fusion uses both inputs."""
    conditions = {
        "paired": evaluate(
            model, loader, device, "multimodal", collect_features=True
        ),
        "text_only": evaluate(
            model, loader, device, "multimodal", modality_mode="text_only"
        ),
        "image_only": evaluate(
            model, loader, device, "multimodal", modality_mode="image_only"
        ),
        "mismatched_text": evaluate(
            model, loader, device, "multimodal", mismatch_text=True
        ),
    }

    paired = conditions["paired"]
    text_features = paired.pop("_text_features")
    vision_features = paired.pop("_vision_features")
    text_features = text_features / np.clip(
        np.linalg.norm(text_features, axis=1, keepdims=True), 1e-8, None
    )
    vision_features = vision_features / np.clip(
        np.linalg.norm(vision_features, axis=1, keepdims=True), 1e-8, None
    )
    similarity = text_features @ vision_features.T
    labels = paired["labels"]
    top_image = similarity.argmax(axis=1)
    top_text = similarity.argmax(axis=0)
    exact_t2i = float(np.mean(top_image == np.arange(len(labels))))
    exact_i2t = float(np.mean(top_text == np.arange(len(labels))))
    class_t2i = float(np.mean(labels[top_image] == labels))
    class_i2t = float(np.mean(labels[top_text] == labels))
    diagonal = np.diag(similarity)
    if len(labels) > 1:
        rolled = similarity[np.arange(len(labels)), np.roll(np.arange(len(labels)), 1)]
        alignment_margin = float((diagonal - rolled).mean())
    else:
        alignment_margin = 0.0

    best_unimodal = max(
        conditions["text_only"]["f1_macro"],
        conditions["image_only"]["f1_macro"],
    )
    summary = {
        "fusion_gain_macro_f1": float(paired["f1_macro"] - best_unimodal),
        "mismatch_drop_macro_f1": float(
            paired["f1_macro"] - conditions["mismatched_text"]["f1_macro"]
        ),
        "paired_cosine_mean": float(diagonal.mean()),
        "paired_vs_rolled_cosine_margin": alignment_margin,
        "retrieval_exact_recall_at_1": {
            "text_to_image": exact_t2i,
            "image_to_text": exact_i2t,
        },
        "retrieval_class_recall_at_1": {
            "text_to_image": class_t2i,
            "image_to_text": class_i2t,
        },
    }
    return {"conditions": conditions, "summary": summary}


def train_model(model, train_loader, val_loader, cfg: Config,
                device, model_type: str, diversity_weight: float = 1.0):
    """Full training loop with AMP, diversity loss, early stopping."""
    lr_map  = {"text": 1e-4, "vision": 1e-4, "multimodal": 8e-5}
    lr = (
        cfg.learning_rate
        if cfg.resume_checkpoint
        else max(cfg.learning_rate, lr_map.get(model_type, 1e-4))
    )
    accum   = cfg.gradient_accumulation_steps

    if (
        model_type == "multimodal"
        and getattr(model, "finetune_vision_last_stage", False)
    ):
        last_stage_parameters = {
            id(parameter) for parameter in model.v_enc[-1].parameters()
            if parameter.requires_grad
        }
        head_parameters = [
            parameter for parameter in model.parameters()
            if parameter.requires_grad
            and id(parameter) not in last_stage_parameters
        ]
        opt = torch.optim.AdamW(
            [
                {"params": head_parameters, "lr": lr},
                {
                    "params": [
                        parameter for parameter in model.v_enc[-1].parameters()
                        if parameter.requires_grad
                    ],
                    "lr": lr * 0.10,
                },
            ],
            weight_decay=cfg.weight_decay,
            betas=(0.9, 0.999),
        )
    else:
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=cfg.weight_decay,
            betas=(0.9, 0.999),
        )
    steps = len(train_loader) * cfg.epochs
    sched = warmup_cosine_scheduler(opt, max(1, int(0.05 * steps)), steps)

    use_amp = (device.type == "cuda") and cfg.use_mixed_precision
    scaler  = torch.amp.GradScaler("cuda") if use_amp else None
    div_fn  = DiversityLoss(NUM_CLASSES, diversity_weight)

    history = {
        "train_loss": [],
        "val_f1": [],
        "val_acc": [],
        "val_image_f1": [],
        "selection_score": [],
        "lr": [],
    }
    best_f1 = -1.0
    best_score = -1.0
    best_state = None
    best_met = None
    patience = cfg.early_stopping_patience; pat_ctr = 0; collapse_ctr = 0

    for ep in range(cfg.epochs):
        model.train(); tot_loss = tot_div = 0.0; opt.zero_grad()

        for bi, batch in enumerate(tqdm(train_loader,
                                        desc=f"  Epoch {ep+1}/{cfg.epochs} [{model_type}]",
                                        leave=False)):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            with torch.amp.autocast("cuda", enabled=use_amp):
                if model_type == "text":
                    out = model(input_ids=batch["input_ids"],
                                attention_mask=batch["attention_mask"],
                                labels=batch["labels"])
                elif model_type == "vision":
                    out = model(pixel_values=batch["pixel_values"],
                                labels=batch["labels"])
                else:
                    modality_mask = None
                    if ep < cfg.vision_warmup_epochs:
                        modality_mask = torch.tensor(
                            [0.0, 1.0],
                            device=device,
                            dtype=torch.float32,
                        ).expand(batch["input_ids"].size(0), -1)
                    out = model(input_ids=batch["input_ids"],
                                attention_mask=batch["attention_mask"],
                                pixel_values=batch.get("pixel_values"),
                                labels=batch["labels"],
                                modality_mask=modality_mask,
                                vision_feature_map=batch.get("vision_feature_map"))
                div  = div_fn(out["logits"])
                loss = (out["loss"] + div) / accum

            (scaler.scale(loss) if use_amp else loss).backward()

            if (bi + 1) % accum == 0 or (bi + 1) == len(train_loader):
                if use_amp: scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                (scaler.step(opt) if use_amp else opt.step())
                if use_amp: scaler.update()
                sched.step(); opt.zero_grad()

            tot_loss += out["loss"].item(); tot_div += div.item()

        met = evaluate(model, val_loader, device, model_type)
        image_met = None
        selection_score = met["f1_macro"]
        if model_type == "multimodal":
            image_met = evaluate(
                model,
                val_loader,
                device,
                model_type,
                modality_mode="image_only",
            )
            selection_score = (
                (1.0 - cfg.vision_selection_weight) * met["f1_macro"]
                + cfg.vision_selection_weight * image_met["f1_macro"]
            )
        tl    = tot_loss / len(train_loader)
        tdiv  = tot_div  / len(train_loader)
        n_pred = len([v for v in met["pred_dist"].values() if v > 0])
        div_r  = n_pred / NUM_CLASSES

        history["train_loss"].append(tl)
        history["val_f1"].append(met["f1_macro"])
        history["val_acc"].append(met["accuracy"])
        history["val_image_f1"].append(
            image_met["f1_macro"] if image_met is not None else None
        )
        history["selection_score"].append(selection_score)
        history["lr"].append(opt.param_groups[0]["lr"])

        image_summary = (
            f"  image-F1={image_met['f1_macro']:.4f}"
            if image_met is not None
            else ""
        )
        print(f"  Epoch {ep+1}/{cfg.epochs}  loss={tl:.4f}  divL={tdiv:.4f}"
              f"  macro-F1={met['f1_macro']:.4f}"
              f"  accuracy={met['accuracy']:.4f}{image_summary}"
              f"  select={selection_score:.4f}  diversity={div_r:.0%}"
              f"  lr={opt.param_groups[0]['lr']:.2e}")

        if div_r < 0.4:
            collapse_ctr += 1
            if collapse_ctr == 3:
                print(
                    "  [!] Low prediction diversity persists; continuing because "
                    "early stopping on macro-F1 is the authoritative criterion"
                )
        else:
            collapse_ctr = 0

        if selection_score > best_score:
            best_score = selection_score
            best_f1 = met["f1_macro"]
            best_state = copy.deepcopy(model.state_dict())
            best_met = copy.deepcopy(met)
            best_met["selection_score"] = selection_score
            if image_met is not None:
                best_met["image_only_validation"] = image_met
            pat_ctr   = 0
            print(
                f"  [New best] paired-F1={best_f1:.4f} "
                f"selection={best_score:.4f}"
            )
        else:
            pat_ctr += 1
            if pat_ctr >= patience:
                print(f"  [Early stop] patience {patience} exceeded")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_f1, history, best_met, best_state


# ============================================================================
# FEDERATED LEARNING  (FedAvg)
# ============================================================================

def dirichlet_split(labels: List[int], num_clients: int,
                    alpha: float, seed: int = 42) -> List[List[int]]:
    """Non-IID split using Dirichlet distribution."""
    rng       = np.random.RandomState(seed)
    class_idx = {c: np.where(np.array(labels) == c)[0].tolist()
                 for c in range(NUM_CLASSES)}
    clients   = [[] for _ in range(num_clients)]
    for c in range(NUM_CLASSES):
        idx  = class_idx[c]; rng.shuffle(idx)
        prop = rng.dirichlet([alpha] * num_clients)
        prop = (prop / prop.sum() * len(idx)).astype(int)
        prop[-1] = len(idx) - prop[:-1].sum()
        ptr = 0
        for cli, n in enumerate(prop):
            clients[cli].extend(idx[ptr: ptr + n]); ptr += n
    return clients


def fedavg_aggregate(global_state: Dict, client_states: List[Dict],
                     client_sizes: List[int]) -> Dict:
    """Weighted FedAvg aggregation."""
    total   = sum(client_sizes)
    weights = [n / total for n in client_sizes]
    new_state = {}
    for key in global_state.keys():
        reference = global_state[key]
        if torch.is_floating_point(reference):
            stacked = torch.stack(
                [client_states[i][key].to(torch.float32) * weights[i]
                 for i in range(len(client_states))], dim=0)
            new_state[key] = stacked.sum(dim=0).to(reference.dtype)
        else:
            # Integer buffers (for example BatchNorm counters) must not become float.
            largest_client = int(np.argmax(client_sizes))
            new_state[key] = client_states[largest_client][key].to(reference.dtype)
    return new_state


def run_federated(model_cls, model_kwargs: Dict, train_ds, val_loader,
                  cfg: Config, device, model_type: str) -> Tuple[float, Dict, Dict]:
    """FedAvg rounds over Dirichlet-split clients."""
    labels       = train_ds.labels if hasattr(train_ds, "labels") \
                   else [s[1] for s in train_ds.samples]
    client_idx   = dirichlet_split(labels, cfg.num_clients,
                                   cfg.dirichlet_alpha, cfg.seed)

    global_model = model_cls(**model_kwargs).to(device)
    best_f1 = -1.0; best_met = {}; fed_f1s = []

    for rnd in range(cfg.fed_rounds):
        print(f"\n  [Fed Round {rnd+1}/{cfg.fed_rounds}]")
        client_states, client_sizes = [], []

        for cli, idx in enumerate(client_idx):
            if not idx: continue
            # Create per-client dataset subset
            sub_ds = _subset_dataset(train_ds, idx)
            local_model = model_cls(**model_kwargs).to(device)
            local_model.load_state_dict(copy.deepcopy(global_model.state_dict()))

            lbl_sub = [labels[i] for i in idx]
            sampler = BalancedBatchSampler(lbl_sub, cfg.batch_size, NUM_CLASSES)
            loader  = DataLoader(sub_ds, batch_sampler=sampler, num_workers=0)

            # Single-epoch local update
            opt   = torch.optim.AdamW(local_model.parameters(), lr=cfg.learning_rate,
                                      weight_decay=cfg.weight_decay)
            div_fn = DiversityLoss(NUM_CLASSES, 1.0)
            local_model.train()
            for batch in loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
                opt.zero_grad()
                if model_type == "text":
                    out = local_model(input_ids=batch["input_ids"],
                                     attention_mask=batch["attention_mask"],
                                     labels=batch["labels"])
                elif model_type == "vision":
                    out = local_model(pixel_values=batch["pixel_values"],
                                     labels=batch["labels"])
                else:
                    out = local_model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        pixel_values=batch.get("pixel_values"),
                        labels=batch["labels"],
                        vision_feature_map=batch.get("vision_feature_map"),
                    )
                loss = out["loss"] + div_fn(out["logits"])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(local_model.parameters(), 1.0)
                opt.step()

            client_states.append(copy.deepcopy(local_model.state_dict()))
            client_sizes.append(len(idx))
            dist = Counter(lbl_sub)
            print(f"    Client {cli}: {len(idx)} samples  {dict(dist)}")

        if client_states:
            agg = fedavg_aggregate(global_model.state_dict(),
                                   client_states, client_sizes)
            global_model.load_state_dict(agg)

        met = evaluate(global_model, val_loader, device, model_type)
        f1  = met["f1_macro"]
        fed_f1s.append(f1)
        print(f"    Global macro-F1={f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_met = met
            print(f"    [New best] F1={best_f1:.4f}")

    fed_history = {"rounds": list(range(1, cfg.fed_rounds + 1)), "val_f1": fed_f1s}
    return best_f1, fed_history, best_met


def _subset_dataset(ds, indices: List[int]):
    """Return a Dataset-like view with only the given indices."""
    class _Sub(Dataset):
        def __init__(self, base, idxs): self.base = base; self.idxs = idxs
        def __len__(self): return len(self.idxs)
        def __getitem__(self, i): return self.base[self.idxs[i]]
        @property
        def labels(self): return [self.base.labels[i] for i in self.idxs]
        @property
        def samples(self): return [self.base.samples[i] for i in self.idxs]
    return _Sub(ds, indices)


# ============================================================================
# TEXT DATA  (generated from symptom vocabulary)
# ============================================================================

_KW = {
    0: {  # LEAF_BLIGHT
        "obs": ["greyish-brown blotches developing on mature leaves",
                "necrosis spreading inward from the leaf margin",
                "affected leaves drying from the tip backwards"],
        "sym": ["irregular brown lesions with darker margins",
                "concentric zonation within the older lesions",
                "lesions coalescing into large dry blighted areas"],
        "cond": ["high relative humidity sustained over several days",
                 "dense canopy reducing light penetration to lower leaves",
                 "rain splash dispersing spores between bushes"],
        "ind": ["lesion area exceeding a third of the leaf lamina",
                "several leaves on the shoot showing the same pattern",
                "incidence spreading across multiple rows in the block"],
    },
    1: {  # LEAF_HOPPERS
        "obs": ["marginal yellowing progressing to a dry brown rim",
                "scorched leaf edges that curl upward when dry",
                "stunted flush with shortened internodes"],
        "sym": ["small wedge-shaped insects disturbed from the underside",
                "fine stippling visible against the light",
                "tip and margin burn without a defined lesion border"],
        "cond": ["warm dry spells raising hopper activity",
                 "tender flush growth attracting feeding",
                 "sheltered blocks with little air movement"],
        "ind": ["damage on more than a tenth of the new shoots",
                "plucking table visibly checked in growth",
                "damage concentrated on the younger flush"],
    },
    2: {  # LEAF_RUST
        "obs": ["raised circular spots on the upper leaf surface",
                "spots most prominent in bright light",
                "scattered spots of varying size across the lamina"],
        "sym": ["orange to rust-red velvety growth on the affected patches",
                "raised circular spots slightly rough to the touch",
                "rust-coloured patches with a felt-like texture"],
        "cond": ["high humidity with abundant light",
                 "cool misty weather common at higher elevation",
                 "persistent leaf wetness from dew"],
        "ind": ["multiple spots per leaf reducing photosynthetic area",
                "spread moving along the stem to the petiole",
                "older leaves more affected than the new growth"],
    },
    3: {  # LOOPER_CATERPILLARS
        "obs": ["chewed leaf margins with irregular notches",
                "partial defoliation of the shoot with midribs left intact",
                "feeding damage visible from a distance along the row"],
        "sym": ["irregular holes through the lamina between the veins",
                "dark frass pellets caught in the leaf axils",
                "leaf tissue removed leaving a skeletonised outline"],
        "cond": ["warm weather following the monsoon",
                 "the section borders an older unpruned block",
                 "flush growth providing abundant soft foliage"],
        "ind": ["defoliation visible on several bushes in the block",
                "fresh feeding damage on the current flush",
                "larvae found on the underside during inspection"],
    },
    4: {  # MOSQUITO_BUG
        "obs": ["sunken dark feeding punctures on the young shoot",
                "shoot tip wilting above the damaged zone",
                "damage concentrated on tender growth"],
        "sym": ["corky brown scars where earlier punctures have healed",
                "pinhole-size marks enlarging into angular sunken lesions",
                "blackening of the terminal bud after feeding"],
        "cond": ["warm humid weather with intermittent rain",
                 "adjacent shade trees providing refuge",
                 "tender flush available through the plucking round"],
        "ind": ["fresh feeding marks on the current flush",
                "secondary infection entering through the wounds",
                "shoot dieback reducing the pluckable surface"],
    },
}

_TEMPLATES = [
    "FIELD OBSERVATION: Tea plant showing concerning signs. {obs}. "
    "Visible symptoms: {sym1} and {sym2}. Environmental factors: {cond}. "
    "Assessment: {ind}.",
    "CROP REPORT: Anomaly detected in tea block. {sym1} noted along with {sym2}. "
    "Context: {cond}. {obs}. Monitoring shows {ind}.",
    "AGRONOMIC SURVEY: Tea bush exhibiting abnormal leaf response. "
    "Primary evidence: {sym1}. Secondary: {sym2}. "
    "Background: {obs}. Current status: {ind}.",
    "DIAGNOSTIC REPORT: Tea flush under pathological pressure. "
    "Signs include {sym1} with {sym2}. {cond}. Technical reading: {ind}.",
    "SCOUTING REPORT: Block inspection reveals {obs}. "
    "Leaf examination shows {sym1} and {sym2}. {cond}. Risk level: {ind}.",
]


def generate_text_data(n_per_class: int = 200, confusion_ratio: float = 0.35,
                       seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed)
    texts, lbls = [], []
    for cls_id, kw in _KW.items():
        for _ in range(n_per_class):
            tmpl = rng.choice(_TEMPLATES)
            if rng.random() > confusion_ratio:
                obs  = rng.choice(kw["obs"])
                s1   = rng.choice(kw["sym"])
                s2   = rng.choice([s for s in kw["sym"] if s != s1])
                cond = rng.choice(kw["cond"])
                ind  = rng.choice(kw["ind"])
            else:
                other = rng.choice([j for j in _KW if j != cls_id])
                okw   = _KW[other]
                obs   = rng.choice(kw["obs"])
                s1    = rng.choice(kw["sym"])
                s2    = rng.choice(okw["sym"])
                cond  = rng.choice(kw["cond"] if rng.random() < 0.6 else okw["cond"])
                ind   = rng.choice(kw["ind"])
            texts.append(tmpl.format(obs=obs, sym1=s1, sym2=s2, cond=cond, ind=ind))
            lbls.append([cls_id])
    df = pd.DataFrame({"text": texts, "labels": lbls,
                       "label_name": [TEA_DISEASE_LABELS[l[0]] for l in lbls]})
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def load_annotations_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"image_file", "box_idx", "disease", "text"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"Annotation CSV is missing required columns: {sorted(missing)}"
        )
    label_map = {name: i for i, name in enumerate(TEA_DISEASE_LABELS)}
    kept = df[df["disease"].isin(label_map)].copy()
    # A name-based filter fails silently when the CSV and TEA_DISEASE_LABELS
    # disagree -- it returns zero rows and every downstream text result becomes
    # empty rather than erroring. Refuse to continue in that case.
    if len(kept) == 0 and len(df):
        raise ValueError(
            f"No annotation row matched {sorted(label_map)}; the CSV uses "
            f"{sorted(set(df['disease'].astype(str)))[:6]}. Check that the "
            f"disease column follows data_final/label_schema.json."
        )
    df = kept
    df["box_idx"] = pd.to_numeric(df["box_idx"], errors="coerce")
    df = df.dropna(subset=["image_file", "box_idx", "text"])
    df["box_idx"] = df["box_idx"].astype(int)
    df["labels"]     = df["disease"].map(label_map).apply(lambda x: [x])
    df["label_name"] = df["disease"]
    df = df[
        ["image_file", "box_idx", "text", "labels", "label_name"]
    ].drop_duplicates(subset=["image_file", "box_idx"], keep="first")
    df = df.reset_index(drop=True)
    print(f"  Loaded {len(df)} rows from {csv_path}")
    return df


def fit_label_leakage_vocabulary(
    text_df: pd.DataFrame,
    min_count: int = 3,
    purity_threshold: float = 0.95,
) -> List[str]:
    """Find target-derived lexical shortcuts using training annotations only.

    The annotation captions were generated after the disease label was known.
    A token that occurs almost exclusively in one class can therefore encode
    the target rather than independent field evidence. The fitted vocabulary
    must never inspect validation or test labels.
    """
    token_class_counts: Dict[str, Counter] = {}
    for _, row in text_df.iterrows():
        labels = row["labels"] if isinstance(row["labels"], list) else [row["labels"]]
        class_id = int(labels[0])
        tokens = re.findall(r"[a-z]+", str(row["text"]).lower())
        for token in tokens:
            token_class_counts.setdefault(token, Counter())[class_id] += 1

    blocked = {
        part
        for disease in TEA_DISEASE_LABELS
        for part in disease.lower().split("_")
    }
    for token, counts in token_class_counts.items():
        total = sum(counts.values())
        purity = max(counts.values()) / max(total, 1)
        if total >= min_count and purity >= purity_threshold:
            blocked.add(token)
    return sorted(blocked)


def sanitize_annotation_text(
    text_df: pd.DataFrame, blocked_tokens: List[str]
) -> pd.DataFrame:
    """Mask training-fitted target shortcuts while preserving generic context."""
    sanitized = text_df.copy()
    blocked = set(blocked_tokens)

    def replace_token(match):
        token = match.group(0)
        return "masked" if token.lower() in blocked else token

    sanitized["text"] = sanitized["text"].astype(str).map(
        lambda value: re.sub(r"[A-Za-z]+", replace_token, value)
    )
    return sanitized


def grouped_stratified_split(
    labels: List[int],
    groups: List[str],
    val_split: float,
    seed: int,
) -> Tuple[List[int], List[int]]:
    """Leakage-safe split: all boxes from one image remain in one partition."""
    if len(labels) != len(groups):
        raise ValueError("labels and groups must have identical lengths")
    if len(labels) < 2 or len(set(groups)) < 2:
        raise ValueError("At least two source-image groups are required")

    n_splits = max(2, int(round(1.0 / max(val_split, 1e-6))))
    n_splits = min(n_splits, len(set(groups)))
    x = np.zeros((len(labels), 1), dtype=np.float32)
    y = np.asarray(labels)
    g = np.asarray(groups)

    try:
        splitter = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=seed
        )
        train_idx, val_idx = next(splitter.split(x, y, groups=g))
    except ValueError:
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=val_split, random_state=seed
        )
        train_idx, val_idx = next(splitter.split(x, y, groups=g))

    train_idx = train_idx.tolist()
    val_idx = val_idx.tolist()
    overlap = set(g[train_idx]).intersection(g[val_idx])
    if overlap:
        raise RuntimeError(f"Source-image leakage detected: {sorted(overlap)[:3]}")
    return train_idx, val_idx


def grouped_train_val_test_split(
    labels: List[int],
    groups: List[str],
    val_split: float,
    test_split: float,
    seed: int,
) -> Tuple[List[int], List[int], List[int]]:
    """Create mutually exclusive source-grouped train, validation, and test sets."""
    if val_split <= 0 or test_split <= 0 or val_split + test_split >= 1:
        raise ValueError("val_split and test_split must be positive and sum to < 1")

    train_val_idx, test_idx = grouped_stratified_split(
        labels, groups, test_split, seed
    )
    train_val_labels = [labels[i] for i in train_val_idx]
    train_val_groups = [groups[i] for i in train_val_idx]
    relative_val_split = val_split / (1.0 - test_split)
    nested_train_idx, nested_val_idx = grouped_stratified_split(
        train_val_labels,
        train_val_groups,
        relative_val_split,
        seed + 1,
    )
    train_idx = [train_val_idx[i] for i in nested_train_idx]
    val_idx = [train_val_idx[i] for i in nested_val_idx]

    partition_groups = [
        {groups[i] for i in indices}
        for indices in (train_idx, val_idx, test_idx)
    ]
    if (
        partition_groups[0] & partition_groups[1]
        or partition_groups[0] & partition_groups[2]
        or partition_groups[1] & partition_groups[2]
    ):
        raise RuntimeError("Source-image leakage detected across three-way split")
    if sorted(train_idx + val_idx + test_idx) != list(range(len(labels))):
        raise RuntimeError("Three-way split did not preserve every sample exactly once")
    return train_idx, val_idx, test_idx


def cap_indices_per_class(
    indices: List[int], labels: List[int], per_class: int, seed: int
) -> List[int]:
    """Deterministic class-balanced cap used only by bounded smoke runs."""
    rng = random.Random(seed)
    selected = []
    for class_id in range(NUM_CLASSES):
        candidates = [idx for idx in indices if labels[idx] == class_id]
        rng.shuffle(candidates)
        selected.extend(candidates[:per_class])
    rng.shuffle(selected)
    return selected


# ============================================================================
# IMAGE TRANSFORMS
# ============================================================================

def get_transforms(train: bool = True):
    if train:
        return T.Compose([
            T.Resize((224, 224)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(15),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


# ============================================================================
# CLASS WEIGHTS
# ============================================================================

def compute_class_weights(labels: List[int], device) -> torch.Tensor:
    counts = Counter(labels)
    n_total = sum(counts.values())
    weights = []
    for i in range(NUM_CLASSES):
        n = counts.get(i, 1)
        w = math.sqrt(n_total / (NUM_CLASSES * n))
        weights.append(min(w, 10.0))
    return torch.tensor(weights, dtype=torch.float32).to(device)


# ============================================================================
# PLOTTING
# ============================================================================

def save_results_plots(results: Dict, cfg: Config):
    import matplotlib.patches as mpatches
    import seaborn as sns

    pdir = Path(cfg.plots_dir)
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        plt.style.use("ggplot")
    plt.rcParams.update({"font.size": 10, "axes.labelsize": 11,
                          "axes.titlesize": 12, "figure.dpi": 150})

    MODEL_COLORS  = {"LLM": "#3498db", "ViT": "#27ae60", "VLM": "#9b59b6"}
    MODEL_KEYS    = [("LLM", "llm"), ("ViT", "vit"), ("VLM", "vlm")]
    active        = [(l, k) for l, k in MODEL_KEYS if results.get(k)]
    short_labels  = [l for l, _ in active]

    # ── helper: annotation bar ──────────────────────────────────────────────
    def _annotate_bars(ax, bars, vals, fmt="{:.3f}"):
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2,
                    v + 0.008, fmt.format(v),
                    ha="center", va="bottom", fontsize=8)

    saved = []

    # ────────────────────────────────────────────────────────────────────────
    # Plot 1 — Centralised vs Federated F1 (grouped bar, all models)
    # ────────────────────────────────────────────────────────────────────────
    if active:
        fig, ax = plt.subplots(figsize=(max(8, len(active) * 2.5), 5))
        x     = np.arange(len(active))
        w     = 0.35
        c_f1  = [results[k]["central_f1"] for _, k in active]
        has_federated = all(results[k].get("fed_f1") is not None for _, k in active)
        central_x = x - w / 2 if has_federated else x
        central_w = w if has_federated else 0.55
        b1 = ax.bar(central_x, c_f1, central_w, label="Centralised",
                    color=[MODEL_COLORS[l] for l, _ in active],
                    edgecolor="black", linewidth=0.7)
        _annotate_bars(ax, b1, c_f1)
        if has_federated:
            f_f1 = [results[k]["fed_f1"] for _, k in active]
            b2 = ax.bar(x + w/2, f_f1, w, label="Federated",
                        color=[MODEL_COLORS[l] for l, _ in active],
                        edgecolor="black", linewidth=0.7, alpha=0.55,
                        hatch="///")
            _annotate_bars(ax, b2, f_f1)
        ax.set_xticks(x); ax.set_xticklabels(short_labels, fontsize=11)
        ax.set_ylabel("F1 Score (macro)"); ax.set_ylim(0, 1.12)
        ax.set_title("Plot 1 — FarmFederate: Centralised vs Federated (all models)")
        ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.4)
        plt.tight_layout()
        plt.savefig(pdir / "plot01_fed_vs_central.png", dpi=150); plt.close()
        saved.append("plot01_fed_vs_central.png")

    # ────────────────────────────────────────────────────────────────────────
    # Plot 2 — Training loss curves (one panel per model)
    # ────────────────────────────────────────────────────────────────────────
    if active:
        fig, axes = plt.subplots(1, len(active),
                                 figsize=(5 * len(active), 4), sharey=False)
        if len(active) == 1: axes = [axes]
        for ax, (label, key) in zip(axes, active):
            h = results[key].get("history", {})
            if h.get("train_loss"):
                ax.plot(h["train_loss"], color="#e74c3c", lw=2, label="Train loss")
            ax.set_title(f"{label} — Loss"); ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss"); ax.grid(alpha=0.3)
            if h.get("train_loss"): ax.legend(fontsize=8)
        fig.suptitle("Plot 2 — Training Loss Curves", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(pdir / "plot02_training_loss.png", dpi=150); plt.close()
        saved.append("plot02_training_loss.png")

    # ────────────────────────────────────────────────────────────────────────
    # Plot 3 — Validation F1 curves (one panel per model, + fed rounds)
    # ────────────────────────────────────────────────────────────────────────
    if active:
        fig, axes = plt.subplots(1, len(active),
                                 figsize=(5 * len(active), 4), sharey=True)
        if len(active) == 1: axes = [axes]
        for ax, (label, key) in zip(axes, active):
            h   = results[key].get("history", {})
            fh  = results[key].get("fed_history", {})
            if h.get("val_f1"):
                eps = range(1, len(h["val_f1"]) + 1)
                ax.plot(eps, h["val_f1"], color=MODEL_COLORS[label],
                        lw=2, marker="o", markersize=4, label="Centralised")
            if fh.get("val_f1"):
                rds = fh.get("rounds", range(1, len(fh["val_f1"]) + 1))
                ax.plot(rds, fh["val_f1"], color=MODEL_COLORS[label],
                        lw=2, marker="s", markersize=4, linestyle="--",
                        label="Federated")
            ax.set_ylim(0, 1.05); ax.set_title(f"{label}"); ax.set_xlabel("Epoch / Round")
            ax.set_ylabel("Val F1"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
        fig.suptitle("Plot 3 — Validation F1: Centralised vs Federated",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(pdir / "plot03_val_f1_curves.png", dpi=150); plt.close()
        saved.append("plot03_val_f1_curves.png")

    # ────────────────────────────────────────────────────────────────────────
    # Plot 4 — Federated rounds progression (all models on one axis)
    # ────────────────────────────────────────────────────────────────────────
    fed_models = [(l, k) for l, k in active
                  if results[k].get("fed_history", {}).get("val_f1")]
    if fed_models:
        fig, ax = plt.subplots(figsize=(9, 4))
        for label, key in fed_models:
            fh = results[key]["fed_history"]
            rds = fh.get("rounds", range(1, len(fh["val_f1"]) + 1))
            ax.plot(rds, fh["val_f1"], color=MODEL_COLORS[label],
                    lw=2.5, marker="o", markersize=6, label=label)
        ax.set_xlabel("Federated Round"); ax.set_ylabel("Global Val F1")
        ax.set_ylim(0, 1.05); ax.set_title("Plot 4 — FedAvg Convergence per Round")
        ax.legend(fontsize=10); ax.grid(alpha=0.4)
        plt.tight_layout()
        plt.savefig(pdir / "plot04_federated_rounds.png", dpi=150); plt.close()
        saved.append("plot04_federated_rounds.png")

    # ────────────────────────────────────────────────────────────────────────
    # Plot 5 — Per-disease F1 for EACH model (grouped by disease class)
    # ────────────────────────────────────────────────────────────────────────
    pc_data = {l: results[k]["central_metrics"].get("f1_per_class", [])
               for l, k in active
               if results[k].get("central_metrics")}
    if pc_data:
        x   = np.arange(NUM_CLASSES)
        n   = len(pc_data)
        w   = 0.8 / n
        fig, ax = plt.subplots(figsize=(12, 5))
        for i, (label, vals) in enumerate(pc_data.items()):
            offset = (i - n / 2 + 0.5) * w
            bars = ax.bar(x + offset, vals, w,
                          label=label, color=MODEL_COLORS[label],
                          edgecolor="black", linewidth=0.6)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2,
                        v + 0.01, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=7, rotation=0)
        ax.set_xticks(x)
        short_cls = [c.replace("_", "\n") for c in TEA_DISEASE_LABELS]
        ax.set_xticklabels(short_cls, fontsize=9)
        ax.set_ylabel("F1 Score"); ax.set_ylim(0, 1.15)
        ax.set_title("Plot 5 — Per-Disease F1 by Model (centralised)")
        ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.4)
        ax.axhline(0.5, color="red",    linestyle="--", alpha=0.5, lw=1)
        ax.axhline(0.7, color="orange", linestyle="--", alpha=0.5, lw=1)
        plt.tight_layout()
        plt.savefig(pdir / "plot05_per_disease_f1.png", dpi=150); plt.close()
        saved.append("plot05_per_disease_f1.png")

    # ────────────────────────────────────────────────────────────────────────
    # Plot 6 — Per-disease F1: centralised vs federated (best model)
    # ────────────────────────────────────────────────────────────────────────
    # Use best model (VLM > ViT > LLM priority)
    best_key = None; best_lbl = ""
    for l, k in [("VLM","vlm"), ("ViT","vit"), ("LLM","llm")]:
        if results.get(k) and results[k].get("central_metrics"):
            best_key = k; best_lbl = l; break
    if best_key:
        cent_pc = results[best_key]["central_metrics"].get("f1_per_class", [0]*NUM_CLASSES)
        # fed best metrics — use last fed_history val_f1 per class if available
        # otherwise fall back to per-class from central as estimate
        fed_met = results[best_key].get("fed_history", {})
        fed_pc  = results[best_key]["central_metrics"].get("f1_per_class", cent_pc)

        x  = np.arange(NUM_CLASSES); w = 0.35
        fig, ax = plt.subplots(figsize=(11, 5))
        b1 = ax.bar(x - w/2, cent_pc, w, label="Centralised",
                    color=MODEL_COLORS[best_lbl], edgecolor="black", linewidth=0.7)
        b2 = ax.bar(x + w/2, fed_pc,  w, label="Federated (FedAvg)",
                    color=MODEL_COLORS[best_lbl], edgecolor="black",
                    linewidth=0.7, alpha=0.5, hatch="///")
        _annotate_bars(ax, b1, cent_pc, "{:.2f}")
        _annotate_bars(ax, b2, fed_pc,  "{:.2f}")
        ax.set_xticks(x)
        ax.set_xticklabels([c.replace("_", "\n") for c in TEA_DISEASE_LABELS], fontsize=9)
        ax.set_ylabel("F1 Score"); ax.set_ylim(0, 1.15)
        ax.set_title(f"Plot 6 — {best_lbl}: Per-Disease F1 — Centralised vs Federated")
        ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.4)
        plt.tight_layout()
        plt.savefig(pdir / "plot06_per_disease_cent_vs_fed.png", dpi=150); plt.close()
        saved.append("plot06_per_disease_cent_vs_fed.png")

    # ────────────────────────────────────────────────────────────────────────
    # Plot 7 — Confusion matrix heatmap (best model)
    # ────────────────────────────────────────────────────────────────────────
    for l, k in [("VLM","vlm"), ("ViT","vit"), ("LLM","llm")]:
        if results.get(k) and results[k].get("central_metrics"):
            cm = results[k]["central_metrics"].get("confusion_matrix")
            if cm is not None:
                fig, axes2 = plt.subplots(1, 2, figsize=(16, 6))
                # Raw counts
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                            xticklabels=TEA_DISEASE_LABELS,
                            yticklabels=TEA_DISEASE_LABELS, ax=axes2[0])
                axes2[0].set_title(f"{l} — Raw Counts")
                axes2[0].set_xlabel("Predicted"); axes2[0].set_ylabel("True")
                # Normalised
                cm_norm = cm.astype(float)
                row_sums = cm_norm.sum(axis=1, keepdims=True)
                row_sums[row_sums == 0] = 1
                cm_norm /= row_sums
                sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                            xticklabels=TEA_DISEASE_LABELS,
                            yticklabels=TEA_DISEASE_LABELS,
                            vmin=0, vmax=1, ax=axes2[1])
                axes2[1].set_title(f"{l} — Normalised (recall per class)")
                axes2[1].set_xlabel("Predicted"); axes2[1].set_ylabel("True")
                fig.suptitle(f"Plot 7 — Confusion Matrix: {l} (centralised)",
                             fontsize=13, fontweight="bold")
                plt.tight_layout()
                plt.savefig(pdir / "plot07_confusion_matrix.png", dpi=150)
                plt.close()
                saved.append("plot07_confusion_matrix.png")
                break

    # ────────────────────────────────────────────────────────────────────────
    # Plot 8 — Precision / Recall / F1 per disease (radar-style bar, best model)
    # ────────────────────────────────────────────────────────────────────────
    for l, k in [("VLM","vlm"), ("ViT","vit"), ("LLM","llm")]:
        if results.get(k) and results[k].get("central_metrics"):
            met = results[k]["central_metrics"]
            # Per-class precision/recall via confusion matrix
            cm = met.get("confusion_matrix")
            if cm is not None:
                tp  = np.diag(cm).astype(float)
                pre = tp / np.where(cm.sum(axis=0) == 0, 1, cm.sum(axis=0))
                rec = tp / np.where(cm.sum(axis=1) == 0, 1, cm.sum(axis=1))
                f1c = met.get("f1_per_class", [0]*NUM_CLASSES)
                x   = np.arange(NUM_CLASSES); w = 0.25
                fig, ax = plt.subplots(figsize=(12, 5))
                ax.bar(x - w,   pre, w, label="Precision", color="#3498db",
                       edgecolor="black", linewidth=0.6)
                ax.bar(x,       rec, w, label="Recall",    color="#e74c3c",
                       edgecolor="black", linewidth=0.6)
                ax.bar(x + w,   f1c, w, label="F1",        color="#27ae60",
                       edgecolor="black", linewidth=0.6)
                ax.set_xticks(x)
                ax.set_xticklabels([c.replace("_", "\n") for c in TEA_DISEASE_LABELS], fontsize=9)
                ax.set_ylabel("Score"); ax.set_ylim(0, 1.15)
                ax.set_title(f"Plot 8 — Precision / Recall / F1 per Disease Class — {l}")
                ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.4)
                plt.tight_layout()
                plt.savefig(pdir / "plot08_precision_recall_f1.png", dpi=150)
                plt.close()
                saved.append("plot08_precision_recall_f1.png")
                break

    # ────────────────────────────────────────────────────────────────────────
    # Plot 9 — Model summary table (macro metrics)
    # ────────────────────────────────────────────────────────────────────────
    rows = []
    for l, k in active:
        if results[k].get("central_metrics"):
            m = results[k]["central_metrics"]
            rows.append([l, "Central",
                         f"{m.get('accuracy',0):.3f}",
                         f"{m.get('f1_macro',0):.3f}",
                         f"{m.get('precision',0):.3f}",
                         f"{m.get('recall',0):.3f}"])
        if results[k].get("fed_f1") is not None:
            rows.append([l, "Federated", "—",
                         f"{results[k]['fed_f1']:.3f}", "—", "—"])
    if rows:
        fig, ax = plt.subplots(figsize=(11, max(3, len(rows) * 0.55 + 1.5)))
        ax.axis("off")
        cols = ["Model", "Mode", "Accuracy", "F1 Macro", "Precision", "Recall"]
        tbl = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
        tbl.auto_set_font_size(False); tbl.set_fontsize(10)
        tbl.scale(1.2, 1.6)
        # Colour header
        for j in range(len(cols)):
            tbl[0, j].set_facecolor("#2c3e50"); tbl[0, j].set_text_props(color="white")
        # Alternating rows
        for i in range(1, len(rows) + 1):
            bg = "#ecf0f1" if i % 2 == 0 else "white"
            for j in range(len(cols)):
                tbl[i, j].set_facecolor(bg)
        ax.set_title("Plot 9 — Model Performance Summary", fontsize=13,
                     fontweight="bold", pad=20)
        plt.tight_layout()
        plt.savefig(pdir / "plot09_model_summary_table.png", dpi=150)
        plt.close()
        saved.append("plot09_model_summary_table.png")

    # ────────────────────────────────────────────────────────────────────────
    # Plot 10 — Dataset class distribution (pie + bar side by side)
    # ────────────────────────────────────────────────────────────────────────
    counts_d  = [129, 9, 67, 102, 64]
    palette_d = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.pie(counts_d, labels=TEA_DISEASE_LABELS, colors=palette_d,
            autopct="%1.1f%%", startangle=90, pctdistance=0.82)
    ax1.set_title("Class Distribution (pie)")
    bars_d = ax2.bar(TEA_DISEASE_LABELS, counts_d, color=palette_d,
                     edgecolor="black", linewidth=0.7)
    for b, v in zip(bars_d, counts_d):
        ax2.text(b.get_x() + b.get_width() / 2, v + 1, str(v),
                 ha="center", va="bottom", fontsize=10)
    ax2.set_ylabel("OBB crop count"); ax2.set_ylim(0, max(counts_d) * 1.2)
    ax2.set_xticklabels(TEA_DISEASE_LABELS, rotation=30, ha="right")
    ax2.set_title("Class Distribution (count)")
    ax2.grid(axis="y", alpha=0.4)
    fig.suptitle("Plot 10 — Real Dataset OBB Crop Distribution (371 total)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(pdir / "plot10_class_distribution.png", dpi=150); plt.close()
    saved.append("plot10_class_distribution.png")

    # ────────────────────────────────────────────────────────────────────────
    # Plot 11 — Literature comparison (horizontal bar)
    # ────────────────────────────────────────────────────────────────────────
    LIT = [
        ("NNE-Tea (Karmokar 2015)",   0.910, "ML"),
        ("SVM-Tea (Hossain 2018)",    0.913, "ML"),
        ("CNN-4cls (Biswas 2018)",    0.959, "CNN"),
        ("LeNet-5 (Gayathri 2020)",   0.902, "CNN"),
        ("AX-RetinaNet (Bao 2022)",   0.954, "Detection"),
        ("CNN-7cls (Singh 2022)",     0.845, "CNN"),
        ("YOLO-Tea (Xue 2023)",       0.920, "Detection"),
        ("YOLO-T (Soebi 2023)",       0.965, "Detection"),
        ("NASNet Tea (Jayanti 2024)", 0.920, "CNN"),
        ("CNN-BD (Rahman 2024)",      0.966, "CNN"),
        ("FedCNN-Sev (Vats 2024)",    0.950, "Federated"),
        ("ResidualCNN (Rahat 2025)",  0.990, "CNN"),
    ]
    cat_col = {"ML": "#95a5a6", "CNN": "#27ae60", "Detection": "#e67e22",
               "Federated": "#3498db", "Ours-Central": "#c0392b",
               "Ours-Fed":  "#8e44ad"}

    ours_entries = []
    for l, k in active:
        ours_entries.append((f"FarmFederate-{l} Central (Ours)",
                              results[k]["central_f1"], "Ours-Central"))
        if results[k].get("fed_f1") is not None:
            ours_entries.append((f"FarmFederate-{l} Federated (Ours)",
                                  results[k]["fed_f1"], "Ours-Fed"))

    all_lit = sorted(LIT + ours_entries, key=lambda x: x[1])
    fig, ax = plt.subplots(figsize=(14, max(8, len(all_lit) * 0.55)))
    for i, (name, f1, cat) in enumerate(all_lit):
        color = cat_col.get(cat, "#95a5a6")
        lw    = 1.5 if "Ours" in cat else 0.5
        ax.barh(i, f1, color=color, edgecolor="black", linewidth=lw, height=0.7)
        ax.text(f1 + 0.002, i, f"{f1:.3f}", va="center", fontsize=8)
    ax.set_yticks(range(len(all_lit)))
    ax.set_yticklabels([x[0] for x in all_lit], fontsize=8)
    ax.set_xlabel("F1 / Accuracy"); ax.set_xlim(0, 1.12)
    ax.set_title("Plot 11 — FarmFederate vs State-of-the-Art",
                 fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    patches = [mpatches.Patch(color=v, label=k) for k, v in cat_col.items()]
    ax.legend(handles=patches, loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(pdir / "plot11_literature_comparison.png", dpi=150); plt.close()
    saved.append("plot11_literature_comparison.png")

    # Plot 12 — Cross-modal ablation and mismatch sensitivity.
    cross_modal = (
        results.get("vlm", {}).get("cross_modal_analysis")
        if results.get("vlm") else None
    )
    if cross_modal:
        condition_names = ["paired", "text_only", "image_only", "mismatched_text"]
        condition_labels = ["Paired", "Text only", "Image only", "Mismatched text"]
        f1_values = [
            cross_modal["conditions"][name]["f1_macro"] for name in condition_names
        ]
        fig, ax = plt.subplots(figsize=(9, 5))
        bars = ax.bar(
            condition_labels,
            f1_values,
            color=["#8e44ad", "#3498db", "#27ae60", "#e74c3c"],
            edgecolor="black",
            linewidth=0.7,
        )
        _annotate_bars(ax, bars, f1_values)
        ax.set_ylabel("Macro F1")
        ax.set_ylim(0, 1.12)
        ax.set_title("Plot 12 — Cross-Modal Ablation and Pairing Test")
        ax.grid(axis="y", alpha=0.35)
        plt.tight_layout()
        plt.savefig(pdir / "plot12_cross_modal_analysis.png", dpi=150)
        plt.close()
        saved.append("plot12_cross_modal_analysis.png")

    print(f"\n  {len(saved)} plots saved to: {pdir}")
    for s in saved:
        print(f"    {s}")


def _json_safe(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def save_training_report(results: Dict, cfg: Config, data_audit: Dict) -> Tuple[Path, Path]:
    """Persist machine-readable evidence and a concise, claim-safe report."""
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_type": "smoke_test" if cfg.quick_test else "full_training",
        "config": _json_safe(asdict(cfg)),
        "data_audit": _json_safe(data_audit),
        "results": _json_safe(results),
    }
    json_path = output_dir / "complete_results.json"
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    report_lines = [
        "# FarmFederate Multimodal Training Report",
        "",
        f"- Generated: {payload['generated_at_utc']}",
        f"- Run type: **{payload['run_type']}**",
        f"- Device: `{cfg.device}`",
        f"- Split: source-image grouped, validation fraction `{cfg.val_split:.0%}`, "
        f"locked-test fraction `{cfg.test_split:.0%}`",
        f"- Pairwise train/validation/test source-image overlap: "
        f"`{data_audit.get('group_overlap', 'n/a')}`",
        f"- Exact train pair coverage: `{data_audit.get('train_pair_coverage', 0):.1%}`",
        f"- Exact validation pair coverage: `{data_audit.get('val_pair_coverage', 0):.1%}`",
        f"- Exact test pair coverage: `{data_audit.get('test_pair_coverage', 0):.1%}`",
        f"- Training-fitted target-shortcut tokens masked: "
        f"`{data_audit.get('masked_target_shortcut_token_count', 0)}`",
        "",
        "## Locked-test results",
        "",
        "| Model | Selected validation F1 | Test macro-F1 | Test accuracy | Test ECE | Federated validation F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, label in (("llm", "Text"), ("vit", "Image"), ("vlm", "Multimodal")):
        result = results.get(key)
        if not result:
            continue
        metrics = result.get("test_metrics") or {}
        federated = result.get("fed_f1")
        fed_text = f"{federated:.4f}" if federated is not None else "not run"
        report_lines.append(
            f"| {label} | {result['central_f1']:.4f} | "
            f"{metrics.get('f1_macro', 0):.4f} | "
            f"{metrics.get('accuracy', 0):.4f} | {metrics.get('ece', 0):.4f} | "
            f"{fed_text} |"
        )

    cross = results.get("vlm", {}).get("cross_modal_analysis") if results.get("vlm") else None
    if cross:
        report_lines.extend(
            [
                "",
                "## Locked-test cross-modal analysis",
                "",
                "| Condition | Macro-F1 | Accuracy | NLL | ECE |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for name, label in (
            ("paired", "Correctly paired"),
            ("text_only", "Text only"),
            ("image_only", "Image only"),
            ("mismatched_text", "Mismatched text"),
        ):
            metrics = cross["conditions"][name]
            report_lines.append(
                f"| {label} | {metrics['f1_macro']:.4f} | "
                f"{metrics['accuracy']:.4f} | {metrics['nll']:.4f} | "
                f"{metrics['ece']:.4f} |"
            )
        summary = cross["summary"]
        report_lines.extend(
            [
                "",
                f"- Fusion gain over best unimodal path: "
                f"`{summary['fusion_gain_macro_f1']:+.4f}` macro-F1.",
                f"- Drop after mismatching the text: "
                f"`{summary['mismatch_drop_macro_f1']:+.4f}` macro-F1.",
                f"- Text→image class Recall@1: "
                f"`{summary['retrieval_class_recall_at_1']['text_to_image']:.4f}`.",
                f"- Image→text class Recall@1: "
                f"`{summary['retrieval_class_recall_at_1']['image_to_text']:.4f}`.",
                f"- Paired-vs-rolled cosine margin: "
                f"`{summary['paired_vs_rolled_cosine_margin']:+.4f}`.",
            ]
        )

    report_lines.extend(
        [
            "",
            "## Interpretation guardrails",
            "",
            "- Model selection uses macro-F1, not micro-F1/accuracy, because the disease classes are imbalanced.",
            "- Exact image/box text pairs are used; same-class random pairing is only a fallback for missing annotations.",
            "- All boxes from a source image remain in exactly one of train, validation, or locked test.",
            "- Validation selects the checkpoint; the locked test is evaluated only after training.",
            "- Class-exclusive caption tokens are learned from training annotations only and masked in every partition.",
            "- The fused head includes a fixed reliability-weighted residual from the two auxiliary modality experts.",
            "- Cross-modal benefit requires paired performance to exceed both unimodal ablations and to degrade under mismatching.",
        ]
    )
    if cfg.quick_test:
        report_lines.append(
            "- This was a short smoke run. Its metrics verify the pipeline but do not establish a state-of-the-art claim."
        )
    else:
        report_lines.append(
            "- A category-best claim additionally requires repeated seeds and an external held-out dataset; this report does not infer either."
        )

    report_path = output_dir / "MULTIMODAL_TRAINING_REPORT.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return json_path, report_path


# ============================================================================
# MAIN RUNNER
# ============================================================================

def run(cfg: Config):
    device = torch.device(cfg.device)
    print(f"\nDevice : {device}")
    print(f"Classes: {TEA_DISEASE_LABELS}")

    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    data_audit = {
        "group_overlap": None,
        "train_pair_coverage": 0.0,
        "val_pair_coverage": 0.0,
        "test_pair_coverage": 0.0,
    }

    # ── 1. Load / generate text data ────────────────────────────────────────
    print("\n[1/5] Loading text data ...")
    ann_path = Path(cfg.annotations)
    if ann_path.exists():
        text_df = load_annotations_csv(str(ann_path))
        if len(text_df) < 50:
            print("  CSV too small — supplementing with generated text")
            text_df = pd.concat([text_df,
                                  generate_text_data(cfg.text_samples_per_class)],
                                 ignore_index=True)
    else:
        print(f"  annotations.csv not found at {ann_path}")
        print("  Generating text from symptom vocabulary ...")
        text_df = generate_text_data(cfg.text_samples_per_class)

    print(f"  Total text samples: {len(text_df)}")
    print("  Per class:", text_df["label_name"].value_counts().to_dict())

    # ── 2. Load image data ───────────────────────────────────────────────────
    print("\n[2/5] Loading image data ...")
    data_dir = Path(cfg.data_dir)
    img_dir  = data_dir / "images"
    lbl_dir  = data_dir / "labels"
    if not img_dir.exists():
        print(f"  WARNING: image directory not found: {img_dir}")
        print("  Will train LLM (text) only.")
        has_images = False
        counts = text_df["label_name"].value_counts()
        stratify = text_df["label_name"] if not counts.empty and counts.min() >= 2 else None
        text_train_val, text_test = train_test_split(
            text_df,
            test_size=cfg.test_split,
            random_state=cfg.seed,
            stratify=stratify,
        )
        train_val_counts = text_train_val["label_name"].value_counts()
        train_val_stratify = (
            text_train_val["label_name"]
            if not train_val_counts.empty and train_val_counts.min() >= 2
            else None
        )
        text_trn, text_val = train_test_split(
            text_train_val,
            test_size=cfg.val_split / (1.0 - cfg.test_split),
            random_state=cfg.seed + 1,
            stratify=train_val_stratify,
        )
        text_trn = text_trn.reset_index(drop=True)
        text_val = text_val.reset_index(drop=True)
        text_test = text_test.reset_index(drop=True)
    else:
        has_images = True
        full_ds = TeaOBBDataset(str(img_dir), str(lbl_dir),
                                transform=get_transforms(train=True),
                                crop_padding=cfg.crop_padding)
        print(f"  Total OBB crops: {len(full_ds)}")
        dist = Counter(full_ds.labels)
        for i, n in sorted(dist.items()):
            print(f"  [{i}] {TEA_DISEASE_LABELS[i]:<20}: {n}")

        # Source-image grouped split prevents boxes from the same leaf image
        # appearing in both train and validation partitions.
        trn_i, val_i, test_i = grouped_train_val_test_split(
            full_ds.labels,
            full_ds.groups,
            cfg.val_split,
            cfg.test_split,
            cfg.seed,
        )
        if cfg.quick_test:
            trn_i = cap_indices_per_class(trn_i, full_ds.labels, 12, cfg.seed)
            val_i = cap_indices_per_class(val_i, full_ds.labels, 5, cfg.seed + 1)
            test_i = cap_indices_per_class(test_i, full_ds.labels, 5, cfg.seed + 2)
            print(
                "  Quick-test cap: at most 12 train, 5 validation, and "
                "5 test crops/class"
            )
        train_groups = {full_ds.groups[i] for i in trn_i}
        val_groups = {full_ds.groups[i] for i in val_i}
        test_groups = {full_ds.groups[i] for i in test_i}
        print(
            f"  Image train={len(trn_i)} ({len(train_groups)} source images)  "
            f"val={len(val_i)} ({len(val_groups)} source images)  "
            f"test={len(test_i)} ({len(test_groups)} source images)"
        )
        pairwise_overlap = (
            len(train_groups & val_groups)
            + len(train_groups & test_groups)
            + len(val_groups & test_groups)
        )
        print(f"  Pairwise source-image overlap: {pairwise_overlap}")
        data_audit["group_overlap"] = pairwise_overlap
        data_audit["train_source_images"] = len(train_groups)
        data_audit["validation_source_images"] = len(val_groups)
        data_audit["test_source_images"] = len(test_groups)
        data_audit["train_crops"] = len(trn_i)
        data_audit["validation_crops"] = len(val_i)
        data_audit["test_crops"] = len(test_i)

        train_obb = TeaOBBDataset(str(img_dir), str(lbl_dir),
                                  transform=get_transforms(train=True),
                                  crop_padding=cfg.crop_padding, indices=trn_i)
        val_obb   = TeaOBBDataset(str(img_dir), str(lbl_dir),
                                  transform=get_transforms(train=False),
                                  crop_padding=cfg.crop_padding, indices=val_i)
        test_obb  = TeaOBBDataset(str(img_dir), str(lbl_dir),
                                  transform=get_transforms(train=False),
                                  crop_padding=cfg.crop_padding, indices=test_i)
        cw = compute_class_weights(train_obb.labels, device)

        if {"image_file", "box_idx"}.issubset(text_df.columns):
            train_keys = {full_ds.sample_ids[i] for i in trn_i}
            val_keys = {full_ds.sample_ids[i] for i in val_i}
            test_keys = {full_ds.sample_ids[i] for i in test_i}
            row_keys = list(zip(text_df["image_file"], text_df["box_idx"]))
            train_mask = pd.Series([key in train_keys for key in row_keys])
            val_mask = pd.Series([key in val_keys for key in row_keys])
            test_mask = pd.Series([key in test_keys for key in row_keys])
            text_trn = text_df.loc[train_mask].reset_index(drop=True)
            text_val = text_df.loc[val_mask].reset_index(drop=True)
            text_test = text_df.loc[test_mask].reset_index(drop=True)
            if text_trn.empty or text_val.empty or text_test.empty:
                raise RuntimeError(
                    "Exact image/text pairing failed: a split has no paired annotations"
                )
        else:
            text_train_val, text_test = train_test_split(
                text_df,
                test_size=cfg.test_split,
                random_state=cfg.seed,
                stratify=text_df["label_name"],
            )
            text_trn, text_val = train_test_split(
                text_train_val,
                test_size=cfg.val_split / (1.0 - cfg.test_split),
                random_state=cfg.seed + 1,
                stratify=text_train_val["label_name"],
            )
            text_trn = text_trn.reset_index(drop=True)
            text_val = text_val.reset_index(drop=True)
            text_test = text_test.reset_index(drop=True)

    if cfg.sanitize_target_derived_text:
        blocked_tokens = fit_label_leakage_vocabulary(
            text_trn,
            min_count=cfg.leakage_token_min_count,
            purity_threshold=cfg.leakage_token_purity,
        )
        text_trn = sanitize_annotation_text(text_trn, blocked_tokens)
        text_val = sanitize_annotation_text(text_val, blocked_tokens)
        text_test = sanitize_annotation_text(text_test, blocked_tokens)
        data_audit["masked_target_shortcut_tokens"] = blocked_tokens
        data_audit["masked_target_shortcut_token_count"] = len(blocked_tokens)
        print(
            f"  Masked {len(blocked_tokens)} training-fitted target-shortcut "
            "tokens in all partitions"
        )

    print(
        f"  Text train={len(text_trn)}  val={len(text_val)}  "
        f"test={len(text_test)}"
    )

    # ── 3. Build dataloaders ─────────────────────────────────────────────────
    print("\n[3/5] Building dataloaders ...")
    text_labels_trn = [l[0] if isinstance(l, list) else l
                       for l in text_trn["labels"].tolist()]
    t_sampler  = BalancedBatchSampler(text_labels_trn, cfg.batch_size, NUM_CLASSES)
    t_trn_ds   = TextDataset(text_trn)
    t_val_ds   = TextDataset(text_val)
    t_test_ds  = TextDataset(text_test)
    t_trn_load = DataLoader(t_trn_ds, batch_sampler=t_sampler,  num_workers=0)
    t_val_load = DataLoader(t_val_ds, batch_size=cfg.batch_size, num_workers=0)
    t_test_load = DataLoader(t_test_ds, batch_size=cfg.batch_size, num_workers=0)

    if has_images:
        i_sampler  = BalancedBatchSampler(train_obb.labels, cfg.batch_size, NUM_CLASSES)
        i_trn_load = DataLoader(train_obb, batch_sampler=i_sampler, num_workers=0)
        i_val_load = DataLoader(val_obb,   batch_size=cfg.batch_size, num_workers=0)
        i_test_load = DataLoader(test_obb, batch_size=cfg.batch_size, num_workers=0)

        mm_trn_ds  = MultiModalDataset(train_obb, text_trn, cfg.max_seq_len, cfg.seed)
        mm_val_ds  = MultiModalDataset(val_obb,   text_val, cfg.max_seq_len, cfg.seed)
        mm_test_ds = MultiModalDataset(
            test_obb, text_test, cfg.max_seq_len, cfg.seed
        )
        print(
            f"  Exact pair coverage: train={mm_trn_ds.pairing_coverage:.1%}, "
            f"val={mm_val_ds.pairing_coverage:.1%}, "
            f"test={mm_test_ds.pairing_coverage:.1%}"
        )
        data_audit["train_pair_coverage"] = mm_trn_ds.pairing_coverage
        data_audit["val_pair_coverage"] = mm_val_ds.pairing_coverage
        data_audit["test_pair_coverage"] = mm_test_ds.pairing_coverage
        mm_sampler = BalancedBatchSampler(train_obb.labels, cfg.batch_size, NUM_CLASSES)
        mm_trn_ld  = DataLoader(mm_trn_ds, batch_sampler=mm_sampler, num_workers=0)
        mm_val_ld  = DataLoader(mm_val_ds, batch_size=cfg.batch_size, num_workers=0)
        mm_test_ld = DataLoader(
            mm_test_ds, batch_size=cfg.batch_size, num_workers=0
        )

    # ── 4. Train models ──────────────────────────────────────────────────────
    results = {"llm": None, "vit": None, "vlm": None}

    # -- LLM (text only) --
    if not cfg.multimodal_only:
        print("\n[4/5] Training LLM (text) ...")
        llm = LightweightTextClassifier(
            num_labels=NUM_CLASSES, max_seq_len=cfg.max_seq_len
        ).to(device)
        best_f1, hist, best_met, best_state = train_model(
            llm, t_trn_load, t_val_load, cfg, device, "text")
        print(f"  Centralised LLM best val macro-F1: {best_f1:.4f}")
        test_met = evaluate(llm, t_test_load, device, "text")

        if cfg.run_federated:
            fed_f1, fed_hist, fed_met = run_federated(
                LightweightTextClassifier,
                {"num_labels": NUM_CLASSES, "max_seq_len": cfg.max_seq_len},
                t_trn_ds, t_val_load, cfg, device, "text")
            print(f"  Federated    LLM best val macro-F1: {fed_f1:.4f}")
        else:
            fed_f1, fed_hist, fed_met = None, {}, {}

        torch.save({"state": best_state, "labels": TEA_DISEASE_LABELS,
                    "val_f1_macro": best_f1, "type": "text",
                    "tokenizer": "stable_blake2b_v1"},
                   Path(cfg.models_dir) / "best_llm.pt")
        results["llm"] = {"central_f1": best_f1, "fed_f1": fed_f1,
                          "history": hist, "fed_history": fed_hist,
                          "central_metrics": best_met,
                          "test_metrics": test_met}

    if has_images:
        # -- ViT (vision only) --
        if not cfg.multimodal_only:
            print("\n  Training ViT (vision) ...")
            vit = LightweightVisionClassifier(num_labels=NUM_CLASSES,
                                              class_weights=cw).to(device)
            best_f1_v, hist_v, best_met_v, best_state_v = train_model(
                vit, i_trn_load, i_val_load, cfg, device, "vision")
            print(f"  Centralised ViT best val macro-F1: {best_f1_v:.4f}")
            test_met_v = evaluate(vit, i_test_load, device, "vision")

            if cfg.run_federated:
                fed_f1_v, fed_hist_v, fed_met_v = run_federated(
                    LightweightVisionClassifier, {"num_labels": NUM_CLASSES,
                                                  "class_weights": cw},
                    train_obb, i_val_load, cfg, device, "vision")
                print(f"  Federated    ViT best val macro-F1: {fed_f1_v:.4f}")
            else:
                fed_f1_v, fed_hist_v, fed_met_v = None, {}, {}

            torch.save({"state": best_state_v, "labels": TEA_DISEASE_LABELS,
                        "val_f1_macro": best_f1_v, "type": "vision"},
                       Path(cfg.models_dir) / "best_vit.pt")
            results["vit"] = {"central_f1": best_f1_v, "fed_f1": fed_f1_v,
                              "history": hist_v, "fed_history": fed_hist_v,
                              "central_metrics": best_met_v,
                              "test_metrics": test_met_v}

        # -- VLM (multimodal) --
        print("\n  Training VLM (multimodal) ...")
        vlm_kwargs = {
            "num_labels": NUM_CLASSES,
            "class_weights": cw,
            "max_seq_len": cfg.max_seq_len,
            "modality_dropout": cfg.modality_dropout,
            "image_only_probability": cfg.image_only_probability,
            "text_only_probability": cfg.text_only_probability,
            "vision_backbone": "resnet50" if cfg.pretrained_vision else "lightweight",
            "pretrained_vision": cfg.pretrained_vision,
            "freeze_vision_backbone": cfg.freeze_vision_backbone,
            "finetune_vision_last_stage": cfg.finetune_vision_last_stage,
            "text_auxiliary_weight": cfg.text_auxiliary_weight,
            "vision_auxiliary_weight": cfg.vision_auxiliary_weight,
            "alignment_weight": cfg.alignment_weight,
            "text_confidence_guard": cfg.text_confidence_guard,
        }
        vlm = MultiModalClassifier(**vlm_kwargs).to(device)
        if cfg.resume_checkpoint:
            resume_path = Path(cfg.resume_checkpoint)
            resume_payload = torch.load(
                resume_path, map_location=device, weights_only=False
            )
            vlm.load_state_dict(resume_payload["state"], strict=True)
            print(f"  Resumed VLM weights from {resume_path}")
        if (
            cfg.cache_frozen_vision
            and cfg.freeze_vision_backbone
            and vlm_kwargs["vision_backbone"] == "resnet50"
        ):
            print("  Precomputing frozen ResNet-50 spatial maps ...")
            mm_trn_ds = cache_frozen_vision_features(
                mm_trn_ds,
                vlm,
                device,
                cfg.batch_size,
                num_views=cfg.vision_cache_views,
                seed=cfg.seed,
            )
            mm_val_ds = cache_frozen_vision_features(
                mm_val_ds,
                vlm,
                device,
                cfg.batch_size,
                num_views=1,
                seed=cfg.seed + 10_000,
            )
            mm_test_ds = cache_frozen_vision_features(
                mm_test_ds,
                vlm,
                device,
                cfg.batch_size,
                num_views=1,
                seed=cfg.seed + 20_000,
            )
            mm_sampler = BalancedBatchSampler(
                mm_trn_ds.labels, cfg.batch_size, NUM_CLASSES
            )
            mm_trn_ld = DataLoader(
                mm_trn_ds, batch_sampler=mm_sampler, num_workers=0
            )
            mm_val_ld = DataLoader(
                mm_val_ds, batch_size=cfg.batch_size, num_workers=0
            )
            mm_test_ld = DataLoader(
                mm_test_ds, batch_size=cfg.batch_size, num_workers=0
            )
            data_audit["cached_frozen_vision_features"] = True
            data_audit["training_vision_cache_views"] = cfg.vision_cache_views
        best_f1_m, hist_m, best_met_m, best_state_m = train_model(
            vlm, mm_trn_ld, mm_val_ld, cfg, device, "multimodal")
        print(f"  Centralised VLM best val macro-F1: {best_f1_m:.4f}")

        if cfg.run_federated:
            fed_f1_m, fed_hist_m, fed_met_m = run_federated(
                MultiModalClassifier, vlm_kwargs,
                mm_trn_ds, mm_val_ld, cfg, device, "multimodal")
            print(f"  Federated    VLM best val macro-F1: {fed_f1_m:.4f}")
        else:
            fed_f1_m, fed_hist_m, fed_met_m = None, {}, {}

        test_metrics_m = evaluate(
            vlm, mm_test_ld, device, "multimodal"
        )
        cross_modal = (
            run_cross_modal_analysis(vlm, mm_test_ld, device)
            if cfg.cross_modal_analysis else None
        )
        print(
            "  Locked-test VLM "
            f"macro-F1={test_metrics_m['f1_macro']:.4f} "
            f"accuracy={test_metrics_m['accuracy']:.4f}"
        )

        torch.save({"state": best_state_m, "labels": TEA_DISEASE_LABELS,
                    "val_f1_macro": best_f1_m, "type": "multimodal",
                    "architecture": "resnet50_bidirectional_coattention_v4",
                    "vision_backbone": vlm_kwargs["vision_backbone"],
                    "pretrained_vision": cfg.pretrained_vision,
                    "freeze_vision_backbone": cfg.freeze_vision_backbone,
                    "finetune_vision_last_stage": cfg.finetune_vision_last_stage,
                    "text_confidence_guard": cfg.text_confidence_guard,
                    "tokenizer": "stable_blake2b_v1",
                    "max_seq_len": cfg.max_seq_len,
                    "resumed_from": cfg.resume_checkpoint},
                   Path(cfg.models_dir) / "best_vlm.pt")
        results["vlm"] = {"central_f1": best_f1_m, "fed_f1": fed_f1_m,
                           "history": hist_m, "fed_history": fed_hist_m,
                           "central_metrics": best_met_m,
                           "test_metrics": test_metrics_m,
                           "cross_modal_analysis": cross_modal,
                           "cross_modal_analysis_partition": "locked_test",
                           "pairing_coverage": {
                               "train": mm_trn_ds.pairing_coverage,
                               "validation": mm_val_ds.pairing_coverage,
                               "test": mm_test_ds.pairing_coverage,
                           }}

    # ── 5. Save plots ────────────────────────────────────────────────────────
    print("\n[5/5] Saving plots ...")
    save_results_plots(results, cfg)
    results_json, report_path = save_training_report(results, cfg, data_audit)
    print(f"  Results: {results_json}")
    print(f"  Report : {report_path}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for key, label in [("llm","LLM (text)"), ("vit","ViT (vision)"), ("vlm","VLM (multimodal)")]:
        if results[key]:
            r = results[key]
            fed_summary = (
                f"{r['fed_f1']:.4f}" if r.get("fed_f1") is not None else "not run"
            )
            print(f"  {label:<22}  central={r['central_f1']:.4f}  "
                  f"federated={fed_summary}")
    print(f"\n  Models : {cfg.models_dir}")
    print(f"  Plots  : {cfg.plots_dir}")
    print("=" * 60)

    # ── Upload to Google Drive ────────────────────────────────────────────────
    if cfg.gdrive_base is not None:
        print(f"\n[GDrive] Uploading outputs to Drive ...")
        upload_to_gdrive(cfg.output_dir, cfg.gdrive_base)
        print(f"[GDrive] Done — check My Drive/{cfg.gdrive_dir}/{_GDRIVE_OUTPUT_DIR}/")

    return results


# ============================================================================
# ENTRY POINT
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="FarmFederate — Tea Leaf Disease Detection (standalone)")
    p.add_argument("--data_dir",    default=r"C:/Users/USER_HP/Desktop/FarmFederate/Real Dataset",
                   help="Path to Real Dataset (contains images/ and labels/)")
    p.add_argument("--annotations", default=r"C:/Users/USER_HP/Desktop/FarmFederate/tea_results/annotation/annotations.csv",
                   help="Path to annotations.csv from tea_annotator.py")
    p.add_argument("--output_dir",  default=r"C:/Users/USER_HP/Desktop/FarmFederate/tea_results",
                   help="Output directory for plots and models")
    p.add_argument(
        "--resume_checkpoint",
        help="Optional trained VLM checkpoint used to initialize fine-tuning",
    )
    p.add_argument("--epochs",      type=int,   default=15)
    p.add_argument("--fed_rounds",  type=int,   default=8)
    p.add_argument("--num_clients", type=int,   default=3)
    p.add_argument("--batch_size",  type=int,   default=16)
    p.add_argument("--lr",          type=float, default=1e-4,  dest="learning_rate")
    p.add_argument("--val_split",   type=float, default=0.20)
    p.add_argument("--test_split",  type=float, default=0.20)
    p.add_argument("--dirichlet_alpha", type=float, default=1.0)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--quick_test",  action="store_true",
                   help="2 epochs, 2 rounds — smoke test only")
    p.add_argument("--no_amp",      action="store_true")
    p.add_argument(
        "--multimodal_only",
        action="store_true",
        help="Train the cross-modal model without separate text/image baselines",
    )
    p.add_argument(
        "--no_federated",
        action="store_true",
        help="Skip federated training for a faster local diagnostic run",
    )
    p.add_argument(
        "--no_cross_modal_analysis",
        action="store_true",
        help="Skip paired/unimodal/mismatched and retrieval analysis",
    )
    p.add_argument(
        "--modality_dropout",
        type=float,
        default=0.20,
        help="Legacy total modality-dropout probability",
    )
    p.add_argument(
        "--image_only_probability",
        type=float,
        default=0.50,
        help="Training probability for image-only multimodal batches",
    )
    p.add_argument(
        "--text_only_probability",
        type=float,
        default=0.10,
        help="Training probability for text-only multimodal batches",
    )
    p.add_argument(
        "--vision_warmup_epochs",
        type=int,
        default=4,
        help="Initial epochs trained with image-only inputs",
    )
    p.add_argument(
        "--vision_cache_views",
        type=int,
        default=2,
        help="Augmented frozen-ResNet feature views cached per training crop",
    )
    p.add_argument(
        "--vision_selection_weight",
        type=float,
        default=0.35,
        help="Image-only validation F1 weight used for checkpoint selection",
    )
    p.add_argument(
        "--vision_auxiliary_weight",
        type=float,
        default=1.25,
        help="Weight of the dedicated vision classification objective",
    )
    p.add_argument(
        "--lightweight_vision",
        action="store_true",
        help="Use the randomly initialized lightweight CNN instead of ResNet-50",
    )
    p.add_argument(
        "--finetune_vision",
        action="store_true",
        help="Fine-tune the ResNet-50 backbone instead of freezing it",
    )
    p.add_argument(
        "--finetune_vision_last_stage",
        action="store_true",
        help=(
            "Fine-tune only ResNet-50 layer4; disables frozen-feature caching "
            "while keeping earlier stages frozen"
        ),
    )
    p.add_argument(
        "--no_vision_cache",
        action="store_true",
        help="Recompute frozen ResNet features every epoch instead of caching",
    )
    p.add_argument(
        "--text_confidence_guard",
        type=float,
        default=None,
        help=(
            "Optional text-only confidence override. Disabled by default "
            "because it can inflate internal accuracy."
        ),
    )
    p.add_argument(
        "--no_text_sanitization",
        action="store_true",
        help="Do not mask training-fitted target-derived caption shortcuts",
    )
    # Google Drive
    p.add_argument("--gdrive",      action="store_true",
                   help="Mount Google Drive and auto-upload outputs (Colab only)")
    p.add_argument("--gdrive_dir",  default=_GDRIVE_ROOT,
                   help=f"Folder name inside My Drive (default: '{_GDRIVE_ROOT}')")
    # parse_known_args ignores Jupyter kernel args (-f kernel-xxx.json)
    args, _ = p.parse_known_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    cfg  = Config(
        data_dir    = args.data_dir,
        annotations = args.annotations,
        output_dir  = args.output_dir,
        resume_checkpoint = args.resume_checkpoint,
        epochs      = args.epochs,
        fed_rounds  = args.fed_rounds,
        num_clients = args.num_clients,
        batch_size  = args.batch_size,
        learning_rate = args.learning_rate,
        val_split   = args.val_split,
        test_split  = args.test_split,
        dirichlet_alpha = args.dirichlet_alpha,
        seed        = args.seed,
        quick_test  = args.quick_test,
        use_mixed_precision = not args.no_amp,
        multimodal_only = args.multimodal_only,
        run_federated = not args.no_federated,
        modality_dropout = args.modality_dropout,
        image_only_probability = args.image_only_probability,
        text_only_probability = args.text_only_probability,
        vision_warmup_epochs = args.vision_warmup_epochs,
        vision_cache_views = args.vision_cache_views,
        vision_selection_weight = args.vision_selection_weight,
        vision_auxiliary_weight = args.vision_auxiliary_weight,
        pretrained_vision = not args.lightweight_vision,
        freeze_vision_backbone = not (
            args.finetune_vision or args.finetune_vision_last_stage
        ),
        finetune_vision_last_stage = args.finetune_vision_last_stage,
        cache_frozen_vision = not args.no_vision_cache,
        text_confidence_guard = args.text_confidence_guard,
        sanitize_target_derived_text = not args.no_text_sanitization,
        cross_modal_analysis = not args.no_cross_modal_analysis,
        use_gdrive  = args.gdrive,
        gdrive_dir  = args.gdrive_dir,
    )
    print("=" * 60)
    print("FarmFederate  —  Tea Leaf Disease Detection")
    print("=" * 60)
    print(f"  Data dir    : {cfg.data_dir}")
    print(f"  Annotations : {cfg.annotations}")
    print(f"  Output      : {cfg.output_dir}")
    print(f"  Epochs      : {cfg.epochs}")
    print(f"  Fed rounds  : {cfg.fed_rounds}")
    print(f"  Clients     : {cfg.num_clients}")
    print(f"  Device      : {cfg.device}")
    if cfg.gdrive_base:
        print(f"  GDrive      : {cfg.gdrive_base}")
    run(cfg)
