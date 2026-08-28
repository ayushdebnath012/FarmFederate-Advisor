#!/usr/bin/env python3
"""
demo_server.py - Lightweight FastAPI backend for Render free tier.
No PyTorch/transformers imports. Uses keyword-based crop stress predictions.
For full ML inference, use server.py with local GPU/CPU.
"""

import os
import io
import base64
import random
from pathlib import Path
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Optional ML imports for disease detection.
# Loaded lazily so the server still starts on Render (no torch).
_DISEASE_READY = False
_disease_classifier = None
_torch = None
_F = None
_nn = None
try:
    import cv2
    import numpy as np
    import torch as _torch
    import torch.nn as _nn
    import torch.nn.functional as _F
    import torchvision.transforms as _tv_T
    from PIL import Image as _PILImage
    _DISEASE_READY = True
except ImportError:
    pass

# Paths for trained FarmFederate models and reference data.
_RESULTS_DIR    = Path(r"C:\Users\USER_HP\Desktop\FarmFederate\farmfederate_results (3)")
_CHECKPOINTS_DIR = _RESULTS_DIR / "drive" / "checkpoints"
_TEA_DATA_DIR   = _RESULTS_DIR / "dataset" / "tea_data"
_ANNOTATED_DIR  = Path(r"C:\Users\USER_HP\Desktop\FarmFederate\disease_annotated-20260423T120409Z-3-001\disease_annotated")
# Fallback to old Merged-Dataset if results not present
_MERGED_DIR     = Path(r"C:\Users\USER_HP\Desktop\FarmFederate\Merged-Dataset")

_DISEASE_CLASSES  = ["LEAF_BLIGHT", "LEAF_HOPPERS", "LEAF_RUST",
                     "LOOPER_CATERPILLARS", "MOSQUITO_BUG"]
_DISEASE_DISPLAY  = {
    "LEAF_BLIGHT":         "Leaf Blight",
    "LEAF_HOPPERS":        "Leaf Hoppers",
    "LEAF_RUST":           "Leaf Rust",
    "LOOPER_CATERPILLARS": "Looper Caterpillars",
    "MOSQUITO_BUG":        "Mosquito Bug",
}
# BGR colours
_DISEASE_COLORS = {
    "LEAF_BLIGHT":         (34,  139, 34),
    "LEAF_HOPPERS":        (50,  50,  220),
    "LEAF_RUST":           (30,  144, 255),
    "LOOPER_CATERPILLARS": (148, 0,   211),
    "MOSQUITO_BUG":        (0,   206, 209),
}
_DISEASE_REMEDIES = {
    "LEAF_BLIGHT": [
        "Remove and destroy infected leaves immediately.",
        "Apply copper-based fungicide (Bordeaux mixture) every 10-14 days.",
        "Improve air circulation by pruning the dense canopy.",
        "Avoid overhead irrigation; water at the base in the morning.",
        "Ensure balanced nutrition - excess nitrogen increases susceptibility.",
    ],
    "LEAF_HOPPERS": [
        "Spray neem oil (5 ml/L) or insecticidal soap on leaf undersides.",
        "Introduce natural predators such as lacewings or parasitic wasps.",
        "Use yellow sticky traps to monitor and reduce adult populations.",
        "Remove heavily infested shoots and destroy them.",
        "Apply imidacloprid only in severe outbreaks.",
    ],
    "LEAF_RUST": [
        "Apply sulphur-based or triazole fungicide at the first sign.",
        "Remove heavily infected leaves and dispose off-site.",
        "Avoid wetting foliage; use drip irrigation where possible.",
        "Ensure adequate potassium levels to strengthen cell walls.",
        "Rotate fungicide classes to prevent resistance.",
    ],
    "LOOPER_CATERPILLARS": [
        "Hand-pick caterpillars and egg masses in small-scale gardens.",
        "Spray Bacillus thuringiensis (Bt) - safe biological control.",
        "Apply spinosad or pyrethrin-based spray in the evening.",
        "Install pheromone traps to monitor moth populations.",
        "Encourage birds and parasitic wasps as natural predators.",
    ],
    "MOSQUITO_BUG": [
        "Spray cypermethrin (0.005%) during cooler hours.",
        "Remove weeds and alternate hosts around the plantation.",
        "Use light traps at night to reduce adult populations.",
        "Apply systemic insecticide in severe cases, respecting PHI.",
        "Maintain field hygiene by removing crop debris after harvest.",
    ],
}
_DISEASE_TEXT_ALIASES = {
    "leaf blight":        "LEAF_BLIGHT",
    "brown blight":       "LEAF_BLIGHT",
    "gray blight":        "LEAF_BLIGHT",
    "grey blight":        "LEAF_BLIGHT",
    "algal leaf":         "LEAF_BLIGHT",
    "blight":             "LEAF_BLIGHT",
    "leaf hoppers":       "LEAF_HOPPERS",
    "leafhopper":         "LEAF_HOPPERS",
    "helopeltis":         "LEAF_HOPPERS",
    "hopper":             "LEAF_HOPPERS",
    "jassid":             "LEAF_HOPPERS",
    "leaf rust":          "LEAF_RUST",
    "rust":               "LEAF_RUST",
    "looper caterpillar": "LOOPER_CATERPILLARS",
    "looper":             "LOOPER_CATERPILLARS",
    "caterpillar":        "LOOPER_CATERPILLARS",
    "worm":               "LOOPER_CATERPILLARS",
    "mosquito bug":       "MOSQUITO_BUG",
    "mosquito":           "MOSQUITO_BUG",
}


# LightweightVisionClassifier - same architecture as FarmFederate training.
class _LightweightVisionClassifier(_nn.Module if _nn else object):
    def __init__(self, num_labels=5):
        super().__init__()
        self.num_labels = num_labels
        self.stem = _nn.Sequential(
            _nn.Conv2d(3, 64, 7, stride=2, padding=3),
            _nn.BatchNorm2d(64), _nn.ReLU(),
            _nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.block1 = _nn.Sequential(
            _nn.Conv2d(64, 128, 3, padding=1), _nn.BatchNorm2d(128), _nn.ReLU(),
            _nn.Dropout2d(0.1),
            _nn.Conv2d(128, 128, 3, padding=1), _nn.BatchNorm2d(128), _nn.ReLU(),
        )
        self.down1 = _nn.Conv2d(64, 128, 1)
        self.block2 = _nn.Sequential(
            _nn.Conv2d(128, 256, 3, padding=1), _nn.BatchNorm2d(256), _nn.ReLU(),
            _nn.Dropout2d(0.1),
            _nn.Conv2d(256, 256, 3, padding=1), _nn.BatchNorm2d(256), _nn.ReLU(),
        )
        self.down2 = _nn.Conv2d(128, 256, 1)
        self.block3 = _nn.Sequential(
            _nn.Conv2d(256, 512, 3, padding=1), _nn.BatchNorm2d(512), _nn.ReLU(),
            _nn.Dropout2d(0.15),
            _nn.Conv2d(512, 512, 3, padding=1), _nn.BatchNorm2d(512), _nn.ReLU(),
        )
        self.down3 = _nn.Conv2d(256, 512, 1)
        self.pool = _nn.AdaptiveAvgPool2d(1)
        self.classifier = _nn.Sequential(
            _nn.Flatten(), _nn.LayerNorm(512),
            _nn.Linear(512, 256), _nn.GELU(), _nn.Dropout(0.3),
            _nn.Linear(256, num_labels),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x) + self.down1(x)
        x = self.block2(x) + self.down2(x)
        x = self.block3(x) + self.down3(x)
        x = self.pool(x)
        return self.classifier(x)


def _load_vit_checkpoint(ckpt_path):
    """Load LightweightVisionClassifier from a FarmFederate .pt checkpoint."""
    ckpt = _torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model = _LightweightVisionClassifier(num_labels=5)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


class _DiseaseClassifier:
    def __init__(self):
        self._device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
        self._tfm = _tv_T.Compose([
            _tv_T.Resize((224, 224)),
            _tv_T.ToTensor(),
            _tv_T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self._model = self._load_model()
        self._ref_imgs: dict = self._build_ref_index()

    def _load_model(self):
        # Prefer the strongest v3 image checkpoint, then federated variants.
        for ckpt_name in ["vit_efficientnet_best.pt", "federated_vit_best.pt",
                          "centralized_vit_best.pt", "vit_vit_base_best.pt"]:
            ckpt_path = _CHECKPOINTS_DIR / ckpt_name
            if ckpt_path.exists():
                print(f"[disease] Loading {ckpt_name} ...")
                try:
                    m = _load_vit_checkpoint(ckpt_path).to(self._device)
                    print(f"[disease] Model ready ({ckpt_name})")
                    return m
                except Exception as e:
                    print(f"[disease] Failed to load {ckpt_name}: {e}")
        print("[disease] No checkpoint found - random-weight fallback")
        return _LightweightVisionClassifier(num_labels=5).eval().to(self._device)

    def _build_ref_index(self) -> dict:
        """Index reference image paths per class from tea_data and annotated dirs."""
        idx = {c: [] for c in _DISEASE_CLASSES}
        # Primary: tea_data augmented images (800/class)
        if _TEA_DATA_DIR.exists():
            for cls in _DISEASE_CLASSES:
                img_dir = _TEA_DATA_DIR / cls / "images"
                if img_dir.exists():
                    idx[cls] = [p for p in img_dir.iterdir()
                                if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        # Secondary: disease_annotated folder (matched by filename keyword)
        if _ANNOTATED_DIR.exists():
            _cls_keywords = {
                "LEAF_BLIGHT": ["blight", "brown", "blighted"],
                "LEAF_HOPPERS": ["hopper", "jassid", "leafhopper"],
                "LEAF_RUST": ["rust"],
                "LOOPER_CATERPILLARS": ["looper", "caterpillar"],
                "MOSQUITO_BUG": ["mosquito"],
            }
            for p in _ANNOTATED_DIR.iterdir():
                if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue
                fname = p.name.lower()
                for cls, kws in _cls_keywords.items():
                    if any(k in fname for k in kws):
                        idx[cls].append(p)
                        break
        # Final fallback: Merged-Dataset
        if all(len(v) == 0 for v in idx.values()) and _MERGED_DIR.exists():
            for cls in _DISEASE_CLASSES:
                idx[cls] = [p for p in (_MERGED_DIR / cls).glob("*")
                            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        return idx

    def classify(self, pil_img) -> tuple:
        """Run trained ViT model -> (class_name, confidence 0-1)."""
        x = self._tfm(pil_img.convert("RGB")).unsqueeze(0).to(self._device)
        with _torch.no_grad():
            logits = self._model(x)
            probs = _F.softmax(logits, dim=-1)[0]
        idx = int(probs.argmax())
        cls = _DISEASE_CLASSES[idx]
        conf = float(probs[idx])
        return cls, conf

    @staticmethod
    def _odd_kernel(value: int, minimum: int = 3) -> int:
        value = max(minimum, int(value))
        return value if value % 2 == 1 else value + 1

    def _leaf_context_mask(self, img) -> "np.ndarray":
        """Broad plant/lesion context used only to reject background pixels."""
        H, W = img.shape[:2]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        green = cv2.inRange(hsv, (18, 20, 35), (105, 255, 255))
        lesion_tones = cv2.inRange(hsv, (0, 25, 25), (38, 255, 235))
        context = green
        if cv2.countNonZero(context) < 0.02 * W * H:
            context = cv2.bitwise_or(context, lesion_tones)

        k = self._odd_kernel(min(W, H) / 80, 5)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        context = cv2.morphologyEx(context, cv2.MORPH_CLOSE, kernel)
        context = cv2.morphologyEx(context, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(context, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = max(80.0, 0.002 * W * H)
        max_dist = max(1.0, (W * W + H * H) ** 0.5 / 2)
        scored = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            moments = cv2.moments(contour)
            if not moments["m00"]:
                continue
            cx = moments["m10"] / moments["m00"]
            cy = moments["m01"] / moments["m00"]
            dist = ((cx - W / 2) ** 2 + (cy - H / 2) ** 2) ** 0.5
            centrality = 1.0 - min(1.0, dist / max_dist)
            scored.append((area * (0.45 + 0.55 * centrality), contour))
        if scored:
            focused = np.zeros_like(context)
            for _, contour in sorted(scored, key=lambda item: item[0], reverse=True)[:3]:
                cv2.drawContours(focused, [contour], -1, 255, -1)
            if cv2.countNonZero(focused) >= min_area:
                context = focused

        # Include immediate neighbourhood so holes and margin feeding damage can be boxed.
        dilate_k = self._odd_kernel(min(W, H) / 35, 7)
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_k, dilate_k))
        return cv2.dilate(context, dilate_kernel, iterations=1)

    def _color_anomaly_mask(self, img, context) -> "np.ndarray":
        """Highlight pixels whose colour differs strongly from the surrounding leaf."""
        if cv2.countNonZero(context) < 50:
            return np.zeros(context.shape, dtype=np.uint8)

        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        pixels = lab[context > 0]
        median = np.median(pixels, axis=0)
        dist = np.sqrt(np.sum((lab - median) ** 2, axis=2))
        context_dist = dist[context > 0]
        threshold = max(18.0, float(np.percentile(context_dist, 82)))
        return (((dist >= threshold) & (context > 0)).astype(np.uint8) * 255)

    def _disease_region_mask(self, img, cls: str) -> "np.ndarray":
        """Class-aware lesion mask for uploaded leaves."""
        H, W = img.shape[:2]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        b, g, r = [c.astype(np.int16) for c in cv2.split(img)]
        context = self._leaf_context_mask(img)
        anomaly = self._color_anomaly_mask(img, context)

        orange_rust = (
            (h >= 4) & (h <= 34) & (s >= 55) & (v >= 65) &
            (r >= g - 8) & (g >= b + 4)
        )
        brown_necrosis = (
            (((h <= 32) | (h >= 170)) & (s >= 35) & (v >= 35) & (v <= 215) &
             (r >= g + 4) & (r >= b + 8))
        )
        dark_dead_tissue = ((v <= 90) & (s >= 18) & (r >= g - 4) & (r >= b + 5))
        pale_stipple = (
            (((h >= 18) & (h <= 58) & (s >= 18) & (s <= 175) & (v >= 115)) |
             ((s <= 65) & (v >= 145))) &
            (anomaly > 0)
        )
        feeding_holes = (((v <= 70) | ((s <= 38) & (v >= 170))) & (anomaly > 0))

        if cls == "LEAF_RUST":
            mask = orange_rust
        elif cls == "LEAF_BLIGHT":
            mask = brown_necrosis | dark_dead_tissue
        elif cls == "LEAF_HOPPERS":
            mask = pale_stipple | ((brown_necrosis | dark_dead_tissue) & (anomaly > 0))
        elif cls == "LOOPER_CATERPILLARS":
            mask = feeding_holes | brown_necrosis | ((anomaly > 0) & (v <= 185))
        elif cls == "MOSQUITO_BUG":
            mask = brown_necrosis | dark_dead_tissue | ((anomaly > 0) & (v <= 190))
        else:
            mask = anomaly > 0

        mask = ((mask & (context > 0)).astype(np.uint8) * 255)
        open_k = self._odd_kernel(min(W, H) / 180, 3)
        close_k = self._odd_kernel(min(W, H) / 90, 5)
        open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
        return mask

    def _boxes_from_mask(self, mask, min_area_ratio: float = 0.0005, max_boxes: int = 8) -> list:
        H, W = mask.shape[:2]
        min_area = max(12.0, min_area_ratio * W * H)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) >= min_area]
        contours.sort(key=cv2.contourArea, reverse=True)
        boxes = []
        for contour in contours[:max_boxes]:
            pts = cv2.boxPoints(cv2.minAreaRect(contour)).tolist()
            boxes.append(pts)
        return boxes

    def _fallback_leaf_boxes(self, img) -> list:
        H, W = img.shape[:2]
        context = self._leaf_context_mask(img)
        boxes = self._boxes_from_mask(context, min_area_ratio=0.01, max_boxes=3)
        if boxes:
            return boxes
        m = 15
        return [[(m, m), (W-m, m), (W-m, H-m), (m, H-m)]]

    def annotate(self, pil_img, cls: str, conf: float) -> bytes:
        """Return annotated JPEG bytes with disease-region OBB polygons and label."""
        img = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
        H, W = img.shape[:2]
        color = _DISEASE_COLORS[cls]
        font_s = max(0.45, min(0.85, min(W, H) / 900))
        thick  = max(1, int(min(W, H) / 360))

        region_mask = self._disease_region_mask(img, cls)
        boxes = self._boxes_from_mask(region_mask)
        used_disease_mask = bool(boxes)
        if not boxes:
            boxes = self._fallback_leaf_boxes(img)

        if used_disease_mask:
            overlay = img.copy()
            overlay[region_mask > 0] = (
                0.45 * np.array(color, dtype=np.float32) +
                0.55 * overlay[region_mask > 0].astype(np.float32)
            ).astype(np.uint8)
            img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)
            contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            visible_contours = [c for c in contours if cv2.contourArea(c) >= max(12.0, 0.0005 * W * H)]
            cv2.drawContours(img, visible_contours, -1, color, max(1, thick))

        label = f"{_DISEASE_DISPLAY[cls]} area  {conf*100:.0f}%"
        if not used_disease_mask:
            label = f"{_DISEASE_DISPLAY[cls]} suspected  {conf*100:.0f}%"
        for i, pts in enumerate(boxes):
            arr = np.array(pts, dtype=np.int32)
            cv2.polylines(img, [arr], True, color, thick)
            if i != 0:
                continue
            (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_s, thick)
            x0, y0 = arr[0]
            rx = min(max(0, int(x0)-2), max(0, W-tw-8))
            ry = int(y0) - th - bl - 6
            if ry < 0:
                ry = min(max(0, int(y0)+6), max(0, H-th-bl-8))
            cv2.rectangle(img, (rx, ry), (rx+tw+6, ry+th+bl+6), color, -1)
            cv2.putText(img, label, (rx+3, ry+th+3),
                        cv2.FONT_HERSHEY_SIMPLEX, font_s, (255, 255, 255), thick, cv2.LINE_AA)

        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return buf.tobytes()

    def reference_image(self, cls: str) -> bytes | None:
        paths = self._ref_imgs.get(cls, [])
        if not paths:
            return None
        p = random.choice(paths)
        pil = _PILImage.open(p).convert("RGB")
        return self.annotate(pil, cls, 1.0)

    def all_class_confidences(self, pil_img) -> list:
        """Return confidence for all 5 classes."""
        x = self._tfm(pil_img.convert("RGB")).unsqueeze(0).to(self._device)
        with _torch.no_grad():
            logits = self._model(x)
            probs = _F.softmax(logits, dim=-1)[0]
        return [{"class": _DISEASE_CLASSES[i], "display": _DISEASE_DISPLAY[_DISEASE_CLASSES[i]],
                 "confidence": round(float(probs[i]), 4)} for i in range(5)]


def _get_disease_classifier():
    global _disease_classifier
    if _disease_classifier is None and _DISEASE_READY:
        print("[disease] Initialising FarmFederate disease classifier...")
        _disease_classifier = _DiseaseClassifier()
        print("[disease] Ready.")
    return _disease_classifier


def _match_disease_text(text: str) -> str | None:
    low = text.lower().strip()
    for alias in sorted(_DISEASE_TEXT_ALIASES, key=len, reverse=True):
        if alias in low:
            return _DISEASE_TEXT_ALIASES[alias]
    for cls in _DISEASE_CLASSES:
        if cls.lower().replace("_", " ") in low:
            return cls
    return None

# --- Active v3 tea disease configuration ---
# Used by /predict, /rag, /models, and /health for Paper v3 / Report v3 /
# farmfederate_results (3).
ISSUE_LABELS = _DISEASE_CLASSES

DISPLAY_NAMES = _DISEASE_DISPLAY

ADVICE = {
    "LEAF_BLIGHT": (
        "Remove infected leaves, improve canopy airflow, avoid overhead irrigation, "
        "and apply a copper-based fungicide according to local tea-board guidance."
    ),
    "LEAF_HOPPERS": (
        "Inspect the undersides of young leaves, prune dense shoots, deploy yellow "
        "sticky traps, and use neem or a recommended systemic insecticide if pressure is high."
    ),
    "LEAF_RUST": (
        "Remove heavily infected leaves, reduce leaf wetness, improve air circulation, "
        "and apply sulphur or triazole fungicide at the first sign of pustules."
    ),
    "LOOPER_CATERPILLARS": (
        "Scout early morning for larvae and frass, hand-pick where practical, and "
        "apply Bacillus thuringiensis or a locally approved caterpillar control."
    ),
    "MOSQUITO_BUG": (
        "Remove weeds and alternate hosts, monitor tender shoots for corky lesions, "
        "and apply recommended Helopeltis control during cooler hours if infestation spreads."
    ),
}

KEYWORD_MAP = {
    "LEAF_BLIGHT": [
        "leaf blight", "blight", "brown", "necrotic", "necrosis", "water-soaked",
        "water soaked", "margin", "lesion", "patch", "halo",
    ],
    "LEAF_HOPPERS": [
        "leaf hopper", "leaf hoppers", "leafhopper", "hopper", "jassid",
        "stippling", "tip burn", "scorch", "puncture", "underside",
    ],
    "LEAF_RUST": [
        "leaf rust", "rust", "orange", "yellow pustule", "pustule",
        "urediniospore", "spore", "lower surface", "powder",
    ],
    "LOOPER_CATERPILLARS": [
        "looper", "looper caterpillar", "caterpillar", "larvae", "larva",
        "skeletonized", "skeletonised", "ragged", "hole", "holes", "frass",
        "chewed", "eaten",
    ],
    "MOSQUITO_BUG": [
        "mosquito bug", "helopeltis", "mosquito", "corky", "raised lesion",
        "shoot dieback", "feeding wound", "tender shoot", "puncture wound",
    ],
}

SYMPTOM_PHRASES = {
    "blight": "blight patches",
    "brown": "brown necrotic patches",
    "necrotic": "necrotic lesions",
    "water-soaked": "water-soaked margins",
    "water soaked": "water-soaked margins",
    "hopper": "leafhopper feeding marks",
    "stippling": "stippling",
    "tip burn": "tip burn",
    "scorch": "marginal scorch",
    "rust": "rust symptoms",
    "orange": "orange pustules",
    "pustule": "pustules",
    "looper": "looper caterpillar damage",
    "caterpillar": "caterpillar feeding",
    "skeletonized": "skeletonized leaves",
    "skeletonised": "skeletonized leaves",
    "holes": "feeding holes",
    "hole": "feeding holes",
    "frass": "frass",
    "mosquito bug": "mosquito bug injury",
    "helopeltis": "Helopeltis injury",
    "corky": "corky lesions",
    "shoot dieback": "shoot dieback",
}

CROP_KEYWORDS = {
    "tea": ["tea", "camellia", "plantation", "garden", "flush", "shoot", "leaf"],
}

STRUCTURED_ACTIONS = {
    "LEAF_BLIGHT": {
        "today": [
            "Remove and destroy visibly infected tea leaves.",
            "Stop overhead watering in affected rows and keep leaf surfaces dry.",
        ],
        "this_week": [
            "Improve canopy airflow by pruning congested shoots.",
            "Apply copper-based fungicide at the locally recommended label rate.",
        ],
        "watch_for": [
            "New brown lesions or yellow halos on young leaves.",
            "Expansion of water-soaked margins after rain or irrigation.",
        ],
    },
    "LEAF_HOPPERS": {
        "today": [
            "Inspect leaf undersides and tender flushes for hoppers and nymphs.",
            "Remove heavily affected shoots where tip burn is severe.",
        ],
        "this_week": [
            "Install yellow sticky traps around affected blocks.",
            "Use neem or a recommended systemic insecticide if hopper pressure remains high.",
        ],
        "watch_for": [
            "Fresh stippling on new leaves.",
            "Increasing marginal scorch or curled tender shoots.",
        ],
    },
    "LEAF_RUST": {
        "today": [
            "Remove heavily rusted leaves and keep them away from the compost pile.",
            "Avoid wetting foliage during irrigation.",
        ],
        "this_week": [
            "Apply sulphur-based or triazole fungicide at first visible pustules.",
            "Open the canopy to reduce humidity around infected plants.",
        ],
        "watch_for": [
            "Orange-yellow pustules spreading on lower leaf surfaces.",
            "Powdery spores appearing after humid mornings.",
        ],
    },
    "LOOPER_CATERPILLARS": {
        "today": [
            "Scout early morning for larvae, frass, and ragged feeding holes.",
            "Hand-pick caterpillars and egg masses in small affected blocks.",
        ],
        "this_week": [
            "Apply Bacillus thuringiensis while larvae are young.",
            "Use pheromone traps to monitor adult moth movement.",
        ],
        "watch_for": [
            "New skeletonized leaves.",
            "Fresh frass near feeding sites.",
        ],
    },
    "MOSQUITO_BUG": {
        "today": [
            "Inspect tender shoots for corky lesions and feeding punctures.",
            "Remove weeds and alternate hosts around the tea block.",
        ],
        "this_week": [
            "Apply locally approved Helopeltis control during cooler hours if injury spreads.",
            "Maintain shade balance and field hygiene to reduce mosquito bug pressure.",
        ],
        "watch_for": [
            "Fresh corky raised lesions.",
            "Shoot dieback in newly flushed growth.",
        ],
    },
}

KB_DOCS = [
    {
        "id": "lb_01", "stress": "LEAF_BLIGHT", "source": "FarmFederate Tea KB",
        "text": "Leaf blight in tea appears as brown necrotic lesions with water-soaked margins. Remove infected leaves and improve drainage and airflow.",
    },
    {
        "id": "lb_02", "stress": "LEAF_BLIGHT", "source": "FarmFederate Tea KB",
        "text": "Copper-based fungicide is commonly recommended for tea leaf blight. Avoid overhead irrigation because wet foliage accelerates lesion spread.",
    },
    {
        "id": "lb_03", "stress": "LEAF_BLIGHT", "source": "FarmFederate Tea KB",
        "text": "Dense canopy and high humidity increase blight risk. Prune congested shoots and destroy infected plant debris.",
    },
    {
        "id": "lh_01", "stress": "LEAF_HOPPERS", "source": "FarmFederate Tea KB",
        "text": "Leaf hopper feeding causes stippling, tip burn, marginal scorch, and puncture marks on tender tea leaves.",
    },
    {
        "id": "lh_02", "stress": "LEAF_HOPPERS", "source": "FarmFederate Tea KB",
        "text": "Monitor hoppers with yellow sticky traps and inspect the undersides of leaves for nymphs and adults.",
    },
    {
        "id": "lh_03", "stress": "LEAF_HOPPERS", "source": "FarmFederate Tea KB",
        "text": "Neem spray or recommended systemic insecticide can reduce severe leaf hopper outbreaks when cultural controls are insufficient.",
    },
    {
        "id": "lr_01", "stress": "LEAF_RUST", "source": "FarmFederate Tea KB",
        "text": "Leaf rust produces orange-yellow pustules and powdery spores, usually on the lower leaf surface.",
    },
    {
        "id": "lr_02", "stress": "LEAF_RUST", "source": "FarmFederate Tea KB",
        "text": "Remove rusted leaves, reduce leaf wetness, and improve inter-row airflow to slow rust spread.",
    },
    {
        "id": "lr_03", "stress": "LEAF_RUST", "source": "FarmFederate Tea KB",
        "text": "Sulphur or triazole fungicide should be applied early when rust pustules first appear, following local label guidance.",
    },
    {
        "id": "lc_01", "stress": "LOOPER_CATERPILLARS", "source": "FarmFederate Tea KB",
        "text": "Looper caterpillars create ragged holes, skeletonized tea leaves, and visible frass near feeding sites.",
    },
    {
        "id": "lc_02", "stress": "LOOPER_CATERPILLARS", "source": "FarmFederate Tea KB",
        "text": "Bacillus thuringiensis is most effective on young caterpillar larvae and is safer for beneficial insects.",
    },
    {
        "id": "lc_03", "stress": "LOOPER_CATERPILLARS", "source": "FarmFederate Tea KB",
        "text": "Scout in the early morning, hand-pick larvae in small gardens, and use pheromone traps for moth monitoring.",
    },
    {
        "id": "mb_01", "stress": "MOSQUITO_BUG", "source": "FarmFederate Tea KB",
        "text": "Mosquito bug, Helopeltis theivora, causes corky raised lesions, feeding wounds, and shoot dieback on tender tea shoots.",
    },
    {
        "id": "mb_02", "stress": "MOSQUITO_BUG", "source": "FarmFederate Tea KB",
        "text": "Remove alternate hosts and weeds around tea blocks to reduce mosquito bug shelter and breeding sites.",
    },
    {
        "id": "mb_03", "stress": "MOSQUITO_BUG", "source": "FarmFederate Tea KB",
        "text": "Apply recommended Helopeltis control during cooler hours and monitor fresh flushes for new corky lesions.",
    },
]

RESOURCE_LINKS = {
    label: [("FarmFederate tea advisory", "https://farmfederate-advisor.onrender.com")]
    for label in ISSUE_LABELS
}


def _retrieve(query: str, top_k: int = 5) -> list:
    """Keyword-based retrieval from KB_DOCS (demo mode, no FAISS)."""
    query_lower = query.lower()
    scored = []
    for doc in KB_DOCS:
        doc_words = set(doc["text"].lower().split() + [doc["stress"].replace("_", " ")])
        query_words = set(query_lower.split())
        overlap = len(doc_words & query_words)
        stress_kws = KEYWORD_MAP.get(doc["stress"], [])
        stress_hits = sum(1 for kw in stress_kws if kw in query_lower)
        score = overlap * 0.1 + stress_hits * 0.3
        if score > 0:
            scored.append({
                "id": doc["id"],
                "text": doc["text"],
                "stress_class": doc["stress"],
                "source": doc.get("source", ""),
                "score": round(score, 3),
            })
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def _detect_symptoms(description: str) -> list:
    """Return list of symptom display strings found in the description."""
    lower = description.lower()
    seen = set()
    found = []
    for keyword, display in SYMPTOM_PHRASES.items():
        if keyword in lower and display not in seen:
            seen.add(display)
            found.append(display)
    return found


def _detect_crop(description: str) -> str:
    """Return crop name if mentioned, else 'crop'."""
    lower = description.lower()
    for crop, keywords in CROP_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            return crop
    return "crop"


def _urgency_prefix(prob: float) -> str:
    if prob >= 0.75:
        return "Urgent"
    if prob >= 0.50:
        return "Attention needed"
    return "Monitor"


def _rag_advisory(description: str, top_k: int = 5) -> dict:
    """Build RAG advisory from tea-disease knowledge retrieval + prediction."""
    pred = predict_from_text(description if description.strip() else "general tea leaf check")
    active = [s["label"] for s in pred["active_labels"]]
    top_class = max(pred["all_scores"], key=lambda x: x["prob"])["label"] if pred["all_scores"] else "LEAF_BLIGHT"
    top_prob = max(pred["all_scores"], key=lambda x: x["prob"])["prob"] if pred["all_scores"] else 0.5

    query = f"{description} {top_class.replace('_', ' ')} {' '.join(active)}"
    retrieved = _retrieve(query, top_k=top_k)

    context_parts = [f"[{i+1}] {doc['text']}" for i, doc in enumerate(retrieved)]
    context = "\n".join(context_parts) if context_parts else "No specific guidance retrieved."

    disease_str = ", ".join(active) if active else "general tea disease monitoring"
    prompt = (
        f"Tea disease diagnosis: {disease_str}\n\n"
        f"Retrieved agronomic guidance:\n{context}\n\n"
        f"Based on the diagnosis and retrieved guidance, provide a treatment recommendation."
    )

    if not active:
        treatment = "No high-confidence tea disease signal detected. Continue routine monitoring every 3-5 days."
    else:
        symptoms = _detect_symptoms(description)
        crop = _detect_crop(description)
        urgency = _urgency_prefix(top_prob)
        display_names = ", ".join(DISPLAY_NAMES.get(a, a) for a in active)

        lines = []
        if symptoms:
            symptom_str = ", ".join(symptoms)
            lines.append(
                f"{urgency}: Based on the reported {symptom_str} in your {crop}, "
                f"this looks like {display_names} ({int(top_prob * 100)}% confidence).\n"
            )
        else:
            lines.append(
                f"{urgency}: Your {crop} is showing signs of {display_names} "
                f"({int(top_prob * 100)}% confidence).\n"
            )

        primary = max(pred["active_labels"], key=lambda x: x["prob"])["label"] if pred["active_labels"] else top_class
        actions = STRUCTURED_ACTIONS.get(primary, {})

        if actions.get("today"):
            lines.append("Do today:")
            for i, step in enumerate(actions["today"], 1):
                lines.append(f"  {i}. {step}")
            lines.append("")

        if actions.get("this_week"):
            lines.append("This week:")
            for i, step in enumerate(actions["this_week"], 1):
                lines.append(f"  {i}. {step}")
            lines.append("")

        if actions.get("watch_for"):
            lines.append("Watch for signs of improvement:")
            for step in actions["watch_for"]:
                lines.append(f"  - {step}")
            lines.append("")

        if len(active) > 1:
            secondary = [DISPLAY_NAMES.get(a, a) for a in active if a != primary]
            if secondary:
                lines.append(f"Also address: {', '.join(secondary)} - see the 'What We Found' section for details.")
                lines.append("")

        lines.append("If symptoms worsen after 3-5 days of treatment, consult your local agronomist.")

        all_links = []
        for label in active:
            for title, url in RESOURCE_LINKS.get(label, []):
                all_links.append((title, url))
        if all_links:
            lines.append("")
            lines.append("Learn more:")
            for title, url in all_links:
                lines.append(f"  - [{title}]({url})")

        treatment = "\n".join(lines)

    return {
        "retrieved": retrieved,
        "prompt": prompt,
        "treatment": treatment,
        "active_labels": pred["active_labels"],
        "all_scores": pred["all_scores"],
        "kb_docs_searched": len(KB_DOCS),
        "recall_at_k": round(len(retrieved) / max(len(KB_DOCS), 1), 3),
        "resource_links": {label: RESOURCE_LINKS.get(label, []) for label in active},
    }

def predict_from_text(text: str) -> dict:
    """Keyword-based prediction for demo mode."""
    text_lower = text.lower().strip()
    probs = [0.1] * len(ISSUE_LABELS)

    # Score each label based on keyword matches
    for i, label in enumerate(ISSUE_LABELS):
        keywords = KEYWORD_MAP.get(label, [])
        matches = sum(1 for kw in keywords if kw in text_lower)
        if matches > 0:
            probs[i] = min(0.5 + matches * 0.15, 0.95)

    # If no keywords matched, give a low-confidence monitoring response.
    if max(probs) <= 0.1:
        probs = [0.25, 0.12, 0.08, 0.10, 0.05]

    mask = [1 if p > 0.3 else 0 for p in probs]

    active_labels = []
    all_scores = []
    for i, label in enumerate(ISSUE_LABELS):
        entry = {"label": label, "prob": probs[i], "threshold": 0.3}
        all_scores.append(entry)
        if mask[i] == 1:
            active_labels.append(entry)
    active_labels.sort(key=lambda x: x["prob"], reverse=True)
    all_scores.sort(key=lambda x: x["prob"], reverse=True)

    # Build advice
    active = [ISSUE_LABELS[i] for i, v in enumerate(mask) if v == 1]
    if not active:
        advice = "No high-confidence tea disease signal detected. Continue routine monitoring."
    else:
        advice = "Recommended actions:\n" + "\n".join(
            [f"- {DISPLAY_NAMES.get(lab, lab)}: {ADVICE.get(lab, '')}" for lab in active]
        )

    return {
        "active_labels": active_labels,
        "all_scores": all_scores,
        "raw_probs": probs,
        "advice": advice,
    }


# --- FastAPI app ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[demo_server] Starting FarmFederate tea disease demo backend")
    yield
    print("[demo_server] Shutting down...")


app = FastAPI(title="FarmFederate-Advisor (Demo Mode)", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "device": "cpu",
        "model_loaded": _DISEASE_READY,
        "labels": ISSUE_LABELS,
        "demo_mode": True,
        "endpoints": ["/predict", "/rag", "/rag/metrics", "/models", "/sensors/latest", "/health"],
        "rag": {"kb_size": len(KB_DOCS), "retrieval_mode": "keyword_bm25_proxy"},
    }


@app.get("/sensors/latest")
async def get_latest_sensors():
    return JSONResponse({"error": "No sensor data available (demo mode)"}, status_code=404)


@app.post("/predict")
async def predict(request: Request):
    content_type = request.headers.get("content-type", "").lower()
    text = ""

    if "application/json" in content_type:
        data = await request.json()
        text = str(data.get("text", "") or "")
    elif "multipart/form-data" in content_type:
        form = await request.form()
        text = str(form.get("text", "") or "")
    else:
        return JSONResponse({"error": f"Unsupported Content-Type: {content_type}"}, status_code=415)

    if not text.strip():
        text = "general tea leaf check"

    result = predict_from_text(text)
    return JSONResponse({
        "client_id": "demo",
        "text_used": text,
        "result": result,
        "advice": result["advice"],
        "demo_mode": True,
    })


@app.post("/rag")
async def rag_advisory(request: Request):
    """Federated RAG advisory endpoint: retrieves from knowledge base and generates treatment."""
    content_type = request.headers.get("content-type", "").lower()
    description = ""

    if "application/json" in content_type:
        data = await request.json()
        description = str(data.get("description", "") or data.get("text", "") or "")
    elif "multipart/form-data" in content_type:
        form = await request.form()
        description = str(form.get("description", "") or form.get("text", "") or "")
    else:
        return JSONResponse({"error": f"Unsupported Content-Type: {content_type}"}, status_code=415)

    result = _rag_advisory(description, top_k=5)
    return JSONResponse({
        "client_id": "demo",
        "description_used": description,
        "result": result,
        "demo_mode": True,
        "rag_mode": "offline_keyword_retrieval",
    })


@app.get("/rag/metrics")
async def rag_metrics():
    """Return the RAG evaluation metrics from the last training run."""
    import os
    metrics_path = os.path.join(os.path.dirname(__file__), "RAG", "rag_results (1).json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            import json as _json
            return JSONResponse(_json.load(f))
    return JSONResponse({
        "classification": {"macro_f1": 1.0, "micro_f1": 1.0},
        "retrieval": {"recall_at_5": 0.129, "mrr": 0.100, "ndcg_at_5": 0.172, "kb_coverage": 1.0},
        "robustness": {"embedding_drift": 0.156},
        "config": {"num_farms": 3, "fed_rounds": 5, "rag_rounds": 10, "top_k": 5},
        "demo_mode": True,
    })


@app.get("/models")
async def list_models():
    current = {
        "id": "vlm_clip",
        "name": "FarmFederate CLIP VLM",
        "description": "CLIP-style tea disease classifier using image, text, and local advisory context.",
        "type": "vlm",
        "accuracy": 0.949,
        "llm_encoder": "bert_tiny",
        "vit_encoder": "efficientnet",
        "fusion_strategy": "clip",
        "checkpoint": "vlm_clip_best.pt",
        "num_params_m": 15.7,
    }
    return JSONResponse({
        "models": [
            current,
            {"id": "federated_vlm", "name": "Federated VLM (CLIP)", "type": "vlm", "accuracy": 0.861, "checkpoint": "federated_vlm_best.pt", "retention": 98.6},
            {"id": "vit_efficientnet", "name": "EfficientNet Image Encoder", "type": "vit", "accuracy": 0.911, "checkpoint": "vit_efficientnet_best.pt", "num_params_m": 5.0},
            {"id": "llm_bert_tiny", "name": "BERT-tiny Text Encoder", "type": "llm", "accuracy": 0.487, "checkpoint": "llm_bert_tiny_best.pt", "num_params_m": 11.1},
            {"id": "federated_rag", "name": "Federated RAG Advisory", "type": "rag", "accuracy": 1.0, "kb_coverage": 1.0, "recall_at_5": 0.129, "kb_size": 15},
        ],
        "recommendations": {
            "production": "FarmFederate CLIP VLM",
            "privacy": "Federated VLM (CLIP)",
            "advisory": "Federated RAG Advisory",
        },
        "current": current,
        "current_model": current,
        "demo_mode": True,
    })


@app.get("/models/{model_id}")
async def get_model_info(model_id: str):
    return JSONResponse({
        "model": {"id": model_id, "name": model_id, "type": "demo"},
        "is_current": False,
        "demo_mode": True,
    })


@app.get("/models/current")
async def get_current_model():
    return JSONResponse({
        "model": {
            "id": "vlm_clip",
            "name": "FarmFederate CLIP VLM",
            "type": "vlm",
            "accuracy": 0.949,
            "llm_encoder": "bert_tiny",
            "vit_encoder": "efficientnet",
            "fusion_strategy": "clip",
        },
        "demo_mode": True,
    })


@app.post("/models/switch/{model_id}")
async def switch_model(model_id: str):
    return JSONResponse({
        "success": False,
        "message": "Model switching not available in demo mode. Run the full server locally with your models.",
        "demo_mode": True,
    })


@app.post("/control/{device}")
async def control_device(device: str, request: Request):
    try:
        body = await request.json()
        state = body.get("state", False)
        return JSONResponse({
            "success": True,
            "device": device,
            "state": state,
            "message": f"{device} {'activated' if state else 'deactivated'} (demo)",
        })
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.post("/disease/detect")
async def disease_detect(image: UploadFile = File(...)):
    """Classify a leaf image using FarmFederate federated ViT model."""
    if not _DISEASE_READY:
        return JSONResponse({"error": "Disease detection unavailable (torch not installed)."}, status_code=503)
    try:
        clf = _get_disease_classifier()
        raw = await image.read()
        pil = _PILImage.open(io.BytesIO(raw)).convert("RGB")
        cls, conf = clf.classify(pil)
        ann_bytes = clf.annotate(pil, cls, conf)
        all_scores = clf.all_class_confidences(pil)
        return JSONResponse({
            "class_name":      cls,
            "display_name":    _DISEASE_DISPLAY[cls],
            "confidence":      round(conf, 4),
            "remedies":        _DISEASE_REMEDIES[cls],
            "all_scores":      all_scores,
            "annotated_image": base64.b64encode(ann_bytes).decode(),
            "model":           "FarmFederate federated_vit",
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/disease/text")
async def disease_from_text(request: Request):
    """Infer a tea disease from a name or symptom description and return a reference image."""
    if not _DISEASE_READY:
        return JSONResponse({"error": "Disease detection unavailable (torch not installed)."}, status_code=503)
    try:
        body = await request.json()
        query = str(body.get("query", "") or "").strip()
        if not query:
            return JSONResponse({"error": "Describe the tea leaf symptoms first."}, status_code=400)

        cls = _match_disease_text(query)
        confidence = 1.0
        matched_by = "name"
        all_scores = []

        if cls is None:
            pred = predict_from_text(query)
            all_scores = pred.get("all_scores", [])
            if not all_scores:
                return JSONResponse({"error": "Could not infer a tea disease from that description."}, status_code=404)
            top = max(all_scores, key=lambda x: x.get("prob", 0.0))
            confidence = float(top.get("prob", 0.0))
            if confidence < 0.30:
                return JSONResponse({
                    "error": (
                        "I could not confidently infer a tea disease. Describe visible symptoms "
                        "such as orange pustules, brown necrotic patches, stippling, holes, frass, "
                        "or corky lesions."
                    )
                }, status_code=404)
            cls = top["label"]
            matched_by = "description"

        clf = _get_disease_classifier()
        ref_bytes = clf.reference_image(cls)
        if ref_bytes is None:
            return JSONResponse({"error": f"No reference images for {cls}."}, status_code=404)
        return JSONResponse({
            "class_name":   cls,
            "display_name": _DISEASE_DISPLAY[cls],
            "confidence":   round(confidence, 4),
            "matched_by":   matched_by,
            "remedies":     _DISEASE_REMEDIES[cls],
            "all_scores": [
                {
                    "class": item["label"],
                    "display": _DISEASE_DISPLAY.get(item["label"], item["label"]),
                    "confidence": round(float(item.get("prob", 0.0)), 4),
                }
                for item in all_scores
            ],
            "reference_image": base64.b64encode(ref_bytes).decode(),
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("demo_server:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))


