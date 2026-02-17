#!/usr/bin/env python3
"""
model_manager.py — Manages trained model checkpoints from the models/ folder.
Supports LLM, ViT, and VLM (multimodal) models trained by FarmFederate_Colab_Complete.py.
"""

import os
import pathlib
import torch
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

# Fix PosixPath on Windows (models saved on Linux/Colab)
if os.name == 'nt':
    pathlib.PosixPath = pathlib.WindowsPath

ISSUE_LABELS = ["water_stress", "nutrient_def", "pest_risk", "disease_risk", "heat_stress"]


@dataclass
class ModelInfo:
    name: str
    path: str
    model_type: str  # 'LLM', 'ViT', 'VLM'
    fusion_type: Optional[str] = None
    f1_score: Optional[float] = None
    config: Optional[Dict[str, Any]] = field(default_factory=dict)
    training_mode: str = "centralized"  # 'centralized' or 'federated'

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "path": self.path,
            "model_type": self.model_type,
            "fusion_type": self.fusion_type,
            "f1_score": self.f1_score,
            "training_mode": self.training_mode,
            "config": {k: str(v) for k, v in (self.config or {}).items()},
        }


class ModelManager:
    """Discovers and manages trained model checkpoints."""

    def __init__(self, models_dir: str = "../models"):
        self.models_dir = os.path.abspath(models_dir)
        self.models: Dict[str, ModelInfo] = {}
        self._scan_models()

    def _scan_models(self):
        """Scan models directory and extract metadata from checkpoints."""
        if not os.path.isdir(self.models_dir):
            print(f"[ModelManager] Models directory not found: {self.models_dir}")
            return

        for fname in sorted(os.listdir(self.models_dir)):
            if not fname.endswith(".pt"):
                continue
            fpath = os.path.join(self.models_dir, fname)
            model_id = fname.replace(".pt", "")

            # Determine type from filename
            model_type, fusion_type, training_mode = self._parse_filename(fname)

            # Try to load metadata from checkpoint
            f1_score = None
            config = {}
            try:
                ck = torch.load(fpath, map_location="cpu", weights_only=False)
                if isinstance(ck, dict):
                    f1_score = ck.get("f1_score")
                    config = ck.get("config", {})
                    fusion_type = ck.get("fusion_type", fusion_type)
                    model_type = ck.get("model_type", model_type)
            except Exception as e:
                print(f"[ModelManager] Warning: could not read metadata from {fname}: {e}")

            self.models[model_id] = ModelInfo(
                name=model_id,
                path=fpath,
                model_type=model_type,
                fusion_type=fusion_type,
                f1_score=float(f1_score) if f1_score is not None else None,
                config=config if isinstance(config, dict) else {},
                training_mode=training_mode,
            )

        print(f"[ModelManager] Found {len(self.models)} models")

    @staticmethod
    def _parse_filename(fname: str):
        """Parse model type, fusion type, and training mode from filename."""
        name = fname.replace(".pt", "").replace("_best", "")
        training_mode = "centralized"
        fusion_type = None

        if name.startswith("federated_"):
            training_mode = "federated"
            name = name.replace("federated_", "")
        elif name.startswith("centralized_"):
            training_mode = "centralized"
            name = name.replace("centralized_", "")

        if name.startswith("vlm_"):
            model_type = "VLM"
            fusion_type = name.replace("vlm_", "")
        elif name.startswith("vit_"):
            model_type = "ViT"
        elif name.startswith("llm_"):
            model_type = "LLM"
        else:
            model_type = name.upper()

        return model_type, fusion_type, training_mode

    def list_models(self) -> List[dict]:
        return [m.to_dict() for m in self.models.values()]

    def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        return self.models.get(model_id)

    def get_recommended_model(self, use_case: str = "production") -> Optional[ModelInfo]:
        """Get recommended model for a use case."""
        candidates = list(self.models.values())
        if not candidates:
            return None

        if use_case == "privacy":
            # Prefer federated VLM
            fed_vlm = [m for m in candidates if m.training_mode == "federated" and m.model_type == "VLM"]
            if fed_vlm:
                return max(fed_vlm, key=lambda m: m.f1_score or 0)

        if use_case == "production":
            # Best F1 VLM
            vlm = [m for m in candidates if m.model_type == "VLM"]
            if vlm:
                return max(vlm, key=lambda m: m.f1_score or 0)

        if use_case == "fast":
            # Fastest = LLM (text only)
            llm = [m for m in candidates if m.model_type == "LLM"]
            if llm:
                return max(llm, key=lambda m: m.f1_score or 0)

        if use_case == "vision":
            vit = [m for m in candidates if m.model_type == "ViT"]
            if vit:
                return max(vit, key=lambda m: m.f1_score or 0)

        if use_case == "text":
            llm = [m for m in candidates if m.model_type == "LLM"]
            if llm:
                return max(llm, key=lambda m: m.f1_score or 0)

        # Fallback: best F1
        return max(candidates, key=lambda m: m.f1_score or 0)

    def export_model_catalog(self, output_path: str = "model_catalog.json"):
        """Export model catalog as JSON."""
        import json
        catalog = {
            "models": self.list_models(),
            "total": len(self.models),
        }
        with open(output_path, "w") as f:
            json.dump(catalog, f, indent=2)
        print(f"[ModelManager] Exported catalog to {output_path}")
