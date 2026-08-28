#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
farmfederate_rag_standalone.py
==============================
Standalone RAG implementation and evaluation for FarmFederate.

Extracted from FarmFederate_Colab_Complete.py — runs independently.

Contains:
  1. Core RAG data structures  (Document, AgriculturalChunker, FarmVectorStore)
  2. Dual encoder              (RetrieverEncoder, RAGQueryBuilder)
  3. Context assembler         (ContextAssembler)
  4. Local data loaders        (LocalDatasetLoader, ImageCaptionLoader)
  5. Advisory generation       (LLMAdvisoryGenerator)
  6. Evaluation metrics        (RAGEvaluator — Recall@K, MRR, NDCG, drift)
  7. Federated RAG training     (FedRAGClient, FedRAGServer, FedRAGConfig)
  8. End-to-end pipeline       (run_rag_diagnosis, run_rag_evaluation)

Usage (Colab / local):
    # Quick demo — no model training required
    results = run_rag_diagnosis(data_dir="data")

    # Full federated RAG training + evaluation
    results = run_full_rag_pipeline(data_dir="data", num_rounds=5)
"""

from __future__ import annotations

# ============================================================================
# STANDARD IMPORTS
# ============================================================================

import csv
import hashlib
import json
import logging
import math
import os
import re
import sys
import time
import uuid
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

# Optional heavy dependencies — graceful fallback when missing
try:
    import faiss
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    _SBERT_AVAILABLE = True
except ImportError:
    _SBERT_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS
# ============================================================================

STRESS_LABELS = ["water_stress", "nutrient_def", "pest_risk", "disease_risk", "heat_stress"]
STRESS_TYPES  = STRESS_LABELS          # alias used by rag_core
NUM_CLASSES   = len(STRESS_LABELS)
IOT_KEYS      = ["temperature", "humidity", "soil_moisture", "nitrogen", "phosphorus", "potassium"]
IOT_DIM       = len(IOT_KEYS)
QUERY_DIM     = 128
FUSION_DIM    = 512

LABEL_TO_STRESS = {i: st for i, st in enumerate(STRESS_LABELS)}

# Agriculture keywords used for captions.csv filtering
_AG_KEYWORDS = frozenset([
    "plant", "crop", "leaf", "leaves", "soil", "farm", "seed", "root",
    "flower", "fruit", "pest", "weed", "drought", "irrigat", "fertil",
    "harvest", "grain", "rice", "wheat", "maize", "cotton", "tomato",
    "potato", "bean", "stress", "disease", "blight", "wilt", "scorch",
    "chlorosis", "deficiency", "nutrient", "nitrogen", "phosphorus",
    "potassium", "insect", "fungal", "rust", "mildew", "rot",
])


# ============================================================================
# BUILT-IN KNOWLEDGE BASE  (used as fallback when no data directory is found)
# ============================================================================

_BUILTIN_KB: List[Dict] = [
    # ── water_stress ──────────────────────────────────────────────────────────
    {"stress_type": "water_stress", "content": "Water stress symptoms: severe wilting, leaf curl, drooping stems, and dry cracked soil. Plants close stomata to reduce transpiration, causing leaf rolling and turgor loss. Maize is especially sensitive during tasseling and silking stages."},
    {"stress_type": "water_stress", "content": "Treatment for water stress: irrigate immediately with 40-60 mm water. Drip or furrow irrigation preferred. Mulch soil surface to retain moisture. Irrigate during early morning or evening to minimize evaporation losses. Repeat every 5-7 days during dry spells."},
    {"stress_type": "water_stress", "content": "Prevention of water stress: install drip irrigation systems, use mulch to conserve soil moisture, plant drought-tolerant varieties, monitor soil moisture sensors, and schedule irrigation based on evapotranspiration data."},
    {"stress_type": "water_stress", "content": "Severe drought causes wilting in wheat, maize, and rice. Soil moisture below 20% triggers water stress. Leaves curl inward, turn grey-green then yellow. Root zone drying causes abscisic acid signaling that closes stomata."},
    {"stress_type": "water_stress", "content": "Rice plants under water deficit show rolled leaves, reduced tillering, and poor grain filling. Alternate wetting and drying (AWD) technique saves 30% water while maintaining yield. Maintain soil moisture at -20 kPa threshold."},
    {"stress_type": "water_stress", "content": "Soil moisture monitoring: tensiometers at 20 cm and 40 cm depth. Irrigate when reading exceeds 40 kPa for sandy soils or 60 kPa for clay soils. Drip irrigation at 80% field capacity maintains optimal growth."},

    # ── nutrient_def ──────────────────────────────────────────────────────────
    {"stress_type": "nutrient_def", "content": "Nitrogen deficiency symptoms: yellowing (chlorosis) starting from older lower leaves progressing upward, stunted growth, pale green to yellow leaves, reduced tillering. Interveinal chlorosis is characteristic of nitrogen shortage in cereals."},
    {"stress_type": "nutrient_def", "content": "Treatment for nitrogen deficiency: apply urea at 46-0-0 NPK at 50-80 kg/ha. Split application: 30% at planting, 40% at tillering, 30% at panicle initiation. Foliar spray of 2% urea solution provides quick green-up within 5-7 days."},
    {"stress_type": "nutrient_def", "content": "Potassium deficiency: brown scorching of leaf margins and tips, weak stems prone to lodging, poor fruit quality. Apply muriate of potash (0-0-60) at 60 kg K2O/ha. Deficiency common in sandy soils with low CEC."},
    {"stress_type": "nutrient_def", "content": "Phosphorus deficiency: purple or reddish discoloration of leaves, delayed maturity, poor root development. Apply DAP (18-46-0) at 50 kg/ha. Soil pH 6.0-7.0 maximizes phosphorus availability. Mycorrhizal fungi improve phosphorus uptake."},
    {"stress_type": "nutrient_def", "content": "Iron deficiency (chlorosis): interveinal yellowing on young leaves while veins remain green. Common in high-pH calcareous soils. Apply chelated iron (EDTA or DTPA) as foliar spray at 2 g/L every 10 days. Acidify soil to pH 6.0-6.5."},
    {"stress_type": "nutrient_def", "content": "Zinc deficiency: white or bronze striping on young maize leaves, shortened internodes, small leaves. Apply zinc sulfate at 5-10 kg/ha. Zinc uptake inhibited by high phosphorus levels. Soil test zinc below 0.5 ppm requires correction."},

    # ── pest_risk ─────────────────────────────────────────────────────────────
    {"stress_type": "pest_risk", "content": "Caterpillar damage symptoms: irregular holes in leaves, skeletonization of leaf tissue, frass (insect excrement) on leaves. Armyworms attack cereals in large numbers overnight. Stem borers create dead heart in young plants and white ear syndrome at flowering."},
    {"stress_type": "pest_risk", "content": "Treatment for caterpillar infestation: apply chlorpyrifos 20 EC at 2 ml/L or lambda-cyhalothrin 5 EC at 1 ml/L. Spray in evening when larvae are active. Bacillus thuringiensis (Bt) biological insecticide is effective and safe for beneficial insects."},
    {"stress_type": "pest_risk", "content": "Aphid infestation causes honeydew production, sooty mold, leaf curl, and mosaic virus transmission. Yellow sticky traps for monitoring. Apply imidacloprid 17.8 SL at 0.5 ml/L or neem oil 1500 ppm at 3 ml/L. Preserve natural enemies like ladybird beetles."},
    {"stress_type": "pest_risk", "content": "Fall armyworm (Spodoptera frugiperda) management: early morning scouting for leaf damage and egg masses. Economic threshold: 5 egg masses or 10% infested plants. Apply emamectin benzoate 5 SG at 0.4 g/L or spinosad. Pheromone traps for adult monitoring at 5/ha."},
    {"stress_type": "pest_risk", "content": "Thrips damage: silver streaking on leaves, distorted growth, black fecal spots. Common on onion, cotton, chilli. Apply spinosad at 0.3 ml/L. Reflective mulch repels thrips. Blue sticky traps for population monitoring. Maintain field hygiene."},
    {"stress_type": "pest_risk", "content": "Whitefly infestation causes yellowing, wilting, and virus transmission in tomato and cotton. Yellow sticky traps (30/ha) for monitoring. Apply buprofezin or spiromesifen. Remove heavily infested leaves. Botanical: neem seed kernel extract 5%."},

    # ── disease_risk ──────────────────────────────────────────────────────────
    {"stress_type": "disease_risk", "content": "Powdery mildew symptoms: white powdery fungal growth on leaf surface, primarily on upper side. Causes reduced photosynthesis and premature defoliation. Common on cucurbits, grapes, wheat during warm days and cool nights with high humidity. Favored by temperatures 20-25°C."},
    {"stress_type": "disease_risk", "content": "Treatment for powdery mildew: apply propiconazole 25 EC at 1 ml/L or tebuconazole 250 EW at 1 ml/L. Spray at first sign of disease. Potassium bicarbonate (5 g/L) effective organic alternative. Improve air circulation by pruning. Resistant varieties preferred."},
    {"stress_type": "disease_risk", "content": "Late blight (Phytophthora infestans) in potato and tomato: water-soaked lesions that turn brown with white sporulation on undersides. Spreads rapidly in cool wet conditions. Apply mancozeb 75 WP at 2.5 g/L preventively. Metalaxyl-mancozeb for active infections. Remove and destroy infected plant debris."},
    {"stress_type": "disease_risk", "content": "Bacterial leaf blight in rice: water-soaked margins turning yellow to white, wilting of leaves. Caused by Xanthomonas oryzae. No effective chemical control; use resistant varieties. Apply copper hydroxide as preventive. Reduce nitrogen, improve drainage, avoid flooding."},
    {"stress_type": "disease_risk", "content": "Fungal rust diseases show orange or reddish-brown pustules on leaves. Wheat stripe rust and leaf rust spread via airborne spores. Apply propiconazole or triadimefon at first sign. Plant resistant varieties. Scout weekly during cool humid weather. Remove volunteer plants."},
    {"stress_type": "disease_risk", "content": "Fusarium wilt causes yellowing and wilting on one side of plant. Infected vascular tissue shows brown discoloration when cut. Soil-borne pathogen persists for years. Use certified disease-free seed, crop rotation with non-host, soil solarization, and resistant varieties."},

    # ── heat_stress ──────────────────────────────────────────────────────────
    {"stress_type": "heat_stress", "content": "Heat stress symptoms: leaf margin scorch and browning, wilting during daytime, reduced pollen viability, flower drop, and premature fruit ripening. Temperatures above 35°C for more than 4 hours cause irreversible damage to photosynthesis machinery. Maize pollen is killed at 38°C."},
    {"stress_type": "heat_stress", "content": "Treatment for heat stress: irrigate during early morning (4-6 AM) or evening to lower canopy temperature. Shade netting (30-50% reduction) for high-value crops. Foliar spray of kaolin clay (50 g/L) reflects sunlight. Apply potassium nitrate 1% foliar to improve heat tolerance."},
    {"stress_type": "heat_stress", "content": "Heat stress management in wheat: crop duration shortens under high temperature; 1°C above optimum reduces yield 6%. Use early maturing heat-tolerant varieties. Timely sowing to avoid terminal heat stress. Micro-irrigation during anthesis protects pollen viability."},
    {"stress_type": "heat_stress", "content": "Tomato fruit set fails when night temperature exceeds 20°C or day temperature exceeds 35°C. Shade cloth 30% reduces temperature 3-5°C. Spray salicylic acid 100 ppm to improve heat tolerance. Pick fruits regularly to reduce plant load."},
    {"stress_type": "heat_stress", "content": "Cooling strategies for greenhouses: evaporative cooling pads reduce temperature 8-10°C. Reflective roofing paint reduces heat absorption. Ventilation fans maintain air movement. Humidification by fogging reduces leaf temperature through latent heat. Whitewashing roof reduces solar gain."},
    {"stress_type": "heat_stress", "content": "Heat stress indicators: leaf rolling during morning hours, canopy temperature 3°C above air temperature (measured by infrared thermometer), wilting before noon, flower abscission, and reduced fruit set. Monitor weather forecasts and pre-irrigate before heat waves above 40°C."},
]


# ============================================================================
# 1. DOCUMENT DATACLASS
# ============================================================================

@dataclass
class Document:
    """Agricultural document with rich metadata for filtered retrieval."""
    doc_id:     str
    content:    str
    crop:       str                    = "general"
    stress_type: str                   = "general"
    region:     str                    = "global"
    source:     str                    = "unknown"
    embedding:  Optional[np.ndarray]   = None
    metadata:   Dict[str, Any]         = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "doc_id": self.doc_id,
            "content": self.content,
            "crop": self.crop,
            "stress_type": self.stress_type,
            "region": self.region,
            "source": self.source,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "Document":
        return cls(
            doc_id=d["doc_id"],
            content=d["content"],
            crop=d.get("crop", "general"),
            stress_type=d.get("stress_type", "general"),
            region=d.get("region", "global"),
            source=d.get("source", "unknown"),
            metadata=d.get("metadata", {}),
        )


# ============================================================================
# 2. AGRICULTURAL CHUNKER
# ============================================================================

class AgriculturalChunker:
    """
    Domain-aware chunker that splits agricultural documents at section
    boundaries (Symptoms / Treatment / Prevention …) then at word windows.

    Each chunk inherits parent crop / stress_type / region metadata.
    """

    SECTION_HEADERS = re.compile(
        r"(?i)(symptoms?|causes?|treatment|prevention|management|control"
        r"|diagnosis|description|background|recommendation|advisory"
        r"|field\s+observation|spray\s+schedule|dose)",
        re.MULTILINE,
    )

    def __init__(self, chunk_size: int = 256, overlap: int = 32):
        self.chunk_size = chunk_size
        self.overlap    = overlap

    def chunk(self, doc: Document) -> List[Document]:
        sections = self._split_sections(doc.content)
        chunks: List[Document] = []
        for section_text in sections:
            for chunk_text in self._split_words(section_text):
                chunk_text = chunk_text.strip()
                if not chunk_text:
                    continue
                chunks.append(Document(
                    doc_id=f"{doc.doc_id}_chunk{len(chunks)}",
                    content=chunk_text,
                    crop=doc.crop,
                    stress_type=doc.stress_type,
                    region=doc.region,
                    source=doc.source,
                    metadata={**doc.metadata, "parent_doc_id": doc.doc_id, "chunk_index": len(chunks)},
                ))
        return chunks if chunks else [doc]

    def _split_sections(self, text: str) -> List[str]:
        parts = self.SECTION_HEADERS.split(text)
        if len(parts) <= 1:
            return [text]
        out = []
        for i in range(0, len(parts) - 1, 2):
            combined = (
                parts[i] + "\n" + parts[i + 1] + " " + parts[i + 2]
            ).strip() if i + 2 < len(parts) else (parts[i] + "\n" + parts[i + 1]).strip()
            out.append(combined)
        return out

    def _split_words(self, text: str) -> List[str]:
        words = text.split()
        if len(words) <= self.chunk_size:
            return [text]
        chunks, start = [], 0
        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunks.append(" ".join(words[start:end]))
            start += self.chunk_size - self.overlap
        return chunks


# ============================================================================
# 3. RETRIEVER ENCODER  (dual-encoder sharing LLM backbone)
# ============================================================================

class RetrieverEncoder(nn.Module):
    """
    Dual-encoder that projects a text-encoder's [CLS] token to QUERY_DIM (128).

    Pass the LLM backbone from MultiModalModel to share weights.
    When used standalone (no backbone), call encode_query(text_str) directly
    — it will use a simple TF-IDF / random fallback for demo purposes.
    """

    def __init__(
        self,
        text_encoder: Optional[nn.Module] = None,
        hidden_size: int = 768,
        query_dim: int = QUERY_DIM,
        tie_encoders: bool = True,
    ):
        super().__init__()
        self.text_encoder = text_encoder
        self.tie_encoders = tie_encoders
        self.query_proj = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, query_dim),
        )
        self.doc_proj = self.query_proj if tie_encoders else nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, query_dim),
        )

    def _cls(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            return out.pooler_output
        return out.last_hidden_state[:, 0, :]

    def encode_query(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.query_proj(self._cls(input_ids, attention_mask)), dim=-1)

    def encode_doc(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.doc_proj(self._cls(input_ids, attention_mask)), dim=-1)

    def forward(
        self,
        query_ids: torch.Tensor,
        query_mask: torch.Tensor,
        doc_ids: Optional[torch.Tensor] = None,
        doc_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        q_emb = self.encode_query(query_ids, query_mask)
        out: Dict[str, torch.Tensor] = {"query_emb": q_emb}
        if doc_ids is not None and doc_mask is not None:
            out["doc_emb"] = self.encode_doc(doc_ids, doc_mask)
        return out


# ============================================================================
# 4. RAG QUERY BUILDER  (multimodal → 128-d query vector)
# ============================================================================

class RAGQueryBuilder(nn.Module):
    """
    Fuses h_f (512), class_probs (5), IoT readings (6) → query_dim (128).

    Used when a trained MultiModalModel is available to build semantically
    rich query vectors from the vision + text fusion + sensor context.
    """

    def __init__(
        self,
        fusion_dim: int   = FUSION_DIM,
        num_classes: int  = NUM_CLASSES,
        iot_dim: int      = IOT_DIM,
        query_dim: int    = QUERY_DIM,
        dropout: float    = 0.1,
    ):
        super().__init__()
        input_dim = fusion_dim + num_classes + iot_dim
        self.proj = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, query_dim),
        )
        self.register_buffer("iot_mean", torch.zeros(iot_dim))
        self.register_buffer("iot_std",  torch.ones(iot_dim))

    def normalise_iot(self, iot_vals: torch.Tensor) -> torch.Tensor:
        return (iot_vals - self.iot_mean.to(iot_vals.device)) / (self.iot_std.to(iot_vals.device) + 1e-6)

    def forward(
        self,
        h_f: torch.Tensor,
        class_probs: torch.Tensor,
        iot_vals: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = h_f.size(0)
        if iot_vals is None:
            iot_vals = torch.zeros(B, IOT_DIM, device=h_f.device, dtype=h_f.dtype)
        else:
            iot_vals = self.normalise_iot(iot_vals.to(h_f.device, dtype=h_f.dtype))
        x = torch.cat([h_f, class_probs, iot_vals], dim=-1)
        return F.normalize(self.proj(x), dim=-1)


# ============================================================================
# 5. FARM VECTOR STORE  (per-farm FAISS / brute-force index)
# ============================================================================

class FarmVectorStore:
    """
    Per-farm knowledge base. Documents are indexed by L2-normalised embeddings.

    Backed by FAISS (inner product = cosine on normalised vectors) when
    available, otherwise brute-force numpy cosine similarity.

    KB documents NEVER leave the farm — this object is never sent to the server.
    """

    def __init__(self, dim: int = QUERY_DIM, farm_id: str = "farm_0"):
        self.dim       = dim
        self.farm_id   = farm_id
        self.documents: List[Document]        = []
        self._embeddings: Optional[np.ndarray] = None

        if _FAISS_AVAILABLE:
            self._index = faiss.IndexFlatIP(dim)
        else:
            self._index = None

    def add_documents(self, docs: List[Document], embeddings: np.ndarray) -> None:
        assert len(docs) == len(embeddings)
        embeddings = embeddings.astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9
        embeddings = embeddings / norms

        self.documents.extend(docs)
        self._embeddings = (
            embeddings if self._embeddings is None
            else np.vstack([self._embeddings, embeddings])
        )
        if _FAISS_AVAILABLE:
            self._index.add(embeddings)

    def search(
        self,
        query_emb: np.ndarray,
        top_k: int = 5,
        filters: Optional[Dict[str, str]] = None,
    ) -> List[Tuple[Document, float]]:
        if not self.documents:
            return []

        query_emb = query_emb.astype(np.float32).reshape(1, -1)
        query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-9)

        allowed = (
            [i for i, d in enumerate(self.documents) if self._matches(d, filters)]
            if filters else list(range(len(self.documents)))
        )
        if not allowed:
            allowed = list(range(len(self.documents)))

        cand = self._embeddings[allowed]
        scores = (cand @ query_emb.T).squeeze(-1)
        k = min(top_k, len(allowed))
        top_local = np.argpartition(-scores, k - 1)[:k]
        top_local = top_local[np.argsort(-scores[top_local])]

        return [(self.documents[allowed[i]], float(scores[i])) for i in top_local]

    def __len__(self) -> int:
        return len(self.documents)

    @staticmethod
    def _matches(doc: Document, filters: Dict[str, str]) -> bool:
        for key, val in filters.items():
            doc_val = getattr(doc, key, None) or doc.metadata.get(key)
            if doc_val != val and val not in ("general", "global"):
                return False
        return True


# ============================================================================
# 6. CONTEXT ASSEMBLER
# ============================================================================

class ContextAssembler:
    """
    Builds a structured LLM prompt from:
        - class_probs   (5 stress probabilities)
        - retrieved_docs (top-k from FarmVectorStore)
        - iot_readings  (dict of sensor values)
        - farmer_text   (free-text symptom description)
    """

    IOT_THRESHOLDS = {
        "temperature":  (10,  40,  "°C"),
        "humidity":     (30,  90,  "%"),
        "soil_moisture":(20,  80,  "%"),
        "nitrogen":     (0,   100, "ppm"),
        "phosphorus":   (0,   100, "ppm"),
        "potassium":    (0,   150, "ppm"),
    }

    def assemble(
        self,
        class_probs: List[float],
        retrieved_docs: List[Tuple[Document, float]],
        iot_readings: Optional[Dict[str, float]] = None,
        farmer_text: str = "",
        top_k_docs: int = 3,
    ) -> str:
        lines: List[str] = []

        lines.append("=== CROP STRESS DIAGNOSIS ===")
        lines.append("Predicted stress probabilities:")
        for label, prob in zip(STRESS_LABELS, class_probs):
            bar = "#" * int(prob * 20)
            lines.append(f"  {label:<15} {prob:.3f}  [{bar:<20}]")
        top_stress = STRESS_LABELS[int(np.argmax(class_probs))]
        lines.append(f"Primary concern: {top_stress}\n")

        if farmer_text:
            lines.append("=== FARMER OBSERVATION ===")
            lines.append(farmer_text.strip())
            lines.append("")

        if iot_readings:
            lines.append("=== SENSOR READINGS ===")
            for key, val in iot_readings.items():
                lo, hi, unit = self.IOT_THRESHOLDS.get(key, (None, None, ""))
                flag = (" [LOW]" if val < lo else " [HIGH]" if val > hi else "") if lo else ""
                lines.append(f"  {key}: {val:.1f}{unit}{flag}")
            lines.append("")

        docs_to_show = retrieved_docs[:top_k_docs]
        if docs_to_show:
            lines.append("=== RETRIEVED KNOWLEDGE BASE ENTRIES ===")
            for rank, (doc, score) in enumerate(docs_to_show, 1):
                lines.append(
                    f"[{rank}] Source: {doc.source} | Crop: {doc.crop} | "
                    f"Stress: {doc.stress_type} | Score: {score:.3f}"
                )
                lines.append(doc.content.strip())
                lines.append("")

        lines.append("=== ADVISORY REQUEST ===")
        lines.append(
            "Based on the diagnosis, sensor readings, farmer observation, and the "
            "retrieved agronomic knowledge above, provide:\n"
            "1. A brief explanation of the likely cause.\n"
            "2. Immediate recommended actions (within 24 hours).\n"
            "3. Medium-term management plan (next 2 weeks).\n"
            "4. Preventive measures for the rest of the season.\n"
            "Keep the language simple and actionable for smallholder farmers."
        )
        return "\n".join(lines)

    def assemble_device_template(
        self,
        class_probs: List[float],
        iot_readings: Optional[Dict[str, float]] = None,
    ) -> str:
        """Lightweight on-device advisory template (no LLM needed)."""
        top_idx   = int(np.argmax(class_probs))
        top_label = STRESS_LABELS[top_idx]
        top_prob  = class_probs[top_idx]
        templates = {
            "water_stress":  "Likely water stress detected ({p:.0%}). Check soil moisture and consider irrigation.",
            "nutrient_def":  "Likely nutrient deficiency detected ({p:.0%}). Test soil and apply balanced fertiliser.",
            "pest_risk":     "Elevated pest risk detected ({p:.0%}). Inspect crops and consult IPM guidelines.",
            "disease_risk":  "Disease risk detected ({p:.0%}). Apply recommended fungicide and remove infected material.",
            "heat_stress":   "Heat stress detected ({p:.0%}). Irrigate during cooler hours and consider shade netting.",
        }
        advisory = templates.get(top_label, "Stress detected ({p:.0%}). Consult local agronomist.").format(p=top_prob)
        if iot_readings:
            temp = iot_readings.get("temperature")
            soil = iot_readings.get("soil_moisture")
            if temp and temp > 38:
                advisory += f" Current temperature {temp:.1f}°C is critically high."
            if soil and soil < 25:
                advisory += f" Soil moisture {soil:.1f}% is critically low."
        return advisory


# ============================================================================
# 7. LOCAL DATA LOADERS
# ============================================================================

def _content_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8", errors="replace")).hexdigest()[:12]


def _is_agriculture_text(text: str) -> bool:
    lower = text.lower()
    return any(kw in lower for kw in _AG_KEYWORDS)


class LocalDatasetLoader:
    """
    Loads per-stress-type text CSVs from data/<stress_type>/text.csv
    plus the consolidated crop_stress_text_dataset.csv and captions.csv.

    Returns a deduplicated list of Document objects.
    """

    def __init__(self, max_docs_per_class: int = 200):
        self.max_docs_per_class = max_docs_per_class

    def load_stress_text_csvs(self, data_dir: str) -> List[Document]:
        data_path = Path(data_dir)
        docs: List[Document] = []
        seen: Set[str] = set()

        for stress_type in STRESS_LABELS:
            csv_path = data_path / stress_type / "text.csv"
            if not csv_path.exists():
                continue
            class_docs: List[Document] = []
            try:
                with open(csv_path, encoding="utf-8", errors="replace") as f:
                    for row in csv.DictReader(f):
                        text = (row.get("text") or "").strip()
                        if not text or len(text) < 20:
                            continue
                        h = _content_hash(text)
                        if h in seen:
                            continue
                        seen.add(h)
                        source = (row.get("source") or "local_text").strip()
                        class_docs.append(Document(
                            doc_id=f"local_{stress_type}_{h}",
                            content=text,
                            crop="general",
                            stress_type=stress_type,
                            region="global",
                            source=f"local_{source}",
                            metadata={"loader": "stress_text_csv"},
                        ))
            except Exception as e:
                logger.warning("Failed to read %s: %s", csv_path, e)
                continue
            if self.max_docs_per_class > 0:
                class_docs = class_docs[: self.max_docs_per_class]
            docs.extend(class_docs)
        return docs

    def load_crop_stress_dataset(self, csv_path: str) -> List[Document]:
        path = Path(csv_path)
        if not path.exists():
            return []
        docs: List[Document] = []
        seen: Set[str] = set()
        class_counts: Dict[str, int] = {st: 0 for st in STRESS_LABELS}
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                for row in csv.DictReader(f):
                    text = (row.get("text") or "").strip()
                    if not text or len(text) < 20:
                        continue
                    h = _content_hash(text)
                    if h in seen:
                        continue
                    seen.add(h)
                    label_name = (row.get("label_name") or "").strip()
                    if label_name in STRESS_LABELS:
                        stress_type = label_name
                    else:
                        try:
                            stress_type = LABEL_TO_STRESS.get(int(row.get("label", -1)), "general")
                        except (ValueError, TypeError):
                            stress_type = "general"
                    if self.max_docs_per_class > 0:
                        if class_counts.get(stress_type, 0) >= self.max_docs_per_class:
                            continue
                        class_counts[stress_type] = class_counts.get(stress_type, 0) + 1
                    source = (row.get("source") or "local_dataset").strip()
                    docs.append(Document(
                        doc_id=f"local_csd_{h}",
                        content=text,
                        crop="general",
                        stress_type=stress_type,
                        region="global",
                        source=f"local_{source}",
                        metadata={"loader": "crop_stress_dataset"},
                    ))
        except Exception as e:
            logger.warning("Failed to read %s: %s", path, e)
        return docs

    def load_captions(self, csv_path: str) -> List[Document]:
        path = Path(csv_path)
        if not path.exists():
            return []
        docs: List[Document] = []
        seen: Set[str] = set()
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                for row in csv.DictReader(f):
                    text = (row.get("text") or row.get("caption") or "").strip()
                    if not text or len(text) < 20 or not _is_agriculture_text(text):
                        continue
                    h = _content_hash(text)
                    if h in seen:
                        continue
                    seen.add(h)
                    label_name = (row.get("label_name") or "").strip()
                    if label_name in STRESS_LABELS:
                        stress_type = label_name
                    else:
                        try:
                            stress_type = LABEL_TO_STRESS.get(int(row.get("label", -1)), "general")
                        except (ValueError, TypeError):
                            stress_type = "general"
                    docs.append(Document(
                        doc_id=f"local_cap_{h}",
                        content=text,
                        crop="general",
                        stress_type=stress_type,
                        region="global",
                        source="local_caption",
                        metadata={"loader": "captions_csv"},
                    ))
        except Exception as e:
            logger.warning("Failed to read %s: %s", path, e)
        return docs

    def load_all(self, data_dir: str) -> List[Document]:
        """Load all local sources with deduplication."""
        data_path = Path(data_dir)
        if not data_path.exists():
            logger.warning("Data directory not found: %s", data_path)
            return []
        all_docs: List[Document] = []
        seen: Set[str] = set()

        def _add(new_docs: List[Document]) -> None:
            for doc in new_docs:
                h = _content_hash(doc.content)
                if h not in seen:
                    seen.add(h)
                    all_docs.append(doc)

        _add(self.load_stress_text_csvs(data_dir))
        csd = data_path / "crop_stress_text_dataset.csv"
        if csd.exists():
            _add(self.load_crop_stress_dataset(str(csd)))
        cap = data_path / "captions.csv"
        if cap.exists():
            _add(self.load_captions(str(cap)))

        logger.info("LocalDatasetLoader: %d unique documents from %s", len(all_docs), data_path)
        return all_docs


class ImageCaptionLoader:
    """
    Creates Document objects from BLIP-generated image captions.
    Reads text.csv for each stress type and pairs entries with image files.
    """

    def load_image_captions(self, data_dir: str, max_per_class: int = 50) -> List[Document]:
        data_path = Path(data_dir)
        docs: List[Document] = []

        for stress_type in STRESS_LABELS:
            img_dir  = data_path / stress_type / "images"
            csv_path = data_path / stress_type / "text.csv"
            if not img_dir.exists() or not csv_path.exists():
                continue

            image_files = sorted([
                f.name for f in img_dir.iterdir()
                if f.suffix.lower() in (".jpg", ".jpeg", ".png")
            ])
            if not image_files:
                continue

            captions: List[str] = []
            try:
                with open(csv_path, encoding="utf-8", errors="replace") as f:
                    for row in csv.DictReader(f):
                        if (row.get("source") or "").strip() == "blip_caption":
                            text = (row.get("text") or "").strip()
                            if text and len(text) >= 10:
                                captions.append(text)
            except Exception as e:
                logger.warning("Failed to read captions from %s: %s", csv_path, e)
                continue

            count = min(max_per_class, len(image_files), max(1, len(captions)))
            for i in range(count):
                caption  = captions[i % len(captions)] if captions else f"{stress_type} crop stress image"
                img_name = image_files[i % len(image_files)]
                docs.append(Document(
                    doc_id=f"local_img_{stress_type}_{i:04d}",
                    content=caption,
                    crop="general",
                    stress_type=stress_type,
                    region="global",
                    source="local_image_caption",
                    metadata={"loader": "image_caption", "image_path": str(img_dir / img_name), "image_name": img_name},
                ))
        logger.info("ImageCaptionLoader: %d total image-caption documents", len(docs))
        return docs


# ============================================================================
# 8. ADVISORY GENERATOR  (Gemini API → Gemma local → offline fallback)
# ============================================================================

class LLMAdvisoryGenerator:
    """
    Generates natural-language agronomic advisories from a structured prompt.

    Priority:
      1. Gemini API  (set GOOGLE_API_KEY env var)
      2. Local Gemma (pass gemma_model_name)
      3. Offline fallback (echoes prompt summary)
    """

    def __init__(
        self,
        gemma_model_name: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        device: str = "cpu",
    ):
        self.max_new_tokens = max_new_tokens
        self.temperature    = temperature
        self.top_p          = top_p
        self._mode          = "offline"
        self._gemini_model  = None
        self._gemma_pipeline = None

        api_key = os.environ.get("GOOGLE_API_KEY")
        if api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                self._gemini_model = genai.GenerativeModel("gemini-1.5-flash")
                self._mode = "gemini"
                logger.info("LLMAdvisoryGenerator: using Gemini API.")
                return
            except ImportError:
                logger.warning("google-generativeai not installed. pip install google-generativeai")

        if gemma_model_name:
            try:
                from transformers import pipeline as hf_pipeline
                self._gemma_pipeline = hf_pipeline(
                    "text-generation",
                    model=gemma_model_name,
                    device=0 if device == "cuda" else -1,
                    torch_dtype="auto",
                )
                self._mode = "gemma"
                logger.info("LLMAdvisoryGenerator: using local Gemma (%s).", gemma_model_name)
                return
            except Exception as e:
                logger.warning("Failed to load Gemma model: %s", e)

        logger.warning("LLMAdvisoryGenerator: offline mode. Set GOOGLE_API_KEY for real advisories.")

    def generate(self, prompt: str) -> str:
        if self._mode == "gemini":
            return self._gemini_generate(prompt)
        if self._mode == "gemma":
            return self._gemma_generate(prompt)
        return self._offline_generate(prompt)

    def _gemini_generate(self, prompt: str) -> str:
        try:
            resp = self._gemini_model.generate_content(
                prompt,
                generation_config={
                    "max_output_tokens": self.max_new_tokens,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                },
            )
            return resp.text
        except Exception as e:
            logger.error("Gemini generation failed: %s", e)
            return f"[Gemini error: {e}]\n{self._offline_generate(prompt)}"

    def _gemma_generate(self, prompt: str) -> str:
        try:
            outputs = self._gemma_pipeline(
                prompt,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
            )
            generated = outputs[0]["generated_text"]
            if generated.startswith(prompt):
                generated = generated[len(prompt):].strip()
            return generated
        except Exception as e:
            logger.error("Gemma generation failed: %s", e)
            return f"[Gemma error: {e}]\n{self._offline_generate(prompt)}"

    @staticmethod
    def _offline_generate(prompt: str) -> str:
        return (
            "[OFFLINE MODE] Real LLM advisory not available.\n"
            "Set GOOGLE_API_KEY or provide a local Gemma model.\n\n"
            "Prompt received:\n" + prompt[:500] + ("..." if len(prompt) > 500 else "")
        )


# ============================================================================
# 9. CONTRASTIVE LOSS  (for fine-tuning RetrieverEncoder)
# ============================================================================

class RetrieverContrastiveLoss(nn.Module):
    """
    In-batch negatives contrastive loss (InfoNCE) for training the dual-encoder.

    Given a batch of (query, positive_doc) pairs, uses other docs in the batch
    as hard negatives. Stress-type aware: penalises same-stress-type negatives
    less harshly via a soft masking coefficient.
    """

    def __init__(self, temperature: float = 0.07, stress_neg_penalty: float = 0.5):
        super().__init__()
        self.temperature       = temperature
        self.stress_neg_penalty = stress_neg_penalty

    def forward(
        self,
        query_embs:   torch.Tensor,       # [B, D]
        pos_doc_embs: torch.Tensor,       # [B, D]
        neg_doc_embs: Optional[torch.Tensor] = None,  # [B*K, D] optional hard negs
        query_stress_types:   Optional[List[str]] = None,
        pos_doc_stress_types: Optional[List[str]] = None,
    ) -> torch.Tensor:
        B = query_embs.size(0)
        # Compute similarity matrix [B, B]
        if neg_doc_embs is not None:
            all_docs = torch.cat([pos_doc_embs, neg_doc_embs], dim=0)
        else:
            all_docs = pos_doc_embs

        sim = torch.matmul(query_embs, all_docs.T) / self.temperature  # [B, N]

        # Build soft mask — downweight same-stress negatives
        if query_stress_types and pos_doc_stress_types:
            mask = torch.zeros(B, B, device=sim.device)
            for i, qs in enumerate(query_stress_types):
                for j, ds in enumerate(pos_doc_stress_types):
                    if i != j and qs == ds:
                        mask[i, j] = -self.stress_neg_penalty * self.temperature
            if neg_doc_embs is None:
                sim[:, :B] = sim[:, :B] + mask

        labels = torch.arange(B, device=sim.device)
        return F.cross_entropy(sim, labels)


# ============================================================================
# 10. FEDERATED RAG TRAINING
# ============================================================================

@dataclass
class FedRAGConfig:
    """Configuration for federated RAG training."""
    num_rounds:       int   = 8
    local_epochs:     int   = 3
    learning_rate:    float = 2e-5
    batch_size:       int   = 16
    temperature:      float = 0.07
    top_k_retrieval:  int   = 5
    min_docs_per_farm: int  = 50
    privacy_noise_std: float = 0.0   # >0 adds Gaussian noise to shared params (DP)
    aggregation:      str   = "fedavg"  # "fedavg" | "fedprox"
    fedprox_mu:       float = 0.01


class FedRAGClient:
    """
    Federated RAG client — holds a local vector store and trains a local
    RetrieverEncoder on its own data, sharing only encoder weights.

    The vector store (KB) is NEVER shared.
    """

    def __init__(
        self,
        farm_id: int,
        documents: List[Document],
        config: FedRAGConfig,
        tokenizer=None,
        device: str = "cpu",
    ):
        self.farm_id      = farm_id
        self.config       = config
        self.tokenizer    = tokenizer
        self.device       = device
        self.vector_store = FarmVectorStore(dim=QUERY_DIM, farm_id=f"farm_{farm_id}")
        self.documents    = documents

        # Encoder is initialised externally and updated by the server each round
        self.encoder: Optional[RetrieverEncoder] = None

    def set_encoder(self, encoder: RetrieverEncoder) -> None:
        self.encoder = deepcopy(encoder).to(self.device)

    def get_encoder_state(self) -> Dict[str, torch.Tensor]:
        return {k: v.cpu() for k, v in self.encoder.state_dict().items()}

    def build_local_index(self, embeddings: np.ndarray) -> None:
        """Index pre-computed embeddings into the local vector store."""
        self.vector_store = FarmVectorStore(dim=embeddings.shape[1], farm_id=f"farm_{self.farm_id}")
        self.vector_store.add_documents(self.documents, embeddings)
        logger.info("Farm %d: indexed %d documents", self.farm_id, len(self.documents))

    def local_train(
        self,
        global_encoder_state: Dict[str, torch.Tensor],
        positives: List[Tuple[str, str]],    # list of (query_text, pos_doc_text)
    ) -> Dict[str, torch.Tensor]:
        """
        Fine-tune the encoder on local (query, positive_doc) pairs.
        Returns updated encoder state dict.

        When no tokenizer is available (offline/demo mode), returns global state
        unchanged — simulating a no-op local update.
        """
        if self.encoder is None or self.tokenizer is None or not positives:
            return global_encoder_state

        self.encoder.load_state_dict(global_encoder_state)
        self.encoder.to(self.device).train()

        optimizer  = torch.optim.AdamW(self.encoder.parameters(), lr=self.config.learning_rate)
        loss_fn    = RetrieverContrastiveLoss(temperature=self.config.temperature)
        total_loss = 0.0

        for epoch in range(self.config.local_epochs):
            np.random.shuffle(positives)
            for i in range(0, len(positives), self.config.batch_size):
                batch = positives[i : i + self.config.batch_size]
                if len(batch) < 2:
                    continue
                queries   = [p[0] for p in batch]
                pos_docs  = [p[1] for p in batch]

                q_tok  = self.tokenizer(queries,  padding=True, truncation=True, return_tensors="pt", max_length=128)
                d_tok  = self.tokenizer(pos_docs, padding=True, truncation=True, return_tensors="pt", max_length=128)

                q_tok  = {k: v.to(self.device) for k, v in q_tok.items()}
                d_tok  = {k: v.to(self.device) for k, v in d_tok.items()}

                q_emb  = self.encoder.encode_query(q_tok["input_ids"], q_tok["attention_mask"])
                d_emb  = self.encoder.encode_doc(d_tok["input_ids"],   d_tok["attention_mask"])

                # FedProx regularisation
                if self.config.aggregation == "fedprox":
                    global_params = [p for p in
                                     RetrieverEncoder.__new__(RetrieverEncoder).parameters()]
                    prox = sum(
                        (p - g.to(self.device)).pow(2).sum()
                        for p, g in zip(self.encoder.parameters(), global_params)
                    )
                    loss = loss_fn(q_emb, d_emb) + (self.config.fedprox_mu / 2) * prox
                else:
                    loss = loss_fn(q_emb, d_emb)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()

        logger.info("Farm %d: local training done, avg loss=%.4f",
                    self.farm_id, total_loss / max(1, self.config.local_epochs))
        return self.get_encoder_state()


class FedRAGServer:
    """
    Federated RAG server — aggregates encoder updates from clients each round.
    Never receives vector stores or raw documents.
    """

    def __init__(self, config: FedRAGConfig, encoder: RetrieverEncoder):
        self.config  = config
        self.encoder = encoder    # global encoder
        self.round   = 0

    def aggregate(self, client_states: List[Dict[str, torch.Tensor]]) -> None:
        """FedAvg aggregation of encoder weights."""
        if not client_states:
            return

        global_state = self.encoder.state_dict()
        n = len(client_states)

        for key in global_state:
            stacked = torch.stack([s[key].float() for s in client_states])
            avg = stacked.mean(dim=0)

            # Optional differential privacy noise
            if self.config.privacy_noise_std > 0:
                avg = avg + torch.randn_like(avg) * self.config.privacy_noise_std

            global_state[key] = avg

        self.encoder.load_state_dict(global_state)
        self.round += 1
        logger.info("Server: aggregated round %d from %d clients", self.round, n)

    def get_global_state(self) -> Dict[str, torch.Tensor]:
        return {k: v.cpu() for k, v in self.encoder.state_dict().items()}


# ============================================================================
# 11. RAG EVALUATOR  (Recall@K, MRR, NDCG, drift, fed/cent ratio)
# ============================================================================

class RAGEvaluator:
    """
    Evaluation metrics for the RAG retrieval pipeline.

    Phase 3  — Retrieval Quality:  Recall@K, MRR, NDCG@K
    Phase 5  — Federated Robustness: embedding drift, fed/cent ratio, KB coverage
    """

    def __init__(self, top_k: int = 5):
        self.top_k = top_k
        self._drift_history: List[np.ndarray] = []

    # ── Phase 3 ────────────────────────────────────────────────────────────

    def recall_at_k(
        self,
        retrieved_ids: List[List[str]],
        relevant_ids:  List[List[str]],
        k: Optional[int] = None,
    ) -> float:
        k = k or self.top_k
        recalls = [
            len(set(ret[:k]) & set(rel)) / len(rel)
            for ret, rel in zip(retrieved_ids, relevant_ids) if rel
        ]
        return float(np.mean(recalls)) if recalls else 0.0

    def mrr(
        self,
        retrieved_ids: List[List[str]],
        relevant_ids:  List[List[str]],
    ) -> float:
        rr_list = []
        for ret, rel in zip(retrieved_ids, relevant_ids):
            rel_set = set(rel)
            rr = next((1.0 / (rank + 1) for rank, did in enumerate(ret) if did in rel_set), 0.0)
            rr_list.append(rr)
        return float(np.mean(rr_list)) if rr_list else 0.0

    def ndcg_at_k(
        self,
        retrieved_ids: List[List[str]],
        relevant_ids:  List[List[str]],
        k: Optional[int] = None,
    ) -> float:
        k = k or self.top_k

        def dcg(hits: List[int]) -> float:
            return sum(h / math.log2(i + 2) for i, h in enumerate(hits))

        ndcg_list = []
        for ret, rel in zip(retrieved_ids, relevant_ids):
            if not rel:
                continue
            rel_set = set(rel)
            gains = [1 if did in rel_set else 0 for did in ret[:k]]
            ideal_dcg  = dcg(sorted(gains, reverse=True))
            actual_dcg = dcg(gains)
            ndcg_list.append(actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0)
        return float(np.mean(ndcg_list)) if ndcg_list else 0.0

    def evaluate_retrieval(
        self,
        retrieved_results: List[List[Tuple]],
        ground_truth_ids:  List[List[str]],
    ) -> Dict[str, float]:
        retrieved_ids = [[doc.doc_id for doc, _ in r] for r in retrieved_results]
        return {
            f"recall_at_{self.top_k}": self.recall_at_k(retrieved_ids, ground_truth_ids),
            "mrr":                     self.mrr(retrieved_ids, ground_truth_ids),
            f"ndcg_at_{self.top_k}":   self.ndcg_at_k(retrieved_ids, ground_truth_ids),
        }

    # ── Phase 5 ────────────────────────────────────────────────────────────

    def record_embeddings(self, query_embeddings: np.ndarray) -> None:
        self._drift_history.append(query_embeddings.copy())

    def embedding_drift(self) -> float:
        if len(self._drift_history) < 2:
            return 0.0
        drifts = []
        for prev, curr in zip(self._drift_history[:-1], self._drift_history[1:]):
            pn = prev / (np.linalg.norm(prev, axis=1, keepdims=True) + 1e-9)
            cn = curr / (np.linalg.norm(curr, axis=1, keepdims=True) + 1e-9)
            drifts.append(float(np.mean(1.0 - (pn * cn).sum(axis=1))))
        return float(np.mean(drifts))

    def fed_vs_centralised_ratio(self, fed_recall: float, centralised_recall: float) -> float:
        return fed_recall / centralised_recall if centralised_recall > 0 else 1.0

    def kb_coverage(self, retrieved_results: List[List[Tuple]]) -> float:
        seen = {doc.stress_type for results in retrieved_results for doc, _ in results}
        return len(seen & set(STRESS_LABELS)) / len(STRESS_LABELS)

    def expert_relevance_score(self, ratings: List[float]) -> Dict[str, float]:
        if not ratings:
            return {"mean_relevance": 0.0, "mean_actionability": 0.0, "count": 0}
        return {
            "mean_relevance":     float(np.mean(ratings)),
            "mean_actionability": float(np.mean(ratings)),
            "count":              len(ratings),
        }


# ============================================================================
# 12. VECTOR STORE BUILDER  (encode + index in one call)
# ============================================================================

def build_vector_store(
    docs: List[Document],
    encoder_model: str = "all-MiniLM-L6-v2",
    farm_id: str = "rag_demo",
    batch_size: int = 64,
) -> Tuple[FarmVectorStore, Any]:
    """
    Encode a list of Documents and build a FarmVectorStore.

    Uses sentence-transformers if available, otherwise a deterministic
    hash-based dummy encoder (for testing / offline runs).

    Returns:
        (FarmVectorStore, encoder)  — encoder can be called later for query encoding
    """
    if _SBERT_AVAILABLE:
        encoder = SentenceTransformer(encoder_model)
        embeddings = encoder.encode(
            [d.content for d in docs],
            show_progress_bar=True,
            batch_size=batch_size,
            convert_to_numpy=True,
        )
        dim = embeddings.shape[1]
    else:
        logger.warning("sentence-transformers not installed. Using hash-based dummy encoder.")

        # Deterministic 128-d embedding from content hash (demo only — not semantic)
        def _dummy_encode(texts: List[str]) -> np.ndarray:
            out = np.zeros((len(texts), QUERY_DIM), dtype=np.float32)
            for i, t in enumerate(texts):
                h = hashlib.sha256(t.encode()).digest()
                arr = np.frombuffer(h, dtype=np.uint8).astype(np.float32)
                # tile/truncate to QUERY_DIM
                tiled = np.tile(arr, (QUERY_DIM // len(arr)) + 1)[:QUERY_DIM]
                tiled = tiled / (np.linalg.norm(tiled) + 1e-9)
                out[i] = tiled
            return out

        class _DummyEncoder:
            def encode(self, texts, **kwargs):
                return _dummy_encode(texts)

        encoder = _DummyEncoder()
        embeddings = _dummy_encode([d.content for d in docs])
        dim = QUERY_DIM

    vs = FarmVectorStore(dim=dim, farm_id=farm_id)
    vs.add_documents(docs, embeddings)
    logger.info("Built vector store: %d vectors, dim=%d, farm=%s", len(vs), dim, farm_id)
    return vs, encoder


# ============================================================================
# 13. END-TO-END RAG DIAGNOSIS
# ============================================================================

def run_rag_diagnosis(
    data_dir: Optional[str] = None,
    demo_queries: Optional[List[str]] = None,
    llm_generator: Optional[LLMAdvisoryGenerator] = None,
    encoder_model: str = "all-MiniLM-L6-v2",
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    RAG-enhanced crop stress diagnosis using the local dataset.

    Steps:
      1. Auto-detect data directory
      2. Load local CSV + image-caption documents
      3. Build FAISS vector store (or TF-IDF fallback)
      4. Retrieve top-k passages per query
      5. Produce context-enriched diagnosis + optional LLM advisory

    Args:
        data_dir:      Path to the data/ directory (auto-detected if None).
        demo_queries:  List of symptom description strings to diagnose.
        llm_generator: Optional LLMAdvisoryGenerator for full advisory text.
        encoder_model: sentence-transformers model name (default: all-MiniLM-L6-v2).
        top_k:         Number of passages to retrieve per query.

    Returns:
        Dict with 'diagnoses', metadata, and optional evaluation metrics.
    """
    print("\n" + "=" * 90)
    print("FARMFEDERATE  —  RAG-ENHANCED CROP STRESS DIAGNOSIS")
    print("=" * 90)

    # ── 0. Resolve data directory ──────────────────────────────────────────
    if data_dir is None:
        candidates = [
            Path("/content/data"),
            Path("/content/FarmFederate/data"),
            Path("/content/drive/MyDrive/FarmFederate/data"),
            Path("/content/drive/MyDrive/data"),
            Path("data"),
        ]
        try:
            candidates.insert(0, Path(__file__).parent.parent / "data")
        except NameError:
            pass
        for p in candidates:
            try:
                if p.exists():
                    data_dir = str(p)
                    break
            except Exception:
                continue
        if data_dir is None:
            data_dir = "data"
    print(f"  [Data Dir] {Path(data_dir).resolve()}")

    # ── 1. Load documents ──────────────────────────────────────────────────
    loader  = LocalDatasetLoader(max_docs_per_class=200)
    all_docs = loader.load_all(data_dir)

    img_loader = ImageCaptionLoader()
    img_docs   = img_loader.load_image_captions(data_dir, max_per_class=50)
    all_docs.extend(img_docs)

    # ── Fallback: built-in KB when no external data found ─────────────────
    if not all_docs:
        print("  [RAG] No external data found — using built-in knowledge base")
        for entry in _BUILTIN_KB:
            all_docs.append(Document(
                doc_id=f"builtin_{entry['stress_type']}_{_content_hash(entry['content'])}",
                content=entry["content"],
                crop="general",
                stress_type=entry["stress_type"],
                region="global",
                source="builtin",
                metadata={"loader": "builtin_kb"},
            ))

    print(f"  [RAG] Loaded {len(all_docs)} documents")
    dist = Counter(d.stress_type for d in all_docs)
    for st in STRESS_LABELS:
        print(f"    {st}: {dist.get(st, 0)} docs")

    # ── 2. Chunk ───────────────────────────────────────────────────────────
    chunker = AgriculturalChunker(chunk_size=256, overlap=32)
    chunks: List[Document] = []
    for doc in all_docs:
        chunks.extend(chunker.chunk(doc))
    print(f"  [RAG] Created {len(chunks)} chunks from {len(all_docs)} documents")

    # ── 3. Build vector store (FAISS/SBERT or TF-IDF fallback) ────────────
    rag_available = False
    vs = None
    encoder = None
    _tfidf_vec    = None
    _tfidf_matrix = None
    _tfidf_corpus: List[Tuple[str, str]] = []

    if _SBERT_AVAILABLE and chunks:
        try:
            vs, encoder = build_vector_store(chunks, encoder_model=encoder_model)
            rag_available = True
            retrieval_method = "faiss"
            print(f"  [RAG] Vector store built: {len(vs)} vectors ✅")
        except Exception as e:
            logger.warning("Vector store build failed: %s", e)

    if not rag_available:
        print("  [RAG] sentence-transformers not available — using TF-IDF fallback")
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            _tfidf_corpus = [(c.content[:300], c.stress_type) for c in chunks]
            if _tfidf_corpus:
                _tfidf_vec    = TfidfVectorizer(max_features=8000, ngram_range=(1, 2), stop_words="english")
                _tfidf_matrix = _tfidf_vec.fit_transform([t for t, _ in _tfidf_corpus])
                retrieval_method = "tfidf"
                print(f"  [RAG TF-IDF] Indexed {len(_tfidf_corpus)} chunks across {len(STRESS_LABELS)} classes")
            else:
                retrieval_method = "none"
        except ImportError:
            retrieval_method = "none"
            print("  [RAG] sklearn not available — no retrieval possible")

    # ── 4. Default demo queries ────────────────────────────────────────────
    if demo_queries is None:
        demo_queries = [
            "The maize plants show severe wilting and leaf curl. Soil is dry and cracked.",
            "Tomato leaves have yellow spots with interveinal chlorosis. Stunted growth observed.",
            "Small holes visible on cabbage leaves with caterpillar feeding damage.",
            "White powdery coating spreading across grape leaves. Humidity is high.",
            "Leaf edges appear scorched and brown after the heat wave. Temperature exceeded 40°C.",
        ]

    # ── 5. Run diagnosis ───────────────────────────────────────────────────
    print("\n" + "-" * 90)
    print("RAG-ENHANCED DIAGNOSES")
    print("-" * 90)

    assembler = ContextAssembler()
    diagnoses: List[Dict[str, Any]] = []

    for i, query in enumerate(demo_queries):
        print(f"\n{'─' * 80}")
        print(f"  Query {i + 1}: \"{query[:80]}{'...' if len(query) > 80 else ''}\"")
        print(f"{'─' * 80}")

        diagnosis: Dict[str, Any] = {
            "query": query,
            "retrieved_context": [],
            "recommendation": "",
            "retrieval_method": retrieval_method,
        }

        retrieved_types: List[str] = []

        # ── FAISS / SBERT path ──
        if rag_available and vs is not None:
            try:
                q_emb = np.array(encoder.encode([query])[0], dtype=np.float32)
                retrieved = vs.search(q_emb, top_k=top_k)

                print(f"\n  Retrieved {len(retrieved)} passages:")
                for j, (doc, score) in enumerate(retrieved):
                    print(f"    [{j+1}] score={score:.3f}  [{doc.stress_type}]  {doc.content[:100]}")
                    diagnosis["retrieved_context"].append({
                        "content":     doc.content[:200],
                        "stress_type": doc.stress_type,
                        "score":       float(score),
                        "source":      doc.source,
                    })
                    if doc.stress_type != "general":
                        retrieved_types.append(doc.stress_type)

                if retrieved_types:
                    type_counts      = Counter(retrieved_types)
                    predicted_stress = type_counts.most_common(1)[0][0]
                    confidence       = type_counts.most_common(1)[0][1] / len(retrieved_types)
                else:
                    predicted_stress, confidence = "unknown", 0.0

                print(f"\n  RAG Diagnosis: {predicted_stress}  (confidence: {confidence:.0%})")

                # Build full LLM prompt and optionally generate advisory
                uniform_probs = [1.0 / NUM_CLASSES] * NUM_CLASSES
                prompt = assembler.assemble(
                    class_probs=uniform_probs,
                    retrieved_docs=retrieved,
                    farmer_text=query,
                    top_k_docs=3,
                )
                if llm_generator is not None:
                    advisory = llm_generator.generate(prompt)
                else:
                    advisory = assembler.assemble_device_template(uniform_probs)

                diagnosis["predicted_stress"] = predicted_stress
                diagnosis["confidence"]       = confidence
                diagnosis["recommendation"]   = advisory
                print(f"  Advisory: {advisory[:120]}...")

            except Exception as e:
                logger.error("FAISS retrieval failed: %s", e)
                diagnosis["error"] = str(e)

        # ── TF-IDF path ──
        elif _tfidf_vec is not None and _tfidf_matrix is not None:
            try:
                from sklearn.metrics.pairwise import cosine_similarity
                q_vec   = _tfidf_vec.transform([query])
                sims    = cosine_similarity(q_vec, _tfidf_matrix)[0]
                top_idx = sims.argsort()[-top_k:][::-1]

                print(f"\n  Retrieved {len(top_idx)} passages (TF-IDF):")
                for j, idx in enumerate(top_idx):
                    text, label = _tfidf_corpus[idx]
                    score = float(sims[idx])
                    print(f"    [{j+1}] score={score:.3f}  [{label}]  {text[:100]}")
                    diagnosis["retrieved_context"].append({
                        "content": text[:200], "stress_type": label,
                        "score": score, "source": "local_csv",
                    })
                    if score > 0.0:
                        retrieved_types.append(label)

                if retrieved_types:
                    type_counts      = Counter(retrieved_types)
                    predicted_stress = type_counts.most_common(1)[0][0]
                    confidence       = type_counts.most_common(1)[0][1] / len(retrieved_types)
                else:
                    predicted_stress, confidence = "unknown", 0.0

                context_summary = "; ".join([_tfidf_corpus[idx][0][:60] for idx in top_idx[:3]])
                advisory = (
                    f"Based on {len(retrieved_types)} retrieved observations (TF-IDF), "
                    f"the symptoms most closely match **{predicted_stress}** "
                    f"(confidence: {confidence:.0%}). Key context: {context_summary}"
                )
                print(f"\n  TF-IDF Diagnosis: {predicted_stress}  (confidence: {confidence:.0%})")
                print(f"  Advisory: {advisory[:120]}...")

                diagnosis["predicted_stress"] = predicted_stress
                diagnosis["confidence"]       = confidence
                diagnosis["recommendation"]   = advisory

            except Exception as e:
                logger.error("TF-IDF retrieval failed: %s", e)
                diagnosis["error"] = str(e)

        else:
            print("  [Info] No retrieval backend available — local data not found.")

        diagnoses.append(diagnosis)

    # ── 6. Summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("RAG DIAGNOSIS SUMMARY")
    print("=" * 90)
    predicted = [d.get("predicted_stress", "?") for d in diagnoses]
    for i, (d, pred) in enumerate(zip(diagnoses, predicted)):
        conf = d.get("confidence", 0.0)
        print(f"  Query {i+1}: {pred:<15}  confidence={conf:.0%}")
    print("=" * 90)

    return {
        "diagnoses":         diagnoses,
        "total_docs_loaded": len(all_docs),
        "total_chunks":      len(chunks),
        "rag_available":     rag_available,
        "retrieval_method":  retrieval_method,
    }


# ============================================================================
# 14. FEDERATED RAG EVALUATION PROTOCOL  (Phases 3–5)
# ============================================================================

def run_rag_evaluation(
    server: FedRAGServer,
    clients: List[FedRAGClient],
    test_queries: List[Dict[str, Any]],
    ground_truth_ids: List[List[str]],
    llm_generator: Optional[LLMAdvisoryGenerator] = None,
    centralised_recall: Optional[float] = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Full federated RAG evaluation (Phases 3–5 from the paper).

    Args:
        server:              FedRAGServer with global encoder.
        clients:             List of FedRAGClient (each has a local FarmVectorStore).
        test_queries:        List of {"query_emb": np.ndarray [128], "iot": dict, "text": str}.
        ground_truth_ids:    List of lists of relevant doc_ids per query.
        llm_generator:       Optional LLMAdvisoryGenerator for Phase 4 advisory sampling.
        centralised_recall:  Optional Recall@K from a centralised baseline (Phase 5).
        top_k:               Retrieval depth.

    Returns:
        Dict with phase3_retrieval, phase4_advisory (optional), phase5_robustness, targets.
    """
    evaluator = RAGEvaluator(top_k=top_k)
    assembler = ContextAssembler()
    results: Dict[str, Any] = {}

    # ── Phase 3: Retrieval Quality ─────────────────────────────────────────
    logger.info("Phase 3: Evaluating retrieval quality ...")
    all_retrieved: List[List[Tuple]] = []
    for query_info in test_queries:
        qemb = query_info["query_emb"].astype(np.float32)
        vs   = clients[0].vector_store                   # representative client
        retrieved = vs.search(qemb, top_k=top_k)
        all_retrieved.append(retrieved)
        evaluator.record_embeddings(qemb.reshape(1, -1))

    phase3 = evaluator.evaluate_retrieval(all_retrieved, ground_truth_ids)
    phase3["kb_coverage"] = evaluator.kb_coverage(all_retrieved)
    results["phase3_retrieval"] = phase3
    logger.info(
        "Phase 3: Recall@%d=%.3f  MRR=%.3f  NDCG@%d=%.3f  KB_coverage=%.3f",
        top_k, phase3.get(f"recall_at_{top_k}", 0),
        phase3.get("mrr", 0),
        top_k, phase3.get(f"ndcg_at_{top_k}", 0),
        phase3["kb_coverage"],
    )

    # ── Phase 4: Advisory Quality ──────────────────────────────────────────
    if llm_generator is not None:
        logger.info("Phase 4: Generating sample advisories ...")
        sample_advisories: List[str] = []
        for query_info, retrieved in zip(test_queries[:5], all_retrieved[:5]):
            prompt = assembler.assemble(
                class_probs=[1.0 / NUM_CLASSES] * NUM_CLASSES,
                retrieved_docs=retrieved,
                iot_readings=query_info.get("iot"),
                farmer_text=query_info.get("text", ""),
            )
            sample_advisories.append(llm_generator.generate(prompt))
        results["phase4_advisory"] = {
            "num_generated":  len(sample_advisories),
            "sample_advisory": sample_advisories[0] if sample_advisories else "",
            "note": (
                "Submit sample_advisories to domain experts for 1–5 relevance and "
                "actionability ratings, then call RAGEvaluator.expert_relevance_score()."
            ),
        }

    # ── Phase 5: Federated Robustness ──────────────────────────────────────
    logger.info("Phase 5: Evaluating federated robustness ...")
    drift       = evaluator.embedding_drift()
    fed_recall  = phase3.get(f"recall_at_{top_k}", 0.0)
    fed_vs_cent = (
        evaluator.fed_vs_centralised_ratio(fed_recall, centralised_recall)
        if centralised_recall is not None else None
    )
    results["phase5_robustness"] = {
        "embedding_drift_per_round": drift,
        "fed_vs_centralised_ratio":  fed_vs_cent,
        "drift_target":  "< 0.05",
        "ratio_target":  "> 0.95",
        "drift_ok":      drift < 0.05,
        "ratio_ok":      (fed_vs_cent is not None and fed_vs_cent > 0.95),
    }
    logger.info("Phase 5: drift=%.4f  fed/cent=%s", drift, f"{fed_vs_cent:.3f}" if fed_vs_cent else "N/A")

    # ── Targets check ──────────────────────────────────────────────────────
    targets = {
        f"recall_at_{top_k}": (phase3.get(f"recall_at_{top_k}", 0), 0.85, ">="),
        "mrr":                 (phase3.get("mrr", 0),                0.70, ">="),
        f"ndcg_at_{top_k}":   (phase3.get(f"ndcg_at_{top_k}", 0),  0.70, ">="),
        "drift":               (drift,                                0.05, "<"),
    }
    results["targets"] = {
        name: {"value": val, "target": tgt, "op": op, "passed": (val >= tgt if op == ">=" else val < tgt)}
        for name, (val, tgt, op) in targets.items()
    }
    return results


# ============================================================================
# 15. FULL FEDERATED RAG PIPELINE
# ============================================================================

def run_full_rag_pipeline(
    data_dir: Optional[str] = None,
    num_rounds: int = 5,
    num_clients: int = 3,
    llm_generator: Optional[LLMAdvisoryGenerator] = None,
    encoder_model: str = "all-MiniLM-L6-v2",
    config: Optional[FedRAGConfig] = None,
) -> Dict[str, Any]:
    """
    Full federated RAG pipeline:
      1. Load and partition data across simulated farm clients
      2. Build per-farm vector stores
      3. Run federated encoder training (num_rounds × FedAvg)
      4. Run RAG diagnosis on demo queries
      5. Evaluate retrieval quality (Phases 3–5)

    Returns combined results dict.
    """
    print("\n" + "=" * 90)
    print("FARMFEDERATE  —  FULL FEDERATED RAG PIPELINE")
    print("=" * 90)

    if config is None:
        config = FedRAGConfig(num_rounds=num_rounds)

    # ── 1. Load data ───────────────────────────────────────────────────────
    if data_dir is None:
        data_dir = "data"
    loader   = LocalDatasetLoader(max_docs_per_class=100)
    all_docs = loader.load_all(data_dir)
    img_docs = ImageCaptionLoader().load_image_captions(data_dir, max_per_class=20)
    all_docs.extend(img_docs)
    print(f"  Loaded {len(all_docs)} documents total")

    # ── 2. Partition across farm clients ──────────────────────────────────
    np.random.seed(42)
    indices    = np.random.permutation(len(all_docs))
    partitions = np.array_split(indices, num_clients)
    client_docs = [[all_docs[i] for i in part] for part in partitions]

    # ── 3. Build per-farm vector stores ───────────────────────────────────
    clients: List[FedRAGClient] = []
    for farm_id, docs in enumerate(client_docs):
        print(f"  Building index for farm {farm_id} ({len(docs)} docs) ...")
        client = FedRAGClient(farm_id=farm_id, documents=docs, config=config)
        if docs:
            vs, _ = build_vector_store(docs, encoder_model=encoder_model, farm_id=f"farm_{farm_id}")
            client.vector_store = vs
        clients.append(client)

    # ── 4. Global encoder (minimal stub — no backbone needed for demo) ────
    # In a full training run, inject the LLM backbone from MultiModalModel here.
    # For the standalone demo, we skip encoder fine-tuning (no tokenizer).
    server = FedRAGServer(
        config=config,
        encoder=RetrieverEncoder(text_encoder=None),  # no backbone → demo mode
    )
    global_state = server.get_global_state()
    print(f"  Federated training: {num_rounds} rounds × {num_clients} clients (demo — no tokenizer)")
    for rnd in range(num_rounds):
        # In demo mode, clients return global state unchanged (no-op local train)
        client_states = [c.local_train(global_state, []) for c in clients]
        server.aggregate(client_states)
        global_state = server.get_global_state()

    # ── 5. RAG diagnosis ───────────────────────────────────────────────────
    diag_results = run_rag_diagnosis(
        data_dir=data_dir,
        llm_generator=llm_generator,
        encoder_model=encoder_model,
    )

    # ── 6. Evaluation (Phase 3–5) — stress-type label matching ────────────
    # Build pseudo ground truth: for each query, docs with matching stress type
    demo_stress_map = {
        "The maize plants show severe wilting":        "water_stress",
        "Tomato leaves have yellow spots":             "nutrient_def",
        "Small holes visible on cabbage":              "pest_risk",
        "White powdery coating":                       "disease_risk",
        "Leaf edges appear scorched":                  "heat_stress",
    }
    test_queries: List[Dict[str, Any]] = []
    ground_truth_ids: List[List[str]] = []

    for diag in diag_results["diagnoses"]:
        q = diag["query"]
        # dummy 128-d query embedding
        qemb = np.random.randn(QUERY_DIM).astype(np.float32)
        qemb = qemb / (np.linalg.norm(qemb) + 1e-9)
        test_queries.append({"query_emb": qemb, "text": q})

        # Ground truth: IDs of docs with correct stress type
        expected_stress = next((v for k, v in demo_stress_map.items() if k.lower() in q.lower()), None)
        gt_ids = [
            doc.doc_id for doc in all_docs
            if expected_stress and doc.stress_type == expected_stress
        ][:20]
        ground_truth_ids.append(gt_ids)

    eval_results = run_rag_evaluation(
        server=server,
        clients=clients,
        test_queries=test_queries,
        ground_truth_ids=ground_truth_ids,
        llm_generator=llm_generator,
    )

    return {
        "diagnosis":  diag_results,
        "evaluation": eval_results,
        "num_clients": num_clients,
        "num_rounds":  num_rounds,
    }


# ============================================================================
# 16. RAG PLOTS
# ============================================================================

# Consistent colour palette — one colour per stress type
_STRESS_COLORS = {
    "water_stress":  "#4EA8DE",   # blue
    "nutrient_def":  "#74C69D",   # green
    "pest_risk":     "#F4A261",   # orange
    "disease_risk":  "#E76F51",   # red-orange
    "heat_stress":   "#E63946",   # red
    "general":       "#ADB5BD",   # grey
    "unknown":       "#6C757D",   # dark grey
}


def plot_rag_results(
    rag_results: Dict[str, Any],
    output_dir: str = ".",
    show: bool = True,
    prefix: str = "rag",
) -> List[str]:
    """
    Generate six diagnostic plots from run_rag_diagnosis() output.

    Plots produced:
      1. rag_retrieval_scores    — top-5 retrieval scores per query (grouped bars)
      2. rag_score_heatmap       — query × passage score heatmap
      3. rag_type_distribution   — stacked bar of retrieved stress types per query
      4. rag_confidence          — predicted stress + confidence per query
      5. rag_kb_distribution     — KB document class balance (bar chart)
      6. rag_score_boxplot       — score spread per stress type across all retrieved passages

    Args:
        rag_results:  Dict returned by run_rag_diagnosis() or run_full_rag_pipeline()["diagnosis"].
        output_dir:   Directory to save PNG files.
        show:         Whether to call plt.show() (set False in headless/Colab batch mode).
        prefix:       Filename prefix for saved plots.

    Returns:
        List of saved file paths.
    """
    try:
        import matplotlib
        matplotlib.use("Agg" if not show else matplotlib.get_backend())
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.gridspec import GridSpec
    except ImportError:
        logger.error("matplotlib not installed. pip install matplotlib")
        return []

    # ── unwrap nested result if run_full_rag_pipeline() was passed ────────
    if "diagnosis" in rag_results and "diagnoses" not in rag_results:
        rag_results = rag_results["diagnosis"]

    diagnoses        = rag_results.get("diagnoses", [])
    total_docs       = rag_results.get("total_docs_loaded", 0)
    retrieval_method = rag_results.get("retrieval_method", "unknown")

    if not diagnoses:
        logger.warning("No diagnoses found in results — nothing to plot.")
        return []

    os.makedirs(output_dir, exist_ok=True)
    saved: List[str] = []

    # Short query labels
    query_labels = [f"Q{i+1}" for i in range(len(diagnoses))]
    query_full   = [d["query"][:55] + ("…" if len(d["query"]) > 55 else "") for d in diagnoses]

    # ── Plot 1: Retrieval scores grouped bar ───────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    n_queries  = len(diagnoses)
    top_k      = max(len(d["retrieved_context"]) for d in diagnoses)
    bar_w      = 0.8 / max(top_k, 1)
    x_base     = np.arange(n_queries)

    for rank in range(top_k):
        scores     = []
        bar_colors = []
        for d in diagnoses:
            ctx = d["retrieved_context"]
            if rank < len(ctx):
                scores.append(ctx[rank]["score"])
                bar_colors.append(_STRESS_COLORS.get(ctx[rank]["stress_type"], "#ADB5BD"))
            else:
                scores.append(0.0)
                bar_colors.append("#FFFFFF")
        offset = (rank - top_k / 2 + 0.5) * bar_w
        bars = ax.bar(x_base + offset, scores, width=bar_w * 0.9, color=bar_colors, alpha=0.85,
                      label=f"Rank {rank+1}" if rank == 0 else "_nolegend_")
        for b, c in zip(bars, bar_colors):
            if c != "#FFFFFF":
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
                        f"{b.get_height():.2f}", ha="center", va="bottom", fontsize=6.5, rotation=90)

    ax.set_xticks(x_base)
    ax.set_xticklabels([f"{ql}\n{qf}" for ql, qf in zip(query_labels, query_full)], fontsize=8)
    ax.set_ylabel("Retrieval Score (cosine similarity)")
    ax.set_title(f"RAG Retrieval Scores — Top-{top_k} Passages per Query  [{retrieval_method.upper()}]",
                 fontweight="bold")
    ax.set_ylim(0, min(1.0, max(
        (ctx["score"] for d in diagnoses for ctx in d["retrieved_context"]), default=1.0
    ) * 1.25))

    legend_patches = [mpatches.Patch(color=c, label=st) for st, c in _STRESS_COLORS.items()
                      if st not in ("general", "unknown")]
    ax.legend(handles=legend_patches, title="Stress Type", bbox_to_anchor=(1.01, 1), loc="upper left",
              fontsize=8)
    ax.axhline(0.3, color="grey", linestyle="--", linewidth=0.8, alpha=0.6, label="Score=0.3 threshold")
    fig.tight_layout()
    p = os.path.join(output_dir, f"{prefix}_01_retrieval_scores.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    saved.append(p)
    if show:
        plt.show()
    plt.close(fig)
    print(f"  Saved: {p}")

    # ── Plot 2: Score heatmap ──────────────────────────────────────────────
    score_matrix = np.zeros((n_queries, top_k))
    type_matrix  = [["" for _ in range(top_k)] for _ in range(n_queries)]
    for i, d in enumerate(diagnoses):
        for j, ctx in enumerate(d["retrieved_context"]):
            score_matrix[i, j] = ctx["score"]
            type_matrix[i][j]  = ctx["stress_type"][:8]

    fig, ax = plt.subplots(figsize=(max(6, top_k * 1.5), max(4, n_queries * 0.9)))
    im = ax.imshow(score_matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Cosine Similarity Score")

    ax.set_xticks(range(top_k))
    ax.set_xticklabels([f"Rank {r+1}" for r in range(top_k)], fontsize=9)
    ax.set_yticks(range(n_queries))
    ax.set_yticklabels([f"{ql}: {qf}" for ql, qf in zip(query_labels, query_full)], fontsize=8)

    for i in range(n_queries):
        for j in range(top_k):
            s  = score_matrix[i, j]
            st = type_matrix[i][j]
            if s > 0:
                color = "white" if s > 0.5 else "black"
                ax.text(j, i, f"{s:.2f}\n{st}", ha="center", va="center",
                        fontsize=7, color=color, fontweight="bold")

    ax.set_title("RAG Score Heatmap — Query × Retrieved Passage", fontweight="bold")
    fig.tight_layout()
    p = os.path.join(output_dir, f"{prefix}_02_score_heatmap.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    saved.append(p)
    if show:
        plt.show()
    plt.close(fig)
    print(f"  Saved: {p}")

    # ── Plot 3: Retrieved stress type distribution per query ───────────────
    fig, ax = plt.subplots(figsize=(11, 5))
    type_counts_per_query = []
    for d in diagnoses:
        cnt = Counter(ctx["stress_type"] for ctx in d["retrieved_context"])
        type_counts_per_query.append(cnt)

    bottom = np.zeros(n_queries)
    for st in STRESS_LABELS + ["general"]:
        vals = np.array([tc.get(st, 0) for tc in type_counts_per_query], dtype=float)
        if vals.sum() > 0:
            bars = ax.bar(query_labels, vals, bottom=bottom,
                          color=_STRESS_COLORS.get(st, "#ADB5BD"), label=st, alpha=0.88)
            for b, v in zip(bars, vals):
                if v > 0:
                    ax.text(b.get_x() + b.get_width() / 2,
                            b.get_y() + b.get_height() / 2,
                            st[:4], ha="center", va="center", fontsize=7, color="white",
                            fontweight="bold")
            bottom += vals

    # Mark predicted stress above each bar
    for i, d in enumerate(diagnoses):
        pred = d.get("predicted_stress", "?")
        conf = d.get("confidence", 0.0)
        ax.text(i, bottom[i] + 0.05, f"→ {pred}\n({conf:.0%})",
                ha="center", va="bottom", fontsize=7.5, fontweight="bold",
                color=_STRESS_COLORS.get(pred, "black"))

    ax.set_ylim(0, max(top_k * 1.7, 1))
    ax.set_ylabel("Number of Retrieved Passages")
    ax.set_title("Retrieved Stress-Type Distribution per Query + Predicted Label",
                 fontweight="bold")
    handles, lbls = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, lbls, title="Stress Type", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    else:
        ax.text(0.5, 0.5, "No retrieved passages\n(knowledge base empty)", transform=ax.transAxes,
                ha="center", va="center", fontsize=11, color="grey", style="italic")
    ax.set_xticks(range(n_queries))
    ax.set_xticklabels([f"{ql}\n{qf}" for ql, qf in zip(query_labels, query_full)], fontsize=8)
    fig.tight_layout()
    p = os.path.join(output_dir, f"{prefix}_03_type_distribution.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    saved.append(p)
    if show:
        plt.show()
    plt.close(fig)
    print(f"  Saved: {p}")

    # ── Plot 4: Confidence per query ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4.5))
    confidences = [d.get("confidence", 0.0) for d in diagnoses]
    preds       = [d.get("predicted_stress", "unknown") for d in diagnoses]
    bar_colors  = [_STRESS_COLORS.get(p, "#ADB5BD") for p in preds]

    bars = ax.barh(query_labels[::-1], [c * 100 for c in confidences[::-1]],
                   color=bar_colors[::-1], alpha=0.85, edgecolor="white", linewidth=0.5)

    for bar, pred, conf in zip(bars, preds[::-1], confidences[::-1]):
        w = bar.get_width()
        ax.text(min(w + 1.5, 102), bar.get_y() + bar.get_height() / 2,
                f"{pred}  {conf:.0%}", va="center", ha="left", fontsize=9, fontweight="bold",
                color=_STRESS_COLORS.get(pred, "black"))

    ax.axvline(80, color="green",  linestyle="--", linewidth=1.2, alpha=0.7, label="80% threshold")
    ax.axvline(60, color="orange", linestyle="--", linewidth=1.0, alpha=0.7, label="60% threshold")
    ax.set_xlim(0, 130)
    ax.set_xlabel("Confidence (%)")
    ax.set_title("RAG Diagnosis Confidence per Query", fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")

    for i, (ql, qf) in enumerate(zip(query_labels[::-1], query_full[::-1])):
        ax.text(0.5, i, qf, va="center", ha="left", fontsize=7.5, color="#555", style="italic")

    fig.tight_layout()
    p = os.path.join(output_dir, f"{prefix}_04_confidence.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    saved.append(p)
    if show:
        plt.show()
    plt.close(fig)
    print(f"  Saved: {p}")

    # ── Plot 5: KB class balance ───────────────────────────────────────────
    # Infer from retrieved passages since we don't have per-class counts directly
    all_retrieved_types = [ctx["stress_type"] for d in diagnoses for ctx in d["retrieved_context"]]
    type_dist = Counter(all_retrieved_types)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Bar
    labels_kb = [st for st in STRESS_LABELS + ["general"] if type_dist.get(st, 0) > 0]
    counts_kb  = [type_dist[st] for st in labels_kb]
    colors_kb  = [_STRESS_COLORS.get(st, "#ADB5BD") for st in labels_kb]
    axes[0].bar(labels_kb, counts_kb, color=colors_kb, alpha=0.88, edgecolor="white")
    for i, (lbl, cnt) in enumerate(zip(labels_kb, counts_kb)):
        axes[0].text(i, cnt + 0.1, str(cnt), ha="center", va="bottom", fontsize=9, fontweight="bold")
    axes[0].set_title("Retrieved Passage Types — Frequency across All Queries", fontweight="bold")
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis="x", rotation=20)

    # Pie
    wedge_colors = [_STRESS_COLORS.get(st, "#ADB5BD") for st in labels_kb]
    if counts_kb:
        axes[1].pie(counts_kb, labels=labels_kb, colors=wedge_colors,
                    autopct="%1.0f%%", startangle=140, pctdistance=0.78,
                    wedgeprops={"edgecolor": "white", "linewidth": 1.5})
    else:
        axes[1].text(0.5, 0.5, "No data", transform=axes[1].transAxes,
                     ha="center", va="center", fontsize=13, color="grey", style="italic")
    axes[1].set_title("Distribution Share", fontweight="bold")

    fig.suptitle(f"Knowledge Base Retrieval Profile  ({total_docs} docs loaded, {retrieval_method.upper()})",
                 fontweight="bold", y=1.01)
    fig.tight_layout()
    p = os.path.join(output_dir, f"{prefix}_05_kb_distribution.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    saved.append(p)
    if show:
        plt.show()
    plt.close(fig)
    print(f"  Saved: {p}")

    # ── Plot 6: Score boxplot per retrieved stress type ────────────────────
    scores_by_type: Dict[str, List[float]] = {}
    for d in diagnoses:
        for ctx in d["retrieved_context"]:
            st = ctx["stress_type"]
            scores_by_type.setdefault(st, []).append(ctx["score"])

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_types  = sorted(scores_by_type.keys(), key=lambda x: -np.mean(scores_by_type[x]))
    plot_data   = [scores_by_type[st] for st in plot_types]
    plot_colors = [_STRESS_COLORS.get(st, "#ADB5BD") for st in plot_types]

    if plot_data:
        bp = ax.boxplot(plot_data, patch_artist=True, notch=False,
                        medianprops={"color": "white", "linewidth": 2.5},
                        whiskerprops={"linewidth": 1.2},
                        capprops={"linewidth": 1.2},
                        flierprops={"marker": "o", "markersize": 5, "alpha": 0.6})
        for patch, color in zip(bp["boxes"], plot_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.82)

        # Overlay individual points
        for i, (vals, color) in enumerate(zip(plot_data, plot_colors), start=1):
            jitter = np.random.uniform(-0.15, 0.15, size=len(vals))
            ax.scatter([i + j for j in jitter], vals, color=color, s=30, alpha=0.7, zorder=3,
                       edgecolors="white", linewidths=0.5)

        ax.set_xticks(range(1, len(plot_types) + 1))
        ax.set_xticklabels(plot_types, fontsize=9)
        ax.axhline(0.3, color="grey", linestyle="--", linewidth=0.9, alpha=0.7, label="0.3 threshold")
        ax.legend(fontsize=8)
        all_vals = [v for vals in plot_data for v in vals]
        ax.set_ylim(0, min(1.0, max(all_vals) * 1.2) if all_vals else 1.0)
    else:
        ax.text(0.5, 0.5, "No retrieved passages\n(knowledge base empty or no scores recorded)",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=11, color="grey", style="italic")
        ax.set_ylim(0, 1.0)

    ax.set_ylabel("Retrieval Score (cosine similarity)")
    ax.set_title("Score Distribution per Retrieved Stress Type", fontweight="bold")
    fig.tight_layout()
    p = os.path.join(output_dir, f"{prefix}_06_score_boxplot.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    saved.append(p)
    if show:
        plt.show()
    plt.close(fig)
    print(f"  Saved: {p}")

    print(f"\n  RAG plots saved ({len(saved)} files) → {output_dir}")
    return saved


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import argparse

    # parse_known_args silently drops unrecognised args like the Jupyter kernel's
    # "-f /root/.../kernel-xxxx.json" so the script works in both Colab and CLI.
    parser = argparse.ArgumentParser(description="FarmFederate RAG Standalone")
    parser.add_argument("--data_dir",      default=None,              help="Path to data/ directory")
    parser.add_argument("--mode",          default="diagnosis",       choices=["diagnosis", "full"],
                        help="'diagnosis' = RAG demo only; 'full' = federated pipeline + eval")
    parser.add_argument("--rounds",        type=int, default=5,       help="Federated training rounds")
    parser.add_argument("--clients",       type=int, default=3,       help="Number of farm clients")
    parser.add_argument("--encoder_model", default="all-MiniLM-L6-v2", help="sentence-transformers model")
    parser.add_argument("--gemini_key",    default=None,              help="Google API key for Gemini advisory")
    parser.add_argument("--output",        default=None,              help="Save results to JSON file")
    args, _unknown = parser.parse_known_args()

    if args.gemini_key:
        os.environ["GOOGLE_API_KEY"] = args.gemini_key

    llm_gen = LLMAdvisoryGenerator()   # will use Gemini if GOOGLE_API_KEY is set

    if args.mode == "diagnosis":
        results = run_rag_diagnosis(
            data_dir=args.data_dir,
            llm_generator=llm_gen,
            encoder_model=args.encoder_model,
        )
    else:
        results = run_full_rag_pipeline(
            data_dir=args.data_dir,
            num_rounds=args.rounds,
            num_clients=args.clients,
            llm_generator=llm_gen,
            encoder_model=args.encoder_model,
        )

    # Always generate plots
    plot_dir = args.output.replace(".json", "_plots") if args.output else "rag_plots"
    print(f"\n  Generating RAG plots → {plot_dir}/")
    plot_rag_results(results, output_dir=plot_dir, show=False)

    if args.output:
        # Remove non-serialisable objects before saving
        def _clean(obj):
            if isinstance(obj, dict):
                return {k: _clean(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_clean(v) for v in obj]
            if isinstance(obj, (np.ndarray, np.floating, np.integer)):
                return obj.tolist()
            if isinstance(obj, torch.Tensor):
                return obj.tolist()
            return obj

        with open(args.output, "w") as f:
            json.dump(_clean(results), f, indent=2)
        print(f"\n  Results saved to: {args.output}")
