#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rag_core.py — Core RAG components for FarmFederate.

Components:
  Document             — Agricultural document dataclass with ag metadata
  AgriculturalChunker  — Domain-aware chunking (crop × stress × region)
  RetrieverEncoder     — Dual-encoder sharing LLM backbone from MultiModalModel
  RAGQueryBuilder      — Multimodal query: h_f (512) + class_probs (5) + IoT → 128
  FarmVectorStore      — FAISS index with metadata filtering (per-farm, never transmitted)
  ContextAssembler     — Builds LLM prompt from all signals
"""

from __future__ import annotations

import re
import uuid
import json
import math
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

# FAISS optional — falls back to brute-force cosine if not installed
try:
    import faiss
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False
    logging.warning(
        "faiss not installed; FarmVectorStore will use brute-force cosine similarity. "
        "Install with: pip install faiss-cpu"
    )

logger = logging.getLogger(__name__)

STRESS_TYPES = ["water_stress", "nutrient_def", "pest_risk", "disease_risk", "heat_stress"]
IOT_KEYS = ["temperature", "humidity", "soil_moisture", "nitrogen", "phosphorus", "potassium"]
IOT_DIM = len(IOT_KEYS)
QUERY_DIM = 128
FUSION_DIM = 512       # h_f output from MultiModalModel.fusion
NUM_CLASSES = 5


# ---------------------------------------------------------------------------
# Document dataclass
# ---------------------------------------------------------------------------

@dataclass
class Document:
    """Agricultural document with rich metadata for filtered retrieval."""
    doc_id: str
    content: str
    crop: str = "general"
    stress_type: str = "general"
    region: str = "global"
    source: str = "unknown"
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

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


# ---------------------------------------------------------------------------
# AgriculturalChunker
# ---------------------------------------------------------------------------

class AgriculturalChunker:
    """
    Domain-aware chunker that splits agricultural documents by crop × stress × region
    semantic units rather than arbitrary token windows.

    Strategy:
    1. Split on section headers (e.g., "Symptoms:", "Treatment:", "Prevention:").
    2. Further split large chunks at sentence boundaries up to `chunk_size` words.
    3. Tag each chunk with inherited crop / stress_type / region metadata.
    """

    SECTION_HEADERS = re.compile(
        r"(?i)(symptoms?|causes?|treatment|prevention|management|control"
        r"|diagnosis|description|background|recommendation|advisory"
        r"|field\s+observation|spray\s+schedule|dose)",
        re.MULTILINE,
    )

    def __init__(self, chunk_size: int = 256, overlap: int = 32):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, doc: Document) -> List[Document]:
        """Returns list of sub-documents, each inheriting parent metadata."""
        sections = self._split_sections(doc.content)
        chunks: List[Document] = []
        for section_text in sections:
            for i, chunk_text in enumerate(self._split_words(section_text)):
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
        # Re-attach header to the content that follows it
        out = []
        for i in range(0, len(parts) - 1, 2):
            combined = (parts[i] + "\n" + parts[i + 1] + " " + parts[i + 2]).strip() \
                       if i + 2 < len(parts) else (parts[i] + "\n" + parts[i + 1]).strip()
            out.append(combined)
        return out

    def _split_words(self, text: str) -> List[str]:
        words = text.split()
        if len(words) <= self.chunk_size:
            return [text]
        chunks = []
        start = 0
        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunks.append(" ".join(words[start:end]))
            start += self.chunk_size - self.overlap
        return chunks


# ---------------------------------------------------------------------------
# RetrieverEncoder
# ---------------------------------------------------------------------------

class RetrieverEncoder(nn.Module):
    """
    Dual-encoder for retrieval.

    Query side: shared LLM backbone (text_encoder from MultiModalModel) + projection.
    Doc side:   same backbone weights (tied) or separate projection — configurable.

    Embedding dim: QUERY_DIM (128) for both query and doc.

    The backbone is passed in (not loaded here) so that weight sharing with the
    existing MultiModalModel is automatic: gradients flow back through both uses.
    """

    def __init__(
        self,
        text_encoder: nn.Module,
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
        if tie_encoders:
            self.doc_proj = self.query_proj   # shared weights
        else:
            self.doc_proj = nn.Sequential(
                nn.Linear(hidden_size, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Linear(256, query_dim),
            )

    def _encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            cls_hidden = out.pooler_output
        else:
            cls_hidden = out.last_hidden_state[:, 0, :]
        return cls_hidden  # [B, hidden_size]

    def encode_query(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Returns L2-normalised query embeddings [B, query_dim]."""
        h = self._encode(input_ids, attention_mask)
        return F.normalize(self.query_proj(h), dim=-1)

    def encode_doc(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Returns L2-normalised doc embeddings [B, query_dim]."""
        h = self._encode(input_ids, attention_mask)
        return F.normalize(self.doc_proj(h), dim=-1)

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


# ---------------------------------------------------------------------------
# RAGQueryBuilder
# ---------------------------------------------------------------------------

class RAGQueryBuilder(nn.Module):
    """
    Builds a query embedding from multimodal signals:
        h_f        — fused visual-language feature [B, FUSION_DIM=512]
        class_probs — softmax probabilities [B, NUM_CLASSES=5]
        iot_vals    — normalised IoT readings [B, IOT_DIM=6]   (optional)

    Projects to query_dim (128) and L2-normalises.
    """

    def __init__(
        self,
        fusion_dim: int = FUSION_DIM,
        num_classes: int = NUM_CLASSES,
        iot_dim: int = IOT_DIM,
        query_dim: int = QUERY_DIM,
        dropout: float = 0.1,
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
        # Per-feature normalisation statistics (running, not learnable)
        self.register_buffer("iot_mean", torch.zeros(iot_dim))
        self.register_buffer("iot_std", torch.ones(iot_dim))

    def normalise_iot(self, iot_vals: torch.Tensor) -> torch.Tensor:
        return (iot_vals - self.iot_mean.to(iot_vals.device)) / (self.iot_std.to(iot_vals.device) + 1e-6)

    def forward(
        self,
        h_f: torch.Tensor,
        class_probs: torch.Tensor,
        iot_vals: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            h_f:          [B, 512]
            class_probs:  [B, 5]   (softmax output)
            iot_vals:     [B, 6]   or None (zero-filled)
        Returns:
            query_emb:    [B, 128] L2-normalised
        """
        B = h_f.size(0)
        if iot_vals is None:
            iot_vals = torch.zeros(B, IOT_DIM, device=h_f.device, dtype=h_f.dtype)
        else:
            iot_vals = self.normalise_iot(iot_vals.to(h_f.device, dtype=h_f.dtype))

        x = torch.cat([h_f, class_probs, iot_vals], dim=-1)  # [B, 512+5+6=523]
        return F.normalize(self.proj(x), dim=-1)              # [B, 128]


# ---------------------------------------------------------------------------
# FarmVectorStore  (per-farm FAISS index, never transmitted)
# ---------------------------------------------------------------------------

class FarmVectorStore:
    """
    Per-farm vector store backed by FAISS (or brute-force fallback).

    Documents are indexed by embedding. Supports:
    - add_documents(docs, embeddings)
    - search(query_emb, top_k, filters)  — metadata filtering before scoring

    KB documents NEVER leave the farm: this object is never serialised to the server.
    """

    def __init__(self, dim: int = QUERY_DIM, farm_id: str = "farm_0"):
        self.dim = dim
        self.farm_id = farm_id
        self.documents: List[Document] = []
        self._embeddings: Optional[np.ndarray] = None  # [N, dim]

        if _FAISS_AVAILABLE:
            self._index = faiss.IndexFlatIP(dim)   # Inner-product (L2-normed = cosine)
        else:
            self._index = None

    # ------------------------------------------------------------------ I/O

    def add_documents(self, docs: List[Document], embeddings: np.ndarray) -> None:
        """
        Args:
            docs:       list of Document objects (len N)
            embeddings: float32 array [N, dim], L2-normalised
        """
        assert len(docs) == len(embeddings), "docs and embeddings must have same length"
        embeddings = embeddings.astype(np.float32)
        # L2-normalise just in case
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9
        embeddings = embeddings / norms

        self.documents.extend(docs)
        if self._embeddings is None:
            self._embeddings = embeddings
        else:
            self._embeddings = np.vstack([self._embeddings, embeddings])

        if _FAISS_AVAILABLE:
            self._index.add(embeddings)

        logger.debug(
            "FarmVectorStore [%s]: added %d docs, total=%d",
            self.farm_id, len(docs), len(self.documents),
        )

    def search(
        self,
        query_emb: np.ndarray,
        top_k: int = 5,
        filters: Optional[Dict[str, str]] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Returns list of (Document, score) sorted by descending relevance.

        filters: e.g. {"stress_type": "disease_risk", "crop": "rice"}
        Metadata pre-filtering happens before FAISS scoring when filters are given.
        """
        if len(self.documents) == 0:
            return []

        query_emb = query_emb.astype(np.float32).reshape(1, -1)
        norm = np.linalg.norm(query_emb) + 1e-9
        query_emb = query_emb / norm

        # --- metadata pre-filter ---
        if filters:
            allowed_indices = [
                i for i, d in enumerate(self.documents)
                if self._matches_filter(d, filters)
            ]
            if not allowed_indices:
                allowed_indices = list(range(len(self.documents)))  # fallback: no filter
        else:
            allowed_indices = list(range(len(self.documents)))

        candidate_embs = self._embeddings[allowed_indices]  # [M, dim]

        # --- score ---
        scores = (candidate_embs @ query_emb.T).squeeze(-1)  # [M]

        k = min(top_k, len(allowed_indices))
        top_local = np.argpartition(-scores, k - 1)[:k]
        top_local = top_local[np.argsort(-scores[top_local])]

        return [
            (self.documents[allowed_indices[i]], float(scores[i]))
            for i in top_local
        ]

    def __len__(self) -> int:
        return len(self.documents)

    @staticmethod
    def _matches_filter(doc: Document, filters: Dict[str, str]) -> bool:
        for key, val in filters.items():
            doc_val = getattr(doc, key, None)
            if doc_val is None:
                doc_val = doc.metadata.get(key)
            if doc_val != val and val != "general" and val != "global":
                return False
        return True


# ---------------------------------------------------------------------------
# ContextAssembler
# ---------------------------------------------------------------------------

class ContextAssembler:
    """
    Builds the LLM prompt from all signals:
        - class_probs   (5 stress-class probabilities)
        - retrieved_docs (top-k from FarmVectorStore)
        - iot_readings  (dict of sensor values)
        - farmer_text   (free-text symptom description)

    Returns a structured prompt string suitable for Gemma/Gemini or any LLM.
    """

    STRESS_LABELS = STRESS_TYPES

    def assemble(
        self,
        class_probs: List[float],
        retrieved_docs: List[Tuple[Document, float]],
        iot_readings: Optional[Dict[str, float]] = None,
        farmer_text: str = "",
        top_k_docs: int = 3,
    ) -> str:
        lines: List[str] = []

        # --- diagnosis summary ---
        lines.append("=== CROP STRESS DIAGNOSIS ===")
        lines.append("Predicted stress probabilities:")
        for label, prob in zip(self.STRESS_LABELS, class_probs):
            bar = "#" * int(prob * 20)
            lines.append(f"  {label:<15} {prob:.3f}  [{bar:<20}]")

        top_stress = self.STRESS_LABELS[int(np.argmax(class_probs))]
        lines.append(f"Primary concern: {top_stress}\n")

        # --- farmer description ---
        if farmer_text:
            lines.append("=== FARMER OBSERVATION ===")
            lines.append(farmer_text.strip())
            lines.append("")

        # --- IoT sensor readings ---
        if iot_readings:
            lines.append("=== SENSOR READINGS ===")
            thresholds = {
                "temperature": (10, 40, "°C"),
                "humidity": (30, 90, "%"),
                "soil_moisture": (20, 80, "%"),
                "nitrogen": (0, 100, "ppm"),
                "phosphorus": (0, 100, "ppm"),
                "potassium": (0, 150, "ppm"),
            }
            for key, val in iot_readings.items():
                lo, hi, unit = thresholds.get(key, (None, None, ""))
                flag = ""
                if lo is not None:
                    if val < lo:
                        flag = " [LOW]"
                    elif val > hi:
                        flag = " [HIGH]"
                lines.append(f"  {key}: {val:.1f}{unit}{flag}")
            lines.append("")

        # --- retrieved knowledge ---
        docs_to_show = retrieved_docs[:top_k_docs]
        if docs_to_show:
            lines.append("=== RETRIEVED KNOWLEDGE BASE ENTRIES ===")
            for rank, (doc, score) in enumerate(docs_to_show, 1):
                lines.append(f"[{rank}] Source: {doc.source} | Crop: {doc.crop} | "
                              f"Stress: {doc.stress_type} | Score: {score:.3f}")
                lines.append(doc.content.strip())
                lines.append("")

        # --- advisory request ---
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
        top_idx = int(np.argmax(class_probs))
        top_label = self.STRESS_LABELS[top_idx]
        top_prob = class_probs[top_idx]

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
            if temp is not None and temp > 38:
                advisory += f" Current temperature {temp:.1f}°C is critically high."
            if soil is not None and soil < 25:
                advisory += f" Soil moisture {soil:.1f}% is critically low."

        return advisory
