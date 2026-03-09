#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
integration.py — End-to-end FarmFederate-RAG pipeline.

Components:
  KnowledgeBaseBuilder — Seeds per-farm FAISS indices with agricultural documents
  FarmFederateRAG      — Complete inference pipeline: classify → retrieve → advise
  setup_federated_rag  — Wire all components together from existing FarmFederate models
  example_inference    — Demo function: classify + retrieve + advise on one sample
"""

from __future__ import annotations

import copy
import logging
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# --- sibling imports (backend/ is on sys.path when run from there) ---
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_core import (
    Document,
    AgriculturalChunker,
    RetrieverEncoder,
    RAGQueryBuilder,
    FarmVectorStore,
    ContextAssembler,
    STRESS_TYPES,
    QUERY_DIM,
)
from federated_rag_training import (
    RetrieverContrastiveLoss,
    AdvisoryGenerator,
    FedRAGConfig,
    FedRAGClient,
    FedRAGServer,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Seed knowledge documents
# ---------------------------------------------------------------------------

_SEED_KNOWLEDGE: List[Dict] = [
    # water_stress
    {
        "content": (
            "Symptoms: Wilting, leaf rolling, grey-green discoloration, stunted growth. "
            "Treatment: Irrigate immediately, preferably at dawn or dusk. "
            "Apply mulch to reduce evaporation. Check irrigation system for blockages. "
            "Prevention: Install tensiometers at 15 cm depth; irrigate when soil moisture falls below 40% field capacity."
        ),
        "crop": "general", "stress_type": "water_stress", "region": "global",
        "source": "ICAR irrigation guide",
    },
    {
        "content": (
            "Rice water stress: Panicle initiation and flowering are most sensitive stages. "
            "Maintain 2–5 cm standing water. If AWD (alternate wetting & drying) is used, "
            "allow only up to 15 cm below soil surface before re-flooding. "
            "Apply 25 kg/ha potassium to improve drought tolerance."
        ),
        "crop": "rice", "stress_type": "water_stress", "region": "south_asia",
        "source": "IRRI water management",
    },
    # nutrient_def
    {
        "content": (
            "Nitrogen deficiency: yellowing starts at older/lower leaves, progresses upward. "
            "Apply 46 kg N/ha urea split: 50% basal, 25% at tillering, 25% at panicle initiation. "
            "Potassium deficiency: marginal leaf scorch, purple-bronze discoloration. "
            "Apply muriate of potash at 60 kg K₂O/ha."
        ),
        "crop": "rice", "stress_type": "nutrient_def", "region": "global",
        "source": "FAO fertiliser guide",
    },
    {
        "content": (
            "Wheat nutrient deficiency management: Iron deficiency shows interveinal chlorosis "
            "on young leaves. Spray ferrous sulphate 0.5% at 3 sprays, 10-day interval. "
            "Zinc deficiency: white striping on new leaves. Apply zinc sulphate 25 kg/ha soil "
            "or 0.5% foliar spray."
        ),
        "crop": "wheat", "stress_type": "nutrient_def", "region": "south_asia",
        "source": "ICAR nutrient management",
    },
    # pest_risk
    {
        "content": (
            "Stem borer (Chilo suppressalis): adult moths lay eggs on leaf blades at night. "
            "Symptoms: dead heart at vegetative stage, white ear at reproductive stage. "
            "Management: Release Trichogramma chilonis @ 50,000/ha at egg-laying peak. "
            "Chemical: Chlorpyrifos 20EC @ 1.5 L/ha if >5% dead hearts."
        ),
        "crop": "rice", "stress_type": "pest_risk", "region": "south_asia",
        "source": "IRRI IPM manual",
    },
    {
        "content": (
            "Cotton bollworm (Helicoverpa armigera): Larval feeding on squares, flowers and bolls. "
            "ETL: 1 larva / plant. Spray: Emamectin benzoate 5SG @ 0.4 g/L water. "
            "Pheromone traps: set 5/ha for monitoring. Avoid broad-spectrum insecticides to "
            "conserve natural enemies."
        ),
        "crop": "cotton", "stress_type": "pest_risk", "region": "global",
        "source": "FAO IPM guide",
    },
    # disease_risk
    {
        "content": (
            "Rice blast (Magnaporthe oryzae): diamond-shaped lesions with grey centre and brown border. "
            "Spray Tricyclazole 75WP @ 6 g/10L at booting stage. "
            "Resistant varieties: IR64, Swarna Sub1. "
            "Avoid excess nitrogen; maintain silica nutrition."
        ),
        "crop": "rice", "stress_type": "disease_risk", "region": "south_asia",
        "source": "IRRI plant pathology",
    },
    {
        "content": (
            "Late blight (Phytophthora infestans) on potato: water-soaked lesions on leaves, "
            "white sporulation on underside. "
            "Management: spray Metalaxyl + Mancozeb (Ridomil Gold) @ 2.5 g/L at first sign. "
            "Destroy infected haulms before harvest. Use certified seed tubers."
        ),
        "crop": "potato", "stress_type": "disease_risk", "region": "global",
        "source": "PlantWise factsheet",
    },
    {
        "content": (
            "Tomato bacterial wilt (Ralstonia solanacearum): wilting at midday even with adequate moisture. "
            "Cut stem test: creamy bacterial ooze. No curative treatment. "
            "Management: remove and burn infected plants, solarise soil for 6 weeks, "
            "rotate with non-solanaceous crops for 3 years."
        ),
        "crop": "tomato", "stress_type": "disease_risk", "region": "global",
        "source": "AVRDC disease guide",
    },
    # heat_stress
    {
        "content": (
            "Heat stress in wheat above 32°C during grain fill: shrivelled grain, reduced test weight. "
            "Irrigate at 10–12 day intervals during grain filling. "
            "Spray KNO₃ (1%) + salicylic acid (100 ppm) to mitigate heat damage. "
            "Prefer heat-tolerant varieties: PBW677, HI8498."
        ),
        "crop": "wheat", "stress_type": "heat_stress", "region": "south_asia",
        "source": "CIMMYT heat stress advisory",
    },
    {
        "content": (
            "Maize heat stress: temperatures >35°C during silking cause poor pollination and "
            "barren ears. Irrigate every 5–7 days during silking and grain fill. "
            "Apply potassium (K₂O 40 kg/ha) to improve heat tolerance. "
            "Use short-duration hybrid varieties to escape peak heat."
        ),
        "crop": "maize", "stress_type": "heat_stress", "region": "global",
        "source": "FAO climate-smart agriculture",
    },
]


# ---------------------------------------------------------------------------
# KnowledgeBaseBuilder
# ---------------------------------------------------------------------------

class KnowledgeBaseBuilder:
    """
    Builds per-farm FAISS knowledge bases from seed knowledge documents.

    Usage:
        builder = KnowledgeBaseBuilder(retriever, tokenizer, device)
        vector_store = builder.build_farm_kb(farm_config, extra_docs=[...])
    """

    def __init__(
        self,
        retriever: RetrieverEncoder,
        tokenizer: AutoTokenizer,
        device: str = "cpu",
        max_length: int = 128,
    ):
        self.retriever = retriever
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        self.chunker = AgriculturalChunker(chunk_size=256, overlap=32)

    @torch.no_grad()
    def _embed_documents(self, docs: List[Document]) -> np.ndarray:
        """Embed a list of documents with the retriever doc encoder."""
        self.retriever.eval()
        embeddings: List[np.ndarray] = []
        batch_size = 16
        for i in range(0, len(docs), batch_size):
            batch_docs = docs[i : i + batch_size]
            texts = [d.content for d in batch_docs]
            enc = self.tokenizer(
                texts,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids    = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)
            emb = self.retriever.encode_doc(input_ids, attention_mask)  # [B, 128]
            embeddings.append(emb.cpu().numpy())
        return np.vstack(embeddings).astype(np.float32)

    def build_farm_kb(
        self,
        farm_config: Dict[str, Any],
        extra_docs: Optional[List[Document]] = None,
    ) -> FarmVectorStore:
        """
        Builds a FarmVectorStore for one farm.

        Args:
            farm_config: {"farm_id": 0, "region": "south_asia", "crops": ["rice", "wheat"]}
            extra_docs:  additional farm-specific documents (historical logs, local guides, etc.)
        """
        farm_id = str(farm_config.get("farm_id", 0))
        farm_region = farm_config.get("region", "global")
        farm_crops = set(farm_config.get("crops", []))

        # filter seed knowledge to farm's region and crops
        relevant_raw = [
            d for d in _SEED_KNOWLEDGE
            if d["region"] in ("global", farm_region)
            and (not farm_crops or d["crop"] in farm_crops or d["crop"] == "general")
        ]
        # always include all "general" docs so every farm has full coverage
        for d in _SEED_KNOWLEDGE:
            if d["crop"] == "general" and d not in relevant_raw:
                relevant_raw.append(d)

        seed_docs = [
            Document(
                doc_id=f"farm{farm_id}_seed_{i}",
                content=raw["content"],
                crop=raw["crop"],
                stress_type=raw["stress_type"],
                region=raw["region"],
                source=raw["source"],
            )
            for i, raw in enumerate(relevant_raw)
        ]

        # chunk + combine with extra docs
        all_docs: List[Document] = []
        for doc in seed_docs:
            all_docs.extend(self.chunker.chunk(doc))
        if extra_docs:
            for doc in extra_docs:
                all_docs.extend(self.chunker.chunk(doc))

        vs = FarmVectorStore(dim=QUERY_DIM, farm_id=farm_id)
        if all_docs:
            embs = self._embed_documents(all_docs)
            vs.add_documents(all_docs, embs)

        logger.info(
            "KnowledgeBaseBuilder: farm %s KB has %d chunks (from %d seed docs)",
            farm_id, len(vs), len(seed_docs),
        )
        return vs


# ---------------------------------------------------------------------------
# FarmFederateRAG  — complete inference pipeline
# ---------------------------------------------------------------------------

class FarmFederateRAG:
    """
    End-to-end inference pipeline:
        image + text → classify → build query → retrieve → assemble → advise

    Components are wired together here; no training logic.
    """

    def __init__(
        self,
        classification_model: torch.nn.Module,
        query_builder: RAGQueryBuilder,
        vector_store: FarmVectorStore,
        context_assembler: ContextAssembler,
        tokenizer: AutoTokenizer,
        device: str = "cpu",
        top_k: int = 5,
    ):
        self.cls_model   = classification_model.to(device).eval()
        self.qb          = query_builder.to(device).eval()
        self.vs          = vector_store
        self.assembler   = context_assembler
        self.tokenizer   = tokenizer
        self.device      = device
        self.top_k       = top_k

    @torch.no_grad()
    def diagnose(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        iot_readings: Optional[Dict[str, float]] = None,
        farmer_text: str = "",
        retrieval_filters: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Full pipeline inference.

        Args:
            input_ids:        [B, L] tokenised text
            attention_mask:   [B, L]
            pixel_values:     [B, 3, 224, 224] or None
            iot_readings:     dict e.g. {"temperature": 38.2, "humidity": 45, "soil_moisture": 22}
            farmer_text:      free-text symptom description (for prompt)
            retrieval_filters: optional FAISS metadata filter e.g. {"crop": "rice"}

        Returns:
            dict with keys:
                logits            [B, 5]
                class_probs       [B, 5]
                predicted_classes list[str]
                query_emb         [B, 128] np.ndarray
                retrieved_docs    list of (Document, score)
                device_advisory   str  (on-device template)
                llm_prompt        str  (full context for server LLM)
        """
        input_ids     = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        if pixel_values is not None:
            pixel_values = pixel_values.to(self.device)

        # 1. classify
        out = self.cls_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            return_features=True,
        )
        logits    = out.logits          # [B, 5]
        h_f       = out.fused_features   # [B, 512]
        class_probs = torch.softmax(logits, dim=-1)  # [B, 5]

        # 2. build IoT tensor
        iot_vals: Optional[torch.Tensor] = None
        if iot_readings:
            iot_order = ["temperature", "humidity", "soil_moisture",
                         "nitrogen", "phosphorus", "potassium"]
            vals = [float(iot_readings.get(k, 0.0)) for k in iot_order]
            iot_vals = torch.tensor(vals, dtype=torch.float32, device=self.device).unsqueeze(0)
            iot_vals = iot_vals.expand(h_f.size(0), -1)

        # 3. build query
        query_emb_t = self.qb(h_f, class_probs, iot_vals)  # [B, 128]
        query_emb_np = query_emb_t.cpu().numpy()

        # 4. retrieve (use first sample in batch for retrieval, typical for B=1)
        retrieved = self.vs.search(
            query_emb=query_emb_np[0],
            top_k=self.top_k,
            filters=retrieval_filters,
        )

        # 5. assemble prompt
        probs_list = class_probs[0].cpu().tolist()
        predicted_classes = [
            STRESS_TYPES[i] for i, p in enumerate(probs_list) if p > 0.3
        ] or [STRESS_TYPES[int(torch.argmax(class_probs[0]).item())]]

        device_advisory = self.assembler.assemble_device_template(
            probs_list, iot_readings
        )
        llm_prompt = self.assembler.assemble(
            class_probs=probs_list,
            retrieved_docs=retrieved,
            iot_readings=iot_readings,
            farmer_text=farmer_text,
            top_k_docs=3,
        )

        return {
            "logits":            logits.cpu(),
            "class_probs":       class_probs.cpu(),
            "predicted_classes": predicted_classes,
            "query_emb":         query_emb_np,
            "retrieved_docs":    retrieved,
            "device_advisory":   device_advisory,
            "llm_prompt":        llm_prompt,
        }

    def text_diagnose(
        self,
        text: str,
        iot_readings: Optional[Dict[str, float]] = None,
        retrieval_filters: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Convenience wrapper: raw text → diagnose (no image)."""
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=160,
            padding="max_length",
            return_tensors="pt",
        )
        return self.diagnose(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            pixel_values=None,
            iot_readings=iot_readings,
            farmer_text=text,
            retrieval_filters=retrieval_filters,
        )


# ---------------------------------------------------------------------------
# setup_federated_rag
# ---------------------------------------------------------------------------

def setup_federated_rag(
    existing_classification_models: List[torch.nn.Module],
    existing_llm_encoders: Optional[List[torch.nn.Module]],
    farm_configs: List[Dict[str, Any]],
    config: Optional[FedRAGConfig] = None,
    tokenizer_name: str = "roberta-base",
) -> Tuple[FedRAGServer, List[FedRAGClient]]:
    """
    Wire all RAG components together from existing FarmFederate models.

    Args:
        existing_classification_models : one MultiModalModel per farm
        existing_llm_encoders          : text_encoder from each model (can be None;
                                         will be extracted from classification_models)
        farm_configs                   : list of {"farm_id", "region", "crops"}
        config                         : FedRAGConfig (defaults used if None)
        tokenizer_name                 : HF tokenizer for KB embedding

    Returns:
        (server, clients)
    """
    config = config or FedRAGConfig()
    num_farms = len(existing_classification_models)
    assert num_farms == len(farm_configs), "One farm_config per classification model required."

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # --- build one shared global set of components (then copy to clients) ---
    ref_model = existing_classification_models[0]
    text_enc = (
        existing_llm_encoders[0]
        if existing_llm_encoders
        else ref_model.text_encoder
    )
    hidden_size = text_enc.config.hidden_size

    global_retriever     = RetrieverEncoder(text_encoder=text_enc, hidden_size=hidden_size)
    global_query_builder = RAGQueryBuilder()
    global_generator     = AdvisoryGenerator()

    # --- build per-farm components ---
    kb_builder = KnowledgeBaseBuilder(
        retriever=global_retriever,
        tokenizer=tokenizer,
        device=config.device,
    )

    clients: List[FedRAGClient] = []
    for i, (cls_model, farm_cfg) in enumerate(
        zip(existing_classification_models, farm_configs)
    ):
        # shallow-copy global RAG components so each client starts from same weights
        client_retriever     = copy.deepcopy(global_retriever)
        client_query_builder = copy.deepcopy(global_query_builder)
        client_generator     = copy.deepcopy(global_generator)

        # wire retriever's text_encoder to this farm's cls_model backbone (weight sharing)
        farm_text_enc = (
            existing_llm_encoders[i]
            if existing_llm_encoders and i < len(existing_llm_encoders)
            else cls_model.text_encoder
        )
        client_retriever.text_encoder = farm_text_enc

        # build local KB
        vector_store = kb_builder.build_farm_kb(farm_cfg)

        client = FedRAGClient(
            farm_id=i,
            classification_model=cls_model,
            retriever=client_retriever,
            query_builder=client_query_builder,
            generator=client_generator,
            vector_store=vector_store,
            config=config,
        )
        clients.append(client)
        logger.info(
            "setup_federated_rag: client %d (farm_id=%s, crops=%s) ready with KB size=%d",
            i, farm_cfg.get("farm_id"), farm_cfg.get("crops"), len(vector_store),
        )

    # --- global server model (use ref copies; shared across clients via aggregation) ---
    server = FedRAGServer(
        global_cls_model     = copy.deepcopy(existing_classification_models[0]),
        global_retriever     = global_retriever,
        global_query_builder = global_query_builder,
        global_generator     = global_generator,
        config               = config,
    )

    return server, clients


# ---------------------------------------------------------------------------
# example_inference
# ---------------------------------------------------------------------------

def example_inference(pipeline: FarmFederateRAG) -> Dict[str, Any]:
    """
    Demo: run one inference on a synthetic rice water-stress sample.

    Returns the full diagnosis dict from FarmFederateRAG.diagnose().
    """
    result = pipeline.text_diagnose(
        text=(
            "My rice paddy leaves are yellowing and rolling inward. "
            "The plants look wilted even in the morning. Soil feels dry."
        ),
        iot_readings={
            "temperature": 38.2,
            "humidity": 45.0,
            "soil_moisture": 18.0,
            "nitrogen": 40.0,
            "phosphorus": 25.0,
            "potassium": 35.0,
        },
        retrieval_filters={"crop": "rice"},
    )

    print("=== Example Inference ===")
    print("Predicted classes:", result["predicted_classes"])
    print("\nDevice advisory:")
    print(result["device_advisory"])
    print("\nTop retrieved doc:")
    if result["retrieved_docs"]:
        doc, score = result["retrieved_docs"][0]
        print(f"  [{score:.3f}] {doc.source}: {doc.content[:120]}...")
    print("\nLLM prompt (first 400 chars):")
    print(result["llm_prompt"][:400])

    return result
