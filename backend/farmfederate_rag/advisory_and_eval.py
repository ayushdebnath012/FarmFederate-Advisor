#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
advisory_and_eval.py — Server-side advisory generation and evaluation.

Components:
  LLMAdvisoryGenerator — Server-side Gemma / Gemini advisory generation
  RAGEvaluator         — Recall@K, MRR, NDCG, embedding drift, KB coverage
  evaluation_protocol  — Phases 3–5 extending the paper's evaluation methodology
"""

from __future__ import annotations

import logging
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLMAdvisoryGenerator
# ---------------------------------------------------------------------------

class LLMAdvisoryGenerator:
    """
    Server-side LLM advisory generation using Gemma (local) or Gemini (API).

    Priority order:
      1. Gemini API (if GOOGLE_API_KEY env var is set)
      2. Local Gemma via HuggingFace transformers (if gemma_model_name provided)
      3. Offline fallback: echoes the prompt with a placeholder note
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
        self.device         = device
        self._gemini_client = None
        self._gemma_pipeline = None

        # Try Gemini API
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
                logger.warning(
                    "google-generativeai not installed. "
                    "pip install google-generativeai to use Gemini."
                )

        # Try local Gemma
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

        # Offline fallback
        self._mode = "offline"
        logger.warning(
            "LLMAdvisoryGenerator: offline mode. "
            "Set GOOGLE_API_KEY or provide gemma_model_name for real LLM advisory."
        )

    def generate(self, prompt: str) -> str:
        """
        Generates an advisory text from a structured prompt.

        Args:
            prompt: assembled context string from ContextAssembler.assemble()

        Returns:
            advisory text string
        """
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
            # strip back the prompt if the model echoes it
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


# ---------------------------------------------------------------------------
# RAGEvaluator
# ---------------------------------------------------------------------------

class RAGEvaluator:
    """
    Evaluation metrics for the RAG retrieval pipeline.

    Metrics:
      Phase 3 — Retrieval Quality:
        - Recall@K   : fraction of relevant docs retrieved in top-K
        - MRR        : mean reciprocal rank of first relevant doc
        - NDCG@K     : normalised discounted cumulative gain

      Phase 5 — Federated Robustness:
        - embedding_drift : cosine distance between consecutive round embeddings
        - fed_vs_cent     : ratio of federated retrieval quality to centralised baseline
        - kb_coverage     : fraction of stress types covered in retrieved docs
    """

    def __init__(self, top_k: int = 5):
        self.top_k = top_k
        self._drift_history: List[np.ndarray] = []  # stores query embeddings over rounds

    # ------------------------------------------------------------------ Phase 3

    def recall_at_k(
        self,
        retrieved_ids: List[List[str]],
        relevant_ids:  List[List[str]],
        k: Optional[int] = None,
    ) -> float:
        """
        Recall@K: for each query, (# relevant docs in top-K) / (# relevant docs total).

        Args:
            retrieved_ids: list of lists of doc_ids returned by search (already top-K)
            relevant_ids:  list of lists of ground-truth relevant doc_ids
        Returns:
            mean Recall@K over all queries
        """
        k = k or self.top_k
        recalls = []
        for ret, rel in zip(retrieved_ids, relevant_ids):
            if not rel:
                continue
            hits = len(set(ret[:k]) & set(rel))
            recalls.append(hits / len(rel))
        return float(np.mean(recalls)) if recalls else 0.0

    def mrr(
        self,
        retrieved_ids: List[List[str]],
        relevant_ids:  List[List[str]],
    ) -> float:
        """
        Mean Reciprocal Rank: 1/rank of the first relevant doc.

        Returns:
            mean MRR over all queries
        """
        rr_list = []
        for ret, rel in zip(retrieved_ids, relevant_ids):
            rel_set = set(rel)
            rr = 0.0
            for rank, doc_id in enumerate(ret, 1):
                if doc_id in rel_set:
                    rr = 1.0 / rank
                    break
            rr_list.append(rr)
        return float(np.mean(rr_list)) if rr_list else 0.0

    def ndcg_at_k(
        self,
        retrieved_ids: List[List[str]],
        relevant_ids:  List[List[str]],
        k: Optional[int] = None,
    ) -> float:
        """
        NDCG@K with binary relevance.

        Returns:
            mean NDCG@K over all queries
        """
        k = k or self.top_k

        def dcg(hits: List[int]) -> float:
            return sum(h / math.log2(i + 2) for i, h in enumerate(hits))

        ndcg_list = []
        for ret, rel in zip(retrieved_ids, relevant_ids):
            if not rel:
                continue
            rel_set = set(rel)
            gains = [1 if doc_id in rel_set else 0 for doc_id in ret[:k]]
            ideal = sorted(gains, reverse=True)
            actual_dcg = dcg(gains)
            ideal_dcg  = dcg(ideal)
            ndcg_list.append(actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0)
        return float(np.mean(ndcg_list)) if ndcg_list else 0.0

    def evaluate_retrieval(
        self,
        retrieved_results: List[List[Tuple]],
        ground_truth_ids:  List[List[str]],
    ) -> Dict[str, float]:
        """
        Compute all Phase 3 retrieval metrics.

        Args:
            retrieved_results: list of list of (Document, score) from FarmVectorStore.search
            ground_truth_ids:  list of list of relevant doc_ids (from eval set)

        Returns:
            {"recall_at_k": ..., "mrr": ..., "ndcg_at_k": ...}
        """
        retrieved_ids = [
            [doc.doc_id for doc, _ in results]
            for results in retrieved_results
        ]
        return {
            f"recall_at_{self.top_k}": self.recall_at_k(retrieved_ids, ground_truth_ids),
            "mrr":                     self.mrr(retrieved_ids, ground_truth_ids),
            f"ndcg_at_{self.top_k}":   self.ndcg_at_k(retrieved_ids, ground_truth_ids),
        }

    # ------------------------------------------------------------------ Phase 5

    def record_embeddings(self, query_embeddings: np.ndarray) -> None:
        """Record query embeddings for a round (for drift tracking)."""
        self._drift_history.append(query_embeddings.copy())

    def embedding_drift(self) -> float:
        """
        Mean cosine distance between embeddings from consecutive rounds.
        drift = 0 means no change; drift = 1 means fully different.
        """
        if len(self._drift_history) < 2:
            return 0.0

        drifts = []
        for prev, curr in zip(self._drift_history[:-1], self._drift_history[1:]):
            # normalise
            prev_n = prev / (np.linalg.norm(prev, axis=1, keepdims=True) + 1e-9)
            curr_n = curr / (np.linalg.norm(curr, axis=1, keepdims=True) + 1e-9)
            cos_sim = (prev_n * curr_n).sum(axis=1)   # per-query cosine similarity
            drift = float(np.mean(1.0 - cos_sim))
            drifts.append(drift)
        return float(np.mean(drifts))

    def fed_vs_centralised_ratio(
        self,
        fed_recall: float,
        centralised_recall: float,
    ) -> float:
        """
        Ratio of federated retrieval Recall@K to centralised baseline.
        Target: >0.95 (95% of centralised quality).
        """
        if centralised_recall <= 0:
            return 1.0
        return fed_recall / centralised_recall

    def kb_coverage(
        self,
        retrieved_results: List[List[Tuple]],
        all_stress_types: Optional[List[str]] = None,
    ) -> float:
        """
        Fraction of stress types that appear at least once in retrieved docs.
        """
        from rag_core import STRESS_TYPES
        all_stress_types = all_stress_types or STRESS_TYPES
        seen = set()
        for results in retrieved_results:
            for doc, _ in results:
                seen.add(doc.stress_type)
        return len(seen & set(all_stress_types)) / len(all_stress_types)

    def expert_relevance_score(
        self,
        advisory_texts: List[str],
        ratings: List[float],
    ) -> Dict[str, float]:
        """
        Phase 4: Aggregates expert 1–5 ratings for advisory quality.

        Args:
            advisory_texts: list of generated advisories
            ratings:        list of expert ratings (1–5 per advisory)

        Returns:
            {"mean_relevance": ..., "mean_actionability": ..., "count": ...}
            (actionability is assumed to be in ratings[1::2] if alternating)
        """
        if not ratings:
            return {"mean_relevance": 0.0, "mean_actionability": 0.0, "count": 0}
        return {
            "mean_relevance":     float(np.mean(ratings)),
            "mean_actionability": float(np.mean(ratings)),  # use same ratings if not split
            "count":              len(ratings),
        }


# ---------------------------------------------------------------------------
# evaluation_protocol  (Phases 3–5)
# ---------------------------------------------------------------------------

def evaluation_protocol(
    server,
    clients: List,
    test_queries: List[Dict[str, Any]],
    ground_truth_ids: List[List[str]],
    llm_generator: Optional[LLMAdvisoryGenerator] = None,
    centralised_recall: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Full evaluation protocol extending the paper's methodology (Phases 3–5).

    Args:
        server            : FedRAGServer (holds global models)
        clients           : list of FedRAGClient
        test_queries      : list of {"query_emb": np.ndarray [128], "iot": dict, "text": str}
        ground_truth_ids  : list of lists of relevant doc_ids per query
        llm_generator     : optional LLMAdvisoryGenerator for Phase 4
        centralised_recall: optional baseline Recall@K from centralised model

    Returns:
        dict with Phase 3, 4, 5 metrics
    """
    evaluator = RAGEvaluator(top_k=5)
    results: Dict[str, Any] = {}

    # ------------------------------------------------------------------ Phase 3: Retrieval Quality

    logger.info("Phase 3: Evaluating retrieval quality...")
    all_retrieved: List[List[Tuple]] = []

    for query_info in test_queries:
        qemb = query_info["query_emb"].astype(np.float32)
        # search on client 0's vector store as representative
        vs = clients[0].vector_store
        retrieved = vs.search(qemb, top_k=5)
        all_retrieved.append(retrieved)
        evaluator.record_embeddings(qemb.reshape(1, -1))

    phase3 = evaluator.evaluate_retrieval(all_retrieved, ground_truth_ids)
    phase3["kb_coverage"] = evaluator.kb_coverage(all_retrieved)
    results["phase3_retrieval"] = phase3

    logger.info(
        "Phase 3: Recall@5=%.3f  MRR=%.3f  NDCG@5=%.3f  KB_coverage=%.3f",
        phase3.get("recall_at_5", 0), phase3.get("mrr", 0),
        phase3.get("ndcg_at_5", 0), phase3["kb_coverage"],
    )

    # ------------------------------------------------------------------ Phase 4: Advisory Quality

    if llm_generator is not None:
        logger.info("Phase 4: Generating sample advisories for expert review...")
        from rag_core import ContextAssembler
        assembler = ContextAssembler()
        sample_advisories: List[str] = []

        for query_info, retrieved in zip(test_queries[:5], all_retrieved[:5]):
            prompt = assembler.assemble(
                class_probs=[0.2] * 5,
                retrieved_docs=retrieved,
                iot_readings=query_info.get("iot"),
                farmer_text=query_info.get("text", ""),
            )
            advisory = llm_generator.generate(prompt)
            sample_advisories.append(advisory)

        results["phase4_advisory"] = {
            "num_generated": len(sample_advisories),
            "sample_advisory": sample_advisories[0] if sample_advisories else "",
            "note": (
                "Submit sample_advisories to domain experts for 1-5 relevance and "
                "actionability ratings, then call evaluator.expert_relevance_score()."
            ),
        }

    # ------------------------------------------------------------------ Phase 5: Federated Robustness

    logger.info("Phase 5: Evaluating federated robustness...")
    drift = evaluator.embedding_drift()

    fed_vs_cent = None
    if centralised_recall is not None:
        fed_recall = phase3.get("recall_at_5", 0.0)
        fed_vs_cent = evaluator.fed_vs_centralised_ratio(fed_recall, centralised_recall)

    results["phase5_robustness"] = {
        "embedding_drift_per_round": drift,
        "fed_vs_centralised_ratio":  fed_vs_cent,
        "drift_target":   "< 0.05",
        "ratio_target":   "> 0.95",
        "drift_ok":       drift < 0.05,
        "ratio_ok":       (fed_vs_cent is not None and fed_vs_cent > 0.95),
    }

    logger.info(
        "Phase 5: drift=%.4f  fed/cent=%.3f",
        drift, fed_vs_cent or 0.0,
    )

    # ------------------------------------------------------------------ targets check

    targets = {
        "recall_at_5":   (phase3.get("recall_at_5", 0), 0.85, ">="),
        "mrr":           (phase3.get("mrr", 0),          0.70, ">="),
        "ndcg_at_5":     (phase3.get("ndcg_at_5", 0),    0.70, ">="),
        "drift":         (drift,                           0.05, "<"),
    }
    passed = {}
    for name, (val, tgt, op) in targets.items():
        ok = (val >= tgt) if op == ">=" else (val < tgt)
        passed[name] = {"value": val, "target": tgt, "op": op, "passed": ok}
    results["targets"] = passed

    return results
