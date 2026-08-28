"""Measured evaluation of the Classify-Retrieve-Advise advisory module.

The paper previously reported this module as designed but unevaluated because
the retrieval corpus for that run was empty. This script supplies a corpus from
``data_final`` and measures the two stages that can be measured offline:

  Retrieve  : Sentence-BERT (all-MiniLM-L6-v2) embeddings + exact inner-product
              search. Exact search is what a FAISS ``IndexFlatIP`` computes, so
              the index choice is not a source of approximation here.
  Advise    : the top-ranked passage becomes the recommendation; it is correct
              only when it carries the query's disease class, because a
              wrong-class passage is wrong treatment advice.

Two routes are compared: retrieval over the whole advisory base, and the full
Classify-Retrieve path where a classifier first narrows the base to one class.

The corpus is template-composed, so the audit below is part of the result, not
a preamble to it: identical observations are collapsed, and queries are split
into those that contain at least one class-specific sentence and those that do
not. The latter are unanswerable from text alone and bound the attainable score.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _load_evaluator() -> Any:
    """Load RAGEvaluator by path; the package __init__ pulls unrelated modules."""
    path = ROOT / "backend" / "farmfederate_rag" / "advisory_and_eval.py"
    spec = importlib.util.spec_from_file_location("ff_advisory_eval", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.RAGEvaluator


RAGEvaluator = _load_evaluator()

DATA = ROOT / "data_final"
ANNOTATIONS = DATA / "text_data" / "annotations.csv"
SCHEMA = DATA / "label_schema.json"
OUTPUT = ROOT / "tea_results" / "advisory_retrieval_v1"

ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
SEED = 42
QUERY_FRACTION = 0.30
TOP_K = 5
BOOTSTRAP = 2000


def sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def load_unique_observations() -> tuple[list[str], list[str], list[str], dict[str, Any]]:
    """Collapse duplicate observations and record the deduplication audit."""
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    class_order = list(schema["class_order"])

    with ANNOTATIONS.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("annotations.csv is empty")

    label_of: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        label_of[row["text"].strip()].add(row["class_name"])
    conflicting = sorted(t for t, labels in label_of.items() if len(labels) > 1)
    if conflicting:
        raise ValueError(
            f"{len(conflicting)} observations carry conflicting class labels"
        )

    unique: dict[str, str] = {}
    for row in rows:
        unique.setdefault(row["text"].strip(), row["class_name"])
    unknown = sorted(set(unique.values()) - set(class_order))
    if unknown:
        raise ValueError(f"Observations reference classes outside the schema: {unknown}")

    texts = list(unique)
    labels = [unique[t] for t in texts]

    # A sentence is class-specific when it never appears under another class.
    sentence_classes: dict[str, set[str]] = defaultdict(set)
    for text, label in unique.items():
        for sentence in sentences(text):
            sentence_classes[sentence].add(label)
    generic = {s for s, cs in sentence_classes.items() if len(cs) > 1}

    audit = {
        "annotation_rows": len(rows),
        "unique_observations": len(unique),
        "duplicate_rows_removed": len(rows) - len(unique),
        "distinct_sentences": len(sentence_classes),
        "sentences_shared_across_classes": len(generic),
        "class_counts_unique": dict(Counter(labels)),
    }
    return texts, labels, class_order, {"audit": audit, "generic_sentences": generic}


def embed(texts: list[str]) -> np.ndarray:
    """Mean-pooled, L2-normalised MiniLM embeddings (the Sentence-BERT recipe)."""
    import os

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import torch
    from transformers import AutoModel, AutoTokenizer

    torch.manual_seed(SEED)
    tokenizer = AutoTokenizer.from_pretrained(ENCODER, local_files_only=True)
    model = AutoModel.from_pretrained(ENCODER, local_files_only=True)
    model.eval()

    vectors: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(texts), 64):
            batch = tokenizer(
                texts[start : start + 64],
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            hidden = model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            pooled = torch.nn.functional.normalize(pooled, dim=-1)
            vectors.append(pooled.cpu().numpy())
    matrix = np.vstack(vectors).astype(np.float64)
    norms = np.linalg.norm(matrix, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-5):
        raise ValueError("Embeddings are not unit-normalised")
    return matrix


def stratified_split(
    labels: list[str], class_order: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Hold out a fixed fraction of each class as queries; the rest is the base."""
    rng = np.random.default_rng(SEED)
    base_idx: list[int] = []
    query_idx: list[int] = []
    for name in class_order:
        members = np.array([i for i, l in enumerate(labels) if l == name])
        rng.shuffle(members)
        cut = int(round(QUERY_FRACTION * len(members)))
        query_idx.extend(members[:cut].tolist())
        base_idx.extend(members[cut:].tolist())
    return np.sort(np.array(base_idx)), np.sort(np.array(query_idx))


def bootstrap_interval(correct: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    n = len(correct)
    draws = rng.integers(0, n, size=(BOOTSTRAP, n))
    means = correct[draws].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    texts, labels, class_order, extra = load_unique_observations()
    audit = extra["audit"]
    generic = extra["generic_sentences"]
    print("audit:", json.dumps(audit, indent=2))

    embeddings = embed(texts)
    base_idx, query_idx = stratified_split(labels, class_order)
    label_array = np.array(labels)

    base_vectors = embeddings[base_idx]
    base_labels = label_array[base_idx]
    query_vectors = embeddings[query_idx]
    query_labels = label_array[query_idx]
    if set(base_idx) & set(query_idx):
        raise ValueError("Base and query splits overlap")

    # Exact inner-product search over unit vectors == cosine == FAISS IndexFlatIP.
    scores = query_vectors @ base_vectors.T
    order = np.argsort(-scores, axis=1)
    retrieved = base_labels[order]

    # Route A: retrieve over the whole advisory base.
    top1_open = retrieved[:, 0] == query_labels
    p_at_k_open = (retrieved[:, :TOP_K] == query_labels[:, None]).mean()

    # Route B: classify first, then retrieve inside the predicted class.
    from sklearn.linear_model import LogisticRegression

    classifier = LogisticRegression(max_iter=2000, C=4.0, random_state=SEED)
    classifier.fit(base_vectors, base_labels)
    predicted = classifier.predict(query_vectors)
    # Retrieval inside one class can only return that class, so the advisory is
    # correct exactly when the classify stage is correct.
    top1_routed = predicted == query_labels

    evaluator = RAGEvaluator(top_k=TOP_K)
    retrieved_ids = [[f"{c}#{r}" for r, c in enumerate(row[:TOP_K])] for row in retrieved]
    relevant_ids = [
        [f"{q}#{r}" for r in range(TOP_K)] for q in query_labels
    ]
    mrr = evaluator.mrr(retrieved_ids, relevant_ids)
    ndcg = evaluator.ndcg_at_k(retrieved_ids, relevant_ids)

    # Answerable queries carry at least one class-specific sentence.
    answerable = np.array(
        [
            any(s not in generic for s in sentences(texts[i]))
            for i in query_idx
        ]
    )
    prior = Counter(base_labels.tolist())
    chance = sum((c / len(base_labels)) ** 2 for c in prior.values())

    low_open, high_open = bootstrap_interval(top1_open.astype(float))
    low_routed, high_routed = bootstrap_interval(top1_routed.astype(float))

    per_class = {}
    for name in class_order:
        mask = query_labels == name
        per_class[name] = {
            "queries": int(mask.sum()),
            "precision_at_1_open": float(top1_open[mask].mean()),
            "precision_at_1_routed": float(top1_routed[mask].mean()),
        }

    results: dict[str, Any] = {
        "experiment": "advisory_classify_retrieve_advise",
        "dataset": "data_final",
        "encoder": ENCODER,
        "index": "exact inner product over unit vectors (FAISS IndexFlatIP equivalent)",
        "class_order": class_order,
        "seed": SEED,
        "top_k": TOP_K,
        "corpus_audit": audit,
        "advisory_base": int(len(base_idx)),
        "queries": int(len(query_idx)),
        "chance_precision_at_1": float(chance),
        # per-query top-1 similarity, kept so the score distribution can be
        # plotted per class rather than summarised away to a mean
        "per_query": [
            {"true_label": str(query_labels[i]),
             "top1_label": str(retrieved[i, 0]),
             "top1_score": float(scores[i, order[i, 0]]),
             "top1_correct": bool(top1_open[i])}
            for i in range(len(query_idx))
        ],

        "queries_with_class_specific_sentence": float(answerable.mean()),
        "retrieve_only": {
            "precision_at_1": float(top1_open.mean()),
            "precision_at_1_ci95": [low_open, high_open],
            f"precision_at_{TOP_K}": float(p_at_k_open),
            "mrr": float(mrr),
            f"ndcg_at_{TOP_K}": float(ndcg),
            "precision_at_1_with_specific_sentence": float(top1_open[answerable].mean()),
            "precision_at_1_generic_only": float(top1_open[~answerable].mean()),
        },
        "classify_then_retrieve": {
            "precision_at_1": float(top1_routed.mean()),
            "precision_at_1_ci95": [low_routed, high_routed],
            "precision_at_1_with_specific_sentence": float(top1_routed[answerable].mean()),
            "precision_at_1_generic_only": float(top1_routed[~answerable].mean()),
        },
        "per_class": per_class,
        "not_evaluated": [
            "fusion-feature re-ranking (needs a fusion model on these five classes)",
            "OBB localization overlay",
            "agronomic correctness of the advice text itself",
        ],
    }

    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "advisory_retrieval_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in results.items() if k != "corpus_audit"}, indent=2))
    print("wrote", OUTPUT / "advisory_retrieval_results.json")


if __name__ == "__main__":
    main()
