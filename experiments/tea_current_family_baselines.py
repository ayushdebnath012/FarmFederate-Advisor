"""Evaluate genuine model-family baselines on the current tea split.

This experiment does not train or modify the proposed FarmFederate model.
It adds a frozen DistilBERT text-only baseline to the exact source-grouped
split used by the audited v6 checkpoint, verifies the existing genuine
ViT-tiny specialist on the same support, and carries forward the unchanged
exploratory v6 VLM and proposed predictions.

The DistilBERT classifier is selected using validation only:

1. Fit the target-shortcut vocabulary on training annotations only.
2. Apply the existing deterministic sparse-note mask at a fixed 50% keep rate.
3. Extract frozen ``distilbert-base-uncased`` masked-mean embeddings.
4. Select a ridge penalty, representation, and class-weight power on validation.
5. Extract and evaluate the fixed internal-test embeddings only after selection.

The script fails closed if any validation/test label sequence differs across
the regenerated split, ViT artifact, VLM artifact, or proposed-model artifact.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from transformers import AutoModel, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tea_train as tea  # noqa: E402
from modality_hierarchy_benchmark import (  # noqa: E402
    cache_partition as cache_v6_partition,
    predict_conditions as predict_v6_conditions,
    sparse_attention,
)
from vision_tta import load_model as load_v6_model  # noqa: E402


NUM_CLASSES = tea.NUM_CLASSES
DEFAULT_BASE_RESULTS = (
    ROOT
    / "tea_results"
    / "multimodal_v6_vision_full_20260727"
    / "complete_results.json"
)
DEFAULT_VIT_RESULTS = (
    DEFAULT_BASE_RESULTS.parent
    / "vit_specialist"
    / "vit_visual_specialist_results.json"
)
DEFAULT_HIERARCHY_RESULTS = (
    DEFAULT_BASE_RESULTS.parent
    / "modality_hierarchy_exploratory"
    / "modality_hierarchy_results.json"
)
DEFAULT_OUTPUT_DIR = (
    ROOT / "tea_results" / "current_family_baselines_20260728"
)

RIDGE_ALPHAS = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)
CLASS_WEIGHT_POWERS = (0.0, 0.25, 0.5, 0.75, 1.0)
REPRESENTATIONS = ("standardized", "l2", "raw")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sequence_digest(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def sample_id_strings(
    full_dataset: tea.TeaOBBDataset, indices: Sequence[int]
) -> List[str]:
    return [
        f"{full_dataset.sample_ids[index][0]}::{full_dataset.sample_ids[index][1]}"
        for index in indices
    ]


def annotation_frame_for_indices(
    annotations: pd.DataFrame,
    full_dataset: tea.TeaOBBDataset,
    indices: Sequence[int],
) -> pd.DataFrame:
    """Return annotation rows in exact OBB-dataset order."""

    by_key = {}
    for _, row in annotations.iterrows():
        key = (str(row["image_file"]), int(row["box_idx"]))
        if key in by_key:
            raise RuntimeError(f"Duplicate annotation key after loading: {key}")
        by_key[key] = row

    ordered_rows = []
    for index in indices:
        key = full_dataset.sample_ids[index]
        if key not in by_key:
            raise RuntimeError(f"Missing exact annotation for crop {key}")
        row = by_key[key].copy()
        annotation_label = int(row["labels"][0])
        image_label = int(full_dataset.labels[index])
        if annotation_label != image_label:
            raise RuntimeError(
                "Annotation/image label mismatch for "
                f"{key}: text={annotation_label}, image={image_label}"
            )
        ordered_rows.append(row)
    return pd.DataFrame(ordered_rows).reset_index(drop=True)


def build_current_split(base_payload: dict):
    config = base_payload["config"]
    data_dir = Path(config["data_dir"])
    full_dataset = tea.TeaOBBDataset(
        str(data_dir / "images"),
        str(data_dir / "labels"),
        transform=tea.get_transforms(train=False),
        crop_padding=float(config["crop_padding"]),
    )
    train_indices, validation_indices, test_indices = (
        tea.grouped_train_val_test_split(
            full_dataset.labels,
            full_dataset.groups,
            float(config["val_split"]),
            float(config["test_split"]),
            int(config["seed"]),
        )
    )
    annotations = tea.load_annotations_csv(config["annotations"])
    train_text = annotation_frame_for_indices(
        annotations, full_dataset, train_indices
    )
    validation_text = annotation_frame_for_indices(
        annotations, full_dataset, validation_indices
    )
    test_text = annotation_frame_for_indices(
        annotations, full_dataset, test_indices
    )
    blocked_tokens = tea.fit_label_leakage_vocabulary(
        train_text,
        min_count=int(config["leakage_token_min_count"]),
        purity_threshold=float(config["leakage_token_purity"]),
    )
    sanitized = {
        "train": tea.sanitize_annotation_text(train_text, blocked_tokens),
        "validation": tea.sanitize_annotation_text(
            validation_text, blocked_tokens
        ),
        "test": tea.sanitize_annotation_text(test_text, blocked_tokens),
    }
    indices = {
        "train": train_indices,
        "validation": validation_indices,
        "test": test_indices,
    }
    return full_dataset, indices, sanitized, blocked_tokens


def labels_for(
    full_dataset: tea.TeaOBBDataset, indices: Sequence[int]
) -> np.ndarray:
    return np.asarray(
        [int(full_dataset.labels[index]) for index in indices],
        dtype=np.int64,
    )


def class_counts(labels: Sequence[int]) -> Dict[str, int]:
    counts = Counter(int(label) for label in labels)
    return {
        tea.TEA_DISEASE_LABELS[class_id]: int(counts.get(class_id, 0))
        for class_id in range(NUM_CLASSES)
    }


def macro_f1(
    labels: np.ndarray, predictions: np.ndarray
) -> Tuple[float, List[float]]:
    per_class = []
    for class_id in range(NUM_CLASSES):
        true_positive = int(
            np.sum((labels == class_id) & (predictions == class_id))
        )
        false_positive = int(
            np.sum((labels != class_id) & (predictions == class_id))
        )
        false_negative = int(
            np.sum((labels == class_id) & (predictions != class_id))
        )
        denominator = (
            2 * true_positive + false_positive + false_negative
        )
        per_class.append(
            0.0
            if denominator == 0
            else float(2 * true_positive / denominator)
        )
    return float(np.mean(per_class)), per_class


def classification_metrics(
    labels: Sequence[int],
    predictions: Sequence[int],
) -> dict:
    labels_array = np.asarray(labels, dtype=np.int64)
    predictions_array = np.asarray(predictions, dtype=np.int64)
    if labels_array.shape != predictions_array.shape:
        raise ValueError(
            "Labels and predictions must have identical shape: "
            f"{labels_array.shape} != {predictions_array.shape}"
        )
    f1, per_class = macro_f1(labels_array, predictions_array)
    correct = int(np.sum(labels_array == predictions_array))
    accuracy = float(correct / len(labels_array))
    matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for actual, predicted in zip(labels_array, predictions_array):
        matrix[int(actual), int(predicted)] += 1
    return {
        "accuracy": accuracy,
        "macro_f1": f1,
        "selection_score": float(0.60 * f1 + 0.40 * accuracy),
        "correct": correct,
        "per_class_f1": per_class,
        "confusion_matrix": matrix.tolist(),
        "predictions": predictions_array.tolist(),
        "labels": labels_array.tolist(),
    }


def tokenize_sparse_partition(
    tokenizer,
    frame: pd.DataFrame,
    sample_ids: Sequence[Tuple[str, int]],
    max_length: int,
    keep_rate: float,
    sparsity_seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    encoded = tokenizer(
        frame["text"].astype(str).tolist(),
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].cpu()
    original_mask = encoded["attention_mask"].cpu()
    sparse_mask = sparse_attention(
        {
            "input_ids": input_ids,
            "attention_mask": original_mask,
            "sample_ids": list(sample_ids),
        },
        keep_rate,
        sparsity_seed,
    )
    special_ids = {
        int(value)
        for value in (
            tokenizer.pad_token_id,
            tokenizer.cls_token_id,
            tokenizer.sep_token_id,
        )
        if value is not None
    }
    content = torch.ones_like(original_mask, dtype=torch.bool)
    for token_id in special_ids:
        content &= input_ids.ne(token_id)
    content &= original_mask.bool()
    original_content_tokens = int(content.sum())
    retained_content_tokens = int(
        (content & sparse_mask.bool()).sum()
    )
    audit = {
        "nominal_keep_rate": float(keep_rate),
        "original_content_tokens": original_content_tokens,
        "retained_content_tokens": retained_content_tokens,
        "realized_keep_rate": float(
            retained_content_tokens / max(original_content_tokens, 1)
        ),
        "sparsity_seed": int(sparsity_seed),
        "hash_rule": (
            "blake2b(seed,image_file,box_idx,token_position,token_id), "
            "person=FieldTxt"
        ),
    }
    return input_ids, sparse_mask, audit


def extract_frozen_embeddings(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    embeddings = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(input_ids), batch_size):
            end = start + batch_size
            ids = input_ids[start:end].to(device)
            mask = attention_mask[start:end].to(device)
            output = model(input_ids=ids, attention_mask=mask)
            tokens = output.last_hidden_state
            weights = mask.to(tokens.dtype).unsqueeze(-1)
            pooled = (
                (tokens * weights).sum(dim=1)
                / weights.sum(dim=1).clamp_min(1.0)
            )
            embeddings.append(pooled.cpu().numpy())
    return np.concatenate(embeddings).astype(np.float64, copy=False)


def fit_representation(
    train_features: np.ndarray, representation: str
) -> dict:
    if representation == "standardized":
        mean = train_features.mean(axis=0, keepdims=True)
        scale = train_features.std(axis=0, keepdims=True)
        scale = np.maximum(scale, 1e-6)
        return {"name": representation, "mean": mean, "scale": scale}
    if representation in {"l2", "raw"}:
        return {"name": representation}
    raise ValueError(f"Unsupported representation: {representation}")


def apply_representation(features: np.ndarray, state: dict) -> np.ndarray:
    if state["name"] == "standardized":
        return (features - state["mean"]) / state["scale"]
    if state["name"] == "l2":
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        return features / np.maximum(norms, 1e-8)
    if state["name"] == "raw":
        return features
    raise ValueError(f"Unsupported representation: {state['name']}")


def sample_weights_for(
    labels: np.ndarray, class_weight_power: float
) -> Tuple[np.ndarray, List[float]]:
    counts = np.bincount(labels, minlength=NUM_CLASSES).clip(1)
    class_weights = counts.astype(np.float64) ** (-class_weight_power)
    class_weights /= class_weights.mean()
    sample_weights = class_weights[labels]
    sample_weights /= sample_weights.mean()
    return sample_weights, class_weights.tolist()


def fit_ridge_classifier(
    features: np.ndarray,
    labels: np.ndarray,
    alpha: float,
    class_weight_power: float,
) -> Tuple[Ridge, List[float]]:
    sample_weights, class_weights = sample_weights_for(
        labels, class_weight_power
    )
    targets = np.eye(NUM_CLASSES, dtype=np.float64)[labels]
    classifier = Ridge(
        alpha=float(alpha),
        fit_intercept=True,
        solver="auto",
    )
    classifier.fit(features, targets, sample_weight=sample_weights)
    return classifier, class_weights


def predict_ridge(classifier: Ridge, features: np.ndarray) -> np.ndarray:
    scores = np.asarray(classifier.predict(features), dtype=np.float64)
    if scores.shape != (len(features), NUM_CLASSES):
        raise RuntimeError(
            f"Unexpected ridge score shape: {scores.shape}"
        )
    return scores.argmax(axis=1).astype(np.int64)


def select_text_classifier(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
) -> Tuple[dict, List[dict]]:
    candidates = []
    for representation_index, representation in enumerate(REPRESENTATIONS):
        representation_state = fit_representation(
            train_features, representation
        )
        represented_train = apply_representation(
            train_features, representation_state
        )
        represented_validation = apply_representation(
            validation_features, representation_state
        )
        for alpha in RIDGE_ALPHAS:
            for class_weight_power in CLASS_WEIGHT_POWERS:
                classifier, class_weights = fit_ridge_classifier(
                    represented_train,
                    train_labels,
                    alpha,
                    class_weight_power,
                )
                predictions = predict_ridge(
                    classifier, represented_validation
                )
                result = classification_metrics(
                    validation_labels, predictions
                )
                candidates.append(
                    {
                        "representation": representation,
                        "representation_index": representation_index,
                        "alpha": float(alpha),
                        "class_weight_power": float(class_weight_power),
                        "class_weights": class_weights,
                        **result,
                    }
                )
    selected = max(
        candidates,
        key=lambda candidate: (
            candidate["selection_score"],
            candidate["macro_f1"],
            candidate["accuracy"],
            -candidate["representation_index"],
            -abs(np.log10(candidate["alpha"])),
            -abs(candidate["class_weight_power"] - 0.5),
        ),
    )
    return selected, candidates


def require_identical_labels(
    expected: np.ndarray,
    observed: Sequence[int],
    description: str,
) -> None:
    observed_array = np.asarray(observed, dtype=np.int64)
    if not np.array_equal(expected, observed_array):
        mismatches = np.flatnonzero(expected != observed_array)
        first = int(mismatches[0]) if len(mismatches) else None
        raise RuntimeError(
            f"{description} label/order mismatch; "
            f"expected shape={expected.shape}, observed={observed_array.shape}, "
            f"first mismatch={first}"
        )


def compact_validation_record(record: dict) -> dict:
    return {
        key: record[key]
        for key in (
            "accuracy",
            "macro_f1",
            "selection_score",
            "correct",
            "per_class_f1",
            "confusion_matrix",
            "predictions",
            "labels",
        )
        if key in record
    }


def make_row(
    system_key: str,
    display_name: str,
    architecture: str,
    support: dict,
    metrics: dict,
) -> dict:
    return {
        "system_key": system_key,
        "display_name": display_name,
        "architecture": architecture,
        "support": support,
        "accuracy": float(metrics["accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "correct": int(metrics["correct"]),
        "predictions": [int(value) for value in metrics["predictions"]],
        "labels": [int(value) for value in metrics["labels"]],
        "per_class_f1": [
            float(value) for value in metrics["per_class_f1"]
        ],
        "confusion_matrix": metrics["confusion_matrix"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_results", type=Path, default=DEFAULT_BASE_RESULTS
    )
    parser.add_argument(
        "--vit_results", type=Path, default=DEFAULT_VIT_RESULTS
    )
    parser.add_argument(
        "--hierarchy_results",
        type=Path,
        default=DEFAULT_HIERARCHY_RESULTS,
    )
    parser.add_argument(
        "--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR
    )
    parser.add_argument(
        "--model_name", default="distilbert-base-uncased"
    )
    parser.add_argument("--keep_rate", type=float, default=0.50)
    parser.add_argument("--sparsity_seed", type=int, default=20260727)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not (0.0 < args.keep_rate <= 1.0):
        raise ValueError("--keep_rate must be in (0, 1]")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        device = torch.device(args.device)

    base_results_path = args.base_results.resolve()
    vit_results_path = args.vit_results.resolve()
    hierarchy_results_path = args.hierarchy_results.resolve()
    output_dir = args.output_dir.resolve()
    base_payload = read_json(base_results_path)
    full_dataset, indices, text_frames, blocked_tokens = (
        build_current_split(base_payload)
    )

    labels = {
        partition: labels_for(full_dataset, partition_indices)
        for partition, partition_indices in indices.items()
    }
    sample_ids = {
        partition: [
            full_dataset.sample_ids[index]
            for index in partition_indices
        ]
        for partition, partition_indices in indices.items()
    }
    sample_id_text = {
        partition: sample_id_strings(full_dataset, partition_indices)
        for partition, partition_indices in indices.items()
    }

    expected_split_counts = {
        "train": 222,
        "validation": 74,
        "test": 75,
    }
    actual_split_counts = {
        partition: len(partition_indices)
        for partition, partition_indices in indices.items()
    }
    if actual_split_counts != expected_split_counts:
        raise RuntimeError(
            "Current split no longer matches audited 222/74/75 support: "
            f"{actual_split_counts}"
        )
    partition_groups = {
        partition: {
            full_dataset.groups[index]
            for index in partition_indices
        }
        for partition, partition_indices in indices.items()
    }
    group_overlap = {
        "train_validation": len(
            partition_groups["train"] & partition_groups["validation"]
        ),
        "train_test": len(
            partition_groups["train"] & partition_groups["test"]
        ),
        "validation_test": len(
            partition_groups["validation"] & partition_groups["test"]
        ),
    }
    if any(group_overlap.values()):
        raise RuntimeError(
            f"Source-group leakage detected: {group_overlap}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        local_files_only=True,
        use_fast=True,
    )
    model = AutoModel.from_pretrained(
        args.model_name,
        local_files_only=True,
    ).to(device)
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()
    encoder_parameters = int(
        sum(parameter.numel() for parameter in model.parameters())
    )
    max_length = int(base_payload["config"]["max_seq_len"])

    tokenized = {}
    sparse_audit = {}
    # Test tokenization is deterministic preprocessing, but test embeddings and
    # predictions are intentionally deferred until after validation selection.
    for partition in ("train", "validation"):
        (
            input_ids,
            attention_mask,
            sparse_audit[partition],
        ) = tokenize_sparse_partition(
            tokenizer,
            text_frames[partition],
            sample_ids[partition],
            max_length,
            args.keep_rate,
            args.sparsity_seed,
        )
        tokenized[partition] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    train_features = extract_frozen_embeddings(
        model,
        tokenized["train"]["input_ids"],
        tokenized["train"]["attention_mask"],
        device,
        args.batch_size,
    )
    validation_features = extract_frozen_embeddings(
        model,
        tokenized["validation"]["input_ids"],
        tokenized["validation"]["attention_mask"],
        device,
        args.batch_size,
    )
    selected_text, candidates = select_text_classifier(
        train_features,
        labels["train"],
        validation_features,
        labels["validation"],
    )

    # Freeze the full text pipeline after validation selection.
    representation_state = fit_representation(
        train_features, selected_text["representation"]
    )
    represented_train = apply_representation(
        train_features, representation_state
    )
    text_classifier, selected_class_weights = fit_ridge_classifier(
        represented_train,
        labels["train"],
        selected_text["alpha"],
        selected_text["class_weight_power"],
    )

    (
        test_input_ids,
        test_attention_mask,
        sparse_audit["test"],
    ) = tokenize_sparse_partition(
        tokenizer,
        text_frames["test"],
        sample_ids["test"],
        max_length,
        args.keep_rate,
        args.sparsity_seed,
    )
    test_features = extract_frozen_embeddings(
        model,
        test_input_ids,
        test_attention_mask,
        device,
        args.batch_size,
    )
    represented_test = apply_representation(
        test_features, representation_state
    )
    text_test_predictions = predict_ridge(
        text_classifier, represented_test
    )
    text_test_metrics = classification_metrics(
        labels["test"], text_test_predictions
    )

    # Load all fixed-test comparison artifacts only after the DistilBERT
    # classifier and representation have been selected on validation.
    vit_payload = read_json(vit_results_path)
    hierarchy_payload = read_json(hierarchy_results_path)
    if abs(float(hierarchy_payload["selected_validation"]["keep_rate"])
           - args.keep_rate) > 1e-12:
        raise RuntimeError(
            "Hierarchy artifact keep rate differs from requested sparse-note "
            f"rate: {hierarchy_payload['selected_validation']['keep_rate']} "
            f"!= {args.keep_rate}"
        )

    hierarchy_validation = hierarchy_payload["selected_validation"]
    hierarchy_test = hierarchy_payload["fixed_test"]
    require_identical_labels(
        labels["validation"],
        vit_payload["selected_validation"]["labels"],
        "ViT validation",
    )
    require_identical_labels(
        labels["test"],
        vit_payload["fixed_test"]["labels"],
        "ViT test",
    )
    for key in (
        "text_only",
        "image_only",
        "raw_cross_attention",
        "fusion",
        "proposed",
    ):
        require_identical_labels(
            labels["validation"],
            hierarchy_validation[key]["labels"],
            f"Hierarchy validation {key}",
        )
    for key in (
        "text_only",
        "image_only",
        "text_image_fusion",
        "proposed_multimodal",
    ):
        require_identical_labels(
            labels["test"],
            hierarchy_test[key]["labels"],
            f"Hierarchy test {key}",
        )

    # Recover the true raw paired v6 cross-attention output. The exploratory
    # artifact stores the post-hoc text_image_fusion gate but not the raw test
    # predictions, so those predictions are recomputed from the unchanged
    # validation-selected checkpoint on the identical sparse-note support.
    del model
    gc.collect()
    v6_checkpoint_path = (
        base_results_path.parent / "models" / "best_vlm.pt"
    )
    v6_model, _ = load_v6_model(v6_checkpoint_path, device)
    v6_args = argparse.Namespace(
        data_dir=base_payload["config"]["data_dir"],
        crop_padding=float(base_payload["config"]["crop_padding"]),
        max_seq_len=max_length,
        batch_size=int(args.batch_size),
        seed=int(base_payload["config"]["seed"]),
        sparsity_seed=int(args.sparsity_seed),
    )
    v6_test_cache = cache_v6_partition(
        v6_model,
        v6_args,
        full_dataset,
        indices["test"],
        text_frames["test"],
        device,
    )
    require_identical_labels(
        labels["test"],
        v6_test_cache["labels"],
        "Recomputed raw-v6 test",
    )
    v6_test_outputs = predict_v6_conditions(
        v6_model,
        v6_test_cache,
        args.keep_rate,
        v6_args,
        device,
    )
    recomputed_v6_sparse_text_predictions = v6_test_outputs[
        "text_only"
    ].argmax(axis=1)
    require_identical_labels(
        np.asarray(
            hierarchy_test["text_only"]["predictions"], dtype=np.int64
        ),
        recomputed_v6_sparse_text_predictions,
        "Recomputed v6 sparse-text predictions",
    )
    raw_vlm_test_metrics = classification_metrics(
        labels["test"],
        v6_test_outputs["fusion"].argmax(axis=1),
    )
    del v6_model, v6_test_cache, v6_test_outputs
    gc.collect()

    vit_test_metrics = classification_metrics(
        labels["test"],
        vit_payload["fixed_test"]["predictions"],
    )
    late_fusion_test_metrics = classification_metrics(
        labels["test"],
        hierarchy_test["text_image_fusion"]["predictions"],
    )
    proposed_test_metrics = classification_metrics(
        labels["test"],
        hierarchy_test["proposed_multimodal"]["predictions"],
    )
    for name, recomputed, stored in (
        (
            "ViT",
            vit_test_metrics,
            vit_payload["fixed_test"],
        ),
        (
            "late-fusion VLM diagnostic",
            late_fusion_test_metrics,
            hierarchy_test["text_image_fusion"],
        ),
        (
            "proposed",
            proposed_test_metrics,
            hierarchy_test["proposed_multimodal"],
        ),
    ):
        if (
            abs(recomputed["accuracy"] - float(stored["accuracy"])) > 1e-12
            or abs(
                recomputed["macro_f1"] - float(stored["macro_f1"])
            ) > 1e-12
        ):
            raise RuntimeError(
                f"Stored {name} metrics do not match recomputed predictions"
            )

    test_support = {
        "partition": "fixed_internal_test",
        "crops": int(len(labels["test"])),
        "source_images": int(len(partition_groups["test"])),
        "class_counts": class_counts(labels["test"]),
        "sample_order_sha256": sequence_digest(sample_id_text["test"]),
    }
    current_exact_pair = {
        "text_only_llm": make_row(
            "text_only_llm",
            "Text-only PLM/LLM (Frozen DistilBERT)",
            (
                "distilbert-base-uncased frozen encoder, deterministic 50% "
                "sparse-note attention mask, masked-mean 768-d embedding, "
                "validation-selected class-weighted ridge"
            ),
            dict(test_support),
            text_test_metrics,
        ),
        "image_only_vit": make_row(
            "image_only_vit",
            "Image-only ViT (Frozen ViT-tiny)",
            (
                f"{vit_payload['model_name']} frozen CLS embeddings; "
                f"{vit_payload['selected_validation']['model']} classifier "
                "selected on validation"
            ),
            dict(test_support),
            vit_test_metrics,
        ),
        "image_text_vlm": make_row(
            "image_text_vlm",
            "Image+text VLM (Raw v6 Cross-Attention)",
            (
                "unchanged validation-selected v6 paired bidirectional "
                "cross-attention output under deterministic 50% sparse notes; "
                "no post-hoc class gate"
            ),
            dict(test_support),
            raw_vlm_test_metrics,
        ),
        "proposed_farmfederate": make_row(
            "proposed_farmfederate",
            "Proposed FarmFederate (Unchanged)",
            (
                "unchanged v6 ResNet-50 + text Transformer + bidirectional "
                "cross-attention core with the existing validation-selected "
                "reliability-aware proposed_multimodal router"
            ),
            dict(test_support),
            proposed_test_metrics,
        ),
    }
    common_test_labels = [
        tuple(row["labels"]) for row in current_exact_pair.values()
    ]
    if len(set(common_test_labels)) != 1:
        raise RuntimeError(
            "Final family comparison rows do not share an identical label array"
        )

    candidate_summaries = []
    for candidate in candidates:
        candidate_summaries.append(
            {
                "representation": candidate["representation"],
                "alpha": candidate["alpha"],
                "class_weight_power": candidate["class_weight_power"],
                "class_weights": candidate["class_weights"],
                "accuracy": candidate["accuracy"],
                "macro_f1": candidate["macro_f1"],
                "selection_score": candidate["selection_score"],
                "correct": candidate["correct"],
            }
        )
    validation_selection = {
        "text_only_llm": {
            "partition": "validation",
            "test_used_for_selection": False,
            "candidate_count": int(len(candidates)),
            "selection_metric": "0.60 * macro_f1 + 0.40 * accuracy",
            "tie_break": (
                "macro_f1, accuracy, representation order, alpha near 1, "
                "class-weight power near 0.5"
            ),
            "selected": {
                "model_name": args.model_name,
                "encoder_parameters": encoder_parameters,
                "frozen_encoder": True,
                "pooling": "sparse-attention masked mean",
                "representation": selected_text["representation"],
                "alpha": selected_text["alpha"],
                "class_weight_power": (
                    selected_text["class_weight_power"]
                ),
                "class_weights": selected_class_weights,
                **compact_validation_record(selected_text),
            },
            "candidates": candidate_summaries,
        },
        "image_only_vit": {
            "artifact": str(vit_results_path),
            "partition": vit_payload["selection_partition"],
            "test_used_for_selection": False,
            "candidate_count": int(vit_payload["candidate_count"]),
            "selected": {
                key: value
                for key, value in vit_payload["selected_validation"].items()
            },
        },
        "image_text_vlm": {
            "artifact": str(hierarchy_results_path),
            "partition": hierarchy_payload["selection_partition"],
            "test_used_for_selection": False,
            "keep_rate": hierarchy_validation["keep_rate"],
            "selected": compact_validation_record(
                hierarchy_validation["raw_cross_attention"]
            ),
            "checkpoint_role": (
                "unchanged validation-selected v6 raw paired "
                "cross-attention output"
            ),
            "post_hoc_gate_used": False,
        },
        "proposed_farmfederate": {
            "artifact": str(hierarchy_results_path),
            "partition": hierarchy_payload["selection_partition"],
            "test_used_for_selection": False,
            "keep_rate": hierarchy_validation["keep_rate"],
            "selected": compact_validation_record(
                hierarchy_validation["proposed"]
            ),
            "threshold": hierarchy_validation["proposed"]["threshold"],
            "low_confidence_routes": hierarchy_validation[
                "proposed"
            ]["low_confidence_routes"],
        },
    }

    base_data_audit = base_payload["data_audit"]
    data_audit = {
        "split_seed": int(base_payload["config"]["seed"]),
        "source_grouped": True,
        "split_crops": actual_split_counts,
        "split_source_images": {
            partition: int(len(groups))
            for partition, groups in partition_groups.items()
        },
        "source_group_overlap": group_overlap,
        "pairing_coverage": {
            "train": float(base_data_audit["train_pair_coverage"]),
            "validation": float(base_data_audit["val_pair_coverage"]),
            "test": float(base_data_audit["test_pair_coverage"]),
        },
        "class_counts": {
            partition: class_counts(partition_labels)
            for partition, partition_labels in labels.items()
        },
        "sample_order_sha256": {
            partition: sequence_digest(values)
            for partition, values in sample_id_text.items()
        },
        "label_order_sha256": {
            partition: sequence_digest(
                str(int(value)) for value in partition_labels
            )
            for partition, partition_labels in labels.items()
        },
        "training_fitted_blocked_token_count": int(len(blocked_tokens)),
        "training_fitted_blocked_token_sha256": sequence_digest(
            blocked_tokens
        ),
        "sparse_note_audit": sparse_audit,
        "all_validation_label_arrays_identical": True,
        "all_test_label_arrays_identical": True,
        "current_exact_pair_label_arrays_identical": True,
    }

    protocol = {
        "name": "current_exact_split_model_family_baselines_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "comparison_scope": (
            "same current source-grouped fixed internal-test support"
        ),
        "base_results": str(base_results_path),
        "base_results_sha256": sha256_file(base_results_path),
        "v6_checkpoint": str(
            base_results_path.parent / "models" / "best_vlm.pt"
        ),
        "v6_checkpoint_sha256": sha256_file(
            base_results_path.parent / "models" / "best_vlm.pt"
        ),
        "vit_results": str(vit_results_path),
        "vit_results_sha256": sha256_file(vit_results_path),
        "hierarchy_results": str(hierarchy_results_path),
        "hierarchy_results_sha256": sha256_file(
            hierarchy_results_path
        ),
        "text_model": args.model_name,
        "text_model_local_files_only": True,
        "text_encoder_frozen": True,
        "text_sparse_keep_rate": float(args.keep_rate),
        "text_sparsity_seed": int(args.sparsity_seed),
        "blocked_vocabulary_fit_partition": "train only",
        "model_selection_partition": "validation only",
        "test_used_for_selection": False,
        "test_embedding_extracted_after_text_model_selection": True,
        "proposed_core_modified": False,
        "proposed_predictions_reused_unchanged": True,
        "vlm_baseline_role": (
            "unchanged raw v6 paired cross-attention under the existing 50% "
            "sparse-note protocol; the post-hoc text_image_fusion class gate "
            "is reported only as a diagnostic"
        ),
        "late_probability_fusion_diagnostic": {
            "computed": False,
            "reason": (
                "The existing ViT specialist artifact stores hard "
                "predictions but not validation/test probability scores. "
                "A probability-fusion diagnostic would require rerunning the "
                "separate ViT feature pipeline and is intentionally omitted."
            ),
        },
        "evidence_boundary": (
            "internal development test previously observed during protocol "
            "development; not fresh external or category-best evidence"
        ),
    }
    result = {
        "protocol": protocol,
        "data_audit": data_audit,
        "current_exact_pair": current_exact_pair,
        "validation_selection": validation_selection,
        "diagnostics": {
            "late_fusion_vlm": make_row(
                "late_fusion_vlm",
                "Existing v6 post-hoc late fusion (Diagnostic)",
                (
                    "existing validation-selected class gate over raw v6 "
                    "cross-attention and the enhanced image expert"
                ),
                dict(test_support),
                late_fusion_test_metrics,
            )
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "current_family_baselines.json"
    json_path.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    report_lines = [
        "# Current Exact-Split Model-Family Baselines",
        "",
        "The proposed FarmFederate v6 core was not modified or retrained. "
        "All four rows use the same fixed internal-test label sequence of "
        f"{len(labels['test'])} crops from {len(partition_groups['test'])} "
        "source photographs.",
        "",
        "| System | Architecture role | Accuracy | Macro-F1 | Correct |",
        "|---|---|---:|---:|---:|",
    ]
    for row in current_exact_pair.values():
        report_lines.append(
            f"| {row['display_name']} | {row['architecture']} | "
            f"{row['accuracy']:.2%} | {row['macro_f1']:.4f} | "
            f"{row['correct']}/{row['support']['crops']} |"
        )
    selected = validation_selection["text_only_llm"]["selected"]
    report_lines.extend(
        [
            "",
            "## DistilBERT validation selection",
            "",
            f"- Frozen encoder: `{args.model_name}` "
            f"({encoder_parameters:,} parameters)",
            f"- Sparse-note keep rate: `{args.keep_rate:.0%}` using the "
            "existing deterministic FieldTxt hash rule",
            f"- Candidates: `{len(candidates)}` ridge/class-weight settings",
            f"- Selected representation: `{selected['representation']}`",
            f"- Selected ridge alpha: `{selected['alpha']}`",
            "- Selected class-weight power: "
            f"`{selected['class_weight_power']}`",
            f"- Validation accuracy: `{selected['accuracy']:.2%}`",
            f"- Validation macro-F1: `{selected['macro_f1']:.4f}`",
            "- Test embeddings were extracted only after these choices were "
            "frozen.",
            "",
            "## VLM diagnostic",
            "",
            "The headline image+text VLM row is the unchanged raw v6 paired "
            "cross-attention output, without a post-hoc class gate. The "
            "existing validation-selected late-fusion diagnostic reaches "
            f"`{late_fusion_test_metrics['accuracy']:.2%}` accuracy and "
            f"`{late_fusion_test_metrics['macro_f1']:.4f}` macro-F1 "
            f"({late_fusion_test_metrics['correct']}/75). It does not replace "
            "the raw VLM baseline.",
            "",
            "## Data and compatibility audit",
            "",
            "- Split: `222 / 74 / 75` crops and "
            f"`{len(partition_groups['train'])} / "
            f"{len(partition_groups['validation'])} / "
            f"{len(partition_groups['test'])}` source photographs",
            "- Pairwise source-photo overlap: `0`",
            "- Exact text/image pairing coverage: `100%` in all partitions",
            f"- Training-fitted blocked lexical shortcuts: "
            f"`{len(blocked_tokens)}`",
            "- Validation and fixed-test label arrays were verified against "
            "both the genuine ViT specialist and v6 hierarchy artifacts.",
            "",
            "## Claim boundary",
            "",
            "This is an internal, same-support model-family comparison. The "
            "fixed test was observed during earlier protocol development, so "
            "these values are not fresh external or category-best evidence. "
            "The VLM and proposed rows are the existing exploratory v6 "
            "predictions; this experiment does not alter the core architecture.",
        ]
    )
    report_path = output_dir / "CURRENT_FAMILY_BASELINES_REPORT.md"
    report_path.write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "json": str(json_path),
                "report": str(report_path),
                "selected_text_validation": {
                    "representation": selected["representation"],
                    "alpha": selected["alpha"],
                    "class_weight_power": (
                        selected["class_weight_power"]
                    ),
                    "accuracy": selected["accuracy"],
                    "macro_f1": selected["macro_f1"],
                },
                "current_exact_pair": {
                    key: {
                        "accuracy": value["accuracy"],
                        "macro_f1": value["macro_f1"],
                        "correct": value["correct"],
                    }
                    for key, value in current_exact_pair.items()
                },
                "data_audit": {
                    "split_crops": actual_split_counts,
                    "split_source_images": {
                        partition: len(groups)
                        for partition, groups in partition_groups.items()
                    },
                    "source_group_overlap": group_overlap,
                    "blocked_token_count": len(blocked_tokens),
                    "all_label_arrays_identical": True,
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
