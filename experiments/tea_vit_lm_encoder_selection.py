"""Fair encoder-family selection for the audited tea multimodal split.

This experiment compares the existing validation-selected FarmFederate core
(frozen ResNet-50 plus the compact text Transformer) with a genuine
ViT + pretrained-language-model multimodal candidate.  The candidate uses
frozen ``WinKawaks/vit-tiny-patch16-224`` and frozen
``distilbert-base-uncased`` encoders.  Only a compact supervised fusion head
is fitted on the 222-crop training partition.

All architecture and hyperparameter choices use the 74-crop validation
partition.  The 75-crop internal test partition is embedded and evaluated
only after the candidate is frozen.  The script fails closed if the regenerated
source-grouped split differs from the audited artifacts.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, ViTModel


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tea_train as tea  # noqa: E402
from experiments import tea_current_family_baselines as family  # noqa: E402
from vit_visual_specialist import make_transform  # noqa: E402


DEFAULT_BASE_RESULTS = (
    ROOT
    / "tea_results"
    / "multimodal_v6_vision_full_20260727"
    / "complete_results.json"
)
DEFAULT_HIERARCHY_RESULTS = (
    DEFAULT_BASE_RESULTS.parent
    / "modality_hierarchy_exploratory"
    / "modality_hierarchy_results.json"
)
DEFAULT_OUTPUT_DIR = (
    ROOT / "tea_results" / "vit_lm_encoder_selection_20260729"
)

NUM_CLASSES = tea.NUM_CLASSES
TRAIN_VIEW_NAMES = ("augmented_1", "augmented_2")
EVALUATION_VIEW_NAMES = ("identity", "hflip")
EVALUATION_VIEW_SETS = (
    ("identity",),
    ("identity", "hflip"),
)
PCA_COMPONENTS = (32, 64)
FEATURE_RECIPES = ("concatenation", "symmetric_interactions")
IMAGE_SCALES = (0.5, 1.0, 2.0)
RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
CLASS_WEIGHT_POWERS = (0.0, 0.5, 1.0)


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_results", type=Path, default=DEFAULT_BASE_RESULTS
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
        "--vision_model",
        default="WinKawaks/vit-tiny-patch16-224",
    )
    parser.add_argument(
        "--language_model", default="distilbert-base-uncased"
    )
    parser.add_argument("--keep_rate", type=float, default=0.50)
    parser.add_argument("--sparsity_seed", type=int, default=20260727)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    return parser.parse_args()


def labels_for(
    full_dataset: tea.TeaOBBDataset, indices: Sequence[int]
) -> np.ndarray:
    return np.asarray(
        [int(full_dataset.labels[index]) for index in indices],
        dtype=np.int64,
    )


def sample_ids_for(
    full_dataset: tea.TeaOBBDataset, indices: Sequence[int]
) -> List[Tuple[str, int]]:
    return [full_dataset.sample_ids[index] for index in indices]


def extract_vision_embeddings(
    model: ViTModel,
    dataset: tea.TeaOBBDataset,
    device: torch.device,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    features: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            pixels = batch["pixel_values"].to(device)
            output = model(pixel_values=pixels)
            features.append(
                output.last_hidden_state[:, 0].cpu().numpy()
            )
            labels.append(batch["labels"].argmax(dim=1).numpy())
    return (
        np.concatenate(features).astype(np.float64, copy=False),
        np.concatenate(labels).astype(np.int64, copy=False),
    )


def make_image_dataset(
    data_dir: Path,
    indices: Sequence[int],
    crop_padding: float,
    view: str,
    train: bool,
) -> tea.TeaOBBDataset:
    return tea.TeaOBBDataset(
        str(data_dir / "images"),
        str(data_dir / "labels"),
        transform=make_transform(view, train=train),
        crop_padding=crop_padding,
        indices=list(indices),
    )


def fit_pca(
    train_features: np.ndarray, components: int, seed: int
) -> PCA:
    maximum = min(
        int(components),
        train_features.shape[0] - 1,
        train_features.shape[1],
    )
    if maximum < 2:
        raise RuntimeError(
            f"Insufficient support for PCA: {train_features.shape}"
        )
    pca = PCA(
        n_components=maximum,
        whiten=True,
        svd_solver="randomized",
        random_state=seed,
    )
    pca.fit(train_features)
    return pca


def stable_standardize(
    train_features: np.ndarray,
    features: np.ndarray,
) -> np.ndarray:
    mean = train_features.mean(axis=0, keepdims=True)
    scale = train_features.std(axis=0, keepdims=True)
    return (features - mean) / np.maximum(scale, 1e-6)


def build_fusion_features(
    text_features: np.ndarray,
    image_features: np.ndarray,
    recipe: str,
    image_scale: float,
) -> np.ndarray:
    if text_features.shape != image_features.shape:
        raise RuntimeError(
            "Projected text and image features must have identical shape: "
            f"{text_features.shape} != {image_features.shape}"
        )
    scaled_image = float(image_scale) * image_features
    if recipe == "concatenation":
        return np.concatenate([text_features, scaled_image], axis=1)
    if recipe == "symmetric_interactions":
        return np.concatenate(
            [
                text_features,
                scaled_image,
                np.abs(text_features - scaled_image),
                text_features * scaled_image,
            ],
            axis=1,
        )
    raise ValueError(f"Unsupported fusion recipe: {recipe}")


def class_weighted_ridge(
    features: np.ndarray,
    labels: np.ndarray,
    alpha: float,
    class_weight_power: float,
) -> Tuple[Ridge, List[float]]:
    sample_weights, class_weights = family.sample_weights_for(
        labels, class_weight_power
    )
    targets = np.eye(NUM_CLASSES, dtype=np.float64)[labels]
    model = Ridge(
        alpha=float(alpha),
        fit_intercept=True,
        solver="lsqr",
        tol=1e-5,
    )
    model.fit(features, targets, sample_weight=sample_weights)
    return model, class_weights


def ridge_scores(model: Ridge, features: np.ndarray) -> np.ndarray:
    scores = np.asarray(model.predict(features), dtype=np.float64)
    if scores.shape != (len(features), NUM_CLASSES):
        raise RuntimeError(f"Unexpected score shape: {scores.shape}")
    return scores


def metrics_from_scores(
    labels: np.ndarray, scores: np.ndarray
) -> dict:
    return family.classification_metrics(
        labels, scores.argmax(axis=1)
    )


def compact_metrics(metrics: dict) -> dict:
    return {
        key: metrics[key]
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
    }


def exact_mcnemar(
    labels: np.ndarray,
    first_predictions: Sequence[int],
    second_predictions: Sequence[int],
) -> dict:
    first = np.asarray(first_predictions, dtype=np.int64)
    second = np.asarray(second_predictions, dtype=np.int64)
    first_only = int(
        np.sum((first == labels) & (second != labels))
    )
    second_only = int(
        np.sum((first != labels) & (second == labels))
    )
    discordant = first_only + second_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(
            math.comb(discordant, value)
            for value in range(min(first_only, second_only) + 1)
        ) / (2**discordant)
        p_value = min(1.0, 2.0 * tail)
    return {
        "first_only_correct": first_only,
        "second_only_correct": second_only,
        "discordant": discordant,
        "exact_two_sided_p": float(p_value),
    }


def main() -> None:
    args = parse_args()
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
    hierarchy_results_path = args.hierarchy_results.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    base_payload = read_json(base_results_path)
    hierarchy_payload = read_json(hierarchy_results_path)
    full_dataset, indices, text_frames, blocked_tokens = (
        family.build_current_split(base_payload)
    )
    split_counts = {
        partition: len(partition_indices)
        for partition, partition_indices in indices.items()
    }
    if split_counts != {"train": 222, "validation": 74, "test": 75}:
        raise RuntimeError(
            f"Audited 222/74/75 split changed: {split_counts}"
        )
    groups = {
        partition: {
            full_dataset.groups[index]
            for index in partition_indices
        }
        for partition, partition_indices in indices.items()
    }
    group_overlap = {
        "train_validation": len(groups["train"] & groups["validation"]),
        "train_test": len(groups["train"] & groups["test"]),
        "validation_test": len(groups["validation"] & groups["test"]),
    }
    if any(group_overlap.values()):
        raise RuntimeError(
            f"Source-group leakage detected: {group_overlap}"
        )

    labels = {
        partition: labels_for(full_dataset, partition_indices)
        for partition, partition_indices in indices.items()
    }
    sample_ids = {
        partition: sample_ids_for(full_dataset, partition_indices)
        for partition, partition_indices in indices.items()
    }
    hierarchy_validation = hierarchy_payload["selected_validation"]
    current_validation = hierarchy_validation["proposed"]
    if not np.array_equal(
        labels["validation"],
        np.asarray(current_validation["labels"], dtype=np.int64),
    ):
        raise RuntimeError(
            "Validation label sequence differs from current architecture"
        )
    if abs(float(hierarchy_validation["keep_rate"]) - args.keep_rate) > 1e-12:
        raise RuntimeError(
            "Sparse-note keep rate differs from current architecture"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.language_model,
        local_files_only=True,
        use_fast=True,
    )
    language_model = AutoModel.from_pretrained(
        args.language_model,
        local_files_only=True,
    ).to(device)
    for parameter in language_model.parameters():
        parameter.requires_grad = False
    language_model.eval()
    language_parameters = int(
        sum(parameter.numel() for parameter in language_model.parameters())
    )
    max_length = int(base_payload["config"]["max_seq_len"])

    sparse_audit: Dict[str, dict] = {}
    text_embeddings: Dict[str, np.ndarray] = {}
    for partition in ("train", "validation"):
        input_ids, attention_mask, audit = (
            family.tokenize_sparse_partition(
                tokenizer,
                text_frames[partition],
                sample_ids[partition],
                max_length,
                args.keep_rate,
                args.sparsity_seed,
            )
        )
        sparse_audit[partition] = audit
        text_embeddings[partition] = (
            family.extract_frozen_embeddings(
                language_model,
                input_ids,
                attention_mask,
                device,
                args.batch_size,
            )
        )

    vision_model = ViTModel.from_pretrained(
        args.vision_model,
        local_files_only=True,
    ).to(device)
    for parameter in vision_model.parameters():
        parameter.requires_grad = False
    vision_model.eval()
    vision_parameters = int(
        sum(parameter.numel() for parameter in vision_model.parameters())
    )
    data_dir = Path(base_payload["config"]["data_dir"])
    crop_padding = float(base_payload["config"]["crop_padding"])

    train_image_views: List[np.ndarray] = []
    for view_index, view_name in enumerate(TRAIN_VIEW_NAMES):
        torch.manual_seed(args.seed + view_index)
        random.seed(args.seed + view_index)
        dataset = make_image_dataset(
            data_dir,
            indices["train"],
            crop_padding,
            view_name,
            train=True,
        )
        features, observed_labels = extract_vision_embeddings(
            vision_model, dataset, device, args.batch_size
        )
        if not np.array_equal(labels["train"], observed_labels):
            raise RuntimeError(
                f"Training label mismatch in {view_name}"
            )
        train_image_views.append(features)

    validation_image_views: Dict[str, np.ndarray] = {}
    for view_name in EVALUATION_VIEW_NAMES:
        dataset = make_image_dataset(
            data_dir,
            indices["validation"],
            crop_padding,
            view_name,
            train=False,
        )
        features, observed_labels = extract_vision_embeddings(
            vision_model, dataset, device, args.batch_size
        )
        if not np.array_equal(labels["validation"], observed_labels):
            raise RuntimeError(
                f"Validation label mismatch in {view_name}"
            )
        validation_image_views[view_name] = features

    unique_train_text = text_embeddings["train"]
    repeated_train_text = np.tile(
        unique_train_text, (len(TRAIN_VIEW_NAMES), 1)
    )
    expanded_train_image = np.concatenate(train_image_views, axis=0)
    expanded_train_labels = np.tile(
        labels["train"], len(TRAIN_VIEW_NAMES)
    )

    candidates: List[dict] = []
    fitted_states: Dict[Tuple, dict] = {}
    for components in PCA_COMPONENTS:
        text_pca = fit_pca(
            unique_train_text, components, args.seed + components
        )
        image_pca = fit_pca(
            expanded_train_image,
            components,
            args.seed + 1000 + components,
        )
        projected_unique_train_text = text_pca.transform(
            unique_train_text
        )
        projected_repeated_train_text = np.tile(
            projected_unique_train_text,
            (len(TRAIN_VIEW_NAMES), 1),
        )
        projected_validation_text = text_pca.transform(
            text_embeddings["validation"]
        )
        projected_train_image = image_pca.transform(
            expanded_train_image
        )
        projected_validation_images = {
            view_name: image_pca.transform(features)
            for view_name, features in validation_image_views.items()
        }

        text_reference = projected_repeated_train_text
        image_reference = projected_train_image
        standardized_train_text = stable_standardize(
            text_reference, projected_repeated_train_text
        )
        standardized_validation_text = stable_standardize(
            text_reference, projected_validation_text
        )
        standardized_train_image = stable_standardize(
            image_reference, projected_train_image
        )
        standardized_validation_images = {
            view_name: stable_standardize(
                image_reference, features
            )
            for view_name, features
            in projected_validation_images.items()
        }

        for recipe in FEATURE_RECIPES:
            for image_scale in IMAGE_SCALES:
                train_fusion = build_fusion_features(
                    standardized_train_text,
                    standardized_train_image,
                    recipe,
                    image_scale,
                )
                validation_fusion = {
                    view_name: build_fusion_features(
                        standardized_validation_text,
                        image_features,
                        recipe,
                        image_scale,
                    )
                    for view_name, image_features
                    in standardized_validation_images.items()
                }
                for alpha in RIDGE_ALPHAS:
                    for class_weight_power in CLASS_WEIGHT_POWERS:
                        classifier, class_weights = (
                            class_weighted_ridge(
                                train_fusion,
                                expanded_train_labels,
                                alpha,
                                class_weight_power,
                            )
                        )
                        view_scores = {
                            view_name: ridge_scores(
                                classifier, fusion_features
                            )
                            for view_name, fusion_features
                            in validation_fusion.items()
                        }
                        state_key = (
                            components,
                            recipe,
                            image_scale,
                            alpha,
                            class_weight_power,
                        )
                        fitted_states[state_key] = {
                            "text_pca": text_pca,
                            "image_pca": image_pca,
                            "text_reference": text_reference,
                            "image_reference": image_reference,
                            "classifier": classifier,
                            "class_weights": class_weights,
                        }
                        for view_set in EVALUATION_VIEW_SETS:
                            scores = np.mean(
                                [view_scores[view] for view in view_set],
                                axis=0,
                            )
                            result = metrics_from_scores(
                                labels["validation"], scores
                            )
                            candidates.append(
                                {
                                    "components": int(components),
                                    "recipe": recipe,
                                    "image_scale": float(image_scale),
                                    "alpha": float(alpha),
                                    "class_weight_power": float(
                                        class_weight_power
                                    ),
                                    "class_weights": class_weights,
                                    "views": list(view_set),
                                    **compact_metrics(result),
                                }
                            )

    selected = max(
        candidates,
        key=lambda candidate: (
            candidate["accuracy"],
            candidate["macro_f1"],
            candidate["selection_score"],
            -len(candidate["views"]),
            -candidate["components"],
            candidate["recipe"] == "concatenation",
            -abs(np.log10(candidate["alpha"])),
            -abs(candidate["image_scale"] - 1.0),
            -abs(candidate["class_weight_power"] - 0.5),
        ),
    )
    selected_key = (
        selected["components"],
        selected["recipe"],
        selected["image_scale"],
        selected["alpha"],
        selected["class_weight_power"],
    )
    selected_state = fitted_states[selected_key]

    # Freeze the candidate before reading the incumbent test artifact or
    # computing candidate test features and predictions.
    hierarchy_test = hierarchy_payload["fixed_test"]
    current_test = hierarchy_test["proposed_multimodal"]
    if not np.array_equal(
        labels["test"],
        np.asarray(current_test["labels"], dtype=np.int64),
    ):
        raise RuntimeError(
            "Test label sequence differs from current architecture"
        )
    test_input_ids, test_attention_mask, sparse_audit["test"] = (
        family.tokenize_sparse_partition(
            tokenizer,
            text_frames["test"],
            sample_ids["test"],
            max_length,
            args.keep_rate,
            args.sparsity_seed,
        )
    )
    text_embeddings["test"] = family.extract_frozen_embeddings(
        language_model,
        test_input_ids,
        test_attention_mask,
        device,
        args.batch_size,
    )
    test_image_views: Dict[str, np.ndarray] = {}
    for view_name in selected["views"]:
        dataset = make_image_dataset(
            data_dir,
            indices["test"],
            crop_padding,
            view_name,
            train=False,
        )
        features, observed_labels = extract_vision_embeddings(
            vision_model, dataset, device, args.batch_size
        )
        if not np.array_equal(labels["test"], observed_labels):
            raise RuntimeError(f"Test label mismatch in {view_name}")
        test_image_views[view_name] = features

    projected_test_text = selected_state["text_pca"].transform(
        text_embeddings["test"]
    )
    standardized_test_text = stable_standardize(
        selected_state["text_reference"], projected_test_text
    )
    test_scores = []
    for view_name in selected["views"]:
        projected_test_image = selected_state["image_pca"].transform(
            test_image_views[view_name]
        )
        standardized_test_image = stable_standardize(
            selected_state["image_reference"],
            projected_test_image,
        )
        test_fusion = build_fusion_features(
            standardized_test_text,
            standardized_test_image,
            selected["recipe"],
            selected["image_scale"],
        )
        test_scores.append(
            ridge_scores(selected_state["classifier"], test_fusion)
        )
    candidate_test = metrics_from_scores(
        labels["test"], np.mean(test_scores, axis=0)
    )

    current_validation_accuracy = float(current_validation["accuracy"])
    candidate_validation_accuracy = float(selected["accuracy"])
    if candidate_validation_accuracy > current_validation_accuracy:
        winner = "vit_distilbert_fusion"
        winner_reason = "higher validation accuracy"
    elif candidate_validation_accuracy < current_validation_accuracy:
        winner = "resnet50_compact_transformer"
        winner_reason = "higher validation accuracy"
    elif float(selected["macro_f1"]) > float(
        current_validation["macro_f1"]
    ):
        winner = "vit_distilbert_fusion"
        winner_reason = "validation-accuracy tie broken by macro-F1"
    else:
        winner = "resnet50_compact_transformer"
        winner_reason = (
            "validation-accuracy tie retained by macro-F1/simple incumbent"
        )
    paired_test = exact_mcnemar(
        labels["test"],
        candidate_test["predictions"],
        current_test["predictions"],
    )

    sample_id_sequences = {
        partition: [
            f"{image_file}::{box_idx}"
            for image_file, box_idx in sample_ids[partition]
        ]
        for partition in ("train", "validation", "test")
    }
    top_candidates = sorted(
        candidates,
        key=lambda candidate: (
            candidate["accuracy"],
            candidate["macro_f1"],
            candidate["selection_score"],
        ),
        reverse=True,
    )[:20]
    output = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "selection_rule": (
            "Select architecture by validation accuracy, with macro-F1 as "
            "the first tie-break; candidate test inference is never used "
            "for selection."
        ),
        "winner": winner,
        "winner_reason": winner_reason,
        "protocol": {
            "base_results": str(base_results_path),
            "hierarchy_results": str(hierarchy_results_path),
            "split_counts": split_counts,
            "source_group_counts": {
                partition: len(partition_groups)
                for partition, partition_groups in groups.items()
            },
            "source_group_overlap": group_overlap,
            "keep_rate": float(args.keep_rate),
            "sparsity_seed": int(args.sparsity_seed),
            "masked_training_fitted_token_count": len(blocked_tokens),
            "sample_id_digests": {
                partition: sequence_digest(sequence)
                for partition, sequence in sample_id_sequences.items()
            },
            "base_results_sha256": sha256_file(base_results_path),
            "hierarchy_results_sha256": sha256_file(
                hierarchy_results_path
            ),
            "test_used_for_selection": False,
        },
        "candidate": {
            "name": "Frozen ViT-tiny + frozen DistilBERT fusion",
            "vision_model": args.vision_model,
            "vision_parameters": vision_parameters,
            "language_model": args.language_model,
            "language_parameters": language_parameters,
            "encoders_frozen": True,
            "trainable_component": (
                "validation-selected PCA projection plus "
                "class-weighted ridge fusion head"
            ),
            "training_image_views": list(TRAIN_VIEW_NAMES),
            "validation_candidate_count": len(candidates),
            "selected_validation": selected,
            "fixed_test": compact_metrics(candidate_test),
            "sparse_note_audit": sparse_audit,
        },
        "incumbent": {
            "name": (
                "Frozen ResNet-50 + compact text Transformer + "
                "cross-attention/reliability router"
            ),
            "selected_validation": compact_metrics(
                current_validation
            ),
            "fixed_test": compact_metrics(current_test),
            "artifact_predictions_reused_unchanged": True,
        },
        "paired_test_comparison": {
            "ordering": (
                "first=vit_distilbert_fusion, "
                "second=resnet50_compact_transformer"
            ),
            **paired_test,
        },
        "top_candidate_configurations": top_candidates,
    }
    json_path = output_dir / "encoder_selection_results.json"
    json_path.write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )

    candidate_validation = output["candidate"]["selected_validation"]
    candidate_fixed_test = output["candidate"]["fixed_test"]
    incumbent_validation = output["incumbent"]["selected_validation"]
    incumbent_fixed_test = output["incumbent"]["fixed_test"]
    report = [
        "# ViT + Language-Model Encoder Selection",
        "",
        "The architecture decision used validation accuracy only, with "
        "macro-F1 as the first tie-break. "
        "Candidate test features and predictions were computed only after "
        "the candidate configuration had been frozen; the incumbent test "
        "artifact was reused unchanged and never entered selection.",
        "",
        "## Result",
        "",
        f"- Selected architecture: `{winner}` ({winner_reason}).",
        (
            "- Incumbent validation: "
            f"{incumbent_validation['accuracy']:.4f} accuracy, "
            f"{incumbent_validation['macro_f1']:.4f} macro-F1, "
            f"{incumbent_validation['selection_score']:.4f} score."
        ),
        (
            "- ViT + DistilBERT validation: "
            f"{candidate_validation['accuracy']:.4f} accuracy, "
            f"{candidate_validation['macro_f1']:.4f} macro-F1, "
            f"{candidate_validation['selection_score']:.4f} score."
        ),
        (
            "- Incumbent locked test: "
            f"{incumbent_fixed_test['accuracy']:.4f} accuracy, "
            f"{incumbent_fixed_test['macro_f1']:.4f} macro-F1."
        ),
        (
            "- ViT + DistilBERT locked test: "
            f"{candidate_fixed_test['accuracy']:.4f} accuracy, "
            f"{candidate_fixed_test['macro_f1']:.4f} macro-F1."
        ),
        (
            "- Paired test discordance (candidate-only/incumbent-only): "
            f"`{paired_test['first_only_correct']}/"
            f"{paired_test['second_only_correct']}`; exact two-sided "
            f"McNemar p=`{paired_test['exact_two_sided_p']:.4f}`."
        ),
        "",
        "## Selected ViT + DistilBERT candidate",
        "",
        f"- PCA components per modality: `{selected['components']}`",
        f"- Fusion recipe: `{selected['recipe']}`",
        f"- Image scale: `{selected['image_scale']}`",
        f"- Ridge alpha: `{selected['alpha']}`",
        (
            "- Class-weight power: "
            f"`{selected['class_weight_power']}`"
        ),
        f"- Test-time image views: `{selected['views']}`",
        f"- Validation configurations checked: `{len(candidates)}`",
        "",
        "## Audit",
        "",
        f"- Split: `{split_counts}` crops",
        (
            "- Source groups: "
            f"`{output['protocol']['source_group_counts']}`"
        ),
        f"- Source-group overlap: `{group_overlap}`",
        f"- Sparse-note keep rate: `{args.keep_rate}`",
        f"- Test used for selection: `False`",
        (
            "- Both pretrained encoders were loaded from the local "
            "model cache and remained frozen."
        ),
    ]
    report_path = output_dir / "ENCODER_SELECTION_REPORT.md"
    report_path.write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    print(json.dumps(
        {
            "winner": winner,
            "candidate_validation": {
                "accuracy": candidate_validation["accuracy"],
                "macro_f1": candidate_validation["macro_f1"],
                "selection_score": candidate_validation[
                    "selection_score"
                ],
            },
            "candidate_test": {
                "accuracy": candidate_fixed_test["accuracy"],
                "macro_f1": candidate_fixed_test["macro_f1"],
            },
            "incumbent_validation": {
                "accuracy": incumbent_validation["accuracy"],
                "macro_f1": incumbent_validation["macro_f1"],
                "selection_score": incumbent_validation[
                    "selection_score"
                ],
            },
            "incumbent_test": {
                "accuracy": incumbent_fixed_test["accuracy"],
                "macro_f1": incumbent_fixed_test["macro_f1"],
            },
            "json": str(json_path),
            "report": str(report_path),
        },
        indent=2,
    ))

    del language_model, vision_model
    gc.collect()


if __name__ == "__main__":
    main()
