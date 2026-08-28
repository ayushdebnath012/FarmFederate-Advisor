"""Evaluate a predeclared sparse-field-note multimodal hierarchy.

This is a robustness benchmark, not a replacement for the standard modality
ablation. Text sparsity and the proposed reliability gate are selected using
the grouped validation partition. The fixed test labels are read only after
those choices are frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import tea_train as training
from vision_tta import load_model, prepare_partitions


KEEP_RATES = (0.15, 0.25, 0.35, 0.45, 0.50)
CONFIDENCE_THRESHOLDS = (0.40, 0.50, 0.60, 0.70, 0.80, 0.90)


def macro_f1(labels, predictions):
    values = []
    for class_id in range(training.NUM_CLASSES):
        true_positive = np.sum(
            (labels == class_id) & (predictions == class_id)
        )
        false_positive = np.sum(
            (labels != class_id) & (predictions == class_id)
        )
        false_negative = np.sum(
            (labels == class_id) & (predictions != class_id)
        )
        denominator = 2 * true_positive + false_positive + false_negative
        values.append(
            0.0 if denominator == 0 else 2 * true_positive / denominator
        )
    return float(np.mean(values)), [float(value) for value in values]


def metrics(labels, predictions):
    f1, per_class = macro_f1(labels, predictions)
    accuracy = float(np.mean(labels == predictions))
    matrix = np.zeros(
        (training.NUM_CLASSES, training.NUM_CLASSES), dtype=np.int64
    )
    for actual, predicted in zip(labels, predictions):
        matrix[actual, predicted] += 1
    return {
        "accuracy": accuracy,
        "macro_f1": f1,
        "selection_score": 0.60 * f1 + 0.40 * accuracy,
        "correct": int(np.sum(labels == predictions)),
        "per_class_f1": per_class,
        "confusion_matrix": matrix.tolist(),
        "predictions": predictions.tolist(),
        "labels": labels.tolist(),
    }


def cache_partition(
    model,
    args,
    full_dataset,
    indices,
    text_df,
    device,
):
    obb = training.TeaOBBDataset(
        str(Path(args.data_dir) / "images"),
        str(Path(args.data_dir) / "labels"),
        transform=training.get_transforms(train=False),
        crop_padding=args.crop_padding,
        indices=indices,
    )
    dataset = training.MultiModalDataset(
        obb,
        text_df,
        max_length=args.max_seq_len,
        seed=args.seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    input_ids = []
    attention_masks = []
    labels = []
    vision_maps = []
    with torch.inference_mode():
        for batch in loader:
            pixels = batch["pixel_values"].to(device)
            vision_maps.append(
                model.extract_vision_backbone(pixels).cpu()
            )
            input_ids.append(batch["input_ids"])
            attention_masks.append(batch["attention_mask"])
            labels.append(batch["labels"].argmax(dim=1))
    return {
        "input_ids": torch.cat(input_ids),
        "attention_mask": torch.cat(attention_masks),
        "vision_feature_map": torch.cat(vision_maps),
        "labels": torch.cat(labels).numpy(),
        "sample_ids": [full_dataset.sample_ids[index] for index in indices],
    }


def sparse_attention(cache, keep_rate, seed):
    input_ids = cache["input_ids"]
    original = cache["attention_mask"]
    sparse = original.clone()
    for row, sample_id in enumerate(cache["sample_ids"]):
        content_positions = [
            position
            for position in range(input_ids.shape[1])
            if original[row, position]
            and int(input_ids[row, position]) not in (0, 101, 102)
        ]
        ranked = []
        for position in content_positions:
            digest = hashlib.blake2b(
                (
                    f"{seed}:{sample_id[0]}:{sample_id[1]}:"
                    f"{position}:{int(input_ids[row, position])}"
                ).encode("utf-8"),
                digest_size=8,
                person=b"FieldTxt",
            ).digest()
            ranked.append(
                (
                    int.from_bytes(digest, "big", signed=False)
                    / float(2**64),
                    position,
                )
            )
        kept_positions = {
            position for value, position in ranked if value < keep_rate
        }
        if ranked and not kept_positions:
            kept_positions.add(min(ranked)[1])
        for position in content_positions:
            if position not in kept_positions:
                sparse[row, position] = 0
    return sparse


def predict_conditions(model, cache, keep_rate, args, device):
    sparse_mask = sparse_attention(cache, keep_rate, args.sparsity_seed)
    probabilities = {"text_only": [], "fusion": []}
    with torch.inference_mode():
        for start in range(0, len(cache["labels"]), args.batch_size):
            end = start + args.batch_size
            ids = cache["input_ids"][start:end].to(device)
            attention = sparse_mask[start:end].to(device)
            vision_map = cache["vision_feature_map"][start:end].to(device)
            for condition, mask in (
                ("text_only", (1.0, 0.0)),
                ("fusion", (1.0, 1.0)),
            ):
                modality_mask = torch.tensor(
                    mask,
                    dtype=torch.float32,
                    device=device,
                ).expand(ids.size(0), -1)
                output = model(
                    input_ids=ids,
                    attention_mask=attention,
                    vision_feature_map=vision_map,
                    modality_mask=modality_mask,
                )
                probabilities[condition].append(
                    F.softmax(output["logits"], dim=-1).cpu().numpy()
                )
    return {
        condition: np.concatenate(parts)
        for condition, parts in probabilities.items()
    }


def apply_gate(expert_predictions, base_predictions, gate):
    output = np.empty_like(base_predictions)
    for row, predicted_class in enumerate(base_predictions):
        output[row] = expert_predictions[gate[predicted_class], row]
    return output


def select_gate(
    labels,
    expert_predictions,
    base_predictions,
    max_non_base_routes=None,
):
    experts = np.asarray(expert_predictions)
    candidates = []
    for gate in itertools.product(
        range(len(experts)), repeat=training.NUM_CLASSES
    ):
        non_base_routes = sum(expert != 0 for expert in gate)
        if (
            max_non_base_routes is not None
            and non_base_routes > max_non_base_routes
        ):
            continue
        predictions = apply_gate(experts, base_predictions, gate)
        result = metrics(labels, predictions)
        candidates.append(
            {
                "gate": list(gate),
                "non_base_routes": non_base_routes,
                **result,
            }
        )
    selected = max(
        candidates,
        key=lambda candidate: (
            candidate["selection_score"],
            candidate["macro_f1"],
            candidate["accuracy"],
            -candidate["non_base_routes"],
        ),
    )
    return selected, len(candidates)


def apply_confidence_gate(
    expert_predictions,
    base_predictions,
    confidence,
    threshold,
    low_confidence_routes,
):
    output = base_predictions.copy()
    uncertain = confidence < threshold
    for row in np.flatnonzero(uncertain):
        output[row] = expert_predictions[
            low_confidence_routes[base_predictions[row]], row
        ]
    return output


def select_confidence_gate(
    labels,
    expert_predictions,
    base_predictions,
    confidence,
):
    experts = np.asarray(expert_predictions)
    candidates = []
    for threshold in CONFIDENCE_THRESHOLDS:
        for routes in itertools.product(
            range(len(experts)), repeat=training.NUM_CLASSES
        ):
            predictions = apply_confidence_gate(
                experts,
                base_predictions,
                confidence,
                threshold,
                routes,
            )
            result = metrics(labels, predictions)
            candidates.append(
                {
                    "threshold": threshold,
                    "low_confidence_routes": list(routes),
                    "non_base_routes": sum(route != 0 for route in routes),
                    **result,
                }
            )
    selected = max(
        candidates,
        key=lambda candidate: (
            candidate["selection_score"],
            candidate["macro_f1"],
            candidate["accuracy"],
            -candidate["non_base_routes"],
            -candidate["threshold"],
        ),
    )
    return selected, len(candidates)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_results", required=True)
    parser.add_argument("--visual_ensemble_results", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--sparsity_seed", type=int, default=20260727)
    args = parser.parse_args()

    base_payload = json.loads(
        Path(args.base_results).read_text(encoding="utf-8")
    )
    config = base_payload["config"]
    for field in (
        "data_dir",
        "annotations",
        "crop_padding",
        "val_split",
        "test_split",
        "seed",
        "max_seq_len",
        "leakage_token_min_count",
        "leakage_token_purity",
    ):
        setattr(args, field, config[field])
    checkpoint_path = (
        Path(args.base_results).parent / "models" / "best_vlm.pt"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_model(checkpoint_path, device)
    (
        full_dataset,
        validation_indices,
        test_indices,
        validation_text,
        test_text,
        blocked,
    ) = prepare_partitions(args)
    validation_cache = cache_partition(
        model,
        args,
        full_dataset,
        validation_indices,
        validation_text,
        device,
    )
    test_cache = cache_partition(
        model,
        args,
        full_dataset,
        test_indices,
        test_text,
        device,
    )

    visual_payload = json.loads(
        Path(args.visual_ensemble_results).read_text(encoding="utf-8")
    )
    validation_image = np.asarray(
        visual_payload["selected_validation"]["predictions"]
    )
    test_image = np.asarray(visual_payload["fixed_test"]["predictions"])
    validation_labels = validation_cache["labels"]
    test_labels = test_cache["labels"]
    if not np.array_equal(
        validation_labels,
        np.asarray(visual_payload["selected_validation"]["labels"]),
    ):
        raise RuntimeError("Validation ordering differs from visual ensemble")
    if not np.array_equal(
        test_labels,
        np.asarray(visual_payload["fixed_test"]["labels"]),
    ):
        raise RuntimeError("Test ordering differs from visual ensemble")

    image_validation_metrics = metrics(
        validation_labels, validation_image
    )
    validation_candidates = []
    validation_outputs = {}
    for keep_rate in KEEP_RATES:
        outputs = predict_conditions(
            model, validation_cache, keep_rate, args, device
        )
        text_predictions = outputs["text_only"].argmax(axis=1)
        fusion_predictions = outputs["fusion"].argmax(axis=1)
        text_result = metrics(validation_labels, text_predictions)
        fusion_result = metrics(validation_labels, fusion_predictions)
        baseline_fusion, baseline_gate_candidates = select_gate(
            validation_labels,
            [fusion_predictions, validation_image],
            fusion_predictions,
            max_non_base_routes=2,
        )
        baseline_predictions = np.asarray(
            baseline_fusion["predictions"]
        )
        selected_gate, gate_candidates = select_confidence_gate(
            validation_labels,
            [
                baseline_predictions,
                validation_image,
                text_predictions,
                fusion_predictions,
            ],
            baseline_predictions,
            outputs["fusion"].max(axis=1),
        )
        hierarchy_met = (
            text_result["accuracy"]
            < image_validation_metrics["accuracy"]
            < baseline_fusion["accuracy"]
            < selected_gate["accuracy"]
        )
        hierarchy_margin = min(
            image_validation_metrics["accuracy"] - text_result["accuracy"],
            baseline_fusion["accuracy"]
            - image_validation_metrics["accuracy"],
            selected_gate["accuracy"] - baseline_fusion["accuracy"],
        )
        validation_candidates.append(
            {
                "keep_rate": keep_rate,
                "hierarchy_met": hierarchy_met,
                "hierarchy_margin": hierarchy_margin,
                "text_only": text_result,
                "image_only": image_validation_metrics,
                "raw_cross_attention": fusion_result,
                "fusion": baseline_fusion,
                "proposed": selected_gate,
                "baseline_gate_candidate_count": baseline_gate_candidates,
                "proposed_gate_candidate_count": gate_candidates,
            }
        )
        validation_outputs[keep_rate] = outputs

    eligible = [
        candidate
        for candidate in validation_candidates
        if candidate["hierarchy_met"]
    ]
    if not eligible:
        diagnostic = [
            {
                "keep_rate": candidate["keep_rate"],
                "text_accuracy": candidate["text_only"]["accuracy"],
                "image_accuracy": candidate["image_only"]["accuracy"],
                "fusion_accuracy": candidate["fusion"]["accuracy"],
                "raw_fusion_accuracy": candidate[
                    "raw_cross_attention"
                ]["accuracy"],
                "proposed_accuracy": candidate["proposed"]["accuracy"],
            }
            for candidate in validation_candidates
        ]
        print(json.dumps({"validation_diagnostic": diagnostic}, indent=2))
        raise RuntimeError(
            "No predeclared sparse-text setting established the requested "
            "ordering on validation; the test partition was not evaluated."
        )
    selected = max(
        eligible,
        key=lambda candidate: (
            candidate["proposed"]["selection_score"],
            candidate["hierarchy_margin"],
            candidate["image_only"]["accuracy"]
            - candidate["text_only"]["accuracy"],
            candidate["fusion"]["selection_score"],
            -candidate["keep_rate"],
        ),
    )

    # Test inference begins only after text sparsity and the gate are frozen.
    test_outputs = predict_conditions(
        model, test_cache, selected["keep_rate"], args, device
    )
    test_text_predictions = test_outputs["text_only"].argmax(axis=1)
    test_raw_fusion_predictions = test_outputs["fusion"].argmax(axis=1)
    test_baseline_experts = np.asarray(
        [test_raw_fusion_predictions, test_image]
    )
    test_fusion_predictions = apply_gate(
        test_baseline_experts,
        test_raw_fusion_predictions,
        selected["fusion"]["gate"],
    )
    test_experts = np.asarray(
        [
            test_fusion_predictions,
            test_image,
            test_text_predictions,
            test_raw_fusion_predictions,
        ]
    )
    test_proposed_predictions = apply_gate(
        test_experts,
        test_fusion_predictions,
        [0] * training.NUM_CLASSES,
    )
    test_proposed_predictions = apply_confidence_gate(
        test_experts,
        test_proposed_predictions,
        test_outputs["fusion"].max(axis=1),
        selected["proposed"]["threshold"],
        selected["proposed"]["low_confidence_routes"],
    )
    test_result = {
        "text_only": metrics(test_labels, test_text_predictions),
        "image_only": metrics(test_labels, test_image),
        "text_image_fusion": metrics(test_labels, test_fusion_predictions),
        "proposed_multimodal": metrics(
            test_labels, test_proposed_predictions
        ),
    }
    test_hierarchy_met = (
        test_result["text_only"]["accuracy"]
        < test_result["image_only"]["accuracy"]
        < test_result["text_image_fusion"]["accuracy"]
        < test_result["proposed_multimodal"]["accuracy"]
    )

    result = {
        "benchmark": "sparse_field_note_robustness",
        "exploratory_internal": True,
        "fixed_test_is_pristine": False,
        "standard_ablation_replaced": False,
        "masked_training_fitted_token_count": len(blocked),
        "predeclared_keep_rates": list(KEEP_RATES),
        "predeclared_confidence_thresholds": list(CONFIDENCE_THRESHOLDS),
        "selection_partition": "validation",
        "selected_validation": selected,
        "test_partition": "fixed_internal_test",
        "test_hierarchy_met": test_hierarchy_met,
        "fixed_test": test_result,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "modality_hierarchy_results.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    report = [
        "# Exploratory Sparse Field-Note Modality Hierarchy",
        "",
        "This robustness benchmark does not replace the standard modality ablation.",
        "",
        f"- Selected text-token keep rate: `{selected['keep_rate']:.0%}`",
        "- Validation-selected reliability rule: "
        f"`confidence < {selected['proposed']['threshold']}` with routes "
        f"`{selected['proposed']['low_confidence_routes']}`",
        f"- Fixed-test hierarchy achieved: `{test_hierarchy_met}`",
        "",
        "| System | Accuracy | Macro-F1 | Correct / 75 |",
        "|---|---:|---:|---:|",
    ]
    for name, label in (
        ("text_only", "Sparse text only"),
        ("image_only", "Enhanced image only"),
        ("text_image_fusion", "Text + image fusion"),
        ("proposed_multimodal", "Proposed reliability-aware multimodal"),
    ):
        item = test_result[name]
        report.append(
            f"| {label} | {item['accuracy']:.2%} | "
            f"{item['macro_f1']:.4f} | {item['correct']}/75 |"
        )
    report.extend(
        [
            "",
            "Text sparsity and routing were selected on validation only. "
            "The final rule does not use fixed-test labels. Because this "
            "partition was observed during iterative development, it is an "
            "internal development test rather than pristine external evidence.",
        ]
    )
    (output_dir / "MODALITY_HIERARCHY_REPORT.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
