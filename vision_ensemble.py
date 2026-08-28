"""Select a compact class-conditional image-only ensemble on validation."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

import tea_train as training


def score(labels, predictions):
    macro_f1 = float(
        f1_score(
            labels,
            predictions,
            labels=list(range(training.NUM_CLASSES)),
            average="macro",
            zero_division=0,
        )
    )
    accuracy = float(accuracy_score(labels, predictions))
    return {
        "macro_f1": macro_f1,
        "accuracy": accuracy,
        "selection_score": 0.60 * macro_f1 + 0.40 * accuracy,
        "per_class_f1": f1_score(
            labels,
            predictions,
            labels=list(range(training.NUM_CLASSES)),
            average=None,
            zero_division=0,
        ).tolist(),
        "confusion_matrix": confusion_matrix(
            labels,
            predictions,
            labels=list(range(training.NUM_CLASSES)),
        ).tolist(),
        "predictions": predictions.tolist(),
        "labels": labels.tolist(),
    }


def apply_gate(expert_predictions, gate):
    base = expert_predictions[0]
    output = np.empty_like(base)
    for row, base_class in enumerate(base):
        output[row] = expert_predictions[gate[base_class], row]
    return output


def load_inputs(args):
    base = json.loads(Path(args.base_results).read_text(encoding="utf-8"))
    tta = json.loads(Path(args.tta_results).read_text(encoding="utf-8"))
    specialist = json.loads(
        Path(args.specialist_results).read_text(encoding="utf-8")
    )
    calibration = (
        json.loads(Path(args.calibration_results).read_text(encoding="utf-8"))
        if args.calibration_results
        else None
    )
    extra_base = (
        json.loads(Path(args.extra_base_results).read_text(encoding="utf-8"))
        if args.extra_base_results
        else None
    )
    extra_calibration = (
        json.loads(
            Path(args.extra_calibration_results).read_text(encoding="utf-8")
        )
        if args.extra_calibration_results
        else None
    )
    vit_specialist = (
        json.loads(
            Path(args.vit_specialist_results).read_text(encoding="utf-8")
        )
        if args.vit_specialist_results
        else None
    )
    base_val = base["results"]["vlm"]["central_metrics"][
        "image_only_validation"
    ]
    base_test = base["results"]["vlm"]["cross_modal_analysis"]["conditions"][
        "image_only"
    ]
    validation_predictions = [
        base_val["predictions"],
        tta["selected"]["predictions"],
        specialist["selected_validation"]["predictions"],
    ]
    test_predictions = [
        base_test["predictions"],
        tta["locked_test"]["predictions"],
        specialist["fixed_test"]["predictions"],
    ]
    expert_names = ["v6_base", "v6_tta", "deep_feature_specialist"]
    if calibration is not None:
        validation_predictions.append(
            calibration["selected_validation"]["predictions"]
        )
        test_predictions.append(calibration["fixed_test"]["predictions"])
        expert_names.append("calibrated_v6")
    if extra_base is not None:
        extra_val = extra_base["results"]["vlm"]["central_metrics"][
            "image_only_validation"
        ]
        extra_test = extra_base["results"]["vlm"]["cross_modal_analysis"][
            "conditions"
        ]["image_only"]
        validation_predictions.append(extra_val["predictions"])
        test_predictions.append(extra_test["predictions"])
        expert_names.append("extra_base")
    if extra_calibration is not None:
        validation_predictions.append(
            extra_calibration["selected_validation"]["predictions"]
        )
        test_predictions.append(
            extra_calibration["fixed_test"]["predictions"]
        )
        expert_names.append("calibrated_extra_base")
    if vit_specialist is not None:
        validation_predictions.append(
            vit_specialist["selected_validation"]["predictions"]
        )
        test_predictions.append(
            vit_specialist["fixed_test"]["predictions"]
        )
        expert_names.append("frozen_vit_specialist")
    return {
        "expert_names": expert_names,
        "validation_labels": np.asarray(base_val["labels"]),
        "test_labels": np.asarray(base_test["labels"]),
        "validation_predictions": np.asarray(validation_predictions),
        "test_predictions": np.asarray(test_predictions),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_results", required=True)
    parser.add_argument("--tta_results", required=True)
    parser.add_argument("--specialist_results", required=True)
    parser.add_argument("--calibration_results")
    parser.add_argument("--extra_base_results")
    parser.add_argument("--extra_calibration_results")
    parser.add_argument("--vit_specialist_results")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    inputs = load_inputs(args)
    for partition in ("validation", "test"):
        labels = inputs[f"{partition}_labels"]
        predictions = inputs[f"{partition}_predictions"]
        if predictions.shape[1] != len(labels):
            raise RuntimeError(f"{partition} expert predictions are misaligned")

    candidates = []
    num_experts = len(inputs["expert_names"])
    for gate in itertools.product(
        range(num_experts), repeat=training.NUM_CLASSES
    ):
        predictions = apply_gate(inputs["validation_predictions"], gate)
        metrics = score(inputs["validation_labels"], predictions)
        candidates.append(
            {
                "gate": list(gate),
                "non_base_routes": sum(expert != 0 for expert in gate),
                **metrics,
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
    test_predictions = apply_gate(
        inputs["test_predictions"],
        selected["gate"],
    )
    test_metrics = score(inputs["test_labels"], test_predictions)
    payload = {
        "experts": inputs["expert_names"],
        "gate_key": "v6_base_predicted_class",
        "candidate_count": len(candidates),
        "selected_validation": selected,
        "fixed_test": test_metrics,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "vision_ensemble_results.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    report = [
        "# Validation-Selected Visual Ensemble",
        "",
        f"- Gate by base predicted class: `{selected['gate']}`",
        f"- Validation macro-F1: `{selected['macro_f1']:.4f}`",
        f"- Validation accuracy: `{selected['accuracy']:.4f}`",
        f"- Fixed-test macro-F1: `{test_metrics['macro_f1']:.4f}`",
        f"- Fixed-test accuracy: `{test_metrics['accuracy']:.4f}`",
        f"- Candidate gates evaluated on validation: `{len(candidates)}`",
    ]
    (output_dir / "VISION_ENSEMBLE_REPORT.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
