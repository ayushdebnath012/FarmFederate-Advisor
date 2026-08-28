"""Fit small class-bias calibrators on image-only validation probabilities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

import tea_train as training


def metrics(labels, probabilities):
    predictions = probabilities.argmax(axis=1)
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


def fit_bias(probabilities, labels, class_weight_power, regularization):
    logits = torch.log(
        torch.as_tensor(probabilities, dtype=torch.float32).clamp_min(1e-8)
    )
    targets = torch.as_tensor(labels, dtype=torch.long)
    counts = torch.bincount(
        targets, minlength=training.NUM_CLASSES
    ).to(torch.float32)
    class_weights = counts.clamp_min(1.0).pow(-class_weight_power)
    class_weights /= class_weights.mean()
    bias = torch.zeros(training.NUM_CLASSES, requires_grad=True)
    optimizer = torch.optim.Adam([bias], lr=0.05)
    for _ in range(500):
        optimizer.zero_grad()
        centered_bias = bias - bias.mean()
        loss = F.cross_entropy(
            logits + centered_bias,
            targets,
            weight=class_weights,
        )
        loss = loss + regularization * centered_bias.square().mean()
        loss.backward()
        optimizer.step()
    return (bias - bias.mean()).detach().numpy()


def calibrated_probabilities(probabilities, bias):
    logits = np.log(np.clip(probabilities, 1e-8, 1.0)) + bias[None, :]
    logits -= logits.max(axis=1, keepdims=True)
    exponentiated = np.exp(logits)
    return exponentiated / exponentiated.sum(axis=1, keepdims=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_results", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    payload = json.loads(Path(args.base_results).read_text(encoding="utf-8"))
    validation = payload["results"]["vlm"]["central_metrics"][
        "image_only_validation"
    ]
    test = payload["results"]["vlm"]["cross_modal_analysis"]["conditions"][
        "image_only"
    ]
    val_probabilities = np.asarray(validation["probabilities"], dtype=np.float32)
    val_labels = np.asarray(validation["labels"])
    test_probabilities = np.asarray(test["probabilities"], dtype=np.float32)
    test_labels = np.asarray(test["labels"])

    candidates = []
    for class_weight_power in (0.0, 0.25, 0.50, 0.75, 1.0):
        for regularization in (0.01, 0.10, 1.0, 10.0):
            bias = fit_bias(
                val_probabilities,
                val_labels,
                class_weight_power,
                regularization,
            )
            result = metrics(
                val_labels,
                calibrated_probabilities(val_probabilities, bias),
            )
            candidates.append(
                {
                    "class_weight_power": class_weight_power,
                    "regularization": regularization,
                    "bias": bias.tolist(),
                    **result,
                }
            )
    selected = max(
        candidates,
        key=lambda candidate: (
            candidate["selection_score"],
            candidate["macro_f1"],
            candidate["accuracy"],
            candidate["regularization"],
        ),
    )
    test_calibrated = calibrated_probabilities(
        test_probabilities,
        np.asarray(selected["bias"]),
    )
    test_metrics = metrics(test_labels, test_calibrated)
    result_payload = {
        "candidate_count": len(candidates),
        "selection_partition": "validation",
        "test_partition": "fixed_test",
        "selected_validation": selected,
        "fixed_test": test_metrics,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "vision_calibration_results.json").write_text(
        json.dumps(result_payload, indent=2),
        encoding="utf-8",
    )
    report = [
        "# Validation-Fitted Image Calibration",
        "",
        f"- Class-weight power: `{selected['class_weight_power']}`",
        f"- Regularization: `{selected['regularization']}`",
        f"- Validation macro-F1: `{selected['macro_f1']:.4f}`",
        f"- Validation accuracy: `{selected['accuracy']:.4f}`",
        f"- Fixed-test macro-F1: `{test_metrics['macro_f1']:.4f}`",
        f"- Fixed-test accuracy: `{test_metrics['accuracy']:.4f}`",
    ]
    (output_dir / "VISION_CALIBRATION_REPORT.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result_payload, indent=2))


if __name__ == "__main__":
    main()
