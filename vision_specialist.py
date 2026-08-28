"""Train a dedicated image-only specialist on frozen FarmFederate features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF

import tea_train as training


def load_feature_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = training.MultiModalClassifier(
        num_labels=len(checkpoint["labels"]),
        vision_backbone=checkpoint["vision_backbone"],
        pretrained_vision=False,
        freeze_vision_backbone=checkpoint["freeze_vision_backbone"],
        max_seq_len=checkpoint["max_seq_len"],
        text_confidence_guard=None,
    )
    model.load_state_dict(checkpoint["state"], strict=True)
    model.to(device).eval()
    return model, checkpoint


def handcrafted_features(normalized_pixels):
    mean = torch.tensor(
        [0.485, 0.456, 0.406],
        device=normalized_pixels.device,
    ).view(1, 3, 1, 1)
    std = torch.tensor(
        [0.229, 0.224, 0.225],
        device=normalized_pixels.device,
    ).view(1, 3, 1, 1)
    rgb = (normalized_pixels * std + mean).clamp(0.0, 1.0)
    flattened = rgb.flatten(2)
    moments = [
        flattened.mean(dim=2),
        flattened.std(dim=2),
    ]
    quantiles = torch.quantile(
        flattened,
        torch.tensor(
            [0.10, 0.25, 0.50, 0.75, 0.90],
            device=rgb.device,
        ),
        dim=2,
    ).permute(1, 0, 2).flatten(1)
    red, green, blue = rgb.unbind(dim=1)
    value, max_indices = rgb.max(dim=1)
    minimum = rgb.min(dim=1).values
    delta = value - minimum
    saturation = torch.where(
        value > 1e-8,
        delta / value.clamp_min(1e-8),
        torch.zeros_like(value),
    )
    safe_delta = delta.clamp_min(1e-8)
    hue_red = torch.remainder((green - blue) / safe_delta, 6.0)
    hue_green = (blue - red) / safe_delta + 2.0
    hue_blue = (red - green) / safe_delta + 4.0
    hue = torch.where(
        max_indices == 0,
        hue_red,
        torch.where(max_indices == 1, hue_green, hue_blue),
    ) / 6.0
    hue = torch.where(delta > 1e-8, hue, torch.zeros_like(hue))
    hsv = torch.stack([hue, saturation, value], dim=1)
    histograms = []
    for sample in hsv:
        channel_histograms = []
        for channel, bins in zip(sample, (12, 6, 6)):
            histogram = torch.histc(channel, bins=bins, min=0.0, max=1.0)
            channel_histograms.append(histogram / histogram.sum().clamp_min(1.0))
        histograms.append(torch.cat(channel_histograms))
    histograms = torch.stack(histograms)

    gray = (
        0.299 * rgb[:, 0]
        + 0.587 * rgb[:, 1]
        + 0.114 * rgb[:, 2]
    )
    gradient_x = gray[:, :, 1:] - gray[:, :, :-1]
    gradient_y = gray[:, 1:, :] - gray[:, :-1, :]
    texture = torch.stack(
        [
            gradient_x.abs().mean(dim=(1, 2)),
            gradient_x.std(dim=(1, 2)),
            gradient_y.abs().mean(dim=(1, 2)),
            gradient_y.std(dim=(1, 2)),
        ],
        dim=1,
    )
    return torch.cat([*moments, quantiles, histograms, texture], dim=1)


def extract_features(model, dataset, device, batch_size, views=1, seed=42):
    feature_parts = {"raw": [], "projected": [], "handcrafted": []}
    labels = []
    for view_index in range(views):
        torch.manual_seed(seed + view_index)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
        for batch in loader:
            pixels = batch["pixel_values"].to(device)
            with torch.no_grad():
                raw_map = model.extract_vision_backbone(pixels)
                projected_map = model.v_projection(raw_map)
                raw = raw_map.mean(dim=(2, 3))
                projected = projected_map.mean(dim=(2, 3))
                handcrafted = handcrafted_features(pixels)
            feature_parts["raw"].append(raw.cpu())
            feature_parts["projected"].append(projected.cpu())
            feature_parts["handcrafted"].append(handcrafted.cpu())
            labels.append(batch["labels"].argmax(-1))
    features = {
        name: torch.cat(parts).numpy()
        for name, parts in feature_parts.items()
    }
    return features, torch.cat(labels).numpy()


def representation(features, name):
    if name == "raw":
        return features["raw"]
    if name == "raw_color":
        return np.concatenate(
            [features["raw"], features["handcrafted"]],
            axis=1,
        )
    if name == "projected_color":
        return np.concatenate(
            [features["projected"], features["handcrafted"]],
            axis=1,
        )
    if name == "all":
        return np.concatenate(
            [
                features["raw"],
                features["projected"],
                features["handcrafted"],
            ],
            axis=1,
        )
    raise ValueError(f"Unknown representation {name}")


def fit_standardizer(features):
    mean = features.mean(axis=0, keepdims=True)
    scale = features.std(axis=0, keepdims=True)
    scale[scale < 1e-6] = 1.0
    return mean.astype(np.float32), scale.astype(np.float32)


def standardize(features, mean, scale):
    return ((features - mean) / scale).astype(np.float32)


def l2_normalize(features):
    norm = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.clip(norm, 1e-8, None)


def fit_specialist(model_type, parameter, features, labels):
    mean, scale = fit_standardizer(features)
    standardized = standardize(features, mean, scale)
    if model_type == "ridge":
        design = np.concatenate(
            [standardized, np.ones((len(standardized), 1), dtype=np.float32)],
            axis=1,
        )
        counts = np.bincount(labels, minlength=training.NUM_CLASSES)
        sample_weights = 1.0 / np.maximum(counts[labels], 1)
        sample_weights *= len(sample_weights) / sample_weights.sum()
        root_weight = np.sqrt(sample_weights).astype(np.float32)[:, None]
        weighted_design = design * root_weight
        targets = np.eye(training.NUM_CLASSES, dtype=np.float32)[labels]
        weighted_targets = targets * root_weight
        gram = weighted_design @ weighted_design.T
        regularized = gram + float(parameter) * np.eye(
            len(gram), dtype=np.float32
        )
        dual = np.linalg.solve(regularized, weighted_targets)
        weights = weighted_design.T @ dual
        return {
            "type": model_type,
            "parameter": float(parameter),
            "mean": mean,
            "scale": scale,
            "weights": weights.astype(np.float32),
        }
    if model_type == "knn":
        counts = np.bincount(labels, minlength=training.NUM_CLASSES)
        class_weights = 1.0 / np.maximum(counts, 1)
        return {
            "type": model_type,
            "parameter": int(parameter),
            "mean": mean,
            "scale": scale,
            "train_features": l2_normalize(standardized),
            "train_labels": labels.copy(),
            "class_weights": class_weights.astype(np.float32),
        }
    if model_type == "centroid":
        normalized = l2_normalize(standardized)
        centroids = np.stack(
            [normalized[labels == class_id].mean(axis=0)
             for class_id in range(training.NUM_CLASSES)]
        )
        return {
            "type": model_type,
            "parameter": 0,
            "mean": mean,
            "scale": scale,
            "centroids": l2_normalize(centroids),
        }
    raise ValueError(f"Unknown model type {model_type}")


def specialist_predict(model, features):
    standardized = standardize(features, model["mean"], model["scale"])
    if model["type"] == "ridge":
        design = np.concatenate(
            [standardized, np.ones((len(standardized), 1), dtype=np.float32)],
            axis=1,
        )
        return (design @ model["weights"]).argmax(axis=1)
    normalized = l2_normalize(standardized)
    if model["type"] == "centroid":
        return (normalized @ model["centroids"].T).argmax(axis=1)
    if model["type"] == "knn":
        similarities = normalized @ model["train_features"].T
        k_value = min(model["parameter"], similarities.shape[1])
        neighbor_indices = np.argpartition(
            -similarities, kth=k_value - 1, axis=1
        )[:, :k_value]
        predictions = []
        for row, indices in enumerate(neighbor_indices):
            votes = np.zeros(training.NUM_CLASSES, dtype=np.float32)
            for index in indices:
                class_id = model["train_labels"][index]
                similarity_weight = max(float(similarities[row, index]), 0.0)
                votes[class_id] += (
                    (similarity_weight + 1e-3)
                    * model["class_weights"][class_id]
                )
            predictions.append(int(votes.argmax()))
        return np.asarray(predictions)
    raise ValueError(f"Unknown model type {model['type']}")


def metrics(labels, predictions):
    return {
        "macro_f1": float(
            f1_score(
                labels,
                predictions,
                labels=list(range(training.NUM_CLASSES)),
                average="macro",
                zero_division=0,
            )
        ),
        "accuracy": float(accuracy_score(labels, predictions)),
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--data_dir", default="Real Dataset")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--train_views", type=int, default=3)
    parser.add_argument("--val_split", type=float, default=0.20)
    parser.add_argument("--test_split", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_dir = str(Path(args.data_dir) / "images")
    label_dir = str(Path(args.data_dir) / "labels")
    full_dataset = training.TeaOBBDataset(
        image_dir,
        label_dir,
        transform=training.get_transforms(train=False),
    )
    train_indices, val_indices, test_indices = (
        training.grouped_train_val_test_split(
            full_dataset.labels,
            full_dataset.groups,
            args.val_split,
            args.test_split,
            args.seed,
        )
    )
    train_dataset = training.TeaOBBDataset(
        image_dir,
        label_dir,
        transform=training.get_transforms(train=True),
        indices=train_indices,
    )
    val_dataset = training.TeaOBBDataset(
        image_dir,
        label_dir,
        transform=training.get_transforms(train=False),
        indices=val_indices,
    )
    test_dataset = training.TeaOBBDataset(
        image_dir,
        label_dir,
        transform=training.get_transforms(train=False),
        indices=test_indices,
    )
    model, checkpoint = load_feature_model(args.checkpoint, device)
    train_features, train_labels = extract_features(
        model,
        train_dataset,
        device,
        args.batch_size,
        views=args.train_views,
        seed=args.seed,
    )
    val_features, val_labels = extract_features(
        model,
        val_dataset,
        device,
        args.batch_size,
        views=1,
        seed=args.seed + 10_000,
    )

    candidates = []
    for representation_name in ("raw", "raw_color", "projected_color", "all"):
        train_x = representation(train_features, representation_name)
        val_x = representation(val_features, representation_name)
        model_rules = [
            ("centroid", 0),
            *[("knn", k_value) for k_value in (1, 3, 5, 9)],
            *[
                ("ridge", regularization)
                for regularization in (0.01, 0.1, 1.0, 10.0, 100.0)
            ],
        ]
        for model_type, parameter in model_rules:
            classifier = fit_specialist(
                model_type,
                parameter,
                train_x,
                train_labels,
            )
            predictions = specialist_predict(classifier, val_x)
            result = metrics(val_labels, predictions)
            result["selection_score"] = (
                0.60 * result["macro_f1"] + 0.40 * result["accuracy"]
            )
            result["representation"] = representation_name
            result["model"] = model_type
            result["parameter"] = parameter
            candidates.append(result)

    selected = max(
        candidates,
        key=lambda candidate: (
            candidate["selection_score"],
            candidate["macro_f1"],
            candidate["accuracy"],
        ),
    )
    # Standard protocol: after validation chooses the fixed hyperparameters,
    # refit that rule on train plus validation before touching the test set.
    train_val_features = {
        name: np.concatenate([train_features[name], val_features[name]], axis=0)
        for name in train_features
    }
    train_val_labels = np.concatenate([train_labels, val_labels], axis=0)
    selected_classifier = fit_specialist(
        selected["model"],
        selected["parameter"],
        representation(train_val_features, selected["representation"]),
        train_val_labels,
    )
    test_features, test_labels = extract_features(
        model,
        test_dataset,
        device,
        args.batch_size,
        views=1,
        seed=args.seed + 20_000,
    )
    test_predictions = specialist_predict(
        selected_classifier,
        representation(test_features, selected["representation"])
    )
    test_metrics = metrics(test_labels, test_predictions)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "classifier": selected_classifier,
            "representation": selected["representation"],
            "checkpoint": str(Path(args.checkpoint)),
            "classes": checkpoint["labels"],
        },
        output_dir / "vision_specialist.pt",
    )
    payload = {
        "checkpoint": str(Path(args.checkpoint)),
        "selection_partition": "validation",
        "test_partition": "fixed_test",
        "train_views": args.train_views,
        "candidate_count": len(candidates),
        "selected_validation": selected,
        "fixed_test": test_metrics,
    }
    (output_dir / "vision_specialist_results.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    report = [
        "# Deep-Feature Image Specialist",
        "",
        f"- Representation: `{selected['representation']}`",
        f"- Classifier: `{selected['model']}`",
        f"- Parameter: `{selected['parameter']}`",
        f"- Validation macro-F1: `{selected['macro_f1']:.4f}`",
        f"- Validation accuracy: `{selected['accuracy']:.4f}`",
        f"- Fixed-test macro-F1: `{test_metrics['macro_f1']:.4f}`",
        f"- Fixed-test accuracy: `{test_metrics['accuracy']:.4f}`",
        f"- Candidate rules evaluated on validation: `{len(candidates)}`",
        "",
        "The selected classifier was refit on train plus validation before its",
        "single fixed-test evaluation.",
    ]
    (output_dir / "VISION_SPECIALIST_REPORT.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
