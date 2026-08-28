"""Train a validation-selected frozen-ViT image specialist.

The model family and all hyperparameters are selected on the grouped
validation partition. The fixed test partition is evaluated once after
selection.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
from transformers import ViTModel

import tea_train as training


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def make_transform(view: str, train: bool = False):
    operations = [T.Resize((224, 224))]
    if train:
        operations.extend(
            [
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.RandomRotation(20),
                T.ColorJitter(
                    brightness=0.25,
                    contrast=0.25,
                    saturation=0.20,
                    hue=0.03,
                ),
            ]
        )
    elif view == "hflip":
        operations.append(T.RandomHorizontalFlip(p=1.0))
    elif view == "vflip":
        operations.append(T.RandomVerticalFlip(p=1.0))
    elif view == "r90":
        operations.append(T.RandomRotation((90, 90)))
    elif view == "r270":
        operations.append(T.RandomRotation((270, 270)))
    operations.extend([T.ToTensor(), T.Normalize(MEAN, STD)])
    return T.Compose(operations)


def extract(model, dataset, device, batch_size):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    features = []
    labels = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            pixels = batch["pixel_values"].to(device)
            output = model(pixel_values=pixels)
            features.append(output.last_hidden_state[:, 0].cpu().numpy())
            labels.append(batch["labels"].argmax(dim=1).numpy())
    return np.concatenate(features), np.concatenate(labels)


def normalize(train_features, features):
    mean = train_features.mean(axis=0, keepdims=True)
    std = train_features.std(axis=0, keepdims=True).clip(1e-5)
    return (features - mean) / std


def l2(features):
    return features / np.linalg.norm(features, axis=1, keepdims=True).clip(1e-8)


def macro_f1(labels, predictions):
    scores = []
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
        scores.append(
            0.0 if denominator == 0 else 2 * true_positive / denominator
        )
    return float(np.mean(scores)), [float(value) for value in scores]


def metrics(labels, scores):
    predictions = scores.argmax(axis=1)
    f1, per_class = macro_f1(labels, predictions)
    accuracy = float(np.mean(labels == predictions))
    matrix = np.zeros(
        (training.NUM_CLASSES, training.NUM_CLASSES), dtype=np.int64
    )
    for actual, predicted in zip(labels, predictions):
        matrix[actual, predicted] += 1
    return {
        "macro_f1": f1,
        "accuracy": accuracy,
        "selection_score": 0.60 * f1 + 0.40 * accuracy,
        "per_class_f1": per_class,
        "confusion_matrix": matrix.tolist(),
        "predictions": predictions.tolist(),
        "labels": labels.tolist(),
    }


def fit_ridge(features, labels, alpha, class_power):
    one_hot = np.eye(training.NUM_CLASSES)[labels]
    counts = np.bincount(labels, minlength=training.NUM_CLASSES).clip(1)
    sample_weights = counts[labels].astype(np.float64) ** (-class_power)
    sample_weights /= sample_weights.mean()
    augmented = np.concatenate(
        [features, np.ones((len(features), 1))], axis=1
    )
    weighted_features = augmented * np.sqrt(sample_weights[:, None])
    weighted_targets = one_hot * np.sqrt(sample_weights[:, None])
    regularizer = np.eye(augmented.shape[1]) * alpha
    regularizer[-1, -1] = alpha * 0.01
    return np.linalg.solve(
        weighted_features.T @ weighted_features + regularizer,
        weighted_features.T @ weighted_targets,
    )


def ridge_scores(features, weights):
    augmented = np.concatenate(
        [features, np.ones((len(features), 1))], axis=1
    )
    return augmented @ weights


def knn_scores(train_features, train_labels, features, neighbors):
    train_features = l2(train_features)
    features = l2(features)
    similarities = features @ train_features.T
    nearest = np.argpartition(
        -similarities,
        kth=min(neighbors, similarities.shape[1] - 1),
        axis=1,
    )[:, :neighbors]
    scores = np.zeros((len(features), training.NUM_CLASSES))
    for row in range(len(features)):
        local_similarity = similarities[row, nearest[row]]
        weights = np.maximum(local_similarity + 1.0, 1e-4) ** 3
        for sample_index, weight in zip(nearest[row], weights):
            scores[row, train_labels[sample_index]] += weight
    return scores


def aggregate_views(view_scores, views):
    return np.mean([view_scores[view] for view in views], axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_results", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--model_name", default="WinKawaks/vit-tiny-patch16-224"
    )
    parser.add_argument("--train_views", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260727)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_payload = json.loads(
        Path(args.base_results).read_text(encoding="utf-8")
    )
    config = base_payload["config"]
    data_dir = Path(config["data_dir"])
    full = training.TeaOBBDataset(
        str(data_dir / "images"),
        str(data_dir / "labels"),
        crop_padding=config["crop_padding"],
    )
    train_indices, validation_indices, test_indices = (
        training.grouped_train_val_test_split(
            full.labels,
            full.groups,
            config["val_split"],
            config["test_split"],
            config["seed"],
        )
    )

    model = ViTModel.from_pretrained(
        args.model_name,
        local_files_only=True,
    ).to(device)
    for parameter in model.parameters():
        parameter.requires_grad = False

    train_feature_views = []
    train_labels = None
    for view_index in range(args.train_views):
        torch.manual_seed(args.seed + view_index)
        random.seed(args.seed + view_index)
        dataset = training.TeaOBBDataset(
            str(data_dir / "images"),
            str(data_dir / "labels"),
            transform=make_transform("train", train=True),
            crop_padding=config["crop_padding"],
            indices=train_indices,
        )
        features, labels = extract(
            model, dataset, device, args.batch_size
        )
        train_feature_views.append(features)
        if train_labels is None:
            train_labels = labels
        else:
            assert np.array_equal(train_labels, labels)
    train_features = np.concatenate(train_feature_views)
    expanded_train_labels = np.tile(train_labels, args.train_views)

    evaluation_views = ["identity", "hflip", "vflip", "r90", "r270"]
    validation_features = {}
    test_features = {}
    validation_labels = None
    test_labels = None
    for view in evaluation_views:
        validation_dataset = training.TeaOBBDataset(
            str(data_dir / "images"),
            str(data_dir / "labels"),
            transform=make_transform(view),
            crop_padding=config["crop_padding"],
            indices=validation_indices,
        )
        test_dataset = training.TeaOBBDataset(
            str(data_dir / "images"),
            str(data_dir / "labels"),
            transform=make_transform(view),
            crop_padding=config["crop_padding"],
            indices=test_indices,
        )
        validation_features[view], current_validation_labels = extract(
            model, validation_dataset, device, args.batch_size
        )
        test_features[view], current_test_labels = extract(
            model, test_dataset, device, args.batch_size
        )
        if validation_labels is None:
            validation_labels = current_validation_labels
            test_labels = current_test_labels
        else:
            assert np.array_equal(
                validation_labels, current_validation_labels
            )
            assert np.array_equal(test_labels, current_test_labels)

    view_sets = [
        ["identity"],
        ["identity", "hflip"],
        ["identity", "vflip"],
        ["identity", "hflip", "vflip"],
        ["identity", "hflip", "vflip", "r90", "r270"],
    ]
    candidates = []
    fitted_models = {}
    representations = {
        "standardized": (
            normalize(train_features, train_features),
            {
                view: normalize(train_features, features)
                for view, features in validation_features.items()
            },
            {
                view: normalize(train_features, features)
                for view, features in test_features.items()
            },
        ),
        "l2": (
            l2(train_features),
            {
                view: l2(features)
                for view, features in validation_features.items()
            },
            {
                view: l2(features)
                for view, features in test_features.items()
            },
        ),
    }
    test_score_cache = {}
    for representation, (
        represented_train,
        represented_validation,
        represented_test,
    ) in representations.items():
        for alpha in (0.01, 0.1, 1.0, 10.0, 100.0):
            for class_power in (0.0, 0.5, 1.0):
                key = ("ridge", representation, alpha, class_power)
                weights = fit_ridge(
                    represented_train,
                    expanded_train_labels,
                    alpha,
                    class_power,
                )
                fitted_models[key] = weights
                validation_scores = {
                    view: ridge_scores(features, weights)
                    for view, features in represented_validation.items()
                }
                test_score_cache[key] = {
                    view: ridge_scores(features, weights)
                    for view, features in represented_test.items()
                }
                for views in view_sets:
                    result = metrics(
                        validation_labels,
                        aggregate_views(validation_scores, views),
                    )
                    candidates.append(
                        {
                            "model": "ridge",
                            "representation": representation,
                            "parameter": alpha,
                            "class_power": class_power,
                            "views": views,
                            **result,
                        }
                    )
        for neighbors in (1, 3, 5, 9, 15, 25):
            key = ("knn", representation, neighbors, None)
            validation_scores = {
                view: knn_scores(
                    represented_train,
                    expanded_train_labels,
                    features,
                    neighbors,
                )
                for view, features in represented_validation.items()
            }
            test_score_cache[key] = {
                view: knn_scores(
                    represented_train,
                    expanded_train_labels,
                    features,
                    neighbors,
                )
                for view, features in represented_test.items()
            }
            for views in view_sets:
                result = metrics(
                    validation_labels,
                    aggregate_views(validation_scores, views),
                )
                candidates.append(
                    {
                        "model": "knn",
                        "representation": representation,
                        "parameter": neighbors,
                        "class_power": None,
                        "views": views,
                        **result,
                    }
                )

    selected = max(
        candidates,
        key=lambda candidate: (
            candidate["selection_score"],
            candidate["macro_f1"],
            candidate["accuracy"],
            -len(candidate["views"]),
        ),
    )
    selected_key = (
        selected["model"],
        selected["representation"],
        selected["parameter"],
        selected["class_power"],
    )
    fixed_test = metrics(
        test_labels,
        aggregate_views(
            test_score_cache[selected_key],
            selected["views"],
        ),
    )

    output = {
        "model_name": args.model_name,
        "train_views": args.train_views,
        "candidate_count": len(candidates),
        "selection_partition": "validation",
        "test_partition": "fixed_test",
        "selected_validation": selected,
        "fixed_test": fixed_test,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "vit_visual_specialist_results.json").write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )
    report = [
        "# Frozen ViT Visual Specialist",
        "",
        f"- Pretrained model: `{args.model_name}`",
        f"- Selected classifier: `{selected['model']}`",
        f"- Representation: `{selected['representation']}`",
        f"- Parameter: `{selected['parameter']}`",
        f"- Class-weight power: `{selected['class_power']}`",
        f"- Evaluation views: `{selected['views']}`",
        f"- Validation accuracy: `{selected['accuracy']:.4f}`",
        f"- Validation macro-F1: `{selected['macro_f1']:.4f}`",
        f"- Fixed-test accuracy: `{fixed_test['accuracy']:.4f}`",
        f"- Fixed-test macro-F1: `{fixed_test['macro_f1']:.4f}`",
    ]
    (output_dir / "VIT_VISUAL_SPECIALIST_REPORT.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
