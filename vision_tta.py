"""Validation-selected image-only TTA for a trained FarmFederate checkpoint.

The selector evaluates label-preserving visual views on validation only. The
chosen rule is then frozen and evaluated on the source-grouped locked test.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.transforms import functional as TF

import tea_train as training


@dataclass(frozen=True)
class ViewSpec:
    name: str
    padding: float = 0.10
    horizontal_flip: bool = False
    vertical_flip: bool = False
    rotation: int = 0


VIEW_SPECS = (
    ViewSpec("base"),
    ViewSpec("horizontal_flip", horizontal_flip=True),
    ViewSpec("vertical_flip", vertical_flip=True),
    ViewSpec("rotate_180", rotation=180),
    ViewSpec("tight_crop", padding=0.05),
    ViewSpec("context_crop", padding=0.15),
)

# These families are declared before validation scores are inspected.
VIEW_FAMILIES = (
    ("base",),
    ("base", "horizontal_flip"),
    ("base", "horizontal_flip", "vertical_flip", "rotate_180"),
    ("base", "tight_crop", "context_crop"),
    tuple(spec.name for spec in VIEW_SPECS),
)
VISION_SCALES = (1.0, 1.5, 2.0, 2.5, 3.0)


def build_transform(spec: ViewSpec):
    operations = [T.Resize((224, 224))]
    if spec.horizontal_flip:
        operations.append(T.Lambda(TF.hflip))
    if spec.vertical_flip:
        operations.append(T.Lambda(TF.vflip))
    if spec.rotation:
        operations.append(
            T.Lambda(lambda image, angle=spec.rotation: TF.rotate(image, angle))
        )
    operations.extend(
        [
            T.ToTensor(),
            T.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225],
            ),
        ]
    )
    return T.Compose(operations)


def annotation_partition(text_df, full_dataset, indices):
    keys = {full_dataset.sample_ids[index] for index in indices}
    row_keys = list(zip(text_df["image_file"], text_df["box_idx"]))
    mask = pd.Series([key in keys for key in row_keys])
    return text_df.loc[mask].reset_index(drop=True)


def prepare_partitions(args):
    full_dataset = training.TeaOBBDataset(
        str(Path(args.data_dir) / "images"),
        str(Path(args.data_dir) / "labels"),
        transform=training.get_transforms(train=False),
        crop_padding=args.crop_padding,
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
    text_df = training.load_annotations_csv(args.annotations)
    text_train = annotation_partition(text_df, full_dataset, train_indices)
    text_val = annotation_partition(text_df, full_dataset, val_indices)
    text_test = annotation_partition(text_df, full_dataset, test_indices)
    blocked = training.fit_label_leakage_vocabulary(
        text_train,
        min_count=args.leakage_token_min_count,
        purity_threshold=args.leakage_token_purity,
    )
    return (
        full_dataset,
        val_indices,
        test_indices,
        training.sanitize_annotation_text(text_val, blocked),
        training.sanitize_annotation_text(text_test, blocked),
        blocked,
    )


def load_model(checkpoint_path, device):
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


def collect_view_outputs(
    model,
    args,
    indices,
    text_df,
    view_specs,
    device,
):
    outputs = {}
    expected_labels = None
    for spec in view_specs:
        obb_dataset = training.TeaOBBDataset(
            str(Path(args.data_dir) / "images"),
            str(Path(args.data_dir) / "labels"),
            transform=build_transform(spec),
            crop_padding=spec.padding,
            indices=indices,
        )
        dataset = training.MultiModalDataset(
            obb_dataset,
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
        fusion_logits = []
        vision_logits = []
        labels = []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                pixels = batch["pixel_values"].to(device)
                modality_mask = torch.tensor(
                    [0.0, 1.0],
                    device=device,
                    dtype=torch.float32,
                ).expand(input_ids.size(0), -1)
                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixels,
                    modality_mask=modality_mask,
                )
                fusion_logits.append(output["fusion_logits"].cpu())
                vision_logits.append(output["vision_logits"].cpu())
                labels.append(batch["labels"].argmax(-1))
        labels = torch.cat(labels).numpy()
        if expected_labels is None:
            expected_labels = labels
        elif not np.array_equal(expected_labels, labels):
            raise RuntimeError("TTA views produced inconsistent sample ordering")
        outputs[spec.name] = {
            "fusion_logits": torch.cat(fusion_logits),
            "vision_logits": torch.cat(vision_logits),
        }
    return outputs, expected_labels


def ensemble_probabilities(outputs, view_names, vision_scale):
    probabilities = []
    for view_name in view_names:
        view = outputs[view_name]
        logits = view["fusion_logits"] + vision_scale * view["vision_logits"]
        probabilities.append(F.softmax(logits, dim=-1))
    return torch.stack(probabilities).mean(dim=0).numpy()


def score_probabilities(probabilities, labels):
    predictions = probabilities.argmax(axis=1)
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


def select_on_validation(outputs, labels):
    candidates = []
    for view_names in VIEW_FAMILIES:
        for vision_scale in VISION_SCALES:
            probabilities = ensemble_probabilities(
                outputs, view_names, vision_scale
            )
            metrics = score_probabilities(probabilities, labels)
            candidates.append(
                {
                    "views": list(view_names),
                    "vision_scale": vision_scale,
                    **metrics,
                }
            )
    selected = max(
        candidates,
        key=lambda candidate: (
            candidate["macro_f1"],
            candidate["accuracy"],
            -len(candidate["views"]),
            -abs(candidate["vision_scale"] - 2.0),
        ),
    )
    return selected, candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--data_dir", default="Real Dataset")
    parser.add_argument(
        "--annotations",
        default="tea_results/annotation/annotations.csv",
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--crop_padding", type=float, default=0.10)
    parser.add_argument("--val_split", type=float, default=0.20)
    parser.add_argument("--test_split", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--leakage_token_min_count", type=int, default=3)
    parser.add_argument("--leakage_token_purity", type=float, default=0.95)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (
        _,
        val_indices,
        test_indices,
        text_val,
        text_test,
        blocked_tokens,
    ) = prepare_partitions(args)
    model, checkpoint = load_model(args.checkpoint, device)

    val_outputs, val_labels = collect_view_outputs(
        model,
        args,
        val_indices,
        text_val,
        VIEW_SPECS,
        device,
    )
    selected, candidates = select_on_validation(val_outputs, val_labels)

    selected_specs = tuple(
        spec for spec in VIEW_SPECS if spec.name in selected["views"]
    )
    test_outputs, test_labels = collect_view_outputs(
        model,
        args,
        test_indices,
        text_test,
        selected_specs,
        device,
    )
    test_probabilities = ensemble_probabilities(
        test_outputs,
        selected["views"],
        selected["vision_scale"],
    )
    test_metrics = score_probabilities(test_probabilities, test_labels)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint": str(Path(args.checkpoint)),
        "selection_partition": "validation",
        "test_partition": "locked_test",
        "candidate_count": len(candidates),
        "selected": selected,
        "locked_test": test_metrics,
        "masked_training_fitted_tokens": len(blocked_tokens),
        "checkpoint_validation_f1": checkpoint["val_f1_macro"],
    }
    (output_dir / "vision_tta_results.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    report = [
        "# Validation-Selected Image-Only TTA",
        "",
        f"- Selected views: `{', '.join(selected['views'])}`",
        f"- Selected vision scale: `{selected['vision_scale']:.1f}`",
        f"- Validation macro-F1: `{selected['macro_f1']:.4f}`",
        f"- Validation accuracy: `{selected['accuracy']:.4f}`",
        f"- Locked-test macro-F1: `{test_metrics['macro_f1']:.4f}`",
        f"- Locked-test accuracy: `{test_metrics['accuracy']:.4f}`",
        f"- Candidate rules evaluated on validation: `{len(candidates)}`",
        "",
        "The locked test was evaluated only after the validation rule was frozen.",
    ]
    (output_dir / "VISION_TTA_REPORT.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
