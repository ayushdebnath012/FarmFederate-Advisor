"""Component ablation of the proposed multimodal architecture.

Every variant is retrained from scratch on the same source-grouped split and
scored on the same locked 75-crop test, so the differences are attributable to
the component and not to a different partition. The data pipeline mirrors
``tea_train.run``: source-grouped split, exact (image_file, box_idx) pairing,
training-fitted leakage mask, and cached frozen ResNet-50 feature views.

Variants:
  full                 the reported architecture
  no_cross_attention   parents pass through without token interaction
  no_reliability_gate  fixed equal modality weighting instead of a learned gate
  no_expert_residual   lambda = 0, so the fusion head alone decides
  no_interaction_feats drop |diff| and product blocks from the fusion input
  lightweight_vision   trainable small CNN instead of frozen ResNet-50
  text_1_layer         one Transformer layer instead of two
  no_aux_losses        drop the auxiliary parent losses
  no_alignment_loss    drop the supervised cross-modal alignment term
"""

from __future__ import annotations

import copy
import json
import sys
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tea_train as T  # noqa: E402

OUTPUT = ROOT / "tea_results" / "architecture_ablation_v1"

VARIANTS: list[tuple[str, str, dict[str, Any]]] = [
    ("full", "Proposed (unchanged)", {}),
    ("no_cross_attention", "No cross-attention", {"use_cross_attention": False}),
    ("no_reliability_gate", "No reliability gate", {"use_reliability_gate": False}),
    ("no_expert_residual", r"No expert residual ($\lambda$=0)",
     {"expert_residual_weight": 0.0}),
    ("residual_lambda1", r"Expert residual ($\lambda$=1)",
     {"expert_residual_weight": 1.0}),
    ("residual_lambda4", r"Expert residual ($\lambda$=4)",
     {"expert_residual_weight": 4.0}),
    (
        "no_interaction_feats",
        "No interaction features",
        {"use_interaction_features": False},
    ),
    (
        "lightweight_vision",
        "Lightweight CNN vision",
        {"vision_backbone": "lightweight", "freeze_vision_backbone": False},
    ),
    ("text_1_layer", "Text encoder: 1 layer", {"text_layers": 1}),
    (
        "no_aux_losses",
        "No auxiliary parent losses",
        {"text_auxiliary_weight": 0.0, "vision_auxiliary_weight": 0.0},
    ),
    ("no_alignment_loss", "No alignment loss", {"alignment_weight": 0.0}),
]


def build_data(cfg: T.Config, device: torch.device) -> dict[str, Any]:
    """Reproduce the paper's split, pairing, and leakage mask exactly once."""
    img_dir = Path(cfg.data_dir) / "images"
    lbl_dir = Path(cfg.data_dir) / "labels"
    if not img_dir.is_dir() or not lbl_dir.is_dir():
        raise FileNotFoundError(f"Expected images/ and labels/ under {cfg.data_dir}")

    full_ds = T.TeaOBBDataset(
        str(img_dir), str(lbl_dir),
        transform=T.get_transforms(train=True), crop_padding=cfg.crop_padding,
    )
    trn_i, val_i, test_i = T.grouped_train_val_test_split(
        full_ds.labels, full_ds.groups, cfg.val_split, cfg.test_split, cfg.seed
    )
    groups = [
        {full_ds.groups[i] for i in idx} for idx in (trn_i, val_i, test_i)
    ]
    overlap = (
        len(groups[0] & groups[1])
        + len(groups[0] & groups[2])
        + len(groups[1] & groups[2])
    )
    if overlap:
        raise RuntimeError(f"Source-group overlap is {overlap}, expected 0")

    train_obb = T.TeaOBBDataset(
        str(img_dir), str(lbl_dir), transform=T.get_transforms(train=True),
        crop_padding=cfg.crop_padding, indices=trn_i,
    )
    val_obb = T.TeaOBBDataset(
        str(img_dir), str(lbl_dir), transform=T.get_transforms(train=False),
        crop_padding=cfg.crop_padding, indices=val_i,
    )
    test_obb = T.TeaOBBDataset(
        str(img_dir), str(lbl_dir), transform=T.get_transforms(train=False),
        crop_padding=cfg.crop_padding, indices=test_i,
    )

    text_df = T.load_annotations_csv(cfg.annotations)
    if not {"image_file", "box_idx"}.issubset(text_df.columns):
        raise RuntimeError("Annotations lack the exact (image_file, box_idx) key")
    keys = [
        {full_ds.sample_ids[i] for i in idx} for idx in (trn_i, val_i, test_i)
    ]
    row_keys = list(zip(text_df["image_file"], text_df["box_idx"]))
    splits = [
        text_df.loc[pd.Series([k in key for k in row_keys])].reset_index(drop=True)
        for key in keys
    ]
    text_trn, text_val, text_test = splits
    if any(part.empty for part in splits):
        raise RuntimeError("A split has no paired annotations")

    blocked: list[str] = []
    if cfg.sanitize_target_derived_text:
        blocked = T.fit_label_leakage_vocabulary(
            text_trn,
            min_count=cfg.leakage_token_min_count,
            purity_threshold=cfg.leakage_token_purity,
        )
        text_trn = T.sanitize_annotation_text(text_trn, blocked)
        text_val = T.sanitize_annotation_text(text_val, blocked)
        text_test = T.sanitize_annotation_text(text_test, blocked)

    mm_trn = T.MultiModalDataset(train_obb, text_trn, cfg.max_seq_len, cfg.seed)
    mm_val = T.MultiModalDataset(val_obb, text_val, cfg.max_seq_len, cfg.seed)
    mm_test = T.MultiModalDataset(test_obb, text_test, cfg.max_seq_len, cfg.seed)
    for name, ds in (("train", mm_trn), ("val", mm_val), ("test", mm_test)):
        if ds.pairing_coverage < 1.0:
            raise RuntimeError(f"{name} pairing coverage is {ds.pairing_coverage}")

    return {
        "train_obb": train_obb,
        "mm_trn": mm_trn,
        "mm_val": mm_val,
        "mm_test": mm_test,
        "class_weights": T.compute_class_weights(train_obb.labels, device),
        "audit": {
            "train_crops": len(trn_i),
            "validation_crops": len(val_i),
            "test_crops": len(test_i),
            "source_group_overlap": overlap,
            "masked_shortcut_tokens": len(blocked),
            "test_class_counts": dict(Counter(test_obb.labels)),
        },
    }


def loaders_for(cfg, data, model, device, cache: bool):
    """Cache frozen ResNet views when the variant keeps a frozen ResNet."""
    trn, val, test = data["mm_trn"], data["mm_val"], data["mm_test"]
    if cache:
        trn = T.cache_frozen_vision_features(
            copy.copy(trn), model, device, cfg.batch_size,
            num_views=cfg.vision_cache_views, seed=cfg.seed,
        )
        val = T.cache_frozen_vision_features(
            copy.copy(val), model, device, cfg.batch_size,
            num_views=1, seed=cfg.seed + 10_000,
        )
        test = T.cache_frozen_vision_features(
            copy.copy(test), model, device, cfg.batch_size,
            num_views=1, seed=cfg.seed + 20_000,
        )
        labels = trn.labels
    else:
        labels = data["train_obb"].labels
    sampler = T.BalancedBatchSampler(labels, cfg.batch_size, T.NUM_CLASSES)
    return (
        DataLoader(trn, batch_sampler=sampler, num_workers=0),
        DataLoader(val, batch_size=cfg.batch_size, num_workers=0),
        DataLoader(test, batch_size=cfg.batch_size, num_workers=0),
    )


SEEDS = [0, 1, 2]


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = T.Config(
        data_dir=str(ROOT / "Real Dataset"),
        annotations=str(ROOT / "tea_results" / "annotation" / "annotations.csv"),
        output_dir=str(ROOT / "tea_results"),
        run_federated=False,
        multimodal_only=True,
        cross_modal_analysis=False,
    )
    print(f"device: {device}")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    data = build_data(cfg, device)
    print("data audit:", json.dumps(data["audit"], indent=2))

    OUTPUT.mkdir(parents=True, exist_ok=True)
    partial_path = OUTPUT / "architecture_ablation_partial.json"

    # Resume support: a long CPU sweep must not lose finished variants if the
    # process dies, so each result is flushed as soon as it is produced.
    rows: list[dict[str, Any]] = []
    if partial_path.is_file():
        rows = json.loads(partial_path.read_text(encoding="utf-8"))
        done = {r["variant"] for r in rows}
        print(f"resuming; {len(done)} variant(s) already complete: {sorted(done)}")

    plan = [(f"{key}|seed={sd}", key, label, overrides, sd)
            for key, label, overrides in VARIANTS for sd in SEEDS]
    for run_id, key, label, overrides, sd in plan:
        if any(r["variant"] == run_id for r in rows):
            print(f"  skip {label} seed={sd} (already recorded)")
            continue
        torch.manual_seed(sd)
        np.random.seed(sd)

        kwargs: dict[str, Any] = {
            "num_labels": T.NUM_CLASSES,
            "class_weights": data["class_weights"],
            "max_seq_len": cfg.max_seq_len,
            "modality_dropout": cfg.modality_dropout,
            "image_only_probability": cfg.image_only_probability,
            "text_only_probability": cfg.text_only_probability,
            "vision_backbone": "resnet50",
            "pretrained_vision": True,
            "freeze_vision_backbone": True,
            "text_auxiliary_weight": cfg.text_auxiliary_weight,
            "vision_auxiliary_weight": cfg.vision_auxiliary_weight,
            "alignment_weight": cfg.alignment_weight,
            "text_confidence_guard": None,
        }
        kwargs.update(overrides)

        model = T.MultiModalClassifier(**kwargs).to(device)
        cache = (
            kwargs["vision_backbone"] == "resnet50"
            and kwargs["freeze_vision_backbone"]
        )
        trn_ld, val_ld, test_ld = loaders_for(cfg, data, model, device, cache)

        started = time.time()
        best_val, _, _, best_state = T.train_model(
            model, trn_ld, val_ld, cfg, device, "multimodal"
        )
        if best_state is not None:
            model.load_state_dict(best_state)
        test_met = T.evaluate(model, test_ld, device, "multimodal")
        elapsed = time.time() - started

        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        row = {
            "variant": run_id,
            "component": key,
            "seed": sd,
            "label": label,
            "overrides": {k: str(v) for k, v in overrides.items()},
            "val_macro_f1": float(best_val),
            "test_accuracy": float(test_met["accuracy"]),
            "test_macro_f1": float(test_met["f1_macro"]),
            "test_correct": int(round(test_met["accuracy"] * data["audit"]["test_crops"])),
            "total_parameters": int(total),
            "trainable_parameters": int(trainable),
            "seconds": round(elapsed, 1),
        }
        rows.append(row)
        partial_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(
            f"  {label:<28} val {row['val_macro_f1']:.4f} | "
            f"test acc {row['test_accuracy']:.4f} F1 {row['test_macro_f1']:.4f} "
            f"({row['seconds']}s)"
        )

    base = next(r for r in rows if r["variant"] == "full")
    for row in rows:
        row["delta_accuracy"] = row["test_accuracy"] - base["test_accuracy"]
        row["delta_macro_f1"] = row["test_macro_f1"] - base["test_macro_f1"]

    payload = {
        "experiment": "multimodal_architecture_ablation",
        "protocol": (
            "each variant retrained from scratch on the identical source-grouped "
            "split and scored on the same locked test crops"
        ),
        "seed": cfg.seed,
        "epochs": cfg.epochs,
        "data_audit": data["audit"],
        "variants": rows,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "architecture_ablation_results.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print("wrote", OUTPUT / "architecture_ablation_results.json")


if __name__ == "__main__":
    main()
