#!/usr/bin/env python3
"""Every text, image, and multimodal encoder on the identical 222/74/75 support.

Why this exists
---------------
The paper's text-encoder table mixes supports. Only frozen DistilBERT has a
text-only result on the crop-note split; the BERT-tiny/mini/small/medium rows
come from a different template benchmark and cannot be placed beside it. The
same applies to any image row not measured on these 75 crops.

This script evaluates every encoder on ONE support:

  * 222 train / 74 validation / 75 test crops, source-grouped, seed 42
  * exact (photograph, box) crop-note linkage
  * leakage vocabulary fitted on TRAIN ONLY, then applied to all three splits
  * configuration chosen on the 74 validation crops alone; test is touched once,
    after the choice is frozen

It emits three aligned tables -- text-only, image-only, multimodal -- all on the
same 75 test crops in the same label order, so rows are directly comparable.

Heads are ridge classifiers, matching the paper's frozen-encoder protocol.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import tea_train as tea  # noqa: E402

TEXT_ENCODERS = {
    "DistilBERT": "distilbert-base-uncased",
    "BERT-tiny": "prajjwal1/bert-tiny",
    "BERT-mini": "prajjwal1/bert-mini",
    "BERT-small": "prajjwal1/bert-small",
    "BERT-medium": "prajjwal1/bert-medium",
}
IMAGE_ENCODERS = {
    "ViT-Base": "google/vit-base-patch16-224",
    "DeiT-tiny": "facebook/deit-tiny-patch16-224",
    "Swin-tiny": "microsoft/swin-tiny-patch4-window7-224",
    "ConvNeXT-tiny": "facebook/convnext-tiny-224",
    "EfficientNet": "google/efficientnet-b0",
}
RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]


# --------------------------------------------------------------------------
def build_split(seed: int = 42):
    """The paper's audited split: source-grouped, crop-linked, train-only mask."""
    import pandas as pd

    data_dir = ROOT / "Real Dataset"
    ann = ROOT / "tea_results" / "annotation" / "annotations.csv"
    full = tea.TeaOBBDataset(str(data_dir / "images"), str(data_dir / "labels"),
                             transform=tea.get_transforms(train=False),
                             crop_padding=0.10)
    tr, va, te = tea.grouped_train_val_test_split(full.labels, full.groups,
                                                  0.20, 0.20, seed)
    text_df = tea.load_annotations_csv(str(ann))
    row_keys = list(zip(text_df["image_file"], text_df["box_idx"]))

    def slice_for(idxs):
        keys = {full.sample_ids[i] for i in idxs}
        return text_df.loc[pd.Series([k in keys for k in row_keys])].reset_index(drop=True)

    parts = {"train": slice_for(tr), "validation": slice_for(va), "test": slice_for(te)}
    blocked = tea.fit_label_leakage_vocabulary(parts["train"], min_count=3,
                                               purity_threshold=0.95)
    parts = {k: tea.sanitize_annotation_text(v, blocked) for k, v in parts.items()}
    obb = {n: tea.TeaOBBDataset(str(data_dir / "images"), str(data_dir / "labels"),
                                transform=tea.get_transforms(train=False),
                                crop_padding=0.10, indices=i)
           for n, i in (("train", tr), ("validation", va), ("test", te))}
    audit = {"crops": {"train": len(tr), "validation": len(va), "test": len(te)},
             "source_groups": {n: len({full.groups[i] for i in idx})
                               for n, idx in (("train", tr), ("validation", va),
                                              ("test", te))},
             "blocked_tokens": len(blocked), "split_seed": seed,
             "pairing": "exact (photograph, box) crop-note linkage"}
    return parts, obb, audit


def load_text_encoder(model_id):
    """AutoModel, falling back to BertModel for configs without model_type."""
    from transformers import AutoModel, BertConfig, BertModel
    try:
        return AutoModel.from_pretrained(model_id)
    except ValueError as err:
        if "model_type" not in str(err):
            raise
        cfg = BertConfig.from_pretrained(model_id)
        return BertModel.from_pretrained(model_id, config=cfg)


@torch.inference_mode()
def text_features(texts: Sequence[str], model_id, device, batch_size=32):
    from transformers import AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained(model_id)
    except Exception:
        # the prajjwal1 BERT checkpoints ship no tokenizer.json, so the fast
        # backend cannot be built; fall back to the slow WordPiece tokenizer,
        # and finally to bert-base-uncased whose vocab they share
        try:
            tok = AutoTokenizer.from_pretrained(model_id, use_fast=False)
        except Exception:
            tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    enc = load_text_encoder(model_id).to(device).eval()
    out = []
    for s in range(0, len(texts), batch_size):
        b = tok(list(texts[s:s + batch_size]), max_length=128, padding="max_length",
                truncation=True, return_tensors="pt").to(device)
        h = enc(**b).last_hidden_state
        w = b["attention_mask"].unsqueeze(-1).to(h.dtype)
        out.append(((h * w).sum(1) / w.sum(1).clamp_min(1.0)).cpu().numpy())
    del enc
    torch.cuda.empty_cache()
    return np.concatenate(out).astype(np.float64)


@torch.inference_mode()
def image_features(obb_ds, model_id, device, batch_size=32):
    from torch.utils.data import DataLoader
    from transformers import AutoModel
    enc = AutoModel.from_pretrained(model_id).to(device).eval()
    out = []
    for batch in DataLoader(obb_ds, batch_size=batch_size, shuffle=False):
        o = enc(pixel_values=batch["pixel_values"].to(device))
        h = o.last_hidden_state
        pooled = getattr(o, "pooler_output", None)
        if h is not None and h.dim() == 3:
            feat = h[:, 0, :]
        elif h is not None and h.dim() == 4:
            feat = h.mean(dim=(-2, -1))
        else:
            feat = pooled
        out.append(feat.cpu().numpy())
    del enc
    torch.cuda.empty_cache()
    return np.concatenate(out).astype(np.float64)


def metrics(y_true, y_pred) -> Dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
    }


def ridge_select_and_test(Xtr, ytr, Xva, yva, Xte, yte):
    """Pick the ridge penalty on validation only, then score test once."""
    from sklearn.linear_model import Ridge
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-8
    Ztr, Zva, Zte = (Xtr - mu) / sd, (Xva - mu) / sd, (Xte - mu) / sd
    onehot = np.eye(tea.NUM_CLASSES)[ytr]
    best = None
    for a in RIDGE_ALPHAS:
        clf = Ridge(alpha=a).fit(Ztr, onehot)
        val = metrics(yva, clf.predict(Zva).argmax(1))
        key = (val["macro_f1"], val["accuracy"], -a)
        if best is None or key > best[0]:
            best = (key, a, clf, val)
    _, alpha, clf, val = best
    test = metrics(yte, clf.predict(Zte).argmax(1))
    return {"ridge_alpha": alpha,
            "validation_accuracy": val["accuracy"],
            "validation_macro_f1": val["macro_f1"],
            "test_accuracy": test["accuracy"],
            "test_macro_f1": test["macro_f1"],
            "test_micro_f1": test["micro_f1"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tea_results/aligned_support_tables")
    a = ap.parse_args()
    out_dir = ROOT / a.out
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parts, obb, audit = build_split()
    print(f"device: {device}\nsplit audit: {audit}\n")
    lab = lambda v: int(v[0]) if isinstance(v, (list, tuple)) else int(v)
    y = {k: np.array([lab(v) for v in parts[k]["labels"]]) for k in parts}
    assert (len(y["train"]), len(y["validation"]), len(y["test"])) == (222, 74, 75), \
        f"unexpected support {tuple(len(v) for v in y.values())}"

    results = {"_meta": {**audit,
                         "protocol": "frozen encoder + ridge head; penalty chosen "
                                     "on the 74 validation crops only; test "
                                     "evaluated once after the choice was frozen",
                         "identical_support_asserted": True}}

    # ---- text ------------------------------------------------------------
    tfeat, results["text_only"] = {}, {}
    for name, mid in TEXT_ENCODERS.items():
        try:
            f = {k: text_features(parts[k]["text"].tolist(), mid, device) for k in parts}
        except Exception as exc:
            print(f"  text {name}: FAILED {exc}"); continue
        tfeat[name] = f
        r = ridge_select_and_test(f["train"], y["train"], f["validation"],
                                  y["validation"], f["test"], y["test"])
        r["encoder"], r["dim"] = mid, int(f["train"].shape[1])
        results["text_only"][name] = r
        print(f"  text  {name:<14} val_acc={r['validation_accuracy']:.4f} "
              f"test_acc={r['test_accuracy']:.4f} test_macroF1={r['test_macro_f1']:.4f}")

    # ---- image -----------------------------------------------------------
    ifeat, results["image_only"] = {}, {}
    for name, mid in IMAGE_ENCODERS.items():
        try:
            f = {k: image_features(obb[k], mid, device) for k in parts}
        except Exception as exc:
            print(f"  image {name}: FAILED {exc}"); continue
        ifeat[name] = f
        r = ridge_select_and_test(f["train"], y["train"], f["validation"],
                                  y["validation"], f["test"], y["test"])
        r["encoder"], r["dim"] = mid, int(f["train"].shape[1])
        results["image_only"][name] = r
        print(f"  image {name:<14} val_acc={r['validation_accuracy']:.4f} "
              f"test_acc={r['test_accuracy']:.4f} test_macroF1={r['test_macro_f1']:.4f}")

    # ---- multimodal: every text encoder against the validation-best image --
    results["multimodal"] = {}
    if tfeat and ifeat:
        best_img = max(ifeat, key=lambda n: (
            results["image_only"][n]["validation_macro_f1"],
            results["image_only"][n]["validation_accuracy"]))
        print(f"\n  validation-selected image encoder for fusion: {best_img}")
        results["_meta"]["fusion_image_encoder"] = best_img
        for name in tfeat:
            f = {k: np.hstack([ifeat[best_img][k], tfeat[name][k]]) for k in parts}
            r = ridge_select_and_test(f["train"], y["train"], f["validation"],
                                      y["validation"], f["test"], y["test"])
            r["dim"] = int(f["train"].shape[1])
            label = f"{best_img} + {name}"
            results["multimodal"][label] = r
            print(f"  fused {label:<28} val_acc={r['validation_accuracy']:.4f} "
                  f"test_acc={r['test_accuracy']:.4f} test_macroF1={r['test_macro_f1']:.4f}")

    path = out_dir / "aligned_support_tables.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {path}")

    for section in ("text_only", "image_only", "multimodal"):
        rows = results.get(section, {})
        if not rows:
            continue
        print(f"\n=== {section} — identical 75-crop test support ===")
        print(f"  {'model':<30}{'val acc':<10}{'test acc':<10}{'test macro F1':<14}")
        for n, r in sorted(rows.items(),
                           key=lambda kv: -kv[1]["validation_macro_f1"]):
            print(f"  {n:<30}{r['validation_accuracy']:<10.4f}"
                  f"{r['test_accuracy']:<10.4f}{r['test_macro_f1']:<14.4f}")


if __name__ == "__main__":
    main()
