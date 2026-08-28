#!/usr/bin/env python3
"""Modality corruption on the locked test, holding the checkpoint fixed.

Table VII was the one live table with no generating script in the repository,
so it kept the numbers of a checkpoint trained on the pre-correction notes. This
rebuilds it against whatever checkpoint is passed, reusing the split, the
tokenizer masking and the model loader that the hierarchy benchmark already
uses, so the support and the note masking are identical to the rest of the
paper.

One trained model is held constant and the *input* is varied, so a change in
prediction is attributable to the evidence the model consumes rather than to
retraining:

  text deletion   the deterministic blake2b token mask, at keep rates
                  1 - p. Deletion is keyed on (seed, image, box, position,
                  token), so every condition deletes the same tokens of the
                  same notes that the rest of the paper deletes. At p = 1 every
                  content token is masked and only [CLS]/[SEP] survive.

  image Gaussian  additive noise at sigma, applied to the normalized pixel
                  tensor before the vision backbone -- not to the cached
                  feature map, which would corrupt the wrong representation.

  image zeroed    the pixel tensor replaced by zeros
  image uniform   the pixel tensor replaced by uniform noise on [0, 1),
                  renormalized, so the image carries no signal but retains
                  the statistics the backbone expects

Flips count predictions differing from the clean run on the same crop, which is
the quantity that distinguishes "the model ignores this modality" from "the
model uses it and the corruption happens not to change the argmax".
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
import tea_train as training  # noqa: E402
from modality_hierarchy_benchmark import metrics, sparse_attention  # noqa: E402
from vision_tta import load_model, prepare_partitions  # noqa: E402

TEXT_DELETION = (0.0, 0.25, 0.50, 0.75, 1.00)
GAUSSIAN_SIGMA = (0.09, 0.18, 0.26, 0.35)


def build_cache(model, args, full_dataset, indices, text_df, device,
                corrupt=None):
    """Cache backbone features, optionally corrupting pixels on the way in."""
    obb = training.TeaOBBDataset(
        str(Path(args.data_dir) / "images"),
        str(Path(args.data_dir) / "labels"),
        transform=training.get_transforms(train=False),
        crop_padding=args.crop_padding,
        indices=indices,
    )
    dataset = training.MultiModalDataset(
        obb, text_df, max_length=args.max_seq_len, seed=args.seed,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=0)
    ids, masks, labels, maps = [], [], [], []
    generator = torch.Generator(device="cpu").manual_seed(args.corruption_seed)
    with torch.inference_mode():
        for batch in loader:
            pixels = batch["pixel_values"]
            if corrupt is not None:
                pixels = corrupt(pixels, generator)
            maps.append(model.extract_vision_backbone(pixels.to(device)).cpu())
            ids.append(batch["input_ids"])
            masks.append(batch["attention_mask"])
            labels.append(batch["labels"].argmax(dim=1))
    return {
        "input_ids": torch.cat(ids),
        "attention_mask": torch.cat(masks),
        "vision_feature_map": torch.cat(maps),
        "labels": torch.cat(labels).numpy(),
        "sample_ids": [full_dataset.sample_ids[i] for i in indices],
    }


def predict(model, cache, attention, args, device):
    """Predictions plus the mean text reliability weight the gate assigns.

    r_t is reported alongside accuracy because it separates "the model stopped
    using this modality" from "the model kept weighting it and was wrong": a
    gate that leaves r_t unchanged under a destroyed modality is not reacting to
    the corruption at all.
    """
    out, weights = [], []
    with torch.inference_mode():
        for s in range(0, len(cache["labels"]), args.batch_size):
            e = s + args.batch_size
            mm = torch.tensor((1.0, 1.0), dtype=torch.float32,
                              device=device).expand(
                                  cache["input_ids"][s:e].size(0), -1)
            o = model(
                input_ids=cache["input_ids"][s:e].to(device),
                attention_mask=attention[s:e].to(device),
                vision_feature_map=cache["vision_feature_map"][s:e].to(device),
                modality_mask=mm,
            )
            out.append(F.softmax(o["logits"], dim=-1).cpu().numpy())
            w = o.get("modality_weights")
            if w is not None:
                weights.append(w.detach().cpu().numpy())
    r_t = float(np.concatenate(weights)[:, 0].mean()) if weights else None
    return np.concatenate(out).argmax(axis=1), r_t


def content_free_mask(cache):
    """Mask every content token, leaving only [CLS] and [SEP]."""
    ids, original = cache["input_ids"], cache["attention_mask"]
    out = original.clone()
    for r in range(ids.shape[0]):
        for p in range(ids.shape[1]):
            if original[r, p] and int(ids[r, p]) not in (0, 101, 102):
                out[r, p] = 0
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_results", type=Path, required=True)
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--sparsity_seed", type=int, default=20260727)
    ap.add_argument("--corruption_seed", type=int, default=20260827)
    args = ap.parse_args()

    payload = json.loads(args.base_results.read_text(encoding="utf-8"))
    for field in ("data_dir", "annotations", "crop_padding", "val_split",
                  "test_split", "seed", "max_seq_len",
                  "leakage_token_min_count", "leakage_token_purity"):
        setattr(args, field, payload["config"][field])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_model(args.base_results.parent / "models" / "best_vlm.pt",
                          device)
    full, _, test_idx, _, test_text, blocked = prepare_partitions(args)
    print(f"device {device}; locked test {len(test_idx)} crops; "
          f"{len(blocked)} masked tokens")

    clean_cache = build_cache(model, args, full, test_idx, test_text, device)
    labels = clean_cache["labels"]
    clean_pred, clean_rt = predict(model, clean_cache,
                                  clean_cache["attention_mask"], args, device)
    clean_m = metrics(labels, clean_pred)
    rows = [("Clean", clean_m, None, clean_rt)]
    print(f"  clean: acc {clean_m['accuracy']:.4f}  "
          f"macro-F1 {clean_m['macro_f1']:.4f}")

    for p in TEXT_DELETION:
        if p == 0.0:
            continue
        att = (content_free_mask(clean_cache) if p >= 1.0
               else sparse_attention(clean_cache, 1.0 - p, args.sparsity_seed))
        pred, rt = predict(model, clean_cache, att, args, device)
        m = metrics(labels, pred)
        rows.append((f"Text deletion {100 * p:.0f}", m,
                     int((pred != clean_pred).sum()), rt))
        print(f"  text -{100 * p:.0f}%: acc {m['accuracy']:.4f}  "
              f"F1 {m['macro_f1']:.4f}  flips {rows[-1][2]}")

    def gaussian(sigma):
        def f(px, gen):
            return px + torch.randn(px.shape, generator=gen) * sigma
        return f

    def zeroed(px, gen):
        return torch.zeros_like(px)

    def uniform(px, gen):
        raw = torch.rand(px.shape, generator=gen)
        return (raw - raw.mean()) / raw.std().clamp_min(1e-6) * px.std() + px.mean()

    conditions = [(f"Image Gaussian $\\sigma{{=}}{s:.2f}$", gaussian(s))
                  for s in GAUSSIAN_SIGMA]
    conditions += [("Image zeroed", zeroed), ("Image uniform noise", uniform)]
    for name, fn in conditions:
        cache = build_cache(model, args, full, test_idx, test_text, device,
                            corrupt=fn)
        pred, rt = predict(model, cache, cache["attention_mask"], args, device)
        m = metrics(labels, pred)
        rows.append((name, m, int((pred != clean_pred).sum()), rt))
        print(f"  {name}: acc {m['accuracy']:.4f}  F1 {m['macro_f1']:.4f}  "
              f"flips {rows[-1][2]}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "experiment": "modality_corruption_locked_test",
        "checkpoint": str(args.base_results.parent / "models" / "best_vlm.pt"),
        "test_crops": int(len(labels)),
        "masked_training_fitted_tokens": len(blocked),
        "sparsity_seed": args.sparsity_seed,
        "corruption_seed": args.corruption_seed,
        "conditions": [
            {"condition": n, "accuracy": m["accuracy"],
             "macro_f1": m["macro_f1"], "flips": f, "mean_text_weight": w}
            for n, m, f, w in rows
        ],
    }
    path = args.output_dir / "modality_corruption_results.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {path}")
    print("Done.")


if __name__ == "__main__":
    main()
