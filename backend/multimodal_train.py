#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
multimodal_train.py — centralized training of the multimodal model.

- Builds a mixed text corpus from HF datasets + synthetic logs + weather-grounded logs.
- Pairs every sample with a real observed day from the AMFU Kharagpur (IMD 42893)
  station: SENSORS/WEATHER lines in the text, a numeric weather vector for the
  model's weather branch, and agromet-rule labels merged into the weak labels.
- Uses multiple plant-stress image datasets via Hugging Face
  (merged by load_stress_image_datasets_hf).
- Trains MultimodalClassifier end-to-end (no federated splitting), evaluates on a
  held-out split every epoch and keeps the best macro-F1 checkpoint.
- Saves <out>/global_central.pt (+ metrics.json, history.json).

Weather ablations (same corpus recipe, different use of the station data):
    --weather-mode full       text line + numeric branch + rule labels   (default)
    --weather-mode text-only  text line + rule labels, numeric branch off
    --weather-mode none       legacy Gaussian sensors, no station data
"""

import os
import sys
import json
import time
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from multimodal_model import MultimodalClassifier, build_image_processor
from datasets_loader import (
    build_text_corpus_mix,
    load_stress_image_datasets_hf,
    summarize_labels,
    configure_weather,
    ISSUE_LABELS,
    NUM_LABELS,
)
from federated_core import (
    MultiModalDataset,
    make_weights_for_balanced_classes,
    FocalLoss,
    weather_batch,
)
from weather_data import WEATHER_FEATURES

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 123


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="checkpoints", help="output directory")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--max-len", type=int, default=160)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--mix-sources", default="gardian,argilla,agnews,localmini,weather")
    ap.add_argument("--max-per-source", type=int, default=2000)
    ap.add_argument("--max-samples", type=int, default=6000)
    ap.add_argument("--max-images", type=int, default=20000)
    ap.add_argument("--max-per-image-dataset", type=int, default=6000)
    ap.add_argument("--no-images", action="store_true", help="skip HF image download (gray dummy image)")
    ap.add_argument("--text-model", default="roberta-base")
    ap.add_argument("--image-model", default="google/vit-base-patch16-224-in21k")
    ap.add_argument("--freeze-backbones", action="store_true")
    ap.add_argument("--weather-mode", choices=["full", "text-only", "none"], default="full")
    ap.add_argument("--weather-xlsx", default=None, help="path to the KGP met workbook (default: repo copy)")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--amp", action="store_true", help="mixed precision on CUDA")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--tag", default="", help="suffix for the run directory name")
    return ap.parse_args()


def make_loader(df, tokenizer, image_processor, image_ds, args, shuffle):
    ds = MultiModalDataset(df, tokenizer, image_processor, image_ds, max_len=args.max_len)
    return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                      num_workers=args.num_workers, pin_memory=DEVICE == "cuda")


@torch.no_grad()
def evaluate(model, loader, loss_fn):
    from sklearn.metrics import f1_score, precision_score, recall_score

    model.eval()
    probs, trues, total_loss = [], [], 0.0
    for batch in loader:
        labels = batch["labels"].to(DEVICE)
        out = model(
            input_ids=batch["input_ids"].to(DEVICE),
            attention_mask=batch["attention_mask"].to(DEVICE),
            pixel_values=batch["pixel_values"].to(DEVICE),
            weather_features=weather_batch(batch, DEVICE),
        )
        total_loss += loss_fn(out.logits, labels).item() * labels.size(0)
        probs.append(torch.sigmoid(out.logits).float().cpu().numpy())
        trues.append(labels.cpu().numpy())
    probs, trues = np.vstack(probs), np.vstack(trues)
    preds = (probs > 0.5).astype(int)
    per_label = f1_score(trues, preds, average=None, zero_division=0)
    return {
        "loss": total_loss / max(1, len(trues)),
        "f1_macro": float(f1_score(trues, preds, average="macro", zero_division=0)),
        "f1_micro": float(f1_score(trues, preds, average="micro", zero_division=0)),
        "precision_macro": float(precision_score(trues, preds, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(trues, preds, average="macro", zero_division=0)),
        "f1_per_label": {k: float(v) for k, v in zip(ISSUE_LABELS, per_label)},
        "support": {k: int(v) for k, v in zip(ISSUE_LABELS, trues.sum(axis=0))},
    }


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    run_name = f"central_{args.weather_mode}{('_' + args.tag) if args.tag else ''}"
    save_dir = os.path.join(args.out, run_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"[run] {run_name} → {save_dir} (device={DEVICE})")

    # 0) weather modality
    sampler = configure_weather(args.weather_mode, args.weather_xlsx)
    use_weather_branch = sampler is not None and args.weather_mode == "full"

    # 1) text corpus (+ paired weather days)
    df = build_text_corpus_mix(
        mix_sources=args.mix_sources,
        max_per_source=args.max_per_source,
        max_samples=args.max_samples,
    )
    summarize_labels(df, "all")
    df_train = df.sample(frac=1.0 - args.val_frac, random_state=args.seed)
    df_val = df.drop(df_train.index).reset_index(drop=True)
    df_train = df_train.reset_index(drop=True)
    print(f"[data] train={len(df_train)} val={len(df_val)} weather_column={'weather' in df.columns}")

    # 2) images: merged HF plant stress datasets (auto-downloadable)
    image_ds = None
    if not args.no_images:
        image_ds = load_stress_image_datasets_hf(
            max_total_images=args.max_images,
            max_per_dataset=args.max_per_image_dataset,
        )

    # 3) model + tokenizers
    tokenizer = AutoTokenizer.from_pretrained(args.text_model)
    image_processor = build_image_processor(args.image_model)
    model = MultimodalClassifier(
        text_model_name=args.text_model,
        image_model_name=args.image_model,
        num_labels=NUM_LABELS,
        freeze_backbones=args.freeze_backbones,
    ).to(DEVICE)

    # 4) dataloaders
    train_loader = make_loader(df_train, tokenizer, image_processor, image_ds, args, shuffle=True)
    val_loader = make_loader(df_val, tokenizer, image_processor, image_ds, args, shuffle=False)

    # 5) loss + optim
    _, counts = make_weights_for_balanced_classes(df_train)
    inv = 1.0 / np.maximum(counts, 1)
    alpha = torch.tensor((inv / inv.mean()).astype(np.float32))
    loss_fn = FocalLoss(alpha=alpha.to(DEVICE))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    use_amp = args.amp and DEVICE == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print(f"Training on {len(train_loader.dataset)} samples, device={DEVICE}, "
          f"weather_branch={use_weather_branch}, amp={use_amp}")

    history, best_f1, ckpt_path = [], -1.0, os.path.join(save_dir, "global_central.pt")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, t0 = 0.0, time.time()
        for step, batch in enumerate(train_loader, start=1):
            opt.zero_grad(set_to_none=True)
            labels = batch["labels"].to(DEVICE)
            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(
                    input_ids=batch["input_ids"].to(DEVICE),
                    attention_mask=batch["attention_mask"].to(DEVICE),
                    pixel_values=batch["pixel_values"].to(DEVICE),
                    weather_features=weather_batch(batch, DEVICE),
                )
                loss = loss_fn(out.logits.float(), labels)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            total_loss += loss.item()
            if step % 50 == 0:
                print(f"  [epoch {epoch} step {step}/{len(train_loader)}] loss={total_loss / step:.4f}", flush=True)

        train_loss = total_loss / max(1, len(train_loader))
        val = evaluate(model, val_loader, loss_fn)
        gate = float(torch.tanh(model.weather_gate).item()) if use_weather_branch else None
        rec = {"epoch": epoch, "train_loss": train_loss, "val": val, "weather_gate": gate,
               "seconds": round(time.time() - t0, 1)}
        history.append(rec)
        print(f"[epoch {epoch}] train_loss={train_loss:.4f} val_loss={val['loss']:.4f} "
              f"val_f1_macro={val['f1_macro']:.4f} val_f1_micro={val['f1_micro']:.4f} "
              f"per_label={ {k: round(v, 3) for k, v in val['f1_per_label'].items()} } gate={gate}", flush=True)

        if val["f1_macro"] > best_f1:
            best_f1 = val["f1_macro"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_type": "VLM",
                "fusion_type": "cross_attention",
                "f1_score": best_f1,
                "epoch": epoch,
                "labels": ISSUE_LABELS,
                "weather_mode": args.weather_mode,
                "weather_features": WEATHER_FEATURES if use_weather_branch else [],
                "text_model": args.text_model,
                "image_model": args.image_model,
                "val_metrics": val,
            }, ckpt_path)
            print(f"[save] best model (f1_macro={best_f1:.4f}) → {ckpt_path}")

        with open(os.path.join(save_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

    best = max(history, key=lambda r: r["val"]["f1_macro"])
    summary = {
        "run": run_name, "weather_mode": args.weather_mode, "device": DEVICE,
        "train_size": len(df_train), "val_size": len(df_val),
        "images": 0 if image_ds is None else len(image_ds),
        "best_epoch": best["epoch"], "best_val": best["val"], "args": vars(args),
    }
    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[done] best epoch {best['epoch']} val_f1_macro={best['val']['f1_macro']:.4f} → {save_dir}/metrics.json")


if __name__ == "__main__":
    main()
