#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_fed_multimodal.py — federated training for the multimodal FarmFederate model.

- Text data: mix of HF agri corpora + synthetic LocalMini + weather-grounded logs
- Weather: every sample is paired with a real AMFU Kharagpur (IMD 42893) day
  (see weather_data.py); FARMFED_WEATHER_MODE=full|text-only|none selects the ablation
- Images: multiple plant stress / disease datasets from HF (PlantVillage mirrors + others)
- FedAvg over non-IID clients (Dirichlet)
- Saves checkpoints/global_round{r}.pt (and final .pt) with val metrics per round

This is the federated counterpart to multimodal_train.py
adapted for your backend + Flutter demo.
"""

import os
import json
from typing import List

import numpy as np
import torch
from transformers import AutoTokenizer

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
    split_clients_dirichlet,
    train_one_client,
    make_weights_for_balanced_classes,
    evaluate_model,
)
from weather_data import WEATHER_FEATURES

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 123
np.random.seed(SEED)
torch.manual_seed(SEED)


def fedavg_weighted(states: List[dict], sizes: List[int]):
    total = float(sum(sizes))
    weights = [s / total for s in sizes]
    out = {}
    keys = states[0].keys()
    for k in keys:
        stacked = torch.stack([st[k].float() * w for st, w in zip(states, weights)], dim=0)
        out[k] = stacked.sum(dim=0)
    return out


def main():
    rounds = int(os.environ.get("FED_ROUNDS", 2))
    clients = int(os.environ.get("FED_CLIENTS", 4))
    local_epochs = int(os.environ.get("FED_LOCAL_EPOCHS", 1))
    batch_size = int(os.environ.get("FED_BATCH_SIZE", 16))
    lr = 3e-5
    dirichlet_alpha = 0.25
    max_images = int(os.environ.get("FED_MAX_IMAGES", 20000))

    weather_mode = os.environ.get("FARMFED_WEATHER_MODE", "full")
    sampler = configure_weather(weather_mode)
    save_dir = os.path.join(os.environ.get("FED_OUT", "checkpoints"), f"fed_{weather_mode}")
    os.makedirs(save_dir, exist_ok=True)

    print("Building text corpus (mix of HF + LocalMini + weather logs)...")
    df = build_text_corpus_mix(
        mix_sources="gardian,argilla,agnews,localmini,weather",
        max_per_source=2000,
        max_samples=6000,
    )
    summarize_labels(df, "global")

    # train/val split
    df_train = df.sample(frac=0.85, random_state=SEED)
    df_val = df.drop(df_train.index).reset_index(drop=True)
    df_train = df_train.reset_index(drop=True)

    # class weights for FocalLoss alpha
    _, counts = make_weights_for_balanced_classes(df_train)
    inv = 1.0 / np.maximum(counts, 1)
    alpha = (inv / inv.mean()).astype(np.float32)
    alpha = torch.tensor(alpha)

    # images: multiple datasets (PlantVillage mirror + others)
    image_ds = load_stress_image_datasets_hf(
        max_total_images=max_images,
        max_per_dataset=min(6000, max_images),
    )

    # tokenizer / image processor
    text_model_name = "roberta-base"
    image_model_name = "google/vit-base-patch16-224-in21k"
    tokenizer = AutoTokenizer.from_pretrained(text_model_name)
    image_processor = build_image_processor(image_model_name)

    # global model
    global_model = MultimodalClassifier(
        text_model_name=text_model_name,
        image_model_name=image_model_name,
        num_labels=NUM_LABELS,
        freeze_backbones=False,
    ).to(DEVICE)

    # federated client splits on train set
    client_dfs = split_clients_dirichlet(df_train, n_clients=clients, alpha=dirichlet_alpha)

    print(f"Non-IID clients sizes: {[len(cdf) for cdf in client_dfs]}")

    val_loader = torch.utils.data.DataLoader(
        MultiModalDataset(df_val, tokenizer, image_processor, image_ds, max_len=160),
        batch_size=32, shuffle=False, num_workers=0,
    )
    history = []

    for r in range(1, rounds + 1):
        print(f"\n=== Federated round {r}/{rounds} ===")
        states = []
        sizes = []

        for cid, cdf in enumerate(client_dfs):
            if len(cdf) < 100:
                print(f"[client {cid}] skipped (too small: n={len(cdf)})")
                continue

            # local train/val split for this client
            n = len(cdf)
            val_n = max(1, int(0.15 * n))
            df_c_val = cdf.iloc[:val_n].reset_index(drop=True)
            df_c_train = cdf.iloc[val_n:].reset_index(drop=True)
            print(f"[client {cid}] train={len(df_c_train)}, val={len(df_c_val)}")

            # fresh model copy
            local_model = MultimodalClassifier(
                text_model_name=text_model_name,
                image_model_name=image_model_name,
                num_labels=NUM_LABELS,
                freeze_backbones=False,
            ).to(DEVICE)

            local_model.load_state_dict(global_model.state_dict(), strict=True)

            state_cpu, train_loss, val_loss = train_one_client(
                model=local_model,
                df_train=df_c_train,
                df_val=df_c_val,
                tokenizer=tokenizer,
                image_processor=image_processor,
                image_ds=image_ds,
                max_len=160,
                batch_size=batch_size,
                local_epochs=local_epochs,
                lr=lr,
                device=DEVICE,
                grad_accum=1,
                weight_decay=0.01,
                alpha_per_class=alpha,
            )

            print(
                f"[client {cid}] n={len(cdf)} "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f}"
            )

            states.append(state_cpu)
            sizes.append(len(df_c_train))

            del local_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if not states:
            print("No client updates this round, skipping FedAvg.")
            continue

        # FedAvg update
        avg_state = fedavg_weighted(states, sizes)
        global_state = global_model.state_dict()
        for k, v in avg_state.items():
            if k in global_state:
                global_state[k] = v
        global_model.load_state_dict(global_state)

        metrics = evaluate_model(global_model, val_loader, device=DEVICE)
        metrics = {k: float(v) for k, v in metrics.items()}
        history.append({"round": r, "clients": len(states), **metrics})
        print(f"[round {r}] val f1_macro={metrics['f1_macro']:.4f} f1_micro={metrics['f1_micro']:.4f}")

        ckpt_path = os.path.join(save_dir, f"global_round{r}.pt")
        torch.save(global_model.state_dict(), ckpt_path)
        print(f"[save] round {r} → {ckpt_path}")

    final_path = os.path.join(save_dir, "global_round_final.pt")
    torch.save({
        "model_state_dict": global_model.state_dict(),
        "model_type": "VLM",
        "fusion_type": "cross_attention",
        "f1_score": history[-1]["f1_macro"] if history else None,
        "labels": ISSUE_LABELS,
        "weather_mode": weather_mode,
        "weather_features": WEATHER_FEATURES if (sampler is not None and weather_mode == "full") else [],
        "history": history,
    }, final_path)
    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump({"weather_mode": weather_mode, "rounds": rounds, "clients": clients,
                   "history": history}, f, indent=2)
    print(f"[save] final model → {final_path}")


if __name__ == "__main__":
    main()
