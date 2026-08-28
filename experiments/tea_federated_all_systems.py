#!/usr/bin/env python3
"""FedAvg across every system the paper reports, not only ResNet+compact.

Why this exists
---------------
tea_federated_adaptation.py federates one architecture: the ResNet-50 plus
compact-Transformer VLM. The remaining systems in the paper's comparison table
are frozen encoders followed by a closed-form sklearn Ridge/linear head, and a
closed-form solve has no gradient updates for FedAvg to average. They were
therefore never federated, so every federated claim in the paper is really a
claim about one network.

This script federates them as linear probes: the encoder stays frozen (exactly
as the paper describes those systems) and the head is trained by SGD, which is
what makes FedAvg well defined. The head is consequently NOT the same estimator
as the centralized Ridge head, so federated and centralized rows for those
systems are not the same fit. That is recorded in the output rather than left
implicit.

Within a cell every system trains on the IDENTICAL Dirichlet client partition,
so a difference between systems is the system and not the split.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import tea_train as tea  # noqa: E402


# --------------------------------------------------------------------------
# split -- the paper's audited, source-grouped, leakage-masked protocol
# --------------------------------------------------------------------------
def build_split(seed: int = 42):
    import pandas as pd

    data_dir = ROOT / "Real Dataset"
    ann = ROOT / "tea_results" / "annotation" / "annotations.csv"
    full = tea.TeaOBBDataset(
        str(data_dir / "images"), str(data_dir / "labels"),
        transform=tea.get_transforms(train=False), crop_padding=0.10,
    )
    tr, va, te = tea.grouped_train_val_test_split(
        full.labels, full.groups, 0.20, 0.20, seed
    )
    text_df = tea.load_annotations_csv(str(ann))
    row_keys = list(zip(text_df["image_file"], text_df["box_idx"]))

    def slice_for(idxs):
        keys = {full.sample_ids[i] for i in idxs}
        mask = pd.Series([k in keys for k in row_keys])
        return text_df.loc[mask].reset_index(drop=True)

    parts = {
        "train": slice_for(tr),
        "validation": slice_for(va),
        "test": slice_for(te),
    }
    blocked = tea.fit_label_leakage_vocabulary(
        parts["train"], min_count=3, purity_threshold=0.95
    )
    parts = {k: tea.sanitize_annotation_text(v, blocked) for k, v in parts.items()}
    obb = {
        name: tea.TeaOBBDataset(
            str(data_dir / "images"), str(data_dir / "labels"),
            transform=tea.get_transforms(train=False),
            crop_padding=0.10, indices=idx,
        )
        for name, idx in (("train", tr), ("validation", va), ("test", te))
    }
    audit = {
        "train": len(tr), "validation": len(va), "test": len(te),
        "source_groups": {
            n: len({full.groups[i] for i in idx})
            for n, idx in (("train", tr), ("validation", va), ("test", te))
        },
        "blocked_tokens": len(blocked),
        "split_seed": seed,
    }
    return parts, obb, audit


# --------------------------------------------------------------------------
# frozen encoders -- extracted once, cached, never trained
# --------------------------------------------------------------------------
@torch.inference_mode()
def frozen_text_features(texts: Sequence[str], device,
                         model_id: str = "distilbert-base-uncased",
                         batch_size: int = 32) -> np.ndarray:
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    enc = AutoModel.from_pretrained(model_id).to(device).eval()
    out = []
    for s in range(0, len(texts), batch_size):
        b = tok(list(texts[s:s + batch_size]), max_length=128,
                padding="max_length", truncation=True,
                return_tensors="pt").to(device)
        h = enc(**b).last_hidden_state
        w = b["attention_mask"].unsqueeze(-1).to(h.dtype)
        out.append(((h * w).sum(1) / w.sum(1).clamp_min(1.0)).cpu().numpy())
    del enc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.concatenate(out).astype(np.float32)


@torch.inference_mode()
def frozen_image_features(obb_ds, device,
                          model_id: str = "facebook/deit-tiny-patch16-224",
                          batch_size: int = 32) -> np.ndarray:
    from torch.utils.data import DataLoader
    from transformers import AutoModel
    enc = AutoModel.from_pretrained(model_id).to(device).eval()
    out = []
    for batch in DataLoader(obb_ds, batch_size=batch_size, shuffle=False):
        h = enc(pixel_values=batch["pixel_values"].to(device)).last_hidden_state
        out.append((h[:, 0, :] if h.dim() == 3 else h.mean((-2, -1))).cpu().numpy())
    del enc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.concatenate(out).astype(np.float32)


# --------------------------------------------------------------------------
# FedAvg over a linear head on frozen features
# --------------------------------------------------------------------------
def dirichlet_partition(labels, num_clients: int, alpha: float, seed: int):
    rng = np.random.RandomState(seed)
    lab = np.asarray(labels)
    parts: List[List[int]] = [[] for _ in range(num_clients)]
    for c in range(tea.NUM_CLASSES):
        idx = np.where(lab == c)[0]
        rng.shuffle(idx)
        if not len(idx):
            continue
        p = rng.dirichlet([alpha] * num_clients)
        cuts = (np.cumsum(p)[:-1] * len(idx)).astype(int)
        for cid, part in enumerate(np.split(idx, cuts)):
            parts[cid].extend(part.tolist())
    return [sorted(p) for p in parts]


def macro_f1(y_true, y_pred) -> float:
    scores = []
    for c in range(tea.NUM_CLASSES):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        d = 2 * tp + fp + fn
        scores.append(2.0 * tp / d if d else 0.0)
    return float(np.mean(scores))


def fedavg_linear_probe(Xtr, ytr, Xva, yva, client_idx, rounds, local_epochs,
                        lr, device, seed, weight_decay: float = 1e-4):
    torch.manual_seed(seed)
    d = Xtr.shape[1]
    g = nn.Linear(d, tea.NUM_CLASSES).to(device)
    Xtr_t = torch.tensor(Xtr, device=device)
    ytr_t = torch.tensor(ytr, device=device, dtype=torch.long)
    Xva_t = torch.tensor(Xva, device=device)
    sizes = [len(c) for c in client_idx]
    total = float(sum(sizes)) or 1.0
    history = []
    for _ in range(rounds):
        acc = {k: torch.zeros_like(v) for k, v in g.state_dict().items()}
        for cid, idx in enumerate(client_idx):
            if not idx:
                continue
            local = nn.Linear(d, tea.NUM_CLASSES).to(device)
            local.load_state_dict(g.state_dict())
            opt = torch.optim.AdamW(local.parameters(), lr=lr,
                                    weight_decay=weight_decay)
            xi, yi = Xtr_t[idx], ytr_t[idx]
            for _ in range(local_epochs):
                opt.zero_grad()
                nn.functional.cross_entropy(local(xi), yi).backward()
                opt.step()
            for k, v in local.state_dict().items():
                acc[k] += v * (sizes[cid] / total)
        g.load_state_dict(acc)
        with torch.no_grad():
            history.append(macro_f1(yva, g(Xva_t).argmax(-1).cpu().numpy()))
    return {
        "final_macro_f1": history[-1] if history else 0.0,
        "best_macro_f1": max(history) if history else 0.0,
        "round_macro_f1": history,
    }


def local_only_linear(Xtr, ytr, Xva, yva, client_idx, local_epochs, lr,
                      device, seed):
    scores = []
    for idx in client_idx:
        if not idx:
            continue
        r = fedavg_linear_probe(Xtr, ytr, Xva, yva, [idx], 1, local_epochs,
                                lr, device, seed)
        scores.append(r["final_macro_f1"])
    return {
        "mean_macro_f1": float(np.mean(scores)) if scores else 0.0,
        "best_macro_f1": float(np.max(scores)) if scores else 0.0,
        "per_client": scores,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tea_results/federated_all_systems")
    ap.add_argument("--client-counts", default="2,3,5,10,20,50")
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--local-epochs", type=int, default=3)
    ap.add_argument("--alphas", default="0.1,1.0")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--lr", type=float, default=1e-2)
    a = ap.parse_args()

    out_dir = ROOT / a.out
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    parts, obb, audit = build_split()
    print(f"split: {audit}")

    cache = out_dir / "frozen_features.npz"
    if cache.exists():
        z = np.load(cache)
        F = {k: z[k] for k in z.files}
        print("reused cached frozen features")
    else:
        F = {}
        for name in ("train", "validation"):
            F[f"text_{name}"] = frozen_text_features(
                parts[name]["text"].tolist(), device)
            F[f"image_{name}"] = frozen_image_features(obb[name], device)
            print(f"  {name}: text {F['text_' + name].shape}  "
                  f"image {F['image_' + name].shape}")
        np.savez_compressed(cache, **F)

    # Standardize with train-fitted statistics before the SGD head sees them.
    # Without this the comparison is not between modalities: DistilBERT features
    # are 768-d with mean |x| = 0.20 and DeiT-tiny CLS features 192-d at 1.31, so
    # at one shared learning rate the text head barely trains (0.352 against
    # 0.646 centralized -- see diag_probe_scaling.py) and the text row reads as a
    # collapse caused by whatever variable is being swept. Applied identically to
    # every arm, so no cell is advantaged relative to its control. The stored
    # statistics are global, which a real server would not hold; that is a
    # simulation convenience, recorded in the output rather than hidden.
    for mod in ("text", "image"):
        tr = F[f"{mod}_train"]
        mu, sd = tr.mean(0, keepdims=True), tr.std(0, keepdims=True) + 1e-6
        for name in ("train", "validation"):
            F[f"{mod}_{name}"] = ((F[f"{mod}_{name}"] - mu) / sd).astype(np.float32)
    print("standardized frozen features (train-fitted mean/SD)")

    def label_index(value):
        # tea_train stores single labels as a one-element list
        if isinstance(value, (list, tuple)):
            return int(value[0])
        return int(value)

    ytr = np.array([label_index(l) for l in parts["train"]["labels"]])
    yva = np.array([label_index(l) for l in parts["validation"]["labels"]])

    SYSTEMS = {
        "text_distilbert": (F["text_train"], F["text_validation"]),
        "image_vit_tiny": (F["image_train"], F["image_validation"]),
        "vit_distilbert": (
            np.hstack([F["image_train"], F["text_train"]]),
            np.hstack([F["image_validation"], F["text_validation"]]),
        ),
    }
    results: Dict = {
        "_meta": {
            "protocol": "frozen encoders, SGD linear head, FedAvg; "
                        "shared partition per cell",
            "caveat": "heads are SGD-trained, not the centralized closed-form "
                      "Ridge heads, so federated and centralized rows are not "
                      "the same estimator",
            "encoders_frozen": True,
            "features": "standardized with train-fitted mean/SD; results before "
                        "2026-08-27 were not standardized and are not comparable",
            "split_audit": audit,
            "clients": "see client_counts",
            "rounds": a.rounds,
            "local_epochs": a.local_epochs,
            "systems": list(SYSTEMS),
        },
        "runs": {},
    }

    client_counts = [int(x) for x in a.client_counts.split(",")]
    results["_meta"]["client_counts"] = client_counts
    for alpha in [float(x) for x in a.alphas.split(",")]:
      for n_clients in client_counts:
        for seed in [int(x) for x in a.seeds.split(",")]:
            client_idx = dirichlet_partition(ytr, n_clients, alpha, seed)
            nonempty = sum(1 for c in client_idx if c)
            print("")
            print(f"alpha={alpha} K={n_clients} seed={seed}  "
                  f"non-empty={nonempty}  sizes={[len(c) for c in client_idx][:8]}")
            for name, (Xtr, Xva) in SYSTEMS.items():
                fed = fedavg_linear_probe(Xtr, ytr, Xva, yva, client_idx,
                                          a.rounds, a.local_epochs, a.lr,
                                          device, seed)
                loc = local_only_linear(Xtr, ytr, Xva, yva, client_idx,
                                        a.rounds * a.local_epochs, a.lr,
                                        device, seed)
                gain = fed["final_macro_f1"] - loc["mean_macro_f1"]
                results["runs"][f"alpha={alpha}|K={n_clients}|seed={seed}|system={name}"] = {
                    "system": name, "alpha": alpha, "seed": seed,
                    "num_clients": n_clients,
                    "feature_dim": int(Xtr.shape[1]),
                    "fedavg_final_macro_f1": fed["final_macro_f1"],
                    "fedavg_best_macro_f1": fed["best_macro_f1"],
                    "round_macro_f1": fed["round_macro_f1"],
                    "local_only_mean_macro_f1": loc["mean_macro_f1"],
                    "local_only_best_macro_f1": loc["best_macro_f1"],
                    "federation_gain": gain,
                    "shared_partition": True,
                }
                print(f"  {name:<18} FedAvg={fed['final_macro_f1']:.4f}  "
                      f"local={loc['mean_macro_f1']:.4f}  gain={gain:+.4f}")

    path = out_dir / "federated_all_systems.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {path}")

    print("\n=== summary: FedAvg macro-F1 (mean over seeds) ===")
    agg = defaultdict(list)
    for r in results["runs"].values():
        agg[(r["system"], r["alpha"])].append(r["fedavg_final_macro_f1"])
    alphas = sorted({al for _, al in agg})
    print(f"  {'system':<20}" + "".join(f"a={al:<8}" for al in alphas))
    for s in SYSTEMS:
        print(f"  {s:<20}"
              + "".join(f"{np.mean(agg[(s, al)]):<10.4f}" for al in alphas))


if __name__ == "__main__":
    main()
