#!/usr/bin/env python3
"""Classical, LLM/ViT, and VLM systems side by side -- centralized and federated.

Everything in here is measured on ONE support:

  * 222 train / 74 validation / 75 test crops, source-grouped, seed 42
  * exact (photograph, box) crop-note linkage
  * leakage vocabulary fitted on TRAIN ONLY and applied to all three splits
  * every configuration chosen on the 74 validation crops; test is scored once

Three families are clubbed together so a classical baseline, a frozen
transformer encoder, and a fused vision-language system can be read off the
same rows:

  text-only        TF-IDF + linear SVM   |  DistilBERT, BERT-tiny/mini/small/medium
  image-only       colour hist + HOG SVM |  ViT-Base, DeiT-tiny, Swin-tiny,
                                            ConvNeXT-tiny, EfficientNet-B0
  multimodal       both classical blocks |  ViT-Base fused with each text encoder

Each system is reported twice:

  Centralized   one ridge (or linear-SVM) head fitted on all 222 train crops.
  Federated     the SAME frozen features, but the head is trained by FedAvg
                over K Dirichlet-partitioned clients. A closed-form solve has
                no gradients to average, so the federated head is SGD-trained.
                It is therefore not the same estimator as the centralized head,
                and that is recorded rather than left implicit.

Notes are also degraded, because at complete notes the text branch saturates
and no fusion result can be read: a ceiling hides the comparison. Deletion is
deterministic in (row, token), so every system sees the identical corruption of
the identical notes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
import tea_train as tea  # noqa: E402
from tea_aligned_support_tables import (  # noqa: E402
    build_split, load_text_encoder, metrics, RIDGE_ALPHAS,
    TEXT_ENCODERS, IMAGE_ENCODERS,
)

SPARSITIES = [0.0, 0.5]
SVM_CS = [0.01, 0.1, 1.0, 10.0]
FUSION_IMAGE = "ViT-Base"
SPLITS = ("train", "validation", "test")


# ---------------------------------------------------------------------------
# deterministic note degradation
# ---------------------------------------------------------------------------
def drop_words(texts: Sequence[str], sparsity: float) -> list:
    """Delete words with the same deterministic rule the ladders use on tokens."""
    if sparsity <= 0:
        return list(texts)
    cut = int(sparsity * 10000)
    out = []
    for r, t in enumerate(texts):
        words = str(t).split()
        kept = [w for c, w in enumerate(words)
                if (r * 3571 + c * 7919) % 10000 >= cut]
        out.append(" ".join(kept) if kept else "")
    return out


# ---------------------------------------------------------------------------
# classical features -- no pretrained weights anywhere in this block
# ---------------------------------------------------------------------------
def tfidf_features(parts_text: Dict[str, list], max_features: int = 4000):
    """Word 1-2gram TF-IDF, vocabulary fitted on the training split only."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2),
                          sublinear_tf=True, min_df=1)
    F = {"train": vec.fit_transform(parts_text["train"]).toarray()}
    for s in ("validation", "test"):
        F[s] = vec.transform(parts_text[s]).toarray()
    return {k: v.astype(np.float64) for k, v in F.items()}


def _hog(gray: np.ndarray, cells: int = 8, bins: int = 9) -> np.ndarray:
    """Plain histogram-of-oriented-gradients, no external image library."""
    gx = np.zeros_like(gray); gy = np.zeros_like(gray)
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    mag = np.sqrt(gx ** 2 + gy ** 2)
    ang = (np.rad2deg(np.arctan2(gy, gx)) % 180.0)
    h, w = gray.shape
    ch, cw = h // cells, w // cells
    feat = []
    for i in range(cells):
        for j in range(cells):
            m = mag[i * ch:(i + 1) * ch, j * cw:(j + 1) * cw].ravel()
            a = ang[i * ch:(i + 1) * ch, j * cw:(j + 1) * cw].ravel()
            hist, _ = np.histogram(a, bins=bins, range=(0, 180), weights=m)
            n = np.linalg.norm(hist) + 1e-8
            feat.append(hist / n)
    return np.concatenate(feat)


def classical_image_features(obb_ds, bins: int = 16) -> np.ndarray:
    """Per-channel colour histogram + colour moments + HOG on the same crops."""
    from torch.utils.data import DataLoader
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    out = []
    for batch in DataLoader(obb_ds, batch_size=32, shuffle=False):
        px = batch["pixel_values"].numpy()
        for k in range(px.shape[0]):
            rgb = np.clip(px[k] * std + mean, 0.0, 1.0)      # undo normalization
            hists = [np.histogram(rgb[c], bins=bins, range=(0, 1),
                                  density=True)[0] for c in range(3)]
            moments = np.concatenate([rgb.mean((1, 2)), rgb.std((1, 2))])
            gray = rgb.mean(0)
            out.append(np.concatenate(hists + [moments, _hog(gray)]))
    return np.asarray(out, dtype=np.float64)


# ---------------------------------------------------------------------------
# frozen deep features
# ---------------------------------------------------------------------------
def tokenizer_for(model_id):
    from transformers import AutoTokenizer
    for kwargs in ({}, {"use_fast": False}):
        try:
            return AutoTokenizer.from_pretrained(model_id, **kwargs)
        except Exception:
            continue
    return AutoTokenizer.from_pretrained("bert-base-uncased")


@torch.inference_mode()
def text_features(texts: Sequence[str], model_id, device, batch_size=32):
    tok = tokenizer_for(model_id)
    enc = load_text_encoder(model_id).to(device).eval()
    out = []
    for s in range(0, len(texts), batch_size):
        b = tok(list(texts[s:s + batch_size]), max_length=128,
                padding="max_length", truncation=True,
                return_tensors="pt").to(device)
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
        if h is not None and h.dim() == 3:
            feat = h[:, 0, :]
        elif h is not None and h.dim() == 4:
            feat = h.mean(dim=(-2, -1))
        else:
            feat = o.pooler_output
        out.append(feat.cpu().numpy())
    del enc
    torch.cuda.empty_cache()
    return np.concatenate(out).astype(np.float64)


# ---------------------------------------------------------------------------
# heads
# ---------------------------------------------------------------------------
def standardize(F: Dict[str, np.ndarray]):
    mu = F["train"].mean(0, keepdims=True)
    sd = F["train"].std(0, keepdims=True) + 1e-8
    return {k: (v - mu) / sd for k, v in F.items()}


def centralized_ridge(Z, y):
    from sklearn.linear_model import Ridge
    onehot = np.eye(tea.NUM_CLASSES)[y["train"]]
    best = None
    for a in RIDGE_ALPHAS:
        clf = Ridge(alpha=a).fit(Z["train"], onehot)
        v = metrics(y["validation"], clf.predict(Z["validation"]).argmax(1))
        key = (v["macro_f1"], v["accuracy"], -a)
        if best is None or key > best[0]:
            best = (key, clf, v)
    _, clf, v = best
    t = metrics(y["test"], clf.predict(Z["test"]).argmax(1))
    return v, t


def centralized_svm(Z, y):
    """The classical rows get an actual classical classifier, not a ridge."""
    from sklearn.svm import LinearSVC
    best = None
    for C in SVM_CS:
        clf = LinearSVC(C=C, max_iter=5000).fit(Z["train"], y["train"])
        v = metrics(y["validation"], clf.predict(Z["validation"]))
        key = (v["macro_f1"], v["accuracy"], -C)
        if best is None or key > best[0]:
            best = (key, clf, v)
    _, clf, v = best
    t = metrics(y["test"], clf.predict(Z["test"]))
    return v, t


def dirichlet_partition(labels, num_clients: int, alpha: float, seed: int):
    rng = np.random.RandomState(seed)
    lab = np.asarray(labels)
    clients = [[] for _ in range(num_clients)]
    for c in range(tea.NUM_CLASSES):
        idx = np.where(lab == c)[0]
        rng.shuffle(idx)
        p = rng.dirichlet(np.repeat(alpha, num_clients))
        cuts = (np.cumsum(p) * len(idx)).astype(int)[:-1]
        for k, part in enumerate(np.split(idx, cuts)):
            clients[k].extend(part.tolist())
    return [sorted(c) for c in clients]


def fedavg_head(Z, y, client_idx, rounds, local_epochs, lr, device, seed):
    """FedAvg over a linear softmax head; round chosen on validation only."""
    torch.manual_seed(seed)
    d = Z["train"].shape[1]
    T = {k: torch.tensor(v, dtype=torch.float32, device=device) for k, v in Z.items()}
    Y = {k: torch.tensor(y[k], dtype=torch.long, device=device) for k in y}
    glob = nn.Linear(d, tea.NUM_CLASSES).to(device)
    loss_fn = nn.CrossEntropyLoss()
    sizes = np.array([len(c) for c in client_idx], dtype=np.float64)
    active = [i for i, s in enumerate(sizes) if s > 0]
    if not active:
        raise ValueError("every client is empty")
    w = sizes[active] / sizes[active].sum()

    best = (-1.0, None, -1)
    for rnd in range(rounds):
        states = []
        for i in active:
            local = nn.Linear(d, tea.NUM_CLASSES).to(device)
            local.load_state_dict(glob.state_dict())
            opt = torch.optim.SGD(local.parameters(), lr=lr, momentum=0.9)
            idx = torch.tensor(client_idx[i], dtype=torch.long, device=device)
            for _ in range(local_epochs):
                opt.zero_grad()
                loss_fn(local(T["train"][idx]), Y["train"][idx]).backward()
                opt.step()
            states.append({k: v.detach().clone()
                           for k, v in local.state_dict().items()})
        merged = {k: sum(wi * st[k] for wi, st in zip(w, states))
                  for k in states[0]}
        glob.load_state_dict(merged)
        with torch.no_grad():
            pv = glob(T["validation"]).argmax(1).cpu().numpy()
        f1 = metrics(y["validation"], pv)["macro_f1"]
        if f1 > best[0]:
            best = (f1, {k: v.clone() for k, v in merged.items()}, rnd)

    glob.load_state_dict(best[1])
    with torch.no_grad():
        pv = glob(T["validation"]).argmax(1).cpu().numpy()
        pt = glob(T["test"]).argmax(1).cpu().numpy()
    return metrics(y["validation"], pv), metrics(y["test"], pt), best[2]


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tea_results/clubbed_tables")
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--local-epochs", type=int, default=3)
    ap.add_argument("--clients", type=int, default=5)
    ap.add_argument("--dirichlet", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--sparsities", default=",".join(str(x) for x in SPARSITIES))
    a = ap.parse_args()

    out_dir = ROOT / a.out
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = [int(s) for s in a.seeds.split(",")]
    sparsities = [float(x) for x in a.sparsities.split(",")]

    parts, obb, audit = build_split()
    lab = lambda v: int(v[0]) if isinstance(v, (list, tuple)) else int(v)
    y = {k: np.array([lab(v) for v in parts[k]["labels"]]) for k in SPLITS}
    assert (len(y["train"]), len(y["validation"]), len(y["test"])) == (222, 74, 75), \
        f"unexpected support {tuple(len(y[k]) for k in SPLITS)}"
    print(f"device: {device}\nsupport: {audit['crops']}  "
          f"sources: {audit['source_groups']}\n")

    # ---- image side: independent of note sparsity, so computed once --------
    t0 = time.time()
    img_feats = {"Colour+HOG SVM": {k: classical_image_features(obb[k])
                                    for k in SPLITS}}
    print(f"  classical image features {img_feats['Colour+HOG SVM']['train'].shape} "
          f"({time.time() - t0:.0f}s)")
    for name, mid in IMAGE_ENCODERS.items():
        try:
            img_feats[name] = {k: image_features(obb[k], mid, device) for k in SPLITS}
            print(f"  image {name:<16} dim={img_feats[name]['train'].shape[1]}")
        except Exception as exc:
            print(f"  image {name}: FAILED {exc}")

    results = {"_meta": {**audit,
                         "sparsities": sparsities,
                         "federated": {"algorithm": "FedAvg", "clients": a.clients,
                                       "dirichlet_alpha": a.dirichlet,
                                       "rounds": a.rounds,
                                       "local_epochs": a.local_epochs,
                                       "lr": a.lr, "seeds": seeds,
                                       "round_selected_on": "validation only"},
                         "caveat": "centralized heads are closed-form (ridge, or "
                                   "linear SVM for the classical rows); federated "
                                   "heads are SGD-trained because a closed-form "
                                   "solve has no gradients to average, so the two "
                                   "columns are not the same estimator",
                         "classical_note": "TF-IDF vocabulary is fitted on the "
                                           "training split centrally; a federated "
                                           "deployment would have to agree it first",
                         "fusion_image_encoder": FUSION_IMAGE},
               "tables": {}}

    def evaluate(name, feats, family, sparsity, classical: bool):
        Z = standardize(feats)
        v, t = (centralized_svm if classical else centralized_ridge)(Z, y)
        fed_v, fed_t, rounds_used = [], [], []
        for s in seeds:
            ci = dirichlet_partition(y["train"], a.clients, a.dirichlet, s)
            fv, ft, r = fedavg_head(Z, y, ci, a.rounds, a.local_epochs,
                                    a.lr, device, s)
            fed_v.append(fv); fed_t.append(ft); rounds_used.append(r)
        agg = lambda rs, k: (float(np.mean([r[k] for r in rs])),
                             float(np.std([r[k] for r in rs])))
        fa, fa_sd = agg(fed_t, "accuracy")
        ff, ff_sd = agg(fed_t, "macro_f1")
        row = {"family": family, "classical": classical,
               "dim": int(feats["train"].shape[1]),
               "central_val_accuracy": v["accuracy"],
               "central_val_macro_f1": v["macro_f1"],
               "central_test_accuracy": t["accuracy"],
               "central_test_macro_f1": t["macro_f1"],
               "fed_val_macro_f1": agg(fed_v, "macro_f1")[0],
               "fed_test_accuracy": fa, "fed_test_accuracy_sd": fa_sd,
               "fed_test_macro_f1": ff, "fed_test_macro_f1_sd": ff_sd,
               "fed_rounds_selected": rounds_used,
               "delta_test_accuracy": fa - t["accuracy"]}
        results["tables"][f"sparsity={sparsity}"][family][name] = row
        print(f"  {family:<11}{name:<24} central {t['accuracy']:.3f}/"
              f"{t['macro_f1']:.3f}   fed {fa:.3f}+-{fa_sd:.3f}/{ff:.3f}")

    for sp in sparsities:
        print(f"\n=== note sparsity {sp:.0%} ===")
        results["tables"][f"sparsity={sp}"] = {"text_only": {}, "image_only": {},
                                               "multimodal": {}}
        texts = {k: drop_words(parts[k]["text"].tolist(), sp) for k in SPLITS}

        txt_feats = {"TF-IDF + linear SVM": tfidf_features(texts)}
        for name, mid in TEXT_ENCODERS.items():
            try:
                txt_feats[name] = {k: text_features(texts[k], mid, device)
                                   for k in SPLITS}
            except Exception as exc:
                print(f"  text {name}: FAILED {exc}")

        for name, f in txt_feats.items():
            evaluate(name, f, "text_only", sp, name.startswith("TF-IDF"))
        for name, f in img_feats.items():
            evaluate(name, f, "image_only", sp, name.startswith("Colour"))

        # multimodal: classical block fused with classical block, and the
        # validation-selected vision encoder fused with each text encoder
        pairs = {"TF-IDF + Colour/HOG SVM":
                 ({k: np.hstack([img_feats["Colour+HOG SVM"][k],
                                 txt_feats["TF-IDF + linear SVM"][k]])
                   for k in SPLITS}, True)}
        if FUSION_IMAGE in img_feats:
            for name in TEXT_ENCODERS:
                if name in txt_feats:
                    pairs[f"{FUSION_IMAGE} + {name}"] = (
                        {k: np.hstack([img_feats[FUSION_IMAGE][k], txt_feats[name][k]])
                         for k in SPLITS}, False)
        for name, (f, classical) in pairs.items():
            evaluate(name, f, "multimodal", sp, classical)

    path = out_dir / "clubbed_tables.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {path}")
    print("Done.")


if __name__ == "__main__":
    main()
