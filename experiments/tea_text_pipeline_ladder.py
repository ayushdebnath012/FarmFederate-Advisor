#!/usr/bin/env python3
"""A text-pipeline ladder on the common support, mirroring the visual Table IV.

The visual table varies the pipeline over one backbone family: direct path,
TTA, deep-feature head, calibrated base, then per-class expert gates, each
choosing among a stated number of validation-fitted candidates. This does the
same for text.

The complication is headroom. At full notes every text encoder already scores
0.987-1.000 on the 75 test crops, so a ladder there would be separating rungs by
a single crop, which is noise. Each rung is therefore run at three note
sparsities: complete notes, 50% of attended tokens deleted, and 75% deleted.
A ladder that is flat at 0% and separates as tokens are removed shows the
ceiling belongs to the corpus rather than to the pipeline.

Selection uses the 74 validation crops only; test is scored once per rung and
sparsity after the choice is frozen.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import tea_train as tea  # noqa: E402
sys.path.insert(0, str(ROOT / "experiments"))
from tea_aligned_support_tables import (  # noqa: E402
    build_split, load_text_encoder, metrics, RIDGE_ALPHAS, TEXT_ENCODERS,
)

SPARSITIES = [0.0, 0.5, 0.75]
PCA_DIMS = [16, 32, 64, 128, 256]


def tokenizer_for(model_id):
    from transformers import AutoTokenizer
    for kwargs in ({}, {"use_fast": False}):
        try:
            return AutoTokenizer.from_pretrained(model_id, **kwargs)
        except Exception:
            continue
    return AutoTokenizer.from_pretrained("bert-base-uncased")


@torch.inference_mode()
def encode(texts: Sequence[str], model_id, device, sparsity: float,
           batch_size: int = 32):
    """Return (cls, mean, last4) representations under a fixed token dropout.

    Deletion is deterministic given the row and token index, so every rung and
    every encoder sees the identical corruption of the identical notes.
    """
    tok = tokenizer_for(model_id)
    enc = load_text_encoder(model_id).to(device).eval()
    cls_out, mean_out, l4_out = [], [], []
    for s in range(0, len(texts), batch_size):
        b = tok(list(texts[s:s + batch_size]), max_length=128,
                padding="max_length", truncation=True,
                return_tensors="pt").to(device)
        ids, mask = b["input_ids"], b["attention_mask"]
        if sparsity > 0:
            rows = torch.arange(ids.size(0), device=ids.device).unsqueeze(1) + s
            cols = torch.arange(ids.size(1), device=ids.device).unsqueeze(0)
            score = (rows * 3571 + cols * 7919) % 10000
            ordinary = mask.bool() & ids.ne(0) & ids.ne(101) & ids.ne(102)
            drop = ordinary & score.lt(int(sparsity * 10000))
            ids = ids.masked_fill(drop, 0)
            mask = mask.masked_fill(drop, 0)
        out = enc(input_ids=ids, attention_mask=mask, output_hidden_states=True)
        h = out.last_hidden_state
        w = mask.unsqueeze(-1).to(h.dtype)
        cls_out.append(h[:, 0, :].cpu().numpy())
        mean_out.append(((h * w).sum(1) / w.sum(1).clamp_min(1.0)).cpu().numpy())
        hs = out.hidden_states[-4:]
        l4 = torch.cat([((x * w).sum(1) / w.sum(1).clamp_min(1.0)) for x in hs], -1)
        l4_out.append(l4.cpu().numpy())
    del enc
    torch.cuda.empty_cache()
    return tuple(np.concatenate(z).astype(np.float64)
                 for z in (cls_out, mean_out, l4_out))


def fit_ridge(Xtr, ytr, alpha):
    from sklearn.linear_model import Ridge
    return Ridge(alpha=alpha).fit(Xtr, np.eye(tea.NUM_CLASSES)[ytr])


def standardize(Xtr, *others):
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-8
    return ((Xtr - mu) / sd,) + tuple((o - mu) / sd for o in others)


def sweep(feats, y, alphas=RIDGE_ALPHAS):
    """Choose the ridge penalty on validation; return val/test metrics + count."""
    Ztr, Zva, Zte = standardize(feats["train"], feats["validation"], feats["test"])
    best = None
    for a in alphas:
        clf = fit_ridge(Ztr, y["train"], a)
        v = metrics(y["validation"], clf.predict(Zva).argmax(1))
        k = (v["macro_f1"], v["accuracy"], -a)
        if best is None or k > best[0]:
            best = (k, clf, v, Zte)
    _, clf, v, Zte = best
    t = metrics(y["test"], clf.predict(Zte).argmax(1))
    return v, t, len(alphas), clf.predict(Zte)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tea_results/text_pipeline_ladder")
    a = ap.parse_args()
    out_dir = ROOT / a.out
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parts, _obb, audit = build_split()
    lab = lambda v: int(v[0]) if isinstance(v, (list, tuple)) else int(v)
    y = {k: np.array([lab(v) for v in parts[k]["labels"]]) for k in parts}
    print(f"device: {device}\nsupport: {audit['crops']}\n")

    results = {"_meta": {**audit, "sparsities": SPARSITIES,
                         "selection": "74 validation crops only",
                         "note": "rungs mirror the visual pipeline table"},
               "ladder": {}}

    for sp in SPARSITIES:
        print(f"=== note sparsity {sp:.0%} ===")
        reps = {}
        for name, mid in TEXT_ENCODERS.items():
            try:
                per_split = {k: encode(parts[k]["text"].tolist(), mid, device, sp)
                             for k in parts}
            except Exception as exc:
                print(f"  {name}: FAILED {exc}"); continue
            reps[name] = {
                "cls": {k: per_split[k][0] for k in parts},
                "mean": {k: per_split[k][1] for k in parts},
                "last4": {k: per_split[k][2] for k in parts},
            }
        if not reps:
            continue
        primary = "DistilBERT" if "DistilBERT" in reps else next(iter(reps))
        rungs = {}

        v, t, n, _ = sweep(reps[primary]["cls"], y, alphas=[1.0])
        rungs["Direct CLS path"] = (v, t, n)

        v, t, n, _ = sweep(reps[primary]["mean"], y)
        rungs["Masked-mean pooling"] = (v, t, n)

        v, t, n, _ = sweep(reps[primary]["last4"], y)
        rungs["Last-4-layer concat"] = (v, t, 4 * n)

        # PCA rung: dimensionality is itself a validation-fitted choice
        from sklearn.decomposition import PCA
        best = None
        for d in PCA_DIMS:
            p = PCA(n_components=min(d, reps[primary]["mean"]["train"].shape[0] - 1))
            f = {"train": p.fit_transform(reps[primary]["mean"]["train"])}
            f["validation"] = p.transform(reps[primary]["mean"]["validation"])
            f["test"] = p.transform(reps[primary]["mean"]["test"])
            v, t, _, _ = sweep(f, y)
            if best is None or (v["macro_f1"], v["accuracy"]) > best[0]:
                best = ((v["macro_f1"], v["accuracy"]), v, t)
        rungs["PCA-reduced head"] = (best[1], best[2], len(PCA_DIMS) * len(RIDGE_ALPHAS))

        # Per-class expert gate over the encoders: one expert per predicted
        # class, exactly the structure the visual table calls a gate.
        names = sorted(reps)
        preds = {}
        for nm in names:
            v, t, _, raw = sweep(reps[nm]["mean"], y)
            Ztr, Zva, Zte = standardize(reps[nm]["mean"]["train"],
                                        reps[nm]["mean"]["validation"],
                                        reps[nm]["mean"]["test"])
            clf = fit_ridge(Ztr, y["train"], 1.0)
            preds[nm] = {"validation": clf.predict(Zva).argmax(1),
                         "test": clf.predict(Zte).argmax(1)}
        base = names[0]
        best_gate = None
        for combo in itertools.product(range(len(names)), repeat=tea.NUM_CLASSES):
            pv = preds[base]["validation"].copy()
            for cls_id, ex in enumerate(combo):
                sel = preds[base]["validation"] == cls_id
                pv[sel] = preds[names[ex]]["validation"][sel]
            v = metrics(y["validation"], pv)
            k = (v["macro_f1"], v["accuracy"])
            if best_gate is None or k > best_gate[0]:
                best_gate = (k, combo, v)
        _, combo, v = best_gate
        pt = preds[base]["test"].copy()
        for cls_id, ex in enumerate(combo):
            sel = preds[base]["test"] == cls_id
            pt[sel] = preds[names[ex]]["test"][sel]
        rungs[f"{len(names)}-expert gate (not single model)"] = (
            v, metrics(y["test"], pt), len(names) ** tea.NUM_CLASSES)

        results["ladder"][f"sparsity={sp}"] = {
            k: {"candidates": n,
                "validation_accuracy": v["accuracy"],
                "validation_macro_f1": v["macro_f1"],
                "test_accuracy": t["accuracy"],
                "test_macro_f1": t["macro_f1"]}
            for k, (v, t, n) in rungs.items()
        }
        print(f"  {'rung':<38}{'cand':>6}{'val acc':>10}{'test acc':>10}{'test F1':>10}")
        for k, (v, t, n) in rungs.items():
            print(f"  {k:<38}{n:>6}{v['accuracy']:>10.3f}"
                  f"{t['accuracy']:>10.3f}{t['macro_f1']:>10.3f}")
        print()

    path = out_dir / "text_pipeline_ladder.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
