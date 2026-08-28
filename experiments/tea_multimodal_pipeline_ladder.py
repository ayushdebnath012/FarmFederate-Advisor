#!/usr/bin/env python3
"""A text+image pipeline ladder on the common support, matching the visual table.

Same column structure as the visual table: candidate count, validation accuracy,
the pre-declared validation score, test accuracy and test macro F1. The score is
the paper's own checkpoint-selection criterion,

    0.65 * F1_paired + 0.35 * F1_image

evaluated on the 74 validation crops, so a rung here is ranked by exactly the
rule the paper declares. Test is scored once per rung after the choice is frozen.

As with the text ladder, each rung is also run at three note sparsities, because
at complete notes the paired branch saturates and rung differences collapse to a
single crop.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import tea_train as tea  # noqa: E402
sys.path.insert(0, str(ROOT / "experiments"))
from tea_aligned_support_tables import (  # noqa: E402
    build_split, image_features, metrics, RIDGE_ALPHAS,
)
from tea_text_pipeline_ladder import encode, fit_ridge, standardize  # noqa: E402

SPARSITIES = [0.0, 0.5, 0.75]
TEXT_ID = "distilbert-base-uncased"
IMAGE_ID = "google/vit-base-patch16-224"
PCA_DIMS = [16, 32, 64, 128]
MIX_WEIGHTS = [0.0, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0]


def val_score(f1_paired: float, f1_image: float) -> float:
    """The paper's pre-declared checkpoint-selection score."""
    return 0.65 * f1_paired + 0.35 * f1_image


def ridge_probs(feats, y, alphas=RIDGE_ALPHAS):
    """Validation-selected ridge; returns (val_pred, test_pred, n_candidates)."""
    Ztr, Zva, Zte = standardize(feats["train"], feats["validation"], feats["test"])
    best = None
    for a in alphas:
        clf = fit_ridge(Ztr, y["train"], a)
        pv = clf.predict(Zva)
        v = metrics(y["validation"], pv.argmax(1))
        k = (v["macro_f1"], v["accuracy"], -a)
        if best is None or k > best[0]:
            best = (k, clf.predict(Zva), clf.predict(Zte))
    return best[1], best[2], len(alphas)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tea_results/multimodal_pipeline_ladder")
    a = ap.parse_args()
    out_dir = ROOT / a.out
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parts, obb, audit = build_split()
    lab = lambda v: int(v[0]) if isinstance(v, (list, tuple)) else int(v)
    y = {k: np.array([lab(v) for v in parts[k]["labels"]]) for k in parts}
    print(f"device: {device}\nsupport: {audit['crops']}\n")

    img = {k: image_features(obb[k], IMAGE_ID, device) for k in parts}
    iv, it, _ = ridge_probs(img, y)
    f1_image_val = metrics(y["validation"], iv.argmax(1))["macro_f1"]
    print(f"image-only branch: val macro F1 = {f1_image_val:.4f}\n")

    results = {"_meta": {**audit, "sparsities": SPARSITIES,
                         "validation_score": "0.65*F1_paired + 0.35*F1_image",
                         "text_encoder": TEXT_ID, "image_encoder": IMAGE_ID},
               "ladder": {}}

    for sp in SPARSITIES:
        per = {k: encode(parts[k]["text"].tolist(), TEXT_ID, device, sp) for k in parts}
        txt = {k: per[k][1] for k in parts}          # masked-mean
        txt4 = {k: per[k][2] for k in parts}         # last-4-layer concat
        tv, tt, _ = ridge_probs(txt, y)
        rungs = {}

        def record(name, pv, pt, cand):
            v = metrics(y["validation"], pv.argmax(1))
            t = metrics(y["test"], pt.argmax(1))
            rungs[name] = (cand, v, val_score(v["macro_f1"], f1_image_val), t)

        # 1. direct concatenation, one fixed penalty
        cat = {k: np.hstack([img[k], txt[k]]) for k in parts}
        pv, pt, _ = ridge_probs(cat, y, alphas=[1.0])
        record("Direct concat path", pv, pt, 1)

        # 2. concatenation with a swept penalty
        pv, pt, n = ridge_probs(cat, y)
        record("Concat + tuned head", pv, pt, n)

        # 3. deeper text representation concatenated
        cat4 = {k: np.hstack([img[k], txt4[k]]) for k in parts}
        pv, pt, n = ridge_probs(cat4, y)
        record("Deep-feature concat", pv, pt, 4 * n)

        # 4. per-modality PCA before concatenation
        from sklearn.decomposition import PCA
        best = None
        for d in PCA_DIMS:
            pi, ptx = PCA(n_components=d), PCA(n_components=d)
            f = {"train": np.hstack([pi.fit_transform(img["train"]),
                                     ptx.fit_transform(txt["train"])])}
            for s in ("validation", "test"):
                f[s] = np.hstack([pi.transform(img[s]), ptx.transform(txt[s])])
            v_, t_, _ = ridge_probs(f, y)
            m = metrics(y["validation"], v_.argmax(1))
            if best is None or (m["macro_f1"], m["accuracy"]) > best[0]:
                best = ((m["macro_f1"], m["accuracy"]), v_, t_)
        record("PCA-per-modality concat", best[1], best[2],
               len(PCA_DIMS) * len(RIDGE_ALPHAS))

        # 5. validation-calibrated late fusion over the two branches
        best = None
        for w in MIX_WEIGHTS:
            pv_ = w * tv + (1 - w) * iv
            m = metrics(y["validation"], pv_.argmax(1))
            if best is None or (m["macro_f1"], m["accuracy"]) > best[0]:
                best = ((m["macro_f1"], m["accuracy"]), pv_, w * tt + (1 - w) * it)
        record("Calibrated late fusion", best[1], best[2], len(MIX_WEIGHTS))

        # 6. per-class gate over {image, text, concat} -- a diagnostic, as in
        #    the visual table, not a deployable single model
        cv, ct, _ = ridge_probs(cat, y)
        experts = [(iv, it), (tv, tt), (cv, ct)]
        base_v, base_t = cv.argmax(1), ct.argmax(1)
        best = None
        for combo in itertools.product(range(3), repeat=tea.NUM_CLASSES):
            gv = base_v.copy()
            for cls_id, ex in enumerate(combo):
                sel = base_v == cls_id
                gv[sel] = experts[ex][0].argmax(1)[sel]
            m = metrics(y["validation"], gv)
            if best is None or (m["macro_f1"], m["accuracy"]) > best[0]:
                best = ((m["macro_f1"], m["accuracy"]), combo, gv)
        _, combo, gv = best
        gt = base_t.copy()
        for cls_id, ex in enumerate(combo):
            sel = base_t == cls_id
            gt[sel] = experts[ex][1].argmax(1)[sel]
        v = metrics(y["validation"], gv)
        t = metrics(y["test"], gt)
        rungs["Three-expert gate (not single model)"] = (
            3 ** tea.NUM_CLASSES, v, val_score(v["macro_f1"], f1_image_val), t)

        results["ladder"][f"sparsity={sp}"] = {
            k: {"candidates": c, "validation_accuracy": v["accuracy"],
                "validation_macro_f1": v["macro_f1"], "validation_score": s,
                "test_accuracy": t["accuracy"], "test_macro_f1": t["macro_f1"]}
            for k, (c, v, s, t) in rungs.items()}

        print(f"=== note sparsity {sp:.0%} ===")
        print(f"  {'multimodal system':<40}{'cand':>6}{'val acc':>9}"
              f"{'val score':>11}{'test acc':>10}{'test F1':>9}")
        for k, (c, v, s, t) in rungs.items():
            print(f"  {k:<40}{c:>6}{v['accuracy']:>9.3f}{s:>11.3f}"
                  f"{t['accuracy']:>10.3f}{t['macro_f1']:>9.3f}")
        print()

    path = out_dir / "multimodal_pipeline_ladder.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
