#!/usr/bin/env python3
"""Does concatenating several frozen vision encoders beat the best single one?

The aligned-support tables sample the vision axis one encoder at a time, and the
architecture ablation says vision capacity is the only component whose removal
clears the seed noise floor. Those two facts together make multi-encoder
concatenation the obvious untried point: five encoders are already scored on
this support, the top three are architecturally diverse (patch transformer,
distilled transformer, convolutional), and combining frozen features costs
nothing but a wider ridge.

Protocol is the published one, unchanged: same 222/74/75 source-grouped split,
same train-fitted leakage mask, per-dimension standardization, closed-form ridge
with the penalty chosen on validation, test scored once. Text is fixed at
BERT-small, the published best, so the only thing varying is the vision side.

The search is reported, not hidden. All 31 non-empty subsets are evaluated, and
the summary prints the validation-to-test gap of the winner next to the
pre-declared single-encoder baseline. On 74 validation crops a 31-way search can
buy validation accuracy that does not transfer -- that is exactly what the gate
rungs of the pipeline ladders do -- so the honest output is the transfer gap,
not the best test cell. One test crop is 1.33 accuracy points; differences under
about four crops are not distinguishable on this support.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
import tea_train as tea  # noqa: E402
from tea_aligned_support_tables import (  # noqa: E402
    IMAGE_ENCODERS, build_split, image_features, metrics,
    ridge_select_and_test, text_features,
)

TEXT_ENCODER = ("BERT-small", "prajjwal1/bert-small")
BASELINE_IMAGE = "ViT-Base"          # pre-declared single-encoder reference


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tea_results/multiencoder_fusion")
    a = ap.parse_args()
    out_dir = ROOT / a.out
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parts, obb, audit = build_split()
    print(f"device {device}; split {audit}")

    lab = lambda v: int(v[0]) if isinstance(v, (list, tuple)) else int(v)
    y = {s: np.array([lab(l) for l in parts[s]["labels"]])
         for s in ("train", "validation", "test")}

    img = {}
    for name, mid in IMAGE_ENCODERS.items():
        img[name] = {s: image_features(obb[s], mid, device)
                     for s in ("train", "validation", "test")}
        print(f"  image {name:<15} dim {img[name]['train'].shape[1]}")
    tname, tmid = TEXT_ENCODER
    txt = {s: text_features(parts[s]["text"].tolist(), tmid, device)
           for s in ("train", "validation", "test")}
    print(f"  text  {tname:<15} dim {txt['train'].shape[1]}")

    names = list(IMAGE_ENCODERS)
    rows = []
    for k in range(1, len(names) + 1):
        for combo in itertools.combinations(names, k):
            for with_text in (False, True):
                X = {}
                for s in ("train", "validation", "test"):
                    blocks = [img[n][s] for n in combo]
                    if with_text:
                        blocks.append(txt[s])
                    X[s] = np.hstack(blocks)
                r = ridge_select_and_test(
                    X["train"], y["train"], X["validation"], y["validation"],
                    X["test"], y["test"])
                r.update({"encoders": list(combo), "n_encoders": k,
                          "with_text": with_text, "dim": int(X["train"].shape[1])})
                rows.append(r)
    print(f"\nevaluated {len(rows)} configurations")

    def best_by_val(pool):
        return max(pool, key=lambda r: (r["validation_macro_f1"],
                                        r["validation_accuracy"]))

    baseline = next(r for r in rows if r["encoders"] == [BASELINE_IMAGE]
                    and r["with_text"])
    winner = best_by_val(rows)
    img_only_winner = best_by_val([r for r in rows if not r["with_text"]])
    img_baseline = next(r for r in rows if r["encoders"] == [BASELINE_IMAGE]
                        and not r["with_text"])

    n_test = len(y["test"])
    crops = lambda d: d * n_test

    print("\n=== pre-declared baseline (published) ===")
    print(f"  {BASELINE_IMAGE} + {tname}: val {baseline['validation_accuracy']:.4f} "
          f"-> test {baseline['test_accuracy']:.4f} / F1 {baseline['test_macro_f1']:.4f}")
    print("\n=== validation-selected over all configurations ===")
    print(f"  {' + '.join(winner['encoders'])}"
          f"{' + ' + tname if winner['with_text'] else ''} (dim {winner['dim']})")
    print(f"  val {winner['validation_accuracy']:.4f} -> test "
          f"{winner['test_accuracy']:.4f} / F1 {winner['test_macro_f1']:.4f}")
    print(f"  change against baseline: "
          f"{winner['test_accuracy'] - baseline['test_accuracy']:+.4f} "
          f"({crops(winner['test_accuracy'] - baseline['test_accuracy']):+.1f} crops)")
    print(f"  validation-to-test gap: "
          f"{winner['validation_accuracy'] - winner['test_accuracy']:+.4f} "
          f"(baseline {baseline['validation_accuracy'] - baseline['test_accuracy']:+.4f})")

    print("\n=== image-only side ===")
    print(f"  baseline {BASELINE_IMAGE}: test {img_baseline['test_accuracy']:.4f}")
    print(f"  selected {' + '.join(img_only_winner['encoders'])}: test "
          f"{img_only_winner['test_accuracy']:.4f} "
          f"({crops(img_only_winner['test_accuracy'] - img_baseline['test_accuracy']):+.1f} crops)")

    print("\n=== best test cell per size (reported, not selected on) ===")
    for k in range(1, len(names) + 1):
        pool = [r for r in rows if r["n_encoders"] == k and r["with_text"]]
        b = max(pool, key=lambda r: r["test_accuracy"])
        v = best_by_val(pool)
        print(f"  k={k}: val-selected test {v['test_accuracy']:.4f} | "
              f"best-possible test {b['test_accuracy']:.4f} "
              f"({' + '.join(b['encoders'])})")

    payload = {
        "_meta": {
            "protocol": "frozen encoders, per-dimension standardization, "
                        "closed-form ridge; penalty and configuration chosen on "
                        "validation only; test scored once",
            "split_audit": audit, "text_encoder": tname,
            "baseline": f"{BASELINE_IMAGE} + {tname}",
            "configurations": len(rows), "test_crops": int(n_test),
            "one_crop_accuracy": 1.0 / n_test,
        },
        "baseline": baseline, "validation_selected": winner,
        "rows": rows,
    }
    path = out_dir / "multiencoder_fusion.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {path}")
    print("Done.")


if __name__ == "__main__":
    main()
