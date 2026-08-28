#!/usr/bin/env python3
"""Per-encoder view of the clubbed tables, which the modality figure summarises.

The clubbed modality figure plots only the best system per family, so the
spread across encoders -- and the fact that the classical text baseline beats
every transformer -- is visible only by reading a twelve-row table.

Panel (a): every text and image encoder at 50% note deletion, classical rows
hatched, centralized bar with the federated value drawn as a tick inside it, so
the federation cost is read without a second chart.

Panel (b): centralized minus federated per system, which is the quantity the
table makes the reader compute. Bars near zero mean federating that encoder is
free; the image encoders cluster there because their features do not move.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "tea_results" / "clubbed_tables" / "clubbed_tables.json"
OUTS = [ROOT / "overleaf_final" / "plots", ROOT / "overleaf_final_slim" / "plots"]

TEXT_ORDER = ["TF-IDF + linear SVM", "BERT-tiny", "BERT-mini", "BERT-small",
              "BERT-medium", "DistilBERT"]
IMAGE_ORDER = ["Colour+HOG SVM", "Swin-tiny", "ConvNeXT-tiny", "EfficientNet",
               "DeiT-tiny", "ViT-Base"]
SHORT = {"TF-IDF + linear SVM": "TF-IDF", "Colour+HOG SVM": "Colour/HOG",
         "EfficientNet": "EffNet-B0", "ViT-Base": "ViT-B/16",
         "ConvNeXT-tiny": "ConvNeXT", "DistilBERT": "DistilBERT"}
TEXT_C, IMAGE_C = "#0072B2", "#D55E00"
INK, MUTED, GRID = "#1a1a1a", "#6b6b6b", "#d9d9d9"


def label(name):
    return SHORT.get(name, name)


def main() -> None:
    d = json.load(open(SRC, encoding="utf-8"))
    t = d["tables"]
    keys = sorted(t, key=lambda k: float(k.split("=")[1]))
    op = keys[1] if len(keys) > 1 else keys[0]        # 50% deletion
    sp = float(op.split("=")[1])

    rows = []
    for fam, order, colour in (("text_only", TEXT_ORDER, TEXT_C),
                               ("image_only", IMAGE_ORDER, IMAGE_C)):
        for name in order:
            r = t[op][fam].get(name)
            if r:
                rows.append((label(name), r["central_test_accuracy"],
                             r["fed_test_accuracy"], colour, r["classical"], fam))
    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
        "xtick.labelsize": 7, "ytick.labelsize": 7.5, "legend.fontsize": 7,
        "axes.edgecolor": MUTED, "text.color": INK,
        "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    })
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.0, 2.9))

    # ---- (a) every encoder, centralized bar with federated tick ----------
    x = np.arange(len(rows))
    for i, (nm, cen, fed, colour, classical, _f) in enumerate(rows):
        ax_a.bar(i, cen, 0.72, color=colour, alpha=0.55 if classical else 1.0,
                 hatch="///" if classical else None, edgecolor="white",
                 linewidth=0.7, zorder=3)
        ax_a.plot([i - 0.36, i + 0.36], [fed, fed], color=INK, lw=1.0,
                  ls=(0, (2, 1.6)), zorder=5)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([r[0] for r in rows], rotation=45, ha="right")
    ax_a.set_ylabel("Test accuracy")
    ax_a.set_ylim(0, 1.05)
    ax_a.set_title(f"(a) Every encoder at {sp:.0%} note deletion",
                   loc="left", color=INK)
    ax_a.axvline(len(TEXT_ORDER) - 0.5, color=MUTED, lw=0.8, ls=":", zorder=2)
    ax_a.text(len(TEXT_ORDER) / 2 - 0.5, 1.0, "text", ha="center",
              fontsize=7, color=TEXT_C)
    ax_a.text(len(TEXT_ORDER) + len(IMAGE_ORDER) / 2 - 0.5, 1.0, "image",
              ha="center", fontsize=7, color=IMAGE_C)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=MUTED, alpha=0.55,
                             hatch="///", edgecolor="white"),
               plt.Line2D([], [], color=INK, lw=1.0, ls=(0, (2, 1.6)))]
    # opaque frame: the legend sits over the bars, so it needs a ground
    ax_a.legend(handles, ["Classical features", "Federated"], frameon=True,
                facecolor="white", edgecolor="none", framealpha=0.92,
                loc="lower left", ncol=1, handlelength=1.4)

    # ---- (b) what federating costs each encoder --------------------------
    deltas = [(nm, cen - fed, colour) for nm, cen, fed, colour, _c, _f in rows]
    ax_b.barh(np.arange(len(deltas)), [d[1] for d in deltas], 0.72,
              color=[d[2] for d in deltas], edgecolor="white", linewidth=0.7,
              zorder=3)
    ax_b.axvline(0.0, color=MUTED, lw=0.9, zorder=4)
    ax_b.set_yticks(np.arange(len(deltas)))
    ax_b.set_yticklabels([d[0] for d in deltas], fontsize=7)
    ax_b.invert_yaxis()
    ax_b.set_xlabel("Centralized $-$ federated test accuracy")
    ax_b.set_title("(b) Cost of federating", loc="left", color=INK)

    for ax in (ax_a, ax_b):
        ax.grid(axis="y" if ax is ax_a else "x", color=GRID, lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        for sp_ in ("top", "right"):
            ax.spines[sp_].set_visible(False)

    fig.tight_layout(pad=0.6)
    for out in OUTS:
        if out.parent.exists():
            out.mkdir(parents=True, exist_ok=True)
            fig.savefig(out / "plot87_encoder_detail.pdf", bbox_inches="tight")
            print(f"wrote {out / 'plot87_encoder_detail.pdf'}")
    png = ROOT / "experiments" / "plot87_encoder_detail.png"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
