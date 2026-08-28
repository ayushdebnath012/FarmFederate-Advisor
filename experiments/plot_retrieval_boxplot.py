#!/usr/bin/env python3
"""Retrieval-score distribution per tea class, as a box plot.

Replaces plots/rag_06_score_boxplot.png, which shows the same idea on the
superseded five-class crop-stress corpus (water stress, nutrient deficiency,
pest risk, disease risk, heat stress) and therefore cannot sit beside the tea
results. This one is built from the current advisory run on the five tea
classes, from the per-query top-1 similarities that run now records.

The distribution is split by whether the top-1 hit was the
correct class, so a high median is not mistaken for a correct one -- the scores
are high everywhere, which is the point: similarity does not separate classes.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "tea_results" / "advisory_retrieval_v1" / "advisory_retrieval_results.json"
OUTS = [ROOT / "overleaf_final" / "plots", ROOT / "overleaf_final_slim" / "plots"]

PRETTY = {"LEAF_BLIGHT": "Blight", "LEAF_HOPPERS": "Hoppers",
          "LEAF_RUST": "Rust", "LOOPER_CATERPILLARS": "Looper",
          "MOSQUITO_BUG": "Mosq. bug"}
HIT, MISS = "#0072B2", "#D55E00"
INK, MUTED, GRID = "#1a1a1a", "#6b6b6b", "#d9d9d9"


def main() -> None:
    d = json.load(open(SRC, encoding="utf-8"))
    order = d["class_order"]
    hit, miss = defaultdict(list), defaultdict(list)
    for r in d["per_query"]:
        (hit if r["top1_correct"] else miss)[r["true_label"]].append(r["top1_score"])

    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
        "axes.edgecolor": MUTED, "text.color": INK,
        "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    })
    fig, ax_a = plt.subplots(1, 1, figsize=(3.45, 2.7))

    # ---- (a) score distribution, correct vs incorrect top-1 --------------
    pos = np.arange(len(order))
    for data, colour, off, tag in ((hit, HIT, -0.18, "top-1 correct"),
                                   (miss, MISS, 0.18, "top-1 wrong")):
        vals = [data[c] for c in order]
        keep = [i for i, v in enumerate(vals) if v]
        bp = ax_a.boxplot([vals[i] for i in keep], positions=pos[keep] + off,
                          widths=0.3, patch_artist=True, showfliers=True,
                          medianprops=dict(color=INK, lw=1.2),
                          flierprops=dict(marker="o", ms=2.4, markerfacecolor="none",
                                          markeredgecolor=colour, markeredgewidth=0.6),
                          whiskerprops=dict(color=colour, lw=0.9),
                          capprops=dict(color=colour, lw=0.9))
        for box in bp["boxes"]:
            box.set(facecolor=colour, alpha=0.55, edgecolor=colour, linewidth=0.9)
        ax_a.plot([], [], color=colour, lw=6, alpha=0.55, label=tag)
    ax_a.set_xticks(pos)
    ax_a.set_xticklabels([PRETTY[c] for c in order], rotation=18, ha="right")
    ax_a.set_ylabel("Top-1 cosine similarity")
    ax_a.set_title("Retrieval score by class", loc="left", color=INK)
    ax_a.legend(frameon=False, loc="lower left", ncol=1)

    for ax in (ax_a,):
        ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    fig.tight_layout(pad=0.6)
    for out in OUTS:
        if out.parent.exists():
            out.mkdir(parents=True, exist_ok=True)
            fig.savefig(out / "plot89_retrieval_boxplot.pdf", bbox_inches="tight")
            print(f"wrote {out / 'plot89_retrieval_boxplot.pdf'}")
    png = ROOT / "experiments" / "plot89_retrieval_boxplot.png"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    print(f"wrote {png}")
    for c in order:
        print(f"  {PRETTY[c]:<10} correct n={len(hit[c]):<3} wrong n={len(miss[c])}")


if __name__ == "__main__":
    main()
