#!/usr/bin/env python3
"""Figure for the aggregator comparison, which exists only as Table V(b).

Panel (a): FedAvg, FedProx, SCAFFOLD and FedBN per architecture at the strongest
skew, with the local-only control drawn as a tick inside each group so the
federation gain is readable without a second chart. Three aggregators coincide
for most architectures once the runs are trained out; SCAFFOLD is the outlier
and the bars show how far it falls rather than leaving it to prose.

Panel (b): the same architectures across all four skews under FedAvg, so the
one arm that never approaches the ceiling (image-only) is visible against the
four that do.

Local-only records store `mean_f1`; federated records store `f1_macro`. Reading
the wrong field silently yields zero, so the accessor is shared.
"""

from __future__ import annotations

import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "experiments" / "farm_results_standard_genuine.json"
OUTS = [ROOT / "overleaf_final" / "plots", ROOT / "overleaf_final_slim" / "plots"]

ARCHS = [("image_only", "Image"), ("text_only", "Text"),
         ("concat_vlm", "Concat"), ("attention_vlm", "Attn"),
         ("cross_attention_vlm", "Cross-attn")]
AGGS = [("fedavg", "FedAvg", "#0072B2"), ("fedprox", "FedProx", "#009E73"),
        ("scaffold", "SCAFFOLD", "#D55E00"), ("fedbn", "FedBN", "#CC79A7")]
INK, MUTED, GRID = "#1a1a1a", "#6b6b6b", "#d9d9d9"


def parse(key):
    out = {}
    for part in key.split("|"):
        if "=" in part:
            a, b = part.split("=", 1)
            out[a] = b
        else:
            out["_alg"] = part
    return out


def score(rec):
    return rec.get("f1_macro", rec.get("mean_f1"))


def main() -> None:
    d = json.load(open(SRC, encoding="utf-8"))
    agg, alpha_sweep = defaultdict(list), defaultdict(list)
    for k, v in d["E8_baselines"].items():
        f = parse(k)
        agg[(f["arch"], f["alpha"], f["_alg"])].append(score(v))
    for k, v in d["E1_alpha_sweep"].items():
        f = parse(k)
        if f.get("split") == "corrected":
            alpha_sweep[(f["arch"], float(f["alpha"]))].append(score(v))
    alphas = sorted({a for _, a in alpha_sweep})

    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
        "axes.edgecolor": MUTED, "text.color": INK,
        "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    })
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.0, 2.75))

    # ---- (a) aggregator per architecture at the strongest skew -----------
    x = np.arange(len(ARCHS))
    w = 0.20
    for j, (key, label, colour) in enumerate(AGGS):
        vals = [st.mean(agg[(a, "0.1", key)]) if agg[(a, "0.1", key)] else np.nan
                for a, _ in ARCHS]
        ax_a.bar(x + (j - 1.5) * w, vals, w * 0.9, color=colour,
                 edgecolor="white", linewidth=0.7, label=label, zorder=3)
    for i, (a, _) in enumerate(ARCHS):                 # local-only control tick
        loc = agg[(a, "0.1", "local_only")]
        if loc:
            ax_a.plot([x[i] - 2 * w, x[i] + 2 * w], [st.mean(loc)] * 2,
                      color=INK, lw=1.0, ls=(0, (2, 1.6)), zorder=5)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([n for _, n in ARCHS], rotation=12, ha="right")
    ax_a.set_ylabel("Validation macro F1")
    ax_a.set_ylim(0, 1.12)
    ax_a.set_title(r"(a) Aggregators at $\alpha=0.1$", loc="left", color=INK)
    ax_a.legend(frameon=False, loc="upper left", ncol=2, columnspacing=1.0,
                handlelength=1.2)

    # ---- (b) FedAvg across skews -----------------------------------------
    cmap = ["#D55E00", "#0072B2", "#CC79A7", "#E69F00", "#009E73"]
    marks = ["s", "o", "^", "D", "v"]
    # several arms sit on the 1.000 ceiling at every skew; drawing them all
    # solid hides all but the last. Duplicates are dashed and the coincidence
    # is named, rather than nudging any series off its true value.
    seen, dashes = {}, [(0, (5, 2)), (0, (1.5, 1.5)), (0, (4, 1, 1, 1))]
    for (key, label), colour, mk in zip(ARCHS, cmap, marks):
        vals = tuple(round(st.mean(alpha_sweep[(key, a)]), 6) for a in alphas)
        twin = seen.get(vals)
        seen.setdefault(vals, label)
        n_twin = sum(1 for v in seen if v == vals) if twin else 0
        style = dict(ls=dashes[len(
            [1 for k, v in seen.items() if k == vals]) % len(dashes)], lw=1.6) \
            if twin else dict(ls="-", lw=1.8)
        ax_b.plot(range(len(alphas)), vals, color=colour, marker=mk, ms=4.5,
                  markeredgecolor="white", markeredgewidth=0.6,
                  label=label + (f" (={twin})" if twin else ""),
                  zorder=4 if twin else 3, **style)
    ax_b.axhline(1.0, color=MUTED, lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax_b.text(-0.15, 1.008, "ceiling", fontsize=6.4, color=MUTED, va="bottom")
    ax_b.set_xticks(range(len(alphas)))
    ax_b.set_xticklabels([f"{a:g}" for a in alphas])
    ax_b.set_xlabel(r"Dirichlet $\alpha$ (label skew)")
    ax_b.set_ylabel("Validation macro F1")
    ax_b.set_ylim(0, 1.12)
    ax_b.set_title("(b) FedAvg across skews", loc="left", color=INK)
    ax_b.legend(frameon=False, loc="lower right", ncol=2, columnspacing=1.0)

    for ax in (ax_a, ax_b):
        ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    fig.tight_layout(pad=0.6)
    for out in OUTS:
        if out.parent.exists():
            out.mkdir(parents=True, exist_ok=True)
            fig.savefig(out / "plot86_aggregators.pdf", bbox_inches="tight")
            print(f"wrote {out / 'plot86_aggregators.pdf'}")
    png = ROOT / "experiments" / "plot86_aggregators.png"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
