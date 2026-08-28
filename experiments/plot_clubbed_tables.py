#!/usr/bin/env python3
"""Figure for the clubbed modality comparison.

Panel (a) is an identity/magnitude comparison: within each modality, the
classical feature baseline beside the best pretrained system, so the value the
transformer encoders actually add is readable per modality rather than asserted.

Panel (b) is change-along-an-ordered-axis: test accuracy as field notes are
progressively deleted, one line per modality. Colour carries the modality (the
entity), line style carries the deployment -- solid centralized, dashed
federated -- so identity is never encoded by rank and the federation cost is
read as the vertical gap between a pair of same-coloured lines.

Colour is the repo's existing Okabe-Ito categorical theme, assigned in fixed
order. Values are labelled on the bars in panel (a), which is also the required
relief for the pink slot's contrast warning from the dataviz validator.
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

MODALITIES = [
    ("text_only", "Text only", "#0072B2", "o"),
    ("image_only", "Image only", "#D55E00", "s"),
    ("multimodal", "Image $+$ text", "#009E73", "^"),
]
INK, MUTED, GRID = "#1a1a1a", "#6b6b6b", "#d9d9d9"


def best(rows, classical: bool, field="central_test_accuracy"):
    """Best row of one kind within a family; None if that kind is absent."""
    pool = {n: r for n, r in rows.items() if bool(r["classical"]) == classical}
    if not pool:
        return None, None
    n = max(pool, key=lambda k: (pool[k][field], pool[k]["central_test_macro_f1"]))
    return n, pool[n]


def main() -> None:
    d = json.load(open(SRC, encoding="utf-8"))
    t = d["tables"]
    keys = sorted(t, key=lambda k: float(k.split("=")[1]))
    sparsities = [float(k.split("=")[1]) for k in keys]
    completeness = [100 * (1 - s) for s in sparsities]
    # the panel-(a) operating point: the sparsest setting where notes still
    # exist, i.e. where the modalities are not hidden by the text ceiling
    op = keys[1] if len(keys) > 1 else keys[0]

    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
        "axes.edgecolor": MUTED, "text.color": INK,
        "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    })
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.0, 2.75))

    # ---- (a) classical floor vs best pretrained system, per modality ------
    x = np.arange(len(MODALITIES))
    width = 0.34
    for j, (kind, hatch, alpha, tag) in enumerate(
            ((True, "///", 0.55, "Classical features"),
             (False, None, 1.0, "Pretrained encoder"))):
        vals, labels = [], []
        for fam, _, _, _ in MODALITIES:
            n, r = best(t[op][fam], kind)
            vals.append(r["central_test_accuracy"] if r else np.nan)
            labels.append(n or "")
        colours = [c for _, _, c, _ in MODALITIES]
        bars = ax_a.bar(x + (j - 0.5) * width, vals, width * 0.92,
                        color=colours, alpha=alpha, hatch=hatch,
                        edgecolor="white", linewidth=0.8,
                        label=tag, zorder=3)
        for b, v in zip(bars, vals):
            if not np.isnan(v):
                ax_a.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}",
                          ha="center", va="bottom", fontsize=6.6, color=INK)
    sp_a = float(op.split("=")[1])
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([m[1] for m in MODALITIES])
    ax_a.set_ylabel("Test accuracy")
    ax_a.set_ylim(0, 1.30)
    ax_a.set_title(f"(a) Classical vs. pretrained, {sp_a:.0%} of notes deleted",
                   loc="left", color=INK)
    ax_a.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    ax_a.set_axisbelow(True)
    for s in ("top", "right"):
        ax_a.spines[s].set_visible(False)
    # legend keyed on texture, not colour: colour already carries modality
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=MUTED, alpha=0.55,
                             hatch="///", edgecolor="white"),
               plt.Rectangle((0, 0), 1, 1, facecolor=MUTED, edgecolor="white")]
    ax_a.legend(handles, ["Classical features", "Pretrained encoder"],
                frameon=False, loc="upper center", ncol=2,
                handlelength=1.4, columnspacing=1.0)

    # ---- (b) accuracy as notes are deleted, centralized vs federated ------
    for fam, label, colour, marker in MODALITIES:
        cen, fed = [], []
        for k in keys:
            _, r = best(t[k][fam], False)
            if r is None:
                _, r = best(t[k][fam], True)
            cen.append(r["central_test_accuracy"])
            fed.append(r["fed_test_accuracy"])
        ax_b.plot(completeness, cen, color=colour, marker=marker,
                  linewidth=2, markersize=5, label=label, zorder=3)
        ax_b.plot(completeness, fed, color=colour, marker=marker,
                  linestyle="--", linewidth=1.6, markersize=4.5,
                  markerfacecolor="white", alpha=0.95, zorder=3)

    ax_b.set_xlabel("Field notes retained (%)")
    ax_b.set_ylabel("Test accuracy")
    ax_b.set_xticks(completeness)
    ax_b.set_xticklabels([f"{c:.0f}" for c in completeness])
    ax_b.invert_xaxis()
    ax_b.set_ylim(0, 1.06)
    ax_b.set_title("(b) Centralized (solid) vs. federated (dashed)",
                   loc="left", color=INK)
    ax_b.grid(color=GRID, linewidth=0.6, zorder=0)
    ax_b.set_axisbelow(True)
    for s in ("top", "right"):
        ax_b.spines[s].set_visible(False)
    ax_b.legend(frameon=False, loc="lower left", ncol=1)

    fig.tight_layout(pad=0.6)
    for out in OUTS:
        if out.parent.exists():
            out.mkdir(parents=True, exist_ok=True)
            p = out / "plot82_clubbed_modality.pdf"
            fig.savefig(p, bbox_inches="tight")
            print(f"wrote {p}")
    png = ROOT / "experiments" / "plot82_clubbed_modality.png"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
