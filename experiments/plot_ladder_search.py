#!/usr/bin/env python3
"""Figure for the three pipeline ladders, which currently exist only as tables.

Panel (a) is change-along-an-ordered-axis: test accuracy against the number of
validation-fitted configurations each rung searched. The visual ladder climbs
across that span while the text and fusion ladders are flat at the ceiling --
the paper's central claim, which a table of numbers states but does not show.

Panel (b) is the cost of that search: validation minus test accuracy for every
rung of every ladder. Gates sit at the right-hand end and fall furthest, so the
penalty for fitting thousands of routings on 74 validation crops is visible as
a trend rather than asserted in prose.

Colour carries the ladder (the entity); marker fill carries whether a rung is a
deployable single model or a per-class gate. The visual ladder is read out of
a frozen JSON extracted from tab:visual_gate; the other two come from their
own result JSON.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
VISUAL = ROOT / "tea_results" / "visual_ladder" / "visual_ladder.json"
TEXT = ROOT / "tea_results" / "text_pipeline_ladder" / "text_pipeline_ladder.json"
MM = ROOT / "tea_results" / "multimodal_pipeline_ladder" / "multimodal_pipeline_ladder.json"
OUTS = [ROOT / "overleaf_final" / "plots", ROOT / "overleaf_final_slim" / "plots"]

LADDERS = [("Visual", "#D55E00"), ("Text", "#0072B2"), ("Text $+$ image", "#009E73")]
INK, MUTED, GRID = "#1a1a1a", "#6b6b6b", "#d9d9d9"
B = chr(92)


def visual_rungs():
    """Read the frozen visual ladder rungs.

    These were extracted once from tab:visual_gate; keeping them in a JSON means
    the figure no longer breaks if that table is trimmed from the paper.
    """
    d = json.load(open(VISUAL, encoding="utf-8"))
    return [(r["name"], r["candidates"], r["validation_accuracy"],
             r["test_accuracy"], r["is_gate"]) for r in d["rungs"]]


def json_rungs(path):
    d = json.load(open(path, encoding="utf-8"))
    rows = d["ladder"]["sparsity=0.0"]
    return [(k, v["candidates"], v["validation_accuracy"], v["test_accuracy"],
             "gate" in k.lower()) for k, v in rows.items()]


def main() -> None:
    data = [visual_rungs(), json_rungs(TEXT), json_rungs(MM)]
    for (name, _), rows in zip(LADDERS, data):
        print(f"{name}: {len(rows)} rungs, "
              f"{sum(1 for r in rows if r[4])} gates")
        if not rows:
            raise SystemExit(f"no rungs parsed for {name} -- refusing to plot")

    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
        "axes.edgecolor": MUTED, "text.color": INK,
        "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    })
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.0, 2.75))

    # ---- (a) does a bigger search buy accuracy? --------------------------
    # rungs are alternative methods, not stages of one progression, so they are
    # not joined: a connecting line would imply a trajectory that does not exist
    for (label, colour), rows in zip(LADDERS, data):
        single = [r for r in rows if not r[4]]
        gates = [r for r in rows if r[4]]
        ax_a.scatter([r[1] for r in single], [r[3] for r in single],
                     facecolors=colour, edgecolors=colour, s=30,
                     linewidths=0.8, label=label, zorder=3)
        if gates:
            ax_a.scatter([r[1] for r in gates], [r[3] for r in gates],
                         facecolors="white", edgecolors=colour, s=38,
                         linewidths=1.4, zorder=4)
    ax_a.set_xscale("log")
    ax_a.set_xlabel("Validation-fitted configurations searched")
    ax_a.set_ylabel("Test accuracy")
    ax_a.set_ylim(0.55, 1.06)
    ax_a.set_title("(a) Search size vs. accuracy", loc="left", color=INK)
    ax_a.axhline(1.0, color=MUTED, lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax_a.text(1.1, 1.008, "ceiling", fontsize=6.4, color=MUTED, va="bottom")
    ax_a.legend(frameon=False, loc="center right", ncol=1)

    # ---- (b) what the search costs on 74 validation crops ----------------
    for (label, colour), rows in zip(LADDERS, data):
        for _n, cand, val, test, is_gate in rows:
            ax_b.scatter(cand, val - test, s=40 if is_gate else 28,
                         facecolors="white" if is_gate else colour,
                         edgecolors=colour, linewidths=1.4 if is_gate else 0.8,
                         zorder=3)
    ax_b.axhline(0.0, color=MUTED, lw=0.9, zorder=2)
    ax_b.set_xscale("log")
    ax_b.set_xlabel("Validation-fitted configurations searched")
    ax_b.set_ylabel("Validation $-$ test accuracy")
    ax_b.set_title("(b) Cost of the search", loc="left", color=INK)
    handles = [plt.Line2D([], [], ls="none", marker="o", ms=5.5,
                          markerfacecolor=MUTED, markeredgecolor=MUTED,
                          label="Single model"),
               plt.Line2D([], [], ls="none", marker="o", ms=6,
                          markerfacecolor="white", markeredgecolor=MUTED,
                          markeredgewidth=1.4, label="Per-class gate")]
    ax_b.legend(handles=handles, frameon=False, loc="upper left")

    for ax in (ax_a, ax_b):
        ax.grid(color=GRID, lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    fig.tight_layout(pad=0.6)
    for out in OUTS:
        if out.parent.exists():
            out.mkdir(parents=True, exist_ok=True)
            fig.savefig(out / "plot85_ladder_search.pdf", bbox_inches="tight")
            print(f"wrote {out / 'plot85_ladder_search.pdf'}")
    png = ROOT / "experiments" / "plot85_ladder_search.png"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
