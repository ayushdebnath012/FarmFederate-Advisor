#!/usr/bin/env python3
"""Ladder rungs as field notes are deleted -- the axis the search figure omits.

The pipeline-search figure fixes notes at complete and varies the search size.
This one fixes nothing and varies the notes, which is where the ladders stop
being flat: every rung of the text and text+image ladders is redrawn at three
note sparsities on the identical 75 crops.

Panel (a): text rungs. Panel (b): text+image rungs, with the best text rung and
the best single visual model repeated as reference lines, so the condition under
which fusion actually leads both parents is readable rather than asserted.

Deletion is deterministic in (row, token), so every rung at a given sparsity
sees the identical corruption of the identical notes.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TEXT = ROOT / "tea_results" / "text_pipeline_ladder" / "text_pipeline_ladder.json"
MM = ROOT / "tea_results" / "multimodal_pipeline_ladder" / "multimodal_pipeline_ladder.json"
VISUAL = ROOT / "tea_results" / "visual_ladder" / "visual_ladder.json"
OUTS = [ROOT / "overleaf_final" / "plots", ROOT / "overleaf_final_slim" / "plots"]

# fixed categorical order; colour follows the rung, never its rank
CYCLE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
MARKS = ["o", "s", "^", "D", "v", "P"]
INK, MUTED, GRID = "#1a1a1a", "#6b6b6b", "#d9d9d9"


def ladder(path):
    d = json.load(open(path, encoding="utf-8"))
    keys = sorted(d["ladder"], key=lambda k: float(k.split("=")[1]))
    sparsities = [float(k.split("=")[1]) for k in keys]
    names = list(d["ladder"][keys[0]].keys())
    series = {n: [d["ladder"][k][n]["test_accuracy"] for k in keys] for n in names}
    return sparsities, series


def draw(ax, sparsities, series, title):
    xs = np.arange(len(sparsities))
    for i, (name, vals) in enumerate(series.items()):
        gate = "gate" in name.lower()
        ax.plot(xs, vals, color=CYCLE[i % len(CYCLE)], marker=MARKS[i % len(MARKS)],
                ms=4.2, lw=1.5 if not gate else 1.3,
                ls="-" if not gate else (0, (4, 2)),
                markerfacecolor="white" if gate else CYCLE[i % len(CYCLE)],
                markeredgecolor=CYCLE[i % len(CYCLE)], markeredgewidth=1.0,
                label=name.replace(" (not single model)", "*"), zorder=3)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{100 * (1 - s):.0f}" for s in sparsities])
    ax.set_xlabel("Field notes retained (%)")
    ax.set_ylim(0.3, 1.06)
    ax.set_title(title, loc="left", color=INK)
    ax.grid(color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main() -> None:
    sp_t, s_text = ladder(TEXT)
    sp_m, s_mm = ladder(MM)
    assert sp_t == sp_m, f"ladders disagree on sparsity grid: {sp_t} vs {sp_m}"
    vis = json.load(open(VISUAL, encoding="utf-8"))["rungs"]
    best_visual = max(r["test_accuracy"] for r in vis if not r["is_gate"])

    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 6.2,
        "axes.edgecolor": MUTED, "text.color": INK,
        "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    })
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)

    draw(ax_a, sp_t, s_text, "(a) Text rungs")
    ax_a.set_ylabel("Test accuracy")
    ax_a.legend(frameon=False, loc="lower left", ncol=1)

    draw(ax_b, sp_m, s_mm, "(b) Text $+$ image rungs")
    best_text = [max(v[i] for v in s_text.values()) for i in range(len(sp_m))]
    ax_b.plot(np.arange(len(sp_m)), best_text, color=MUTED, lw=1.2,
              ls=(0, (1.5, 1.5)), zorder=2)
    ax_b.text(len(sp_m) - 1, best_text[-1] - 0.045, "best text rung",
              fontsize=6.2, color=MUTED, ha="right")
    ax_b.axhline(best_visual, color=MUTED, lw=1.2, ls=(0, (5, 2)), zorder=2)
    ax_b.text(0.02, best_visual + 0.012, "best single visual model",
              fontsize=6.2, color=MUTED)
    ax_b.legend(frameon=False, loc="lower left", ncol=1)

    fig.tight_layout(pad=0.6)
    for out in OUTS:
        if out.parent.exists():
            out.mkdir(parents=True, exist_ok=True)
            fig.savefig(out / "plot88_sparsity_ladder.pdf", bbox_inches="tight")
            print(f"wrote {out / 'plot88_sparsity_ladder.pdf'}")
    png = ROOT / "experiments" / "plot88_sparsity_ladder.png"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    print(f"wrote {png}")
    print(f"  best single visual model: {best_visual:.3f}")
    print(f"  best text rung per sparsity:  {[round(b, 3) for b in best_text]}")
    print(f"  best fusion rung per sparsity: "
          f"{[round(max(v[i] for v in s_mm.values()), 3) for i in range(len(sp_m))]}")


if __name__ == "__main__":
    main()
