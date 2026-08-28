#!/usr/bin/env python3
"""Figure for the per-system federated comparison.

Panel (a) is a magnitude comparison: FedAvg macro-F1 per system at each skew,
with the local-only control drawn as a tick inside each bar so the federation
gain is readable without a second chart.

Panel (b) is change-over-time: per-round convergence, one line per system.

Colour is the repo's existing Okabe-Ito categorical theme, assigned by system
identity in fixed order (never by rank). The palette was checked with the
dataviz validator: all checks pass; the pink slot warns on contrast versus the
surface, so every bar carries a visible value label as the required relief.
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
SRC = ROOT / "tea_results" / "federated_all_systems_full" / "federated_all_systems.json"
OUT = ROOT / "overleaf_final" / "plots"
OUT_SLIM = ROOT / "overleaf_final_slim" / "plots"

# categorical theme: fixed order, colour follows the entity not its rank
SYSTEMS = [
    ("text_distilbert", "Text (frozen DistilBERT)", "#0072B2"),
    ("image_vit_tiny", "Image (frozen ViT-tiny)", "#D55E00"),
    ("vit_distilbert", "ViT + DistilBERT fusion", "#CC79A7"),
]
INK = "#1a1a1a"
MUTED = "#6b6b6b"
GRID = "#d9d9d9"


def load():
    d = json.load(open(SRC, encoding="utf-8"))
    fed, loc, curves = defaultdict(list), defaultdict(list), defaultdict(list)
    for r in d["runs"].values():
        key = (r["system"], r["alpha"])
        fed[key].append(r["fedavg_final_macro_f1"])
        loc[key].append(r["local_only_mean_macro_f1"])
        curves[key].append(r["round_macro_f1"])
    return d, fed, loc, curves


def main() -> None:
    d, fed, loc, curves = load()
    alphas = sorted({a for _, a in fed})

    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
        "axes.edgecolor": MUTED, "text.color": INK,
        "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    })
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.0, 2.7))

    # ---- (a) magnitude: grouped bars, local-only as an inset tick ----------
    x = np.arange(len(alphas))
    width = 0.24
    for i, (key, label, colour) in enumerate(SYSTEMS):
        off = (i - 1) * (width + 0.02)          # 2px-equivalent gap between fills
        vals = [st.mean(fed[(key, a)]) for a in alphas]
        locs = [st.mean(loc[(key, a)]) for a in alphas]
        bars = ax_a.bar(x + off, vals, width, color=colour, label=label,
                        edgecolor="white", linewidth=0.6, zorder=3)
        # local-only control drawn inside its own bar
        for xi, lv in zip(x + off, locs):
            ax_a.plot([xi - width / 2, xi + width / 2], [lv, lv],
                      color="white", lw=1.4, zorder=5)
            ax_a.plot([xi - width / 2, xi + width / 2], [lv, lv],
                      color=INK, lw=0.8, ls=(0, (2, 1.4)), zorder=6)
        # visible value labels: required relief for the contrast WARN
        for b, v in zip(bars, vals):
            ax_a.text(b.get_x() + b.get_width() / 2, v + 0.018, f"{v:.2f}",
                      ha="center", va="bottom", fontsize=6.6, color=INK)

    ax_a.set_xticks(x)
    ax_a.set_xticklabels([rf"$\alpha$ = {a}" for a in alphas])
    ax_a.set_ylabel("Validation macro F1")
    ax_a.set_ylim(0, 1.0)
    ax_a.set_title("(a) FedAvg by system and label skew", loc="left")
    ax_a.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    ax_a.set_axisbelow(True)
    for s in ("top", "right"):
        ax_a.spines[s].set_visible(False)
    ax_a.plot([], [], color=INK, lw=0.8, ls=(0, (2, 1.4)),
              label="local-only control")

    # ---- (b) change over time: per-round convergence ----------------------
    alpha_b = alphas[-1]
    for key, label, colour in SYSTEMS:
        runs = curves[(key, alpha_b)]
        n = min(len(r) for r in runs)
        mean = [st.mean([r[i] for r in runs]) for i in range(n)]
        rounds = np.arange(1, n + 1)
        ax_b.plot(rounds, mean, color=colour, lw=2.0, marker="o", ms=3.6,
                  markeredgecolor="white", markeredgewidth=0.6, zorder=3)
        ax_b.text(rounds[-1] + 0.12, mean[-1], f"{mean[-1]:.2f}",
                  fontsize=6.6, color=INK, va="center")

    ax_b.set_xlabel("Federated round")
    ax_b.set_ylabel("Validation macro F1")
    ax_b.set_xlim(0.6, len(rounds) + 0.9)
    ax_b.set_ylim(0, 1.0)
    ax_b.set_title(rf"(b) Convergence at $\alpha$ = {alpha_b}", loc="left")
    ax_b.grid(color=GRID, lw=0.6, zorder=0)
    ax_b.set_axisbelow(True)
    for s in ("top", "right"):
        ax_b.spines[s].set_visible(False)

    handles, labels = ax_a.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.07, 1, 1))

    for out_dir in (OUT, OUT_SLIM):
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "plot80_federated_all_systems.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"wrote {path}")
    plt.close(fig)

    # ---- console table (the second contrast relief, and a sanity check) ---
    print("\nFedAvg macro F1 (mean over seeds), local-only in brackets")
    print(f"  {'system':<26}" + "".join(f"a={a:<14}" for a in alphas))
    for key, label, _ in SYSTEMS:
        row = "".join(
            f"{st.mean(fed[(key,a)]):.3f} [{st.mean(loc[(key,a)]):.3f}]  "
            for a in alphas
        )
        print(f"  {label:<26}{row}")


if __name__ == "__main__":
    main()
