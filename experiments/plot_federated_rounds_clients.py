#!/usr/bin/env python3
"""Federated rounds and client-count comparisons.

Panel (a) is change-over-time: validation macro F1 per federated round, one line
per system, from the paper-systems run (frozen encoders, SGD heads).

Panel (b) is change-over-a-scale: validation macro F1 against the number of
clients the same fixed data is split across, one line per architecture, from the
scratch-architecture suite. More clients means less data per client, so a
declining line is the expected shape and a flat one is the notable case.

Colour is a fixed-order categorical assignment by system identity. The five-slot
palette was checked with the dataviz validator: all checks pass, with contrast
warnings on two slots, so every line carries a direct end label as the required
relief.
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
SCRATCH = ROOT / "experiments" / "farm_results_standard_genuine.json"
SYSTEMS_JSON = (ROOT / "tea_results" / "federated_all_systems_full"
                / "federated_all_systems.json")
OUTS = [ROOT / "overleaf_final" / "plots", ROOT / "overleaf_final_slim" / "plots"]

# fixed categorical order; colour follows the entity, never its rank
PAPER_SYSTEMS = [
    ("text_distilbert", "Text (DistilBERT)", "#0072B2"),
    ("image_vit_tiny", "Image (ViT-tiny)", "#D55E00"),
    ("vit_distilbert", "ViT + DistilBERT", "#CC79A7"),
]
SCRATCH_ARCHS = [
    ("image_only", "Image only", "#D55E00"),
    ("text_only", "Text only", "#0072B2"),
    ("concat_vlm", "Concat VLM", "#CC79A7"),
    ("attention_vlm", "Attention VLM", "#E69F00"),
    ("cross_attention_vlm", "Cross-attn VLM", "#009E73"),
]
INK, MUTED, GRID = "#1a1a1a", "#6b6b6b", "#d9d9d9"


def parse(key):
    return dict(p.split("=", 1) for p in key.split("|") if "=" in p)


def main() -> None:
    sysd = json.load(open(SYSTEMS_JSON, encoding="utf-8"))
    scratch = json.load(open(SCRATCH, encoding="utf-8"))

    # (a) rounds, at the milder skew where curves are readable
    curves = defaultdict(list)
    for r in sysd["runs"].values():
        if r["alpha"] == 1.0:
            curves[r["system"]].append(r["round_macro_f1"])

    # (b) client counts, mean over seeds
    by_k = defaultdict(list)
    for k, v in scratch["E2_client_sweep"].items():
        f = parse(k)
        by_k[(f["arch"], int(f["K"]))].append(v.get("f1_macro", 0.0))
    ks = sorted({kk for _, kk in by_k})
    pos = np.arange(len(ks))          # categorical x for the client sweep

    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
        "axes.edgecolor": MUTED, "text.color": INK,
        "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    })
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.0, 2.9))

    # ---- (a) per-round convergence ---------------------------------------
    for key, label, colour in PAPER_SYSTEMS:
        runs = curves[key]
        n = min(len(r) for r in runs)
        mean = [st.mean([r[i] for r in runs]) for i in range(n)]
        rounds = np.arange(1, n + 1)
        # markers every fifth round: at 50 rounds one per point is a solid bar
        ax_a.plot(rounds, mean, color=colour, lw=1.6, marker="o", ms=3.4,
                  markevery=max(1, n // 10),
                  markeredgecolor="white", markeredgewidth=0.6,
                  label=label, zorder=3)
        ax_a.text(n + 0.8, mean[-1], f"{mean[-1]:.2f}",
                  fontsize=6.8, color=INK, va="center")
        last_round = n
    ax_a.set_xlabel("Federated round")
    ax_a.set_ylabel("Validation macro F1")
    # derived from the data: a hardcoded limit silently clipped the curves and
    # threw the end labels outside the axes once the run grew past 8 rounds
    ax_a.set_xlim(0.6, last_round + 5.0)
    ax_a.set_ylim(0, 1.02)
    ax_a.set_title(r"(a) Rounds — paper systems, $\alpha$ = 1", loc="left")
    ax_a.legend(loc="lower right", frameon=False)

    # ---- (b) client-count scaling ----------------------------------------
    # Two architectures produce identical values at every client count, so a
    # solid line for both would hide one entirely and lose its identity. The
    # duplicate is drawn dashed and the coincidence is stated, rather than
    # nudging either series off its true value.
    seen = {}
    drawn_labels = []
    for key, label, colour in SCRATCH_ARCHS:
        vals = tuple(round(st.mean(by_k[(key, k)]), 6) for k in ks)
        twin = seen.get(vals)
        seen.setdefault(vals, label)
        style = dict(ls=(0, (4, 2)), lw=1.8) if twin else dict(ls="-", lw=2.0)
        # evenly spaced positions: on a linear axis K=2,3,5 collide at the left
        # once the sweep reaches K=50
        ax_b.plot(pos, vals, color=colour, marker="o", ms=4.0,
                  markeredgecolor="white", markeredgewidth=0.7,
                  label=label + (" (=" + twin + ")" if twin else ""),
                  zorder=4 if twin else 3, **style)
        if not twin:
            ax_b.text(pos[-1] + 0.08, vals[-1], f"{vals[-1]:.2f}",
                      fontsize=6.8, color=INK, va="center")
        drawn_labels.append((label, twin))
    ax_b.axhline(1.0, color=MUTED, lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax_b.text(pos[0], 1.005, "ceiling", fontsize=6.4, color=MUTED,
              va="bottom")
    ax_b.set_xlabel("Number of clients (fixed total data)")
    ax_b.set_ylabel("Validation macro F1")
    ax_b.set_xticks(pos)
    ax_b.set_xticklabels([str(k) for k in ks])
    ax_b.set_xlim(pos[0] - 0.25, pos[-1] + 0.75)
    ax_b.set_ylim(0, 1.08)
    ax_b.set_title("(b) Clients — scratch architectures", loc="left")
    ax_b.legend(loc="lower left", frameon=False, ncol=2)

    for ax in (ax_a, ax_b):
        ax.grid(color=GRID, lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    fig.tight_layout()
    for out_dir in OUTS:
        out_dir.mkdir(parents=True, exist_ok=True)
        p = out_dir / "plot81_federated_rounds_clients.png"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print(f"wrote {p}")
    plt.close(fig)

    print("\n(b) macro F1 by client count")
    print(f"  {'architecture':<22}" + "".join(f"K={k:<7}" for k in ks))
    for key, label, _ in SCRATCH_ARCHS:
        print(f"  {label:<22}"
              + "".join(f"{st.mean(by_k[(key,k)]):<9.3f}" for k in ks))


if __name__ == "__main__":
    main()
