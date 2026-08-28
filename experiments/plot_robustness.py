#!/usr/bin/env python3
"""The four untested federated failure modes, drawn on the audited split.

Two IEEE-column-pair figures rather than one four-panel block, so each can be
placed next to the paragraph that discusses it:

  plot90  (a) severe label skew   (b) client dropout
  plot91  (a) straggler staleness (b) poisoned updates, mean vs median

Panels (a)-(c) plot all three federated systems, because the question a
multimodal paper has to answer is not "does FedAvg degrade" -- it always does --
but whether the fused system degrades faster than its parents. Panel (d) fixes
the system and varies the aggregator, since that is the only axis on which the
poisoning result is actionable.

Colour follows the system, never its rank, so a panel with a different ordering
still shows the same system in the same hue.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "tea_results" / "federated_robustness" / "federated_robustness.json"
OUTS = [ROOT / "overleaf_final" / "plots", ROOT / "overleaf_final_slim" / "plots"]

SYSTEMS = ["text", "image", "fusion"]
NICE = {"text": "Text", "image": "Image", "fusion": "Text $+$ image"}
COLOR = {"text": "#D55E00", "image": "#0072B2", "fusion": "#009E73"}
MARK = {"text": "s", "image": "o", "fusion": "D"}
AGG_COLOR = {"mean": "#6b6b6b", "median": "#CC79A7"}
INK, MUTED, GRID = "#1a1a1a", "#6b6b6b", "#d9d9d9"


def series(store, keys):
    return {s: [store[f"{k}|system={s}"]["mean_macro_f1"] for k in keys]
            for s in SYSTEMS}


def spread(store, keys):
    return {s: [store[f"{k}|system={s}"]["sd"] for k in keys] for s in SYSTEMS}


def line_panel(ax, keys, ticks, store, title, xlabel):
    xs = np.arange(len(keys))
    vals, sds = series(store, keys), spread(store, keys)
    for s in SYSTEMS:
        ax.fill_between(xs, np.array(vals[s]) - np.array(sds[s]),
                        np.array(vals[s]) + np.array(sds[s]),
                        color=COLOR[s], alpha=0.13, lw=0, zorder=2)
        ax.plot(xs, vals[s], color=COLOR[s], marker=MARK[s], ms=4.0, lw=1.6,
                markeredgecolor="white", markeredgewidth=0.7,
                label=NICE[s], zorder=3)
    ax.set_xticks(xs)
    ax.set_xticklabels(ticks)
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left", color=INK)
    ax.grid(color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def style():
    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
        "xtick.labelsize": 7.2, "ytick.labelsize": 7.2, "legend.fontsize": 6.4,
        "axes.edgecolor": MUTED, "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
    })


def save(fig, name):
    for out in OUTS:
        if out.parent.exists():
            out.mkdir(parents=True, exist_ok=True)
            fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
            print(f"wrote {out / (name + '.pdf')}")
    png = ROOT / "experiments" / f"{name}.png"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    print(f"wrote {png}")


def poison_panel(ax, pois, frac, tag):
    """Fusion only, mean against coordinate-wise median, at the worst fraction."""
    attacks = ["clean", "label_flip", "sign_flip", "gaussian"]
    ticks = ["none", "label\nflip", "sign\nflip", "Gaussian\nnoise"]
    xs = np.arange(len(attacks))
    w = 0.36
    for j, agg in enumerate(("mean", "median")):
        recs = [pois[(f"clean|agg={agg}" if a == "clean"
                      else f"{a}@{frac}|agg={agg}") + "|system=fusion"]
                for a in attacks]
        vals = [r["mean_macro_f1"] for r in recs]
        errs = [r["sd"] for r in recs]
        ax.bar(xs + (j - 0.5) * w, vals, w * 0.92, yerr=errs, capsize=2,
               color=AGG_COLOR[agg], edgecolor="white", linewidth=0.8,
               error_kw={"lw": 0.8, "ecolor": MUTED},
               label=f"FedAvg {agg}", zorder=3)
        # clear the error-bar cap, otherwise the cap strikes through the label
        for x, v, e in zip(xs + (j - 0.5) * w, vals, errs):
            ax.text(x, v + e + 0.022, f"{v:.2f}", ha="center", fontsize=5.8,
                    color=MUTED, zorder=4)
    ax.set_xticks(xs)
    ax.set_xticklabels(ticks)
    ax.set_xlabel(f"Attack, {100 * frac:.0f}% of clients malicious")
    ax.set_title(f"{tag} Poisoned updates (fusion)", loc="left", color=INK)
    ax.set_ylim(0, max(0.85, ax.get_ylim()[1]))
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, loc="upper right", ncol=1)


def main() -> None:
    d = json.load(open(SRC, encoding="utf-8"))
    het, drop, strag, pois = (d["heterogeneity"], d["dropout"],
                              d["stragglers"], d["poisoning"])
    style()

    alphas = sorted({float(k.split("|")[0].split("=")[1])
                     for k in het if "pathological" not in k}, reverse=True)
    h_keys = [f"alpha={a}" for a in alphas] + ["alpha=pathological"]
    h_ticks = [f"{a:g}" for a in alphas] + ["1 cls"]
    h_x = r"Dirichlet $\alpha$ (skew increases $\rightarrow$)"
    d_keys = sorted({k.split("|")[0] for k in drop},
                    key=lambda s: float(s.split("=")[1]))
    d_ticks = [f"{100 * float(k.split('=')[1]):.0f}" for k in d_keys]
    d_x = "Clients failing to report per round (%)"
    s_keys = sorted({k.split("|")[0] for k in strag},
                    key=lambda s: float(s.split("=")[1]))
    s_ticks = [f"{100 * float(k.split('=')[1]):.0f}" for k in s_keys]
    s_x = "Clients reporting a stale update per round (%)"
    frac = max(float(k.split("@")[1].split("|")[0]) for k in pois if "@" in k)

    # --- figure 90: skew and dropout -------------------------------------
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.55))
    line_panel(a1, h_keys, h_ticks, het, "(a) Severe label skew", h_x)
    a1.set_ylabel("Federated macro-F1")
    a1.legend(frameon=False, loc="lower left", ncol=1)
    line_panel(a2, d_keys, d_ticks, drop, "(b) Client dropout", d_x)
    a2.legend(frameon=False, loc="lower left", ncol=1)
    fig.tight_layout(pad=0.6)
    save(fig, "plot90_robustness_skew_dropout")
    plt.close(fig)

    # --- figure 91: the three modes worth plotting, one row ---------------
    # Three panels rather than four: client dropout is flat to 70% and a flat
    # line is the one result a sentence carries as well as a plot, so it stays
    # in the prose and the row keeps the two failures plus the skew ladder.
    fig, (b1, b2, b3) = plt.subplots(1, 3, figsize=(7.16, 2.05))
    line_panel(b1, h_keys, h_ticks, het, "(a) Severe label skew", h_x)
    b1.set_ylabel("Federated macro-F1")
    b1.legend(frameon=False, loc="lower left", ncol=1,
              handlelength=1.3, borderaxespad=0.2)
    line_panel(b2, s_keys, s_ticks, strag, "(b) Straggler staleness", s_x)
    poison_panel(b3, pois, frac, "(c)")
    for ax in (b1, b2, b3):
        ax.title.set_fontsize(7.6)
        ax.xaxis.label.set_fontsize(7.0)
    fig.tight_layout(pad=0.35, w_pad=0.9)
    save(fig, "plot91_robustness_stragglers_poison")
    plt.close(fig)

    # --- figure 92: all four modes in one two-column float ----------------
    # The page budget is 8 pages; this is the version that fits when the two
    # split figures do not.
    fig, ax = plt.subplots(2, 2, figsize=(7.0, 4.5))
    line_panel(ax[0][0], h_keys, h_ticks, het, "(a) Severe label skew", h_x)
    ax[0][0].set_ylabel("Federated macro-F1")
    ax[0][0].legend(frameon=False, loc="lower left", ncol=1)
    line_panel(ax[0][1], d_keys, d_ticks, drop, "(b) Client dropout", d_x)
    line_panel(ax[1][0], s_keys, s_ticks, strag, "(c) Straggler staleness", s_x)
    ax[1][0].set_ylabel("Federated macro-F1")
    poison_panel(ax[1][1], pois, frac, "(d)")
    fig.tight_layout(pad=0.6)
    save(fig, "plot92_robustness_all")
    plt.close(fig)

    # --- digest ----------------------------------------------------------
    print("\nheadline numbers")
    for s in SYSTEMS:
        base = het[f"alpha={alphas[0]}|system={s}"]["mean_macro_f1"]
        worst = het["alpha=pathological|system=" + s]["mean_macro_f1"]
        print(f"  {s:<8} alpha={alphas[0]:g}: {base:.3f} -> one class/client: "
              f"{worst:.3f}  ({worst - base:+.3f})")
    for s in SYSTEMS:
        c = pois[f"clean|agg=mean|system={s}"]["mean_macro_f1"]
        p = pois[f"label_flip@{frac}|agg=mean|system={s}"]["mean_macro_f1"]
        m = pois[f"label_flip@{frac}|agg=median|system={s}"]["mean_macro_f1"]
        print(f"  {s:<8} clean {c:.3f} | label-flip mean {p:.3f} | "
              f"median {m:.3f}")


if __name__ == "__main__":
    main()
