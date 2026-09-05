#!/usr/bin/env python3
"""Compact single-panel figures, sized to sit two-per-column in IEEE format.

The paper previously carried a few large multi-panel figures. Each of those
bundles several claims into one float, which forces the float to be big and
makes the caption carry three sentences. Splitting them into single-panel
figures at roughly 3.4x1.85in lets many more results appear in the same eight
pages, and each caption then states exactly one thing.

Everything is drawn from completed runs whose aggregates the paper quotes.
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
OUTS = [ROOT / "overleaf_final" / "plots", ROOT / "overleaf_final_slim" / "plots"]
LOCAL = ROOT / "experiments" / "compact"

ALIGNED = ROOT / "tea_results" / "aligned_support_tables" / "aligned_support_tables.json"
CLUBBED = ROOT / "tea_results" / "clubbed_tables" / "clubbed_tables.json"
TEXTL = ROOT / "tea_results" / "text_pipeline_ladder" / "text_pipeline_ladder.json"
MML = ROOT / "tea_results" / "multimodal_pipeline_ladder" / "multimodal_pipeline_ladder.json"
VISUAL = ROOT / "tea_results" / "visual_ladder" / "visual_ladder.json"
FARM = ROOT / "experiments" / "farm_results_full_genuine.json"
SYSTEMS = ROOT / "tea_results" / "federated_all_systems_full" / "federated_all_systems.json"
CORE = ROOT / "tea_results" / "federated_adaptation_v6" / "federated_adaptation_results.json"
ADVISORY = ROOT / "tea_results" / "advisory_retrieval_v1" / "advisory_retrieval_results.json"

C = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
INK, MUTED, GRID = "#1a1a1a", "#6b6b6b", "#d9d9d9"
SIZE = (3.4, 1.08)
RC = {"font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7.5,
      "xtick.labelsize": 5.8, "ytick.labelsize": 5.8, "legend.fontsize": 5.0,
      "axes.edgecolor": MUTED, "text.color": INK, "axes.labelcolor": INK,
      "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlepad": 3.0}
MADE: list[str] = []


def new():
    return plt.subplots(figsize=SIZE)


def finish(fig, ax, name, grid="y"):
    ax.grid(axis=grid, color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(pad=0.35)
    LOCAL.mkdir(parents=True, exist_ok=True)
    fig.savefig(LOCAL / f"{name}.png", dpi=200, bbox_inches="tight")
    for out in OUTS:
        if out.parent.exists():
            fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    MADE.append(name)


def parse(key):
    out = {}
    for part in key.split("|"):
        if "=" in part:
            a, b = part.split("=", 1)
            out[a] = b
        else:
            out["_tag"] = part
    return out


def score(rec):
    return rec.get("f1_macro", rec.get("mean_f1"))


def sparsity_axis(path):
    d = json.load(open(path, encoding="utf-8"))
    keys = sorted(d["ladder"], key=lambda k: float(k.split("=")[1]))
    return d, keys, [100 * (1 - float(k.split("=")[1])) for k in keys]


# --- ladders ---------------------------------------------------------------
def c01_text_ladder():
    d, keys, xs = sparsity_axis(TEXTL)
    fig, ax = new()
    for i, n in enumerate(d["ladder"][keys[0]]):
        ax.plot(xs, [d["ladder"][k][n]["test_accuracy"] for k in keys],
                color=C[i % len(C)], marker="o", ms=2.8, lw=1.2,
                ls=(0, (3, 2)) if "gate" in n.lower() else "-",
                label=n.replace(" (not single model)", "*"), zorder=3)
    ax.invert_xaxis(); ax.set_xticks(xs)
    ax.set_xlabel("Notes retained (%)"); ax.set_ylabel("Test acc.")
    ax.set_ylim(0.3, 1.05); ax.legend(frameon=False, loc="lower left", ncol=1)
    finish(fig, ax, "c01_text_ladder", "both")


def c02_fusion_ladder():
    d, keys, xs = sparsity_axis(MML)
    fig, ax = new()
    for i, n in enumerate(d["ladder"][keys[0]]):
        ax.plot(xs, [d["ladder"][k][n]["test_accuracy"] for k in keys],
                color=C[i % len(C)], marker="o", ms=2.8, lw=1.2,
                ls=(0, (3, 2)) if "gate" in n.lower() else "-",
                label=n.replace(" (not single model)", "*"), zorder=3)
    ax.invert_xaxis(); ax.set_xticks(xs)
    ax.set_xlabel("Notes retained (%)"); ax.set_ylabel("Test acc.")
    ax.set_ylim(0.3, 1.05); ax.legend(frameon=False, loc="lower left", ncol=1)
    finish(fig, ax, "c02_fusion_ladder", "both")


def c03_visual_search():
    """The image-only ladder.

    The rungs are alternative methods, not stages of one progression, so they
    are not joined by a line: that would imply a trajectory a system moves
    along. They are also not points on a log axis, which is unreadable at this
    size -- distinct systems compared by magnitude are bars. Search size is kept
    as an annotation so the axis it used to occupy is not lost.
    """
    rungs = json.load(open(VISUAL, encoding="utf-8"))["rungs"]
    rungs = sorted(rungs, key=lambda r: r["test_accuracy"])
    fig, ax = new()
    y = np.arange(len(rungs))
    # one bar per rung, styled by kind: overlaying a hatch on a filled bar
    # renders the hatch in the edge colour and reads as black at this size
    for i, r in enumerate(rungs):
        gate = r["is_gate"]
        ax.barh(i, r["test_accuracy"], 0.72,
                facecolor="white" if gate else C[1],
                hatch="///" if gate else None,
                edgecolor=C[1] if gate else "white", lw=0.7, zorder=3)
        ax.text(r["test_accuracy"] + 0.006, i, f"{r['candidates']}",
                va="center", fontsize=4.8, color=MUTED)
    ax.set_yticks(y)
    ax.set_yticklabels([r["name"] for r in rungs], fontsize=4.8)
    ax.set_xlim(0, 0.86)
    ax.set_xlabel("Test accuracy  (label: configurations searched)")
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=C[1], edgecolor="white"),
               plt.Rectangle((0, 0), 1, 1, facecolor="white", hatch="///",
                             edgecolor=C[1])]
    ax.legend(handles, ["Single model", "Per-class gate"], frameon=True,
              facecolor="white", edgecolor="none", framealpha=0.92,
              loc="lower right", ncol=1, handlelength=1.2)
    finish(fig, ax, "c03_visual_search", "x")


def c04_selection_gap():
    d = json.load(open(ALIGNED, encoding="utf-8"))
    labels, gaps, cols = [], [], []
    for i, sec in enumerate(("text_only", "image_only", "multimodal")):
        for n, r in sorted(d[sec].items(), key=lambda kv: kv[1]["validation_accuracy"]):
            labels.append(n.replace("ViT-Base + ", "+"))
            gaps.append(r["validation_accuracy"] - r["test_accuracy"])
            cols.append(C[i])
    # 15 categorical rows cannot fit the default compact height: at 1.08in the
    # tick labels overlap into an unreadable block. This one figure is taller.
    fig, ax = plt.subplots(figsize=(3.4, 2.15))
    ax.barh(np.arange(len(labels)), gaps, 0.74, color=cols,
            edgecolor="white", lw=0.5, zorder=3)
    ax.axvline(0, color=MUTED, lw=0.8, zorder=4)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=5.2)
    ax.set_ylim(len(labels) - 0.4, -0.6)
    ax.set_xlabel("Validation $-$ test acc.")
    finish(fig, ax, "c04_selection_gap", "x")


# --- aligned support -------------------------------------------------------
def _aligned(sec, name, xlim=1.05):
    d = json.load(open(ALIGNED, encoding="utf-8"))[sec]
    names = sorted(d, key=lambda n: d[n]["test_accuracy"])
    y = np.arange(len(names))
    fig, ax = new()
    ax.barh(y - 0.19, [d[n]["validation_accuracy"] for n in names], 0.36,
            color=C[0], edgecolor="white", lw=0.5, label="Val.", zorder=3)
    ax.barh(y + 0.19, [d[n]["test_accuracy"] for n in names], 0.36,
            color=C[1], edgecolor="white", lw=0.5, label="Test", zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([n.replace("ViT-Base + ", "+") for n in names], fontsize=5.6)
    ax.set_xlim(0, xlim); ax.set_xlabel("Accuracy")
    ax.legend(frameon=True, facecolor="white", edgecolor="none",
              framealpha=0.92, loc="lower right", ncol=2)
    finish(fig, ax, name, "x")


def c05_aligned_text():
    _aligned("text_only", "c05_aligned_text")


def c06_aligned_image():
    _aligned("image_only", "c06_aligned_image")


def c07_aligned_fusion():
    _aligned("multimodal", "c07_aligned_fusion")


# --- clubbed ---------------------------------------------------------------
def _clubbed(family, name):
    d = json.load(open(CLUBBED, encoding="utf-8"))["tables"]
    keys = sorted(d, key=lambda k: float(k.split("=")[1]))
    xs = [100 * (1 - float(k.split("=")[1])) for k in keys]
    fig, ax = new()
    for i, sysname in enumerate(d[keys[0]][family]):
        ax.plot(xs, [d[k][family][sysname]["central_test_accuracy"] for k in keys],
                color=C[i % len(C)], marker="o", ms=2.6, lw=1.2,
                label=sysname.replace("ViT-Base + ", "+").replace(" + linear SVM", ""),
                zorder=3)
    ax.invert_xaxis(); ax.set_xticks(xs)
    ax.set_xlabel("Notes retained (%)"); ax.set_ylabel("Test acc.")
    ax.set_ylim(0.3, 1.05); ax.legend(frameon=False, loc="lower left", ncol=1)
    finish(fig, ax, name, "both")


def c08_clubbed_text():
    _clubbed("text_only", "c08_clubbed_text")


def c09_clubbed_fusion():
    _clubbed("multimodal", "c09_clubbed_fusion")


def c10_federation_cost():
    d = json.load(open(CLUBBED, encoding="utf-8"))["tables"]
    keys = sorted(d, key=lambda k: float(k.split("=")[1]))
    op = keys[1] if len(keys) > 1 else keys[0]
    labels, deltas, cols = [], [], []
    for i, fam in enumerate(("text_only", "image_only")):
        for n, r in d[op][fam].items():
            labels.append(n.replace(" + linear SVM", "").replace("Colour+HOG SVM", "Colour/HOG"))
            deltas.append(r["central_test_accuracy"] - r["fed_test_accuracy"])
            cols.append(C[i])
    fig, ax = new()
    ax.barh(np.arange(len(labels)), deltas, 0.74, color=cols,
            edgecolor="white", lw=0.5, zorder=3)
    ax.axvline(0, color=MUTED, lw=0.8, zorder=4)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=5.0)
    ax.invert_yaxis(); ax.set_xlabel("Centralized $-$ federated acc.")
    finish(fig, ax, "c10_federation_cost", "x")


# --- federated -------------------------------------------------------------
def _alpha_sweep(split, name):
    d = json.load(open(FARM, encoding="utf-8"))
    g = defaultdict(list)
    for k, v in d["E1_alpha_sweep"].items():
        f = parse(k)
        if f["split"] == split:
            g[(f["arch"], float(f["alpha"]))].append(score(v))
    archs = sorted({a for a, _ in g}); alphas = sorted({a for _, a in g})
    fig, ax = new()
    for i, a in enumerate(archs):
        ax.plot(range(len(alphas)), [st.mean(g[(a, al)]) for al in alphas],
                color=C[i % len(C)], marker="o", ms=2.8, lw=1.2,
                label=a.replace("_", " "), zorder=3)
    ax.set_xticks(range(len(alphas)))
    ax.set_xticklabels([f"{a:g}" for a in alphas])
    ax.set_xlabel(r"Dirichlet $\alpha$"); ax.set_ylabel("Val. macro F1")
    ax.set_ylim(0, 1.06); ax.legend(frameon=False, loc="lower right", ncol=1)
    finish(fig, ax, name, "both")


def c11_alpha_corrected():
    _alpha_sweep("corrected", "c11_alpha_corrected")


def c12_alpha_legacy():
    _alpha_sweep("legacy", "c12_alpha_legacy")


def c13_aggregators():
    d = json.load(open(FARM, encoding="utf-8"))
    g = defaultdict(list)
    for k, v in d["E8_baselines"].items():
        f = parse(k)
        if f["alpha"] == "0.1":
            g[(f["arch"], f["_tag"])].append(score(v))
    archs = ["image_only", "text_only", "concat_vlm", "attention_vlm", "cross_attention_vlm"]
    aggs = [("fedavg", "Avg"), ("fedprox", "Prox"), ("scaffold", "SCAF"), ("fedbn", "BN")]
    fig, ax = new()
    x = np.arange(len(archs)); w = 0.2
    for j, (key, lab) in enumerate(aggs):
        ax.bar(x + (j - 1.5) * w, [st.mean(g[(a, key)]) if g[(a, key)] else np.nan
                                   for a in archs], w * 0.9, color=C[j],
               edgecolor="white", lw=0.5, label=lab, zorder=3)
    for i, a in enumerate(archs):
        loc = g[(a, "local_only")]
        if loc:
            ax.plot([x[i] - 2 * w, x[i] + 2 * w], [st.mean(loc)] * 2, color=INK,
                    lw=0.8, ls=(0, (2, 1.4)), zorder=5)
    ax.set_xticks(x)
    ax.set_xticklabels(["Img", "Txt", "Concat", "Attn", "X-attn"])
    ax.set_ylabel("Val. macro F1"); ax.set_ylim(0, 1.12)
    ax.legend(frameon=True, facecolor="white", edgecolor="none",
              framealpha=0.92, loc="upper center", ncol=4, columnspacing=0.8)
    finish(fig, ax, "c13_aggregators")


SYSLABEL = {"image_vit_tiny": "Image (ViT-tiny)",
            "text_distilbert": "Text (DistilBERT)",
            "vit_distilbert": "ViT$+$DistilBERT"}


def syslabel(s):
    return SYSLABEL.get(s, s.replace("_", " "))


def c14_round_curves():
    d = json.load(open(SYSTEMS, encoding="utf-8"))
    curves = defaultdict(list)
    for r in d["runs"].values():
        if r["alpha"] == 1.0:
            curves[r["system"]].append(r["round_macro_f1"])
    fig, ax = new()
    for i, s in enumerate(sorted(curves)):
        runs = curves[s]; n = min(len(r) for r in runs)
        mean = [st.mean([r[j] for r in runs]) for j in range(n)]
        ax.plot(range(1, n + 1), mean, color=C[i], lw=1.3,
                label=syslabel(s), zorder=3)
    ax.set_xlabel("Federated round"); ax.set_ylabel("Val. macro F1")
    ax.set_ylim(0.30, 0.85)
    ax.legend(frameon=False, loc="lower right", ncol=3, columnspacing=0.9,
              handlelength=1.4, borderpad=0.0)
    finish(fig, ax, "c14_round_curves", "both")


def c15_client_scaling():
    d = json.load(open(SYSTEMS, encoding="utf-8"))
    g = defaultdict(list)
    for r in d["runs"].values():
        if r["alpha"] == 1.0:
            g[(r["system"], r["num_clients"])].append(r["fedavg_final_macro_f1"])
    ks = sorted({k for _, k in g}); pos = np.arange(len(ks))
    fig, ax = new()
    for i, s in enumerate(sorted({s for s, _ in g})):
        ax.plot(pos, [st.mean(g[(s, k)]) for k in ks], color=C[i], marker="o",
                ms=2.8, lw=1.3, label=syslabel(s), zorder=3)
    ax.set_xticks(pos); ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("Clients $K$"); ax.set_ylabel("Val. macro F1")
    ax.set_ylim(0.34, 0.90)
    ax.legend(frameon=False, loc="lower left", ncol=3, columnspacing=0.9,
              handlelength=1.4, borderpad=0.0)
    finish(fig, ax, "c15_client_scaling", "both")


# --- warm-start core scaling ------------------------------------------------
def _core():
    d = json.load(open(CORE, encoding="utf-8"))
    ks = sorted(d["summary"], key=int)
    base = d["runs"][0]["baseline_validation"]["f1_macro"]
    return d["summary"], ks, base


def c22_core_rounds():
    """Rounds against macro F1 for the retained core, one line per client count.

    The subsection quotes a round-wise band and one transient peak; a reader
    cannot see from the numbers alone that the K=2 and K=3 curves climb while
    K=5 and K=8 are flat from round 1. The dashed rule is the centralized
    initialization every run starts from, so the gap to it is the adaptation
    cost the text reports as a retention percentage.
    """
    summ, ks, base = _core()
    fig, ax = new()
    ax.axhline(base, color=INK, lw=0.8, ls=(0, (2, 1.4)), zorder=2)
    ax.text(8.0, base + 0.004, "centralized init 0.692", ha="right", va="bottom",
            fontsize=5.0, color=INK)
    for i, k in enumerate(ks):
        ys = summ[k]["round_mean_macro_f1"]
        ax.plot(range(1, len(ys) + 1), ys, color=C[i], marker="o", ms=2.4,
                lw=1.2, label=f"$K={k}$", zorder=3)
    ax.set_xlabel("Federated round"); ax.set_ylabel("Val. macro F1")
    ax.set_xticks(range(1, 9)); ax.set_ylim(0.52, 0.73)
    ax.legend(frameon=False, loc="lower left", ncol=4, columnspacing=0.9,
              handlelength=1.4, borderpad=0.0)
    finish(fig, ax, "c22_core_rounds", "both")


def c23_core_clients():
    """Client count against macro F1 for the retained core, final and best round.

    Splitting the same fixed training set over more owners is the scaling
    question the deployment story rests on, and it is the one axis the paper
    otherwise reports only as a range. Realized label total variation is printed
    under each tick rather than drawn on a second y-axis.
    """
    summ, ks, base = _core()
    x = np.arange(len(ks))
    fin = [summ[k]["final_macro_f1_mean"] for k in ks]
    fsd = [summ[k]["final_macro_f1_std"] for k in ks]
    bst = [summ[k]["best_macro_f1_mean"] for k in ks]
    bsd = [summ[k]["best_macro_f1_std"] for k in ks]
    fig, ax = new()
    ax.axhline(base, color=INK, lw=0.8, ls=(0, (2, 1.4)), zorder=2)
    ax.text(x[-1] + 0.06, base, "centralized init", ha="right", va="bottom",
            fontsize=5.0, color=INK)
    ax.errorbar(x, bst, yerr=bsd, color=C[2], marker="s", ms=2.8, lw=1.2,
                ls=(0, (3, 1.5)), capsize=1.6, elinewidth=0.7,
                label="best round", zorder=3)
    ax.errorbar(x, fin, yerr=fsd, color=C[0], marker="o", ms=2.8, lw=1.3,
                capsize=1.6, elinewidth=0.7, label="final round", zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{k}\n{summ[k]['mean_label_total_variation']:.2f}"
                        for k in ks])
    ax.set_xlim(-0.25, len(ks) - 0.75)
    ax.set_xlabel("Clients $K$ / realized label TV")
    ax.set_ylabel("Val. macro F1"); ax.set_ylim(0.52, 0.76)
    ax.legend(frameon=False, loc="lower left", ncol=2, columnspacing=0.9,
              handlelength=1.6, borderpad=0.0)
    finish(fig, ax, "c23_core_clients", "both")


def c16_cost_pareto():
    """Uplink against accuracy as a cost-ordered line.

    Architectures are joined in ascending uplink so the shape of the trade-off
    is readable: the line is an ordering, not an interpolation, and the caption
    says so. Points stay direct-labelled -- five marks, two of them 1.0 MiB
    apart, are not separable by colour.
    """
    d = json.load(open(FARM, encoding="utf-8"))
    cost = {k.split("|", 1)[1]: v["upload_mib_per_client_per_round"]
            for k, v in d["E6_cost"].items() if k.startswith("arch_cost|")}
    acc = defaultdict(list)
    for k, v in d["E8_baselines"].items():
        f = parse(k)
        if f["_tag"] == "fedavg" and f["alpha"] == "0.1":
            acc[f["arch"]].append(score(v))
    short = {"image_only": ("Image", 0, 7, "center"),
             "text_only": ("Text", 0, 7, "center"),
             "concat_vlm": ("Concat", -4, -11, "right"),
             "attention_vlm": ("Attn", 2, 7, "left"),
             "cross_attention_vlm": ("X-attn", 0, 7, "center")}
    pts = sorted(((cost[a], st.mean(acc[a]), a) for a in short), key=lambda t: t[0])
    fig, ax = new()
    ax.plot([x for x, _, _ in pts], [y for _, y, _ in pts], color=MUTED, lw=1.0,
            zorder=2)
    for i, (x, y, a) in enumerate(pts):
        ax.plot(x, y, "o", ms=4.0, color=C[i % len(C)], mec="white", mew=0.6,
                zorder=4)
        lab, dx, dy, ha = short[a]
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=(dx, dy),
                    ha=ha, fontsize=5.6, color=INK, zorder=5)
    ax.set_xlabel("Uplink per client-round (MiB)")
    ax.set_ylabel("Val. macro F1")
    ax.set_xlim(12, 76); ax.set_ylim(0.43, 0.585)
    finish(fig, ax, "c16_cost_pareto", "both")


def c17_anticollapse():
    d = json.load(open(FARM, encoding="utf-8"))
    g = defaultdict(list)
    for k, v in d["E3_anticollapse"].items():
        f = parse(k)
        g[(f["_tag"], f["alpha"])].append(score(v))
    arms = sorted({a for a, _ in g}); alphas = sorted({al for _, al in g})
    fig, ax = new(); w = 0.8 / len(alphas)
    for j, al in enumerate(alphas):
        ax.bar(np.arange(len(arms)) + (j - (len(alphas) - 1) / 2) * w,
               [st.mean(g[(a, al)]) for a in arms], w * 0.9, color=C[j],
               edgecolor="white", lw=0.5, label=rf"$\alpha={al}$", zorder=3)
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([a.replace("_", " ") for a in arms], rotation=14,
                       ha="right", fontsize=5.8)
    ax.set_ylabel("Val. macro F1"); ax.set_ylim(0, 1.12)
    ax.legend(frameon=False, loc="lower right", ncol=2)
    finish(fig, ax, "c17_anticollapse")


def c18_warm_cold():
    d = json.load(open(FARM, encoding="utf-8"))
    g = defaultdict(list)
    for k, v in d["E4_warmstart"].items():
        g[parse(k)["_tag"]].append(score(v))
    fig, ax = new()
    for i, (k, vals) in enumerate(sorted(g.items())):
        ax.scatter([i] * len(vals), vals, s=30, color=C[i], zorder=3,
                   edgecolors="white", linewidths=0.5)
        ax.plot([i - 0.2, i + 0.2], [st.mean(vals)] * 2, color=INK, lw=1.0, zorder=4)
    ax.set_xticks(range(len(g))); ax.set_xticklabels(sorted(g))
    ax.set_xlim(-0.5, len(g) - 0.5)
    ax.set_ylabel("Val. macro F1"); ax.set_ylim(0, 1.08)
    finish(fig, ax, "c18_warm_cold")


# --- advisory --------------------------------------------------------------
# --- matched-suite arms the paper asserts but never shows -------------------
def c24_anticollapse_arms():
    """E3 arms as grouped bars from zero, with the values printed.

    Bars start at zero because the finding is that no arm collapses, and four
    near-equal bars say that honestly; a truncated axis would manufacture a
    difference. The 0.044 spread that a zero baseline hides is recoverable from
    the printed values.
    """
    d = json.load(open(FARM, encoding="utf-8"))
    g = {}
    for k, v in d["E3_anticollapse"].items():
        f = parse(k)
        g[(f["_tag"], f["alpha"])] = score(v)
    arms = ["both", "no_diversity", "no_balanced", "neither"]
    short = {"both": "Both", "no_diversity": "No div.",
             "no_balanced": "No bal.", "neither": "Neither"}
    alphas = ["0.1", "1.0"]
    x = np.arange(len(arms)); w = 0.38
    fig, ax = plt.subplots(figsize=(3.4, 0.85))
    for j, a in enumerate(alphas):
        vals = [g[(arm, a)] for arm in arms]
        ax.bar(x + (j - 0.5) * w, vals, w * 0.86, color=C[j], edgecolor="white",
               lw=0.5, label=rf"$\alpha={a}$", zorder=3)
        for xi, v in zip(x + (j - 0.5) * w, vals):
            ax.text(xi, v + 0.012, f"{v:.3f}", ha="center", va="bottom",
                    fontsize=4.4, color=INK, zorder=5)
    ax.set_xticks(x); ax.set_xticklabels([short[a] for a in arms])
    ax.set_ylabel("Val. macro F1"); ax.set_ylim(0, 0.78)
    ax.set_yticks([0, 0.25, 0.5])
    ax.legend(frameon=False, loc="upper right", ncol=2, columnspacing=0.8,
              handlelength=1.2, handletextpad=0.3, borderpad=0.0)
    finish(fig, ax, "c24_anticollapse_arms")


def c25_seed_floor():
    """Matched-setting arms as horizontal bars, whiskers spanning the seeds.

    Horizontal because four arm names read better along the axis than under it,
    and it keeps this figure from looking like the one above it. Bars are
    3-seed means from zero; the whisker is the observed seed range, and the
    ranges overlapping is the whole claim.
    """
    d = json.load(open(FARM, encoding="utf-8"))
    g = defaultdict(list)
    for k, v in d["E4_warmstart"].items():
        g[parse(k)["_tag"]].append(score(v))
    for k, v in d["E5_fusion_seeds"].items():
        tag = parse(k)["_tag"]
        if tag != "concat":          # identical runs to the cold arm
            g[tag].append(score(v))
    arms = [("warm", "Warm"), ("cold", "Cold (concat)"),
            ("attention", "Attention"), ("gated", "Gated")]
    y = np.arange(len(arms))[::-1]
    fig, ax = plt.subplots(figsize=(3.4, 0.85))
    for i, (key, _) in enumerate(arms):
        vals = g[key]; m = st.mean(vals)
        ax.barh(y[i], m, 0.62, color=C[i], edgecolor="white", lw=0.5, zorder=3)
        ax.plot([min(vals), max(vals)], [y[i]] * 2, color=INK, lw=0.8, zorder=5)
        ax.plot([min(vals), max(vals)], [y[i]] * 2, "|", ms=2.6, color=INK,
                mew=0.8, zorder=5)
        ax.text(max(vals) + 0.012, y[i], f"{m:.3f}", va="center", ha="left",
                fontsize=4.8, color=INK, zorder=5)
    ax.set_yticks(y); ax.set_yticklabels([lab for _, lab in arms])
    ax.set_xlabel("Val. macro F1"); ax.set_xlim(0, 0.68)
    ax.set_xticks([0, 0.2, 0.4, 0.6])
    finish(fig, ax, "c25_seed_floor", "x")


def c19_retrieval_box():
    a = json.load(open(ADVISORY, encoding="utf-8"))
    pretty = {"LEAF_BLIGHT": "Blight", "LEAF_HOPPERS": "Hoppers", "LEAF_RUST": "Rust",
              "LOOPER_CATERPILLARS": "Looper", "MOSQUITO_BUG": "Mosq."}
    hit, miss = defaultdict(list), defaultdict(list)
    for r in a["per_query"]:
        (hit if r["top1_correct"] else miss)[r["true_label"]].append(r["top1_score"])
    order = a["class_order"]; pos = np.arange(len(order))
    fig, ax = new()
    for data, colour, off, tag in ((hit, C[0], -0.17, "correct"),
                                   (miss, C[1], 0.17, "wrong")):
        vals = [data[c] for c in order]
        keep = [i for i, v in enumerate(vals) if v]
        bp = ax.boxplot([vals[i] for i in keep], positions=pos[keep] + off,
                        widths=0.28, patch_artist=True,
                        medianprops=dict(color=INK, lw=0.9),
                        flierprops=dict(marker="o", ms=1.8, markerfacecolor="none",
                                        markeredgecolor=colour, markeredgewidth=0.5),
                        whiskerprops=dict(color=colour, lw=0.7),
                        capprops=dict(color=colour, lw=0.7))
        for b in bp["boxes"]:
            b.set(facecolor=colour, alpha=0.5, edgecolor=colour, linewidth=0.7)
        ax.plot([], [], color=colour, lw=5, alpha=0.5, label=f"top-1 {tag}")
    ax.set_xticks(pos); ax.set_xticklabels([pretty[c] for c in order], fontsize=6.0)
    ax.set_ylabel("Top-1 similarity")
    ax.legend(frameon=True, facecolor="white", edgecolor="none",
              framealpha=0.92, loc="lower left", ncol=2)
    finish(fig, ax, "c19_retrieval_box")


def c20_advisory_perclass():
    a = json.load(open(ADVISORY, encoding="utf-8"))
    pretty = {"LEAF_BLIGHT": "Blight", "LEAF_HOPPERS": "Hoppers", "LEAF_RUST": "Rust",
              "LOOPER_CATERPILLARS": "Looper", "MOSQUITO_BUG": "Mosq."}
    order = a["class_order"]; per = a["per_class"]; pos = np.arange(len(order))
    fig, ax = new(); w = 0.36
    ax.bar(pos - w / 2, [per[c]["precision_at_1_open"] for c in order], w,
           color=C[0], edgecolor="white", lw=0.5, label="Retrieve", zorder=3)
    ax.bar(pos + w / 2, [per[c]["precision_at_1_routed"] for c in order], w,
           color=C[3], edgecolor="white", lw=0.5, label="Classify$+$retrieve", zorder=3)
    ax.axhline(a["chance_precision_at_1"], color=MUTED, lw=0.8, ls=(0, (3, 2)), zorder=4)
    ax.set_xticks(pos); ax.set_xticklabels([pretty[c] for c in order], fontsize=6.0)
    ax.set_ylabel("Precision@1"); ax.set_ylim(0, 1.14)
    ax.legend(frameon=True, facecolor="white", edgecolor="none",
              framealpha=0.92, loc="lower right", ncol=1)
    finish(fig, ax, "c20_advisory_perclass")


def c21_retrieval_heatmap():
    """RAG retrieval confusion: true class against top-1 retrieved class.

    Row-normalized so each row sums to 1 and the diagonal is per-class
    precision@1. Sequential single hue, light to dark, because the quantity is
    a magnitude with a meaningful zero -- never a rainbow. Counts are printed in
    each cell so a colour need not be decoded back into a number.
    """
    from collections import Counter
    d = json.load(open(ADVISORY, encoding="utf-8"))
    order = d["class_order"]
    short = {"LEAF_BLIGHT": "Blight", "LEAF_HOPPERS": "Hoppers", "LEAF_RUST": "Rust",
             "LOOPER_CATERPILLARS": "Looper", "MOSQUITO_BUG": "Mosq."}
    pairs = Counter((r["true_label"], r["top1_label"]) for r in d["per_query"])
    counts = np.array([[pairs.get((t, q), 0) for q in order] for t in order],
                      dtype=float)
    rates = counts / counts.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(3.4, 1.42))
    im = ax.imshow(rates, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")
    for i in range(len(order)):
        for j in range(len(order)):
            if counts[i, j]:
                ax.text(j, i, f"{int(counts[i, j])}", ha="center", va="center",
                        fontsize=5.0,
                        color="white" if rates[i, j] > 0.55 else INK)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([short[c] for c in order], fontsize=5.2, rotation=18,
                       ha="right")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([short[c] for c in order], fontsize=5.2)
    ax.set_xlabel("Top-1 retrieved class", fontsize=6.2)
    ax.set_ylabel("True class", fontsize=6.2)
    for sp in ("top", "right", "left", "bottom"):
        ax.spines[sp].set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.036, pad=0.02)
    cb.ax.tick_params(labelsize=4.8, color=MUTED)
    cb.outline.set_visible(False)
    cb.set_label("Share of queries", fontsize=5.2)
    fig.tight_layout(pad=0.35)
    LOCAL.mkdir(parents=True, exist_ok=True)
    fig.savefig(LOCAL / "c21_retrieval_heatmap.png", dpi=200, bbox_inches="tight")
    for out in OUTS:
        if out.parent.exists():
            fig.savefig(out / "c21_retrieval_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)
    MADE.append("c21_retrieval_heatmap")


def main() -> None:
    plt.rcParams.update(RC)
    for fn in (c01_text_ladder, c02_fusion_ladder, c03_visual_search, c04_selection_gap,
               c05_aligned_text, c06_aligned_image, c07_aligned_fusion,
               c08_clubbed_text, c09_clubbed_fusion, c10_federation_cost,
               c11_alpha_corrected, c12_alpha_legacy, c13_aggregators,
               c14_round_curves, c15_client_scaling,
               c22_core_rounds, c23_core_clients, c16_cost_pareto,
               c17_anticollapse, c18_warm_cold, c19_retrieval_box,
               c20_advisory_perclass, c21_retrieval_heatmap,
               c24_anticollapse_arms, c25_seed_floor):
        fn()
    print(f"generated {len(MADE)} compact figures in {LOCAL}")
    for n in MADE:
        print("   ", n)


if __name__ == "__main__":
    main()
