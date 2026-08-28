#!/usr/bin/env python3
"""Generate the supplementary figure suite from verified result JSON only.

Every panel here comes from a run that finished and whose aggregates are already
quoted in the paper. Nothing is drawn from the superseded pipelines whose plots
still sit in overleaf_final/plots (the rag_* series, the old 5-class stress
charts); those measure a different corpus and cannot be shown beside the tea
results.

Each figure is written independently so the paper can include any subset. Colour
follows the Okabe-Ito categorical theme used elsewhere in the repo, assigned by
entity in fixed order rather than by rank.
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
LOCAL = ROOT / "experiments" / "figure_suite"

ALIGNED = ROOT / "tea_results" / "aligned_support_tables" / "aligned_support_tables.json"
CLUBBED = ROOT / "tea_results" / "clubbed_tables" / "clubbed_tables.json"
FARM = ROOT / "experiments" / "farm_results_full_genuine.json"
SYSTEMS = ROOT / "tea_results" / "federated_all_systems_full" / "federated_all_systems.json"
ADVISORY = ROOT / "tea_results" / "advisory_retrieval_v1" / "advisory_retrieval_results.json"

C = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
INK, MUTED, GRID = "#1a1a1a", "#6b6b6b", "#d9d9d9"
RC = {"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
      "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
      "axes.edgecolor": MUTED, "text.color": INK, "axes.labelcolor": INK,
      "xtick.color": MUTED, "ytick.color": MUTED}
MADE: list[str] = []


def finish(fig, name):
    fig.tight_layout(pad=0.6)
    LOCAL.mkdir(parents=True, exist_ok=True)
    fig.savefig(LOCAL / f"{name}.png", dpi=200, bbox_inches="tight")
    for out in OUTS:
        if out.parent.exists():
            out.mkdir(parents=True, exist_ok=True)
            fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    MADE.append(name)


def tidy(ax, grid_axis="y"):
    ax.grid(axis=grid_axis, color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


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
    """Federated rows carry f1_macro; local-only rows carry mean_f1."""
    return rec.get("f1_macro", rec.get("mean_f1"))


# ---------------------------------------------------------------------------
def fig_aligned_encoders():
    """Validation-vs-test for every encoder on the aligned support."""
    d = json.load(open(ALIGNED, encoding="utf-8"))
    # each panel lists different systems, so the y axis must NOT be shared:
    # sharing makes the last panel's tick labels overwrite the others
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5))
    for ax, (sec, title) in zip(axes, (("text_only", "Text only"),
                                       ("image_only", "Image only"),
                                       ("multimodal", "Image $+$ text"))):
        rows = d[sec]
        names = sorted(rows, key=lambda n: rows[n]["test_accuracy"])
        y = np.arange(len(names))
        ax.barh(y - 0.19, [rows[n]["validation_accuracy"] for n in names], 0.36,
                color=C[0], edgecolor="white", lw=0.6, label="Validation", zorder=3)
        ax.barh(y + 0.19, [rows[n]["test_accuracy"] for n in names], 0.36,
                color=C[1], edgecolor="white", lw=0.6, label="Test", zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels([n.replace("ViT-Base + ", "+") for n in names], fontsize=6.4)
        ax.set_xlim(0, 1.05)
        ax.set_title(title, loc="left", color=INK)
        tidy(ax, "x")
    axes[0].set_xlabel("Accuracy")
    axes[1].set_xlabel("Accuracy")
    axes[2].set_xlabel("Accuracy")
    axes[1].legend(frameon=False, loc="lower right")
    finish(fig, "sfig01_aligned_encoders")


def fig_generalisation_gap():
    """Validation minus test on the aligned support: who overfits selection."""
    d = json.load(open(ALIGNED, encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(3.45, 2.7))
    labels, gaps, colours = [], [], []
    for i, sec in enumerate(("text_only", "image_only", "multimodal")):
        for n, r in sorted(d[sec].items(), key=lambda kv: kv[1]["validation_accuracy"]):
            labels.append(n.replace("ViT-Base + ", "+"))
            gaps.append(r["validation_accuracy"] - r["test_accuracy"])
            colours.append(C[i])
    y = np.arange(len(labels))
    ax.barh(y, gaps, 0.72, color=colours, edgecolor="white", lw=0.6, zorder=3)
    ax.axvline(0, color=MUTED, lw=0.9, zorder=4)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=6.2)
    ax.invert_yaxis()
    ax.set_xlabel("Validation $-$ test accuracy")
    ax.set_title("Selection gap by system", loc="left", color=INK)
    tidy(ax, "x")
    finish(fig, "sfig02_generalisation_gap")


def fig_clubbed_family(family, name, title):
    """One family's encoders as notes are deleted, centralized vs federated."""
    d = json.load(open(CLUBBED, encoding="utf-8"))
    t = d["tables"]
    keys = sorted(t, key=lambda k: float(k.split("=")[1]))
    xs = [100 * (1 - float(k.split("=")[1])) for k in keys]
    fig, ax = plt.subplots(figsize=(3.45, 2.7))
    for i, sysname in enumerate(t[keys[0]][family]):
        cen = [t[k][family][sysname]["central_test_accuracy"] for k in keys]
        fed = [t[k][family][sysname]["fed_test_accuracy"] for k in keys]
        ax.plot(xs, cen, color=C[i % len(C)], marker="o", ms=3.6, lw=1.5,
                label=sysname.replace("ViT-Base + ", "+"), zorder=3)
        ax.plot(xs, fed, color=C[i % len(C)], marker="o", ms=3.0, lw=1.1,
                ls=(0, (3, 2)), markerfacecolor="white", alpha=0.9, zorder=3)
    ax.invert_xaxis()
    ax.set_xticks(xs)
    ax.set_xlabel("Field notes retained (%)")
    ax.set_ylabel("Test accuracy")
    ax.set_ylim(0.3, 1.06)
    ax.set_title(title, loc="left", color=INK)
    tidy(ax, "both")
    ax.legend(frameon=False, loc="lower left", fontsize=5.8, ncol=1)
    finish(fig, name)


def fig_alpha_sweep():
    """Every architecture across the four skews, both splitters."""
    d = json.load(open(FARM, encoding="utf-8"))
    g = defaultdict(list)
    for k, v in d["E1_alpha_sweep"].items():
        f = parse(k)
        g[(f["arch"], f["split"], float(f["alpha"]))].append(score(v))
    archs = sorted({a for a, _, _ in g})
    alphas = sorted({al for _, _, al in g})
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
    for ax, split in zip(axes, ("corrected", "legacy")):
        for i, a in enumerate(archs):
            vals = [st.mean(g[(a, split, al)]) for al in alphas]
            ax.plot(range(len(alphas)), vals, color=C[i % len(C)], marker="o",
                    ms=3.8, lw=1.5, label=a.replace("_", " "), zorder=3)
        ax.set_xticks(range(len(alphas)))
        ax.set_xticklabels([f"{a:g}" for a in alphas])
        ax.set_xlabel(r"Dirichlet $\alpha$")
        ax.set_ylim(0, 1.08)
        ax.set_title(f"{split} splitter", loc="left", color=INK)
        tidy(ax, "both")
    axes[0].set_ylabel("Validation macro F1")
    axes[0].legend(frameon=False, loc="lower right", fontsize=6.2)
    finish(fig, "sfig05_alpha_sweep_splitters")


def fig_cost_pareto():
    """Communication cost against accuracy: what the uplink buys."""
    d = json.load(open(FARM, encoding="utf-8"))
    cost = {k.split("|", 1)[1]: v["upload_mib_per_client_per_round"]
            for k, v in d["E6_cost"].items() if k.startswith("arch_cost|")}
    acc = defaultdict(list)
    for k, v in d["E8_baselines"].items():
        f = parse(k)
        if f["_tag"] == "fedavg" and f["alpha"] == "1.0":
            acc[f["arch"]].append(score(v))
    fig, ax = plt.subplots(figsize=(3.45, 2.7))
    for i, a in enumerate(sorted(cost)):
        if a not in acc:
            continue
        ax.scatter(cost[a], st.mean(acc[a]), s=54, color=C[i % len(C)],
                   edgecolors="white", linewidths=0.8, zorder=3,
                   label=a.replace("_", " "))
    ax.set_xlabel("Uplink per client-round (MiB)")
    ax.set_ylabel(r"Validation macro F1 ($\alpha=1$)")
    ax.set_ylim(0, 1.08)
    ax.set_title("Cost against accuracy", loc="left", color=INK)
    tidy(ax, "both")
    ax.legend(frameon=False, loc="lower right", fontsize=6.2)
    finish(fig, "sfig06_cost_pareto")


def fig_warm_cold_and_fusion():
    """Warm vs cold start, and fusion variants across seeds."""
    d = json.load(open(FARM, encoding="utf-8"))
    warm = defaultdict(list)
    for k, v in d["E4_warmstart"].items():
        warm[parse(k)["_tag"]].append(score(v))
    fus = defaultdict(list)
    for k, v in d["E5_fusion_seeds"].items():
        fus[parse(k)["_tag"]].append(score(v))
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.0, 2.6))
    for i, (k, vals) in enumerate(sorted(warm.items())):
        ax_a.scatter([i] * len(vals), vals, s=46, color=C[i], zorder=3,
                     edgecolors="white", linewidths=0.7)
        ax_a.plot([i - 0.2, i + 0.2], [st.mean(vals)] * 2, color=INK, lw=1.2, zorder=4)
    ax_a.set_xticks(range(len(warm)))
    ax_a.set_xticklabels(sorted(warm))
    ax_a.set_ylabel("Validation macro F1")
    ax_a.set_ylim(0, 1.08)
    ax_a.set_title("(a) Warm vs cold start (per seed)", loc="left", color=INK)
    tidy(ax_a)
    for i, (k, vals) in enumerate(sorted(fus.items())):
        ax_b.scatter([i] * len(vals), vals, s=46, color=C[i], zorder=3,
                     edgecolors="white", linewidths=0.7)
        ax_b.plot([i - 0.2, i + 0.2], [st.mean(vals)] * 2, color=INK, lw=1.2, zorder=4)
    ax_b.set_xticks(range(len(fus)))
    ax_b.set_xticklabels(sorted(fus))
    ax_b.set_ylim(0, 1.08)
    ax_b.set_title("(b) Fusion variant (per seed)", loc="left", color=INK)
    tidy(ax_b)
    finish(fig, "sfig07_warmstart_fusion_seeds")


def fig_anticollapse():
    """The anti-collapse ablation, reported for the concat model."""
    d = json.load(open(FARM, encoding="utf-8"))
    g = defaultdict(list)
    for k, v in d["E3_anticollapse"].items():
        f = parse(k)
        g[(f["_tag"], f["alpha"])].append(score(v))
    arms = sorted({a for a, _ in g})
    alphas = sorted({al for _, al in g})
    fig, ax = plt.subplots(figsize=(3.45, 2.7))
    w = 0.8 / len(alphas)
    for j, al in enumerate(alphas):
        ax.bar(np.arange(len(arms)) + (j - (len(alphas) - 1) / 2) * w,
               [st.mean(g[(a, al)]) for a in arms], w * 0.9, color=C[j],
               edgecolor="white", lw=0.7, label=rf"$\alpha={al}$", zorder=3)
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([a.replace("_", " ") for a in arms], rotation=18, ha="right")
    ax.set_ylabel("Validation macro F1")
    ax.set_ylim(0, 1.08)
    ax.set_title("Anti-collapse components", loc="left", color=INK)
    tidy(ax)
    ax.legend(frameon=False, loc="lower right")
    finish(fig, "sfig08_anticollapse")


def fig_round_curves_by_skew():
    """Per-round convergence at both skews for the frozen-encoder systems."""
    d = json.load(open(SYSTEMS, encoding="utf-8"))
    curves = defaultdict(list)
    for r in d["runs"].values():
        curves[(r["system"], r["alpha"])].append(r["round_macro_f1"])
    systems = sorted({s for s, _ in curves})
    alphas = sorted({a for _, a in curves})
    fig, axes = plt.subplots(1, len(alphas), figsize=(7.0, 2.6), sharey=True)
    for ax, al in zip(np.atleast_1d(axes), alphas):
        for i, sname in enumerate(systems):
            runs = curves[(sname, al)]
            n = min(len(r) for r in runs)
            mean = [st.mean([r[j] for r in runs]) for j in range(n)]
            lo = [min(r[j] for r in runs) for j in range(n)]
            hi = [max(r[j] for r in runs) for j in range(n)]
            ax.fill_between(range(1, n + 1), lo, hi, color=C[i], alpha=0.16, lw=0)
            ax.plot(range(1, n + 1), mean, color=C[i], lw=1.6,
                    label=sname.replace("_", " "), zorder=3)
        ax.set_xlabel("Federated round")
        ax.set_ylim(0, 1.05)
        ax.set_title(rf"$\alpha={al:g}$", loc="left", color=INK)
        tidy(ax, "both")
    np.atleast_1d(axes)[0].set_ylabel("Validation macro F1")
    np.atleast_1d(axes)[0].legend(frameon=False, loc="lower right", fontsize=6.2)
    finish(fig, "sfig09_round_curves_by_skew")


def fig_corpus_audit():
    """What the advisory corpus is made of, and the class balance."""
    a = json.load(open(ADVISORY, encoding="utf-8"))
    audit = a["corpus_audit"]
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.0, 2.5))
    bars = [("annotation rows", audit["annotation_rows"]),
            ("unique obs.", audit["unique_observations"]),
            ("duplicates removed", audit["duplicate_rows_removed"]),
            ("distinct sentences", audit["distinct_sentences"]),
            ("shared across classes", audit["sentences_shared_across_classes"])]
    ax_a.barh(np.arange(len(bars)), [b[1] for b in bars], 0.66, color=C[0],
              edgecolor="white", lw=0.6, zorder=3)
    for i, (_, v) in enumerate(bars):
        ax_a.text(v, i, f" {v}", va="center", fontsize=6.6, color=INK)
    ax_a.set_yticks(np.arange(len(bars)))
    ax_a.set_yticklabels([b[0] for b in bars], fontsize=6.8)
    ax_a.invert_yaxis()
    ax_a.set_xscale("log")
    ax_a.set_xlabel("Count (log)")
    ax_a.set_title("(a) Advisory corpus audit", loc="left", color=INK)
    tidy(ax_a, "x")
    counts = audit["class_counts_unique"]
    order = a["class_order"]
    ax_b.bar(range(len(order)), [counts[c] for c in order], 0.66,
             color=C[2], edgecolor="white", lw=0.6, zorder=3)
    ax_b.set_xticks(range(len(order)))
    ax_b.set_xticklabels([c.replace("_", " ").title() for c in order],
                         rotation=18, ha="right", fontsize=6.6)
    ax_b.set_ylabel("Unique observations")
    ax_b.set_title("(b) Class balance", loc="left", color=INK)
    tidy(ax_b)
    finish(fig, "sfig10_corpus_audit")


def main() -> None:
    plt.rcParams.update(RC)
    fig_aligned_encoders()
    fig_generalisation_gap()
    fig_clubbed_family("text_only", "sfig03_clubbed_text",
                       "Text encoders as notes are deleted")
    fig_clubbed_family("multimodal", "sfig04_clubbed_multimodal",
                       "Fusions as notes are deleted")
    fig_alpha_sweep()
    fig_cost_pareto()
    fig_warm_cold_and_fusion()
    fig_anticollapse()
    fig_round_curves_by_skew()
    fig_corpus_audit()
    print(f"generated {len(MADE)} figures into {LOCAL} (and both plots/ dirs):")
    for n in MADE:
        print("   ", n)


if __name__ == "__main__":
    main()
