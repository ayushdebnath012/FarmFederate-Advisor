"""Regenerate Fig.10 (federated sweep) and Fig.11 (federated cost) at IEEE print size.

Rendered at the exact physical width each panel occupies in the paper, with fonts
sized for print, so axis labels and legends stay legible instead of being shrunk
into illegibility by \\includegraphics scaling.
"""
import json, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

R = "farmfederate_results_20260719_190712 new/"
D = json.load(open(R + "results/complete_results.json"))
FS = D["fed_sweep"]

# ---- palette (dataviz skill reference instance) -----------------------------
# categorical slots 1,4,8 - chosen by maximising min CVD dE (15.3 vs 7.6 for 1,2,3)
FAM = {"ViT": "#2a78d6", "LLM": "#eda100", "VLM": "#e34948"}
# ordinal blue ramp, steps 250/350/450/600 (lightest clears the 2:1 ordinal floor)
ORD = ["#86b6ef", "#5598e7", "#2a78d6", "#184f95"]
GRID, AXIS, MUTED, INK = "#e1e0d9", "#c3c2b7", "#898781", "#0b0b0b"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 7.0,
    "axes.titlesize": 7.5,
    "axes.labelsize": 7.0,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK,
    "ytick.labelcolor": INK,
    "grid.color": GRID,
    "grid.linewidth": 0.5,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.3,
    "lines.markersize": 3.4,
    "figure.dpi": 400,
    "savefig.dpi": 400,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.012,
})


def style(ax, xlabel, ylabel, title=None):
    ax.grid(True, which="major", axis="both", zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, pad=3)
    ax.tick_params(length=2, width=0.6, pad=1.5)


def save(fig, name):
    fig.savefig(f"plots/panels/{name}.png", facecolor="white")
    plt.close(fig)
    print("  wrote", name)


# =============================== FIG 10 ======================================
# Panel size: 0.325*textwidth(7.16in) ~= 2.33in wide
PW, PH = 2.33, 1.72

# (a) client count vs final-round macro F1
fig, ax = plt.subplots(figsize=(PW, PH))
ks = ["2", "3", "5", "10"]
x = np.arange(len(ks))
for fam, c in FAM.items():
    y = [FS[fam]["clients_rounds"][k]["final_f1_macro"] for k in ks]
    ax.plot(x, y, "-o", color=c, label=fam, zorder=3)
ax.set_xticks(x); ax.set_xticklabels(ks)
style(ax, "Number of clients $K$", "Final-round macro F1")
ax.legend(frameon=False, handlelength=1.4, borderpad=0.2, labelspacing=0.25, loc="best")
save(fig, "f10a_clients")

# (d) per-round convergence at the deployed K=3, all three families in one panel.
# One panel instead of three: the point is that *no* family converges cleanly, and
# overlaying them shows that far more compactly than three separate axes.
fig, ax = plt.subplots(figsize=(PW, PH))
for fam, c in FAM.items():
    h = FS[fam]["clients_rounds"]["3"]["history"]
    ax.plot(h["rounds"], h["val_f1_macro"], "-o", color=c, label=fam,
            zorder=3, markersize=2.8)
style(ax, "Federated round ($K$=3)", "Validation macro F1")
ax.legend(frameon=False, handlelength=1.4, borderpad=0.2, labelspacing=0.25, loc="best")
save(fig, "f10d_rounds")

# (e) local epochs
fig, ax = plt.subplots(figsize=(PW, PH))
es = ["1", "3", "5"]
for fam, c in FAM.items():
    ax.plot([int(e) for e in es], [FS[fam]["local_epochs"][e]["final_f1_macro"] for e in es],
            "-o", color=c, label=fam, zorder=3)
ax.set_xticks([1, 3, 5])
style(ax, "Local epochs $E$", "Final-round macro F1")
ax.legend(frameon=False, handlelength=1.4, borderpad=0.2, labelspacing=0.25, loc="best")
save(fig, "f10e_epochs")

# (f) Dirichlet alpha (log x - it spans 0.1..10)
fig, ax = plt.subplots(figsize=(PW, PH))
al = ["0.1", "0.5", "1.0", "10.0"]
for fam, c in FAM.items():
    ax.plot([float(a) for a in al], [FS[fam]["alpha"][a]["final_f1_macro"] for a in al],
            "-o", color=c, label=fam, zorder=3)
ax.set_xscale("log")
ax.set_xticks([0.1, 0.5, 1.0, 10.0]); ax.set_xticklabels(["0.1", "0.5", "1", "10"])
ax.minorticks_off()
style(ax, r"Dirichlet $\alpha$  (low = more skew)", "Final-round macro F1")
ax.legend(frameon=False, handlelength=1.4, borderpad=0.2, labelspacing=0.25, loc="best")
save(fig, "f10f_alpha")


# =============================== FIG 11 ======================================
# Panel size: 0.49*textwidth ~= 3.5in wide
cent, fed = {}, {}
for r in csv.DictReader(open(R + "results/four_way_federated_vs_centralized.csv")):
    (cent if r["training_mode"] == "centralized" else fed)[r["condition"]] = float(r["f1_macro"])
eff = {r["condition"]: r for r in csv.DictReader(open(R + "plots/federated_paired_effect.csv"))}

order = ["unimodal_image", "unimodal_text", "standard_vlm_fusion",
         "flexible_multimodal_vlm_matched_scratch"]
short = ["Image\nonly", "Text\nonly", "Concat\nfusion", "Proposed"]

# (a) centralized vs federated, grouped bars
fig, ax = plt.subplots(figsize=(3.5, 1.85))
x = np.arange(len(order)); w = 0.36
ax.bar(x - w/2, [cent[c] for c in order], w, label="Centralized",
       color="#2a78d6", edgecolor="white", linewidth=0.8, zorder=3)
ax.bar(x + w/2, [fed[c] for c in order], w, label="Federated (FedAvg)",
       color="#eda100", edgecolor="white", linewidth=0.8, zorder=3)
for xi, c in zip(x, order):
    # nudge the pair apart so near-equal values (text-only: 0.139 / 0.139) don't collide
    ax.text(xi - w/2 - 0.035, cent[c] + 0.012, f"{cent[c]:.3f}", ha="center", va="bottom",
            fontsize=5.6, color=INK)
    ax.text(xi + w/2 + 0.035, fed[c] + 0.012, f"{fed[c]:.3f}", ha="center", va="bottom",
            fontsize=5.6, color=INK)
ax.set_xticks(x); ax.set_xticklabels(short)
ax.set_ylim(0, 0.56)
style(ax, "", "Macro F1 on common test ($n$=30)")
ax.grid(axis="x", visible=False)
ax.legend(frameon=False, ncol=2, handlelength=1.2, borderpad=0.2,
          columnspacing=1.0, loc="upper center", bbox_to_anchor=(0.5, 1.16))
save(fig, "f11a_fedcent")

# (b) paired effect, forest plot with 95% CI
fig, ax = plt.subplots(figsize=(3.5, 1.85))
y = np.arange(len(order))[::-1]
for yi, c in zip(y, order):
    e = eff[c]
    lo, hi, est = float(e["ci_low"]), float(e["ci_high"]), float(e["fed_minus_central_macro_f1"])
    crosses = lo <= 0 <= hi
    col = "#898781" if crosses else "#e34948"
    ax.plot([lo, hi], [yi, yi], "-", color=col, linewidth=1.6, solid_capstyle="round", zorder=3)
    ax.plot([est], [yi], "o", color=col, markersize=4.2, zorder=4,
            markeredgecolor="white", markeredgewidth=0.7)
    ax.text(hi + 0.012, yi, f"{est:+.3f}", va="center", ha="left", fontsize=5.8, color=INK)
ax.axvline(0, color=AXIS, linewidth=0.8, zorder=2)
ax.set_yticks(y); ax.set_yticklabels([s.replace("\n", " ") for s in short])
ax.set_xlim(-0.42, 0.30)
# clear band below the last row so the legend never lands on a CI bar or its label
ax.set_ylim(-1.30, len(order) - 0.45)
style(ax, r"Federated $-$ centralized macro F1 (95% CI)", "")
ax.grid(axis="y", visible=False)
ax.legend(handles=[
    Line2D([], [], color="#898781", lw=1.6, marker="o", markersize=4.2,
           markeredgecolor="white", label="CI includes 0"),
    Line2D([], [], color="#e34948", lw=1.6, marker="o", markersize=4.2,
           markeredgecolor="white", label="CI excludes 0 (pre-Holm)"),
], frameon=False, handlelength=1.4, borderpad=0.15, labelspacing=0.18,
   loc="lower left", bbox_to_anchor=(0.0, 0.0), fontsize=5.8, ncol=2,
   columnspacing=1.1)
save(fig, "f11b_paired")
print("done")
