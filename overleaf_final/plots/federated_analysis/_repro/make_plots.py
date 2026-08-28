"""
Stage 3: Plots from the real federated sweep.

Reads sweep/sweep_results.json and writes PNGs to <repo>/plots/federated_analysis/.
Design: colourblind-safe Okabe-Ito palette, consistent per modality across every
figure, light background, high DPI.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

HERE = Path(__file__).parent
RES = json.load(open(HERE / "sweep" / "sweep_results.json"))
OUT = Path(r"c:\Users\USER_HP\Desktop\FarmFederate\plots\federated_analysis")
OUT.mkdir(parents=True, exist_ok=True)

# ---- style ---------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 150, "font.size": 11,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 12, "axes.titleweight": "bold",
    "figure.facecolor": "white", "axes.facecolor": "white",
    "font.family": "DejaVu Sans",
})
# modality colours (Okabe-Ito)
MC = {"text": "#0072B2", "image": "#D55E00", "text+image": "#009E73"}
ML = {"text": "Text-only (LLM)", "image": "Image-only (ViT)", "text+image": "Text+Image (VLM)"}
MODS = ["text", "image", "text+image"]
CLS_SHORT = {"water_stress": "Water", "nutrient_def": "Nutrient",
             "pest_risk": "Pest", "disease_risk": "Disease", "heat_stress": "Heat"}


def savefig(fig, name):
    fig.tight_layout()
    p = OUT / name
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("  wrote", p.name)


# ==========================================================================
# FIG 1: Convergence vs rounds, one line per #clients, 3 modality panels
# ==========================================================================
def fig_rounds_convergence():
    rc = RES["rounds_x_clients"]
    clients = sorted(rc["text"].keys(), key=int)
    cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(clients)))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
    for ax, mk in zip(axes, MODS):
        for c, col in zip(clients, cmap):
            mean = np.array(rc[mk][c]["mean"])
            std = np.array(rc[mk][c]["std"])
            x = np.arange(1, len(mean) + 1)
            ax.plot(x, mean, color=col, lw=2, label=f"{c}")
            ax.fill_between(x, mean - std, mean + std, color=col, alpha=0.12)
        ax.set_title(ML[mk], color=MC[mk])
        ax.set_xlabel("Communication round")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[0].set_ylabel("Macro-F1 (held-out test)")
    leg = axes[2].legend(title="# clients", fontsize=9, title_fontsize=9,
                         loc="lower right", ncol=2, framealpha=0.9)
    axes[1].set_ylim(0.84, 1.005)  # zoom: image is the only modality that moves
    fig.suptitle("Federated convergence: rounds × number of clients × modality "
                 "(FedAvg, α=1.0, 3 local epochs, mean±sd over 2 seeds)",
                 fontsize=12.5, fontweight="bold", y=1.02)
    savefig(fig, "fig01_rounds_x_clients_convergence.png")


# ==========================================================================
# FIG 2: Final & best F1 vs number of clients (scalability)
# ==========================================================================
def fig_clients_scalability():
    rc = RES["rounds_x_clients"]
    clients = sorted(rc["text"].keys(), key=int)
    x = np.arange(len(clients))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6))
    for mk in MODS:
        final = [rc[mk][c]["final"] for c in clients]
        best = [rc[mk][c]["best"] for c in clients]
        a1.plot(x, final, "-o", color=MC[mk], lw=2, ms=6, label=ML[mk])
        a2.plot(x, best, "-o", color=MC[mk], lw=2, ms=6, label=ML[mk])
    for a, t in ((a1, "Final-round F1"), (a2, "Best-round F1")):
        a.set_xticks(x); a.set_xticklabels(clients)
        a.set_xlabel("Number of federated clients (fixed global data)")
        a.set_ylabel("Macro-F1"); a.set_title(t)
        a.legend(fontsize=9, loc="lower left")
        a.set_ylim(0.955, 1.004)  # zoomed: note the narrow band
    fig.suptitle("Scalability: holding total data fixed, more clients = smaller/noisier shards "
                 "(y-axis zoomed to 0.955–1.0)",
                 fontsize=12.5, fontweight="bold", y=1.02)
    savefig(fig, "fig02_clients_scalability.png")


# ==========================================================================
# FIG 3: The "is it really federated?" diagnostic
#   left  = label-skew (TV) actually realised by each splitter vs alpha
#   right = F1 vs alpha for both splitters (text+image)
# ==========================================================================
def fig_heterogeneity_diagnostic():
    al = RES["alpha"]
    alphas = sorted(al["repo"]["text"].keys(), key=float)
    xa = [float(a) for a in alphas]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8))

    # left: realised label skew (averaged across modalities, same split so identical)
    tv_repo = [np.mean([al["repo"][m][a]["label_skew_tv"] for m in MODS]) for a in alphas]
    tv_lab = [np.mean([al["label"][m][a]["label_skew_tv"] for m in MODS]) for a in alphas]
    cv_repo = [np.mean([al["repo"][m][a]["size_cv"] for m in MODS]) for a in alphas]
    a1.plot(xa, tv_lab, "-o", color="#CC79A7", lw=2.2, ms=7,
            label="Correct label-Dirichlet split")
    a1.plot(xa, tv_repo, "-s", color="#0072B2", lw=2.2, ms=7,
            label="Repo split (paper's method)")
    a1.plot(xa, cv_repo, "--^", color="#999999", lw=1.6, ms=6,
            label="Repo split: size imbalance (CV)")
    a1.set_xscale("log")
    a1.set_xlabel("Dirichlet α  (smaller = more non-IID)")
    a1.set_ylabel("Realised heterogeneity across clients")
    a1.set_title("What non-IID does the split actually create?")
    a1.axhline(0, color="k", lw=0.6)
    a1.legend(fontsize=8.5, loc="upper right")
    a1.annotate("Repo split leaves label mix\n~identical (TV≈0) at every α;\nonly shard SIZE varies",
                xy=(xa[2], tv_repo[2]), xytext=(0.30, 0.30), textcoords="axes fraction",
                fontsize=8.5, color="#0072B2",
                arrowprops=dict(arrowstyle="->", color="#0072B2", lw=1.2))

    # right: F1 vs alpha, both splitters, for IMAGE (the only modality that moves)
    for sp, style, lab in (("repo", "-s", "Repo split (paper's method)"),
                           ("label", "-o", "Correct label-skew split")):
        f = [al[sp]["image"][a]["f1_mean"] for a in alphas]
        s = [al[sp]["image"][a]["f1_std"] for a in alphas]
        col = "#0072B2" if sp == "repo" else "#CC79A7"
        a2.errorbar(xa, f, yerr=s, fmt=style, color=col, lw=2, ms=7,
                    capsize=3, label=lab)
    a2.set_xscale("log")
    a2.set_xlabel("Dirichlet α  (smaller = more non-IID)")
    a2.set_ylabel("Federated Macro-F1 (Image-only)")
    a2.set_title("Does accuracy degrade under heterogeneity?")
    a2.legend(fontsize=9); a2.set_ylim(0.85, 1.01)
    a2.annotate("Real label skew costs\n~9 F1 points at α=0.1",
                xy=(xa[0], al["label"]["image"][alphas[0]]["f1_mean"]),
                xytext=(0.28, 0.35), textcoords="axes fraction",
                fontsize=8.5, color="#CC79A7",
                arrowprops=dict(arrowstyle="->", color="#CC79A7", lw=1.2))
    fig.suptitle("Federation reality check: the shipped splitter perturbs shard SIZE, not label mix",
                 fontsize=12.5, fontweight="bold", y=1.02)
    savefig(fig, "fig03_heterogeneity_diagnostic.png")


# ==========================================================================
# FIG 4: local epochs + participation (2 panels)
# ==========================================================================
def fig_localep_participation():
    le = RES["local_epochs"]; pr = RES["participation"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6))
    les = sorted(le["text"].keys(), key=int)
    for mk in MODS:
        y = [le[mk][k]["f1_mean"] for k in les]
        e = [le[mk][k]["f1_std"] for k in les]
        a1.errorbar([int(k) for k in les], y, yerr=e, fmt="-o", color=MC[mk],
                    lw=2, ms=6, capsize=3, label=ML[mk])
    a1.set_xlabel("Local epochs per round (E)")
    a1.set_ylabel("Macro-F1"); a1.set_title("Local computation vs accuracy\n(10 clients, 8 rounds)")
    a1.legend(fontsize=9); a1.set_ylim(0.955, 1.004)

    prs = sorted(pr["text"].keys(), key=float)
    for mk in MODS:
        y = [pr[mk][k]["f1_mean"] for k in prs]
        e = [pr[mk][k]["f1_std"] for k in prs]
        a2.errorbar([float(k) for k in prs], y, yerr=e, fmt="-o", color=MC[mk],
                    lw=2, ms=6, capsize=3, label=ML[mk])
    a2.set_xlabel("Client participation fraction (C)")
    a2.set_ylabel("Macro-F1"); a2.set_title("Partial participation\n(20 clients, 10 rounds)")
    a2.legend(fontsize=9); a2.set_ylim(0.955, 1.004)
    fig.suptitle("FedAvg knobs: local epochs and client sampling (y-axis zoomed to 0.955–1.0)",
                 fontsize=12.5, fontweight="bold", y=1.02)
    savefig(fig, "fig04_localepochs_participation.png")


# ==========================================================================
# FIG 5: Modality comparison — centralized vs federated grouped bars
# ==========================================================================
def fig_modality_comparison():
    cen = RES["centralized"]; rc = RES["rounds_x_clients"]
    x = np.arange(len(MODS)); w = 0.36
    cvals = [cen[m]["f1_macro_mean"] for m in MODS]
    cerr = [cen[m]["f1_macro_std"] for m in MODS]
    fvals = [rc[m]["3"]["best"] for m in MODS]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    b1 = ax.bar(x - w/2, cvals, w, yerr=cerr, capsize=4, color="#56B4E9",
                edgecolor="black", lw=0.6, label="Centralized")
    b2 = ax.bar(x + w/2, fvals, w, color="#E69F00",
                edgecolor="black", lw=0.6, label="Federated (K=3, paper config)")
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.012,
                f"{b.get_height():.3f}", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels([ML[m] for m in MODS])
    ax.set_ylabel("Macro-F1"); ax.set_ylim(0, 1.05)
    ax.set_title("Modality ablation: text-only vs image-only vs text+image\n"
                 "(centralized vs 3-client federated)")
    ax.legend(fontsize=10)
    savefig(fig, "fig05_modality_comparison.png")


# ==========================================================================
# FIG 6: Per-class F1 by modality (grouped bars)
# ==========================================================================
def fig_per_class():
    pc = RES["per_class"]
    classes = list(pc["text"].keys())
    x = np.arange(len(classes)); w = 0.26
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, mk in enumerate(MODS):
        vals = [pc[mk][c] for c in classes]
        ax.bar(x + (i - 1) * w, vals, w, color=MC[mk], edgecolor="black",
               lw=0.5, label=ML[mk])
    ax.set_xticks(x); ax.set_xticklabels([CLS_SHORT[c] for c in classes])
    ax.set_ylabel("Per-class F1"); ax.set_ylim(0.90, 1.006)
    ax.set_xlabel("Crop-stress class")
    ax.set_title("Where each modality helps: per-class federated F1\n"
                 "(10 clients, 15 rounds; y-axis zoomed to 0.90–1.0)")
    ax.legend(fontsize=9.5, loc="lower left")
    savefig(fig, "fig06_per_class_by_modality.png")


# ==========================================================================
# FIG 7: Modality lift heatmap + summary
# ==========================================================================
def fig_summary():
    cen = RES["centralized"]; rc = RES["rounds_x_clients"]
    fvc = RES["fed_vs_central"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8),
                                 gridspec_kw={"width_ratios": [1.15, 1]})
    # left: retention % (federated/centralized) per modality
    ret = [100 * fvc[m]["federated_3clients"] / max(cen[m]["f1_macro_mean"], 1e-9) for m in MODS]
    cols = [MC[m] for m in MODS]
    b = a1.bar([ML[m] for m in MODS], ret, color=cols, edgecolor="black", lw=0.6)
    a1.axhline(100, color="k", ls="--", lw=1)
    for bb, r in zip(b, ret):
        a1.text(bb.get_x()+bb.get_width()/2, bb.get_height()+1, f"{r:.0f}%",
                ha="center", fontsize=10, fontweight="bold")
    a1.set_ylabel("Federated retains (% of centralized F1)")
    a1.set_title("Federated vs centralized retention")
    a1.set_ylim(0, max(ret) * 1.18)

    # right: absolute F1 summary table-as-bars
    short = {"text": "LLM", "image": "ViT", "text+image": "VLM"}
    labels, vals, colors = [], [], []
    for m in MODS:
        labels += [f"{short[m]}\ncentral", f"{short[m]}\nfed K=3"]
        vals += [cen[m]["f1_macro_mean"], fvc[m]["federated_3clients"]]
        colors += [MC[m], MC[m]]
    xa = np.arange(len(vals))
    bb = a2.bar(xa, vals, color=colors, edgecolor="black", lw=0.5)
    for i, r in enumerate(bb):
        r.set_alpha(1.0 if i % 2 == 0 else 0.55)
        a2.text(r.get_x()+r.get_width()/2, r.get_height()+0.005, f"{vals[i]:.2f}",
                ha="center", fontsize=8.5)
    a2.set_xticks(xa); a2.set_xticklabels(labels, fontsize=8.5)
    a2.set_ylabel("Macro-F1"); a2.set_ylim(0.90, 1.02)
    a2.set_title("All modalities × {central, federated}\n(y-axis zoomed to 0.90–1.0)")
    fig.suptitle("Summary: multimodal fusion wins; federation is nearly loss-free at K=3",
                 fontsize=12.5, fontweight="bold", y=1.02)
    savefig(fig, "fig07_summary_retention.png")


if __name__ == "__main__":
    print("Generating plots ->", OUT)
    fig_rounds_convergence()
    fig_clients_scalability()
    fig_heterogeneity_diagnostic()
    fig_localep_participation()
    fig_modality_comparison()
    fig_per_class()
    fig_summary()
    print("All plots done.")
