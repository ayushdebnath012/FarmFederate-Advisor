"""Architecture ablation figure: two independent runs of the same sweep.

The point of the figure is not a component ranking. Both runs use the same code,
seed, split and locked test set and differ only in execution hardware, so the
spread between the paired points is the run-to-run variance floor. Any component
effect smaller than that spread is unmeasurable at this support.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN_A = ROOT / "tea_results" / "architecture_ablation_v1" / "architecture_ablation_results.json"
RUN_B = ROOT / "tea_results" / "architecture_ablation_gpu" / "architecture_ablation_results.json"
OUTPUT = ROOT / "overleaf_final" / "plots"

INK = "#161616"
MUTED = "#6C757D"
GRID = "#DDE1E5"
RUN_A_C = "#2878B5"
RUN_B_C = "#E67700"
PROPOSED = "#A61E4D"

SHORT = {
    "full": "Proposed (full)",
    "no_cross_attention": "no cross-attention",
    "no_reliability_gate": "no reliability gate",
    "no_expert_residual": "no expert residual",
    "no_interaction_feats": "no interaction feats.",
    "lightweight_vision": "lightweight CNN vision",
    "text_1_layer": "text encoder: 1 layer",
    "no_aux_losses": "no auxiliary losses",
    "no_alignment_loss": "no alignment loss",
}


def configure() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.2,
            "axes.labelsize": 7.4,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.4,
            "legend.fontsize": 6.2,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.linewidth": 0.65,
            "grid.color": GRID,
            "grid.linewidth": 0.5,
            "savefig.dpi": 400,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def clean(axis, which="x") -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis=which, zorder=0)
    axis.set_axisbelow(True)
    axis.tick_params(length=2.2, width=0.6, pad=1.8)


def load(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("experiment") != "multimodal_architecture_ablation":
        raise RuntimeError(f"unexpected provenance in {path}")
    audit = payload["data_audit"]
    if audit["source_group_overlap"] != 0 or audit["test_crops"] != 75:
        raise ValueError(f"{path} did not use the audited split")
    return {r["variant"]: r for r in payload["variants"]}


def main() -> None:
    configure()
    a, b = load(RUN_A), load(RUN_B)
    if set(a) != set(b):
        raise ValueError("the two runs cover different variants")

    order = sorted(a, key=lambda k: -(a[k]["test_accuracy"] + b[k]["test_accuracy"]) / 2)
    names = [SHORT.get(k, k) for k in order]
    acc_a = np.array([a[k]["test_accuracy"] for k in order])
    acc_b = np.array([b[k]["test_accuracy"] for k in order])
    d_a = np.array([a[k]["delta_accuracy"] for k in order])
    d_b = np.array([b[k]["delta_accuracy"] for k in order])

    # Single IEEE column: one panel carrying both messages, since a two-panel
    # 7.16in layout is illegible when scaled into 3.5in.
    figure, left = plt.subplots(figsize=(3.42, 1.98))
    y = np.arange(len(order))

    # Paired test accuracy. Connector length is the run-to-run variance floor;
    # a red connector marks a variant whose measured effect reverses sign.
    flips = 0
    for position, key, va, vb in zip(y, order, acc_a, acc_b):
        reversed_sign = key != "full" and a[key]["delta_accuracy"] * b[key]["delta_accuracy"] < 0
        flips += int(reversed_sign)
        left.hlines(position, min(va, vb), max(va, vb),
                    color=PROPOSED if reversed_sign else MUTED,
                    linewidth=1.7 if reversed_sign else 1.2, zorder=2)
        if key == "full":
            left.axhspan(position - 0.45, position + 0.45, color=PROPOSED,
                         alpha=0.08, zorder=0)
    left.scatter(acc_a, y, s=20, color=RUN_A_C, edgecolor="white", linewidth=0.45,
                 zorder=4, label="Run A (CPU)")
    left.scatter(acc_b, y, s=20, color=RUN_B_C, edgecolor="white", linewidth=0.45,
                 zorder=4, label="Run B (H100)")
    left.set_yticks(y)
    left.set_yticklabels(names, fontsize=6.0)
    left.invert_yaxis()
    left.set_xlim(0.76, 1.005)
    left.set_xlabel("Locked 75-crop test accuracy")
    left.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
                frameon=False, handletextpad=0.25, columnspacing=0.9,
                fontsize=6.0)
    left.text(0.985, 0.03, f"{flips} of 8 effects reverse sign",
              transform=left.transAxes, ha="right", va="bottom", fontsize=5.7,
              color=PROPOSED, fontweight="bold")
    left.text(-0.39, 1.04, "(d)", transform=left.transAxes,
              ha="left", va="bottom", fontsize=6.7, fontweight="bold")
    clean(left)

    figure.subplots_adjust(left=0.40, right=0.985, bottom=0.16, top=0.90)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    target = OUTPUT / "plot81_architecture_ablation.png"
    figure.savefig(target, dpi=400, bbox_inches="tight", pad_inches=0.025,
                   facecolor="white")
    figure.savefig(target.with_suffix(".pdf"), bbox_inches="tight",
                   pad_inches=0.025, facecolor="white")
    plt.close(figure)

    # Short paired-effect strip for the remaining space in an IEEE column.
    # It preserves all eight ablation deltas while avoiding an unreadably small
    # version of the absolute-accuracy dumbbell chart.
    strip_order = [key for key in order if key != "full"]
    strip_names = {
        "lightweight_vision": "Light\nCNN",
        "no_expert_residual": "No expert\nres.",
        "no_alignment_loss": "No align.",
        "text_1_layer": "Text\n1 layer",
        "no_reliability_gate": "No gate",
        "no_cross_attention": "No cross\nattn.",
        "no_interaction_feats": "No inter.\nfeats.",
        "no_aux_losses": "No aux.\nlosses",
    }
    strip_a = 100.0 * np.array([a[key]["delta_accuracy"] for key in strip_order])
    strip_b = 100.0 * np.array([b[key]["delta_accuracy"] for key in strip_order])
    strip_x = np.arange(len(strip_order))
    strip_figure, strip = plt.subplots(figsize=(3.42, 1.22))
    for position, key, va, vb in zip(strip_x, strip_order, strip_a, strip_b):
        reversed_sign = a[key]["delta_accuracy"] * b[key]["delta_accuracy"] < 0
        strip.vlines(position, min(va, vb), max(va, vb),
                     color=PROPOSED if reversed_sign else MUTED,
                     linewidth=1.45 if reversed_sign else 1.05, zorder=2)
    strip.scatter(strip_x - 0.06, strip_a, s=16, color=RUN_A_C,
                  edgecolor="white", linewidth=0.4, zorder=4, label="Run A (CPU)")
    strip.scatter(strip_x + 0.06, strip_b, s=16, color=RUN_B_C,
                  edgecolor="white", linewidth=0.4, zorder=4, label="Run B (H100)")
    strip.axhline(0.0, color=INK, linewidth=0.75, linestyle="--", zorder=1)
    strip.set_xticks(strip_x)
    strip.set_xticklabels([strip_names[key] for key in strip_order], fontsize=5.25)
    strip.set_xlim(-0.45, len(strip_order) - 0.55)
    strip.set_ylim(-15.2, 11.2)
    strip.set_ylabel(r"$\Delta$ accuracy (points)")
    strip.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
                 frameon=False, handletextpad=0.25, columnspacing=0.8,
                 fontsize=5.7)
    strip.text(0.985, 0.97, f"{flips}/8 signs reverse",
               transform=strip.transAxes, ha="right", va="top",
               fontsize=5.5, color=PROPOSED, fontweight="bold")
    clean(strip, "y")
    strip_figure.subplots_adjust(left=0.13, right=0.985, bottom=0.31, top=0.82)
    strip_target = OUTPUT / "plot81_architecture_effect_strip.png"
    strip_figure.savefig(strip_target, dpi=400, bbox_inches="tight",
                         pad_inches=0.025, facecolor="white")
    strip_figure.savefig(strip_target.with_suffix(".pdf"), bbox_inches="tight",
                         pad_inches=0.025, facecolor="white")
    plt.close(strip_figure)
    gap = 100.0 * float(np.mean(np.abs(acc_a - acc_b)))
    print(f"{target}\n{strip_target}\n  mean |A-B| gap {gap:.2f} points, "
          f"{flips}/8 sign reversals")


if __name__ == "__main__":
    main()
