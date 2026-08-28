"""Generate IEEE-sized figures for the updated multimodal experiments."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
RESULTS = (
    ROOT.parent
    / "tea_results"
    / "multimodal_v6_vision_full_20260727"
)
PLOTS = ROOT / "plots"
PLOTS.mkdir(exist_ok=True)

base = json.loads((RESULTS / "complete_results.json").read_text())
visual = json.loads(
    (RESULTS / "ensemble5_vit" / "vision_ensemble_results.json").read_text()
)
hierarchy = json.loads(
    (
        RESULTS
        / "modality_hierarchy_exploratory"
        / "modality_hierarchy_results.json"
    ).read_text()
)

INK = "#151515"
MUTED = "#6f6f6f"
GRID = "#dddddd"
BLUE = "#2878B5"
ORANGE = "#E07A1F"
GREEN = "#2A9D6F"
RED = "#D1495B"
PURPLE = "#7A5195"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 7.2,
        "axes.titlesize": 8.0,
        "axes.labelsize": 7.2,
        "xtick.labelsize": 6.7,
        "ytick.labelsize": 6.7,
        "legend.fontsize": 6.5,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "grid.color": GRID,
        "grid.linewidth": 0.55,
        "axes.linewidth": 0.65,
        "figure.dpi": 400,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.025,
    }
)


def style(ax, ylabel="Score"):
    ax.set_ylim(0, 1.04)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=2.2, width=0.6)


def label_bars(ax, bars, values):
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.025,
            f"{100 * value:.1f}",
            ha="center",
            va="bottom",
            fontsize=6.4,
            fontweight="bold",
        )


def save(fig, filename):
    path = PLOTS / filename
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    print("wrote", path)


# ---------------------------------------------------------------------------
# Standard checkpoint ablation
# ---------------------------------------------------------------------------
conditions = base["results"]["vlm"]["cross_modal_analysis"]["conditions"]
standard_order = ["image_only", "text_only", "paired", "mismatched_text"]
standard_labels = [
    "Image",
    "Text",
    "Paired",
    "Mismatch",
]
standard_acc = [conditions[name]["accuracy"] for name in standard_order]
standard_f1 = [conditions[name]["f1_macro"] for name in standard_order]

fig, ax = plt.subplots(figsize=(3.46, 2.15))
x = np.arange(len(standard_order))
width = 0.36
bars_a = ax.bar(
    x - width / 2,
    standard_acc,
    width,
    label="Accuracy",
    color=BLUE,
    edgecolor="white",
    linewidth=0.7,
    zorder=3,
)
bars_f = ax.bar(
    x + width / 2,
    standard_f1,
    width,
    label="Macro F1",
    color=ORANGE,
    edgecolor="white",
    linewidth=0.7,
    zorder=3,
)
label_bars(ax, bars_a, standard_acc)
label_bars(ax, bars_f, standard_f1)
ax.set_xticks(x)
ax.set_xticklabels(standard_labels)
style(ax)
ax.legend(frameon=False, ncol=2, loc="upper left")
save(fig, "plot65_standard_multimodal_ablation.png")

with (PLOTS / "new_modality_standard.csv").open("w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["condition", "accuracy", "macro_f1"])
    for name, accuracy, f1 in zip(standard_order, standard_acc, standard_f1):
        writer.writerow([name, accuracy, f1])


# ---------------------------------------------------------------------------
# Exploratory sparse-field-note hierarchy
# ---------------------------------------------------------------------------
hierarchy_order = [
    "text_only",
    "image_only",
    "text_image_fusion",
    "proposed_multimodal",
]
hierarchy_labels = [
    "Sparse\ntext",
    "Enhanced\nimage",
    "Late\nfusion",
    "Proposed",
]
hierarchy_colors = [PURPLE, BLUE, ORANGE, GREEN]
hierarchy_acc = [
    hierarchy["fixed_test"][name]["accuracy"] for name in hierarchy_order
]
hierarchy_f1 = [
    hierarchy["fixed_test"][name]["macro_f1"] for name in hierarchy_order
]

fig, ax = plt.subplots(figsize=(3.46, 2.15))
x = np.arange(len(hierarchy_order))
bars = ax.bar(
    x,
    hierarchy_acc,
    0.63,
    color=hierarchy_colors,
    edgecolor="white",
    linewidth=0.8,
    zorder=3,
)
label_bars(ax, bars, hierarchy_acc)
ax.plot(x, hierarchy_f1, "o--", color=INK, linewidth=1.0, markersize=3.6,
        label="Macro F1", zorder=4)
for xi, value in zip(x, hierarchy_f1):
    ax.text(xi, value - 0.055, f"{value:.3f}", ha="center", va="top",
            fontsize=5.8, color=INK)
ax.set_xticks(x)
ax.set_xticklabels(hierarchy_labels)
style(ax, "Accuracy / macro F1")
ax.text(
    0.5,
    0.965,
    "64.0%  <  72.0%  <  78.7%  <  81.3%",
    transform=ax.transAxes,
    ha="center",
    va="top",
    fontsize=6.4,
    fontweight="bold",
)
ax.legend(frameon=False, loc="lower left")
save(fig, "plot66_sparse_field_note_hierarchy.png")

with (PLOTS / "new_modality_hierarchy.csv").open("w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["condition", "accuracy", "macro_f1"])
    for name, accuracy, f1 in zip(
        hierarchy_order, hierarchy_acc, hierarchy_f1
    ):
        writer.writerow([name, accuracy, f1])


# ---------------------------------------------------------------------------
# Image-only improvement path
# ---------------------------------------------------------------------------
tta = json.loads(
    (RESULTS / "tta" / "vision_tta_results.json").read_text()
)
specialist = json.loads(
    (RESULTS / "specialist" / "vision_specialist_results.json").read_text()
)
calibration = json.loads(
    (RESULTS / "calibration" / "vision_calibration_results.json").read_text()
)
vit = json.loads(
    (RESULTS / "vit_specialist" / "vit_visual_specialist_results.json").read_text()
)
ensemble4 = json.loads(
    (RESULTS / "ensemble4" / "vision_ensemble_results.json").read_text()
)
progress_labels = ["Direct", "TTA", "Deep", "Calib.", "ViT", "4-exp.", "5-exp."]
progress_acc = [
    conditions["image_only"]["accuracy"],
    tta["locked_test"]["accuracy"],
    specialist["fixed_test"]["accuracy"],
    calibration["fixed_test"]["accuracy"],
    vit["fixed_test"]["accuracy"],
    ensemble4["fixed_test"]["accuracy"],
    visual["fixed_test"]["accuracy"],
]
progress_f1 = [
    conditions["image_only"]["f1_macro"],
    tta["locked_test"]["macro_f1"],
    specialist["fixed_test"]["macro_f1"],
    calibration["fixed_test"]["macro_f1"],
    vit["fixed_test"]["macro_f1"],
    ensemble4["fixed_test"]["macro_f1"],
    visual["fixed_test"]["macro_f1"],
]

fig, ax = plt.subplots(figsize=(3.46, 2.15))
x = np.arange(len(progress_labels))
colors = [BLUE] * 4 + [PURPLE, ORANGE, GREEN]
bars = ax.bar(
    x,
    progress_acc,
    0.62,
    color=colors,
    edgecolor="white",
    linewidth=0.8,
    zorder=3,
)
label_bars(ax, bars, progress_acc)
ax.plot(
    x,
    progress_f1,
    "o-",
    color=INK,
    linewidth=1.2,
    markersize=4,
    label="Macro F1",
    zorder=4,
)
ax.axhline(0.70, color=RED, linestyle="--", linewidth=0.9, zorder=2)
ax.text(5.78, 0.705, "70%", color=RED, fontsize=6.0, va="bottom")
ax.set_xticks(x)
ax.set_xticklabels(progress_labels)
style(ax, "Image-only score")
ax.legend(frameon=False, loc="lower right")
save(fig, "plot67_image_only_improvement.png")

with (PLOTS / "new_image_ensemble_progress.csv").open(
    "w", newline=""
) as handle:
    writer = csv.writer(handle)
    writer.writerow(["system", "accuracy", "macro_f1"])
    for name, accuracy, f1 in zip(progress_labels, progress_acc, progress_f1):
        writer.writerow([name.replace("\n", " "), accuracy, f1])


# ---------------------------------------------------------------------------
# Confusion matrices: enhanced image vs proposed sparse-field-note model
# ---------------------------------------------------------------------------
classes = ["Blight", "Hoppers", "Rust", "Looper", "Mosq. bug"]
matrices = [
    np.asarray(visual["fixed_test"]["confusion_matrix"]),
    np.asarray(
        hierarchy["fixed_test"]["proposed_multimodal"]["confusion_matrix"]
    ),
]
titles = [
    "Enhanced image only (72.0%)",
    "Proposed sparse multimodal (81.3%)",
]

fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.45))
for ax, matrix, title in zip(axes, matrices, titles):
    image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(m.max() for m in matrices))
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                fontsize=7,
                color="white" if matrix[row, column] >= 10 else INK,
                fontweight="bold" if row == column else "normal",
            )
    ax.set_xticks(range(5))
    ax.set_xticklabels(classes)
    ax.set_yticks(range(5))
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
fig.colorbar(image, ax=axes, fraction=0.025, pad=0.025, label="Count")
save(fig, "plot68_new_confusion_matrices.png")

for matrix, filename in zip(
    matrices,
    ["plot68a_image_confusion.png", "plot68b_proposed_confusion.png"],
):
    fig, ax = plt.subplots(figsize=(3.46, 2.15))
    image = ax.imshow(
        matrix,
        cmap="Blues",
        vmin=0,
        vmax=max(m.max() for m in matrices),
    )
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                fontsize=7.5,
                color="white" if matrix[row, column] >= 10 else INK,
                fontweight="bold" if row == column else "normal",
            )
    ax.set_xticks(range(5))
    ax.set_xticklabels(classes)
    ax.set_yticks(range(5))
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.035)
    save(fig, filename)

# ---------------------------------------------------------------------------
# Dataset audit: normalized class composition for every source-grouped split
# ---------------------------------------------------------------------------
split_names = ["Train\n$n=222$", "Validation\n$n=74$", "Test\n$n=75$"]
split_counts = np.asarray(
    [
        [77, 5, 40, 61, 39],
        [26, 2, 14, 20, 12],
        [26, 2, 13, 21, 13],
    ],
    dtype=float,
)
split_share = split_counts / split_counts.sum(axis=1, keepdims=True)
class_colors = [BLUE, RED, GREEN, ORANGE, PURPLE]

# Match the neighboring compact panels' height so this plot does not sit above
# their shared baseline when all four are placed at equal widths in LaTeX.
fig, ax = plt.subplots(figsize=(3.46, 1.80))
bottom = np.zeros(len(split_names))
for index, (label, color) in enumerate(zip(classes, class_colors)):
    bars = ax.bar(
        np.arange(len(split_names)),
        split_share[:, index],
        bottom=bottom,
        width=0.63,
        color=color,
        edgecolor="white",
        linewidth=0.6,
        label=label,
        zorder=3,
    )
    for bar, count, share in zip(bars, split_counts[:, index], split_share[:, index]):
        if share >= 0.07:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_y() + bar.get_height() / 2,
                f"{int(count)}",
                ha="center",
                va="center",
                fontsize=5.7,
                color="white",
                fontweight="bold",
            )
    bottom += split_share[:, index]
ax.set_xticks(np.arange(len(split_names)))
ax.set_xticklabels(split_names)
style(ax, "Fraction of split")
legend_handles, legend_labels = ax.get_legend_handles_labels()
ax.legend(
    legend_handles,
    legend_labels,
    frameon=False,
    ncol=5,
    mode="expand",
    loc="upper left",
    bbox_to_anchor=(0.0, -0.27, 1.0, 0.01),
    borderaxespad=0.0,
    columnspacing=0.5,
    handlelength=0.9,
    handletextpad=0.25,
    fontsize=5.7,
)
save(fig, "plot69_dataset_split_audit.png")


# ---------------------------------------------------------------------------
# Per-class recall: direct image, enhanced image, and exploratory proposed
# ---------------------------------------------------------------------------
def recall_from_matrix(matrix):
    matrix = np.asarray(matrix, dtype=float)
    totals = matrix.sum(axis=1)
    return np.divide(
        np.diag(matrix),
        totals,
        out=np.zeros_like(totals, dtype=float),
        where=totals > 0,
    )


direct_recall = recall_from_matrix(conditions["image_only"]["confusion_matrix"])
enhanced_recall = recall_from_matrix(visual["fixed_test"]["confusion_matrix"])
proposed_recall = recall_from_matrix(
    hierarchy["fixed_test"]["proposed_multimodal"]["confusion_matrix"]
)

fig, ax = plt.subplots(figsize=(3.46, 2.15))
x = np.arange(len(classes))
width = 0.25
ax.bar(
    x - width,
    direct_recall,
    width,
    label="Direct image",
    color=MUTED,
    edgecolor="white",
    linewidth=0.5,
    zorder=3,
)
ax.bar(
    x,
    enhanced_recall,
    width,
    label="Enhanced image",
    color=BLUE,
    edgecolor="white",
    linewidth=0.5,
    zorder=3,
)
ax.bar(
    x + width,
    proposed_recall,
    width,
    label="Proposed",
    color=GREEN,
    edgecolor="white",
    linewidth=0.5,
    zorder=3,
)
ax.set_xticks(x)
ax.set_xticklabels(classes)
style(ax, "Recall")
ax.legend(
    frameon=False,
    ncol=3,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.18),
    columnspacing=0.7,
)
save(fig, "plot70_classwise_recall.png")


# ---------------------------------------------------------------------------
# Cross-modal diagnostics: learned reliability and retrieval alignment
# ---------------------------------------------------------------------------
paired_weights = conditions["paired"]["mean_modality_weights"]
text_weight = paired_weights["text"] if isinstance(paired_weights, dict) else paired_weights[0]
image_weight = paired_weights["image"] if isinstance(paired_weights, dict) else paired_weights[1]
retrieval_labels = ["Exact\nT$\\to$I", "Class\nT$\\to$I", "Class\nI$\\to$T"]
retrieval_values = [0.0133, 0.6000, 0.6133]

fig, axes = plt.subplots(1, 2, figsize=(3.46, 2.15))
left, right = axes
bars = left.bar(
    ["Text", "Image"],
    [text_weight, image_weight],
    color=[PURPLE, BLUE],
    edgecolor="white",
    linewidth=0.6,
    width=0.58,
    zorder=3,
)
label_bars(left, bars, [text_weight, image_weight])
style(left, "Mean paired weight")

bars = right.bar(
    retrieval_labels,
    retrieval_values,
    color=[RED, ORANGE, GREEN],
    edgecolor="white",
    linewidth=0.6,
    width=0.62,
    zorder=3,
)
label_bars(right, bars, retrieval_values)
style(right, "Recall@1")
right.set_ylabel("")
save(fig, "plot71_crossmodal_diagnostics.png")


# ---------------------------------------------------------------------------
# Source-group bootstrap accuracy intervals for controlled evaluations
# ---------------------------------------------------------------------------
ci_labels = ["Direct\nimage", "Enhanced\nimage", "Paired\nstandard"]
ci_point = np.asarray([0.6133, 0.7200, 0.9200])
ci_low = np.asarray([0.4857, 0.6023, 0.8571])
ci_high = np.asarray([0.7568, 0.8514, 0.9822])
xerr = np.vstack([ci_point - ci_low, ci_high - ci_point])

fig, ax = plt.subplots(figsize=(3.46, 2.15))
x = np.arange(len(ci_labels))
for xi, point, low, high, color in zip(
    x, ci_point, ci_low, ci_high, [BLUE, GREEN, ORANGE]
):
    ax.errorbar(
        [xi],
        [point],
        yerr=[[point - low], [high - point]],
        fmt="o",
        color=INK,
        ecolor=color,
        elinewidth=2.2,
        capsize=5,
        capthick=1.1,
        markersize=5,
        zorder=4,
    )
    ax.text(
        xi,
        point + 0.035,
        f"{100 * point:.1f}",
        ha="center",
        fontsize=6.5,
        fontweight="bold",
    )
    ax.text(
        xi,
        low - 0.04,
        f"{100 * low:.1f}-{100 * high:.1f}",
        ha="center",
        fontsize=5.7,
        color=MUTED,
    )
ax.set_xticks(x)
ax.set_xticklabels(ci_labels)
style(ax, "Accuracy")
ax.set_ylim(0.40, 1.03)
save(fig, "plot72_source_bootstrap_accuracy.png")

print("done")
