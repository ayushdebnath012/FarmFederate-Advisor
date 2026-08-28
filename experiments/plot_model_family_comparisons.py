"""Publication figures for model-family screening and current exact-pair results.

The legacy July 19 family screens use different held-out supports and are shown
only within family.  The current inter-model panel is stricter: it requires four
systems evaluated on the same exact-pair label order before it will render.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import Bbox


ROOT = Path(__file__).resolve().parents[1]
LEGACY_RESULTS = (
    ROOT
    / "overleaf_final"
    / "farmfederate_results_20260719_190712 new"
    / "results"
    / "complete_results.json"
)
CURRENT_RESULTS = (
    ROOT
    / "tea_results"
    / "current_family_baselines_20260728"
    / "current_family_baselines.json"
)
FEDERATED_RESULTS = (
    ROOT
    / "tea_results"
    / "federated_adaptation_v6"
    / "federated_adaptation_results.json"
)
ENCODER_SELECTION_RESULTS = (
    ROOT
    / "tea_results"
    / "vit_lm_encoder_selection_20260729"
    / "encoder_selection_results.json"
)
V6_ROOT = ROOT / "tea_results" / "multimodal_v6_vision_full_20260727"
V6_RESULTS = V6_ROOT / "complete_results.json"
V6_VISION_ENSEMBLE = V6_ROOT / "ensemble5_vit" / "vision_ensemble_results.json"
V6_HIERARCHY = (
    V6_ROOT / "modality_hierarchy_exploratory" / "modality_hierarchy_results.json"
)
V6_STAGES = {
    "tta": V6_ROOT / "tta" / "vision_tta_results.json",
    "specialist": V6_ROOT / "specialist" / "vision_specialist_results.json",
    "calibration": V6_ROOT / "calibration" / "vision_calibration_results.json",
    "vit": V6_ROOT / "vit_specialist" / "vit_visual_specialist_results.json",
    "ensemble4": V6_ROOT / "ensemble4" / "vision_ensemble_results.json",
}
OUTPUT = ROOT / "overleaf_final" / "plots"


INK = "#161616"
MUTED = "#6C757D"
GRID = "#DDE1E5"
TEXT_BLUE = "#2878B5"
VISION_ORANGE = "#E67700"
FUSION_GREEN = "#2A9D6F"
PROPOSED = "#A61E4D"
ALTERNATIVE = "#6F42C1"

CURRENT_ORDER = (
    "text_only_llm",
    "image_only_vit",
    "image_text_vlm",
    "proposed_farmfederate",
)
CURRENT_LABELS = (
    "Text-only\nPLM",
    "Image-only\nViT",
    "Image+text\nVLM",
    "Proposed\nmodel",
)
CURRENT_COLORS = (TEXT_BLUE, VISION_ORANGE, FUSION_GREEN, PROPOSED)

FUSION_LABELS = {
    "concat": "Concat",
    "attention": "Attention",
    "gated": "Gated",
    "clip": "CLIP-style",
    "flamingo": "Flamingo-style",
    "blip2": "BLIP-2-style",
    "coca": "CoCa-style",
    "unified_io": "Unified-IO",
    "flexible_cross_attention": "Flexible cross-attn",
    "pretrained_late_fusion": "Pretrained late fusion",
}


def configure_style() -> None:
    """Set stable IEEE-print typography without plot-internal titles."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.2,
            "axes.labelsize": 7.6,
            "xtick.labelsize": 6.6,
            "ytick.labelsize": 6.4,
            "legend.fontsize": 6.4,
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


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return payload


def metric(row: dict[str, Any], key: str, context: str) -> float:
    if key not in row:
        raise KeyError(f"{context} is missing {key!r}")
    value = float(row[key])
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{context}.{key} must be finite and in [0, 1]")
    return value


def clean_axis(axis: plt.Axes, grid_axis: str = "x") -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis=grid_axis, zorder=0)
    axis.set_axisbelow(True)
    axis.tick_params(length=2.2, width=0.6, pad=1.8)


def save(figure: plt.Figure, filename: str) -> Path:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    target = OUTPUT / filename
    figure.savefig(
        target,
        dpi=400,
        bbox_inches="tight",
        pad_inches=0.025,
        facecolor="white",
    )
    # Keep the PNG for previews and a vector PDF for the manuscript.
    figure.savefig(
        target.with_suffix(".pdf"),
        bbox_inches="tight",
        pad_inches=0.025,
        facecolor="white",
    )
    plt.close(figure)
    print(target)
    return target


def save_panel_crops(
    figure: plt.Figure,
    axes: tuple[plt.Axes, ...],
    filenames: tuple[str, ...],
) -> None:
    """Export compound-figure axes as real manuscript panels.

    Keeping these as separate assets prevents LaTeX from treating a three-axis
    diagnostic as one small subfigure beside a fourth plot.
    """
    if len(axes) != len(filenames):
        raise ValueError("every diagnostic axis needs one output filename")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    for axis, filename in zip(axes, filenames):
        tight = axis.get_tightbbox(renderer).transformed(
            figure.dpi_scale_trans.inverted()
        )
        pad = 0.025
        bounds = Bbox.from_extents(
            tight.x0 - pad,
            tight.y0 - pad,
            tight.x1 + pad,
            tight.y1 + pad,
        )
        target = OUTPUT / filename
        figure.savefig(
            target,
            dpi=400,
            bbox_inches=bounds,
            facecolor="white",
        )
        figure.savefig(
            target.with_suffix(".pdf"),
            bbox_inches=bounds,
            facecolor="white",
        )
        print(target)


def legacy_family_rows(
    payload: dict[str, Any],
) -> list[tuple[str, list[str], list[float], str, str]]:
    """Return label/value groups plus their validation-selected representative."""
    required = ("llm_models", "vit_models", "vlm_models", "fusion_common_test")
    missing = [key for key in required if key not in payload]
    if missing:
        raise KeyError(f"Legacy results are missing: {missing}")

    llm = payload["llm_models"]
    vision = payload["vit_models"]
    fusion = payload["vlm_models"]
    if not all(isinstance(group, dict) for group in (llm, vision, fusion)):
        raise TypeError("Legacy family result groups must be JSON objects")
    if (len(llm), len(vision), len(fusion)) != (5, 5, 10):
        raise ValueError(
            "Expected exactly five PLMs, five vision backbones, and ten "
            f"fusion surrogates; observed {(len(llm), len(vision), len(fusion))}"
        )

    common = payload["fusion_common_test"]
    if not isinstance(common, dict):
        raise TypeError("fusion_common_test must be a JSON object")
    selected_text = str(common.get("text_parent_name", ""))
    selected_vision = str(common.get("vision_parent_name", ""))
    selected_model = payload.get("validation_selected_multimodal_model", {})
    selected_fusion = (
        str(selected_model.get("name", ""))
        if isinstance(selected_model, dict)
        else ""
    )

    groups = [
        ("(a)", llm, selected_text, TEXT_BLUE),
        ("(b)", vision, selected_vision, VISION_ORANGE),
        ("(c)", fusion, selected_fusion, FUSION_GREEN),
    ]
    output: list[tuple[str, list[str], list[float], str, str]] = []
    for panel_label, rows, selected, color in groups:
        if selected not in rows:
            raise KeyError(
                f"Selected representative {selected!r} is absent from {panel_label}"
            )
        names = list(rows)
        labels = [
            FUSION_LABELS.get(name, name) if rows is fusion else name
            for name in names
        ]
        values = [
            metric(rows[name], "f1_macro", f"{panel_label}.{name}")
            for name in names
        ]
        output.append((panel_label, labels, values, selected, color))
    return output


def plot_within_family(payload: dict[str, Any]) -> Path:
    """Draw title-free, within-family legacy screens without cross-family ranking."""
    groups = legacy_family_rows(payload)
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(7.16, 1.72),
        gridspec_kw={"wspace": 0.69},
    )

    for axis, (panel_label, labels, values, selected_key, color) in zip(
        axes, groups
    ):
        names_for_selection = (
            list(payload["llm_models"])
            if panel_label.startswith("(a)")
            else list(payload["vit_models"])
            if panel_label.startswith("(b)")
            else list(payload["vlm_models"])
        )
        selected_index = names_for_selection.index(selected_key)
        y = np.arange(len(labels))
        for row_index, (position, value) in enumerate(zip(y, values)):
            is_selected = row_index == selected_index
            line_color = PROPOSED if is_selected else color
            axis.hlines(
                position,
                0.0,
                value,
                color=line_color,
                alpha=0.88 if is_selected else 0.48,
                linewidth=2.1 if is_selected else 1.35,
                zorder=2,
            )
            axis.scatter(
                value,
                position,
                marker="D" if is_selected else "o",
                s=23 if is_selected else 15,
                color=line_color,
                edgecolor="white",
                linewidth=0.55,
                zorder=4,
            )
            axis.text(
                min(value + 0.025, 0.965),
                position,
                f"{value:.3f}",
                ha="left" if value < 0.94 else "right",
                va="center",
                fontsize=5.7,
                color=PROPOSED if is_selected else INK,
                fontweight="bold" if is_selected else "normal",
            )

        shown_labels = [
            f"{label}  ★" if index == selected_index else label
            for index, label in enumerate(labels)
        ]
        axis.set_yticks(y)
        axis.set_yticklabels(shown_labels)
        axis.invert_yaxis()
        axis.set_xlim(0.0, 1.01)
        axis.set_xticks([0.0, 0.25, 0.50, 0.75, 1.0])
        axis.set_xlabel("Held-out macro-F1")
        clean_axis(axis, "x")
        axis.text(
            0.0,
            1.025,
            panel_label,
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=6.7,
            fontweight="bold",
        )
        tick_labels = axis.get_yticklabels()
        tick_labels[selected_index].set_color(PROPOSED)
        tick_labels[selected_index].set_fontweight("bold")

    # The star is explained in the caption; an in-figure note would sit directly
    # above the lower row and read as if it belonged to that panel.
    figure.subplots_adjust(left=0.13, right=0.995, bottom=0.235, top=0.945)
    return save(figure, "plot75_within_family_models.png")


def current_exact_rows(
    payload: dict[str, Any],
) -> list[dict[str, Any]]:
    """Validate four current systems on one common exact-pair support."""
    current = payload.get("current_exact_pair")
    if not isinstance(current, dict):
        raise TypeError("current_exact_pair must be a JSON object keyed by system")
    missing = [key for key in CURRENT_ORDER if key not in current]
    extras = [key for key in current if key not in CURRENT_ORDER]
    if missing or extras:
        raise KeyError(
            "current_exact_pair must contain exactly "
            f"{list(CURRENT_ORDER)}; missing={missing}, extras={extras}"
        )

    rows: list[dict[str, Any]] = []
    reference_labels: np.ndarray | None = None
    reference_support: int | None = None
    for key in CURRENT_ORDER:
        row = current[key]
        if not isinstance(row, dict):
            raise TypeError(f"current_exact_pair.{key} must be a JSON object")
        if str(row.get("system_key", key)) != key:
            raise ValueError(f"current_exact_pair.{key}.system_key is inconsistent")

        accuracy = metric(row, "accuracy", f"current_exact_pair.{key}")
        macro_f1 = metric(row, "macro_f1", f"current_exact_pair.{key}")
        labels = np.asarray(row.get("labels", []), dtype=int).reshape(-1)
        predictions = np.asarray(row.get("predictions", []), dtype=int).reshape(-1)
        support_record = row.get("support", len(labels))
        if isinstance(support_record, dict):
            support = int(support_record.get("crops", len(labels)))
        else:
            support = int(support_record)
        if support <= 0 or len(labels) != support or len(predictions) != support:
            raise ValueError(
                f"{key} must provide support-matched labels and predictions"
            )
        if reference_support is None:
            reference_support = support
            reference_labels = labels
        elif support != reference_support or not np.array_equal(
            labels, reference_labels
        ):
            raise ValueError(
                "All current exact-pair systems must use identical support "
                "and label order"
            )

        correct = int(np.sum(predictions == labels))
        if "correct" in row and int(row["correct"]) != correct:
            raise ValueError(f"{key}.correct disagrees with predictions")
        if not np.isclose(accuracy, correct / support, atol=5e-7):
            raise ValueError(f"{key}.accuracy disagrees with predictions")

        architecture = str(row.get("architecture", "")).lower()
        if key == "text_only_llm" and not any(
            token in architecture for token in ("bert", "language model", "llm")
        ):
            raise ValueError(
                "text_only_llm architecture is not identified as a pretrained "
                "language encoder"
            )
        if key == "image_only_vit" and "vit" not in architecture:
            raise ValueError("image_only_vit architecture is not identified as a ViT")

        rows.append(
            {
                **row,
                "system_key": key,
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "support": support,
            }
        )
    return rows


def federated_summary(payload: dict[str, Any]) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Extract central initialization and final FedAvg statistics."""
    config = payload.get("config")
    runs = payload.get("runs")
    summary = payload.get("summary")
    if not isinstance(config, dict) or not isinstance(runs, list) or not runs:
        raise TypeError("Federated results require config and non-empty runs")
    if not isinstance(summary, dict):
        raise TypeError("Federated results require a summary object")

    clients = np.asarray([int(value) for value in config.get("clients", [])])
    if clients.tolist() != [2, 3, 5, 8]:
        raise ValueError(f"Expected client counts [2, 3, 5, 8], found {clients}")
    baseline = metric(
        runs[0]["baseline_validation"],
        "f1_macro",
        "runs[0].baseline_validation",
    )
    means = np.asarray(
        [
            metric(
                summary[str(client)],
                "final_macro_f1_mean",
                f"summary.{client}",
            )
            for client in clients
        ],
        dtype=float,
    )
    stds = np.asarray(
        [
            float(summary[str(client)]["final_macro_f1_std"])
            for client in clients
        ],
        dtype=float,
    )
    if np.any(~np.isfinite(stds)) or np.any(stds < 0):
        raise ValueError("Federated final standard deviations must be finite and nonnegative")
    return baseline, clients, means, stds


def plot_current_and_federated(
    current_payload: dict[str, Any],
    federated_payload: dict[str, Any],
    encoder_selection_payload: dict[str, Any],
) -> Path:
    """Draw current family baselines and proposed centralized/FedAvg retention."""
    rows = current_exact_rows(current_payload)
    candidate = encoder_selection_payload["candidate"]["fixed_test"]
    candidate_labels = np.asarray(candidate.get("labels", []), dtype=int).reshape(-1)
    if len(candidate_labels) != rows[0]["support"] or not np.array_equal(
        candidate_labels, np.asarray(rows[0]["labels"], dtype=int).reshape(-1)
    ):
        raise ValueError(
            "The ViT-tiny + DistilBERT candidate must use the identical ordered "
            "75-crop support"
        )
    comparison_rows = [
        rows[0],
        rows[1],
        rows[2],
        {
            **candidate,
            "system_key": "vit_distilbert_candidate",
            "support": len(candidate_labels),
        },
        rows[3],
    ]
    comparison_labels = (
        "Text-only\nPLM",
        "Image-only\nViT",
        "Raw image+text\nVLM",
        "ViT+PLM\nalternative",
        "Retained\nincumbent",
    )
    comparison_colors = (
        TEXT_BLUE,
        VISION_ORANGE,
        FUSION_GREEN,
        ALTERNATIVE,
        PROPOSED,
    )
    baseline, clients, fed_means, fed_stds = federated_summary(federated_payload)

    figure, (left, right) = plt.subplots(
        1,
        2,
        figsize=(7.16, 1.72),
        gridspec_kw={"wspace": 0.31},
    )

    # (a) Current systems on identical ordered labels. A horizontal paired-dot
    # design remains readable after two-column scaling and shows the alternative
    # without implying that it won selection.
    y = np.arange(len(comparison_rows))
    accuracy = np.asarray([row["accuracy"] for row in comparison_rows], dtype=float)
    macro_f1 = np.asarray([row["macro_f1"] for row in comparison_rows], dtype=float)
    for position, acc, f1, color in zip(
        y, accuracy, macro_f1, comparison_colors
    ):
        left.hlines(
            position,
            min(acc, f1),
            max(acc, f1),
            color=color,
            linewidth=1.7,
            alpha=0.55,
            zorder=2,
        )
    left.scatter(
        accuracy,
        y,
        color=comparison_colors,
        marker="o",
        s=23,
        edgecolor="white",
        linewidth=0.55,
        label="Accuracy",
        zorder=4,
    )
    left.scatter(
        macro_f1,
        y,
        facecolor="white",
        edgecolor=comparison_colors,
        marker="s",
        s=22,
        linewidth=1.0,
        label="Macro-F1",
        zorder=4,
    )
    for position, acc, f1 in zip(y, accuracy, macro_f1):
        left.text(
            max(acc, f1) + 0.012,
            position,
            f"{100.0 * acc:.1f}/{100.0 * f1:.1f}",
            ha="left",
            va="center",
            fontsize=5.4,
        )
    left.axhspan(2.55, 3.45, color=ALTERNATIVE, alpha=0.035, zorder=0)
    left.axhspan(3.55, 4.45, color=PROPOSED, alpha=0.045, zorder=0)
    left.set_yticks(y)
    left.set_yticklabels(comparison_labels)
    left.invert_yaxis()
    left.set_xlim(0.50, 0.88)
    left.set_xlabel("Locked-test score (accuracy / macro-F1 labels)")
    clean_axis(left, "x")
    left.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=2,
        frameon=False,
        columnspacing=1.2,
        handlelength=1.3,
    )
    left.text(
        -0.105,
        1.025,
        "(a)",
        transform=left.transAxes,
        ha="left",
        va="bottom",
        fontweight="bold",
        clip_on=False,
    )
    left.get_yticklabels()[-2].set_color(ALTERNATIVE)
    left.get_yticklabels()[-1].set_color(PROPOSED)
    left.get_yticklabels()[-1].set_fontweight("bold")

    # (b) The unchanged proposed model before and after federated adaptation.
    right.axhline(
        baseline,
        color=INK,
        linestyle=(0, (3, 2)),
        linewidth=1.25,
        label=f"Centralized initialization ({baseline:.3f})",
        zorder=2,
    )
    right.plot(
        clients,
        fed_means,
        color=PROPOSED,
        marker="o",
        markerfacecolor=PROPOSED,
        markeredgecolor="white",
        markeredgewidth=0.65,
        linewidth=2.0,
        label="FedAvg final mean",
        zorder=4,
    )
    for client, value in zip(clients, fed_means):
        right.annotate(
            f"{value:.3f}",
            (client, value),
            xytext=(0, -8),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=6.0,
            color=PROPOSED,
            fontweight="bold",
        )
    observed_low = float(min(baseline, np.min(fed_means)))
    observed_high = float(max(baseline, np.max(fed_means)))
    span = observed_high - observed_low
    # Extra room below for the labels; extra room above for the legend.
    right.set_ylim(
        max(0.0, observed_low - max(0.022, 0.42 * span)),
        min(1.0, observed_high + max(0.010, 0.18 * span)),
    )
    right.set_xlim(1.55, 8.45)
    right.set_xticks(clients)
    right.set_xlabel("Number of federated clients, $K$")
    right.set_ylabel("Validation macro-F1")
    right.yaxis.set_major_locator(MaxNLocator(nbins=6))
    clean_axis(right, "y")
    right.legend(
        loc="lower right",
        frameon=True,
        framealpha=0.95,
        edgecolor="#DEE2E6",
        borderpad=0.3,
        handlelength=2.0,
        labelspacing=0.22,
    )
    right.text(
        -0.105,
        1.025,
        "(b)",
        transform=right.transAxes,
        ha="left",
        va="bottom",
        fontweight="bold",
        clip_on=False,
    )

    figure.subplots_adjust(left=0.13, right=0.995, bottom=0.285, top=0.925)
    return save(figure, "plot76_current_intermodel_and_central_fed.png")


CLASS_LABELS = ("Blight", "Hoppers", "Rust", "Looper", "Mosq. bug")
TEST_SUPPORT = 75

# Source-group bootstrap intervals for the three controlled conditions. Only the
# enhanced-image interval appears in a table, so the panel adds the other two and
# their overlap; widths are asserted against the audited values.
BOOTSTRAP = (
    ("Direct\nimage", 0.6133, 0.4857, 0.7568, VISION_ORANGE),
    ("Enhanced\nimage", 0.7200, 0.6023, 0.8514, FUSION_GREEN),
    ("Paired\nstandard", 0.9200, 0.8571, 0.9822, TEXT_BLUE),
)
EXPECTED_WIDTHS = (27.11, 24.91, 12.51)

LADDER = (
    ("Direct", ("conditions", "image_only")),
    ("TTA", ("tta", "locked_test")),
    ("Deep", ("specialist", "fixed_test")),
    ("Calib.", ("calibration", "fixed_test")),
    ("ViT", ("vit", "fixed_test")),
    ("4-exp.", ("ensemble4", "fixed_test")),
    ("5-exp.", ("visual", "fixed_test")),
)


def recall_from_matrix(matrix: Any) -> np.ndarray:
    counts = np.asarray(matrix, dtype=float)
    if counts.ndim != 2 or counts.shape[0] != counts.shape[1]:
        raise ValueError("A confusion matrix must be square")
    totals = counts.sum(axis=1)
    return np.divide(
        np.diag(counts), totals, out=np.zeros_like(totals), where=totals > 0
    )


def _accuracy_pair(row: dict[str, Any]) -> tuple[float, float, int]:
    """Return (accuracy, macro F1, correct count) for one evaluated system."""
    accuracy = float(row["accuracy"] if "accuracy" in row else row["acc"])
    macro = float(row["macro_f1"] if "macro_f1" in row else row["f1_macro"])
    for value, name in ((accuracy, "accuracy"), (macro, "macro F1")):
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be finite and in [0, 1]")
    return accuracy, macro, int(round(accuracy * TEST_SUPPORT))


def plot_score_decomposition_and_ladder(
    base: dict[str, Any],
    visual: dict[str, Any],
    current: dict[str, Any],
    stages: dict[str, dict[str, Any]],
) -> Path:
    """Pairwise deltas, the image-only ladder, and bootstrap accuracy spread.

    None of these three panels restates a table: the pairwise decomposition
    replaces the removed appendix table, the ladder is untabulated, and only one
    of the three bootstrap intervals appears in the evidence ledger.
    """
    conditions = base["results"]["vlm"]["cross_modal_analysis"]["conditions"]
    exact = current["current_exact_pair"]

    def condition(name: str) -> tuple[float, float, int]:
        return _accuracy_pair(conditions[name])

    def system(name: str) -> tuple[float, float, int]:
        return _accuracy_pair(exact[name])

    contrasts = (
        ("DistilBERT $\\to$ ViT", system("text_only_llm"), system("image_only_vit")),
        ("ViT $\\to$ raw VLM", system("image_only_vit"), system("image_text_vlm")),
        (
            "Raw VLM $\\to$ proposed",
            system("image_text_vlm"),
            system("proposed_farmfederate"),
        ),
        (
            "Direct $\\to$ enhanced",
            condition("image_only"),
            _accuracy_pair(visual["fixed_test"]),
        ),
        ("Image $\\to$ paired", condition("image_only"), condition("paired")),
        ("Paired $\\to$ text", condition("paired"), condition("text_only")),
        ("Mismatch $\\to$ paired", condition("mismatched_text"), condition("paired")),
    )

    figure, (left, middle, right) = plt.subplots(
        1, 3, figsize=(7.16, 1.72), gridspec_kw={"wspace": 0.42}
    )

    # (a) Accuracy points gained, coloured by whether macro F1 moved with them.
    names = [row[0] for row in contrasts]
    delta_acc = np.asarray(
        [100.0 * (after[0] - before[0]) for _, before, after in contrasts]
    )
    delta_f1 = np.asarray(
        [after[1] - before[1] for _, before, after in contrasts]
    )
    delta_correct = np.asarray(
        [after[2] - before[2] for _, before, after in contrasts], dtype=int
    )
    y = np.arange(len(contrasts))
    left.barh(
        y,
        delta_acc,
        color=[FUSION_GREEN if value > 0 else PROPOSED for value in delta_f1],
        edgecolor="white",
        linewidth=0.5,
        height=0.68,
        zorder=3,
    )
    for position, points, f1_shift, correct in zip(
        y, delta_acc, delta_f1, delta_correct
    ):
        left.text(
            points + 1.4,
            position,
            f"+{correct}, {f1_shift:+.3f}",
            va="center",
            ha="left",
            fontsize=5.5,
            color=INK,
        )
    left.set_yticks(y)
    left.set_yticklabels(names, fontsize=5.8)
    left.invert_yaxis()
    left.set_xlim(0.0, 88.0)
    left.set_xlabel("Accuracy points gained")
    clean_axis(left, "x")

    # (b) Image-only ladder; every stage is an added component, not a rerun.
    ladder_labels = [row[0] for row in LADDER]
    ladder_acc = []
    for _, (source, key) in LADDER:
        row = conditions["image_only"] if source == "conditions" else stages[source][key]
        ladder_acc.append(_accuracy_pair(row)[0])
    ladder_acc = np.asarray(ladder_acc, dtype=float)
    colors = [TEXT_BLUE] * len(ladder_acc)
    colors[4] = PROPOSED
    colors[-1] = FUSION_GREEN
    bars = middle.bar(
        ladder_labels,
        ladder_acc,
        color=colors,
        edgecolor="white",
        linewidth=0.6,
        width=0.68,
        zorder=3,
    )
    for bar, value in zip(bars, ladder_acc):
        middle.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.018,
            f"{100.0 * value:.1f}",
            ha="center",
            va="bottom",
            fontsize=5.5,
        )
    middle.set_ylim(0.0, 0.92)
    middle.set_ylabel("Image-only accuracy")
    middle.tick_params(axis="x", labelrotation=38)
    clean_axis(middle, "y")

    # (c) Source-group bootstrap accuracy, 5,000 replicates over 38 sources.
    widths = tuple(round(100.0 * (high - low), 2) for _, _, low, high, _ in BOOTSTRAP)
    if widths != EXPECTED_WIDTHS:
        raise ValueError(
            f"Bootstrap interval widths {widths} disagree with the audited "
            f"values {EXPECTED_WIDTHS}"
        )
    positions = np.arange(len(BOOTSTRAP))
    for position, (label, point, low, high, color) in zip(positions, BOOTSTRAP):
        right.errorbar(
            [position],
            [point],
            yerr=[[point - low], [high - point]],
            fmt="o",
            color=INK,
            ecolor=color,
            elinewidth=2.0,
            capsize=3.4,
            capthick=0.95,
            markersize=3.6,
            zorder=4,
        )
        right.text(
            position + 0.14,
            point,
            f"{100.0 * point:.1f}",
            ha="left",
            va="center",
            fontsize=6.0,
            fontweight="bold",
        )
    right.set_xticks(positions)
    right.set_xticklabels([row[0] for row in BOOTSTRAP])
    right.set_xlim(-0.5, len(BOOTSTRAP) - 0.2)
    right.set_ylim(0.42, 1.03)
    right.set_ylabel("Accuracy")
    clean_axis(right, "y")

    for axis, tag in ((left, "(a)"), (middle, "(b)"), (right, "(c)")):
        axis.text(
            -0.10,
            1.04,
            tag,
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=6.7,
            fontweight="bold",
        )

    figure.subplots_adjust(left=0.115, right=0.995, bottom=0.30, top=0.90)
    save_panel_crops(
        figure,
        (left, middle, right),
        (
            "plot78a_score_gains.png",
            "plot78b_image_ladder.png",
            "plot78c_bootstrap_intervals.png",
        ),
    )
    return save(figure, "plot78_score_decomposition.png")


def plot_confusion_matrices(
    base: dict[str, Any],
    visual: dict[str, Any],
    hierarchy: dict[str, Any],
) -> Path:
    """Where errors actually land; tables record only diagonal correct counts."""
    conditions = base["results"]["vlm"]["cross_modal_analysis"]["conditions"]
    panels = (
        ("Direct image", conditions["image_only"]["confusion_matrix"]),
        ("Enhanced image", visual["fixed_test"]["confusion_matrix"]),
        (
            "Proposed router",
            hierarchy["fixed_test"]["proposed_multimodal"]["confusion_matrix"],
        ),
    )

    figure, axes = plt.subplots(
        1, 3, figsize=(7.16, 1.88), gridspec_kw={"wspace": 0.22}
    )
    global_max = max(
        float(np.asarray(matrix, dtype=float).max()) for _, matrix in panels
    )
    for axis, (title, matrix) in zip(axes, panels):
        counts = np.asarray(matrix, dtype=float)
        if counts.shape != (len(CLASS_LABELS), len(CLASS_LABELS)):
            raise ValueError(f"{title} matrix must be {len(CLASS_LABELS)}-square")
        if int(counts.sum()) != TEST_SUPPORT:
            raise ValueError(
                f"{title} matrix sums to {int(counts.sum())}, not {TEST_SUPPORT}"
            )
        axis.imshow(counts, cmap="Blues", vmin=0.0, vmax=global_max)
        for i in range(counts.shape[0]):
            for j in range(counts.shape[1]):
                value = int(counts[i, j])
                if value == 0:
                    continue
                axis.text(
                    j,
                    i,
                    str(value),
                    ha="center",
                    va="center",
                    fontsize=5.9,
                    fontweight="bold" if i == j else "normal",
                    color="white" if counts[i, j] > 0.6 * counts.max() else INK,
                )
        axis.set_xticks(range(len(CLASS_LABELS)))
        axis.set_yticks(range(len(CLASS_LABELS)))
        axis.set_xticklabels(CLASS_LABELS, rotation=38, fontsize=5.8)
        row_labels = [
            f"{label} ({int(total)})"
            for label, total in zip(CLASS_LABELS, counts.sum(axis=1))
        ]
        axis.set_yticklabels(row_labels, fontsize=5.6)
        axis.set_xlabel("Predicted", fontsize=6.4)
        axis.set_title(title, fontsize=6.6, fontweight="bold", pad=3)
        axis.tick_params(length=1.6, width=0.5, pad=1.2)
        for spine in axis.spines.values():
            spine.set_visible(False)
    axes[0].set_ylabel("True", fontsize=6.4)

    figure.subplots_adjust(left=0.088, right=0.995, bottom=0.235, top=0.86)
    return save(figure, "plot79_confusion_matrices.png")


def main() -> None:
    configure_style()
    legacy = load_json(LEGACY_RESULTS)
    plot_within_family(legacy)

    if not CURRENT_RESULTS.is_file():
        print(
            "Waiting for current exact-pair baseline results; plot76 was not "
            f"generated because {CURRENT_RESULTS} does not exist."
        )
        return

    current = load_json(CURRENT_RESULTS)
    federated = load_json(FEDERATED_RESULTS)
    encoder_selection = load_json(ENCODER_SELECTION_RESULTS)
    if encoder_selection.get("winner") != "resnet50_compact_transformer":
        raise ValueError("Unexpected validation-selected architecture winner")
    plot_current_and_federated(current, federated, encoder_selection)

    base = load_json(V6_RESULTS)
    visual = load_json(V6_VISION_ENSEMBLE)
    hierarchy = load_json(V6_HIERARCHY)
    stages = {name: load_json(path) for name, path in V6_STAGES.items()}
    stages["visual"] = visual
    plot_score_decomposition_and_ladder(base, visual, current, stages)
    plot_confusion_matrices(base, visual, hierarchy)


if __name__ == "__main__":
    main()
