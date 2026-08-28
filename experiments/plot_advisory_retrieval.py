"""Figure for the measured Classify-Retrieve-Advise advisory evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = (
    ROOT / "tea_results" / "advisory_retrieval_v1" / "advisory_retrieval_results.json"
)
OUTPUT = ROOT / "overleaf_final" / "plots"

INK = "#161616"
MUTED = "#6C757D"
GRID = "#DDE1E5"
OPEN = "#2878B5"
ROUTED = "#A61E4D"
CHANCE = "#868E96"

SHORT = {
    "LEAF_BLIGHT": "Blight",
    "LEAF_HOPPERS": "Hoppers",
    "LEAF_RUST": "Rust",
    "LOOPER_CATERPILLARS": "Looper",
    "MOSQUITO_BUG": "Mosq. bug",
}


def configure() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.2,
            "axes.labelsize": 7.4,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.4,
            "legend.fontsize": 6.3,
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


def clean(axis: plt.Axes, which: str = "y") -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis=which, zorder=0)
    axis.set_axisbelow(True)
    axis.tick_params(length=2.2, width=0.6, pad=1.8)


def main() -> None:
    configure()
    payload: dict[str, Any] = json.loads(RESULTS.read_text(encoding="utf-8"))
    if payload.get("experiment") != "advisory_classify_retrieve_advise":
        raise RuntimeError("Unexpected advisory result provenance")

    open_route = payload["retrieve_only"]
    routed = payload["classify_then_retrieve"]
    chance = float(payload["chance_precision_at_1"])
    classes = list(payload["class_order"])

    figure, (left, middle, right) = plt.subplots(
        1, 3, figsize=(7.16, 1.42), gridspec_kw={"wspace": 0.52}
    )

    # (a) Headline advisory correctness with bootstrap intervals. Point and
    # whisker marks make the uncertainty visible without a near-ceiling bar.
    labels = ["Retrieve\nonly", "Classify then\nretrieve"]
    values = np.array([open_route["precision_at_1"], routed["precision_at_1"]])
    lows = np.array(
        [open_route["precision_at_1_ci95"][0], routed["precision_at_1_ci95"][0]]
    )
    highs = np.array(
        [open_route["precision_at_1_ci95"][1], routed["precision_at_1_ci95"][1]]
    )
    x = np.arange(len(labels))
    for position, value, low, high, color in zip(
        x, values, lows, highs, (OPEN, ROUTED)
    ):
        left.errorbar(
            [position],
            [value],
            yerr=[[value - low], [high - value]],
            fmt="o",
            color=color,
            ecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.55,
            markersize=5.5,
            elinewidth=1.8,
            capsize=3.2,
            capthick=0.9,
            zorder=5,
        )
        left.text(
            position,
            high + 0.045,
            f"{100.0 * value:.1f}",
            ha="center",
            va="bottom",
            fontsize=6.2,
            fontweight="bold",
        )
    left.axhline(
        chance, color=CHANCE, linestyle=(0, (2, 2)), linewidth=1.0, zorder=2
    )
    # Labelled inline: a legend entry here would sit on the chance line itself.
    left.set_xlim(-0.52, 1.52)
    left.text(
        0.5,
        chance + 0.03,
        f"chance {100.0 * chance:.0f}%",
        ha="center",
        va="bottom",
        fontsize=5.3,
        color=CHANCE,
    )
    left.set_xticks(x)
    left.set_xticklabels(labels)
    left.set_ylim(0.0, 1.12)
    left.set_ylabel("Advisory precision@1")
    clean(left)

    # (b) Split by whether the query carries a uniquely class-specific sentence.
    specific_count = int(
        round(payload["queries"] * payload["queries_with_class_specific_sentence"])
    )
    generic_count = int(payload["queries"] - specific_count)
    groups = [
        f"Class-specific\nsentence ($n={specific_count}$)",
        f"Generic only\n($n={generic_count}$)",
    ]
    open_vals = [
        open_route["precision_at_1_with_specific_sentence"],
        open_route["precision_at_1_generic_only"],
    ]
    routed_vals = [
        routed["precision_at_1_with_specific_sentence"],
        routed["precision_at_1_generic_only"],
    ]
    xs = np.arange(len(groups))
    for position, open_value, routed_value in zip(xs, open_vals, routed_vals):
        middle.plot(
            [position - 0.11, position + 0.11],
            [open_value, routed_value],
            color=MUTED,
            linewidth=1.2,
            zorder=2,
        )
        for offset, value, color in (
            (-0.11, open_value, OPEN),
            (0.11, routed_value, ROUTED),
        ):
            middle.scatter(
                position + offset,
                value,
                s=25,
                color=color,
                edgecolor="white",
                linewidth=0.5,
                zorder=4,
            )
            middle.text(
                position + offset,
                value + 0.02,
                f"{100.0 * value:.1f}",
                ha="center",
                va="bottom",
                fontsize=5.8,
            )
    middle.set_xticks(xs)
    middle.set_xticklabels(groups)
    middle.set_ylim(0.60, 1.27)
    middle.set_ylabel("Advisory precision@1")
    middle.legend(
        handles=[
            plt.Line2D(
                [0], [0], marker="o", color="none", markerfacecolor=OPEN,
                markeredgecolor="white", label="Retrieve only"
            ),
            plt.Line2D(
                [0], [0], marker="o", color="none", markerfacecolor=ROUTED,
                markeredgecolor="white", label="Classify then retrieve"
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=2,
        frameon=False,
        columnspacing=0.7,
        handlelength=0.9,
        handletextpad=0.4,
        fontsize=5.9,
    )
    clean(middle)

    # (c) Per class; routing helps exactly where open retrieval is weakest.
    per_class = payload["per_class"]
    names = [f"{SHORT[c]} ({int(per_class[c]['queries'])})" for c in classes]
    open_c = [per_class[c]["precision_at_1_open"] for c in classes]
    routed_c = [per_class[c]["precision_at_1_routed"] for c in classes]
    y = np.arange(len(classes))
    for position, open_value, routed_value in zip(y, open_c, routed_c):
        right.hlines(
            position,
            min(open_value, routed_value),
            max(open_value, routed_value),
            color=MUTED,
            linewidth=1.25,
            zorder=2,
        )
    right.scatter(
        open_c, y, s=23, color=OPEN, edgecolor="white", linewidth=0.5, zorder=4
    )
    right.scatter(
        routed_c,
        y,
        s=23,
        color=ROUTED,
        edgecolor="white",
        linewidth=0.5,
        zorder=4,
    )
    right.axvline(chance, color=CHANCE, linestyle=(0, (2, 2)), linewidth=1.0, zorder=2)
    right.set_yticks(y)
    right.set_yticklabels(names)
    right.invert_yaxis()
    right.set_xlim(0.0, 1.09)
    right.set_xlabel("Advisory precision@1")
    clean(right, "x")

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

    figure.subplots_adjust(left=0.075, right=0.995, bottom=0.30, top=0.90)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    target = OUTPUT / "plot80_advisory_retrieval.png"
    figure.savefig(
        target, dpi=400, bbox_inches="tight", pad_inches=0.025, facecolor="white"
    )
    figure.savefig(
        target.with_suffix(".pdf"),
        bbox_inches="tight",
        pad_inches=0.025,
        facecolor="white",
    )
    plt.close(figure)
    print(target)


if __name__ == "__main__":
    main()
