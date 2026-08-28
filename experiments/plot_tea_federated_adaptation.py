"""Publication figures for the measured tea VLM federated-adaptation sweep."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator


ROOT = Path(__file__).resolve().parents[1]
RESULTS = (
    ROOT
    / "tea_results"
    / "federated_adaptation_v6"
    / "federated_adaptation_results.json"
)
OUTPUT = ROOT / "overleaf_final" / "plots"

MM = "#A61E4D"
BEST = "#087F5B"
NEUTRALS = {
    2: "#6C757D",
    3: MM,
    5: "#3B5BDB",
    8: "#E67700",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.6,
            "axes.labelsize": 8.2,
            "xtick.labelsize": 7.3,
            "ytick.labelsize": 7.3,
            "legend.fontsize": 6.7,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.65,
            "lines.markersize": 4.2,
            "grid.color": "#CED4DA",
            "grid.alpha": 0.55,
            "grid.linewidth": 0.45,
            "savefig.dpi": 400,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def clean_axis(axis) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", zorder=0)
    axis.tick_params(length=2.5, width=0.65)


def save(figure, filename: str) -> None:
    target = OUTPUT / filename
    figure.savefig(
        target,
        dpi=400,
        bbox_inches="tight",
        pad_inches=0.025,
        facecolor="white",
    )
    figure.savefig(
        target.with_suffix(".pdf"),
        bbox_inches="tight",
        pad_inches=0.025,
        facecolor="white",
    )
    plt.close(figure)


def client_count_figure(payload: dict) -> None:
    clients = [int(value) for value in payload["config"]["clients"]]
    summary = payload["summary"]
    final_mean = np.array(
        [summary[str(value)]["final_macro_f1_mean"] for value in clients]
    )
    best_mean = np.array(
        [summary[str(value)]["best_macro_f1_mean"] for value in clients]
    )
    baseline = float(payload["runs"][0]["baseline_validation"]["f1_macro"])

    figure, axis = plt.subplots(figsize=(2.28, 1.30))
    clean_axis(axis)
    axis.axhline(
        baseline,
        color="#868E96",
        linestyle=(0, (2, 2)),
        linewidth=1.05,
        label=f"Centralized {baseline:.3f}",
        zorder=1,
    )
    axis.plot(
        clients,
        final_mean,
        color=MM,
        marker="o",
        linewidth=2.2,
        label="FedAvg final mean",
        zorder=4,
    )
    axis.plot(
        clients,
        best_mean,
        color=BEST,
        marker="D",
        markerfacecolor="white",
        markeredgewidth=1.0,
        linestyle="--",
        linewidth=1.55,
        label="Best-round mean",
        zorder=3,
    )
    for x_value, y_value in zip(clients, final_mean):
        axis.annotate(
            f"{y_value:.3f}",
            (x_value, y_value),
            xytext=(0, -6),
            textcoords="offset points",
            ha="center",
            va="top",
            color=MM,
            fontsize=6.0,
        )
    axis.set_xlabel("Number of federated clients, $K$")
    axis.set_ylabel("Validation macro-F1")
    axis.set_xticks(clients)
    axis.set_xlim(min(clients) - 0.45, max(clients) + 0.45)
    axis.set_ylim(0.83, 0.985)
    axis.yaxis.set_major_locator(MultipleLocator(0.05))
    axis.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        frameon=False,
        borderpad=0,
        columnspacing=0.9,
        handlelength=1.5,
        handletextpad=0.4,
        labelspacing=0.25,
        fontsize=6.0,
    )
    figure.subplots_adjust(left=0.175, right=0.988, bottom=0.245, top=0.80)
    save(figure, "plot73_fl_clients_vs_performance.png")


def round_figure(payload: dict) -> None:
    clients = [int(value) for value in payload["config"]["clients"]]
    rounds = np.arange(1, int(payload["config"]["rounds"]) + 1)
    summary = payload["summary"]

    figure, axis = plt.subplots(figsize=(2.28, 1.30))
    clean_axis(axis)
    for client_count in clients:
        row = summary[str(client_count)]
        mean = np.asarray(row["round_mean_macro_f1"], dtype=float)
        color = NEUTRALS[client_count]
        deployed = client_count == 3
        axis.plot(
            rounds,
            mean,
            color=color,
            marker="o" if deployed else None,
            markersize=3.8 if deployed else 0,
            linewidth=2.35 if deployed else 1.15,
            linestyle="-" if deployed else "--",
            alpha=1.0 if deployed else 0.78,
            label=f"$K={client_count}$",
            zorder=4 if deployed else 2,
        )
    axis.set_xlabel("Communication round")
    axis.set_ylabel("Validation macro-F1")
    axis.set_xticks(rounds)
    axis.set_xlim(0.75, rounds[-1] + 0.25)
    axis.set_ylim(0.875, 0.965)
    axis.yaxis.set_major_locator(MultipleLocator(0.01))
    axis.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=4,
        frameon=False,
        borderpad=0,
        columnspacing=1.0,
        handlelength=1.6,
    )
    figure.subplots_adjust(left=0.155, right=0.988, bottom=0.235, top=0.855)
    save(figure, "plot74_fl_rounds_vs_performance.png")


def heterogeneity_figure(payload: dict) -> None:
    """Realized label skew per client count; the sweep's non-IID axis."""
    clients = [int(value) for value in payload["config"]["clients"]]
    summary = payload["summary"]
    variation = np.asarray(
        [float(summary[str(value)]["mean_label_total_variation"]) for value in clients],
        dtype=float,
    )
    if np.any(~np.isfinite(variation)) or np.any(variation < 0) or np.any(variation > 1):
        raise ValueError("Label total variation must be finite and within [0, 1]")

    figure, axis = plt.subplots(figsize=(2.28, 1.30))
    clean_axis(axis)
    bars = axis.bar(
        [str(value) for value in clients],
        variation,
        color=[NEUTRALS[value] for value in clients],
        edgecolor="white",
        linewidth=0.7,
        width=0.66,
        zorder=3,
    )
    for bar, value in zip(bars, variation):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.008,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=6.3,
            color=MM,
        )
    axis.set_xlabel("Number of federated clients, $K$")
    axis.set_ylabel("Mean label TV")
    axis.set_ylim(0.0, float(variation.max()) * 1.30)
    figure.subplots_adjust(left=0.20, right=0.985, bottom=0.265, top=0.965)
    save(figure, "plot77_fl_heterogeneity.png")


def main() -> None:
    configure_style()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    payload = json.loads(RESULTS.read_text(encoding="utf-8"))
    if payload.get("experiment") != "validation_only_federated_adaptation":
        raise RuntimeError("Unexpected experiment provenance")
    client_count_figure(payload)
    round_figure(payload)
    heterogeneity_figure(payload)
    print(OUTPUT / "plot73_fl_clients_vs_performance.png")
    print(OUTPUT / "plot74_fl_rounds_vs_performance.png")
    print(OUTPUT / "plot77_fl_heterogeneity.png")


if __name__ == "__main__":
    main()
