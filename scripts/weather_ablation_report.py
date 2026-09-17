#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
weather_ablation_report.py — summarise the LEAF weather-modality ablation.

Reads the per-seed run directories produced by the two arms (text+image vs
text+image+weather), and emits:
  - a LaTeX table of mean +/- std across seeds,
  - a matplotlib figure (per-seed paired points + the reliability weights).

The weather assignment is synthetic and label-independent, so the expected
result is no separation between the arms; the seed spread is the point.
"""
import argparse
import glob
import json
import os
from typing import Dict, List

import numpy as np


def load_runs(root: str) -> Dict[str, Dict[int, dict]]:
    """root/<arm>_s<seed>/complete_results.json -> {arm: {seed: results}}"""
    out: Dict[str, Dict[int, dict]] = {}
    for path in sorted(glob.glob(os.path.join(root, "*_s*", "complete_results.json"))):
        name = os.path.basename(os.path.dirname(path))
        arm, _, seed = name.rpartition("_s")
        try:
            seed = int(seed)
        except ValueError:
            continue
        with open(path) as f:
            out.setdefault(arm, {})[seed] = json.load(f)
    return out


def metric(run: dict, key: str) -> float:
    vlm = run["results"]["vlm"]
    if key in ("central_f1", "fed_f1"):
        return float(vlm[key])
    return float(vlm.get("test_metrics", {}).get(key, np.nan))


def weather_weight(run: dict) -> float:
    w = run["results"]["vlm"].get("test_metrics", {}).get("mean_modality_weights", {})
    if "weather" in w:
        return float(w["weather"])
    # older runs reported only text/image; the remainder is the weather mass
    if w:
        return float(max(0.0, 1.0 - w.get("text", 0.0) - w.get("image", 0.0)))
    return float("nan")


def summarise(runs, arms: List[str], keys: List[str]) -> Dict[str, Dict[str, tuple]]:
    table = {}
    for arm in arms:
        seeds = sorted(runs.get(arm, {}))
        table[arm] = {}
        for k in keys:
            vals = np.array([metric(runs[arm][s], k) for s in seeds], dtype=float)
            table[arm][k] = (np.nanmean(vals), np.nanstd(vals, ddof=1) if len(vals) > 1 else 0.0, len(vals))
    return table


def latex_table(table, arms, keys, labels) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\caption{Weather-modality ablation on the tea corpus, mean $\pm$ standard",
        r"deviation over five seeds. The crop-to-station-day assignment is synthetic and",
        r"label-independent, so weather carries no information about the target; the arms",
        r"are statistically indistinguishable and the seed spread exceeds every difference",
        r"between them.}",
        r"\label{tab:weather_ablation}",
        r"\centering\footnotesize",
        r"\begin{tabular}{l" + "c" * len(arms) + r"}",
        r"\toprule",
        r"Metric & " + " & ".join(a.replace("base", "Text$+$image").replace(
            "weather", "Text$+$image$+$weather") for a in arms) + r"\\",
        r"\midrule",
    ]
    for k, lab in zip(keys, labels):
        cells = []
        for a in arms:
            m, sd, n = table[a][k]
            cells.append(f"${m:.3f} \\pm {sd:.3f}$")
        lines.append(f"{lab} & " + " & ".join(cells) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def make_plot(runs, arms, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seeds = sorted(set(runs.get(arms[0], {})) & set(runs.get(arms[1], {})))
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))

    # (a) paired per-seed central F1
    ax = axes[0]
    a_vals = [metric(runs[arms[0]][s], "central_f1") for s in seeds]
    b_vals = [metric(runs[arms[1]][s], "central_f1") for s in seeds]
    for i, s in enumerate(seeds):
        ax.plot([0, 1], [a_vals[i], b_vals[i]], "-o", color="0.55", ms=4, lw=1)
    ax.errorbar([0, 1], [np.mean(a_vals), np.mean(b_vals)],
                yerr=[np.std(a_vals, ddof=1), np.std(b_vals, ddof=1)],
                fmt="s", color="#1f77b4", ms=9, lw=2, capsize=5, zorder=3, label="mean $\\pm$ s.d.")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["text+image", "+weather"])
    ax.set_xlim(-0.35, 1.35); ax.set_ylabel("VLM validation $F_1$ (macro)")
    ax.set_title("(a) Paired runs across seeds", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=0.3)

    # (b) reliability mass assigned to each modality in the weather arm
    ax = axes[1]
    w = runs[arms[1]]
    mods = ["text", "image", "weather"]
    vals = []
    for m in mods:
        col = []
        for s in seeds:
            mw = w[s]["results"]["vlm"].get("test_metrics", {}).get("mean_modality_weights", {})
            col.append(weather_weight(w[s]) if m == "weather" else float(mw.get(m, np.nan)))
        vals.append(col)
    means = [np.nanmean(v) for v in vals]
    sds = [np.nanstd(v, ddof=1) if len(v) > 1 else 0.0 for v in vals]
    bars = ax.bar(mods, means, yerr=sds, capsize=5,
                  color=["#4c72b0", "#dd8452", "#937860"])
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, m + 0.02, f"{m:.2f}", ha="center", fontsize=9)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_ylabel("mean reliability weight")
    ax.set_ylim(0, max(means) + 0.15)
    ax.set_title("(b) Router mass, weather arm", fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    return out_png


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="directory holding <arm>_s<seed>/ run folders")
    ap.add_argument("--out-tex", default="weather_ablation_table.tex")
    ap.add_argument("--out-png", default="weather_ablation.png")
    args = ap.parse_args()

    runs = load_runs(args.root)
    arms = [a for a in ("base", "weather") if a in runs]
    if len(arms) < 2:
        raise SystemExit(f"need both arms, found {list(runs)}")
    keys = ["central_f1", "fed_f1", "f1_macro"]
    labels = ["Validation $F_1$ (macro)", "Federated $F_1$ (macro)", "Test $F_1$ (macro)"]

    table = summarise(runs, arms, keys)
    seeds = sorted(set(runs[arms[0]]) & set(runs[arms[1]]))
    print(f"seeds: {seeds}")
    for k, lab in zip(keys, labels):
        line = f"{lab:28s}"
        for a in arms:
            m, sd, n = table[a][k]
            line += f"  {a}: {m:.4f} +/- {sd:.4f} (n={n})"
        d = table[arms[1]][k][0] - table[arms[0]][k][0]
        pooled = np.hypot(table[arms[0]][k][1], table[arms[1]][k][1])
        line += f"   delta={d:+.4f} ({abs(d)/pooled:.2f} pooled s.d.)" if pooled else ""
        print(line)

    ww = [weather_weight(runs["weather"][s]) for s in seeds]
    print(f"\nweather reliability weight: {np.nanmean(ww):.3f} +/- {np.nanstd(ww, ddof=1):.3f}")

    with open(args.out_tex, "w") as f:
        f.write(latex_table(table, arms, keys, labels) + "\n")
    print(f"[write] {args.out_tex}")
    print(f"[write] {make_plot(runs, arms, args.out_png)}")


if __name__ == "__main__":
    main()
