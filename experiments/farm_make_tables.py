"""
Turn farm_results.json into LaTeX fragments, figures, and a plain-language
readout.

Run after farm_ablation.py finishes (or after any chunk of it):

    python farm_make_tables.py --results farm_results.json --outdir paper_assets

Read paper_assets/SUMMARY.md before pasting anything into the paper. It states
what each result means, including the cases where a result argues against a
claim the paper currently makes. Tables are generated only for experiments that
have data, so partial runs still produce usable output.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
def _f(x, nd=3):
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return "--"


def _mean_std(values: List[float]):
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    if arr.size == 0:
        return None, None
    return float(arr.mean()), float(arr.std())


def _table(caption: str, label: str, header: str, rows: List[str],
           colspec: str, note: str = "", tabcolsep: str = "5pt") -> str:
    body = "\n".join(rows)
    note_block = ""
    if note:
        note_block = ("\\vspace{1pt}\n\n{\\scriptsize " + note + "}\n")
    return (
        "\\begin{table}[t]\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        "\\centering\\footnotesize\n"
        f"\\setlength\\tabcolsep{{{tabcolsep}}}\n"
        f"\\begin{{tabular}}{{{colspec}}}\n"
        "\\toprule\n"
        f"{header}\\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        f"{note_block}"
        "\\end{table}\n"
    )


# ---------------------------------------------------------------------------
# E1 -- the headline: alpha sweep under the corrected vs legacy splitter
# ---------------------------------------------------------------------------
def build_alpha_split(data: dict, out: Path, summary: List[str]) -> Optional[str]:
    exp = data.get("E1_alpha_sweep") or {}
    if not exp:
        return None

    # alpha -> splitter -> list of (f1, tv, cv)
    grid: Dict[float, Dict[str, List[dict]]] = defaultdict(lambda: defaultdict(list))
    for key, rec in exp.items():
        if "error" in rec:
            continue
        alpha = rec.get("alpha")
        split = rec.get("splitter", "corrected")
        grid[float(alpha)][split].append(rec)

    rows = []
    for alpha in sorted(grid):
        cells = []
        for split in ("corrected", "legacy"):
            recs = grid[alpha].get(split, [])
            if not recs:
                cells += ["--", "--", "--"]
                continue
            f1_m, f1_s = _mean_std([r.get("f1_macro") for r in recs])
            tv_m, _ = _mean_std([r.get("partition", {}).get("label_tv_mean") for r in recs])
            cv_m, _ = _mean_std([r.get("partition", {}).get("size_cv") for r in recs])
            f1_cell = _f(f1_m) if f1_s in (None, 0.0) else f"{_f(f1_m)}\\,$\\pm$\\,{_f(f1_s)}"
            cells += [_f(tv_m, 3), _f(cv_m, 2), f1_cell]
        rows.append(f"$\\alpha={alpha:g}$ & " + " & ".join(cells) + " \\\\")

    header = ("\\textbf{Partition} & \\multicolumn{3}{c}{\\textbf{Class-wise Dirichlet}} "
              "& \\multicolumn{3}{c}{\\textbf{Legacy size-only}}\\\\\n"
              "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\n"
              " & TV & size CV & Macro F1 & TV & size CV & Macro F1")
    tex = _table(
        caption=("Realised heterogeneity and accuracy under the two partitioners. "
                 "TV is the mean total-variation distance between a client's class "
                 "distribution and the global one; size CV is the coefficient of "
                 "variation of shard sizes. The legacy partitioner varies shard "
                 "size while leaving label mix close to IID, so its accuracy is "
                 "flat in $\\alpha$ by construction."),
        label="tab:alpha_split", header=header, rows=rows,
        # seven columns in an IEEE single column: keep the padding tight or the
        # table overflows the column measure
        colspec="lcccccc", tabcolsep="2.2pt",
        note=("Both columns use identical models, budgets and seeds; only the "
              "partitioning rule differs. Report the TV column alongside "
              "$\\alpha$: $\\alpha$ alone does not establish that clients were "
              "non-IID."))
    (out / "tab_alpha_split.tex").write_text(tex, encoding="utf8")

    # -- readout ---------------------------------------------------------
    summary.append("## E1 alpha sweep -- corrected vs legacy partitioner\n")
    for alpha in sorted(grid):
        parts = []
        for split in ("corrected", "legacy"):
            recs = grid[alpha].get(split, [])
            if not recs:
                continue
            f1_m, _ = _mean_std([r.get("f1_macro") for r in recs])
            tv_m, _ = _mean_std([r.get("partition", {}).get("label_tv_mean") for r in recs])
            parts.append(f"{split}: TV={_f(tv_m)} F1={_f(f1_m)}")
        summary.append(f"- alpha={alpha:g} -- " + " | ".join(parts))

    corr = {a: _mean_std([r.get("f1_macro") for r in grid[a].get("corrected", [])])[0]
            for a in sorted(grid) if grid[a].get("corrected")}
    leg = {a: _mean_std([r.get("f1_macro") for r in grid[a].get("legacy", [])])[0]
           for a in sorted(grid) if grid[a].get("legacy")}
    if len(corr) >= 2:
        spread_c = max(corr.values()) - min(corr.values())
        summary.append(f"\n- Corrected partitioner: F1 varies by {_f(spread_c)} across the "
                       f"alpha range. This is the number that supports (or refuses to "
                       f"support) a robustness-to-non-IID claim.")
    if len(leg) >= 2:
        spread_l = max(leg.values()) - min(leg.values())
        summary.append(f"- Legacy partitioner: F1 varies by {_f(spread_l)}. If this is "
                       f"near zero while TV is also near zero, the flatness is an "
                       f"artefact of the splitter, not evidence of robustness. Do not "
                       f"cite legacy-split numbers as non-IID results.")
    summary.append("")
    return "tab_alpha_split.tex"


# ---------------------------------------------------------------------------
def build_client_sweep(data: dict, out: Path, summary: List[str]) -> Optional[str]:
    exp = data.get("E2_client_sweep") or {}
    if not exp:
        return None
    by_k = defaultdict(list)
    for rec in exp.values():
        if "error" not in rec:
            by_k[int(rec.get("num_clients", 0))].append(rec)

    rows = []
    for k in sorted(by_k):
        recs = by_k[k]
        f1_m, _ = _mean_std([r.get("f1_macro") for r in recs])
        tv_m, _ = _mean_std([r.get("partition", {}).get("label_tv_mean") for r in recs])
        comm = _mean_std([r.get("total_comm_bytes") for r in recs])[0] or 0.0
        secs = _mean_std([r.get("wall_seconds") for r in recs])[0] or 0.0
        mem = _mean_std([r.get("peak_mib") for r in recs])[0] or 0.0
        rows.append(f"{k} & {_f(tv_m)} & {_f(f1_m)} & {comm/2**20:.0f} & "
                    f"{secs/60:.1f} & {mem:.0f} \\\\")

    tex = _table(
        caption=("Scalability in the number of clients $K$ at fixed $\\alpha$. "
                 "Communication is the measured round-trip volume over the whole "
                 "run; wall-clock and peak memory are measured on the training GPU."),
        label="tab:client_sweep",
        header="\\textbf{$K$} & TV & \\textbf{Macro F1} & Comm.\\ (MiB) & Time (min) & Peak (MiB)",
        rows=rows, colspec="lccccc",
        note="Total data and all other hyperparameters are held fixed as $K$ grows, "
             "so each client's shard shrinks as $K$ increases.")
    (out / "tab_client_sweep.tex").write_text(tex, encoding="utf8")

    summary.append("## E2 client-count scalability\n")
    for k in sorted(by_k):
        f1_m, _ = _mean_std([r.get("f1_macro") for r in by_k[k]])
        summary.append(f"- K={k}: Macro F1 {_f(f1_m)}")
    summary.append("- Read the drop from smallest to largest K as the cost of "
                   "splitting fixed data across more sites, not as a property of "
                   "the aggregation rule.\n")
    return "tab_client_sweep.tex"


# ---------------------------------------------------------------------------
def build_anticollapse(data: dict, out: Path, summary: List[str]) -> Optional[str]:
    exp = data.get("E3_anticollapse") or {}
    if not exp:
        return None
    by_alpha = defaultdict(dict)
    for rec in exp.values():
        if "error" in rec:
            continue
        by_alpha[float(rec.get("alpha", 1.0))][rec.get("arm")] = rec

    order = [("both", "Both (full system)"), ("no_diversity", "-- diversity loss"),
             ("no_balanced", "-- balanced sampler"), ("neither", "Neither")]
    alphas = sorted(by_alpha)
    rows = []
    for arm, pretty in order:
        cells = []
        for alpha in alphas:
            rec = by_alpha[alpha].get(arm)
            if not rec:
                cells += ["--", "--"]
                continue
            div = rec.get("diversity_ratio")
            cells += [_f(rec.get("f1_macro")),
                      "--" if div is None else f"{float(div)*100:.0f}\\%"]
        rows.append(f"{pretty} & " + " & ".join(cells) + " \\\\")

    head = " & ".join(f"\\multicolumn{{2}}{{c}}{{$\\alpha={a:g}$}}" for a in alphas)
    sub = " & ".join("F1 & div." for _ in alphas)
    tex = _table(
        caption=("Anti-collapse components under federation on the multimodal "
                 "branch. ``div.'' is the fraction of the five classes the model "
                 "actually predicts on validation; a collapsed model scores high "
                 "micro-F1 while predicting one or two classes."),
        label="tab:anticollapse",
        header=f"\\textbf{{Configuration}} & {head}\\\\\n & {sub}",
        rows=rows, colspec="l" + "cc" * len(alphas),
        note="The repository's own run\\_ablation\\_study asks this question "
             "centrally and text-only; collapse is a federated, class-imbalanced "
             "failure, so it is measured here under federation.")
    (out / "tab_anticollapse.tex").write_text(tex, encoding="utf8")

    summary.append("## E3 anti-collapse components\n")
    for alpha in alphas:
        for arm, pretty in order:
            rec = by_alpha[alpha].get(arm)
            if rec:
                summary.append(f"- alpha={alpha:g} {pretty}: F1={_f(rec.get('f1_macro'))}, "
                               f"diversity={_f(rec.get('diversity_ratio'), 2)}")
    neither = [by_alpha[a].get("neither") for a in alphas if by_alpha[a].get("neither")]
    both = [by_alpha[a].get("both") for a in alphas if by_alpha[a].get("both")]
    if neither and both:
        n_div = [r.get("diversity_ratio") for r in neither if r.get("diversity_ratio")]
        b_div = [r.get("diversity_ratio") for r in both if r.get("diversity_ratio")]
        if n_div and b_div and min(n_div) >= 0.8:
            summary.append("\n- IMPORTANT: the 'Neither' arm did NOT collapse "
                           "(diversity stayed high without either component). The "
                           "anti-collapse stack is then not doing the work the paper "
                           "attributes to it on this data. Weaken the claim to match "
                           "the measurement rather than re-running until it agrees.")
        else:
            summary.append("\n- The 'Neither' arm degrades as expected, which is what "
                           "licenses describing the stack as necessary.")
    summary.append("")
    return "tab_anticollapse.tex"


# ---------------------------------------------------------------------------
def build_warmstart(data: dict, out: Path, summary: List[str]) -> Optional[str]:
    exp = data.get("E4_warmstart") or {}
    if not exp:
        return None
    by_mode = defaultdict(list)
    for rec in exp.values():
        if "error" not in rec:
            by_mode[rec.get("mode")].append(rec)

    rows = []
    for mode, pretty in (("warm", "Warm start (centralized init)"),
                         ("cold", "Cold start (random init)")):
        recs = by_mode.get(mode, [])
        if not recs:
            continue
        f1_m, f1_s = _mean_std([r.get("f1_macro") for r in recs])
        hist = [r.get("history", {}).get("round_f1_macro", []) for r in recs]
        r1 = _mean_std([h[0] for h in hist if h])[0]
        cell = _f(f1_m) if not f1_s else f"{_f(f1_m)}\\,$\\pm$\\,{_f(f1_s)}"
        rows.append(f"{pretty} & {_f(r1)} & {cell} \\\\")

    tex = _table(
        caption=("Effect of initializing federated training from a centralized "
                 "checkpoint. The round-1 column shows how much of the benefit is "
                 "simply a better starting point."),
        label="tab:warmstart",
        header="\\textbf{Initialization} & Round 1 F1 & \\textbf{Best Macro F1}",
        rows=rows, colspec="lcc",
        note="Both arms share the partition, budget and seeds; only the initial "
             "weights differ.")
    (out / "tab_warmstart.tex").write_text(tex, encoding="utf8")

    summary.append("## E4 warm start vs cold start\n")
    w = _mean_std([r.get("f1_macro") for r in by_mode.get("warm", [])])[0]
    c = _mean_std([r.get("f1_macro") for r in by_mode.get("cold", [])])[0]
    if w is not None and c is not None:
        summary.append(f"- warm {_f(w)} vs cold {_f(c)} (gap {_f(w - c)})")
        if abs(w - c) < 0.01:
            summary.append("- The gap is within noise. Warm starting is then a "
                           "convenience, not a contribution -- describe it that way.")
    summary.append("")
    return "tab_warmstart.tex"


# ---------------------------------------------------------------------------
def build_fusion_variance(data: dict, out: Path, summary: List[str]) -> Optional[str]:
    exp = data.get("E5_fusion_seeds") or {}
    if not exp:
        return None
    by_fusion = defaultdict(list)
    for rec in exp.values():
        if "error" in rec:
            continue
        by_fusion[rec.get("fusion_type")].append(rec)

    rows = []
    ranked = []
    for fusion, recs in by_fusion.items():
        f1_m, f1_s = _mean_std([r.get("f1_macro") for r in recs])
        if f1_m is None:
            continue
        ranked.append((f1_m, f1_s, fusion, len(recs)))
    for f1_m, f1_s, fusion, n in sorted(ranked, reverse=True):
        rows.append(f"{fusion.capitalize()} & {n} & {_f(f1_m)}\\,$\\pm$\\,{_f(f1_s)} \\\\")

    tex = _table(
        caption=("Fusion strategies across seeds. The spread is what determines "
                 "whether a ranking between strategies is real."),
        label="tab:fusion_variance",
        header="\\textbf{Fusion} & Seeds & \\textbf{Macro F1 (mean\\,$\\pm$\\,std)}",
        rows=rows, colspec="lcc",
        note="Single-run differences smaller than the seed spread are not "
             "evidence that one fusion strategy beats another.")
    (out / "tab_fusion_variance.tex").write_text(tex, encoding="utf8")

    summary.append("## E5 fusion variance\n")
    for f1_m, f1_s, fusion, n in sorted(ranked, reverse=True):
        summary.append(f"- {fusion}: {_f(f1_m)} +/- {_f(f1_s)} over {n} seeds")
    if len(ranked) >= 2:
        top, second = sorted(ranked, reverse=True)[:2]
        gap = top[0] - second[0]
        noise = max(top[1] or 0.0, second[1] or 0.0)
        if gap < noise:
            summary.append(f"\n- The gap between {top[2]} and {second[2]} ({_f(gap)}) is "
                           f"smaller than the seed spread ({_f(noise)}). Do not claim "
                           f"one is better; report them as indistinguishable here.")
    summary.append("")
    return "tab_fusion_variance.tex"


# ---------------------------------------------------------------------------
def build_cost(data: dict, out: Path, summary: List[str]) -> Optional[str]:
    exp = data.get("E6_cost") or {}
    if not exp:
        return None
    rows = []
    for rec in exp.values():
        if "error" in rec:
            continue
        rows.append(f"{str(rec.get('fusion_type','')).capitalize()} & "
                    f"{rec.get('num_params',0)/1e6:.1f} & "
                    f"{rec.get('upload_mib_per_client_per_round',0):.1f} & "
                    f"{2*rec.get('upload_mib_per_client_per_round',0):.1f} \\\\")
    if not rows:
        return None
    tex = _table(
        caption="Per-client communication cost of one federated round.",
        label="tab:cost",
        header=("\\textbf{Variant} & Params (M) & Upload (MiB) & Round trip (MiB)"),
        rows=rows, colspec="lccc",
        note="Sizes are exact for float32 trainable parameters. Measured "
             "wall-clock and peak memory per round are recorded in the E1/E2/E8 "
             "histories in the results file.")
    (out / "tab_cost.tex").write_text(tex, encoding="utf8")
    summary.append("## E6 communication cost\n")
    for rec in exp.values():
        if "error" not in rec:
            summary.append(f"- {rec.get('fusion_type')}: "
                           f"{rec.get('upload_mib_per_client_per_round',0):.1f} MiB "
                           f"per client per round")
    summary.append("")
    return "tab_cost.tex"


# ---------------------------------------------------------------------------
def build_baselines(data: dict, out: Path, summary: List[str]) -> Optional[str]:
    exp = data.get("E8_baselines") or {}
    if not exp:
        return None

    alphas, by_algo = set(), defaultdict(lambda: defaultdict(list))
    local = {}
    for key, rec in exp.items():
        if "error" in rec:
            continue
        algo = key.split("|")[0]
        alpha = float(rec.get("alpha", 1.0))
        alphas.add(alpha)
        if algo == "local_only":
            local[alpha] = rec
        else:
            by_algo[algo][alpha].append(rec)

    alphas = sorted(alphas)
    rows = []
    if local:
        cells = [_f(local[a].get("mean_f1")) if a in local else "--" for a in alphas]
        rows.append("Local only (no federation) & " + " & ".join(cells) + " \\\\")
        rows.append("\\midrule")

    pretty = {"fedavg": "FedAvg", "fedprox": "FedProx",
              "scaffold": "SCAFFOLD", "fedbn": "FedBN"}
    for algo in ("fedavg", "fedprox", "scaffold", "fedbn"):
        if algo not in by_algo:
            continue
        cells = []
        for a in alphas:
            recs = by_algo[algo].get(a, [])
            if not recs:
                cells.append("--")
                continue
            m, s = _mean_std([r.get("f1_macro") for r in recs])
            cells.append(_f(m) if not s else f"{_f(m)}\\,$\\pm$\\,{_f(s)}")
        rows.append(f"{pretty[algo]} & " + " & ".join(cells) + " \\\\")

    header = "\\textbf{Method} & " + " & ".join(f"$\\alpha={a:g}$" for a in alphas)
    tex = _table(
        caption=("Federated aggregation rules under \\emph{matched} settings: same "
                 "multimodal backbone, data, partition, local budget and "
                 "hyperparameters, so the aggregation rule is the only variable. "
                 "Mean\\,$\\pm$\\,std over seeds."),
        label="tab:baselines", header=header, rows=rows,
        colspec="l" + "c" * len(alphas),
        note="Local-only is the mean over clients, each trained on its own shard "
             "alone at matched total local epochs. Every other row is the global "
             "model. Partitions use the class-wise Dirichlet splitter.")
    (out / "tab_baselines.tex").write_text(tex, encoding="utf8")

    summary.append("## E8 matched-setting baselines\n")
    for a in alphas:
        bits = []
        if a in local:
            bits.append(f"local_only {_f(local[a].get('mean_f1'))}")
        for algo in ("fedavg", "fedprox", "scaffold", "fedbn"):
            recs = by_algo.get(algo, {}).get(a, [])
            if recs:
                m, _ = _mean_std([r.get("f1_macro") for r in recs])
                bits.append(f"{algo} {_f(m)}")
        summary.append(f"- alpha={a:g}: " + ", ".join(bits))

    summary.append("\n- This is the comparison reviewers asked for, and the only one "
                   "in the paper run under matched conditions. Two readings matter: "
                   "how far every federated arm sits above local-only (that is what "
                   "federation buys), and whether the drift-correcting rules separate "
                   "from FedAvg at low alpha.")
    low = min(alphas) if alphas else None
    if low is not None and by_algo.get("fedavg", {}).get(low):
        base, _ = _mean_std([r.get("f1_macro") for r in by_algo["fedavg"][low]])
        winners = []
        for algo in ("fedprox", "scaffold", "fedbn"):
            recs = by_algo.get(algo, {}).get(low, [])
            if not recs:
                continue
            m, _ = _mean_std([r.get("f1_macro") for r in recs])
            if m is not None and base is not None and m > base + 0.005:
                winners.append(f"{algo} ({_f(m)} vs {_f(base)})")
        if winners:
            summary.append(f"- At alpha={low:g}, these beat FedAvg: {', '.join(winners)}. "
                           f"Switch the aggregator and say so -- do not keep FedAvg "
                           f"because it is what the submitted version used.")
        else:
            summary.append(f"- At alpha={low:g} nothing beat FedAvg by a clear margin. "
                           f"That is a legitimate, reportable result: state that the "
                           f"simplest rule sufficed at this scale.")
    summary.append("")
    return "tab_baselines.tex"


# ---------------------------------------------------------------------------
def build_alpha_figure(data: dict, out: Path) -> Optional[str]:
    exp = data.get("E1_alpha_sweep") or {}
    if not exp:
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    grid = defaultdict(lambda: defaultdict(list))
    for rec in exp.values():
        if "error" in rec:
            continue
        grid[rec.get("splitter", "corrected")][float(rec.get("alpha"))].append(rec)

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.0), dpi=300)
    display_names = {
        "corrected": "class-wise Dirichlet",
        "legacy": "legacy size-only",
    }
    for split, style in (("corrected", dict(marker="o", lw=1.8)),
                         ("legacy", dict(marker="s", lw=1.4, ls="--"))):
        if split not in grid:
            continue
        xs = sorted(grid[split])
        f1 = [_mean_std([r.get("f1_macro") for r in grid[split][a]])[0] for a in xs]
        tv = [_mean_std([r.get("partition", {}).get("label_tv_mean")
                         for r in grid[split][a]])[0] for a in xs]
        label = display_names[split]
        axes[0].plot(xs, tv, label=label, **style)
        axes[1].plot(xs, f1, label=label, **style)

    for ax, ylab, title in ((axes[0], "mean label TV distance", "Realised label skew"),
                            (axes[1], "Macro F1", "Accuracy")):
        ax.set_xscale("log")
        ax.set_xlabel(r"Dirichlet $\alpha$", fontsize=9)
        ax.set_ylabel(ylab, fontsize=9)
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=0.25, lw=0.5)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=7.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout(pad=0.4)
    fig.savefig(out / "fig_alpha_split.png", bbox_inches="tight")
    plt.close(fig)
    return "fig_alpha_split.png"


def build_standard_suite_figure(data: dict, out: Path) -> Optional[str]:
    """Compact four-panel figure for the IEEE reviewer-response manuscript."""
    required = ("E1_alpha_sweep", "E2_client_sweep", "E3_anticollapse",
                "E8_baselines")
    if any(not data.get(section) for section in required):
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    colors = {
        "blue": "#2b6cb0", "orange": "#dd6b20", "green": "#2f855a",
        "red": "#c53030", "purple": "#6b46c1", "gray": "#718096",
    }
    fig, axes = plt.subplots(1, 4, figsize=(12.0, 2.65), dpi=300)

    # (a) Realized alpha sweep.
    alpha_grid = defaultdict(lambda: defaultdict(list))
    for rec in data["E1_alpha_sweep"].values():
        alpha_grid[rec.get("splitter", "corrected")][float(rec["alpha"])].append(rec)
    for split, label, color, marker, ls in (
        ("corrected", "class-wise", colors["blue"], "o", "-"),
        ("legacy", "size-only", colors["orange"], "s", "--"),
    ):
        xs = sorted(alpha_grid[split])
        ys = [_mean_std([r.get("f1_macro") for r in alpha_grid[split][a]])[0]
              for a in xs]
        axes[0].plot(xs, ys, marker=marker, ls=ls, lw=1.7, ms=4.2,
                     color=color, label=label)
    axes[0].set_xscale("log")
    axes[0].set_xlabel(r"Dirichlet $\alpha$")
    axes[0].set_ylabel("Macro F1")
    axes[0].set_title("(a) Partition audit")
    axes[0].legend(fontsize=9.2, frameon=False, loc="lower right")

    # (b) Fixed-data client scalability.
    client_rows = sorted(
        (int(rec["num_clients"]), float(rec["f1_macro"]))
        for rec in data["E2_client_sweep"].values() if "error" not in rec
    )
    ks = [row[0] for row in client_rows]
    kf1 = [row[1] for row in client_rows]
    axes[1].plot(ks, kf1, marker="o", lw=1.8, ms=4.2, color=colors["green"])
    axes[1].set_xticks(ks)
    axes[1].set_xlabel("Clients, K")
    axes[1].set_title("(b) Client scaling")
    for x, y in zip(ks, kf1):
        axes[1].annotate(f"{y:.3f}", (x, y), xytext=(0, 4),
                         textcoords="offset points", ha="center", fontsize=8.5)

    # (c) Anti-collapse component ablation.
    arms = ["both", "no_diversity", "no_balanced", "neither"]
    arm_labels = ["Both", "No div.", "No bal.", "Neither"]
    e3 = {(rec["arm"], float(rec["alpha"])): float(rec["f1_macro"])
          for rec in data["E3_anticollapse"].values() if "error" not in rec}
    x = np.arange(len(arms))
    width = 0.36
    axes[2].bar(x - width / 2, [e3[(arm, 0.1)] for arm in arms], width,
                color=colors["blue"], label=r"$\alpha=0.1$")
    axes[2].bar(x + width / 2, [e3[(arm, 1.0)] for arm in arms], width,
                color=colors["orange"], label=r"$\alpha=1$")
    axes[2].set_xticks(x, arm_labels, rotation=28, ha="right")
    axes[2].set_title("(c) Components")
    axes[2].legend(fontsize=9.2, frameon=False, loc="lower right")

    # (d) Matched aggregation baselines.
    baseline = defaultdict(lambda: defaultdict(list))
    local = {}
    for key, rec in data["E8_baselines"].items():
        if "error" in rec:
            continue
        algo = key.split("|")[0]
        alpha = float(rec["alpha"])
        if algo == "local_only":
            local[alpha] = float(rec["mean_f1"])
        else:
            baseline[algo][alpha].append(float(rec["f1_macro"]))
    algos = ["local", "fedavg", "fedprox", "scaffold", "fedbn"]
    algo_labels = ["Local", "FedAvg", "FedProx", "SCAFF.", "FedBN"]
    vals01, vals1 = [], []
    for algo in algos:
        if algo == "local":
            vals01.append(local[0.1]); vals1.append(local[1.0])
        else:
            vals01.append(float(np.mean(baseline[algo][0.1])))
            vals1.append(float(np.mean(baseline[algo][1.0])))
    x = np.arange(len(algos))
    axes[3].bar(x - width / 2, vals01, width, color=colors["blue"],
                label=r"$\alpha=0.1$")
    axes[3].bar(x + width / 2, vals1, width, color=colors["orange"],
                label=r"$\alpha=1$")
    axes[3].set_xticks(x, algo_labels, rotation=28, ha="right")
    axes[3].set_title("(d) Matched baselines")

    for ax in axes:
        ax.set_ylim(0.2, 0.96)
        ax.grid(axis="y", alpha=0.22, lw=0.5)
        ax.tick_params(labelsize=9.3)
        ax.xaxis.label.set_size(10.0)
        ax.yaxis.label.set_size(10.0)
        ax.title.set_size(10.5)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.tight_layout(pad=0.55, w_pad=0.8)
    fig.savefig(out / "fig_standard_suite.png", bbox_inches="tight")
    plt.close(fig)
    return "fig_standard_suite.png"


def build_alpha_vertical_figure(data: dict, out: Path) -> Optional[str]:
    """Single-column version of the partition audit for dense IEEE layouts."""
    exp = data.get("E1_alpha_sweep") or {}
    if not exp:
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    grid = defaultdict(lambda: defaultdict(list))
    for rec in exp.values():
        if "error" not in rec:
            grid[rec.get("splitter", "corrected")][float(rec["alpha"])].append(rec)

    fig, axes = plt.subplots(2, 1, figsize=(3.65, 4.1), dpi=300, sharex=True)
    styles = {
        "corrected": ("class-wise", "#2b6cb0", "o", "-"),
        "legacy": ("size-only", "#dd6b20", "s", "--"),
    }
    for split in ("corrected", "legacy"):
        label, color, marker, ls = styles[split]
        xs = sorted(grid[split])
        tv = [_mean_std([r.get("partition", {}).get("label_tv_mean")
                         for r in grid[split][a]])[0] for a in xs]
        f1 = [_mean_std([r.get("f1_macro") for r in grid[split][a]])[0]
              for a in xs]
        axes[0].plot(xs, tv, marker=marker, ls=ls, lw=1.7, ms=4.2,
                     color=color, label=label)
        axes[1].plot(xs, f1, marker=marker, ls=ls, lw=1.7, ms=4.2,
                     color=color, label=label)
    axes[0].set_title("(a) Realized label skew")
    axes[0].set_ylabel("Mean label TV")
    axes[1].set_title("(b) Performance response")
    axes[1].set_ylabel("Macro F1")
    axes[1].set_xlabel(r"Dirichlet $\alpha$")
    axes[1].legend(fontsize=7.5, frameon=False, ncol=2, loc="lower right")
    for ax in axes:
        ax.set_xscale("log")
        ax.grid(alpha=0.22, lw=0.5)
        ax.tick_params(labelsize=8.3)
        ax.title.set_size(9.5)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.tight_layout(pad=0.45, h_pad=0.65)
    fig.savefig(out / "fig_alpha_split_vertical.png", bbox_inches="tight")
    plt.close(fig)
    return "fig_alpha_split_vertical.png"


def build_remaining_ablations_figure(data: dict, out: Path) -> Optional[str]:
    """Plot the reviewer ablations that are otherwise only visible in Table VI."""
    required = ("E4_warmstart", "E5_fusion_seeds", "E6_cost")
    if any(not data.get(section) for section in required):
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    blue, orange, green, gray = "#2b6cb0", "#dd6b20", "#2f855a", "#718096"
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 2.65), dpi=300)

    # (a) Warm versus cold initialization, averaged across seeds.
    warm_groups = defaultdict(list)
    for rec in data["E4_warmstart"].values():
        if "error" not in rec:
            warm_groups[rec["mode"]].append(float(rec["f1_macro"]))
    warm_order = ["cold", "warm"]
    warm_means = [float(np.mean(warm_groups[k])) for k in warm_order]
    warm_std = [float(np.std(warm_groups[k])) for k in warm_order]
    axes[0].bar([0, 1], warm_means, yerr=warm_std, capsize=3,
                color=[gray, green], width=0.62)
    axes[0].set_xticks([0, 1], ["Cold", "Warm"])
    axes[0].set_ylabel("Macro F1")
    axes[0].set_title("(a) Initialization")
    for i, y in enumerate(warm_means):
        axes[0].text(i, y + 0.018, f"{y:.3f}", ha="center", fontsize=9)

    # (b) Fusion strategy across seeds.
    fusion_groups = defaultdict(list)
    for rec in data["E5_fusion_seeds"].values():
        if "error" not in rec:
            fusion_groups[rec["fusion_type"]].append(float(rec["f1_macro"]))
    fusion_order = ["concat", "attention", "gated"]
    fusion_means = [float(np.mean(fusion_groups[k])) for k in fusion_order]
    fusion_std = [float(np.std(fusion_groups[k])) for k in fusion_order]
    axes[1].bar(np.arange(3), fusion_means, yerr=fusion_std, capsize=3,
                color=[blue, orange, green], width=0.64)
    axes[1].set_xticks(np.arange(3), ["Concat", "Attention", "Gated"],
                       rotation=18, ha="right")
    axes[1].set_title("(b) Fusion")
    for i, y in enumerate(fusion_means):
        axes[1].text(i, y + 0.018, f"{y:.3f}", ha="center", fontsize=9)

    # (c) Per-client communication for one float32 round.
    cost = {}
    for rec in data["E6_cost"].values():
        if "error" not in rec:
            cost[rec["fusion_type"]] = rec
    upload = [float(cost[k]["upload_mib_per_client_per_round"])
              for k in fusion_order]
    roundtrip = [float(cost[k]["round_trip_bytes_per_client"]) / 2**20
                 for k in fusion_order]
    x = np.arange(3); width = 0.36
    axes[2].bar(x - width / 2, upload, width, color=blue, label="Upload")
    axes[2].bar(x + width / 2, roundtrip, width, color=orange, label="Round trip")
    axes[2].set_xticks(x, ["Concat", "Attention", "Gated"],
                       rotation=18, ha="right")
    axes[2].set_ylabel("MiB / client-round")
    axes[2].set_title("(c) Communication")
    axes[2].legend(fontsize=8.5, frameon=False)

    for ax in axes[:2]:
        ax.set_ylim(0.75, 0.93)
    axes[2].set_ylim(0, 135)
    for ax in axes:
        ax.grid(axis="y", alpha=0.22, lw=0.5)
        ax.tick_params(labelsize=9.2)
        ax.xaxis.label.set_size(10.0)
        ax.yaxis.label.set_size(10.0)
        ax.title.set_size(10.5)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.tight_layout(pad=0.55, w_pad=1.0)
    fig.savefig(out / "fig_remaining_ablations.png", bbox_inches="tight")
    plt.close(fig)
    return "fig_remaining_ablations.png"


def build_training_dynamics_figure(data: dict, out: Path) -> Optional[str]:
    """Two-seed round trajectories for initialization, fusion, and aggregators."""
    required = ("E4_warmstart", "E5_fusion_seeds", "E8_baselines")
    if any(not data.get(section) for section in required):
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    palette = {
        "warm": "#2f855a", "cold": "#718096", "concat": "#2b6cb0",
        "attention": "#dd6b20", "gated": "#2f855a", "fedavg": "#2b6cb0",
        "fedprox": "#dd6b20", "scaffold": "#c53030", "fedbn": "#6b46c1",
    }
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 5.0), dpi=300)

    def plot_group(ax, records, group_key, order, title):
        grouped = defaultdict(list)
        for rec in records.values():
            if "error" in rec:
                continue
            hist = rec.get("history", {}).get("round_f1_macro", [])
            if hist:
                grouped[str(rec[group_key])].append(np.asarray(hist, dtype=float))
        for name in order:
            curves = grouped.get(name, [])
            if not curves:
                continue
            length = min(len(c) for c in curves)
            arr = np.stack([c[:length] for c in curves])
            rounds = np.arange(1, length + 1)
            mean = arr.mean(axis=0)
            ax.plot(rounds, mean, marker="o", ms=3.0, lw=1.6,
                    color=palette[name], label=name.capitalize())
            if len(curves) > 1:
                ax.fill_between(rounds, arr.min(axis=0), arr.max(axis=0),
                                color=palette[name], alpha=0.12, linewidth=0)
        ax.set_title(title)
        ax.set_xlabel("Communication round")
        ax.set_ylabel("Macro F1")
        ax.set_xticks(np.arange(1, 9))
        ax.set_ylim(0.0, 0.96)
        ax.legend(fontsize=8.0, frameon=False, ncol=2, loc="lower right")

    plot_group(axes[0, 0], data["E4_warmstart"], "mode",
               ["cold", "warm"], "(a) Initialization dynamics")
    plot_group(axes[0, 1], data["E5_fusion_seeds"], "fusion_type",
               ["concat", "attention", "gated"], "(b) Fusion dynamics")

    for ax, alpha, title in ((axes[1, 0], 0.1, r"(c) Aggregators, $\alpha=0.1$"),
                             (axes[1, 1], 1.0, r"(d) Aggregators, $\alpha=1$")):
        subset = {}
        for key, rec in data["E8_baselines"].items():
            if "local_only" in key or "error" in rec or float(rec["alpha"]) != alpha:
                continue
            subset[key] = rec
        plot_group(ax, subset, "algorithm",
                   ["fedavg", "fedprox", "scaffold", "fedbn"], title)

    for ax in axes.flat:
        ax.grid(alpha=0.22, lw=0.5)
        ax.tick_params(labelsize=8.7)
        ax.xaxis.label.set_size(9.4)
        ax.yaxis.label.set_size(9.4)
        ax.title.set_size(10.2)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.tight_layout(pad=0.6, h_pad=1.0, w_pad=1.2)
    fig.savefig(out / "fig_training_dynamics.png", bbox_inches="tight")
    plt.close(fig)
    return "fig_training_dynamics.png"


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="farm_results.json")
    ap.add_argument("--outdir", default="paper_assets")
    args = ap.parse_args()

    res_path = Path(args.results)
    if not res_path.is_file():
        raise SystemExit(f"results file not found: {res_path}")
    data = json.loads(res_path.read_text())

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    summary: List[str] = ["# FarmFederate ablation readout\n"]
    meta = (data.get("_meta") or {}).get("data")
    if meta:
        summary.append("## Data actually used\n")
        summary.append(f"- paired samples: {meta.get('num_paired')} "
                       f"(train {meta.get('train')}, val {meta.get('val')})")
        summary.append(f"- real images before synthetic fill: {meta.get('num_real_images')}")
        summary.append(f"- class counts: {meta.get('class_counts')}")
        summary.append(f"- pairing mode: **{meta.get('pairing_mode')}**")
        summary.append(f"- {meta.get('note')}\n")

    written = []
    for fn in (build_alpha_split, build_client_sweep, build_anticollapse,
               build_warmstart, build_fusion_variance, build_cost, build_baselines):
        name = fn(data, out, summary)
        if name:
            written.append(name)
    for figure_builder in (build_alpha_figure, build_standard_suite_figure,
                           build_alpha_vertical_figure,
                           build_remaining_ablations_figure,
                           build_training_dynamics_figure):
        fig = figure_builder(data, out)
        if fig:
            written.append(fig)

    missing = [tag for tag in ("E1_alpha_sweep", "E2_client_sweep", "E3_anticollapse",
                               "E4_warmstart", "E5_fusion_seeds", "E6_cost",
                               "E8_baselines") if not data.get(tag)]
    if missing:
        summary.append("## Not yet run\n")
        summary.append("- " + ", ".join(missing))
        summary.append("- Tables above cover only the chunks present in the results "
                       "file; re-run the missing chunks and regenerate.\n")

    (out / "SUMMARY.md").write_text("\n".join(summary), encoding="utf8")
    written.append("SUMMARY.md")

    print("Wrote:")
    for name in written:
        print("  ", out / name)
    print("\nRead SUMMARY.md before pasting anything into the paper.")


if __name__ == "__main__":
    main()
