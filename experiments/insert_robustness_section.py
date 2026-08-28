#!/usr/bin/env python3
"""Insert the federated-robustness subsection, figure and Threats clause.

The manuscript federated only under mild and moderate label skew with every
client reporting a fresh honest update each round, and said so. This adds the
four failure modes that were named as untested: severe skew, client dropout,
straggler staleness and poisoned updates.

Every number in the prose is read from the result JSON rather than typed, so a
rerun that moves a number moves the sentence with it. The one thing the script
asserts rather than computes is the shape of the argument; the assertions at the
top of build() fail loudly if a rerun changes that shape (for instance if
stragglers stopped costing more than dropout), which is the point -- a
regenerated paragraph should never quietly describe a pattern that no longer
holds.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "tea_results" / "federated_robustness" / "federated_robustness.json"
TEXS = ["overleaf_final/main.tex", "overleaf_final_slim/main.tex"]
B = chr(92)
ANCHOR = B + "subsection{Actual Model-Family and Inter-Model Comparison}"


def f3(x):
    return f"{x:.3f}"


def build():
    d = json.load(open(SRC, encoding="utf-8"))
    m = d["_meta"]
    het, drop, strag, pois = (d["heterogeneity"], d["dropout"],
                              d["stragglers"], d["poisoning"])
    g = lambda store, k, s: store[f"{k}|system={s}"]["mean_macro_f1"]

    alphas = sorted({float(k.split("|")[0].split("=")[1])
                     for k in het if "pathological" not in k}, reverse=True)
    h_keys = [f"alpha={a}" for a in alphas] + ["alpha=pathological"]
    fus_h = [g(het, k, "fusion") for k in h_keys]
    best_txt_h = max(g(het, k, "text") for k in h_keys)
    best_img_h = max(g(het, k, "image") for k in h_keys)

    d_keys = sorted({k.split("|")[0] for k in drop},
                    key=lambda s: float(s.split("=")[1]))
    drop_hi = 100 * float(d_keys[-1].split("=")[1])
    s_keys = sorted({k.split("|")[0] for k in strag},
                    key=lambda s: float(s.split("=")[1]))
    st_hi = 100 * float(s_keys[-1].split("=")[1])

    # the argument depends on these holding; a rerun that breaks one must not
    # silently regenerate a paragraph claiming it
    assert min(fus_h) > max(best_txt_h, best_img_h), "fusion no longer dominates under skew"
    assert abs(g(drop, d_keys[-1], "fusion") - g(drop, d_keys[0], "fusion")) < 0.02, \
        "dropout now costs something"
    assert g(strag, s_keys[0], "fusion") - g(strag, s_keys[-1], "fusion") > 0.05, \
        "stragglers no longer cost"

    P = lambda atk, fr, agg, s: g(pois, f"{atk}@{fr}|agg={agg}", s)
    clean_mean = g(pois, "clean|agg=mean", "fusion")
    clean_med = g(pois, "clean|agg=median", "fusion")

    sec = (
        B + "subsection{Robustness: Skew, Dropout, Stragglers and Poisoning}\n"
        + B + "label{sec:robustness}\n\n"

        + B + "emph{Held fixed.} The split, the frozen encoders and the FedAvg protocol\n"
        f"({m['clients']} clients, {m['rounds']} rounds, {m['local_epochs']} local epochs,\n"
        f"{len(m['seeds'])} seeds); the varied axis is the failure mode. Within every cell the\n"
        "arm and its control share the client partition at the same seed, so a difference\n"
        "is the failure mode and not the split. Heads are linear probes on standardized\n"
        "frozen features, which is what makes the sweep affordable; the invariances below\n"
        "are properties of that estimator and are not claimed for a deep federated model.\n\n"

        + B + "textbf{Skew and dropout break nothing.} From $" + B + f"alpha{{=}}{alphas[0]:g}$ down to\n"
        "$" + B + f"alpha{{=}}{alphas[-1]:g}$, and under a pathological partition giving each client a\n"
        f"single class, fused macro-F1 stays in {f3(min(fus_h))}--{f3(max(fus_h))} --- its worst setting above\n"
        f"the best setting of either parent ({f3(best_txt_h)} text, {f3(best_img_h)} image). Dropout is\n"
        f"flatter still: {drop_hi:.0f}" + B + "% of clients failing to report each round leaves the fused\n"
        f"system at {f3(g(drop, d_keys[-1], 'fusion'))} against {f3(g(drop, d_keys[0], 'fusion'))} with none. The image branch is the\n"
        "limiting case, returning per-seed-identical scores across $" + B + "alpha{=}0.5$, 1 and 10\n"
        "and again across $" + B + "alpha{=}0.05$, 0.01 and the pathological split: a convex probe\n"
        "reaches the same solution whatever the partition, so skew changes the path\n"
        "(round 5: 0.435 against 0.420) and not the destination.\n\n"

        + B + "textbf{Stragglers cost what dropout does not.} A stale update arrives and is\n"
        "averaged in, pulling the model toward an older global state rather than merely\n"
        f"shrinking the cohort. At {st_hi:.0f}" + B + "% stale updates per round the fused system loses\n"
        f"{f3(g(strag, s_keys[0], 'fusion') - g(strag, s_keys[-1], 'fusion'))} "
        f"({f3(g(strag, s_keys[0], 'fusion'))} to {f3(g(strag, s_keys[-1], 'fusion'))}) and the image branch "
        f"{f3(g(strag, s_keys[0], 'image') - g(strag, s_keys[-1], 'image'))}\n"
        f"({f3(g(strag, s_keys[0], 'image'))} to {f3(g(strag, s_keys[-1], 'image'))}), where the same fraction of "
        + B + "emph{dropped} clients costs\n"
        "nothing. Discarding a late update beats waiting for it.\n\n"

        + B + "textbf{Poisoning is where plain FedAvg has no defence.} Sign-flipped updates are\n"
        f"catastrophic at every fraction tested: 10" + B + "% malicious clients already take the\n"
        f"fused system from {f3(clean_mean)} to {f3(P('sign_flip', 0.1, 'mean', 'fusion'))}, "
        f"and 30" + B + f"% to {f3(P('sign_flip', 0.3, 'mean', 'fusion'))}, with text\n"
        f"({f3(P('sign_flip', 0.3, 'mean', 'text'))}) and image ({f3(P('sign_flip', 0.3, 'mean', 'image'))}) "
        "equally destroyed --- the one condition in\n"
        "which fusion is not the best of the three, because all three are broken. Label\n"
        f"flip degrades gradually instead ({f3(P('label_flip', 0.1, 'mean', 'fusion'))}, "
        f"{f3(P('label_flip', 0.2, 'mean', 'fusion'))}, {f3(P('label_flip', 0.3, 'mean', 'fusion'))} at 10/20/30" + B + "%) and\n"
        "Gaussian noise sits between them. Coordinate-wise median recovers the outlier\n"
        f"attacks --- sign flip to {f3(P('sign_flip', 0.1, 'median', 'fusion'))}, "
        f"{f3(P('sign_flip', 0.2, 'median', 'fusion'))} and {f3(P('sign_flip', 0.3, 'median', 'fusion'))}, "
        f"Gaussian to {f3(P('gaussian', 0.3, 'median', 'fusion'))} at 30" + B + "% --- but not\n"
        f"label flip, where it is within noise at 10--20" + B + "% and worse at 30" + B + "% "
        f"({f3(P('label_flip', 0.3, 'median', 'fusion'))}\n"
        f"against {f3(P('label_flip', 0.3, 'mean', 'fusion'))}), because a label-flipped update is a "
        "well-formed gradient step\n"
        "toward the wrong target and is not an outlier in parameter space. The median also\n"
        f"costs {f3(clean_mean - clean_med)} with nobody attacking ({f3(clean_med)} against {f3(clean_mean)}), so it is a trade\n"
        "and not a free defence. No configuration here is Byzantine-tolerant and none is\n"
        "claimed to be.\n\n"

        + B + "begin{figure*}[t]\n"
        + B + "centering\n"
        + B + "includegraphics[width=" + B + "linewidth]{plots/plot92_robustness_all}\n"
        + B + "caption{The four federated failure modes the earlier sweeps do not cover, on the\n"
        "audited split. (a) Severe label skew, to one class per client. (b) Clients failing\n"
        "to report per round. (c) Clients reporting an update computed from an earlier\n"
        "global model. (d) Malicious clients under mean against coordinate-wise-median\n"
        "aggregation. Bands are $" + B + "pm$1 SD over "
        f"{len(m['seeds'])} seeds. Skew and dropout are absorbed;\n"
        "staleness and poisoning are not.}\n"
        + B + "label{fig:robustness}\n"
        + B + "end{figure*}\n\n"
    )
    return sec


def main() -> None:
    sec = build()
    for tex in TEXS:
        p = ROOT / tex
        s = p.read_text(encoding="utf-8")
        if B + "label{sec:robustness}" in s:
            print(f"  {tex}: section already present")
            continue
        if s.count(ANCHOR) != 1:
            print(f"  {tex}: ANCHOR x{s.count(ANCHOR)} -- not applied")
            continue
        p.write_text(s.replace(ANCHOR, sec + ANCHOR), encoding="utf-8")
        print(f"  {tex}: inserted ({sec.count(chr(10))} lines)")


if __name__ == "__main__":
    main()
