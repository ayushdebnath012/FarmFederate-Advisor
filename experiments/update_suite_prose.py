#!/usr/bin/env python3
"""Rewrite the Table II paragraph against the corrected full tier.

Every quantitative claim in the old paragraph came from the pre-correction run
and is now contradicted by the table beside it: gains of $+0.131$ to $+0.545$
(now $+0.124$ to $+0.229$), FedProx leading four of five (now two of five),
saturation at 1.000 in most cells (nothing above 0.561), and a warm-start
reversal quoted as 1.000 against 0.795.

One claim survives unchanged and is kept: the splitter's label total variation.
The note correction rewrote text only -- the 371 crops, their labels and the
source-grouped split are untouched -- so the partition audit in
Fig.~\\ref{fig:c_alpha_corrected} is unaffected by it.

The replacement is also about half the length, which is where part of the page
for the robustness section comes from.
"""

from __future__ import annotations

import json
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "experiments" / "farm_results_full_genuine.json"
TEXS = ["overleaf_final/main.tex", "overleaf_final_slim/main.tex"]
B = chr(92)
R = B + "ref{"

OLD = (
    "Table~" + R + "tab:standard_suite} closes the remaining gaps without mixing\n"
    "supports. The suite runs on 371 genuine box-linked observations with an\n"
    "identical client partition per cell, so differences are architectural. At\n"
    "$" + B + "alpha=0.1$ the corrected splitter realizes label TV 0.451 against 0.265\n"
    "(Fig.~" + R + "fig:c_alpha_corrected}), and federation beats local-only training for\n"
    "all five architectures (gains $+0.131$ to $+0.545$). Across three seeds FedProx\n"
    "leads or ties in four of the five at $" + B + "alpha=0.1$, most clearly on concat (0.848\n"
    "against 0.795 for FedAvg); only the image-only arm favours FedAvg, and by 0.004.\n"
    "SCAFFOLD is the exception, unstable everywhere (0.088--0.639) and strictly worst\n"
    "in nine of ten cells. Communication spans $3.5" + B + "times$, 18.9 to 67.2" + B + ",MiB per\n"
    "client-round, so the image-only model reaches 0.526 macro F1 at roughly a third\n"
    "the uplink cost of either fusion model. Saturation is common at mild skew ---\n"
    "four of the five reach exactly 1.000 at $" + B + "alpha=0.5$, three at $" + B + "alpha=1$ and two\n"
    "at $" + B + "alpha=10$ --- and on a 53-crop validation split that indicates the templated\n"
    "notes are close to label-determining rather than that federation is solved. Warm\n"
    "start does not help once trained out: over three seeds cold start is identical\n"
    "at two and better at the third (1.000 against 0.795), reversing the eight-round\n"
    "result. The anti-collapse and fusion-variant ablations remain\n"
    "single-architecture."
)

ARCHS = ["image_only", "text_only", "concat_vlm", "attention_vlm",
         "cross_attention_vlm"]
NICE = {"image_only": "image-only", "text_only": "text",
        "concat_vlm": "concat", "attention_vlm": "attention fusion",
        "cross_attention_vlm": "cross-attention"}


def parse(k):
    out = {}
    for part in k.split("|"):
        if "=" in part:
            a, b = part.split("=", 1)
            out[a] = b
        else:
            out["_alg"] = part
    return out


def facts():
    d = json.load(open(SRC, encoding="utf-8"))
    cells = defaultdict(list)
    for k, v in d["E1_alpha_sweep"].items():
        f = parse(k)
        if f.get("split") == "corrected":
            cells[(f["arch"], float(f["alpha"]))].append(v["f1_macro"])
    span = [st.mean(v) for v in cells.values()]

    agg = defaultdict(list)
    for k, v in d["E8_baselines"].items():
        f = parse(k)
        if f.get("alpha") == "0.1":
            agg[(f["arch"], f["_alg"])].append(v.get("f1_macro", v.get("mean_f1")))
    A = lambda ar, al: st.mean(agg[(ar, al)]) if agg[(ar, al)] else None

    gains = [A(a, "fedavg") - A(a, "local_only") for a in ARCHS]
    winners = defaultdict(list)
    for a in ARCHS:
        c = {al: A(a, al) for al in ("fedavg", "fedprox", "fedbn", "scaffold")}
        winners[max(c, key=c.get)].append(NICE[a])
    scaf = [A(a, "scaffold") for a in ARCHS]
    scaf_worst = sum(
        1 for a in ARCHS
        if A(a, "scaffold") == min(A(a, al) for al in
                                   ("fedavg", "fedprox", "fedbn", "scaffold")))

    cost = {k.split("|", 1)[1]: v["upload_mib_per_client_per_round"]
            for k, v in d["E6_cost"].items() if k.startswith("arch_cost|")}
    mib = [cost[a] for a in ARCHS]

    warm = [v["f1_macro"] for k, v in d["E4_warmstart"].items() if k.startswith("warm")]
    cold = [v["f1_macro"] for k, v in d["E4_warmstart"].items() if k.startswith("cold")]
    warm_wins = sum(1 for w, c in zip(warm, cold) if w > c)
    return dict(lo=min(span), hi=max(span), gains=gains, winners=winners,
                scaf=scaf, scaf_worst=scaf_worst, mib=mib, A=A,
                warm=st.mean(warm), cold=st.mean(cold), warm_wins=warm_wins,
                n_seeds=len(warm))


def main() -> None:
    f = facts()
    A = f["A"]
    w = f["winners"]
    lead = lambda alg: ", ".join(w[alg]) if w[alg] else "nothing"

    new = (
        "Table~" + R + "tab:standard_suite} closes the remaining gaps without mixing\n"
        "supports: 371 genuine box-linked observations, an identical client partition\n"
        "per cell, so differences are architectural. The note correction rewrote text\n"
        "only --- crops, labels and split are untouched --- so the corrected splitter\n"
        "still realizes label TV 0.451 against 0.265 at $" + B + "alpha=0.1$\n"
        "(Fig.~" + R + "fig:c_alpha_corrected}). Federation beats local-only training for all\n"
        f"five architectures ($+${min(f['gains']):.3f} to $+${max(f['gains']):.3f}), and nothing saturates: the\n"
        f"sweep spans {f['lo']:.3f}--{f['hi']:.3f}, so the ordering is architectural rather than the\n"
        "note-leakage ceiling the previous corpus imposed. No aggregator dominates ---\n"
        f"FedProx leads on {lead('fedprox')} "
        f"({A('concat_vlm', 'fedprox'):.3f} against {A('concat_vlm', 'fedavg'):.3f} on concat),\n"
        f"FedAvg on {lead('fedavg')}, FedBN on {lead('fedbn')} --- while SCAFFOLD is unstable\n"
        f"everywhere ({min(f['scaf']):.3f}--{max(f['scaf']):.3f}) and strictly worst in "
        f"{f['scaf_worst']} of five. Communication\n"
        f"spans ${max(f['mib']) / min(f['mib']):.1f}" + B + "times$, {:.1f} to {:.1f}".format(min(f["mib"]), max(f["mib"]))
        + B + ",MiB per client-round, so image-only\n"
        f"reaches {A('image_only', 'fedavg'):.3f} macro F1 at roughly a third the uplink of either fusion.\n"
        f"Warm start does not survive training out: it leads on {f['warm_wins']} of {f['n_seeds']} seeds but\n"
        f"loses the mean by {abs(f['cold'] - f['warm']):.3f}. The anti-collapse and fusion-variant ablations\n"
        "remain single-architecture."
    )

    for tex in TEXS:
        p = ROOT / tex
        s = p.read_text(encoding="utf-8")
        if "nothing saturates" in s:
            print(f"  {tex}: already rewritten")
            continue
        if s.count(OLD) != 1:
            print(f"  {tex}: ANCHOR x{s.count(OLD)} -- not applied")
            continue
        p.write_text(s.replace(OLD, new), encoding="utf-8")
        print(f"  {tex}: rewritten ({OLD.count(chr(10))} -> {new.count(chr(10))} lines)")
    print("\n" + new)


if __name__ == "__main__":
    main()
