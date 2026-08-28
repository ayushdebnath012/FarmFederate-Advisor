#!/usr/bin/env python3
"""Rewrite the federated-adaptation paragraph against the retrained checkpoint.

The old paragraph reported 0.8924--0.9099 final macro-F1 and "98.1--100.0% of
the 0.9099 initialization". Both numbers belong to the checkpoint trained on the
pre-correction notes, where the text branch was label-determining and the whole
stack sat near the ceiling. Retrained on the corrected corpus the initialization
is much lower and the retention story changes shape, so the claim is recomputed
rather than rescaled.

Retention is expressed against the centralized validation macro-F1 recorded by
the training run itself, so the denominator is the same checkpoint the clients
start from.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ADAPT = (ROOT / "tea_results" / "federated_adaptation_v6"
         / "federated_adaptation_results.json")
BASE = (ROOT / "tea_results" / "multimodal_v6_vision_full_20260727"
        / "complete_results.json")
TEXS = ["overleaf_final/main.tex", "overleaf_final_slim/main.tex"]
B = chr(92)
R = B + "ref{"

OLD = (
    "Figure~" + R + "fig:fl_scaling} answers the two federated scaling questions\n"
    "directly. Final-round macro F1 ranges from 0.8924\n"
    "to 0.9099 and is never perfect. At the predeclared $K=3$ setting, the final\n"
    "mean is 0.9046, or 99.4" + B + "% of the 0.9099 initialization. Round-wise\n"
    "means stay between 0.8924 and 0.9216; the highest transient mean occurs for\n"
    "$K=5$ at round 4, but it is reported as a validation peak rather than the final\n"
    "model. Increasing $K$ raises realized label-distribution total variation from\n"
    "0.139 to 0.300 without a monotone final-score penalty."
)


def main() -> None:
    a = json.loads(ADAPT.read_text(encoding="utf-8"))
    init = json.loads(BASE.read_text(encoding="utf-8"))["results"]["vlm"]["central_f1"]
    s = a["summary"]
    ks = sorted(s, key=int)
    finals = {k: s[k]["final_macro_f1_mean"] for k in ks}
    tvs = {k: s[k]["mean_label_total_variation"] for k in ks}
    rounds = [v for k in ks for v in s[k]["round_mean_macro_f1"]]
    peak_k = max(ks, key=lambda k: max(s[k]["round_mean_macro_f1"]))
    peak_round = s[peak_k]["round_mean_macro_f1"].index(
        max(s[peak_k]["round_mean_macro_f1"])) + 1
    ret = {k: 100 * finals[k] / init for k in ks}
    n_seeds = len(a["config"]["seeds"])

    new = (
        "Figure~" + R + "fig:fl_scaling} answers the two federated scaling questions\n"
        f"directly. Final-round macro F1 ranges from {min(finals.values()):.4f} to "
        f"{max(finals.values()):.4f} over\n"
        f"$K=2$ to {ks[-1]}, against a centralized initialization of {init:.4f} --- "
        f"{min(ret.values()):.0f}--{max(ret.values()):.0f}" + B + "% retained,\n"
        "so adaptation costs something on the corrected corpus where it cost almost\n"
        f"nothing on the saturated one. At the predeclared $K=3$ setting the final mean is\n"
        f"{finals['3']:.4f} ({ret['3']:.0f}" + B + "%). Round-wise means stay between "
        f"{min(rounds):.4f} and {max(rounds):.4f}; the\n"
        f"highest transient mean occurs for $K={peak_k}$ at round {peak_round}, reported as a validation\n"
        "peak rather than the final model. Increasing $K$ raises realized\n"
        f"label-distribution total variation from {min(tvs.values()):.3f} to "
        f"{max(tvs.values()):.3f} without a monotone\n"
        f"final-score penalty, over {n_seeds} partition seeds."
    )

    for tex in TEXS:
        p = ROOT / tex
        txt = p.read_text(encoding="utf-8")
        if "against a centralized initialization of" in txt:
            print(f"  {tex}: already rewritten")
            continue
        if txt.count(OLD) != 1:
            print(f"  {tex}: ANCHOR x{txt.count(OLD)} -- not applied")
            continue
        p.write_text(txt.replace(OLD, new), encoding="utf-8")
        print(f"  {tex}: rewritten")
    print("\n" + new)


if __name__ == "__main__":
    main()
