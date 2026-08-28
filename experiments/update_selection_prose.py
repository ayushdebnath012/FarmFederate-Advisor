#!/usr/bin/env python3
"""Rewrite the Table VI discussion for the corrected corpus.

The old two paragraphs rested on a premise the rerun removes. They said the
macro-F1 ordering reversed the accuracy ordering and that a single two-crop
class produced almost all of a 0.151 gap. On the corrected corpus the two
systems are 0.0015 apart in test macro-F1, so there is no reversal to explain
and no per-class story worth half a column: the honest statement is that the
support cannot separate them, which takes three sentences rather than eleven.

That is also where the rest of the page for the robustness section comes from.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEL = (ROOT / "tea_results" / "vit_lm_encoder_selection_20260729"
       / "encoder_selection_results.json")
TEXS = ["overleaf_final/main.tex", "overleaf_final_slim/main.tex"]
B = chr(92)
R = B + "ref{"

OLD = (
    "In Table~" + R + "tab:family_comparison} the routed ResNet--compact model is the\n"
    + B + "emph{incumbent} --- the core already\n"
    "carried into the routing and federated experiments --- and the pooled\n"
    "ViT--DistilBERT model is the " + B + "emph{candidate}, adopted only if it beats the\n"
    "incumbent on validation. Validation selects the incumbent before test\n"
    "extraction: 89.19" + B + "% accuracy and 0.8541 F1 versus 82.43" + B + "% and 0.6892 for the\n"
    "candidate.\n"
    "The candidate later predicts one more test crop, but the 11-versus-10 discordance\n"
    "is not significant (McNemar $p=1.0$). The selection/test distinction is what matters here: earlier screens ran on\n"
    "different supports and license within-panel ranking only.\n"
    "\n"
    "The macro-F1 ordering reverses the accuracy ordering, and one two-crop class\n"
    "produces almost all of it. On leaf hoppers the incumbent scores F1 0.667 on\n"
    "validation and 0.000 on test, while the candidate scores 0.000 on validation and\n"
    "0.667 on test --- the two models swap a single crop, which accounts for 0.133 of\n"
    "the 0.151 test macro-F1 gap. Dropping that class, four-class macro F1 is 0.901\n"
    "against 0.862 on validation and 0.835 against 0.857 on test, so the lead changes\n"
    "hands by 0.02--0.04. The incumbent's validation-to-test accuracy movement (66/74\n"
    "to 61/75) is 1.35 standard errors. Macro-F1 differences on this support are\n"
    "therefore reported and not ranked."
)


def main() -> None:
    s = json.loads(SEL.read_text(encoding="utf-8"))
    cand, inc = s["candidate"], s["incumbent"]
    cv, ct = cand["selected_validation"], cand["fixed_test"]
    iv, it = inc["selected_validation"], inc["fixed_test"]
    p = s["paired_test_comparison"]

    new = (
        "In Table~" + R + "tab:family_comparison} the routed ResNet--compact model is the\n"
        + B + "emph{incumbent} --- the core already carried into the routing and federated\n"
        "experiments --- and the pooled ViT--DistilBERT model is the " + B + "emph{candidate},\n"
        "adopted only if it beats the incumbent on validation. Validation selects the\n"
        f"incumbent before test extraction: {100 * iv['accuracy']:.2f}" + B + f"% accuracy and {iv['macro_f1']:.4f} F1\n"
        f"against {100 * cv['accuracy']:.2f}" + B + f"% and {cv['macro_f1']:.4f} for the candidate. On test the candidate\n"
        f"predicts {ct['correct'] - it['correct']} more crops, but the "
        f"{p['first_only_correct']}-versus-{p['second_only_correct']} discordance is not\n"
        f"significant (McNemar $p={p['exact_two_sided_p']:.2f}$) and the two macro-F1 values differ by\n"
        f"{abs(ct['macro_f1'] - it['macro_f1']):.4f}. On 75 crops the support does not separate them, so the\n"
        "selection rule decides and the test column is reported rather than ranked;\n"
        "earlier screens ran on different supports and license within-panel ranking only."
    )

    for tex in TEXS:
        pth = ROOT / tex
        txt = pth.read_text(encoding="utf-8")
        if "the support does not separate them" in txt:
            print(f"  {tex}: already rewritten")
            continue
        if txt.count(OLD) != 1:
            print(f"  {tex}: ANCHOR x{txt.count(OLD)} -- not applied")
            continue
        pth.write_text(txt.replace(OLD, new), encoding="utf-8")
        print(f"  {tex}: rewritten ({OLD.count(chr(10))} -> {new.count(chr(10))} lines)")
    print("\n" + new)


if __name__ == "__main__":
    main()
