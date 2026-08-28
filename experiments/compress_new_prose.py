#!/usr/bin/env python3
"""Compress the two rewritten paragraphs back into the page budget.

The corrected ablation and fusion-figure paragraphs were right but long: they
took the manuscript from 8 pages to 9, and the robustness section still has to
fit. Same claims, same numbers, fewer lines -- nothing is dropped except
restatement.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEXS = ["overleaf_final/main.tex", "overleaf_final_slim/main.tex"]
B = chr(92)
R = B + "ref{"

OLD_ABL = (
    "The sweep was rerun as 11 variants $" + B + "times$ 3 seeds on a single device (33 runs, 2.0 GPU\n"
    "hours), which is what separates a component effect from seed noise; the\n"
    "earlier CPU-versus-H100 comparison confounded the two. The noise floor is\n"
    "large: pooled within-variant SD is 0.032 accuracy and the full model\n"
    "alone spans 0.053 across seeds (0.649 mean). Against that floor one\n"
    "effect survives --- replacing the vision encoder with a lightweight stack\n"
    "costs 0.227 --- and none of the remaining ablations moves the mean by\n"
    "more than 0.071, i.e." + B + " under two SDs. The expert-residual weight is not\n"
    "supported by the sweep at all: removing the residual entirely ($" + B + "lambda=0$,\n"
    "$-0.013$) beats both $" + B + "lambda=1$ ($-0.044$) and $" + B + "lambda=4$ ($-0.040$) relative to the\n"
    "$" + B + "lambda=2$ default, so the ordering is non-monotone and the setting is a\n"
    "choice rather than a tuned optimum. Read together with Table~" + R + "tab:multimodal_cent_fed}, where a frozen-encoder concatenation reaches 0.800 on the\n"
    "same support, the trained architecture is not what produces the multimodal\n"
    "result and is not claimed to be."
)

NEW_ABL = (
    "Eleven variants $" + B + "times$ 3 seeds on one device (33 runs) separate component\n"
    "effects from seed noise, which the earlier CPU-versus-H100 comparison\n"
    "confounded. The floor is high: pooled within-variant SD is 0.032 accuracy and\n"
    "the full model alone spans 0.053 across seeds (0.649 mean). One effect clears\n"
    "it --- a lightweight vision encoder costs 0.227 --- and no other ablation moves\n"
    "the mean by more than 0.071. The expert-residual weight is unsupported:\n"
    "$" + B + "lambda{=}0$ ($-0.013$) beats $" + B + "lambda{=}1$ ($-0.044$) and "
    "$" + B + "lambda{=}4$ ($-0.040$)\n"
    "against the $" + B + "lambda{=}2$ default, so the ordering is non-monotone. With a\n"
    "frozen-encoder concatenation reaching 0.800 on the same support\n"
    "(Table~" + R + "tab:multimodal_cent_fed}), the trained architecture is not what\n"
    "produces the multimodal result and is not claimed to be."
)

OLD_FIG = (
    "Figure~" + R + "fig:c_clubbed_fusion} redraws the fused rows as notes are\n"
    "deleted. Every ViT-Base pairing stays above the best text-only system at\n"
    "every sparsity (0.373 at 75" + B + "% deletion), so the fused ordering is\n"
    "stable even where the note channel has mostly collapsed. Which text partner\n"
    "is best is not stable: BERT-small leads at complete notes and BERT-mini at\n"
    "75" + B + "% deletion, and the spread across the five pairings widens from\n"
    "0.053 to 0.187. The figure therefore supports fusion over its parents, not\n"
    "a particular choice of text encoder --- 75 crops cannot separate those five."
)

NEW_FIG = (
    "Figure~" + R + "fig:c_clubbed_fusion} redraws the fused rows as notes are deleted.\n"
    "Every ViT-Base pairing stays above the best text-only system at every sparsity\n"
    "(0.373 at 75" + B + "% deletion), but which partner is best does not hold: BERT-small\n"
    "leads at complete notes, BERT-mini at 75" + B + "%, and the spread across the five\n"
    "widens from 0.053 to 0.187. The figure supports fusion over its parents, not a\n"
    "choice of text encoder."
)

OLD_FIVE = (
    "All 5 ViT-Base $+$ BERT rows clear the best\n"
    "unimodal system at complete notes (lowest 0.747 against 0.733), so the gain is not\n"
    "the property of one lucky pairing; the classical TF-IDF with colour/HOG fusion\n"
    "does not, which places the gain in the pretrained encoders rather than in concatenation itself."
)

NEW_FIVE = (
    "All five ViT-Base $+$ BERT rows clear the\n"
    "best unimodal system at complete notes (lowest 0.747 against 0.733), so the gain\n"
    "is not one lucky pairing; the classical TF-IDF with colour/HOG fusion does not,\n"
    "placing the gain in the pretrained encoders rather than in concatenation."
)


def main() -> None:
    for tex in TEXS:
        p = ROOT / tex
        s = p.read_text(encoding="utf-8")
        notes = []
        for tag, old, new in (("ablation", OLD_ABL, NEW_ABL),
                              ("figure", OLD_FIG, NEW_FIG),
                              ("five-rows", OLD_FIVE, NEW_FIVE)):
            saved = old.count("\n") - new.count("\n")
            if s.count(old) == 1:
                s = s.replace(old, new)
                notes.append(f"{tag} (-{saved} lines)")
            elif new.split("\n")[0][:40] in s:
                notes.append(f"{tag}: already compressed")
            else:
                notes.append(f"{tag}: ANCHOR x{s.count(old)}")
        p.write_text(s, encoding="utf-8")
        print(f"  {tex}: " + "; ".join(notes))


if __name__ == "__main__":
    main()
