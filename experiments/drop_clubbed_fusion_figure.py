#!/usr/bin/env python3
"""Drop Fig. 7; Table V already prints every value it plots.

Same rationale as Fig. 9: the figure redraws the six fused rows across the three
note sparsities, which is exactly the twelve columns of Table V sitting beside
it, and it is cited once. The finding the paragraph draws from it -- fusion beats
its parents at every sparsity, but which text partner wins does not hold -- is
kept in full; only the redundant plot goes.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEXS = ["overleaf_final/main.tex", "overleaf_final_slim/main.tex"]
B = chr(92)
R = B + "ref{"
LABEL = B + "label{fig:c_clubbed_fusion}"
BEGIN = "%%COMPACT-BEGIN%%"
END = "%%COMPACT-END%%"

OLD_PARA = (
    "Figure~" + R + "fig:c_clubbed_fusion} redraws the fused rows as notes are deleted.\n"
    "Every ViT-Base pairing stays above the best text-only system at every sparsity\n"
    "(0.373 at 75" + B + "% deletion), but which partner is best does not hold: BERT-small\n"
    "leads at complete notes, BERT-mini at 75" + B + "%, and the spread across the five\n"
    "widens from 0.053 to 0.187. The figure supports fusion over its parents, not a\n"
    "choice of text encoder."
)
NEW_PARA = (
    "Across the table every ViT-Base pairing stays above the best text-only system at\n"
    "every sparsity (0.373 at 75" + B + "% deletion), but which partner is best does not hold:\n"
    "BERT-small leads at complete notes, BERT-mini at 75" + B + "%, and the spread across the\n"
    "five widens from 0.053 to 0.187. The evidence supports fusion over its parents,\n"
    "not a choice of text encoder."
)


def main() -> None:
    for tex in TEXS:
        p = ROOT / tex
        s = p.read_text(encoding="utf-8")
        notes = []
        if LABEL in s:
            i = s.index(LABEL)
            a = s.rindex(BEGIN, 0, i)
            b = s.index(END, i) + len(END)
            while s[b:b + 1] == "\n":
                b += 1
            notes.append(f"figure removed ({b - a} chars)")
            s = s[:a] + s[b:]
        else:
            notes.append("figure already gone")

        if "Across the table every ViT-Base pairing" in s:
            notes.append("paragraph already retargeted")
        elif s.count(OLD_PARA) == 1:
            s = s.replace(OLD_PARA, NEW_PARA)
            notes.append("paragraph retargeted")
        else:
            notes.append(f"paragraph ANCHOR x{s.count(OLD_PARA)}")
        p.write_text(s, encoding="utf-8")
        print(f"  {tex}: " + "; ".join(notes))


if __name__ == "__main__":
    main()
