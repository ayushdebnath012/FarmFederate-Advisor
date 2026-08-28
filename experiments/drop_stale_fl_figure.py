#!/usr/bin/env python3
"""Drop Fig. 6 (federated scaling) to reach the eight-page limit.

It is the right float to lose on three counts, not just size:

  correctness   its three panels are plot73/74/77, rendered 2026-07-29, a month
                before the note correction. They still draw the 0.89--0.92
                curves of the pre-correction checkpoint, so the figure
                contradicts the paragraph beside it, which was rewritten to
                0.58--0.65. Fixing it would mean regenerating three plots to
                keep a float the page budget cannot afford.
  redundancy    client scaling is in Table III and realized label skew is now
                Fig. 8(a); only rounds-vs-performance was unique to it.
  cost          it is the largest float in the paper, a figure* carrying three
                subfigures, and it is cited exactly once.

The discussion paragraph stays -- its numbers come from the rerun and are
correct -- with the figure reference replaced by the sweep it describes.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEXS = ["overleaf_final/main.tex", "overleaf_final_slim/main.tex"]
B = chr(92)

OPEN = B + "begin{figure*}"
CLOSE = B + "end{figure*}"
LABEL = B + "label{fig:fl_scaling}"

OLD_REF = ("Figure~" + B + "ref{fig:fl_scaling} answers the two federated scaling questions\n"
           "directly.")
NEW_REF = "The client and round sweep answers the two federated scaling questions directly."


def main() -> None:
    for tex in TEXS:
        p = ROOT / tex
        s = p.read_text(encoding="utf-8")
        notes = []
        if LABEL in s:
            i = s.index(LABEL)
            a = s.rindex(OPEN, 0, i)
            b = s.index(CLOSE, i) + len(CLOSE)
            # swallow the blank line the float leaves behind
            while s[b:b + 1] == "\n":
                b += 1
            removed = b - a
            s = s[:a] + s[b:]
            notes.append(f"float removed ({removed} chars)")
        else:
            notes.append("float already gone")

        if NEW_REF[:40] in s:
            notes.append("reference already retargeted")
        elif s.count(OLD_REF) == 1:
            s = s.replace(OLD_REF, NEW_REF)
            notes.append("reference retargeted")
        else:
            notes.append(f"reference ANCHOR x{s.count(OLD_REF)}")

        p.write_text(s, encoding="utf-8")
        print(f"  {tex}: " + "; ".join(notes))


if __name__ == "__main__":
    main()
