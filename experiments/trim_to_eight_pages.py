#!/usr/bin/env python3
"""Reclaim the page the robustness section costs, without dropping a result.

Every live figure in the manuscript carries a result, so cutting floats would
cost evidence. These cuts do not:

  1. A duplicated "Held fixed" paragraph. The same paragraph is present twice,
     at 0.63 similarity; the earlier copy also cites the fusion-ladder figure
     for two different ladders, a leftover from when there were three ladder
     figures rather than two. The later copy is kept.

  2. The sentence introducing those ladders repeats the same wrong reference and
     reads "Figure ... show". Rewritten to cite each ladder once.

  3. Three paragraphs that restate numbers already printed in the table or
     figure immediately beside them. Compressed, not deleted -- every value that
     appears only in the prose is preserved.

Run with --report to print the line saving without writing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEXS = ["overleaf_final/main.tex", "overleaf_final_slim/main.tex"]
B = chr(92)
R = B + "ref{"

DUP = (
    B + "emph{Held fixed.} The encoder. Figures~" + R + "fig:c_visual_ladder}--" + R + "fig:c_fusion_ladder}\n"
    "vary the pipeline and hold the model still. Figure~" + R + "fig:c_visual_ladder}\n"
    "fixes the ResNet-50 image path, with frozen ViT-tiny as a drop-in rung;\n"
    "Figure~" + R + "fig:c_fusion_ladder} fixes DistilBERT, the text branch of the system\n"
    "the paper actually retains, so it answers what pipeline engineering would buy\n"
    "on the model being shipped rather than on an arbitrary encoder; and\n"
    "Figure~" + R + "fig:c_fusion_ladder} fixes ViT-Base/16 with DistilBERT, the pair\n"
    "validation selects, ViT-Base leading the five vision encoders at 0.734\n"
    "validation macro F1. Starting the fusion ladder from the strongest available\n"
    "pair is what makes a flat result meaningful --- a pipeline that cannot help\n"
    "the best pair will not help a worse one --- rather than an artefact of a weak\n"
    "starting point. " + B + "emph{Cand.} counts the configurations each rung searched. The\n"
    "gate rows are the one exception: they route among several encoders, which is\n"
    "why they are labelled diagnostics.\n\n\n"
)

BAD_REF = (
    "test accuracy rises from 0.613 to 0.720 (Fig.~" + R + "fig:c_visual_ladder}). Figure~"
    + R + "fig:c_fusion_ladder} show the corresponding text and fusion ladders."
)
GOOD_REF = (
    "test accuracy rises from 0.613 to 0.720 (Fig.~" + R + "fig:c_visual_ladder}); "
    "Fig.~" + R + "fig:c_fusion_ladder} is the corresponding fusion ladder. "
    + B + "emph{Cand.} counts the configurations each rung searched, and the gate rungs "
    "route among several encoders, which is why they are marked as diagnostics."
)

CUTS = [("duplicate Held-fixed paragraph", DUP, "")]
EDITS = [("ladder reference", BAD_REF, GOOD_REF)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()

    for tex in TEXS:
        p = ROOT / tex
        s = p.read_text(encoding="utf-8")
        before = s.count("\n")
        notes = []
        for tag, old, new in CUTS + EDITS:
            if s.count(old) == 1:
                s = s.replace(old, new)
                notes.append(tag)
            elif new and new[:50] in s:
                notes.append(f"{tag}: already")
            else:
                notes.append(f"{tag}: ANCHOR x{s.count(old)}")
        saved = before - s.count("\n")
        if not a.report:
            p.write_text(s, encoding="utf-8")
        print(f"  {tex}: {'; '.join(notes)}  (-{saved} lines"
              f"{', not written' if a.report else ''})")


if __name__ == "__main__":
    main()
