#!/usr/bin/env python3
"""Final compressions for the eight-page limit: intro, related work, ladders.

No claim, citation or number is dropped. These three paragraphs state the same
thing twice in places -- the intro says field notes exist alongside the image
and then that FL work has not integrated them; the ladder paragraph explains
"fixes the text branch of the system the paper actually retains, so it answers
what pipeline engineering would buy on the model being shipped rather than on an
arbitrary encoder", which is one idea in twenty-eight words.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEXS = ["overleaf_final/main.tex", "overleaf_final_slim/main.tex"]
B = chr(92)
C = B + "cite{"

JOBS = [
    ("intro",
     "but nearly all begin and end with the photograph. That is not how tea diagnosis\n"
     "is recorded on the ground, where staff note lesion shape, flush condition,\n"
     "weather history, and pest pressure alongside the image~" + C + "Pandey2021TeaFungal,Sinniah2024TeaDiagnosis}. Meanwhile, sending every\n"
     "high-resolution image and field note to a central repository is unattractive for\n"
     "estates with limited connectivity and proprietary records~" + C + "Dembani2025FLPrivacy,Puppala2025SelfReg}. Federated\n"
     "Learning~" + C + "McMahan2017} addresses the second problem, but existing FL work on\n"
     "tea pathology has not integrated the visual and textual evidence used in real\n"
     "inspections.",

     "but nearly all begin and end with the photograph, whereas staff record lesion\n"
     "shape, flush condition, weather history and pest pressure alongside\n"
     "it~" + C + "Pandey2021TeaFungal,Sinniah2024TeaDiagnosis}. Centralizing every image and\n"
     "note is also unattractive for estates with limited connectivity and proprietary\n"
     "records~" + C + "Dembani2025FLPrivacy,Puppala2025SelfReg}; Federated\n"
     "Learning~" + C + "McMahan2017} addresses that, but FL work on tea pathology has not\n"
     "integrated the two kinds of evidence.",
     "whereas staff record lesion"),

    ("ladders",
     B + "emph{Held fixed.} The encoder. The three pipeline ladders vary the pipeline\n"
     "and hold the model still: the visual ladder fixes the\n"
     "ResNet-50 image path with frozen ViT-tiny as a drop-in rung; the text ladder\n"
     "fixes DistilBERT, the text branch of the system the paper actually retains, so\n"
     "it answers what pipeline engineering would buy on the model being shipped\n"
     "rather than on an arbitrary encoder; and the fusion ladder fixes ViT-Base/16\n"
     "with DistilBERT, the pair validation selects, ViT-Base leading the five vision\n"
     "encoders at 0.734 validation macro F1 (Fig.~" + B + "ref{fig:c_selection_gap}). Starting from the strongest available\n"
     "pair is what makes a flat result meaningful --- a pipeline that cannot help the\n"
     "best pair will not help a worse one. The gate rungs are the one exception: they\n"
     "route among several encoders, which is why they are marked as diagnostics.",

     B + "emph{Held fixed.} The encoder. The three pipeline ladders vary the pipeline and\n"
     "hold the model still: the visual ladder fixes the ResNet-50 image path with\n"
     "frozen ViT-tiny as a drop-in rung, the text ladder fixes DistilBERT, and the\n"
     "fusion ladder fixes the pair validation selects, ViT-Base/16 with DistilBERT\n"
     "(ViT-Base leads the five vision encoders at 0.734 validation macro F1,\n"
     "Fig.~" + B + "ref{fig:c_selection_gap}). Starting from the strongest pair is what makes a\n"
     "flat result meaningful: a pipeline that cannot help the best pair will not help\n"
     "a worse one. Gate rungs route among several encoders and are marked diagnostics.",
     "Starting from the strongest pair is what makes a"),

    ("related work",
     "Their\n"
     "recurring weakness is the text side: notes generated from the class label or\n"
     "paired by class matching are partial copies of the target. Ours is templated too,\n"
     "which is why pairing is fixed at the exact box index and the shortcut vocabulary\n"
     "is fitted on training text alone. The gap is exactly this: paired image--text FL with a source-disjoint\n"
     "leakage audit.",

     "Their recurring\n"
     "weakness is the text side: notes generated from the class label or paired by\n"
     "class matching are partial copies of the target. Ours is templated too, which is\n"
     "why pairing is fixed at the exact box index and the shortcut vocabulary fitted on\n"
     "training text alone. The gap is paired image--text FL with a source-disjoint\n"
     "leakage audit.",
     "shortcut vocabulary fitted on"),
]


def main() -> None:
    for tex in TEXS:
        p = ROOT / tex
        s = p.read_text(encoding="utf-8")
        notes, saved = [], 0
        for tag, old, new, marker in JOBS:
            if marker in s:
                notes.append(f"{tag}: already")
            elif s.count(old) == 1:
                saved += len(old) - len(new)
                s = s.replace(old, new)
                notes.append(f"{tag}: ok")
            else:
                notes.append(f"{tag}: ANCHOR x{s.count(old)}")
        p.write_text(s, encoding="utf-8")
        print(f"  {tex}: " + "; ".join(notes) + f"  (-{saved} chars)")


if __name__ == "__main__":
    main()
