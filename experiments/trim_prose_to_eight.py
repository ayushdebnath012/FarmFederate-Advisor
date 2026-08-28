#!/usr/bin/env python3
"""Two compressions that are also corrections, for the eight-page limit.

1. The fusion-cost paragraph restates the corruption probe in full, a column
   before the corruption subsection does the same against Table VII. Keeping
   both is redundant; the earlier one becomes the claim plus a pointer.

2. The "Held fixed. The method." paragraph justifies the frozen-ridge protocol
   with three numbers that no longer hold: a 0.932-against-0.747 gate collapse
   and a 0.920 calibrated late fusion, both from the pre-correction ladders
   (which are also \\iffalse-excluded, so the reader cannot check them). On the
   corrected corpus the best fusion rung is 0.773 while plain concatenation
   reaches 0.800 with a stronger text encoder, so "plain concatenation is the
   weakest rung, therefore this is a lower bound" is no longer an argument the
   data supports. The reasons that survive -- isolation, determinism, minimal
   search -- are kept; the lower-bound claim goes.

Both edits shorten the paper, but neither is made for length alone.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEXS = ["overleaf_final/main.tex", "overleaf_final_slim/main.tex"]
B = chr(92)
R = B + "ref{"

OLD_FUSE = (
    "Both branches carry evidence, and neither dominates. Deleting every note token\n"
    "costs 26.67 accuracy points and moves 41 of 75 predictions; zeroing the image\n"
    "costs 25.33 points and moves 49. The two ablations are a wash on accuracy, but\n"
    "the image ablation collapses macro-F1 much further (0.160 against 0.290) by\n"
    "pushing the model onto a few classes, so the image is not the accessory the\n"
    "previous corpus made it look. The note channel is the more redundant of the\n"
    "two: three quarters of its tokens can go for 2.67 points (11 flips), and the\n"
    "collapse arrives only when the last quarter does. On the\n"
    "image side the model absorbs additive sensor noise --- $" + B + "sigma{=}0.26$ leaves accuracy at 0.6267 ---\n"
    "but not loss of content: noise with matched statistics is the worst condition\n"
    "tested at 0.2800 and 61 flips. Unimodally the image branch leads clearly on this\n"
    "test, 0.6400 against 0.2400. Two limits apply: notes are\n"
    "templated, so every note-side number is an upper bound, and the routing family\n"
    "was refined on this partition."
)
NEW_FUSE = (
    "Both branches carry evidence and neither dominates: the image is not the\n"
    "accessory the previous corpus made it look, and unimodally it leads this test\n"
    "0.6400 against 0.2400. Section~" + R + "sec:modality_inertness} and\n"
    "Table~" + R + "tab:corruption} quantify both ablations. Two limits apply: notes are\n"
    "templated, so every note-side number is an upper bound, and the routing family\n"
    "was refined on this partition."
)

OLD_METHOD = (
    B + "emph{Determinism:} a closed-form solve\n"
    "has one exact solution, no seed and no initialization, so a gap between two\n"
    "rows is the encoder and not luck, which the TTA and gate rungs of\n"
    "Table~" + R + "fig:c_fusion_ladder} cannot promise. " + B + "emph{Minimal search:} one penalty\n"
    "out of six, against the 243--3125 candidates of the ladders; on 74 validation\n"
    "crops that difference is precisely the 0.185 validation-to-test collapse\n"
    "those gates exhibit (0.932 against 0.747). " + B + "emph{Lower bound:} plain\n"
    "concatenation is the weakest fusion rung of Table~" + R + "fig:c_fusion_ladder}, so a\n"
    "multimodal gain reported here understates what better fusion achieves, which\n"
    "is why the discussion below points forward to the 0.920 of calibrated late\n"
    "fusion rather than treating the concatenation number as a ceiling. The varied"
)
NEW_METHOD = (
    B + "emph{Determinism:} a closed-form solve\n"
    "has one exact solution, no seed and no initialization, so a gap between two\n"
    "rows is the encoder and not luck, which the TTA and gate rungs of\n"
    "Fig.~" + R + "fig:c_fusion_ladder} cannot promise. " + B + "emph{Minimal search:} one penalty\n"
    "out of six, against the 243--3125 candidates of the ladders, which on 74\n"
    "validation crops is where those gates lose their selection advantage. The varied"
)


def main() -> None:
    jobs = [("fusion-cost", OLD_FUSE, NEW_FUSE, "quantify both ablations"),
            ("held-fixed method", OLD_METHOD, NEW_METHOD,
             "lose their selection advantage")]
    for tex in TEXS:
        p = ROOT / tex
        s = p.read_text(encoding="utf-8")
        notes, saved = [], 0
        for tag, old, new, marker in jobs:
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
