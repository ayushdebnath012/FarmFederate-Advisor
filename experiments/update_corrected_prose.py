#!/usr/bin/env python3
"""Rewrite the results prose for the corrected note corpus.

The old notes were class-exclusive at sentence level, so the text branch scored
1.000 and every multimodal claim had to be argued from note sparsity: fusion
could only tie a saturated parent. With notes that describe the right condition
and share sentences across classes, fusion beats both parents outright at
complete notes, which is a simpler and stronger claim.

Numbers below come from tea_results/aligned_support_tables/,
tea_results/clubbed_tables/ and the two ladder JSONs, all regenerated on the
corrected corpus. Image-only figures are unchanged by construction and were
verified to reproduce exactly.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
B = chr(92)
R = B + "ref{"
TEXS = ["overleaf_final/main.tex", "overleaf_final_slim/main.tex"]

FIXES = [
    # --- the headline unimodal paragraph -------------------------------------
    ("With complete notes the text branch is already at the ceiling: frozen\n"
     "DistilBERT reaches 1.000 while the best image encoder, ViT-Base, trails at\n"
     "0.733. Federating costs little on either side (1.000 to 0.987; 0.733 to 0.667),\n"
     "so the modality separates these systems, not the deployment mode. The classical\n"
     "rows locate where pretraining earns its place: decisive on images (0.440 for\n"
     "colour histograms with HOG against 0.733), and absent on text, where TF-IDF\n"
     "with a linear SVM is the strongest row at every note sparsity (1.000, 0.933,\n"
     "0.733). That is direct evidence the note channel is carried by lexical form\n"
     "rather than contextual meaning, consistent with Section~" + R + "sec:dataset}.",
     "With complete notes the image branch leads the text branch: ViT-Base reaches\n"
     "0.733 against 0.640 for the best text encoder, BERT-small. Federating costs\n"
     "little on either side (0.733 to 0.667; 0.640 to 0.582), so the modality\n"
     "separates these systems, not the deployment mode. The classical rows locate\n"
     "where pretraining earns its place: decisive on images (0.440 for colour\n"
     "histograms with HOG against 0.733) and absent on text, where TF-IDF with a\n"
     "linear SVM ties the best transformer at 0.640. Neither branch is close to the\n"
     "ceiling, so the comparison between them is informative rather than saturated."),

    # --- the headline multimodal paragraph -----------------------------------
    ("At complete notes fusion cannot demonstrate anything:\n"
     "ViT-Base with DistilBERT scores 1.000, the same as text alone, because there is\n"
     "no headroom above a saturated branch. The comparison becomes informative only\n"
     "once notes are incomplete --- the realistic field condition. At 75" + B + "% deletion\n"
     "fusion leads outright, 0.800 against 0.733 for the best image-only encoder and\n"
     "0.653 for the best text-only encoder, and the same holds encoder-for-encoder at\n"
     "50" + B + "% deletion (0.800 against 0.707 and 0.733). Both tables fuse by plain\n"
     "concatenation, the weakest rung of Fig.~" + R + "fig:c_fusion_ladder}; the\n"
     "deep-feature and calibrated late-fusion rungs reach 0.907 and 0.920 at 50" + B + "%\n"
     "deletion. Federation preserves the ordering rather than creating it: at 75" + B + "%\n"
     "deletion federated fusion holds 0.751 while federated text falls to 0.582.",
     "Fusion beats both parents outright, and does so at complete notes:\n"
     "ViT-Base with BERT-small reaches 0.800 against 0.733 for the best image-only\n"
     "encoder and 0.640 for the best text-only encoder. Every one of the five fused\n"
     "systems beats every unimodal system on this support, so the gain is not the\n"
     "property of one lucky pairing. The advantage widens as notes degrade, because\n"
     "only the text branch degrades: at 75" + B + "% deletion text falls to 0.373 while\n"
     "fusion holds 0.787 and the image branch is unmoved at 0.733 by construction.\n"
     "Federation preserves the ordering rather than creating it, and its cost falls\n"
     "as the note thins --- 0.800 to 0.667 at complete notes, 0.787 to 0.716 at\n"
     "75" + B + "% deletion."),

    # --- the ladder discussion ----------------------------------------------
    ("Two differences from the visual case are immediate. At complete notes both\n"
     "ladders are flat --- a single-candidate path already reaches 0.987 for text\n"
     "and 1.000 for fusion, while the visual ladder gains 0.107 across the same\n"
     "span --- so the binding constraint on the note side is the corpus, not the\n"
     "pipeline. The ladders separate once notes are degraded, and the ordering is\n"
     "stable:",
     "Neither ladder is flat now that the notes no longer determine the label. The\n"
     "best text rung reaches 0.613 at complete notes and 0.360 once three quarters\n"
     "of the tokens are gone, while the best fusion rung holds 0.773 and 0.773 over\n"
     "the same span: the fused pipeline is what survives note loss. The ordering is\n"
     "stable across rungs:"),

    # --- dataset section: the corpus is no longer label-determining ----------
    ("the 371\n"
     "crop-linked notes contain 742 sentences drawn from only 328 distinct ones, each\n"
     "reused 2.26 times on average, and " + B + "emph{no sentence crosses a class boundary},\n"
     "before or after masking. The 178-token filter removes shortcut words but not\n"
     "this sentence-level exclusivity, which is why a bag-of-words classifier still\n"
     "reaches 1.000 on these notes: the audit bounds the shortcut, it does not\n"
     "neutralize it. Every note-side number here therefore upper-bounds what\n"
     "free-form field notes would give.",
     "the 371\n"
     "crop-linked notes are assembled from a pool of 140 distinct sentences, 61" + B + "% of\n"
     "which occur in more than one class, and the diagnostic sign for a class is\n"
     "recorded in only 45" + B + "% of its notes --- a scout does not always see it. A\n"
     "bag-of-words classifier consequently reaches 0.640 rather than saturating, so\n"
     "the note channel is informative without being label-determining, and the\n"
     "comparison between modalities is not decided by the corpus."),

    # --- abstract -----------------------------------------------------------
    ("every record links to its exact box, and a\n"
     "training-fitted filter masks 178 target-shortcut tokens --- which bounds the\n"
     "note shortcut without removing it, since every note sentence remains\n"
     "class-exclusive.",
     "every record links to its exact box, and a\n"
     "training-fitted filter masks 45 target-shortcut tokens; 61" + B + "% of note sentences\n"
     "occur in more than one class, so the notes inform the label without\n"
     "determining it."),
]


def main() -> None:
    for tex in TEXS:
        p = ROOT / tex
        s = p.read_text(encoding="utf-8")
        ok, miss = 0, []
        for old, new in FIXES:
            if s.count(old) == 1:
                s = s.replace(old, new)
                ok += 1
            else:
                miss.append(old.split("\n")[0][:46])
        p.write_text(s, encoding="utf-8")
        print(f"  {tex}: {ok}/{len(FIXES)} applied")
        for m in miss:
            print(f"      not matched: {m!r}")


if __name__ == "__main__":
    main()
