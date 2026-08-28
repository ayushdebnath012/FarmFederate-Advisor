#!/usr/bin/env python3
"""Re-apply the prose compressions lost with the deleted span.

The spliced-back text is the pre-compression version, so the paper runs one page
long again. These are the same condensations applied before: sentences whose
content a figure now carries, and paragraphs that restated numbers already in an
adjacent table.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
B = chr(92)
R = B + "ref{"
TEXS = ["overleaf_final/main.tex", "overleaf_final_slim/main.tex"]

FIXES = [
    # the gate walk-through is now the selection-gap figure
    ("The calibrated base wins on both sides of the split --- the validation score\n"
     "(0.771) and the highest single-model test accuracy (0.680, 51/75) --- on the\n"
     "smallest search in the table. The gates score higher, but " + B + "emph{Cand.} explains\n"
     "them: validation accuracy climbs\n"
     "monotonically with the routings fitted on 74 crops, 0.797 at 243 candidates to\n"
     "0.851 at 3125, while the validation-to-test drop stays near 13 points\n"
     "throughout. The five-expert gate converts 3125 fitted choices into three test\n"
     "crops (54/75 against 51/75), inside the $" + B + "pm0.10$ Wilson half-width here. We\n"
     "carry the calibrated base as the visual path and report the gate as a\n"
     "diagnostic ceiling.",
     "The calibrated base wins on both sides of the split on the smallest search\n"
     "(validation score 0.771, test accuracy 0.680). The gates score higher only by\n"
     "fitting more routings on 74 crops, and Fig.~" + R + "fig:c_selection_gap} shows what\n"
     "that costs. We carry the calibrated base as the visual path and report the\n"
     "gate as a diagnostic ceiling."),
    # the ladder discussion is now two figures
    ("Two differences from the visual case are immediate. First, at complete\n"
     "notes both ladders are flat: a single-candidate path already reaches 0.987\n"
     "accuracy for text and 1.000 for fusion, and searching 3125 or 243 candidates\n"
     "adds at most one crop, whereas the visual ladder gains 0.107 accuracy across\n"
     "the same span. The binding constraint on the text side is therefore the corpus,\n"
     "not the pipeline. Second, the ladders separate once notes are degraded, and the\n"
     "ordering is stable:",
     "Two differences from the visual case are immediate. At complete notes both\n"
     "ladders are flat --- a single-candidate path already reaches 0.987 for text\n"
     "and 1.000 for fusion, while the visual ladder gains 0.107 across the same\n"
     "span --- so the binding constraint on the note side is the corpus, not the\n"
     "pipeline. The ladders separate once notes are degraded, and the ordering is\n"
     "stable:"),
    # the advisory numbers are now in the retrieval figures
    ("Retrieval alone returns a correct-class passage for 90.05" + B + "% of queries (95" + B + "% CI\n"
     "85.57--94.03) against a 20.01" + B + "% chance rate, with precision@5 93.23" + B + "%, MRR\n"
     "0.9469, and NDCG@5 0.9621. The classify stage raises this to 95.52" + B + "% (95" + B + "% CI\n"
     "92.54--98.01), but that second number " + B + "emph{is} the classifier's accuracy:\n"
     "retrieval restricted to one predicted class can only return that class, so\n"
     "routing rather than ranking decides correctness.",
     "Retrieval reaches 90.05" + B + "% correct-class precision@1 against a 20.01" + B + "% chance\n"
     "rate, rising to 95.52" + B + "% with the classify stage. That second number\n"
     + B + "emph{is} the classifier's accuracy, because retrieval restricted to one\n"
     "predicted class can only return that class, so routing rather than ranking\n"
     "decides correctness."),
    ("Three limits apply. The corpus is template-composed, so base and queries share\n"
     "sentence vocabulary even though no observation is shared; these are upper bounds\n"
     "relative to free-form notes. The fusion re-ranker needs a fusion model on these\n"
     "five classes and is still unevaluated, as are the OBB overlay and the agronomic\n"
     "correctness of the advice itself. This is a retrieval result, not evidence that\n"
     "the recommended treatment is right.",
     "Three limits apply. The corpus is template-composed, so base and queries share\n"
     "sentence vocabulary --- visible in Fig.~" + R + "fig:c_retrieval_box} as uniformly\n"
     "high similarity --- and these are upper bounds relative to free-form notes.\n"
     "The fusion re-ranker, the OBB overlay and the agronomic correctness of the\n"
     "advice remain unevaluated. This is a retrieval result, not evidence that the\n"
     "treatment is right."),
    # the matched-suite paragraph restates the table beside it
    ("The suite runs on 371 genuine\n"
     "box-linked observations, and within each cell every architecture trains on the\n"
     "identical client partition, so differences are architectural rather than\n"
     "partition effects.",
     "The suite runs on 371 genuine box-linked observations with an identical client\n"
     "partition per cell, so differences are architectural."),
    ("Communication spans $3.5" + B + "times$, from 18.9" + B + ",MiB per client-round for the\n"
     "image-only model to 67.2" + B + ",MiB for cross-attention,",
     "Communication spans $3.5" + B + "times$, 18.9 to 67.2" + B + ",MiB per client-round,"),
    ("on a 53-crop validation split this indicates that the templated\n"
     "notes are close to label-determining rather than that federation is solved, and\n"
     "it bounds what the fusion comparison can establish on this corpus.",
     "on a 53-crop validation split that indicates the templated notes are close to\n"
     "label-determining rather than that federation is solved."),
    ("The anti-collapse and\n"
     "fusion-variant ablations remain single-architecture and are reported for the\n"
     "concat model only.",
     "The anti-collapse and fusion-variant ablations remain single-architecture."),
]


def main() -> None:
    for tex in TEXS:
        p = ROOT / tex
        s = p.read_text(encoding="utf-8")
        applied, absent = 0, []
        for old, new in FIXES:
            if s.count(old):
                s = s.replace(old, new)
                applied += 1
            else:
                absent.append(old.split("\n")[0][:44])
        p.write_text(s, encoding="utf-8")
        print(f"  {tex}: {applied}/{len(FIXES)} applied"
              + (f"; not found {absent}" if absent else ""))


if __name__ == "__main__":
    main()
