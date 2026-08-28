#!/usr/bin/env python3
"""Close the last five pre-correction passages in the manuscript.

Found by grepping the compiled PDF for the old run's fingerprints rather than
the source, so passages inside \\iffalse regions are correctly ignored.

  1  Fig. 6 caption      dotted initialization line, 0.9099 -> retrained value
  2  image-only gate     the five-expert ensemble numbers are from the old
                         checkpoint; the surviving selector is vision_tta.py,
                         whose search is over 25 view/scale candidates, so the
                         sentence is restated for what actually ran
  3  cross-modal means   mean reliability was 0.575 text / 0.425 image; the
                         retrained gate is image-dominant at 0.263 / 0.737
  4  corruption prose    two paragraphs whose numbers and whose conclusion both
                         change. The reliability finding inverts outright: the
                         old gate down-weighted text when text was deleted
                         (0.575 -> 0.481), the retrained one weights it *more*
                         (0.263 -> 0.361), so the gate is not reacting to the
                         corruption in the direction the design intends. That
                         is worth stating plainly rather than dropping.
  5  conclusion          the matched-suite averages

Every number is read from the result files.
"""

from __future__ import annotations

import json
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
V6 = ROOT / "tea_results" / "multimodal_v6_vision_full_20260727"
CORR = V6 / "corruption" / "modality_corruption_results.json"
TTA = V6 / "ensemble5_vit" / "vision_tta_results.json"
FARM = ROOT / "experiments" / "farm_results_full_genuine.json"
TEXS = ["overleaf_final/main.tex", "overleaf_final_slim/main.tex"]
B = chr(92)
R = B + "ref{"


def main() -> None:
    corr = {r["condition"]: r for r in
            json.loads(CORR.read_text(encoding="utf-8"))["conditions"]}
    tta = json.loads(TTA.read_text(encoding="utf-8"))
    base = json.loads((V6 / "complete_results.json").read_text(encoding="utf-8"))
    init = base["results"]["vlm"]["central_f1"]
    cm = base["results"]["vlm"]["cross_modal_analysis"]
    cond = cm["conditions"]

    clean, kill = corr["Clean"], corr["Text deletion 100"]
    zero, uni = corr["Image zeroed"], corr["Image uniform noise"]
    g35 = corr["Image Gaussian $" + B + "sigma{=}0.35$"]
    rt = lambda r: r["mean_text_weight"]

    farm = json.loads(FARM.read_text(encoding="utf-8"))
    cells = defaultdict(list)
    for k, v in farm["E1_alpha_sweep"].items():
        p = dict(x.split("=", 1) for x in k.split("|") if "=" in x)
        if p.get("split") == "corrected":
            cells[p["alpha"]].append(v["f1_macro"])

    fixes = [
        ("Fig. 6 caption",
         "fixed. The dotted line in (a) is the 0.9099 initialization, and (c) is the",
         f"fixed. The dotted line in (a) is the {init:.4f} initialization, and (c) is the"),

        ("image-only gate",
         "The validation-selected five-expert gate raises image accuracy from 61.33" + B + "%\n"
         "(46/75) to " + B + "textbf{72.00" + B + "%} (54/75; macro F1 0.6759). This is an ensemble, not\n"
         "the 66.67" + B + "% pure-ViT baseline; rare-class behavior remains unstable.",
         "A validation-selected test-time-augmentation search over "
         f"{tta['candidate_count']} view/scale\n"
         f"candidates reaches {100 * tta['selected']['accuracy']:.2f}" + B + "% validation accuracy "
         f"(macro F1 {tta['selected']['macro_f1']:.4f}) with the\n"
         # view names carry underscores, which are LaTeX specials
         + ", ".join(v.replace("_", "-") for v in tta["selected"]["views"])
         + " views, and "
         f"{100 * tta['locked_test']['accuracy']:.2f}" + B + "% on the locked test "
         f"(macro F1 {tta['locked_test']['macro_f1']:.4f}).\n"
         "This is an inference-time ensemble over one backbone, not a stronger encoder;\n"
         "rare-class behaviour remains unstable."),

        ("cross-modal means",
         "Mean reliability is 0.575 text/0.425 image; mismatching text costs 58.67 accuracy\n"
         "points and 0.5949 F1.",
         f"Mean reliability is {rt(clean):.3f} text/{1 - rt(clean):.3f} image --- the retrained gate is\n"
         "image-dominant --- and mismatching the note costs only "
         f"{100 * (cond['paired']['accuracy'] - cond['mismatched_text']['accuracy']):.2f} accuracy\n"
         f"points and {cond['paired']['f1_macro'] - cond['mismatched_text']['f1_macro']:.4f} F1."),

        ("corruption para 1",
         B + "textbf{Both branches are load-bearing.} Table~" + R + "tab:corruption} reports the\n"
         "sweep. Deleting every attended text token costs 0.551 macro F1 and moves 46 of 75\n"
         "predictions; replacing the image with uniform noise costs 0.222 and moves 27. The\n"
         "reliability weight is not constant: mean $w_t$ is 0.575 on clean input, 0.481\n"
         "under complete text deletion, and 0.615 under complete image noise. The note\n"
         "carries more of the decision than the image, but neither branch is inert. No\n"
         "ordering of modalities or architectures is advanced from a single 75-crop\n"
         "support.",
         B + "textbf{Both branches are load-bearing.} Table~" + R + "tab:corruption} reports the\n"
         f"sweep. Deleting every attended text token costs {clean['macro_f1'] - kill['macro_f1']:.3f} macro F1 and moves\n"
         f"{kill['flips']} of 75 predictions; replacing the image with uniform noise costs "
         f"{clean['macro_f1'] - uni['macro_f1']:.3f} and\n"
         f"moves {uni['flips']}. The image now carries more of the decision than the note, reversing\n"
         "the previous corpus, but neither branch is inert. The reliability gate does not\n"
         f"track this: mean $w_t$ is {rt(clean):.3f} on clean input and rises to {rt(kill):.3f} under\n"
         f"complete text deletion and {rt(uni):.3f} under complete image noise, so it weights a\n"
         "destroyed modality more rather than less. No ordering of modalities or\n"
         "architectures is advanced from a single 75-crop support."),

        ("corruption para 2",
         B + "textbf{Ablation strength determines the verdict.} Additive Gaussian noise at\n"
         "$" + B + "sigma{=}0.35$ leaves leaf and lesion plainly visible and moves only 9 of 75\n"
         "predictions, whereas zeroing the image moves 18 and replacing it with uniform\n"
         "noise moves 27.",
         B + "textbf{Ablation strength determines the verdict.} Additive Gaussian noise at\n"
         f"$" + B + f"sigma{{=}}0.35$ leaves leaf and lesion plainly visible and moves {g35['flips']} of 75\n"
         f"predictions, whereas zeroing the image moves {zero['flips']} and replacing it with uniform\n"
         f"noise moves {uni['flips']}."),

        ("conclusion",
         "suite, FedAvg averages 0.816/0.892 across the five architectures at",
         f"suite, FedAvg averages {st.mean(cells['0.1']):.3f}/{st.mean(cells['1.0']):.3f} "
         "across the five architectures at"),
    ]

    for tex in TEXS:
        p = ROOT / tex
        s = p.read_text(encoding="utf-8")
        notes = []
        for tag, old, new in fixes:
            # the longest line, not the first: several replacements keep their
            # opening line, so a first-line check reports a false "already"
            marker = max(new.split("\n"), key=len)[:50]
            if marker in s:
                notes.append(f"{tag}: already")
            elif s.count(old) == 1:
                s = s.replace(old, new)
                notes.append(f"{tag}: ok")
            else:
                notes.append(f"{tag}: x{s.count(old)}")
        p.write_text(s, encoding="utf-8")
        print(f"  {tex}:\n    " + "\n    ".join(notes))


if __name__ == "__main__":
    main()
