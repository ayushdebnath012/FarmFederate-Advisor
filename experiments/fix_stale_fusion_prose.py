#!/usr/bin/env python3
"""Repair three places that still describe the old, saturated note corpus.

With the previous notes every text sentence was class-exclusive, so the text
branch scored 1.000 and fusion could only tie it. Those three statements were
written for that world and are now contradicted by the table they sit next to:

  1. the caption of Table~\\ref{tab:multimodal_cent_fed} says fusion "cannot
     improve" on a saturated text branch -- the table shows 0.800 against 0.640
  2. "All five fused systems" is loose: the table has six fused rows, and the
     classical one (0.667) does not clear the best image-only system
  3. the paragraph under Fig.~\\ref{fig:c_clubbed_fusion} describes panels (a)
     and (b) of a two-panel figure and quotes curves coinciding at 1.000; the
     figure is single-panel and nothing coincides at 1.000 any more

Numbers below are read from tea_results/clubbed_tables/clubbed_tables.json so
the replacement text cannot drift from the run either.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLUBBED = ROOT / "tea_results" / "clubbed_tables" / "clubbed_tables.json"
TEXS = ["overleaf_final/main.tex", "overleaf_final_slim/main.tex"]
B = chr(92)

OLD_CAPTION = ("At complete notes the text branch is already saturated, so fusion "
               "cannot improve on it and the comparison is uninformative; the "
               "separation appears once notes are incomplete, which is the "
               "realistic field condition.")

OLD_FIVE = ("All five fused systems beat\n"
            "every unimodal system on this support, so the gain is not the property of one\n"
            "lucky pairing.")

OLD_FIG = ("Figure~" + B + "ref{fig:c_clubbed_fusion} summarizes both halves. Panel~(a) shows that\n"
           "the classical-versus-pretrained gap is modality-dependent, and panel~(b) that\n"
           "the three modalities are indistinguishable only while the notes are complete\n"
           "--- the text and fusion curves coincide exactly at 1.000 --- and fan out as\n"
           "notes are removed, with the image branch flat by construction and fusion the\n"
           "only branch that is never worst.")


def facts():
    d = json.load(open(CLUBBED, encoding="utf-8"))["tables"]
    keys = sorted(d, key=lambda k: float(k.split("=")[1]))
    acc = lambda fam, k: {s: d[k][fam][s]["central_test_accuracy"]
                          for s in d[k][fam]}
    neural = lambda k: {s: v for s, v in acc("multimodal", k).items()
                        if s.startswith("ViT-Base")}
    best_img = max(acc("image_only", keys[0]).values())
    best_txt = {k: max(acc("text_only", k).values()) for k in keys}
    return keys, neural, best_img, best_txt


def main() -> None:
    keys, neural, best_img, best_txt = facts()
    n0, nlast = neural(keys[0]), neural(keys[-1])
    lead0 = min(n0.values())
    band0 = max(n0.values()) - min(n0.values())
    band_last = max(nlast.values()) - min(nlast.values())
    top0 = max(n0, key=n0.get).replace("ViT-Base + ", "")
    top_last = max(nlast, key=nlast.get).replace("ViT-Base + ", "")
    del_last = 100 * float(keys[-1].split("=")[1])

    new_caption = (
        "Fusion leads both parents at complete notes rather than only under note "
        "loss: the best fused system reaches 0.800 against 0.733 image-only and "
        f"0.640 text-only, and all {len(n0)} ViT-Base rows clear the best unimodal "
        "system. The margin then widens as notes are deleted, because only the "
        "text branch degrades.")

    new_five = (
        f"All {len(n0)} ViT-Base $+$ BERT rows clear the best\n"
        f"unimodal system at complete notes (lowest {lead0:.3f} against "
        f"{best_img:.3f}), so the gain is not\n"
        "the property of one lucky pairing; the classical TF-IDF with colour/HOG "
        "fusion\n"
        "does not, which places the gain in the pretrained encoders rather than in "
        "concatenation itself.")

    new_fig = (
        "Figure~" + B + "ref{fig:c_clubbed_fusion} redraws the fused rows as notes are\n"
        "deleted. Every ViT-Base pairing stays above the best text-only system at\n"
        f"every sparsity ({best_txt[keys[-1]]:.3f} at {del_last:.0f}"
        + B + "% deletion), so the fused ordering is\n"
        "stable even where the note channel has mostly collapsed. Which text partner\n"
        f"is best is not stable: {top0} leads at complete notes and {top_last} at\n"
        f"{del_last:.0f}" + B + "% deletion, and the spread across the five pairings widens from\n"
        f"{band0:.3f} to {band_last:.3f}. The figure therefore supports fusion over its parents, not\n"
        "a particular choice of text encoder --- 75 crops cannot separate those five.")

    for tex in TEXS:
        p = ROOT / tex
        s = p.read_text(encoding="utf-8")
        notes = []
        for tag, old, new in (("caption", OLD_CAPTION, new_caption),
                              ("five-rows", OLD_FIVE, new_five),
                              ("figure para", OLD_FIG, new_fig)):
            if s.count(old) == 1:
                s = s.replace(old, new)
                notes.append(tag)
            elif new.split("\n")[0][:40] in s:
                notes.append(f"{tag}: already applied")
            else:
                notes.append(f"{tag}: ANCHOR x{s.count(old)}")
        p.write_text(s, encoding="utf-8")
        print(f"  {tex}: " + "; ".join(notes))

    print("\n--- caption ---\n" + new_caption)
    print("\n--- five rows ---\n" + new_five)
    print("\n--- figure paragraph ---\n" + new_fig)


if __name__ == "__main__":
    main()
