#!/usr/bin/env python3
"""Rewrite the corruption paragraph; the corrected corpus inverts its claim.

The old paragraph argued that the text path alone was the configuration to
deploy and that "the image earns its place only where the note is incomplete".
On the corrected corpus the probe says the opposite: zeroing the image is the
second most damaging condition tested and costs more than deleting every note
token, while the note channel can lose three quarters of its tokens for about
one accuracy point. The image is load-bearing at every note condition.

Numbers come from the corruption probe and the modality hierarchy, both rerun
against the retrained checkpoint.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORR = (ROOT / "tea_results" / "multimodal_v6_vision_full_20260727"
        / "corruption" / "modality_corruption_results.json")
HIER = (ROOT / "tea_results" / "multimodal_v6_vision_full_20260727"
        / "modality_hierarchy" / "modality_hierarchy_results.json")
TEXS = ["overleaf_final/main.tex", "overleaf_final_slim/main.tex"]
B = chr(92)
R = B + "ref{"

OLD = (
    "Fusion is not uniformly worth its cost, and the two blocks say when it is. On\n"
    "complete notes it does not beat its strongest parent --- 69/75 against 72/75 for\n"
    "text --- so the text path alone is the configuration to deploy, and mismatching\n"
    "the note drops it from 92.00" + B + "% to 33.33" + B + "%, confirming cross-modal dependence.\n"
    "Masking half the note tokens reverses the ordering: text alone falls to 64.00" + B + "%\n"
    "while the router holds 81.33" + B + "%. The image earns its place only where the note is\n"
    "incomplete, which is the condition field notes are written under. With no note at\n"
    "all, the calibrated visual base of Table~" + R + "fig:c_fusion_ladder} gives 68.00" + B + "%.\n"
    "Two limits apply: notes are templated, so the complete-note block is the most\n"
    "inflated, and the routing family was refined on this partition, with three\n"
    "further hierarchy configurations placing the routed row between 73.33" + B + "% and\n"
    "81.33" + B + "%."
)


def main() -> None:
    c = json.loads(CORR.read_text(encoding="utf-8"))
    h = json.loads(HIER.read_text(encoding="utf-8"))
    by = {r["condition"]: r for r in c["conditions"]}
    n = c["test_crops"]
    clean, kill, d75 = by["Clean"], by["Text deletion 100"], by["Text deletion 75"]
    zero, uni = by["Image zeroed"], by["Image uniform noise"]
    g26 = by["Image Gaussian $" + B + "sigma{=}0.26$"]
    t = h["fixed_test"]

    new = (
        "Both branches carry evidence, and the corrected corpus reverses which one\n"
        "carries more. Deleting every note token costs "
        f"{100 * (clean['accuracy'] - kill['accuracy']):.2f} accuracy points and moves\n"
        f"{kill['flips']} of {n} predictions; zeroing the image costs "
        f"{100 * (clean['accuracy'] - zero['accuracy']):.2f} points and moves {zero['flips']},\n"
        "so the image is the more load-bearing modality, not the one that earns its\n"
        "place only when the note is thin. The note channel is also the more\n"
        f"redundant of the two: three quarters of its tokens can go for "
        f"{100 * (clean['accuracy'] - d75['accuracy']):.2f} points\n"
        f"({d75['flips']} flips), and the collapse arrives only when the last quarter does. On the\n"
        "image side the model absorbs additive sensor noise --- "
        f"$" + B + f"sigma{{=}}0.26$ leaves accuracy at {g26['accuracy']:.4f} ---\n"
        "but not loss of content: noise with matched statistics is the worst condition\n"
        f"tested at {uni['accuracy']:.4f} and {uni['flips']} flips. Unimodally the same ordering holds on this\n"
        f"test, image-only {t['image_only']['accuracy']:.4f} against text-only "
        f"{t['text_only']['accuracy']:.4f}. Two limits apply: notes are\n"
        "templated, so every note-side number is an upper bound, and the routing family\n"
        "was refined on this partition."
    )

    for tex in TEXS:
        p = ROOT / tex
        s = p.read_text(encoding="utf-8")
        if "reverses which one" in s:
            print(f"  {tex}: already rewritten")
            continue
        if s.count(OLD) != 1:
            print(f"  {tex}: ANCHOR x{s.count(OLD)} -- not applied")
            continue
        p.write_text(s.replace(OLD, new), encoding="utf-8")
        print(f"  {tex}: rewritten ({OLD.count(chr(10))} -> {new.count(chr(10))} lines)")
    print("\n" + new)


if __name__ == "__main__":
    main()
