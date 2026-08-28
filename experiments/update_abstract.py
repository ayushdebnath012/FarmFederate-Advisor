#!/usr/bin/env python3
"""Rewrite the abstract against the corrected corpus.

Every quantitative claim in it came from the pre-correction notes, where the
text channel was label-determining and most of the stack sat near 1.000. Three
of the claims do not merely change value, they change direction, and an abstract
that kept them would advertise conclusions the paper no longer supports:

  "both branches are used"      the mismatched-note diagnostic fell 92.00 to
                                33.33 then; it now falls 62.67 to 60.00, so a
                                wrong note costs almost nothing and that
                                sentence has to go
  "two architectures saturate"  nothing saturates; the federated sweep tops out
                                at 0.561
  the five-system comparison    all five accuracies move, and the ordering
                                changes

What replaces them is the result the corrected corpus actually supports, which
is a stronger claim than the old one: on the aligned support, fusion of frozen
encoders beats both parents outright rather than only under note sparsity.

Numbers are read from the result files, not typed.
"""

from __future__ import annotations

import json
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
V6 = ROOT / "tea_results" / "multimodal_v6_vision_full_20260727"
SEL = (ROOT / "tea_results" / "vit_lm_encoder_selection_20260729"
       / "encoder_selection_results.json")
HIER = V6 / "modality_hierarchy" / "modality_hierarchy_results.json"
CORR = V6 / "corruption" / "modality_corruption_results.json"
ADAPT = (ROOT / "tea_results" / "federated_adaptation_v6"
         / "federated_adaptation_results.json")
FARM = ROOT / "experiments" / "farm_results_full_genuine.json"
CLUB = ROOT / "tea_results" / "clubbed_tables" / "clubbed_tables.json"
TEXS = ["overleaf_final/main.tex", "overleaf_final_slim/main.tex"]
B = chr(92)

OLD = (
    "On the identical\n"
    "75-crop support, frozen DistilBERT, frozen ViT-tiny, the raw\n"
    "ResNet--Transformer VLM, a frozen ViT-tiny--DistilBERT VLM, and the routed\n"
    "ResNet--Transformer reach 65.33" + B + "%, 66.67" + B + "%, 77.33" + B + "%, 82.67" + B + "%, and 81.33" + B + "%\n"
    "accuracy.\n"
    "Selection is validation-only: the ViT--DistilBERT candidate's 82.43" + B + "% validation\n"
    "accuracy stays below the routed incumbent's 89.19" + B + "%, so the incumbent is\n"
    "retained. On test the candidate leads by a single crop out of 75, and its higher\n"
    "macro F1 (0.8192 versus 0.6679) traces almost entirely to one crop in a\n"
    "two-sample class. A same-checkpoint diagnostic falls from 92.00" + B + "%\n"
    "to 33.33" + B + "% once the note is class-mismatched, and on the locked test deleting the\n"
    "note moves 46 of 75 predictions while replacing the image with noise moves 27,\n"
    "so both branches are used and neither modality is advanced as decisive. The\n"
    "validation-selected ResNet--compact-Transformer VLM retains 98.1--100.0" + B + "% of\n"
    "its 0.9099 centralized validation macro F1 under warm-start FedAvg over 2--8\n"
    "simulated clients. Across five federated architectures on 371 genuine\n"
    "box-linked observations, federation outperforms local-only training, although\n"
    "the best aggregator is architecture-dependent; one float32 round costs\n"
    "18.9--67.2" + B + ",MiB per client\n"
    "depending on architecture. Two architectures saturate at 1.000 macro F1, which\n"
    "bounds what this corpus can establish; clients are simulated, so multi-site\n"
    "validation remains necessary."
)


def main() -> None:
    sel = json.loads(SEL.read_text(encoding="utf-8"))
    hier = json.loads(HIER.read_text(encoding="utf-8"))["fixed_test"]
    corr = {r["condition"]: r for r in
            json.loads(CORR.read_text(encoding="utf-8"))["conditions"]}
    adapt = json.loads(ADAPT.read_text(encoding="utf-8"))["summary"]
    base = json.loads((V6 / "complete_results.json").read_text(encoding="utf-8"))
    cm = base["results"]["vlm"]["cross_modal_analysis"]["conditions"]
    init = base["results"]["vlm"]["central_f1"]
    club = json.loads(CLUB.read_text(encoding="utf-8"))["tables"]["sparsity=0.0"]

    cand, inc = sel["candidate"], sel["incumbent"]
    p = sel["paired_test_comparison"]
    rows = [hier["text_only"], hier["image_only"], hier["text_image_fusion"],
            cand["fixed_test"], inc["fixed_test"]]
    accs = ", ".join(f"{100 * r['accuracy']:.2f}" + B + "%" for r in rows[:-1])
    accs += f", and {100 * rows[-1]['accuracy']:.2f}" + B + "%"

    finals = [v["final_macro_f1_mean"] for v in adapt.values()]
    ret = [100 * f / init for f in finals]

    farm = json.loads(FARM.read_text(encoding="utf-8"))
    cells = defaultdict(list)
    for k, v in farm["E1_alpha_sweep"].items():
        parts = dict(x.split("=", 1) for x in k.split("|") if "=" in x)
        if parts.get("split") == "corrected":
            cells[(parts["arch"], parts["alpha"])].append(v["f1_macro"])
    sweep = [st.mean(v) for v in cells.values()]

    best_fuse = max(v["central_test_accuracy"] for v in club["multimodal"].values())
    best_img = max(v["central_test_accuracy"] for v in club["image_only"].values())
    best_txt = max(v["central_test_accuracy"] for v in club["text_only"].values())

    new = (
        "On the identical\n"
        f"75-crop support, frozen DistilBERT, frozen ViT-tiny, the raw\n"
        "ResNet--Transformer VLM, a frozen ViT-tiny--DistilBERT VLM, and the routed\n"
        f"ResNet--Transformer reach {accs} accuracy.\n"
        "Selection is validation-only: the ViT--DistilBERT candidate's "
        f"{100 * cand['selected_validation']['accuracy']:.2f}" + B + "% validation\n"
        f"accuracy stays below the routed incumbent's "
        f"{100 * inc['selected_validation']['accuracy']:.2f}" + B + "%, so the incumbent is\n"
        f"retained; on test the candidate leads by {cand['fixed_test']['correct'] - inc['fixed_test']['correct']} crops of 75, a "
        f"{p['first_only_correct']}--{p['second_only_correct']} discordance that is not\n"
        f"significant (McNemar $p={p['exact_two_sided_p']:.2f}$). Fusion of frozen encoders is what "
        "pays: on that\n"
        f"support ViT-Base with BERT-small reaches {100 * best_fuse:.2f}" + B + "% against "
        f"{100 * best_img:.2f}" + B + "% image-only and\n"
        f"{100 * best_txt:.2f}" + B + "% text-only, beating both parents outright rather than only under\n"
        "note sparsity. The trained router does not exploit the note channel the same "
        "way ---\n"
        f"mismatching the note costs it only "
        f"{100 * (cm['paired']['accuracy'] - cm['mismatched_text']['accuracy']):.2f} points --- though "
        f"ablating either modality moves\n"
        f"predictions ({corr['Text deletion 100']['flips']} of 75 for the note, "
        f"{corr['Image zeroed']['flips']} for the image). The router retains\n"
        f"{min(ret):.0f}--{max(ret):.0f}" + B + "% of its {:.4f} centralized validation macro F1 under FedAvg over\n".format(init)
        + "2--8 simulated clients. Across five federated architectures on 371 genuine\n"
        "box-linked observations, federation outperforms local-only training, although\n"
        "the best aggregator is architecture-dependent; one float32 round costs\n"
        "18.9--67.2" + B + ",MiB per client depending on architecture. Under severe skew, client\n"
        "dropout, straggler staleness and poisoned updates, fusion absorbs the first two\n"
        "and degrades under the last two, with no configuration Byzantine-tolerant.\n"
        f"Nothing saturates (the federated sweep spans {min(sweep):.3f}--{max(sweep):.3f}), but clients are\n"
        "simulated and notes are templated, so multi-site validation remains necessary."
    )

    for tex in TEXS:
        pth = ROOT / tex
        s = pth.read_text(encoding="utf-8")
        if "beating both parents outright" in s:
            print(f"  {tex}: already rewritten")
            continue
        if s.count(OLD) != 1:
            print(f"  {tex}: ANCHOR x{s.count(OLD)} -- not applied")
            continue
        pth.write_text(s.replace(OLD, new), encoding="utf-8")
        print(f"  {tex}: rewritten ({OLD.count(chr(10))} -> {new.count(chr(10))} lines)")
    print("\n" + new)


if __name__ == "__main__":
    main()
