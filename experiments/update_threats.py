#!/usr/bin/env python3
"""Update Threats to Validity for what the reruns changed, and tighten it.

Two entries are no longer accurate:

  "The router result is non-pristine"  -- it was, because Table VI drew on the
      one hierarchy variant flagged fixed_test_is_pristine: False. The rebuild
      uses the pristine variant, so the caveat is retired rather than restated.

  "earlier fusion screens on proxy pairs have incomparable supports"  -- those
      screens are no longer cited anywhere in the compiled paper, so the caveat
      guards nothing.

Two are added, because the reruns created them: the robustness sweep uses linear
probes on frozen features rather than the deep model, and the corruption probe
now also reports a reliability weight that moves the wrong way, which is a
property of one checkpoint.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEXS = ["overleaf_final/main.tex", "overleaf_final_slim/main.tex"]

OLD = (
    "The locked test is small (38 sources, 75 crops, two leaf-hopper crops), and\n"
    "selecting five experts on 74 validation crops can overfit. The\n"
    "central checkpoint has one seed; FL seeds vary only client partitions. Clients\n"
    "are simulated; E2/E3 use one seed. The router result is\n"
    "non-pristine, DistilBERT is a PLM rather than a generative LLM, and earlier\n"
    "fusion screens on proxy pairs have incomparable supports, so their patterns are\n"
    "read without their levels. The corruption probe is one seeded realization per\n"
    "severity on 75 crops and carries no interval. Deployment requires multi-estate data,\n"
    "independent notes, rare-class coverage, and repeated end-to-end training."
)

NEW = (
    "The locked test is small (38 sources, 75 crops, two leaf-hopper crops), and\n"
    "selection on 74 validation crops can overfit. The central checkpoint has one\n"
    "seed; FL seeds vary only client partitions, and clients are simulated.\n"
    "DistilBERT is a PLM rather than a generative LLM. The corruption probe is one\n"
    "seeded realization per severity on 75 crops and carries no interval, and the\n"
    "reliability weights it reports are one checkpoint's. The robustness sweep\n"
    "federates linear probes on frozen features, so its invariances belong to that\n"
    "estimator and not to the deep model. Deployment requires multi-estate data,\n"
    "independent notes, rare-class coverage, and repeated end-to-end training."
)


def main() -> None:
    for tex in TEXS:
        p = ROOT / tex
        s = p.read_text(encoding="utf-8")
        if "belong to that\nestimator" in s:
            print(f"  {tex}: already updated")
            continue
        if s.count(OLD) != 1:
            print(f"  {tex}: ANCHOR x{s.count(OLD)}")
            continue
        p.write_text(s.replace(OLD, NEW), encoding="utf-8")
        print(f"  {tex}: updated (-{len(OLD) - len(NEW)} chars)")


if __name__ == "__main__":
    main()
