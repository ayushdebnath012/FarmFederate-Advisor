#!/usr/bin/env python3
"""Cut the paper to eight pages by suppressing decorative and diagnostic floats.

Nothing is deleted. Each cut float is wrapped in \\iffalse...\\fi between
%%TRIM-BEGIN%% / %%TRIM-END%% markers, so any single item can be restored by
deleting its two marker lines. The prose that announced each float is rewritten
so the sentence still reads on its own -- in every case the numbers were already
in the text, so what is lost is the illustration, not the evidence.

Run with --restore to put everything back.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))
from trim_floats import float_spans  # noqa: E402

B = chr(92)
TEXS = ["overleaf_final/main.tex", "overleaf_final_slim/main.tex"]

CUT = [
    # decorative / duplicated diagrams
    "fig:cross_modal_arch", "fig:reliability_router", "fig:edge_cloud",
    # summary tables whose content is already prose
    "tab:relwork", "tab:model_blueprint", "tab:manifest",
    # diagnostics and secondary panels
    "fig:score_diagnostics", "fig:compact_diagnostics", "fig:confusion",
    "fig:architecture_stability", "fig:advisory", "fig:model_comparisons",
    "fig:remaining_ablations", "fig:training_dynamics",
    "tab:standard_ablation", "tab:fl_adaptation",
]

OPEN = "%%TRIM-BEGIN%% restore by deleting this line and the matching TRIM-END\n" \
       + B + "iffalse\n"
CLOSE = B + "fi\n%%TRIM-END%%\n"

# (old, new) prose rewrites; each must appear exactly once
REWRITES = [
    ("The raw and routed VLM rows are the cross-attention output before and after\n"
     "the router in Figs.~" + B + "ref{fig:cross_modal_arch} and~"
     + B + "ref{fig:reliability_router}.",
     "The raw and routed VLM rows are the cross-attention output before and after\n"
     "the reliability router."),
    ("Figure~" + B + "ref{fig:edge_cloud} separates the two halves of a round: every",
     "The two halves of a round separate cleanly: every"),
    ("Table~" + B + "ref{tab:relwork} marks the gap: exactly\n"
     "paired image--text FL with a source-disjoint leakage audit.",
     "The gap is exactly this: paired image--text FL with a source-disjoint\n"
     "leakage audit."),
    ("Table~" + B + "ref{tab:model_blueprint} summarizes the implementation. The reliability",
     "The reliability"),
    ("Table~" + B + "ref{tab:manifest} separates mechanically verified controls from properties",
     "It separates mechanically verified controls from properties"),
    ("Figure~" + B + "ref{fig:architecture_stability} therefore reports instability, not a component ranking.",
     "The sweep therefore reports instability, not a component ranking."),
    ("Fig.~" + B + "ref{fig:advisory}(b)\nlocates the failure.",
     "The per-class breakdown\nlocates the failure."),
    ("Figure~" + B + "ref{fig:model_comparisons} makes\n"
     "the selection/test distinction visible. Its top row contains earlier screens on\n"
     "different supports and therefore licenses within-panel ranking only.",
     "The selection/test distinction is what matters here: earlier screens ran on\n"
     "different supports and license within-panel ranking only."),
    ("Figures~" + B + "ref{fig:standard_suite} and~" + B + "ref{fig:remaining_ablations}, with\n"
     "Table~" + B + "ref{tab:standard_suite}, close the",
     "Figure~" + B + "ref{fig:standard_suite}, with\n"
     "Table~" + B + "ref{tab:standard_suite}, closes the"),
    ("Figure~" + B + "ref{fig:fl_scaling} and Table~" + B + "ref{tab:fl_adaptation} answer the two\n"
     "federated scaling questions directly.",
     "Figure~" + B + "ref{fig:fl_scaling} answers the two federated scaling questions\n"
     "directly."),
]


def restore(text: str) -> str:
    text = re.sub(r"%%TRIM-BEGIN%%[^\n]*\n" + re.escape(B) + r"iffalse\n", "", text)
    text = text.replace(CLOSE, "")
    for old, new in REWRITES:
        text = text.replace(new, old)
    return text


def apply(text: str) -> tuple[str, list[str]]:
    notes = []
    spans = [(a, b, l) for l, a, b in float_spans(text) if l in CUT]
    for a, b, lab in sorted(spans, reverse=True):
        text = text[:a] + OPEN + text[a:b] + "\n" + CLOSE + text[b:]
    notes.append(f"{len(spans)}/{len(CUT)} floats suppressed")
    missing = set(CUT) - {l for _, _, l in spans}
    if missing:
        notes.append(f"NOT FOUND: {sorted(missing)}")
    for old, new in REWRITES:
        n = text.count(old)
        if n != 1:
            notes.append(f"REWRITE x{n}: {old[:46]!r}")
            continue
        text = text.replace(old, new)
    return text, notes


def main() -> None:
    restoring = "--restore" in sys.argv
    for tex in TEXS:
        p = ROOT / tex
        s = p.read_text(encoding="utf-8")
        if restoring:
            p.write_text(restore(s), encoding="utf-8")
            print(f"  {tex}: restored")
            continue
        if "%%TRIM-BEGIN%%" in s:
            print(f"  {tex}: already trimmed, skipping"); continue
        out, notes = apply(s)
        p.write_text(out, encoding="utf-8")
        print(f"  {tex}: {'; '.join(notes)}")


if __name__ == "__main__":
    main()
