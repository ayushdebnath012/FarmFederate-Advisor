#!/usr/bin/env python3
"""Insert the compact single-panel figures into the paper.

Each figure is a single-column float carrying one claim, so its caption is one
sentence and several fit per page. References that pointed at the large
multi-panel figures these replace are rewritten, so no sentence is left aimed
at a float that no longer exists.

Blocks are delimited by distinct BEGIN/END markers. An earlier version used the
same token for both, which made a deletion regex unable to tell one block's end
from the next block's start; it deleted across boundaries and left orphans.

Run with --restore to remove them again.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
B = chr(92)
R = B + "ref{"
TEXS = ["overleaf_final/main.tex", "overleaf_final_slim/main.tex"]
OPEN_MARK = "%%COMPACT-BEGIN%%"
CLOSE_MARK = "%%COMPACT-END%%"

# (image, label, caption, subsection this figure is placed before)
FIGS = [
    ("c01_text_ladder", "fig:c_text_ladder",
     "Text ladder rungs as field notes are deleted, on the identical 75 crops. "
     "Every rung falls from the ceiling; gates are dashed and marked $*$.",
     "Reliability-Aware Sparse-Note Rule"),
    ("c02_fusion_ladder", "fig:c_fusion_ladder",
     "Text${+}$image rungs under the same deletion. Fusion degrades far more "
     "gently: the best rung holds 0.853 where the best text rung reaches 0.653.",
     "Reliability-Aware Sparse-Note Rule"),
    ("c04_selection_gap", "fig:c_selection_gap",
     "Validation minus test accuracy per system on the aligned support. A large "
     "positive gap is selection that did not transfer from 74 validation crops.",
     "Multimodal Federated Adaptation"),
    ("c08_clubbed_text", "fig:c_clubbed_text",
     "Text systems as notes are deleted, including the classical baseline: "
     "TF-IDF leads every transformer at every sparsity.",
     "Actual Model-Family and Inter-Model Comparison"),
    ("c09_clubbed_fusion", "fig:c_clubbed_fusion",
     "Fused systems under the same deletion, showing where fusion overtakes "
     "both unimodal parents.",
     "Actual Model-Family and Inter-Model Comparison"),
    ("c10_federation_cost", "fig:c_fed_cost",
     "Centralized minus federated accuracy per encoder at 50\\% note deletion. "
     "Federating is close to free for the image encoders.",
     "Cross-Modal Analysis and Evidence Boundary"),
    ("c11_alpha_corrected", "fig:c_alpha_corrected",
     "All five architectures across skews under the corrected class-wise "
     "splitter: accuracy falls as $" + B + "alpha$ tightens, as intended.",
     "Cross-Modal Analysis and Evidence Boundary"),
    ("c13_aggregators", "fig:c_aggregators",
     "Aggregators at $" + B + "alpha=0.1$; the dashed rule in each group is that "
     "architecture's local-only control. SCAFFOLD collapses to 0.088.",
     "Missing-Modality and Corruption Probes"),
    ("c14_round_curves", "fig:c_round_curves",
     "Per-round convergence of the frozen-encoder systems at $" + B + "alpha=1$, "
     "three seeds. Text overtakes fusion at round 33.",
     "Architecture Component Ablation"),
    ("c15_client_scaling", "fig:c_client_scaling",
     "The same systems against client count. Fusion leads from $K=10$ upward, "
     "where the shards become small.",
     "Architecture Component Ablation"),
    ("c19_retrieval_box", "fig:c_retrieval_box",
     "Top-1 retrieval similarity per class, split by whether that hit was "
     "correct. The distributions overlap, so similarity does not separate a "
     "correct advisory from a wrong one.",
     "Advisory Retrieval Results"),
]

# references that pointed at the large figures these replace
REWRITE = [
    (R + "fig:sparsity_ladder}", R + "fig:c_fusion_ladder}"),
    (R + "fig:fed_rounds_clients}b", R + "fig:c_client_scaling}"),
    (R + "fig:fed_rounds_clients}", R + "fig:c_client_scaling}"),
    (R + "fig:alpha_splitters}", R + "fig:c_alpha_corrected}"),
    (R + "fig:clubbed_modality}", R + "fig:c_clubbed_fusion}"),
    (R + "fig:selection_gap}", R + "fig:c_selection_gap}"),
    (R + "fig:retrieval_box}", R + "fig:c_retrieval_box}"),
    (R + "fig:encoder_detail}", R + "fig:c_fed_cost}"),
    (R + "fig:ladder_search}", R + "fig:c_fusion_ladder}"),
]


def block(name, label, cap):
    return "\n".join([
        OPEN_MARK, B + "begin{figure}[t]", B + "centering",
        B + "includegraphics[width=" + B + "linewidth]{plots/" + name + "}",
        B + "caption{" + cap + "}", B + "label{" + label + "}",
        B + "end{figure}", CLOSE_MARK])


def main() -> None:
    restoring = "--restore" in sys.argv
    for tex in TEXS:
        p = ROOT / tex
        s = p.read_text(encoding="utf-8")
        if restoring:
            s = re.sub(re.escape(OPEN_MARK) + r"\n.*?" + re.escape(CLOSE_MARK)
                       + r"\n\n", "", s, flags=re.S)
            p.write_text(s, encoding="utf-8")
            print(f"  {tex}: compact figures removed")
            continue
        for old, new in REWRITE:
            s = s.replace(old, new)
        added, missed = 0, []
        for name, label, cap, anchor in FIGS:
            # test for the label command, not the bare string: the rewrite above
            # already put the label text inside a reference
            if B + "label{" + label + "}" in s:
                continue
            a = B + "subsection{" + anchor + "}"
            if s.count(a) != 1:
                missed.append(anchor)
                continue
            s = s.replace(a, block(name, label, cap) + "\n\n" + a, 1)
            added += 1
        p.write_text(s, encoding="utf-8")
        note = f"{added}/{len(FIGS)} inserted"
        if missed:
            note += f"; anchors not found: {sorted(set(missed))}"
        print(f"  {tex}: {note}")


if __name__ == "__main__":
    main()
