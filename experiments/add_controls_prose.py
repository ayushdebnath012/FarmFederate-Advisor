#!/usr/bin/env python3
"""State, in every results section, what is held fixed, what is varied, and why.

A reader meeting the same 75-crop support in six different tables cannot tell
which axis each one moves. Each section therefore opens with one \\emph{Held
fixed} paragraph naming the controlled quantity, the varied quantity, and the
reason the control was chosen. The wording is deliberately parallel across
sections so the pattern is recognizable at a glance.

Run with --restore to remove the added paragraphs.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
B = chr(92)
TEXS = ["overleaf_final/main.tex", "overleaf_final_slim/main.tex"]
EM = B + "emph{Held fixed.} "
R = B + "ref{"
E = B + "emph{"

BLOCKS = [
    ("Dataset and Leakage Audit",
     EM + "This split. Every table in the paper is scored on the same 222/74/75\n"
     "source-grouped crops, the same exact box--note pairing, and the same\n"
     "train-fitted leakage mask; nothing below re-splits or re-pairs the corpus,\n"
     "so any difference between tables is the axis that table varies and not the\n"
     "data underneath it."),

    ("Enhanced Image-Only Inference",
     EM + "The encoder. Tables~" + R + "tab:visual_gate}--" + R + "tab:mm_ladder} vary\n"
     "the pipeline and hold the model still. Table~" + R + "tab:visual_gate} fixes the\n"
     "ResNet-50 image path, with frozen ViT-tiny carried as a drop-in row.\n"
     "Table~" + R + "tab:text_ladder} fixes DistilBERT, the text branch of the system\n"
     "the paper actually retains, so the ladder answers what pipeline engineering\n"
     "would buy on the model being shipped rather than on an arbitrary encoder.\n"
     "Table~" + R + "tab:mm_ladder} fixes ViT-Base/16 with DistilBERT, the pair\n"
     "validation selects: ViT-Base leads the five vision encoders at 0.734\n"
     "validation macro F1. Starting the fusion ladder from the strongest available\n"
     "pair is what makes a flat result meaningful --- a pipeline that cannot help\n"
     "the best pair will not help a worse one --- rather than an artefact of a weak\n"
     "starting point. " + E + "Cand.} counts the configurations each rung searched. The\n"
     "gate rows are the one exception: they route among several encoders, which is\n"
     "why they are labelled diagnostics."),

    ("Matched-Setting Controlled Ablations",
     EM + "The client partition. Within every cell all five architectures train on\n"
     "the identical Dirichlet split, so a gap between rows is architectural rather\n"
     "than a lucky partition; the varied axes are the architecture, the label skew\n"
     "$" + B + "alpha$, and the client count $K$."),

    ("Unimodal and Multimodal Systems, Centralized and Federated",
     EM + "The method. Every row uses frozen mean-pooled features --- no test-time\n"
     "augmentation, no layer concatenation, no calibration, no gating --- then one\n"
     "closed-form ridge head whose single penalty is chosen on the 74 validation\n"
     "crops, with plain concatenation for fusion. Four reasons. " + E + "Isolation:}\n"
     "under an elaborate pipeline a good score cannot be attributed to the model\n"
     "rather than to the pipeline, so freezing the simplest method leaves the\n"
     "encoder as the only thing that moves. " + E + "Determinism:} a closed-form solve\n"
     "has one exact solution, no seed and no initialization, so a gap between two\n"
     "rows is the encoder and not luck, which the TTA and gate rungs of\n"
     "Table~" + R + "tab:visual_gate} cannot promise. " + E + "Minimal search:} one penalty\n"
     "out of six, against the 243--3125 candidates of the ladders; on 74 validation\n"
     "crops that difference is precisely the 0.185 validation-to-test collapse\n"
     "those gates exhibit (0.932 against 0.747). " + E + "Lower bound:} plain\n"
     "concatenation is the weakest fusion rung of Table~" + R + "tab:mm_ladder}, so a\n"
     "multimodal gain reported here understates what better fusion achieves, which\n"
     "is why the discussion below points forward to the 0.920 of calibrated late\n"
     "fusion rather than treating the concatenation number as a ceiling. The varied\n"
     "axes are the encoder --- classical, then five text and five vision\n"
     "transformers --- and the deployment. Only the federated column departs from\n"
     "the fixed method, because a closed-form solve has no gradients for FedAvg to\n"
     "average and its head must therefore be trained by SGD."),

    ("Actual Model-Family and Inter-Model Comparison",
     EM + "The support and the selection rule. All five families are screened on the\n"
     "same validation crops under the same pre-declared score, and test is read\n"
     "once afterwards; the varied axis is the model family. The text row alone\n"
     "also fixes the note condition at 50" + B + "% masking, which is why it must not be\n"
     "read against the complete-note rows of Table~" + R + "tab:unimodal_cent_fed}."),

    ("Missing-Modality and Corruption Probes",
     EM + "The checkpoint. One trained model is held constant and the " + E + "input}\n"
     "is varied instead --- each modality corrupted in turn and the image ablated\n"
     "outright --- so a change in prediction is attributable to the evidence the\n"
     "model consumes rather than to retraining."),
]


def anchor(title: str) -> str:
    return B + "subsection{" + title + "}"


def main() -> None:
    restoring = "--restore" in sys.argv
    for tex in TEXS:
        p = ROOT / tex
        s = p.read_text(encoding="utf-8")
        notes = []
        for title, para in BLOCKS:
            a = anchor(title)
            if restoring:
                if "\n\n" + para + "\n" in s:
                    s = s.replace("\n\n" + para + "\n", "")
                    notes.append(title.split()[0])
                continue
            if para[:60] in s:
                notes.append(f"{title.split()[0]}=present"); continue
            if s.count(a) != 1:
                notes.append(f"{title.split()[0]}=ANCHORx{s.count(a)}"); continue
            # insert after the heading, and after any \label line that follows it
            k = s.index(a) + len(a)
            if s[k:k + 8].startswith("\n" + B + "label"):
                k = s.index("\n", k + 1)
            s = s[:k] + "\n\n" + para + "\n" + s[k:]
            notes.append(f"{title.split()[0]}=ok")
        p.write_text(s, encoding="utf-8")
        print(f"  {tex}: {' '.join(notes)}")


if __name__ == "__main__":
    main()
