#!/usr/bin/env python3
"""Define every symbol and weight in the Model Variants and Objective section.

The section previously stated the reliability weights, the residual factor 2,
and the four loss terms without saying what any of them are. Each definition
here was read off the implementation in tea_train.py rather than inferred from
the formula:

  reliability MLP      LayerNorm -> Linear(2d,128) -> GELU -> Linear(128,2),
                       absent modality masked with -1e4 before the softmax
  expert residual      expert_residual_weight = 2.0, applied at train and test
  auxiliary weights    text_auxiliary_weight = 0.20, vision_auxiliary_weight = 1.25
  alignment weight     alignment_weight = 0.05
  label smoothing      0.1, with inverse-frequency class weights
  auxiliary masking    each expert loss is averaged only over samples where that
                       modality is present
  alignment term       supervised contrastive over l2-normalised embeddings of
                       paired samples, both directions, needs >= 2 paired crops
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
B = chr(92)
Z = B + "mathbf{z}"
M = B + "mathcal{L}"
TEXS = ["overleaf_final/main.tex", "overleaf_final_slim/main.tex"]

OLD_LOGITS = (
    "The reliability network\n"
    "produces $(r_t,r_v)$, $r_t+r_v=1$, and the final logits are\n"
    "$" + Z + "=" + Z + "_f+2(r_t" + Z + "_t+r_v" + Z + "_v)$, with a missing\n"
    "modality's expert zeroed. The residual remains active at test time."
)

NEW_LOGITS = "\n".join([
    "The reliability network is a two-layer MLP over the concatenated text and",
    "image embeddings whose two outputs are softmax-normalized, giving weights",
    "$(r_t,r_v)$ with $r_t+r_v=1$: $r_t$ is how far the fused prediction leans on",
    "the note and $r_v$ on the image. An absent modality has its logit set to",
    "$-10^{4}$ before the softmax, so its weight goes to zero and the surviving",
    "modality takes the full unit of mass. The final logits are",
    B + "begin{equation}",
    Z + "=" + Z + "_f+" + B + "lambda" + B + "left(r_t" + Z + "_t+r_v" + Z + "_v"
    + B + "right)," + B + "qquad" + B + "lambda=2,",
    B + "label{eq:logits}",
    B + "end{equation}",
    "where $" + Z + "_f$ is the five-way output of the fusion head and $" + Z + "_t$,",
    "$" + Z + "_v$ are the outputs of two linear " + B + "emph{expert} heads reading the",
    "text and image embeddings directly. The residual weight $" + B + "lambda=2$ is",
    "fixed rather than learned: it makes the two single-modality experts count",
    "twice as much as the fused head, which is what leaves a usable prediction",
    "when one branch is absent. A missing modality's expert is zeroed, and the",
    "residual stays active at test time rather than being a training-only crutch.",
])

OLD_CE = (
    "For one-hot label $" + B + "mathbf{y}$, training uses class-weighted, label-smoothed\n"
    "cross entropy with auxiliary parent losses and supervised cross-modal alignment:"
)
NEW_CE = (
    "For one-hot label $" + B + "mathbf{y}$, training uses class-weighted, label-smoothed\n"
    "cross entropy --- weights inverse to class frequency, smoothing $0.1$ --- with\n"
    "auxiliary parent losses and supervised cross-modal alignment:"
)

OLD_W = (
    "The visual weight is deliberately larger because image-only learning is the\n"
    "harder path."
)
NEW_W = "\n".join([
    "Here $" + M + "_{f}$ is the loss on the fused logits $" + Z + "$; $" + M + "_{t}$ and",
    "$" + M + "_{v}$ apply the same criterion to the two expert heads, each averaged",
    "only over samples in which that modality is present, so a text-only batch",
    "never trains the image expert; and $" + M + "_{align}$ is a supervised contrastive",
    "term over the $" + B + "ell_2$-normalized text and image embeddings of the paired",
    "samples, treating same-class pairs as positives and averaging both",
    "directions. It is evaluated only when a batch holds at least two paired",
    "crops. The three weights are fixed by construction rather than tuned: $1.25$",
    "on the visual expert against $0.20$ on the text expert, because image-only",
    "learning is the harder path while the templated notes make the text expert",
    "easy to satisfy, and $0.05$ on alignment because it is a regularizer, large",
    "enough to shape the embedding geometry and small enough not to compete with",
    "the classification objective.",
])

PAIRS = [("logits", OLD_LOGITS, NEW_LOGITS),
         ("cross-entropy", OLD_CE, NEW_CE),
         ("weights", OLD_W, NEW_W)]


def main() -> None:
    for tex in TEXS:
        p = ROOT / tex
        s = p.read_text(encoding="utf-8")
        notes = []
        for tag, old, new in PAIRS:
            # a prefix test is unsafe: several replacements begin with the
            # same words as the text they replace. Test a phrase unique to new.
            marker = max(new.split("---")[-1].split("."), key=len).strip()
            if marker and marker in s:
                notes.append(f"{tag}=present")
                continue
            n = s.count(old)
            if n != 1:
                notes.append(f"{tag}=ANCHORx{n}")
                continue
            s = s.replace(old, new)
            notes.append(f"{tag}=ok")
        p.write_text(s, encoding="utf-8")
        print(f"  {tex}: {' '.join(notes)}")


if __name__ == "__main__":
    main()
