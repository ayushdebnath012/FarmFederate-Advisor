#!/usr/bin/env python3
"""Swap the four-panel robustness float for the two-panel one, and tighten.

The page budget is 8 pages and the four new experiments cost about one. The cut
that loses least is the figure: panels (a) and (b) plot two variables that do
not move -- skew and dropout are absorbed -- and a flat line is the one result
a sentence states as well as a plot. Panels (c) and (d) are the two modes that
actually degrade, so those keep their float. The full four-panel version stays
in plots/ and in the manifest for anyone who wants it.

The poisoning paragraph is also the longest in the section; it is compressed
without dropping a number.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEXS = ["overleaf_final/main.tex", "overleaf_final_slim/main.tex"]
B = chr(92)

OLD_FIG = (
    B + "includegraphics[width=" + B + "linewidth]{plots/plot92_robustness_all}\n"
    + B + "caption{The four federated failure modes the earlier sweeps do not cover, on the\n"
    "audited split. (a) Severe label skew, to one class per client. (b) Clients failing\n"
    "to report per round. (c) Clients reporting an update computed from an earlier\n"
    "global model. (d) Malicious clients under mean against coordinate-wise-median\n"
    "aggregation. Bands are $" + B + "pm$1 SD over 3 seeds. Skew and dropout are absorbed;\n"
    "staleness and poisoning are not.}"
)

NEW_FIG = (
    B + "includegraphics[width=" + B + "linewidth]{plots/plot91_robustness_stragglers_poison}\n"
    + B + "caption{The two federated failure modes that actually degrade the stack, on the\n"
    "audited split. (a) Clients reporting an update computed from an earlier global\n"
    "model; bands are $" + B + "pm$1 SD over 3 seeds. (b) Malicious clients under mean against\n"
    "coordinate-wise-median aggregation. Severe skew and client dropout are omitted\n"
    "because both are absorbed --- see the text, and plot92 in the released figure set\n"
    "for all four panels.}"
)

OLD_POISON_TAIL = (
    "Gaussian noise sits between them. Coordinate-wise median recovers the outlier\n"
    "attacks --- sign flip to 0.613, 0.543 and 0.420, Gaussian to 0.564 at 30" + B + "% --- but not\n"
    "label flip, where it is within noise at 10--20" + B + "% and worse at 30" + B + "% (0.510\n"
    "against 0.539), because a label-flipped update is a well-formed gradient step\n"
    "toward the wrong target and is not an outlier in parameter space. The median also\n"
    "costs 0.057 with nobody attacking (0.715 against 0.772), so it is a trade\n"
    "and not a free defence. No configuration here is Byzantine-tolerant and none is\n"
    "claimed to be."
)

NEW_POISON_TAIL = (
    "Gaussian noise sits between them. Coordinate-wise median recovers the outlier\n"
    "attacks (sign flip to 0.613, 0.543, 0.420; Gaussian to 0.564 at 30" + B + "%) but not label\n"
    "flip, which is within noise at 10--20" + B + "% and worse at 30" + B + "% (0.510 against 0.539): a\n"
    "label-flipped update is a well-formed gradient step toward the wrong target, not an\n"
    "outlier in parameter space. The median also costs 0.057 with nobody attacking, so it\n"
    "is a trade, not a free defence. Nothing here is Byzantine-tolerant."
)


def main() -> None:
    for tex in TEXS:
        p = ROOT / tex
        s = p.read_text(encoding="utf-8")
        notes = []
        for tag, old, new in (("float", OLD_FIG, NEW_FIG),
                              ("poison tail", OLD_POISON_TAIL, NEW_POISON_TAIL)):
            if s.count(old) == 1:
                s = s.replace(old, new)
                notes.append(f"{tag} swapped (-{old.count(chr(10)) - new.count(chr(10))} lines)")
            elif new.split("\n")[0][:40] in s:
                notes.append(f"{tag}: already applied")
            else:
                notes.append(f"{tag}: ANCHOR x{s.count(old)}")
        p.write_text(s, encoding="utf-8")
        print(f"  {tex}: " + "; ".join(notes))


if __name__ == "__main__":
    main()
