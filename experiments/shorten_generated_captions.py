#!/usr/bin/env python3
"""Shorten the three generated captions to what a caption has to carry.

They were written to be self-contained, which is right for a caption that
carries an argument the prose does not. Here the argument is in the prose
directly beside each table, so the captions restate it: Table II's caption
narrates the whole sweep, Table VI's re-derives the selection decision, and
Table VII's re-lists the numbers in its own rows.

Each keeps what a reader needs to interpret the columns -- support, protocol,
and any condition that is not obvious from the header -- and drops the
narration. The findings stay in the text.

Edits the generators, so a rerun keeps the short form.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
B = chr(92)

JOBS = [
    ("make_standard_suite_table.py",
     ['         "of validation macro F1 over 50 federated rounds. No architecture "\n'
      '         f"approaches saturation: the sweep spans {lo:.3f}--{hi:.3f}, the best cell "\n'
      '         f"being {dict(ARCHS).get(hi_arch, hi_arch)} at $" + BS + f"alpha{{=}}{hi_alpha:g}$, so the "\n'
      '         "ordering reflects architecture rather than the note-leakage ceiling the "\n'
      '         "earlier corpus imposed. Federation beats local-only training for every "\n'
      '         f"architecture ($+${min(gains):.3f} to $+${max(gains):.3f}).}}",\n'],
     ['         "of validation macro F1 over 50 federated rounds. Nothing saturates: the "\n'
      '         f"sweep spans {lo:.3f}--{hi:.3f}.}}",\n']),

    ("make_encoder_selection_table.py",
     ['        "under the deterministic sparse-note mask at "\n'
      '        f"{100 * keep:.0f}" + B + "% of note tokens "\n'
      '        "retained, which is the rate the routing hierarchy selects on the "\n'
      '        "corrected corpus. Validation retains "\n'
      '        f"{winner} ({inc[\'selected_validation\'][\'accuracy\']:.4f} against "\n'
      '        f"{cand[\'selected_validation\'][\'accuracy\']:.4f} accuracy); on test the candidate leads by "\n'
      '        f"{cand[\'fixed_test\'][\'correct\'] - inc[\'fixed_test\'][\'correct\']} crops of 75 and the two macro-F1 values are "\n'
      '        f"within {abs(cand[\'fixed_test\'][\'macro_f1\'] - inc[\'fixed_test\'][\'macro_f1\']):.4f}, so the test gap does not "\n'
      '        "reverse the choice. The image row uses ViT-tiny rather than the "\n'
      '        "ViT-Base of Table~" + B + "ref{tab:unimodal_cent_fed}.")\n'],
     ['        f"under the deterministic sparse-note mask at {100 * keep:.0f}" + B + "% of note "\n'
      '        "tokens retained, the rate the routing hierarchy selects. The image row "\n'
      '        "uses ViT-tiny, not the ViT-Base of Table~" + B + "ref{tab:unimodal_cent_fed}.")\n']),

    ("make_corruption_table.py",
     ['        "from the clean run. Neither modality is inert and neither dominates: "\n'
      '        f"zeroing the image costs {clean[\'accuracy\'] - zero[\'accuracy\']:.4f} accuracy and moves "\n'
      '        f"{zero[\'flips\']} of {n} predictions, against "\n'
      '        f"{clean[\'accuracy\'] - text_kill[\'accuracy\']:.4f} and {text_kill[\'flips\']} "\n'
      '        "for deleting every note token, though the image ablation collapses "\n'
      '        f"macro-F1 further ({zero[\'macro_f1\']:.4f} against {text_kill[\'macro_f1\']:.4f}). "\n'
      '        "Additive Gaussian noise is absorbed up to "\n'
      '        f"$" + B + "sigma{=}0.26$; replacing the image with noise of matched statistics "\n'
      '        f"({uni[\'accuracy\']:.4f}, {uni[\'flips\']} flips) is the most damaging condition tested, "\n'
      '        "which separates robustness to sensor noise from dependence on image "\n'
      '        "content.")\n'],
     ['        "from the clean run. Neither modality is inert and neither dominates; "\n'
      '        "uniform noise of matched statistics is the most damaging condition "\n'
      '        "tested, which separates robustness to sensor noise from dependence on "\n'
      '        "image content.")\n']),
]


def main() -> None:
    for name, olds, news in JOBS:
        p = ROOT / "experiments" / name
        s = p.read_text(encoding="utf-8")
        old, new = "".join(olds), "".join(news)
        # a phrase that exists only after the edit: every replacement here keeps
        # its opening line, so a prefix check reports a false "already"
        marker = {"make_standard_suite_table.py": "Nothing saturates: the ",
                  "make_encoder_selection_table.py":
                      "the rate the routing hierarchy selects. The image row",
                  "make_corruption_table.py":
                      "uniform noise of matched statistics is the most damaging"}[name]
        if marker in s:
            print(f"  {name}: already shortened")
            continue
        if s.count(old) != 1:
            print(f"  {name}: ANCHOR x{s.count(old)}")
            continue
        p.write_text(s.replace(old, new), encoding="utf-8")
        print(f"  {name}: shortened (-{len(old) - len(new)} source chars)")


if __name__ == "__main__":
    main()
