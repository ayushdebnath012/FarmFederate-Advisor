#!/usr/bin/env python3
"""Re-apply the prose corrections that a bad regex deletion wiped out.

A figure-removal regex spanned several blocks and deleted twelve subsections.
The span was spliced back from a backup that predates the completed 50-round
standard tier, so it carries the stale eight-round claims again. These are the
same corrections applied before, restated here so the reconstruction is
auditable and repeatable rather than hand-patched.

Every number below was recomputed from farm_results_standard_genuine.json after
the run printed Done.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
B = chr(92)
R = B + "ref{"
TEXS = ["overleaf_final/main.tex", "overleaf_final_slim/main.tex"]

FIXES = [
    # federation gains, aggregator ties, SCAFFOLD range, image-only accuracy
    ("federation beats local-only\ntraining for all five architectures (gains $+0.121$ to $+0.348$). Aggregator\n"
     "ranking is architecture-dependent: FedProx leads for the text, attention, and\n"
     "cross-attention models at $" + B + "alpha=0.1$, while FedAvg leads for the image-only\n"
     "and concat models; at $" + B + "alpha=1$ FedProx leads or ties throughout. SCAFFOLD is\n"
     "unstable in every configuration (0.088--0.579) and is always worst.",
     "federation beats local-only\ntraining for all five architectures (gains $+0.170$ to $+0.556$). Trained out\n"
     "to 50 rounds, aggregator ranking is mostly a tie: FedAvg, FedProx and FedBN\n"
     "reach the same value for the text, attention and cross-attention models at\n"
     "both skews, while FedProx leads for concat (0.772 against 0.737) and FedAvg\n"
     "for image-only (0.389 against 0.388). SCAFFOLD is the exception, unstable\n"
     "everywhere (0.088--0.581) and strictly worst in nine of ten cells."),
    ("so the image-only model\nreaches 0.499 macro F1 at roughly a third the uplink cost",
     "so the image-only model\nreaches 0.520 macro F1 at roughly a third the uplink cost"),
    # the ceiling statement, now that four of five saturate
    ("The text-only and cross-attention models reach exactly 1.000 at\n"
     "$" + B + "alpha" + B + "geq0.5$;",
     "Four of the five reach exactly 1.000 at $" + B + "alpha=0.5$ and $" + B + "alpha=10$, and\n"
     "text-only and cross-attention do so at every skew;"),
    # warm start reversed sign once trained out
    ("Warm start\nadds 0.034 macro F1 at one seed and nothing at the other.",
     "Warm start does not help once trained out: cold start is better at one seed\n"
     "(1.000 against 0.795) and identical at the other, reversing the eight-round\n"
     "result."),
    # conclusion
    ("In the matched\nsuite, FedAvg reaches 0.715/0.809 at $" + B + "alpha=0.1/1$, warm start adds 0.043, and\n"
     "updates cost about 60" + B + ",MiB.",
     "In the matched\nsuite, FedAvg averages 0.823/0.884 across the five architectures at\n"
     "$" + B + "alpha=0.1/1$, warm start no longer helps, and updates cost about 60" + B + ",MiB."),
    # per-system federated ordering at 50 rounds
    ("Within this protocol fusion leads at both skews\n"
     "(0.697 and 0.831) and its federation gain over local-only training is the\n"
     "largest of the three systems in every cell ($+0.187$ to $+0.446$), so combining\n"
     "the branches helps more under partitioning than either branch alone.",
     "Within this protocol the ordering depends on the skew: fusion leads at strong\n"
     "skew (0.677 against 0.571 text and 0.519 image), text at mild skew (0.833\n"
     "against 0.803). Fusion's gain over local-only training is nevertheless the\n"
     "largest in both cells ($+0.421$, $+0.372$) and grows with clients: at\n"
     "$" + B + "alpha=1$ text leads for $K" + B + "leq5$ but falls to 0.651 by $K=50$, whereas\n"
     "fusion holds 0.740 and leads from $K=10$ up."),
    # references to floats that no longer exist after the trim
    ("Figures~" + R + "fig:standard_suite} and~" + R + "fig:remaining_ablations}, with\n"
     "Table~" + R + "tab:standard_suite}, close the",
     "Table~" + R + "tab:standard_suite} closes the"),
    ("Figure~" + R + "fig:fl_scaling} and Table~" + R + "tab:fl_adaptation} answer the two\n"
     "federated scaling questions directly.",
     "Figure~" + R + "fig:fl_scaling} answers the two federated scaling questions\n"
     "directly."),
    ("Table~" + R + "tab:all_systems_fed}\nand Fig.~" + R + "fig:all_systems_fed} extend it to the frozen-encoder systems.",
     "Table~" + R + "tab:all_systems_fed} extends it to the frozen-encoder systems."),
    ("Figure~" + R + "fig:model_comparisons} makes\n"
     "the selection/test distinction visible. Its top row contains earlier screens on\n"
     "different supports and therefore licenses within-panel ranking only.",
     "The selection/test distinction is what matters here: earlier screens ran on\n"
     "different supports and license within-panel ranking only."),
    # ladder tables were replaced by compact figures
    (R + "tab:mm_ladder}", R + "fig:c_fusion_ladder}"),
    (R + "tab:text_ladder}", R + "fig:c_text_ladder}"),
    (R + "tab:visual_gate}", R + "fig:c_text_ladder}"),
    (R + "fig:clubbed_modality}", R + "fig:c_clubbed_fusion}"),
]


def main() -> None:
    for tex in TEXS:
        p = ROOT / tex
        s = p.read_text(encoding="utf-8")
        applied, absent = 0, []
        for old, new in FIXES:
            n = s.count(old)
            if n:
                s = s.replace(old, new)
                applied += 1
            else:
                absent.append(old.split("\n")[0][:44])
        p.write_text(s, encoding="utf-8")
        msg = f"{applied}/{len(FIXES)} applied"
        if absent:
            msg += f"; not found: {len(absent)}"
        print(f"  {tex}: {msg}")
        for a in absent:
            print(f"      - {a!r}")


if __name__ == "__main__":
    main()
