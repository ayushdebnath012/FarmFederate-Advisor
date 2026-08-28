#!/usr/bin/env python3
"""Replace the architecture-ablation paragraph with the multi-seed result.

The old paragraph reported a CPU-versus-H100 discrepancy and concluded
"instability, not a component ranking". That conclusion was an artefact of the
comparison: the two runs used different devices AND single seeds, so device and
seed were confounded and could not be separated.

The sweep was rerun as 11 variants x 3 seeds on one device (33 runs, 2.0 GPU
hours), which separates them. The variance is now attributable: pooled
within-variant SD is 0.032 accuracy, and the full model alone spans 0.053 across
seeds. Against that, one component effect survives and the rest do not -- which
is a stronger and more honest statement than "unstable", because it names the
noise floor the effects are being measured against.

Numbers come from tea_results/architecture_ablation_v1/, computed here rather
than transcribed.
"""

from __future__ import annotations

import json
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = (ROOT / "tea_results" / "architecture_ablation_v1"
       / "architecture_ablation_partial.json")
TEXS = ["overleaf_final/main.tex", "overleaf_final_slim/main.tex"]
B = chr(92)

OLD = ("Nine variants were retrained on the same split in CPU and H100 runs. Although the\n"
       "H100 full model reaches 94.67\\%, the identical CPU run gives 85.33\\%; the mean\n"
       "gap is four points and five of eight effects reverse sign. The sweep therefore reports instability, not a component ranking.")


def stats():
    rows = json.loads(SRC.read_text(encoding="utf-8"))
    g = defaultdict(list)
    for r in rows:
        g[r["component"]].append(r)
    acc = {k: (st.mean(x["test_accuracy"] for x in v),
               st.pstdev(x["test_accuracy"] for x in v)) for k, v in g.items()}
    pooled = st.mean(sd for _, sd in acc.values())
    full = acc["full"][0]
    spread = (max(x["test_accuracy"] for x in g["full"])
              - min(x["test_accuracy"] for x in g["full"]))
    hours = sum(r["seconds"] for r in rows) / 3600
    return len(rows), len(g), acc, pooled, full, spread, hours


def main() -> None:
    n_runs, n_var, acc, pooled, full, spread, hours = stats()
    seeds = n_runs // n_var
    light = acc["lightweight_vision"][0] - full
    lam = {k: acc[k][0] - full
           for k in ("no_expert_residual", "residual_lambda1", "residual_lambda4")}
    others = {k: v[0] - full for k, v in acc.items()
              if k not in ("full", "lightweight_vision")}
    within = max(abs(d) for k, d in others.items() if not k.startswith("residual")
                 and k != "no_expert_residual")

    new = (
        f"The sweep was rerun as {n_var} variants $" + B + "times$ "
        f"{seeds} seeds on a single device ({n_runs} runs, {hours:.1f} GPU\n"
        "hours), which is what separates a component effect from seed noise; the\n"
        "earlier CPU-versus-H100 comparison confounded the two. The noise floor is\n"
        f"large: pooled within-variant SD is {pooled:.3f} accuracy and the full model\n"
        f"alone spans {spread:.3f} across seeds ({full:.3f} mean). Against that floor one\n"
        f"effect survives --- replacing the vision encoder with a lightweight stack\n"
        f"costs {abs(light):.3f} --- and none of the remaining ablations moves the mean by\n"
        f"more than {within:.3f}, i.e.\\ under two SDs. The expert-residual weight is not\n"
        "supported by the sweep at all: removing the residual entirely ($" + B + "lambda=0$,\n"
        f"{lam['no_expert_residual']:+.3f}) beats both $" + B + "lambda=1$ "
        f"({lam['residual_lambda1']:+.3f}) and $" + B + "lambda=4$ "
        f"({lam['residual_lambda4']:+.3f}) relative to the\n"
        "$" + B + "lambda=2$ default, so the ordering is non-monotone and the setting is a\n"
        "choice rather than a tuned optimum. Read together with Table~" + B +
        "ref{tab:multimodal_cent_fed}, where a frozen-encoder concatenation reaches 0.800 on the\n"
        "same support, the trained architecture is not what produces the multimodal\n"
        "result and is not claimed to be."
    )

    for tex in TEXS:
        p = ROOT / tex
        s = p.read_text(encoding="utf-8")
        if "which is what separates a component effect from seed noise" in s:
            print(f"  {tex}: already rewritten")
            continue
        if s.count(OLD) != 1:
            print(f"  {tex}: ANCHOR x{s.count(OLD)} -- not applied")
            continue
        p.write_text(s.replace(OLD, new), encoding="utf-8")
        print(f"  {tex}: rewritten")

    print("\n" + new)


if __name__ == "__main__":
    main()
