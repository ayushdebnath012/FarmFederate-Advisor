#!/usr/bin/env python3
"""Rewrite the Table III paragraph; its numbers survived the rerun unchanged.

The table was regenerated from the standardized rerun but this paragraph was
not, so it still quoted the pre-correction figures -- and one of its claims
inverts. It said the ordering "depends on the skew", with text leading at mild
skew (0.833 against 0.803). On the corrected corpus fusion leads at both skews
and at every client count, so the hedge is gone and the paragraph gets shorter
saying the stronger thing.

Numbers are computed from the run rather than transcribed.
"""

from __future__ import annotations

import json
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = (ROOT / "tea_results" / "federated_all_systems_full"
       / "federated_all_systems.json")
TEXS = ["overleaf_final/main.tex", "overleaf_final_slim/main.tex"]
B = chr(92)
R = B + "ref{"

OLD = (
    "Within this protocol the ordering depends on the skew: fusion leads under\n"
    "strong label skew (0.677 against 0.571 for text and 0.519 for image), while at\n"
    "mild skew the text branch leads (0.833 against 0.803). Fusion's federation gain\n"
    "over local-only training is nevertheless the largest of the three in both cells\n"
    "($+0.421$ and $+0.372$), and its advantage grows with the number of clients: at\n"
    "$" + B + "alpha=1$ text is ahead for $K" + B + "leq5$ (0.949--0.954) but falls to 0.651 by\n"
    "$K=50$, whereas fusion holds 0.740 and leads from $K=10$ upward\n"
    "(Fig.~" + R + "fig:c_client_scaling})."
)


def main() -> None:
    runs = json.loads(SRC.read_text(encoding="utf-8"))["runs"]
    fed, loc, byk = defaultdict(list), defaultdict(list), defaultdict(list)
    for r in runs.values():
        fed[(r["system"], r["alpha"])].append(r["fedavg_final_macro_f1"])
        loc[(r["system"], r["alpha"])].append(r["local_only_mean_macro_f1"])
        byk[(r["system"], r["alpha"], r["num_clients"])].append(
            r["fedavg_final_macro_f1"])
    F = lambda s, a: st.mean(fed[(s, a)])
    G = lambda s, a: st.mean(fed[(s, a)]) - st.mean(loc[(s, a)])
    T, I, M = "text_distilbert", "image_vit_tiny", "vit_distilbert"
    ks = sorted({k for _, _, k in byk})
    fus = [st.mean(byk[(M, 1.0, k)]) for k in ks]
    txt = [st.mean(byk[(T, 1.0, k)]) for k in ks]

    new = (
        f"Within this protocol fusion leads at both skews ({F(M, 0.1):.3f} against "
        f"{F(T, 0.1):.3f} for\n"
        f"text and {F(I, 0.1):.3f} for image at $" + B + f"alpha=0.1$; {F(M, 1.0):.3f}, {F(T, 1.0):.3f} and "
        f"{F(I, 1.0):.3f} at\n"
        "$" + B + "alpha=1$) and takes the largest federation gain over local-only training in\n"
        f"both cells ($+{G(M, 0.1):.3f}$ and $+{G(M, 1.0):.3f}$). The lead holds across the client sweep:\n"
        f"at $" + B + f"alpha=1$ fusion spans {min(fus):.3f}--{max(fus):.3f} from $K=2$ to {ks[-1]}, above text\n"
        f"({min(txt):.3f}--{max(txt):.3f}) at every count (Fig.~" + R + "fig:c_client_scaling})."
    )

    for tex in TEXS:
        p = ROOT / tex
        s = p.read_text(encoding="utf-8")
        if "fusion leads at both skews" in s:
            print(f"  {tex}: already rewritten")
            continue
        if s.count(OLD) != 1:
            print(f"  {tex}: ANCHOR x{s.count(OLD)}")
            continue
        p.write_text(s.replace(OLD, new), encoding="utf-8")
        print(f"  {tex}: rewritten (-{len(OLD) - len(new)} chars)")
    print("\n" + new)


if __name__ == "__main__":
    main()
