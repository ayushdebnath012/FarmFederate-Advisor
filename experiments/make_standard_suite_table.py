#!/usr/bin/env python3
"""Regenerate the matched-setting suite table from the completed full tier.

The block previously in the paper was transcribed from an 8-round run. Eight
rounds is short of convergence here, and the numbers move a long way once the
run is trained out -- the attention VLM goes from 0.628 to 0.988 at the
strongest skew. Generating the table from the JSON keeps the printed values
tied to the run that produced them.

Local-only records store `mean_f1`; federated records store `f1_macro`. Reading
the wrong field silently yields 0.000, so the accessor is shared.
"""

from __future__ import annotations

import json
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "experiments" / "farm_results_full_genuine.json"
OUT = ROOT / "experiments" / "standard_suite_block.tex"
BS = chr(92)

ARCHS = [("image_only", "Image only"),
         ("text_only", "Text only"),
         ("concat_vlm", "Concat VLM"),
         ("attention_vlm", "Attention VLM"),
         ("cross_attention_vlm", "Cross-attn.\\ VLM")]
AGGS = [("fedavg", "Avg"), ("fedprox", "Prox"),
        ("scaffold", "SCAF"), ("fedbn", "BN")]
SHORT = {"image_only": "Image", "text_only": "Text", "concat_vlm": "Concat",
         "attention_vlm": "Attention", "cross_attention_vlm": "Cross-att"}


def parse(key: str) -> dict:
    out = {}
    for part in key.split("|"):
        if "=" in part:
            a, b = part.split("=", 1)
            out[a] = b
        else:
            out["_alg"] = part
    return out


def score(rec: dict):
    """Federated rows carry f1_macro; local-only rows carry mean_f1."""
    return rec.get("f1_macro", rec.get("mean_f1"))


def main() -> None:
    d = json.load(open(SRC, encoding="utf-8"))

    alpha_cell = defaultdict(list)
    for k, v in d["E1_alpha_sweep"].items():
        f = parse(k)
        if f.get("split") == "corrected":
            alpha_cell[(f["arch"], float(f["alpha"]))].append(score(v))
    alphas = sorted({a for _, a in alpha_cell})

    agg_cell = defaultdict(list)
    for k, v in d["E8_baselines"].items():
        f = parse(k)
        if f.get("alpha") == "0.1":
            agg_cell[(f["arch"], f["_alg"])].append(score(v))

    cost = {}
    for k, v in d["E6_cost"].items():
        if k.startswith("arch_cost|"):
            cost[k.split("|", 1)[1]] = v.get("upload_mib_per_client_per_round")

    seeds = sorted({parse(k).get("seed") for k in d["E1_alpha_sweep"]})

    # Caption facts are computed, not asserted. The previous caption claimed
    # saturation at 1.000 because the old note corpus was label-determining; on
    # the corrected corpus the same cells top out near 0.56, and a hand-written
    # sentence would have kept contradicting the table underneath it.
    cells = {k: st.mean(v) for k, v in alpha_cell.items()}
    lo, hi = min(cells.values()), max(cells.values())
    hi_arch, hi_alpha = max(cells, key=cells.get)
    gains = []
    for arch in {a for a, _ in cells}:
        fed, loc = agg_cell[(arch, "fedavg")], agg_cell[(arch, "local_only")]
        if fed and loc:
            gains.append(st.mean(fed) - st.mean(loc))

    # Split facts are computed from the same file the cells come from, so this
    # caption cannot drift when SRC changes (it previously did: a hardcoded
    # "260/53/58 over 153" survived a switch to farm_results_full_genuine.json).
    md = d["_meta"]["data"]
    n_tr, n_va = md["train"], md["val"]
    n_te = md["num_paired"] - n_tr - n_va
    n_grp = md["num_independent_source_groups"]

    L = [BS + "begin{table*}[t]",
         BS + "caption{Federated results on " + str(md["num_paired"])
         + " genuine box-linked crop--note "
         f"observations ({n_tr}/{n_va}/{n_te} over {n_grp} independent source "
         "groups; no synthetic images). Every architecture within a cell trains on the "
         "identical client partition, so differences are architectural rather "
         "than partition effects. Values are " + str(len(seeds)) + "-seed means "
         "of validation macro F1 over 50 federated rounds. Nothing saturates: panel (a) "
         f"spans {lo:.3f}--{hi:.3f}.}}",
         BS + "label{tab:standard_suite}",
         BS + "centering" + BS + "scriptsize",
         BS + "setlength" + BS + "tabcolsep{2.7pt}"]

    # ---- (a) FedAvg by architecture and skew -----------------------------
    L += [BS + "begin{minipage}[t]{0.38" + BS + "textwidth}",
          BS + "centering",
          BS + "textbf{(a) FedAvg by architecture and skew}",
          BS + "begin{tabular}{lcccc}", BS + "toprule",
          "Architecture & " + " & ".join(
              "$" + BS + "alpha{=}" + (f"{a:g}".lstrip("0") or "0") + "$"
              for a in alphas) + BS + BS,
          BS + "midrule"]
    for key, label in ARCHS:
        vals = [st.mean(alpha_cell[(key, a)]) for a in alphas]
        L.append(f"{label:<17} & " + " & ".join(f"{v:.3f}" for v in vals) + BS + BS)
    L += [BS + "bottomrule", BS + "end{tabular}", BS + "end{minipage}" + BS + "hfill"]

    # ---- (b) aggregator at the strongest skew ----------------------------
    L += [BS + "begin{minipage}[t]{0.30" + BS + "textwidth}",
          BS + "centering",
          BS + "textbf{(b) Aggregator at $" + BS + "alpha{=}0.1$}",
          BS + "begin{tabular}{lcccc}", BS + "toprule",
          "Arch. & " + " & ".join(n for _, n in AGGS) + BS + BS,
          BS + "midrule"]
    for key, _ in ARCHS:
        vals = [(a, st.mean(agg_cell[(key, a)])) for a, _ in AGGS
                if agg_cell[(key, a)]]
        best = max(v for _, v in vals) if vals else None
        cells = []
        for a, _ in AGGS:
            if not agg_cell[(key, a)]:
                cells.append("--"); continue
            v = st.mean(agg_cell[(key, a)])
            txt = f"{v:.3f}".lstrip("0") if v < 1 else "1.000"
            cells.append(BS + "textbf{" + txt + "}" if abs(v - best) < 1e-9 else txt)
        L.append(f"{SHORT[key]:<10} & " + " & ".join(cells) + BS + BS)
    L += [BS + "bottomrule", BS + "end{tabular}", BS + "end{minipage}" + BS + "hfill"]

    # ---- (c) cost and federation gain ------------------------------------
    L += [BS + "begin{minipage}[t]{0.28" + BS + "textwidth}",
          BS + "centering",
          BS + "textbf{(c) Cost and federation gain}",
          BS + "begin{tabular}{lcc}", BS + "toprule",
          "Architecture & MiB & $" + BS + "Delta$ local" + BS + BS,
          BS + "midrule"]
    for key, label in ARCHS:
        mib = cost.get(key)
        fed, loc = agg_cell[(key, "fedavg")], agg_cell[(key, "local_only")]
        gain = st.mean(fed) - st.mean(loc) if fed and loc else None
        L.append(f"{label:<17} & " + (f"{mib:.1f}" if mib else "--") + " & "
                 + (("$+$" + f"{gain:.3f}") if gain is not None else "--") + BS + BS)
    L += [BS + "bottomrule", BS + "end{tabular}", BS + "end{minipage}",
          BS + "end{table*}"]

    body = ("%% generated by experiments/make_standard_suite_table.py "
            "-- do not hand-edit\n\n" + "\n".join(L) + "\n")
    OUT.write_bytes(body.encode("utf-8"))
    print(f"wrote {OUT}")
    for key, label in ARCHS:
        fed, loc = agg_cell[(key, "fedavg")], agg_cell[(key, "local_only")]
        print(f"  {label:<18} fedavg {st.mean(fed):.3f}  local {st.mean(loc):.3f} "
              f" gain {st.mean(fed) - st.mean(loc):+.3f}")


if __name__ == "__main__":
    main()
