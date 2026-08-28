#!/usr/bin/env python3
"""Emit the federated-robustness table as LaTeX, straight from the result JSON.

Four failure modes in one table, each as a block of rows: severe label skew,
client dropout, straggler staleness and poisoned updates. Columns are the three
federated systems, so the reader can see whether a failure mode hurts the fused
system more or less than its parents -- which is the question a multimodal
paper has to answer and a single-number robustness claim cannot.

Generated rather than transcribed so the numbers cannot drift from the run.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "tea_results" / "federated_robustness" / "federated_robustness.json"
OUT = ROOT / "experiments" / "robustness_table.tex"
SYSTEMS = ["text", "image", "fusion"]
B = chr(92)


def cell(store, key, sysname):
    r = store.get(f"{key}|system={sysname}")
    return None if r is None else r["mean_macro_f1"]


def block(store, title, keys, labels, bold_worst=True):
    """One failure mode. The worst cell in a column is bolded, not the best:
    the point of the block is where the system breaks."""
    rows = [B + "addlinespace[1pt]",
            B + "multicolumn{5}{l}{" + B + "emph{" + title + "}}" + B + B]
    vals = {s: [cell(store, k, s) for k in keys] for s in SYSTEMS}
    worst = {s: min([v for v in vals[s] if v is not None], default=None)
             for s in SYSTEMS}
    for i, (k, lab) in enumerate(zip(keys, labels)):
        cells = []
        for s in SYSTEMS:
            v = vals[s][i]
            if v is None:
                cells.append("--")
                continue
            t = f"{v:.3f}"
            if bold_worst and worst[s] is not None and abs(v - worst[s]) < 1e-12:
                t = B + "textbf{" + t + "}"
            cells.append(t)
        rows.append("~~" + lab + " & " + " & ".join(cells) + B + B)
    return rows


def main() -> None:
    d = json.load(open(SRC, encoding="utf-8"))
    meta = d["_meta"]
    het, drop, strag, pois = (d["heterogeneity"], d["dropout"],
                              d["stragglers"], d["poisoning"])

    alphas = sorted({float(k.split("|")[0].split("=")[1])
                     for k in het if "pathological" not in k})
    het_keys = [f"alpha={a}" for a in sorted(alphas, reverse=True)] + \
               ["alpha=pathological"]
    het_labels = ["$" + B + f"alpha={a:g}$" for a in sorted(alphas, reverse=True)] + \
                 ["one class per client"]

    drop_keys = sorted({k.split("|")[0] for k in drop},
                       key=lambda s: float(s.split("=")[1]))
    drop_labels = [f"{100 * float(k.split('=')[1]):.0f}" + B + "% drop per round"
                   for k in drop_keys]

    st_keys = sorted({k.split("|")[0] for k in strag},
                     key=lambda s: float(s.split("=")[1]))
    st_labels = [f"{100 * float(k.split('=')[1]):.0f}" + B + "% stale per round"
                 for k in st_keys]

    frac = max(float(k.split("@")[1].split("|")[0])
               for k in pois if "@" in k)
    pretty = {"label_flip": "label flip", "sign_flip": "sign flip",
              "gaussian": "Gaussian noise"}
    p_keys, p_labels = [], []
    for agg in ("mean", "median"):
        p_keys.append(f"clean|agg={agg}")
        p_labels.append(f"no attack, {agg}")
        for atk in ("label_flip", "sign_flip", "gaussian"):
            p_keys.append(f"{atk}@{frac}|agg={agg}")
            p_labels.append(f"{pretty[atk]}, {agg}")

    cap = ("Federated robustness on the audited split: macro-F1 after "
           f"{meta['rounds']} rounds with {meta['clients']} clients, mean over "
           f"{len(meta['seeds'])} seeds. Each arm shares its client partition "
           "with the matching control at the same seed. Dropout, straggler and "
           "poisoning blocks fix $" + B + "alpha=" + f"{meta['base_alpha']:g}$; "
           f"the poisoning block uses {100 * frac:.0f}" + B + "% malicious "
           "clients under both mean and coordinate-wise-median aggregation. "
           "Bold marks the worst setting in each column.")

    L = [B + "begin{table}[t]",
         B + "caption{" + cap + "}",
         B + "label{tab:robustness}",
         B + "centering" + B + "footnotesize",
         B + "setlength" + B + "tabcolsep{4.5pt}",
         B + "begin{tabular}{lccc}",
         B + "toprule",
         "Condition & Text & Image & Text$+$Image" + B + B,
         B + "midrule"]
    L += block(het, "Severe label skew", het_keys, het_labels)
    L += [B + "midrule"] + block(drop, "Client dropout", drop_keys, drop_labels)
    L += [B + "midrule"] + block(strag, "Straggler staleness", st_keys, st_labels)
    L += [B + "midrule"] + block(pois, "Poisoned updates", p_keys, p_labels)
    L += [B + "bottomrule", B + "end{tabular}", B + "end{table}"]

    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {OUT}  ({len(L)} lines)")

    # a plain-text digest so the prose can be written against real numbers
    def show(tag, store, keys, labels):
        print(f"\n{tag}")
        print("  " + " " * 26 + "".join(f"{s:<10}" for s in SYSTEMS))
        for k, lab in zip(keys, labels):
            vals = "".join(
                f"{cell(store, k, s):<10.3f}" if cell(store, k, s) is not None
                else f"{'--':<10}" for s in SYSTEMS)
            print(f"  {lab.replace(B, '')[:26]:<26}{vals}")

    show("heterogeneity", het, het_keys, het_labels)
    show("dropout", drop, drop_keys, drop_labels)
    show("stragglers", strag, st_keys, st_labels)
    show("poisoning", pois, p_keys, p_labels)


if __name__ == "__main__":
    main()
