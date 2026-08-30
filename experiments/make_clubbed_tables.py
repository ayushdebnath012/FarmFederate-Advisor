#!/usr/bin/env python3
"""Emit the centralized-vs-federated tables from clubbed_tables.json.

Two tables, matching the two-part story:

  Table 1  unimodal   -- text-only and image-only, classical and deep, both
                         centralized and federated
  Table 2  multimodal -- the same protocol on fused features, with the best
                         unimodal rows repeated underneath for reference

Generated rather than transcribed so the printed numbers cannot drift from the
run that produced them.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "tea_results" / "clubbed_tables" / "clubbed_tables.json"
OUT = ROOT / "experiments" / "clubbed_tables.tex"
BS = chr(92)

# printing order: classical floor first, then deep systems weakest-to-strongest
TEXT_ORDER = ["TF-IDF + linear SVM", "BERT-tiny", "BERT-mini", "BERT-small",
              "BERT-medium", "DistilBERT"]
IMAGE_ORDER = ["Colour+HOG SVM", "Swin-tiny", "ConvNeXT-tiny", "EfficientNet",
               "DeiT-tiny", "ViT-Base"]
MM_ORDER = ["TF-IDF + Colour/HOG SVM", "ViT-Base + BERT-tiny",
            "ViT-Base + BERT-mini", "ViT-Base + BERT-small",
            "ViT-Base + BERT-medium", "ViT-Base + DistilBERT"]
PRETTY = {"EfficientNet": "EfficientNet-B0", "ViT-Base": "ViT-Base/16"}


def esc(s: str) -> str:
    return s.replace("%", BS + "%").replace("+", "$+$")


def load():
    d = json.load(open(SRC, encoding="utf-8"))
    keys = sorted(d["tables"], key=lambda k: float(k.split("=")[1]))
    return d, keys


def head(sparsity_keys):
    n = len(sparsity_keys)
    spec = "lr" + "cccc" * n
    lines = [BS + "begin{tabular}{" + spec + "}", BS + "toprule"]
    top = ["", ""]
    rules = []
    col = 3
    for k in sparsity_keys:
        sp = float(k.split("=")[1])
        name = "Complete notes" if sp == 0 else f"{sp:.0%} of notes deleted"
        top.append(BS + "multicolumn{4}{c}{" + esc(name) + "}")
        rules.append(BS + f"cmidrule(lr){{{col}-{col + 3}}}")
        col += 4
    lines.append(" & ".join(top) + BS + BS)
    lines.append("".join(rules))
    mid = ["", ""]
    rules, col = [], 3
    for _ in sparsity_keys:
        mid.append(BS + "multicolumn{2}{c}{Centralized}")
        mid.append(BS + "multicolumn{2}{c}{Federated}")
        rules.append(BS + f"cmidrule(lr){{{col}-{col + 1}}}")
        rules.append(BS + f"cmidrule(lr){{{col + 2}-{col + 3}}}")
        col += 4
    lines.append(" & ".join(mid) + BS + BS)
    lines.append("".join(rules))
    cols = ["System", "Dim."] + ["Acc.", "F1"] * (2 * len(sparsity_keys))
    lines.append(" & ".join(cols) + BS + BS)
    lines.append(BS + "midrule")
    return lines


def row(name, tables, keys, family, bold_on):
    cells = []
    dim = None
    for k in keys:
        r = tables[k][family].get(name)
        if r is None:
            cells += ["--"] * 4
            continue
        dim = r["dim"]
        for field in ("central_test_accuracy", "central_test_macro_f1",
                      "fed_test_accuracy", "fed_test_macro_f1"):
            v = f"{r[field]:.3f}"
            if (k, field) in bold_on and abs(r[field] - bold_on[(k, field)]) < 1e-9:
                v = BS + "textbf{" + v + "}"
            cells.append(v)
    label = esc(PRETTY.get(name, name))
    return f"{label} & {dim if dim else '--'} & " + " & ".join(cells) + BS + BS


def best_map(tables, keys, family, names):
    """Column-wise maxima within one family block, used only for bolding."""
    out = {}
    for k in keys:
        for field in ("central_test_accuracy", "central_test_macro_f1",
                      "fed_test_accuracy", "fed_test_macro_f1"):
            vals = [tables[k][family][n][field] for n in names
                    if n in tables[k][family]]
            if vals:
                out[(k, field)] = max(vals)
    return out


def block(title, family, order, tables, keys):
    names = [n for n in order if n in tables[keys[0]][family]]
    extra = [n for n in tables[keys[0]][family] if n not in names]
    names += extra
    bold = best_map(tables, keys, family, names)
    ncol = 2 + 4 * len(keys)
    lines = [BS + "multicolumn{" + str(ncol) + "}{l}{" + BS + "emph{"
             + title + "}}" + BS + BS]
    lines += [row(n, tables, keys, family, bold) for n in names]
    return lines


def table_unimodal(d, keys):
    t = d["tables"]
    fed = d["_meta"]["federated"]
    cap = (
        "Unimodal systems on the common 75-crop test support, centralized and "
        "federated. Every row uses the identical 222/74/75 source-grouped split, "
        "the identical crop-linked notes, and the identical train-fitted leakage "
        "mask; configurations are chosen on the 74 validation crops and test is "
        "scored once. Federated rows are FedAvg over $K{=}"
        + str(fed["clients"]) + "$ Dirichlet($\\alpha{=}"
        + f"{fed['dirichlet_alpha']:g}"
        + "$) clients for " + str(fed["rounds"]) + " rounds, averaged over "
        + str(len(fed["seeds"])) + " seeds, with the round chosen on validation "
        "only. Centralized heads are closed-form and federated heads are "
        "SGD-trained, because a closed-form solve has no gradients to average, "
        "so the two columns are close but not the same estimator. Image rows do "
        "not move with note sparsity by construction. This table varies the "
        "encoder under one fixed pipeline --- frozen features with a single "
        "closed-form head --- whereas Figs.~\\ref{fig:c_visual_ladder} "
        "and~\\ref{fig:c_fusion_ladder} vary the pipeline for a fixed encoder. "
        "Bold marks the best value in each block and column."
    )
    L = [BS + "begin{table*}[t]", BS + "caption{" + cap + "}",
         BS + "label{tab:unimodal_cent_fed}",
         BS + "centering" + BS + "footnotesize",
         BS + "setlength" + BS + "tabcolsep{4pt}"]
    L += head(keys)
    L += block("Text only", "text_only", TEXT_ORDER, t, keys)
    L.append(BS + "midrule")
    L += block("Image only", "image_only", IMAGE_ORDER, t, keys)
    L += [BS + "bottomrule", BS + "end{tabular}", BS + "end{table*}"]
    return "\n".join(L)


def table_multimodal(d, keys):
    t = d["tables"]
    cap = (
        "Multimodal systems on the same 75 crops, under the same protocol as "
        "Table~" + BS + "ref{tab:unimodal_cent_fed}. The best text-only and "
        "best image-only rows are repeated at the foot for reference. At "
        "complete notes the text branch is already saturated, so fusion cannot "
        "improve on it and the comparison is uninformative; the separation "
        "appears once notes are incomplete, which is the realistic field "
        "condition. Bold marks the best value in each column."
    )
    L = [BS + "begin{table*}[t]", BS + "caption{" + cap + "}",
         BS + "label{tab:multimodal_cent_fed}",
         BS + "centering" + BS + "footnotesize",
         BS + "setlength" + BS + "tabcolsep{4pt}"]
    L += head(keys)
    L += block("Multimodal (image $+$ text)", "multimodal", MM_ORDER, t, keys)
    L.append(BS + "midrule")

    # reference rows: the strongest unimodal system per family, by centralized
    # test accuracy at the sparsest setting actually measured
    ref_key = keys[-1]
    L.append(BS + "multicolumn{" + str(2 + 4 * len(keys)) + "}{l}{" + BS
             + "emph{Best unimodal system, repeated for reference}}" + BS + BS)
    for family, tag in (("text_only", "Text only"), ("image_only", "Image only")):
        rows = t[ref_key][family]
        pick = max(rows, key=lambda n: (rows[n]["central_test_accuracy"],
                                        rows[n]["central_test_macro_f1"]))
        cells, dim = [], None
        for k in keys:
            r = t[k][family][pick]
            dim = r["dim"]
            cells += [f"{r[f]:.3f}" for f in
                      ("central_test_accuracy", "central_test_macro_f1",
                       "fed_test_accuracy", "fed_test_macro_f1")]
        label = f"{tag}: {PRETTY.get(pick, pick)}"
        L.append(esc(label) + f" & {dim} & " + " & ".join(cells) + BS + BS)
    L += [BS + "bottomrule", BS + "end{tabular}", BS + "end{table*}"]
    return "\n".join(L)


def main() -> None:
    d, keys = load()
    body = ("%% generated by experiments/make_clubbed_tables.py "
            "-- do not hand-edit\n\n")
    body += table_unimodal(d, keys) + "\n\n"
    body += table_multimodal(d, keys) + "\n"
    OUT.write_text(body, encoding="utf-8", newline="\n")
    print(f"wrote {OUT}  ({len(body.splitlines())} lines)")


if __name__ == "__main__":
    main()
