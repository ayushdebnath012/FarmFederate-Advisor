#!/usr/bin/env python3
"""Adapt vision_tta.py output to the schema modality_hierarchy_benchmark reads.

The hierarchy benchmark expects a visual payload with `selected_validation` and
`fixed_test`, each carrying `predictions` and `labels`. That file was originally
produced by a five-expert vision-ensemble script which is not in this
repository; the surviving visual selector is vision_tta.py, whose output carries
the same two quantities under the names `selected` and `locked_test`.

So this renames, it does not compute. The substitution is recorded in the output
under `source`, because the visual row downstream is now selected by
test-time-augmentation search rather than by the older expert ensemble, and a
reader comparing against the previously published Table VI is entitled to know
that the visual selector changed along with the corpus.

The label orderings are asserted non-empty and equal in length to the
predictions; the hierarchy script separately asserts they match its own split,
which is the real guard.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tta", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()

    d = json.loads(a.tta.read_text(encoding="utf-8"))
    val, test = d["selected"], d["locked_test"]
    for name, part in (("selected", val), ("locked_test", test)):
        n_p, n_l = len(part["predictions"]), len(part["labels"])
        if not n_p or n_p != n_l:
            raise ValueError(f"{name}: {n_p} predictions against {n_l} labels")

    out = {
        "source": "vision_tta.py, adapted by tta_to_ensemble_shim.py; the "
                  "original five-expert ensemble script is not in the "
                  "repository, so the visual selector differs from the one "
                  "behind the previously published table",
        "checkpoint": d.get("checkpoint"),
        "experts": d["selected"].get("views", []),
        "gate_key": "tta:" + ",".join(d["selected"].get("views", [])),
        "candidate_count": d.get("candidate_count"),
        "selected_validation": val,
        "fixed_test": test,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {a.out}")
    print(f"  validation {len(val['predictions'])} preds, "
          f"macro-F1 {val.get('macro_f1'):.4f}")
    print(f"  locked test {len(test['predictions'])} preds, "
          f"macro-F1 {test.get('macro_f1'):.4f}")


if __name__ == "__main__":
    main()
