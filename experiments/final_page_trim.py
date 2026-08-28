#!/usr/bin/env python3
"""Last trim to eight pages: abstract tail and one duplicated caveat.

The robustness subsection ends by saying its invariances belong to the probe and
not to a deep federated model. That sentence now also appears in Threats to
Validity, where it belongs, so the local copy goes.

The abstract's federated sentence spends a clause on aggregator ranking and
another on communication cost, both of which are Table II's job. Trimmed to the
finding.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEXS = ["overleaf_final/main.tex", "overleaf_final_slim/main.tex"]
B = chr(92)

JOBS = [
    ("robustness caveat",
     "arm and its control share the client partition at the same seed, so a difference\n"
     "is the failure mode and not the split. Heads are linear probes on standardized\n"
     "frozen features, which is what makes the sweep affordable; the invariances below\n"
     "are properties of that estimator and are not claimed for a deep federated model.",
     "arm and its control share the client partition at the same seed, so a difference\n"
     "is the failure mode and not the split. Heads are linear probes on standardized\n"
     "frozen features, which is what makes the sweep affordable.",
     "which is what makes the sweep affordable.\n"),

    ("abstract tail",
     "Across five federated architectures on 371 genuine\n"
     "box-linked observations, federation outperforms local-only training, although\n"
     "the best aggregator is architecture-dependent; one float32 round costs\n"
     "18.9--67.2" + B + ",MiB per client depending on architecture. Under severe skew, client\n"
     "dropout, straggler staleness and poisoned updates, fusion absorbs the first two\n"
     "and degrades under the last two, with no configuration Byzantine-tolerant.\n"
     "Nothing saturates (the federated sweep spans 0.431--0.561), but clients are\n"
     "simulated and notes are templated, so multi-site validation remains necessary.",

     "Across five federated architectures, federation beats\n"
     "local-only training at 18.9--67.2" + B + ",MiB per client-round, and no aggregator\n"
     "dominates. Under severe skew, client dropout, straggler staleness and poisoned\n"
     "updates, fusion absorbs the first two and degrades under the last two, with no\n"
     "configuration Byzantine-tolerant. Nothing saturates, but clients are simulated\n"
     "and notes are templated, so multi-site validation remains necessary.",
     "at 18.9--67.2"),
]


def main() -> None:
    for tex in TEXS:
        p = ROOT / tex
        s = p.read_text(encoding="utf-8")
        notes, saved = [], 0
        for tag, old, new, marker in JOBS:
            if marker in s:
                notes.append(f"{tag}: already")
            elif s.count(old) == 1:
                saved += len(old) - len(new)
                s = s.replace(old, new)
                notes.append(f"{tag}: ok")
            else:
                notes.append(f"{tag}: ANCHOR x{s.count(old)}")
        p.write_text(s, encoding="utf-8")
        print(f"  {tex}: " + "; ".join(notes) + f"  (-{saved} chars)")


if __name__ == "__main__":
    main()
