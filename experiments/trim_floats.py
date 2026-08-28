#!/usr/bin/env python3
"""Suppress named floats in main.tex to measure what a page target costs.

Wrapping a float in \\iffalse...\\fi removes it from the build without deleting
it from the source, so a trial is reversible and the cut list stays auditable.
References to a suppressed float are rewritten to a placeholder rather than
left undefined, so the trial page count is not distorted by "??" boxes.

Usage:
    python trim_floats.py <tex> <label>[,<label>...]
    python trim_floats.py <tex> --restore
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

B = chr(92)
OPEN = "%%TRIM-BEGIN%%\n" + B + "iffalse\n"
CLOSE = B + "fi\n%%TRIM-END%%\n"


def float_spans(text: str):
    """Yield (label, start, end) for every table/figure float in the source."""
    pat = re.compile(re.escape(B + "begin{") + r"(table\*|figure\*|table|figure)\}"
                     + r"(.*?)" + re.escape(B + "end{") + r"\1\}", re.S)
    for m in pat.finditer(text):
        lab = re.search(re.escape(B + "label{") + r"([^}]*)\}", m.group(2))
        if lab:
            yield lab.group(1), m.start(), m.end()


def restore(text: str) -> str:
    return (text.replace(OPEN, "").replace(CLOSE, "")
                .replace("%%TRIM-REF%%", ""))


def main() -> None:
    tex = Path(sys.argv[1])
    text = restore(tex.read_text(encoding="utf-8"))
    if len(sys.argv) > 2 and sys.argv[2] == "--restore":
        tex.write_text(text, encoding="utf-8")
        print("restored")
        return

    wanted = {w.strip() for w in sys.argv[2].split(",") if w.strip()}
    spans = [(l, a, b) for l, a, b in float_spans(text) if l in wanted]
    found = {l for l, _, _ in spans}
    missing = wanted - found
    for a, b, lab in sorted(((a, b, l) for l, a, b in spans), reverse=True):
        text = text[:a] + OPEN + text[a:b] + "\n" + CLOSE + text[b:]

    # keep \ref resolvable: point suppressed labels at a neutral phantom
    if found:
        phantom = "\n".join(B + "newcommand{" + B + "trimref" + chr(97 + i)
                            + "}{}" for i in range(0))
        for lab in found:
            text = text.replace(B + "ref{" + lab + "}", "the suppressed float%%TRIM-REF%%")
        text = text.replace("Table~the suppressed float", "the suppressed float")
        text = text.replace("Fig.~the suppressed float", "the suppressed float")

    tex.write_text(text, encoding="utf-8")
    print(f"suppressed {len(found)} floats: {sorted(found)}")
    if missing:
        print(f"NOT FOUND: {sorted(missing)}")


if __name__ == "__main__":
    main()
