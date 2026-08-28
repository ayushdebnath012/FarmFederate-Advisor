#!/usr/bin/env python3
"""Replace a generated table block in the manuscript with its regenerated form.

Several tables in the paper are produced by a generator and then pasted in. That
works exactly once: after a rerun the generator's output and the pasted copy
drift apart silently, which is how Table II ended up asserting saturation at
1.000 above a body of numbers that top out near 0.56.

This splices by label. It finds the LaTeX float carrying \\label{<label>} in
main.tex, checks that the replacement carries the same label, and swaps the
whole float. Nesting is handled by counting begin/end of the same environment,
so a table* containing minipages is replaced as one unit.

Usage:
    python splice_table_block.py <label> <generated.tex> [<label> <generated.tex> ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEXS = ["overleaf_final/main.tex", "overleaf_final_slim/main.tex"]
B = chr(92)


def find_float(s: str, label: str) -> tuple[int, int, str]:
    """Return (start, end, env) of the float whose body carries the label."""
    marker = B + "label{" + label + "}"
    i = s.index(marker)
    for env in ("table*", "table", "figure*", "figure"):
        open_tag, close_tag = B + "begin{" + env + "}", B + "end{" + env + "}"
        start = s.rfind(open_tag, 0, i)
        if start == -1:
            continue
        # walk forward counting nested opens of the same environment
        depth, j = 0, start
        while j < len(s):
            nxt_o = s.find(open_tag, j + 1)
            nxt_c = s.find(close_tag, j + 1)
            if nxt_c == -1:
                break
            if nxt_o != -1 and nxt_o < nxt_c:
                depth += 1
                j = nxt_o
                continue
            if depth == 0:
                end = nxt_c + len(close_tag)
                if start < i < end:
                    return start, end, env
                break
            depth -= 1
            j = nxt_c
    raise LookupError(f"no float found around {label}")


def main() -> None:
    args = sys.argv[1:]
    if not args or len(args) % 2:
        sys.exit(__doc__)
    jobs = [(args[i], Path(args[i + 1])) for i in range(0, len(args), 2)]

    for label, gen_path in jobs:
        new = gen_path.read_text(encoding="utf-8")
        assert B + "label{" + label + "}" in new, \
            f"{gen_path.name} does not carry label {label}"
        new = new.strip() + "\n"
        for tex in TEXS:
            p = ROOT / tex
            s = p.read_text(encoding="utf-8")
            try:
                a, b, env = find_float(s, label)
            except (LookupError, ValueError) as exc:
                print(f"  {tex}: {label} -- {exc}")
                continue
            if s[a:b].strip() == new.strip():
                print(f"  {tex}: {label} already current")
                continue
            p.write_text(s[:a] + new.rstrip("\n") + s[b:], encoding="utf-8")
            print(f"  {tex}: {label} replaced "
                  f"({b - a} -> {len(new)} chars, {env})")


if __name__ == "__main__":
    main()
