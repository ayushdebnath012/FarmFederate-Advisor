"""Move the three large federated floats earlier in the source.

They are declared in the Federated Training subsection, ~1.5 pages before the
end, so with a FloatBarrier before the bibliography they pile up and force an
extra page. Declaring them earlier lets LaTeX place them on earlier pages while
body text keeps flowing, so the references still come last.
"""
import io

p = "main.tex"
L = io.open(p, encoding="utf-8").read().split("\n")


def find_block(start_pat, end_pat, after=0):
    for i in range(after, len(L)):
        if L[i].startswith(start_pat):
            for j in range(i, len(L)):
                if L[j].startswith(end_pat):
                    return i, j + 1
    raise SystemExit(f"not found: {start_pat}")


# locate by label so we are independent of line numbers
def block_with_label(label):
    for i, l in enumerate(L):
        if label in l:
            s = next(k for k in range(i, -1, -1)
                     if L[k].startswith("\\begin{table*}")
                     or L[k].startswith("\\begin{figure*}"))
            endtok = "\\end{table*}" if L[s].startswith("\\begin{table*}") else "\\end{figure*}"
            e = next(k for k in range(i, len(L)) if L[k].startswith(endtok))
            return s, e + 1
    raise SystemExit(f"label not found: {label}")


spans = [block_with_label("{tab:fed}"),
         block_with_label("{fig:fedsweep}"),
         block_with_label("{fig:fedeffect}")]
spans.sort()
blocks = [L[s:e] for s, e in spans]

# delete from the bottom up so earlier indices stay valid
for s, e in sorted(spans, reverse=True):
    del L[s:e]

payload = []
for b in blocks:
    payload += b + [""]

anchor = next(i for i, l in enumerate(L)
              if l.startswith("\\subsection{Why the Image Branch"))
L[anchor:anchor] = payload

io.open(p, "w", encoding="utf-8").write("\n".join(L))
print(f"moved {len(blocks)} floats to line {anchor + 1}")
