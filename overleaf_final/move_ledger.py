"""Move the eligibility-ledger table out of the Conclusion.

Declared in the Conclusion with [!b] it lands on the references page, below the
REFERENCES heading. Declaring it at the head of Threats to Validity places it
before the bibliography, which is where a summary of what the evaluation
licenses belongs anyway.
"""
import io

p = "main.tex"
L = io.open(p, encoding="utf-8").read().split("\n")

i = next(k for k, l in enumerate(L) if "{tab:ledger}" in l)
s = next(k for k in range(i, -1, -1) if L[k].startswith("\\begin{table}"))
e = next(k for k in range(i, len(L)) if L[k].startswith("\\end{table}")) + 1
block = L[s:e]
block[0] = "\\begin{table}[!t]"        # top placement, not bottom-of-page
del L[s:e]
# drop the now-orphaned blank line left behind
if s < len(L) and L[s].strip() == "" and L[s - 1].strip() == "":
    del L[s]

anchor = next(k for k, l in enumerate(L) if l.startswith("\\subsection{Threats to Validity}"))
L[anchor:anchor] = block + [""]
io.open(p, "w", encoding="utf-8").write("\n".join(L))
print(f"moved ledger table to line {anchor + 1} (head of Threats to Validity)")
