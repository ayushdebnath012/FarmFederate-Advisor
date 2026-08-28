"""Detect real whitespace gaps by rasterising each page and finding blank bands
inside each text column. Catches gaps the 'lowest content block' metric misses."""
import sys
import fitz
import numpy as np

PDF = sys.argv[1] if len(sys.argv) > 1 else "main.pdf"
DPI = 100
S = DPI / 72.0
# IEEEtran letter geometry (pt): text area and the two columns
TOP, BOT = 54, 738
LCOL = (54, 300)
RCOL = (312, 558)
MIN_GAP_PT = 24          # ignore normal inter-paragraph / float spacing

doc = fitz.open(PDF)
total_waste = 0.0
for pno in range(doc.page_count):
    pix = doc[pno].get_pixmap(dpi=DPI)
    img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
    gray = img[:, :, :3].min(axis=2)
    page_lines = []
    for name, (x0, x1) in (("L", LCOL), ("R", RCOL)):
        band = gray[int(TOP * S):int(BOT * S), int(x0 * S):int(x1 * S)]
        blank = (band > 246).all(axis=1)
        runs, st = [], None
        for i, b in enumerate(blank):
            if b and st is None:
                st = i
            elif not b and st is not None:
                runs.append((st, i)); st = None
        tail = None
        if st is not None:
            tail = (st, len(blank))          # trailing blank = unused column bottom
        gaps = [(a / S, (b - a) / S) for a, b in runs if (b - a) / S >= MIN_GAP_PT]
        for top_off, h in gaps:
            page_lines.append(f"    {name}: interior gap {h:5.0f}pt at y={TOP+top_off:.0f}")
            total_waste += h
        if tail:
            h = (tail[1] - tail[0]) / S
            if h >= MIN_GAP_PT:
                page_lines.append(f"    {name}: TRAILING blank {h:5.0f}pt "
                                  f"({h/(BOT-TOP)*100:.0f}% of column)")
                total_waste += h
    if page_lines:
        print(f"  page {pno+1}:")
        print("\n".join(page_lines))
print(f"\ntotal wasted column-height: {total_waste:.0f}pt "
      f"({total_waste/(BOT-TOP)/2*100:.0f}% of one page)")
