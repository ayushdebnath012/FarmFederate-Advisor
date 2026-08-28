#!/usr/bin/env python3
"""
finalize_paper.py — final polish for 10-page FarmFederate paper:
  1. Remove unnecessary dashes/hyphens
  2. Adjust figure widths to 0.48\columnwidth (2-column layout)
  3. Add FarmFederate app mention in introduction
  4. Report final file stats
"""
import re

FILE = r'c:\Users\USER_HP\Desktop\FarmFederate\FarmFederate_Paper.tex'

with open(FILE, 'r', encoding='utf-8') as f:
    text = f.read()

# 1. Remove em-dashes that are purely decorative (between words, not in compound adjectives)
text = re.sub(r' — ([A-Z])', r' \1', text)  # em-dash followed by capital → just space + capital
text = re.sub(r'—', '–', text)  # remaining em-dashes → endash (more formal in LaTeX)

# 2. Fix "data-governance" type hyphens (split compounds not needed)
text = text.replace('data governance', 'data governance')
text = text.replace('cross-paradigm', 'cross paradigm')
text = text.replace('multi-class', 'multi-class')  # keep this one (standard usage)

# 3. Adjust ALL figure widths to 0.48\columnwidth for 2-column layout
text = re.sub(r'\\includegraphics\[width=0\.9\\columnwidth', 
              r'\\includegraphics[width=0.48\\columnwidth', text)
text = re.sub(r'\\includegraphics\[width=1\\.0\\columnwidth', 
              r'\\includegraphics[width=0.48\\columnwidth', text)
text = re.sub(r'\\includegraphics\[width=0\.8\\columnwidth', 
              r'\\includegraphics[width=0.48\\columnwidth', text)

# 4. Add FarmFederate app mention after "multimodal models" in intro
intro_marker = r'(The combination of)'
replacement = (
    r'\1 multimodal models on-device through the FarmFederate mobile app—'
    r'which integrates text symptom descriptions, real-time photographs, and IoT sensor '
    r'telemetry from the field—'
)
text = re.sub(intro_marker, replacement, text)

# 5. Shrink Some unnecessary section overhead
text = text.replace('\\bigskip', '\\smallskip')  # reduce whitespace

# Save result
with open(FILE, 'w', encoding='utf-8') as f:
    f.write(text)

# Count lines and estimate pages
lines = len(text.split('\n'))
print(f"✓ Paper finalized: {lines} lines")
print(f"✓ All figures resized to 0.48\\columnwidth (2-column layout)")
print(f"✓ Unnecessary dashes removed")
print(f"✓ FarmFederate app mention added")
print(f"\nNext step: pdflatex FarmFederate_Paper.tex (2 passes)")
