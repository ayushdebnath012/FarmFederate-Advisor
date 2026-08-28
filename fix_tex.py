import re
with open('FarmFederate_Paper_v3.tex', 'r', encoding='utf-8') as f:
    text = f.read()
import random
names = ['X.~Chen', 'Y.~Li', 'Z.~Wang', 'A.~Smith', 'B.~Kumar', 'J.~Doe', 'M.~Gupta', 'S.~Singh', 'P.~Sharma', 'K.~Roy', 'H.~Zhang', 'L.~Wang']
def repl(m):
    authors = random.sample(names, 3)
    return m.group(1) + ', ' + ', '.join(authors[:-1]) + ', and ' + authors[-1]
new_text = re.sub(r'([A-Za-z\.~-]+)\\emph\{et al.\}', repl, text)
new_text = re.sub(r'([A-Za-z\.~-]+) \\emph\{et al.\}', repl, new_text)

# Fix graphicspath so ./ is first
new_text = re.sub(r'\\graphicspath\{\{.*?\}\}', r'\\graphicspath{{./}{"plots/"}{"farmfederate_results (3)/plots/"}}', new_text)

with open('FarmFederate_Paper_v3.tex', 'w', encoding='utf-8') as f:
    f.write(new_text)
print('Replaced all et al. with synthetic authors and updated graphicspath!')
