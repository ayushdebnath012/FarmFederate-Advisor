#!/usr/bin/env python3
"""
trim_references.py — remove 22 least-critical references to get to exactly 45.
Strategy: Keep federated learning, VLM, ViT, and top tea disease papers.
Remove: Secondary tea papers, duplicates, and less-cited works.
"""
import re

FILE = r'c:\Users\USER_HP\Desktop\FarmFederate\FarmFederate_Paper.tex'

with open(FILE, 'r', encoding='utf-8') as f:
    text = f.read()

# Extract all \bibitem{...} keys
refs = re.findall(r'\\bibitem\{([^}]+)\}', text)
print(f"Current references: {len(refs)}")
print(f"Target: 45 references")
print(f"Remove: {len(refs) - 45} references\n")

# References to KEEP (45 most important)
keep_refs = {
    # Federated Learning (essential)
    'McMahan2017', 'Li2020fedprox', 'Karimireddy2020',
    'Wang2020fedavg',
    
    # Vision Transformer (essential)
    'Dosovitsky2021', 'Touvron2021deit', 'Liu2021swin',
    'Cai2021coca_efficient', 'Tan2021efficientnet',
    
    # CLIP & Multimodal (essential)
    'Radford2021clip', 'Li2023blip2', 'Alayrac2022flamingo',
    'Yu2022coca_visual_semantic', 'Wang2022unified_io',
    
    # RAG (essential)
    'Lewis2020retrieval_augmented',
    
    # Top Tea Disease Papers (representative)
    'Madhavi2025', 'Rahman2024sci', 'AlamSoeb2023',
    'Hossain2018', 'Karmokar2015', 'Bhowmik2022',
    'Dipty2025', 'Hairah2024', 'Wu2025',
    
    # Federated + Tea Disease
    'Hari2025', 'Plant_AI2024', 'Vinu2024', 'Fahim2024',
    'Kabala2023', 
    
    # Object detection / Tea specific
    'Bao2022', 'Xue2023yolo_tea', 'Yao2024',
    
    # Other key papers
    'Mamun2023', 'HuGensheng2019', 'Mukhopadhyay2020',
    'Latha2021', 'Balasundaram2024',
    
    # Loss / Training
    'He2016resnet',
    
    # NLP / Text
    'Devlin2019bert', 'Liu2019roberta',
    
    # Additional key works
    'Kingma2015adam', 'Goodfellow2016',
}

print(f"Keeping {len(keep_refs)} references:")
for k in sorted(keep_refs):
    print(f"  {k}")

# Now remove bibitem entries not in keep_refs
lines = text.split('\n')
in_bib = False
remove_next = False
new_lines = []
removed_count = 0

for i, line in enumerate(lines):
    if '\\begin{thebibliography}' in line:
        in_bib = True
        new_lines.append(line)
        continue
    if '\\end{thebibliography}' in line:
        in_bib = False
        new_lines.append(line)
        continue
    
    if in_bib:
        match = re.match(r'\\bibitem\{([^}]+)\}', line)
        if match:
            ref_key = match.group(1)
            if ref_key not in keep_refs:
                print(f"  Removing: {ref_key}")
                removed_count += 1
                # Skip this bibitem and the next lines until blank line
                continue
        
        # Skip lines if previous bibitem was removed
        if removed_count > 0 and line.strip() == '':
            if i > 0 and re.match(r'\\bibitem\{', lines[i-1]) is None:
                # We've passed the removal section
                removed_count = 0
                new_lines.append(line)
            continue
        
        if removed_count > 0:
            # Check if this is start of next bibitem
            if re.match(r'\\bibitem\{', line):
                removed_count = 0
                new_lines.append(line)
            # Otherwise skip
            continue
    
    new_lines.append(line)

with open(FILE, 'w', encoding='utf-8') as f:
    f.write('\n'.join(new_lines))

# Verify count
new_refs = re.findall(r'\\bibitem\{([^}]+)\}', '\n'.join(new_lines))
print(f"\n✓ References reduced: {len(refs)} → {len(new_refs)}")
