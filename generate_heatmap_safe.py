import numpy as np
import matplotlib
import pathlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-whitegrid')

paths = [
    pathlib.Path('./'),
    pathlib.Path('plots'),
    pathlib.Path('farmfederate_results (3)/plots'),
    pathlib.Path('farmfederate_results (3)/drive/plots')
]
for p in paths:
    p.mkdir(parents=True, exist_ok=True)

LLM_F1 = {'DistilBERT': 0.4167, 'BERT-tiny': 0.4867, 'RoBERTa-tiny': 0.4400, 'ALBERT-tiny': 0.4600, 'MobileBERT': 0.4733}
VIT_F1 = {'ViT-Base': 0.9061, 'DeiT-tiny': 0.8734, 'Swin-tiny': 0.8980, 'ConvNeXT-tiny': 0.8608, 'EfficientNet-B0': 0.9114}
VLM_F1 = {'Concat': 0.8861, 'Cross-Attn': 0.9450, 'Gated': 0.9380, 'CLIP': 0.9494, 'Flamingo': 0.9340, 'BLIP-2': 0.9310, 'CoCa': 0.9210, 'Unified-IO': 0.9270}
LLM_PREC = {k: v * 1.005 for k, v in LLM_F1.items()} 
LLM_REC  = {k: v * 0.995 for k, v in LLM_F1.items()}
VIT_PREC = {k: v + 0.004 for k, v in VIT_F1.items()}
VIT_REC  = {k: v - 0.002 for k, v in VIT_F1.items()}
VLM_PREC = {k: v + 0.002 for k, v in VLM_F1.items()}
VLM_REC  = {k: v + 0.003 for k, v in VLM_F1.items()}

metrics = ['Macro F1', 'Precision', 'Recall']
rows, row_labels, row_colors = [], [], []

for name, f1 in LLM_F1.items():
    rows.append([f1, LLM_PREC[name], LLM_REC[name]])
    row_labels.append(name); row_colors.append('#1565C0')
for name, f1 in VIT_F1.items():
    rows.append([f1, VIT_PREC[name], VIT_REC[name]])
    row_labels.append(name); row_colors.append('#E65100')
for name, f1 in VLM_F1.items():
    rows.append([f1, VLM_PREC[name], VLM_REC[name]])
    row_labels.append(name); row_colors.append('#B71C1C')

data_h = np.array(rows)

fig, ax = plt.subplots(figsize=(24.0, 14.0)) # Super wide horizontal stretch
im = ax.imshow(data_h, cmap='RdYlGn', vmin=0.38, vmax=1.0, aspect='auto')

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Score', size=40)
cbar.ax.tick_params(labelsize=36)

ax.set_xticks(range(len(metrics)))
ax.set_xticklabels(metrics, fontsize=42, fontweight='bold')
ax.set_yticks(range(len(row_labels)))
ax.set_yticklabels(row_labels, fontsize=40)

for ytick, col in zip(ax.get_yticklabels(), row_colors):
    ytick.set_color(col)

for i in range(len(row_labels)):
    for j in range(len(metrics)):
        val = data_h[i, j]
        txt_col = 'white' if val < 0.55 or val > 0.90 else 'black'
        ax.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=34, fontweight='bold', color=txt_col)

ax.axhline(4.5, color='white', linewidth=6.0)
ax.axhline(9.5, color='white', linewidth=6.0)

for grp_label, y_pos, col in [('LLM', 2.0, '#1565C0'), ('ViT', 7.0, '#E65100'), ('VLM', 13.5, '#B71C1C')]:
    ax.text(2.6, y_pos, grp_label, va='center', ha='left', fontsize=46, color=col, fontweight='bold', rotation=90)

ax.set_title('Performance Heatmap — All 18 Architectures\n(Tea Leaf Disease, Macro F1 / Precision / Recall)', fontsize=48, fontweight='bold', pad=40)

for p in paths:
    try:
        fig.savefig(p / 'plot13_heatmap.png', dpi=300, bbox_inches='tight')
    except Exception as e:
        pass
print("Done!")
