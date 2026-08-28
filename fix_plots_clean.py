"""
fix_plots_clean.py — Regenerate plots with proper formatting and spacing.
Fixes overlapping text, improves readability, and uses appropriate figure sizes.
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================
BASE = Path('C:/Users/USER_HP/Desktop/FarmFederate')
RESULTS_FILE = BASE / 'farmfederate_results_20260418_114930' / 'results' / 'complete_results.json'
PLOTS_DIR = BASE / 'plots'
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Load results
with open(RESULTS_FILE) as f:
    results = json.load(f)

# ============================================================================
# PLOT SETTINGS
# ============================================================================
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 11,
    'lines.linewidth': 2.0,
    'axes.linewidth': 1.5,
})

# ============================================================================
# FIX 1: LLM/ViT/VLM Encoder Comparisons (Horizontal Bar Charts)
# ============================================================================

def fix_encoder_comparisons():
    """Generate clean encoder comparison plots with proper spacing."""
    
    llm_data = results.get('llm_models', {})
    vit_data = results.get('vit_models', {})
    vlm_data = results.get('vlm_models', {})
    
    # ────────────────────────────────────────────────────────────────────────
    # LLM Encoders
    # ────────────────────────────────────────────────────────────────────────
    if llm_data:
        fig, ax = plt.subplots(figsize=(10, 5))
        names = list(llm_data.keys())
        f1_scores = [llm_data[n].get('f1', 0) for n in names]
        
        # Sort by F1 score
        sorted_pairs = sorted(zip(names, f1_scores), key=lambda x: x[1])
        names, f1_scores = zip(*sorted_pairs)
        
        colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(names)))
        bars = ax.barh(names, f1_scores, color=colors, edgecolor='black', linewidth=1.5)
        
        # Add value labels
        for i, (bar, score) in enumerate(zip(bars, f1_scores)):
            ax.text(score + 0.015, i, f'{score:.3f}', va='center', fontsize=11, fontweight='bold')
        
        ax.set_xlabel('Macro F1 Score', fontsize=12, fontweight='bold')
        ax.set_title('LLM Encoder Comparison', fontsize=13, fontweight='bold', pad=15)
        ax.set_xlim(0, 1.0)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / 'plot01_llm_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ LLM encoder comparison fixed")
    
    # ────────────────────────────────────────────────────────────────────────
    # ViT Encoders
    # ────────────────────────────────────────────────────────────────────────
    if vit_data:
        fig, ax = plt.subplots(figsize=(10, 5))
        names = list(vit_data.keys())
        f1_scores = [vit_data[n].get('f1', 0) for n in names]
        
        # Sort by F1 score
        sorted_pairs = sorted(zip(names, f1_scores), key=lambda x: x[1])
        names, f1_scores = zip(*sorted_pairs)
        
        colors = plt.cm.Oranges(np.linspace(0.4, 0.9, len(names)))
        bars = ax.barh(names, f1_scores, color=colors, edgecolor='black', linewidth=1.5)
        
        # Add value labels
        for i, (bar, score) in enumerate(zip(bars, f1_scores)):
            ax.text(score + 0.015, i, f'{score:.3f}', va='center', fontsize=11, fontweight='bold')
        
        ax.set_xlabel('Macro F1 Score', fontsize=12, fontweight='bold')
        ax.set_title('ViT Encoder Comparison', fontsize=13, fontweight='bold', pad=15)
        ax.set_xlim(0, 1.0)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / 'plot02_vit_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ ViT encoder comparison fixed")
    
    # ────────────────────────────────────────────────────────────────────────
    # VLM Fusion
    # ────────────────────────────────────────────────────────────────────────
    if vlm_data:
        fig, ax = plt.subplots(figsize=(10, 6))
        names = list(vlm_data.keys())
        f1_scores = [vlm_data[n].get('f1', 0) for n in names]
        
        # Sort by F1 score
        sorted_pairs = sorted(zip(names, f1_scores), key=lambda x: x[1], reverse=True)
        names, f1_scores = zip(*sorted_pairs)
        
        colors = plt.cm.RdYlGn(np.linspace(0.4, 0.9, len(names)))
        bars = ax.barh(names, f1_scores, color=colors, edgecolor='black', linewidth=1.5)
        
        # Add value labels
        for i, (bar, score) in enumerate(zip(bars, f1_scores)):
            ax.text(score + 0.015, i, f'{score:.3f}', va='center', fontsize=11, fontweight='bold')
        
        ax.set_xlabel('Macro F1 Score', fontsize=12, fontweight='bold')
        ax.set_title('VLM Fusion Architecture Comparison', fontsize=13, fontweight='bold', pad=15)
        ax.set_xlim(0, 1.0)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / 'plot03_vlm_fusion_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ VLM fusion comparison fixed")

# ============================================================================
# FIX 2: Performance Heatmaps (with proper sizing and annotations)
# ============================================================================

def fix_performance_heatmap():
    """Generate clean performance heatmap with readable annotations."""
    
    llm_data = results.get('llm_models', {})
    vit_data = results.get('vit_models', {})
    vlm_data = results.get('vlm_models', {})
    
    rows = []
    row_labels = []
    
    # Collect all data
    for model_type, models in [('LLM', llm_data), ('ViT', vit_data), ('VLM', vlm_data)]:
        for name, data in models.items():
            f1 = data.get('f1', 0)
            precision = data.get('precision', 0)
            recall = data.get('recall', 0)
            rows.append([f1, precision, recall])
            row_labels.append(f"{model_type}-{name}")
    
    if rows:
        data_array = np.array(rows)
        
        # Create larger figure for heatmap
        fig, ax = plt.subplots(figsize=(10, max(8, len(rows) * 0.4)))
        
        sns.heatmap(data_array, 
                   annot=True, 
                   fmt='.3f',
                   cmap='RdYlGn',
                   vmin=0,
                   vmax=1,
                   xticklabels=['F1 Score', 'Precision', 'Recall'],
                   yticklabels=row_labels,
                   ax=ax,
                   linewidths=1,
                   linecolor='gray',
                   cbar_kws={'label': 'Score'},
                   annot_kws={'size': 10, 'weight': 'bold'})
        
        ax.set_title('Performance Heatmap (All Models)', fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
        ax.set_ylabel('Model', fontsize=12, fontweight='bold')
        
        # Improve readability
        plt.setp(ax.get_yticklabels(), rotation=0, fontsize=10)
        plt.setp(ax.get_xticklabels(), rotation=0, fontsize=11)
        
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / 'plot_performance_heatmap_fixed.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Performance heatmap fixed")

# ============================================================================
# FIX 3: Federated vs Centralized Comparison
# ============================================================================

def fix_fed_vs_centralized():
    """Generate clean federated vs centralized comparison."""
    
    fed_data = results.get('federated', {})
    cent_data = results.get('centralized', {})
    
    if fed_data and cent_data:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        model_types = list(fed_data.keys())
        x = np.arange(len(model_types))
        width = 0.35
        
        cent_f1 = [cent_data[m].get('f1', 0) for m in model_types]
        fed_f1 = [fed_data[m].get('f1', 0) for m in model_types]
        
        bars1 = ax.bar(x - width/2, cent_f1, width, label='Centralized', 
                      color='#3498db', edgecolor='black', linewidth=1.5)
        bars2 = ax.bar(x + width/2, fed_f1, width, label='Federated', 
                      color='#e74c3c', edgecolor='black', linewidth=1.5)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Model Type', fontsize=12, fontweight='bold')
        ax.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
        ax.set_title('Federated vs Centralized Training', fontsize=13, fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(model_types, fontsize=11)
        ax.set_ylim(0, 1.0)
        ax.legend(fontsize=11, loc='lower right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / 'plot05_fed_vs_centralized_fixed.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Federated vs Centralized comparison fixed")

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("FIXING PLOT FORMATTING")
    print("="*70)
    
    try:
        fix_encoder_comparisons()
        fix_performance_heatmap()
        fix_fed_vs_centralized()
        print("\n✓ All plots fixed successfully!")
        print(f"  Output directory: {PLOTS_DIR}")
    except Exception as e:
        print(f"\n✗ Error during plot generation: {e}")
        import traceback
        traceback.print_exc()
