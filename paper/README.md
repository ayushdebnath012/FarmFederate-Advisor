# FarmFederate Paper - LaTeX Documentation

This directory contains the complete LaTeX paper for the FarmFederate project.

## Files

- `main.tex` - Main paper with all sections (Introduction, Related Work, Methodology, Results, Discussion, Conclusion)
- `architectures.tex` - Standalone TikZ architecture diagrams
- `figures/` - Directory for generated plots from training (created by running FarmFederate_Colab.py)

## Compilation

### Prerequisites

Install a LaTeX distribution:
- **Windows**: MiKTeX or TeX Live
- **Linux**: `sudo apt install texlive-full`
- **macOS**: MacTeX

### Compile Main Paper

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Or use `latexmk` for automatic compilation:
```bash
latexmk -pdf main.tex
```

### Compile Architecture Diagrams

```bash
pdflatex architectures.tex
```

## Generated Plots Integration

After running `FarmFederate_Colab.py`, copy the plots from `output/plots/` to `paper/figures/`:

```bash
# From FarmFederate root directory
cp -r output/plots/* paper/figures/
```

Then add to `main.tex`:

```latex
\begin{figure}[htbp]
\centering
\includegraphics[width=\columnwidth]{figures/model_comparison_f1.png}
\caption{F1 Score comparison across all models.}
\end{figure}
```

## Paper Structure

1. **Abstract** - Summary of contributions and results
2. **Introduction** - Problem statement and contributions
3. **Related Work** - Comparison with 45+ SOTA methods
4. **Methodology**
   - System Architecture
   - LLM Encoders (5 variants)
   - ViT Encoders (5 variants)
   - VLM Fusion (8 architectures)
   - Class Imbalance Handling
   - Federated Learning
   - Semantic Retrieval
5. **Experimental Setup** - Datasets, training config, metrics
6. **Results** - Performance tables, ablation study
7. **Discussion** - Analysis and limitations
8. **Conclusion** - Summary and future work
9. **References** - 35+ citations

## Key Results (Actual Experimental)

### Stress-Biased Datasets (200 samples, 4:1 imbalance)
| Dataset | Test F1 | Baseline |
|---------|---------|----------|
| Water Stress | 0.129 | 0.504 |
| Nutrient Def | 0.097 | 0.504 |
| Pest Risk | 0.129 | 0.504 |
| Disease Risk | 0.484 | 0.504 |
| Heat Stress | 0.194 | 0.504 |
| **COMBINED** | **1.000** | 0.200 |

### Model Performance (General Training Data)
| Model Type | Best F1 |
|------------|---------|
| LLM (DistilBERT) | 0.271 |
| ViT (Swin-tiny) | 0.251 |
| VLM (Attention/CoCa/Unified-IO) | **1.00** |

## Architecture Diagrams Included

1. Complete System Architecture
2. LLM Encoder Architecture
3. ViT Encoder Architecture
4. 8 VLM Fusion Architectures
5. Federated Learning Pipeline
6. Class Imbalance Handling
7. Qdrant Semantic Retrieval

## Citation

```bibtex
@article{farmfederate2024,
  title={FarmFederate: A Comprehensive Multimodal Framework for Federated Crop Stress Detection},
  author={Research Team},
  journal={arXiv preprint},
  year={2024}
}
```
