# FarmFederate Training Results

## Package Contents

This package contains all artifacts from FarmFederate training:

### 📁 models/ (24 files)
- All trained model checkpoints (.pt files)
- Includes: LLM, ViT, VLM, Centralized, and Federated models
- Load with: `torch.load('models/model_name.pt')`

### 📁 plots/ (53 files)
- Training visualizations and comparisons
- Dataset comparison plots (benchmark model on different datasets)
- Intra-model comparison plots (hyperparameter tuning)
- Inter-model comparison plots (LLM vs ViT vs VLM)
- Research paper comparisons (45+ SOTA papers)
- High-resolution PNG files (300 DPI)

### 📁 results/ (2 files)
- complete_results.json - Full training metrics
- dataset_comparison_results.json - Dataset benchmark results
- CSV files for detailed metrics

## Quick Start

1. Extract the zip file
2. Load a model:
```python
import torch
checkpoint = torch.load('models/vlm_attention_best.pt')
model_state = checkpoint['model_state_dict']
f1_score = checkpoint['f1_score']
```

3. View plots in the plots/ directory
4. Check results JSON files for detailed metrics

## Models Trained
- 5 LLM models (DistilBERT, BERT-tiny, RoBERTa-tiny, ALBERT-tiny, MobileBERT)
- 5 ViT models (ViT-Base, DeiT-tiny, Swin-tiny, ConvNeXT-tiny, EfficientNet)
- 8 VLM fusion architectures (concat, attention, gated, CLIP, Flamingo, BLIP2, CoCa, Unified-IO)
- Centralized and Federated variants

Generated: 2026-02-17 15:51:21
Total Files: 79
Package Size: 1047.7 MB
