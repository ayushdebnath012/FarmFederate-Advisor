# Federated Learning Complete Training System

## Overview

Complete implementation of Federated Learning system for plant stress detection with:

- **Federated LLM**: Text-based plant stress detection using T5, GPT-2
- **Federated ViT**: Image-based detection using Vision Transformers
- **Federated VLM**: Multimodal detection using CLIP (text + images)
- **15+ Comprehensive Plots**: Detailed comparison and analysis
- **Baseline Comparisons**: Comparison with FedAvg, FedProx, MOON, PlantVillage, etc.
- **Checkpoint System**: Auto-save and auto-resume on crashes
- **Result Tracking**: Results saved after each model

## Quick Start

### Option 1: Run Everything (Recommended)

```bash
train_federated_all.bat
```

This will:
1. Train all models (LLM, ViT, VLM)
2. Generate 15+ comparison plots
3. Save all results and checkpoints

### Option 2: Train Models Only

```bash
cd backend
python federated_complete_training.py
```

### Option 3: Generate Plots Only (after training)

```bash
cd backend
python comprehensive_plotting.py
```

## What Gets Trained

### 1. Federated LLM Models (Text-based)
- **Flan-T5-Small**: Small efficient model
- **T5-Small**: Standard T5 baseline
- **DistilGPT2**: Distilled GPT-2

### 2. Federated ViT Models (Image-based)
- **ViT-Base**: Standard Vision Transformer
- **ViT-Small**: Smaller efficient variant

### 3. Federated VLM Models (Multimodal)
- **CLIP-ViT-Base**: Vision-Language model combining text and images

## Features

### ✅ Automatic Checkpoint Saving
- Saves checkpoint after each federated round
- Auto-resumes from latest checkpoint if training crashes
- No need to restart from scratch

### ✅ Result Tracking
- Saves metrics after each model completes
- JSON format for easy analysis
- Pickle format for Python loading

### ✅ 15+ Comprehensive Plots

**Performance Comparison:**
1. Overall Performance (F1, Accuracy, Precision, Recall)
2. Model Type Comparison (LLM vs ViT vs VLM)
3. Training Convergence Curves
4. Baseline Paper Comparison

**Analysis Plots:**
5. Precision-Recall Scatter
6. Metrics Heatmap
7. Federated Rounds Impact
8. Best vs Worst Model
9. Improvement Over Rounds
10. Statistical Distribution
11. Multi-Metric Radar Chart
12. Convergence Rate Analysis
13. Performance Ranking
14. Performance Evolution Over Years
15. Loss Landscape

### ✅ Baseline Paper Comparisons

Compares against state-of-the-art papers:
- FedAvg (2017)
- FedProx (2020)
- MOON (2021)
- FedBN (2021)
- PlantVillage (2016)
- DeepPlant (2019)
- AgriVision-ViT (2023)
- FedCrop (2023)
- FedAgri-BERT (2023)

## Directory Structure

```
FarmFederate-Advisor/
├── backend/
│   ├── federated_complete_training.py  # Main training script
│   ├── comprehensive_plotting.py        # Plotting system
│   ├── datasets_loader.py               # Data loading
│   └── federated_core.py                # Core federated logic
├── checkpoints/                         # Model checkpoints
│   ├── flan-t5-small_latest.pt
│   ├── vit-base_latest.pt
│   └── ...
├── results/                             # Training results
│   ├── all_results.json
│   ├── flan-t5-small_results.json
│   └── ...
├── plots/                               # Generated plots
│   ├── plot_01_overall_performance.png
│   ├── plot_02_model_type_comparison.png
│   └── ... (15+ plots)
└── train_federated_all.bat             # Start script
```

## Training Configuration

Each model trains with:
- **Federated Rounds**: 10 (configurable)
- **Number of Clients**: 5 (simulated)
- **Local Epochs per Round**: 3
- **Batch Size**: 12-16 (depending on model)
- **Learning Rate**: 1e-5 to 3e-5 (adaptive)

## Resuming Training

If training crashes or is interrupted:

```bash
cd backend
python federated_complete_training.py
```

The system will automatically:
1. Detect existing checkpoints
2. Load the latest checkpoint
3. Resume from the last saved round
4. Continue training seamlessly

## Viewing Results

### Results Files
- `results/all_results.json` - All model results
- `results/<model_name>_results.json` - Individual model results

### Plot Files
All plots are saved in `plots/` as high-resolution PNG images (300 DPI)

### Checkpoints
- `checkpoints/<model_name>_latest.pt` - Latest checkpoint (auto-resume)
- `checkpoints/<model_name>_round_<N>.pt` - Per-round checkpoints
- `checkpoints/<model_name>_final.pt` - Final trained model

## Requirements

```bash
pip install torch torchvision transformers datasets
pip install numpy pandas matplotlib seaborn scikit-learn
pip install Pillow scipy
```

## Datasets

### Text Data
- CGIAR/gardian-ai-ready-docs (agricultural documents)
- argilla/farming (farming conversations)
- Synthetic agricultural data

### Image Data
- PlantVillage Dataset (plant diseases)
- PlantDoc Dataset (plant documentation)
- Other agricultural image datasets

Datasets are automatically downloaded from Hugging Face on first run.

## Performance Metrics

Each model is evaluated on:
- **F1 Score** (Macro and Micro)
- **Accuracy**
- **Precision**
- **Recall**
- **AUC-ROC** (when applicable)

## Hardware Requirements

**Minimum:**
- CPU: 4+ cores
- RAM: 16 GB
- Storage: 10 GB

**Recommended:**
- GPU: NVIDIA GPU with 8GB+ VRAM (CUDA-enabled)
- RAM: 32 GB
- Storage: 20 GB

Training automatically uses GPU if available, otherwise falls back to CPU.

## Training Time Estimates

On GPU (RTX 3080):
- Flan-T5-Small: ~30 minutes
- ViT-Base: ~40 minutes
- CLIP-ViT-Base: ~50 minutes

On CPU:
- 3-5x longer than GPU times

## Troubleshooting

### Out of Memory
Reduce batch size in model configs:
```python
batch_size=8  # Instead of 16
```

### Dataset Download Issues
Check internet connection. Datasets are downloaded from Hugging Face.

### Checkpoint Loading Fails
Delete corrupted checkpoint:
```bash
del checkpoints\<model_name>_latest.pt
```

## Citation

If you use this system in your research, please cite:

```bibtex
@article{farmfederate2026,
  title={Federated Learning for Agricultural Intelligence: A Comprehensive Framework},
  author={FarmFederate Team},
  journal={Agricultural AI Systems},
  year={2026}
}
```

## License

MIT License - See LICENSE file for details

## Contact

For questions or issues, please open an issue on GitHub.

---

**Happy Training! 🌱🤖**
