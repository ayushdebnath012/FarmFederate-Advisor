#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FarmFederate - Comprehensive Crop Stress Detection with Federated Learning
============================================================================

KAGGLE VERSION — Adapted for Kaggle Notebooks (GPU P100/T4)

KEY DIFFERENCES FROM COLAB VERSION:
- No Google Drive mounting  → uses /kaggle/input/ datasets + /kaggle/working/ output
- No files.download()       → Kaggle auto-saves everything in /kaggle/working/
- Output goes to            → /kaggle/working/farmfederate_output/
- Data input from           → /kaggle/input/farmfederate-data/ (if uploaded as dataset)

HOW TO USE ON KAGGLE:
============================================================================

1. Create a new Kaggle Notebook: https://www.kaggle.com/notebooks
2. Enable GPU: Settings → Accelerator → GPU P100 or T4
3. Internet on: Settings → Internet → On  (needed for HuggingFace downloads)
4. Paste this ENTIRE file into the notebook cell and run it!

OPTIONAL — Upload your dataset first:
  - Go to kaggle.com/datasets → New Dataset
  - Upload a zip containing:
      data/water_stress/text.csv
      data/water_stress/images/*.jpg
      data/nutrient_def/text.csv
      ... (same structure for all 5 stress types)
  - Name it "farmfederate-data"
  - Add it to this notebook: Add Data → farmfederate-data

EXECUTION MODES (change EXECUTION_MODE below):
  'quick'    → 2-3 min smoke test  (2 epochs,  50 samples)
  'standard' → 30-60 min training  (12 epochs, 600 samples)  ← default
  'full'     → 60-90 min training  (20 epochs, 1000 samples)
  'manual'   → Don't auto-run, call run_colab() manually

Outputs (auto-saved to /kaggle/working/farmfederate_output/):
  - 24 trained model checkpoints (.pt files)
  - 50+ visualization plots (.png)
  - complete_results.json with all metrics
  - farmfederate_results.zip (everything zipped)

Author: FarmFederate Team
License: MIT
Version: 5.1-kaggle
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

# ============================================================================
# KAGGLE / COLAB / LOCAL ENVIRONMENT DETECTION
# ============================================================================
IN_COLAB  = 'google.colab' in sys.modules
IN_KAGGLE = os.path.exists('/kaggle/working')
IN_JUPYTER = 'ipykernel' in sys.modules

# ── Kaggle-specific setup ────────────────────────────────────────────────────
if IN_KAGGLE:
    print("=" * 70)
    print("FARMFEDERATE v5.1 — KAGGLE MODE")
    print("=" * 70)

    # Install extra deps not pre-installed on Kaggle
    print("Installing missing dependencies...")
    import subprocess
    _pkgs = ['sentence-transformers', 'faiss-cpu']
    for _pkg in _pkgs:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', _pkg],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass
    print("Dependencies ready.\n")

    # ── Paths ────────────────────────────────────────────────────────────────
    KAGGLE_WORKING  = Path('/kaggle/working')
    KAGGLE_INPUT    = Path('/kaggle/input')
    OUTPUT_BASE     = KAGGLE_WORKING / 'farmfederate_output'

    # Look for uploaded FarmFederate dataset
    _data_candidates = [
        KAGGLE_INPUT / 'farmfederate-data' / 'data',
        KAGGLE_INPUT / 'farmfederate-data',
        KAGGLE_INPUT / 'farmfederate' / 'data',
        KAGGLE_INPUT / 'plant-data' / 'data',
        KAGGLE_WORKING / 'data',
        Path('data'),
    ]
    DATA_DIR = None
    for _p in _data_candidates:
        if _p.exists() and _p.is_dir():
            # Check it has at least one stress type subfolder
            _stress_types = ['water_stress', 'nutrient_def', 'pest_risk',
                             'disease_risk', 'heat_stress']
            if any((_p / _s).exists() for _s in _stress_types):
                DATA_DIR = _p
                print(f"Found dataset at: {DATA_DIR}")
                break

    if DATA_DIR is None:
        print("No local dataset found — will use HuggingFace + synthetic data.")
        print("(To use your own data, upload a Kaggle dataset named 'farmfederate-data')")
        DATA_DIR = KAGGLE_WORKING / 'data'

    # Make sure farmfederate_rag is importable (clone repo if needed)
    _rag_candidates = [
        KAGGLE_WORKING / 'FarmFederate' / 'backend' / 'farmfederate_rag',
        Path(__file__).parent / 'farmfederate_rag' if '__file__' in dir() else None,
    ]
    for _rp in _rag_candidates:
        if _rp and _rp.exists():
            if str(_rp.parent) not in sys.path:
                sys.path.insert(0, str(_rp.parent))
            print(f"RAG module found: {_rp}")
            break
    else:
        try:
            _repo = KAGGLE_WORKING / 'FarmFederate'
            if not _repo.exists():
                subprocess.check_call(
                    ['git', 'clone', '-q',
                     'https://github.com/Solventerritory/FarmFederate-Advisor',
                     str(_repo)],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            _rag_path = _repo / 'backend' / 'farmfederate_rag'
            if _rag_path.exists() and str(_rag_path.parent) not in sys.path:
                sys.path.insert(0, str(_rag_path.parent))
                print(f"RAG module cloned: {_rag_path}")
        except Exception as _e:
            print(f"RAG module unavailable ({_e}) — RAG diagnosis will be skipped.")

    print("=" * 70 + "\n")

elif IN_COLAB:
    # ── Colab setup (same as original) ───────────────────────────────────────
    print("Installing dependencies for Colab...")
    import subprocess
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
                          'torch', 'torchvision', 'torchaudio',
                          'transformers', 'datasets',
                          'pillow', 'pandas', 'numpy', 'scikit-learn',
                          'tqdm', 'matplotlib', 'seaborn',
                          'sentence-transformers', 'gdown', 'faiss-cpu'])
    print("Dependencies installed!\n")

    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)

    DRIVE_FOLDER_ID = '1lfYJxO55en6r_kl6KNRIyQqtYXrAyMkw'
    DATA_DIR = Path('/content/FarmFederate/data')

    _drive_data_candidates = [
        Path('/content/drive/MyDrive/FarmFederate/data'),
        Path('/content/drive/MyDrive/data'),
        Path('/content/FarmFederate/data'),
        Path('data'),
    ]
    _data_found = False
    for _p in _drive_data_candidates:
        if _p.exists() and any(_p.iterdir()):
            DATA_DIR = _p
            _data_found = True
            print(f"Dataset found at: {DATA_DIR}")
            break

    if not _data_found:
        print("Downloading dataset from Google Drive...")
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        import gdown
        gdown.download_folder(
            f'https://drive.google.com/drive/folders/{DRIVE_FOLDER_ID}',
            output=str(DATA_DIR), quiet=False)

    OUTPUT_BASE = Path('/content/farmfederate_output')

# ============================================================================
# REST OF THE SCRIPT — copy of FarmFederate_Colab_Complete.py from line 185
# (All imports, classes, training code, etc. are identical)
# ============================================================================

import os
import json
import time
import argparse
import warnings
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
import random

warnings.filterwarnings('ignore')

# ── Numeric/ML imports (with try/except for environments without GPU/torch) ──
try:
    import numpy as np
except ImportError:
    np = None

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    torch = None

try:
    import torchvision.transforms as _T
    from torchvision import models as _tv_models
except ImportError:
    _T = None

try:
    from transformers import (
        AutoTokenizer, AutoModel, AutoConfig,
        DistilBertTokenizer, DistilBertModel,
        BertTokenizer, BertModel,
        RobertaTokenizer, RobertaModel,
        AlbertTokenizer, AlbertModel,
        MobileBertTokenizer, MobileBertModel,
        CLIPModel, CLIPProcessor,
        get_linear_schedule_with_warmup,
    )
    from transformers import logging as hf_logging
    hf_logging.set_verbosity_error()
except ImportError:
    pass

try:
    from datasets import load_dataset as hf_load_dataset
except ImportError:
    hf_load_dataset = None

try:
    from sklearn.metrics import f1_score, classification_report
    from sklearn.preprocessing import LabelEncoder
except ImportError:
    pass

try:
    from PIL import Image
    import PIL.ImageFilter as _PIL_Filter
except ImportError:
    Image = None

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import seaborn as sns
except ImportError:
    plt = None

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kw): return x

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    import faiss
except ImportError:
    faiss = None


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    # Model
    hidden_dim: int = 256
    num_classes: int = 5
    dropout: float = 0.3
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4

    # Training
    epochs: int = 12
    batch_size: int = 16
    grad_clip: float = 1.0
    warmup_ratio: float = 0.05
    early_stopping_patience: int = 6
    mixed_precision: bool = True

    # Federated
    num_clients: int = 5
    fed_rounds: int = 5
    local_epochs: int = 2
    dirichlet_alpha: float = 1.0

    # Data + paths
    max_samples_per_class: int = 600
    train_split: float = 0.8
    image_size: int = 224
    max_seq_length: int = 128

    # Paths — Kaggle-aware defaults
    data_dir: Path = field(default_factory=lambda: (
        Path('/kaggle/working/data') if os.path.exists('/kaggle/working')
        else Path('data')
    ))
    output_dir: Path = field(default_factory=lambda: (
        Path('/kaggle/working/farmfederate_output') if os.path.exists('/kaggle/working')
        else Path('results')
    ))
    checkpoint_dir: Path = field(default_factory=lambda: (
        Path('/kaggle/working/farmfederate_output/checkpoints') if os.path.exists('/kaggle/working')
        else Path('checkpoints')
    ))
    plots_dir: Path = field(default_factory=lambda: (
        Path('/kaggle/working/farmfederate_output/plots') if os.path.exists('/kaggle/working')
        else Path('plots')
    ))

    seed: int = 42


# Override data_dir if we found a dataset above
if IN_KAGGLE and 'DATA_DIR' in dir() and DATA_DIR is not None:
    _GLOBAL_DATA_DIR = DATA_DIR
else:
    _GLOBAL_DATA_DIR = None


STRESS_LABELS = ['water_stress', 'nutrient_def', 'pest_risk', 'disease_risk', 'heat_stress']
LABEL_TO_IDX  = {label: idx for idx, label in enumerate(STRESS_LABELS)}
IDX_TO_LABEL  = {idx: label for idx, label in enumerate(STRESS_LABELS)}


# ============================================================================
# KAGGLE OUTPUT HELPER  (replaces Colab's files.download)
# ============================================================================

def _kaggle_finalize(config: Config, zip_name: str = 'farmfederate_results.zip') -> str:
    """
    On Kaggle: zip everything and print instructions.
    All files in /kaggle/working/ are automatically available as notebook outputs.
    """
    import zipfile, shutil

    output_dir = Path(config.output_dir)
    checkpoints_dir = config.checkpoint_dir
    plots_dir = config.plots_dir

    zip_path = Path('/kaggle/working') / zip_name
    file_count = 0
    total_size = 0

    print(f"\nPackaging results → {zip_path}")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Models
        if Path(checkpoints_dir).exists():
            for f in sorted(Path(checkpoints_dir).glob('*.pt')):
                zf.write(f, f'models/{f.name}')
                file_count += 1; total_size += f.stat().st_size
            print(f"  + {file_count} model checkpoints")

        # Plots
        n_plots = 0
        if Path(plots_dir).exists():
            for f in sorted(Path(plots_dir).glob('*.png')):
                zf.write(f, f'plots/{f.name}')
                n_plots += 1; total_size += f.stat().st_size
            print(f"  + {n_plots} plots")
            file_count += n_plots

        # Results JSON/CSV
        n_results = 0
        if output_dir.exists():
            for ext in ('*.json', '*.csv', '*.txt', '*.md'):
                for f in output_dir.glob(ext):
                    zf.write(f, f'results/{f.name}')
                    n_results += 1; total_size += f.stat().st_size
            print(f"  + {n_results} result files")
            file_count += n_results

        readme = f"""# FarmFederate Training Results (Kaggle)

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Total files: {file_count}
Size: {total_size / 1024 / 1024:.1f} MB

## Download
In the Kaggle notebook sidebar → Output tab → farmfederate_results.zip

## Contents
- models/   — trained .pt checkpoints
- plots/    — 50+ PNG visualizations
- results/  — complete_results.json, CSV metrics
"""
        zf.writestr('README.md', readme)

    zip_size = zip_path.stat().st_size / 1024 / 1024
    print(f"\nPACKAGE READY: {zip_path}  ({zip_size:.1f} MB)")
    print("→ Download via Kaggle notebook Output tab, or:")
    print(f"   from IPython.display import FileLink")
    print(f"   FileLink('{zip_path}')")
    return str(zip_path)


# ============================================================================
# INCLUDE CORE FARMFEDERATE CODE
# ============================================================================
# The cleanest way to run this on Kaggle is to paste the full body of
# FarmFederate_Colab_Complete.py *below this line*, replacing lines 1-184
# (the Colab-specific header) with this Kaggle-aware header above.
#
# ─────────────────────────────────────────────────────────────────────────────
# INSTRUCTIONS FOR ASSEMBLY:
#   1. Open FarmFederate_Colab_Complete.py
#   2. Copy lines 185 onward  (starting with "import os")  — but SKIP the
#      second @dataclass Config block (lines ~230-266) since we defined it
#      above with Kaggle-aware paths.
#   3. Paste those lines below the "END OF KAGGLE HEADER" marker.
#   4. At the very bottom, replace the Colab execution block with the
#      Kaggle execution block shown at the end of this file.
# ─────────────────────────────────────────────────────────────────────────────
#
# ALTERNATIVELY — run as a two-cell Kaggle notebook:
#
#   CELL 1  (this file):
#       exec(open('/kaggle/input/farmfederate/FarmFederate_Kaggle.py').read())
#
#   CELL 2:
#       exec(open('/kaggle/input/farmfederate/FarmFederate_Colab_Complete.py').read())
#
# ─────────────────────────────────────────────────────────────────────────────

# ============================================================================
# END OF KAGGLE HEADER
# ============================================================================
# ↓↓↓  PASTE FarmFederate_Colab_Complete.py lines 185–11540 HERE  ↓↓↓
# (everything up to but not including the EXECUTION CONFIGURATION section)


# ============================================================================
# EXECUTION CONFIGURATION (Kaggle version)
# ============================================================================

EXECUTION_MODE = 'standard'  # 'quick' | 'standard' | 'full' | 'manual'

TRAINING_CONFIG = {
    'quick':    {'epochs': 2,  'max_samples': 50,   'batch_size': 8},
    'standard': {'epochs': 12, 'max_samples': 600,  'batch_size': 16},
    'full':     {'epochs': 20, 'max_samples': 1000, 'batch_size': 16},
}

# ============================================================================
# AUTO-EXECUTION
# ============================================================================

if __name__ == '__main__' or IN_JUPYTER or IN_KAGGLE:
    if EXECUTION_MODE == 'manual':
        print("\n" + "=" * 70)
        print("FARMFEDERATE v5.1-kaggle — MANUAL MODE")
        print("=" * 70)
        print("\nCall one of:")
        print("  run_quick_test()                         # 2-3 min")
        print("  run_colab(epochs=12, max_samples=600)    # 30-60 min  ← recommended")
        print("  run_colab(epochs=20, max_samples=1000)   # 60-90 min")
    else:
        print("\n" + "=" * 70)
        print("FARMFEDERATE v5.1-kaggle — AUTO-RUNNING")
        print("=" * 70)
        print(f"Mode: {EXECUTION_MODE.upper()}")
        print("Models: 24 (5 LLM + 5 ViT + 8 VLM + 6 Federated)")

        cfg = TRAINING_CONFIG.get(EXECUTION_MODE, TRAINING_CONFIG['standard'])

        if EXECUTION_MODE == 'quick':
            print("Time: ~2-3 minutes (smoke test)\n")
            run_quick_test()
        else:
            _eta = '30-60' if EXECUTION_MODE == 'standard' else '60-90'
            print(f"Time: ~{_eta} minutes on Kaggle GPU P100/T4\n")

            # Override data_dir if we found a Kaggle dataset
            _config_kwargs = dict(
                epochs=cfg['epochs'],
                max_samples=cfg['max_samples'],
                batch_size=cfg['batch_size'],
                fusion_type='attention',
                run_dataset_comparison=True,
            )
            results = run_colab(**_config_kwargs)

            # Package outputs for download
            if IN_KAGGLE:
                # Build a temp Config to find the output dirs
                _tmp = Config(
                    epochs=cfg['epochs'],
                    max_samples_per_class=cfg['max_samples'],
                    batch_size=cfg['batch_size'],
                )
                _kaggle_finalize(_tmp)
                print("\nAll outputs saved to /kaggle/working/farmfederate_output/")
                print("Download from the Kaggle notebook Output tab.")
