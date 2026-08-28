# FarmFederate Tea Leaf Disease Dataset

**Version 1.0** — April 2026

Hybrid multimodal dataset for tea crop stress detection, used in the FarmFederate
federated learning study.

## Overview

| Attribute | Value |
|-----------|-------|
| Classes | 5 tea stress categories |
| Training images | 2,400 (480/class) |
| Validation images | 600 (120/class) |
| Text samples | 2,000 (400/class) |
| Field-collected images | 200 (669 labeled instances) |
| Total unique images | 5,067+ |

## Classes

| ID | Name | Description |
|----|------|-------------|
| 0 | gray_blight | Pestalotiopsis-caused gray necrotic lesions |
| 1 | helopeltis | Tea mosquito bug puncture damage |
| 2 | algal_leaf_spot | Cephaleuros virescens orange algal spots |
| 3 | brown_blight | Colletotrichum brown water-soaked lesions |
| 4 | red_leaf_spot | Phyllosticta reddish circular lesions |

## Folder Structure

```
farmfederate_dataset/
├── images/
│   ├── train/{class}/          # 480 images per class
│   └── val/{class}/            # 120 images per class
├── labels/
│   ├── master_labels.csv         # All 3,000 image labels
│   ├── train.csv                 # (see splits/)
│   ├── {class}_labels.csv      # Per-class label files
│   └── field_labels.csv          # Field-collected multi-label annotations
├── splits/
│   ├── train.csv                 # 2,400 training image paths
│   └── val.csv                   # 600 validation image paths
├── text/
│   └── text_corpus.csv           # 2,000 synthetic text samples
├── field_collected/{class}/    # Original 200 field images (669 instances)
├── metadata/                     # (reserved for future use)
└── dataset_info.json             # Full statistics

```

## Data Sources

### Training Images (3,000 total)
- **Kaggle Tea Leaf Disease Dataset** (Mamun et al., 2023): Primary source.
  4,867 images across 5 classes. Balanced to 600/class for training.
- **Field-collected** (200 images, 669 labeled instances): Tea garden surveys
  in Northeast India. Multi-label annotations; small subset included per class.

### Text Corpus (2,000 samples)
Synthetic agricultural text generated from domain templates:
- 40% class-specific symptom descriptions
- 60% cross-class general agricultural text
- 80/20 train/val split per class

## CSV Columns

**master_labels.csv / splits/train.csv / splits/val.csv:**
```
filename, rel_path, class_name, class_id, source, split
```

**text/text_corpus.csv:**
```
text, class_id, class_name, split, template_type
```

**labels/field_labels.csv:**
```
filename, rel_path, class_name, class_id, source
```

## Usage

```python
import pandas as pd
from PIL import Image

df = pd.read_csv("farmfederate_dataset/labels/master_labels.csv")
train_df = df[df["split"] == "train"]

# Load an image
img_path = "farmfederate_dataset/" + train_df.iloc[0]["rel_path"]
img = Image.open(img_path)
```

## Citation

If you use this dataset, please cite:
```
FarmFederate: Multimodal Federated Learning for Tea Crop Stress Detection (2026)
Mamun et al. Tea Leaf Disease Dataset. Kaggle (2023).
```

## Statistics

- **Source breakdown**: 0 field-collected + 3000 Kaggle (in 3,000-image training set)
- **Class balance**: Perfectly balanced at 600/class (train+val)
- **Original field imbalance**: 26:1 (helopeltis under-represented)
- **Text templates**: 18 cross-class + 25 class-specific
