"""Importable versions of the lightweight demo pieces from RAG_Colab_Demo.ipynb.

The notebook defines these inline for Colab users; CI and unit tests need the same
definitions without executing a notebook, so they are mirrored here (Sections 4, 5,
11 and 13 of the notebook) with the demo print/side-effect cells left out.
"""
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset

# Section 4 - Configuration & Constants
ISSUE_LABELS = ['water_stress', 'nutrient_def', 'pest_risk', 'disease_risk', 'heat_stress']
NUM_LABELS = len(ISSUE_LABELS)


# Section 5 - Synthetic Text & Image Generators
def generate_text_data(n_samples: int = 100, label: Optional[str] = None) -> pd.DataFrame:
    texts = []
    labels = []
    for i in range(n_samples):
        lbl = label if label is not None else np.random.choice(ISSUE_LABELS)
        texts.append(f"Sample {i} for {lbl}: leaf discoloration and small spots")
        labels.append([ISSUE_LABELS.index(lbl)])
    return pd.DataFrame({'text': texts, 'labels': labels})


def generate_image_data(n_samples: int = 50, img_size: int = 224) -> List[Image.Image]:
    imgs = []
    for _ in range(n_samples):
        arr = (np.random.rand(img_size, img_size, 3) * 255).astype('uint8')
        imgs.append(Image.fromarray(arr))
    return imgs


# Section 11 - MultiModalDataset
class MultiModalDataset(Dataset):
    def __init__(self, texts, text_labels, images=None, image_labels=None):
        self.texts = texts
        self.tlabels = text_labels
        self.images = images or []
        self.ilabels = image_labels or []

    def __len__(self):
        return max(len(self.texts), len(self.images))

    def __getitem__(self, idx):
        t = self.texts[idx % len(self.texts)] if self.texts else ""
        tl = self.tlabels[idx % len(self.tlabels)] if self.tlabels else [0]
        if self.images:
            img = self.images[idx % len(self.images)]
            # Return PIL Image; downstream transforms can handle it
        else:
            img = Image.new('RGB', (224, 224))
        labels = torch.tensor([1 if i in tl else 0 for i in range(NUM_LABELS)]).float()
        return {'text': t, 'labels': labels, 'image': img}


# Section 13 - Minimal model architectures (LLM-like & ViT-like) for smoke tests
class SimpleLLM(nn.Module):
    def __init__(self, vocab_size: int = 10000, embed_dim: int = 128, num_labels: int = NUM_LABELS):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(embed_dim, num_labels)

    def forward(self, input_ids):
        x = self.emb(input_ids)
        x = x.mean(dim=1)
        return self.classifier(x)


class SimpleViT(nn.Module):
    def __init__(self, in_ch: int = 3, hidden: int = 128, num_labels: int = NUM_LABELS):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, hidden, 7, stride=4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(self, pixel_values):
        x = self.conv(pixel_values)
        x = self.pool(x).view(x.size(0), -1)
        return self.classifier(x)
