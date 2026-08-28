#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Image vs text vs fusion on ONE identical 3-class split - COLAB / KAGGLE.

Paste this whole file into one cell and run it.

  Colab   Runtime  -> Change runtime type -> T4 GPU
  Kaggle  Settings -> Accelerator: GPU, Internet: ON

Question
--------
The cross-domain run showed the image branch reaching macro-F1 0.907 on three
classes, where the 5-class pipeline reported only 0.503. If image has really
caught up with text, the gate collapse that flattened the proposed multimodal
model onto text-only had a cause that no longer applies.

This script settles it by scoring both modalities, and their fusion, on the
SAME held-out rows:

  image_only   vision encoder on the test images
  text_only    text encoder on the paired test observations
  fusion       p = w*p_text + (1-w)*p_image, with w chosen on VALIDATION only

Honest limits, stated up front
------------------------------
1. Text and images are NOT sample-linked. A pair is label-aligned only: the
   observation was not written about that photograph. This mirrors the main
   pipeline's proxy-pair protocol and is reported as such.
2. The text corpus has 11 independent first-sentence template families per
   class. Row-level text scores therefore rest on far fewer independent units
   than the row count suggests, so a group-level text score is reported too.
3. n is small. Wilson intervals and exact McNemar tests are printed; read the
   intervals, not the point estimates alone.

Data
----
Field images and annotations.csv both come from data_final.zip - Google Drive
on Colab, an attached Dataset on Kaggle (expanded folder or zip, found by
recursive search of /kaggle/input).
"""
from __future__ import annotations

import csv
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

# ===========================================================================
# CONFIGURATION
# ===========================================================================
VISION_BACKBONE = 'microsoft/swin-tiny-patch4-window7-224'
TEXT_BACKBONE = 'distilbert-base-uncased'
EPOCHS_VISION = 15
EPOCHS_TEXT = 12
BATCH_SIZE = 16
VISION_LR, VISION_HEAD_LR = 2e-5, 1e-4
TEXT_LR, TEXT_HEAD_LR = 2e-5, 1.6e-4
MAX_SEQ_LENGTH = 128
SEED = 42

DATA_FINAL_DRIVE_ID = '1PBZJrCGQ8-sxZnKzxMPhA9ySzOZWmMzf'
CLASSES = ['LEAF_BLIGHT', 'LEAF_RUST', 'MOSQUITO_BUG']
CLASS_TO_IDX = {name: index for index, name in enumerate(CLASSES)}
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def _detect_runtime():
    in_kaggle = bool(os.environ.get('KAGGLE_KERNEL_RUN_TYPE')) or (
        Path('/kaggle/input').is_dir() and Path('/kaggle/working').is_dir())
    in_colab = False if in_kaggle else (
        'google.colab' in sys.modules or Path('/content').exists())
    return in_colab, in_kaggle


IN_COLAB, IN_KAGGLE = _detect_runtime()
KAGGLE_INPUT = Path(os.environ.get('KAGGLE_INPUT_DIR', '/kaggle/input'))
KAGGLE_WORKING = Path(os.environ.get('KAGGLE_WORKING_DIR', '/kaggle/working'))

if IN_KAGGLE:
    # Point download caches at the writable area before transformers loads.
    _cache = KAGGLE_WORKING / '.cache'
    os.environ.setdefault('HF_HOME', str(_cache / 'huggingface'))
    os.environ.setdefault('TORCH_HOME', str(_cache / 'torch'))
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')


def work_root() -> Path:
    if IN_KAGGLE:
        return KAGGLE_WORKING
    if IN_COLAB:
        return Path('/content')
    return Path('.')


def log(message):
    print(message, flush=True)


def section(title):
    log('\n' + '=' * 78)
    log(title)
    log('=' * 78)


# ===========================================================================
# 1. ENVIRONMENT + DATA RESOLUTION
# ===========================================================================
def setup_environment():
    section('[1/6] ENVIRONMENT')
    runtime = 'Kaggle' if IN_KAGGLE else ('Colab' if IN_COLAB else 'local')
    log(f'  runtime: {runtime}   work_root: {work_root()}')
    for module, package in (('transformers', 'transformers'),
                            ('PIL', 'pillow'), ('numpy', 'numpy')):
        try:
            __import__(module)
        except ImportError:
            log(f'  installing {package} ...')
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', package])
    import torch
    log(f'  torch {torch.__version__}  cuda={torch.cuda.is_available()}')
    if torch.cuda.is_available():
        log(f'  GPU: {torch.cuda.get_device_name(0)}')
    else:
        log('  [WARN] No GPU. This will be very slow.')
    return torch


def mount_drive():
    if IN_KAGGLE or not IN_COLAB:
        return None
    try:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=False)
        for candidate in (Path('/content/drive/MyDrive'), Path('/content/drive/My Drive')):
            if candidate.is_dir():
                log(f'  Drive mounted: {candidate}')
                return candidate
    except Exception as error:
        log(f'  [WARN] Drive mount failed: {error}')
    return None


def _find_bundle_dir(root: Path):
    """Return the data_final root holding both real_dataset_sorted and text_data."""
    if not root.exists():
        return None
    for candidate in root.rglob('real_dataset_sorted'):
        if not candidate.is_dir():
            continue
        if not any((candidate / c).is_dir() for c in CLASSES):
            continue
        parent = candidate.parent
        if (parent / 'text_data' / 'annotations.csv').is_file():
            return parent
    return None


def resolve_bundle(drive_root):
    """Locate data_final (images + annotations.csv), extracting only if needed."""
    section('[2/6] DATA (images + annotations.csv)')

    existing = _find_bundle_dir(work_root())
    if existing:
        log(f'  Already available: {existing}')
        return existing

    zip_path = None
    if IN_KAGGLE:
        if KAGGLE_INPUT.is_dir():
            found = _find_bundle_dir(KAGGLE_INPUT)
            if found:
                log(f'  Kaggle input (expanded): {found}')
                return found
            for candidate in sorted(KAGGLE_INPUT.rglob('data_final.zip')):
                zip_path = candidate
                log(f'  Kaggle input (zip): {zip_path}')
                break
        if zip_path is None:
            raise FileNotFoundError(
                'No data_final among the attached Kaggle Datasets. Attach a '
                'dataset containing data_final.zip or its expanded contents '
                '(real_dataset_sorted/ next to text_data/annotations.csv).')
    else:
        candidates = []
        if drive_root is not None:
            candidates += [drive_root / 'FarmFederate' / 'data_final.zip',
                           drive_root / 'data_final.zip']
        candidates += [Path('/content/data_final.zip'), Path('data_final.zip')]
        zip_path = next((c for c in candidates if c.is_file()), None)
        if zip_path is None:
            log('  Not on Drive - downloading the public copy ...')
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'gdown'])
            import gdown
            target = work_root() / 'data_final.zip'
            gdown.download(id=DATA_FINAL_DRIVE_ID, output=str(target),
                           quiet=False, fuzzy=True)
            zip_path = target if target.is_file() else None
        if zip_path is None:
            raise FileNotFoundError('data_final.zip not found.')

    destination = work_root() / 'bundle'
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    log(f'  Extracting {zip_path.name} ...')
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(destination)
    bundle = _find_bundle_dir(destination)
    if bundle is None:
        raise FileNotFoundError(
            f'{zip_path} has no real_dataset_sorted/ beside text_data/annotations.csv.')
    log(f'  Bundle: {bundle}')
    return bundle


# ===========================================================================
# 2. SPLITS
# ===========================================================================
def normalise(text: str) -> str:
    return re.sub(r'\s+', ' ', str(text)).strip().casefold()


def template_key(text: str) -> str:
    """First sentence - the template family a synthetic observation came from."""
    normalised = normalise(text)
    return re.split(r'(?<=[.!?])\s+', normalised, maxsplit=1)[0] or normalised


def image_group(path: Path) -> str:
    return re.sub(r'\.rf\.[0-9a-fA-F]+$', '', path.stem).casefold()


def collect_images(sorted_root: Path):
    records = []
    for class_name in CLASSES:
        class_dir = sorted_root / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f'Missing image folder: {class_dir}')
        for image_path in sorted(class_dir.rglob('*')):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_SUFFIXES:
                records.append({'path': image_path,
                                'label': CLASS_TO_IDX[class_name],
                                'group': image_group(image_path)})
    return records


def collect_texts(csv_path: Path):
    """Deduplicate exact text; a repeated row is not independent evidence."""
    seen, records = {}, []
    with open(csv_path, encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            class_name = row['class_name'].strip()
            if class_name not in CLASS_TO_IDX:
                continue
            key = normalise(row['text'])
            if not key:
                continue
            label = CLASS_TO_IDX[class_name]
            if key in seen:
                if seen[key] != label:
                    raise ValueError('Identical text carries conflicting labels.')
                continue
            seen[key] = label
            records.append({'text': row['text'].strip(), 'label': label,
                            'group': template_key(row['text'])})
    return records


def group_split(records, val_groups_ratio=0.2, test_groups_ratio=0.3, seed=SEED):
    """Assign whole source/template groups to one split each."""
    by_class = defaultdict(lambda: defaultdict(list))
    for record in records:
        by_class[record['label']][record['group']].append(record)

    train, val, test = [], [], []
    for label in sorted(by_class):
        groups = list(by_class[label])
        random.Random(seed + label * 7919).shuffle(groups)
        if len(groups) < 3:
            raise ValueError(f'{CLASSES[label]} has only {len(groups)} groups.')
        n_val = max(1, round(len(groups) * val_groups_ratio))
        n_test = max(1, round(len(groups) * test_groups_ratio))
        while n_val + n_test >= len(groups):
            if n_test > n_val and n_test > 1:
                n_test -= 1
            elif n_val > 1:
                n_val -= 1
            else:
                break
        for group in groups[:n_val]:
            val.extend(by_class[label][group])
        for group in groups[n_val:n_val + n_test]:
            test.extend(by_class[label][group])
        for group in groups[n_val + n_test:]:
            train.extend(by_class[label][group])
    for split in (train, val, test):
        random.Random(seed).shuffle(split)
    return train, val, test


def assert_disjoint(name, train, val, test):
    groups = {'train': {r['group'] for r in train},
              'val': {r['group'] for r in val},
              'test': {r['group'] for r in test}}
    for left, right in (('train', 'val'), ('train', 'test'), ('val', 'test')):
        overlap = groups[left] & groups[right]
        if overlap:
            raise AssertionError(
                f'{name}: {len(overlap)} groups shared between {left} and {right}.')
    log(f'  {name} groups disjoint: '
        + '  '.join(f'{k}={len(v)}' for k, v in groups.items()))


def pair_rows(image_records, text_records, seed=SEED):
    """Attach a same-class observation to each image, round-robin by template.

    Round-robin visits every available template family once before reusing one,
    so the paired text is as diverse as the split allows rather than a run of
    near-identical variants. The pairing is label-aligned ONLY - the text was
    not written about the photograph.
    """
    by_class = defaultdict(lambda: defaultdict(list))
    for record in text_records:
        by_class[record['label']][record['group']].append(record)

    cursor = {}
    for label, groups in by_class.items():
        keys = sorted(groups)
        random.Random(seed + label).shuffle(keys)
        for key in keys:
            random.Random(seed + label).shuffle(groups[key])
        cursor[label] = {'keys': keys, 'index': 0, 'depth': 0}

    paired = []
    for record in image_records:
        label = record['label']
        state = cursor.get(label)
        if not state or not state['keys']:
            raise ValueError(f'No text available for {CLASSES[label]}.')
        key = state['keys'][state['index'] % len(state['keys'])]
        bucket = by_class[label][key]
        text_record = bucket[state['depth'] % len(bucket)]
        state['index'] += 1
        if state['index'] % len(state['keys']) == 0:
            state['depth'] += 1
        paired.append({**record, 'text': text_record['text'],
                       'text_group': text_record['group']})
    return paired


# ===========================================================================
# 3. MODELS
# ===========================================================================
def encoder_hidden_dim(config):
    for attribute in ('hidden_size', 'dim', 'd_model', 'n_embd'):
        if hasattr(config, attribute):
            return int(getattr(config, attribute))
    if getattr(config, 'hidden_sizes', None):
        return int(config.hidden_sizes[-1])
    return 768


def build_vision_model(torch, num_labels, dropout=0.20, label_smoothing=0.05):
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import AutoImageProcessor, AutoModel

    class VisionClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.label_smoothing = label_smoothing
            self.encoder = AutoModel.from_pretrained(VISION_BACKBONE)
            try:
                processor = AutoImageProcessor.from_pretrained(VISION_BACKBONE, use_fast=False)
            except TypeError:
                processor = AutoImageProcessor.from_pretrained(VISION_BACKBONE)
            self.register_buffer('processor_mean', torch.tensor(
                getattr(processor, 'image_mean', [0.5] * 3), dtype=torch.float32).view(1, 3, 1, 1))
            self.register_buffer('processor_std', torch.tensor(
                getattr(processor, 'image_std', [0.5] * 3), dtype=torch.float32).view(1, 3, 1, 1))
            dim = encoder_hidden_dim(self.encoder.config)
            self.classifier = nn.Sequential(
                nn.LayerNorm(dim), nn.Dropout(dropout), nn.Linear(dim, num_labels))
            nn.init.normal_(self.classifier[-1].weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.classifier[-1].bias)

        def forward(self, pixel_values, labels=None, **_):
            imagenet_mean = pixel_values.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            imagenet_std = pixel_values.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            raw = torch.clamp(pixel_values * imagenet_std + imagenet_mean, 0.0, 1.0)
            pixels = (raw - self.processor_mean.to(raw.dtype)) / self.processor_std.to(raw.dtype)
            outputs = self.encoder(pixel_values=pixels, return_dict=True)
            hidden = outputs.last_hidden_state
            model_type = str(getattr(self.encoder.config, 'model_type', '')).lower()
            if hidden.dim() == 3 and model_type in {'vit', 'deit'}:
                features = hidden[:, 0, :]
            else:
                pooled = getattr(outputs, 'pooler_output', None)
                if pooled is not None and pooled.dim() == 2:
                    features = pooled
                elif hidden.dim() == 4:
                    features = hidden.mean(dim=(-2, -1))
                else:
                    features = hidden.mean(dim=1)
            logits = self.classifier(features)
            loss = None
            if labels is not None:
                loss = F.cross_entropy(logits, labels.long(),
                                       label_smoothing=self.label_smoothing)
            return {'loss': loss, 'logits': logits}

    return VisionClassifier()


def build_text_model(torch, num_labels, dropout=0.20, label_smoothing=0.05):
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import AutoModel

    class TextClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.label_smoothing = label_smoothing
            self.encoder = AutoModel.from_pretrained(TEXT_BACKBONE)
            dim = encoder_hidden_dim(self.encoder.config)
            self.classifier = nn.Sequential(
                nn.LayerNorm(dim), nn.Dropout(dropout), nn.Linear(dim, num_labels))
            nn.init.normal_(self.classifier[-1].weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.classifier[-1].bias)

        def forward(self, input_ids, attention_mask=None, labels=None, **_):
            outputs = self.encoder(input_ids=input_ids,
                                   attention_mask=attention_mask, return_dict=True)
            tokens = outputs.last_hidden_state
            # Padding-aware mean pooling: pad positions dominate otherwise.
            if attention_mask is None:
                features = tokens.mean(dim=1)
            else:
                weights = attention_mask.to(tokens.dtype).unsqueeze(-1)
                features = (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
            logits = self.classifier(features)
            loss = None
            if labels is not None:
                loss = F.cross_entropy(logits, labels.long(),
                                       label_smoothing=self.label_smoothing)
            return {'loss': loss, 'logits': logits}

    return TextClassifier()


# ===========================================================================
# 4. LOADERS / TRAIN / EVAL
# ===========================================================================
def augment_image(torch, pixel_values):
    import torch.nn.functional as F
    mean = pixel_values.new_tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = pixel_values.new_tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    raw = torch.clamp(pixel_values * std + mean, 0.0, 1.0)
    if random.random() < 0.5:
        raw = torch.flip(raw, dims=[2])
    if random.random() < 0.5:
        height, width = raw.shape[-2:]
        scale = 0.88 + 0.12 * random.random()
        crop_h, crop_w = max(1, round(height * scale)), max(1, round(width * scale))
        top = random.randint(0, max(0, height - crop_h))
        left = random.randint(0, max(0, width - crop_w))
        raw = F.interpolate(raw[:, top:top + crop_h, left:left + crop_w].unsqueeze(0),
                            size=(height, width), mode='bilinear',
                            align_corners=False).squeeze(0)
    brightness = 0.90 + 0.20 * random.random()
    contrast = 0.90 + 0.20 * random.random()
    channel_mean = raw.mean(dim=(1, 2), keepdim=True)
    raw = (raw - channel_mean) * contrast + channel_mean
    raw = torch.clamp(raw * brightness + torch.randn_like(raw) * 0.015, 0.0, 1.0)
    return (raw - mean) / std


def make_loader(torch, records, modality, tokenizer, batch_size, augment, shuffle):
    import torchvision.transforms as T
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

    to_tensor = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                           T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    class _Dataset(Dataset):
        def __len__(self):
            return len(records)

        def __getitem__(self, index):
            record = records[index]
            item = {'labels': torch.tensor(record['label'], dtype=torch.long)}
            if modality in ('image', 'both'):
                pixel_values = to_tensor(Image.open(record['path']).convert('RGB'))
                if augment:
                    pixel_values = augment_image(torch, pixel_values)
                item['pixel_values'] = pixel_values
            if modality in ('text', 'both'):
                encoding = tokenizer(record['text'], max_length=MAX_SEQ_LENGTH,
                                     padding='max_length', truncation=True,
                                     return_tensors='pt')
                item['input_ids'] = encoding['input_ids'].squeeze(0)
                item['attention_mask'] = encoding['attention_mask'].squeeze(0)
            return item

    dataset = _Dataset()
    if not shuffle:
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)
    counts = Counter(r['label'] for r in records)
    weights = [1.0 / counts[r['label']] for r in records]
    sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double),
                                    num_samples=len(records), replacement=True)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)


def predict_probabilities(torch, model, loader, device):
    import numpy as np
    model.eval()
    probabilities, labels = [], []
    with torch.no_grad():
        for batch in loader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            logits = model(**inputs)['logits']
            probabilities.append(torch.softmax(logits, dim=-1).cpu())
            labels.append(batch['labels'])
    return (torch.cat(probabilities).numpy().astype(float),
            torch.cat(labels).numpy().astype(int))


def score(probabilities, labels, groups=None):
    import numpy as np
    predictions = probabilities.argmax(axis=1)
    per_class = []
    for index in range(len(CLASSES)):
        true_positive = int(((predictions == index) & (labels == index)).sum())
        denominator = int((predictions == index).sum()) + int((labels == index).sum())
        per_class.append((2.0 * true_positive / denominator) if denominator else 0.0)
    correct = int((predictions == labels).sum())
    total = int(len(labels))
    result = {
        'n': total, 'correct': correct,
        'accuracy': correct / total if total else 0.0,
        'macro_f1': float(np.mean(per_class)),
        'per_class_f1': {CLASSES[i]: float(v) for i, v in enumerate(per_class)},
        'predictions': predictions.tolist(), 'labels': labels.tolist(),
    }
    if groups is not None:
        # Average probabilities within an independent group before deciding, so
        # a template family with many variants gets one vote, not many.
        buckets = defaultdict(list)
        for row, group in enumerate(groups):
            buckets[group].append(row)
        group_probabilities, group_labels = [], []
        for rows in buckets.values():
            true = {int(labels[r]) for r in rows}
            if len(true) != 1:
                raise ValueError('A group contains multiple labels.')
            group_labels.append(true.pop())
            group_probabilities.append(probabilities[rows].mean(axis=0))
        group_probabilities = np.asarray(group_probabilities)
        group_labels = np.asarray(group_labels, dtype=int)
        group_predictions = group_probabilities.argmax(axis=1)
        group_per_class = []
        for index in range(len(CLASSES)):
            true_positive = int(((group_predictions == index) & (group_labels == index)).sum())
            denominator = int((group_predictions == index).sum()) + int((group_labels == index).sum())
            group_per_class.append((2.0 * true_positive / denominator) if denominator else 0.0)
        result['independent_groups'] = int(len(group_labels))
        result['group_correct'] = int((group_predictions == group_labels).sum())
        result['group_accuracy'] = float((group_predictions == group_labels).mean())
        result['group_macro_f1'] = float(np.mean(group_per_class))
    return result


def wilson(correct, total, z=1.959963984540054):
    if total < 1:
        return 0.0, 1.0
    proportion = correct / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    centre = (proportion + z2 / (2.0 * total)) / denominator
    half = z * math.sqrt(proportion * (1.0 - proportion) / total
                         + z2 / (4.0 * total * total)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def exact_mcnemar(labels, a, b):
    a_only = sum(1 for y, x, z in zip(labels, a, b) if x == y and z != y)
    b_only = sum(1 for y, x, z in zip(labels, a, b) if x != y and z == y)
    discordant = a_only + b_only
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(a_only, b_only) + 1))
    return float(min(1.0, 2.0 * tail / (2.0 ** discordant)))


def train(torch, model, train_loader, val_loader, device, epochs,
          encoder_lr, head_lr, tag):
    optimizer = torch.optim.AdamW([
        {'params': model.encoder.parameters(), 'lr': encoder_lr},
        {'params': model.classifier.parameters(), 'lr': head_lr},
    ], weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs * max(1, len(train_loader))))
    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    best_key = (-1.0, -1.0)
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            inputs = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast('cuda', enabled=use_amp):
                loss = model(**inputs)['loss']
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()
            total_loss += float(loss.item())
        probabilities, labels = predict_probabilities(torch, model, val_loader, device)
        metrics = score(probabilities, labels)
        key = (metrics['macro_f1'], metrics['accuracy'])
        marker = ''
        if key > best_key:
            best_key = key
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            marker = '   <- best'
        log(f'    [{tag}] epoch {epoch + 1}/{epochs} '
            f'loss={total_loss / max(1, len(train_loader)):.4f} '
            f'val_macro={metrics["macro_f1"]:.4f}{marker}')
    model.load_state_dict(best_state)
    return model


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    random.seed(SEED)
    torch = setup_environment()
    import numpy as np
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    bundle = resolve_bundle(mount_drive())
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(TEXT_BACKBONE)

    section('[3/6] SPLITS')
    images = collect_images(bundle / 'real_dataset_sorted')
    texts = collect_texts(bundle / 'text_data' / 'annotations.csv')
    log(f'  images {len(images)} in {len({r["group"] for r in images})} source groups')
    log(f'  texts  {len(texts)} unique in {len({r["group"] for r in texts})} template groups')

    # Image uses 70/15/15 so image_only here is directly comparable with the
    # cross-domain run's field_only. Text gets a larger held-out share because
    # it has only 11 independent template families per class; a 15% slice would
    # leave under two of them per class to test on.
    image_train, image_val, image_test = group_split(
        images, val_groups_ratio=0.15, test_groups_ratio=0.15)
    text_train, text_val, text_test = group_split(
        texts, val_groups_ratio=0.18, test_groups_ratio=0.27)
    assert_disjoint('image', image_train, image_val, image_test)
    assert_disjoint('text', text_train, text_val, text_test)

    # Common rows: each held-out image paired with a held-out observation of the
    # same class, drawn ONLY from the matching text split.
    val_pairs = pair_rows(image_val, text_val)
    test_pairs = pair_rows(image_test, text_test)
    log(f'  common validation pairs: {len(val_pairs)}   common test pairs: {len(test_pairs)}')
    log(f'  test pairs draw on {len({p["text_group"] for p in test_pairs})} '
        f'independent text templates and '
        f'{len({p["group"] for p in test_pairs})} independent photographs')

    section('[4/6] TRAINING')
    log('\n--- image branch ---')
    vision = build_vision_model(torch, len(CLASSES)).to(device)
    vision = train(
        torch, vision,
        make_loader(torch, image_train, 'image', None, BATCH_SIZE, True, True),
        make_loader(torch, image_val, 'image', None, BATCH_SIZE, False, False),
        device, EPOCHS_VISION, VISION_LR, VISION_HEAD_LR, 'image')

    log('\n--- text branch ---')
    text_model = build_text_model(torch, len(CLASSES)).to(device)
    text_model = train(
        torch, text_model,
        make_loader(torch, text_train, 'text', tokenizer, BATCH_SIZE, False, True),
        make_loader(torch, text_val, 'text', tokenizer, BATCH_SIZE, False, False),
        device, EPOCHS_TEXT, TEXT_LR, TEXT_HEAD_LR, 'text')

    section('[5/6] COMMON-ROW EVALUATION')
    val_image_probabilities, val_labels = predict_probabilities(
        torch, vision, make_loader(torch, val_pairs, 'image', None, BATCH_SIZE, False, False), device)
    val_text_probabilities, val_labels_text = predict_probabilities(
        torch, text_model, make_loader(torch, val_pairs, 'text', tokenizer, BATCH_SIZE, False, False), device)
    assert (val_labels == val_labels_text).all(), 'validation label order drift'

    test_image_probabilities, test_labels = predict_probabilities(
        torch, vision, make_loader(torch, test_pairs, 'image', None, BATCH_SIZE, False, False), device)
    test_text_probabilities, test_labels_text = predict_probabilities(
        torch, text_model, make_loader(torch, test_pairs, 'text', tokenizer, BATCH_SIZE, False, False), device)
    assert (test_labels == test_labels_text).all(), 'test label order drift'

    # Fusion weight chosen on VALIDATION only; the test rows never inform it.
    best_weight, best_key, sweep = 0.5, (-1.0, -1.0), []
    for step in range(101):
        weight = step / 100.0
        mixed = weight * val_text_probabilities + (1.0 - weight) * val_image_probabilities
        metrics = score(mixed, val_labels)
        sweep.append({'w_text': weight, 'val_macro_f1': metrics['macro_f1']})
        key = (metrics['macro_f1'], metrics['accuracy'], -abs(weight - 0.5))
        if key[:2] > best_key[:2] or (key[:2] == best_key[:2]
                                      and abs(weight - 0.5) < abs(best_weight - 0.5)):
            best_key, best_weight = key[:2], weight
    log(f'  validation-selected fusion weight: w_text={best_weight:.2f} '
        f'(val macro-F1={best_key[0]:.4f})')

    text_groups = [p['text_group'] for p in test_pairs]
    results = {
        'classes': CLASSES,
        'vision_backbone': VISION_BACKBONE, 'text_backbone': TEXT_BACKBONE,
        'seed': SEED,
        'pairing': 'label-aligned proxy pairs; text was NOT written about the photograph',
        'identical_rows': True, 'test_used_for_selection': False,
        'fusion_weight_text': best_weight,
        'fusion_weight_sweep': sweep,
        'split_sizes': {'image_train': len(image_train), 'image_val': len(image_val),
                        'image_test': len(image_test), 'text_train': len(text_train),
                        'text_val': len(text_val), 'text_test': len(text_test)},
        'conditions': {},
    }
    results['conditions']['image_only'] = score(test_image_probabilities, test_labels)
    results['conditions']['text_only'] = score(
        test_text_probabilities, test_labels, groups=text_groups)
    results['conditions']['fusion'] = score(
        best_weight * test_text_probabilities + (1.0 - best_weight) * test_image_probabilities,
        test_labels)

    for condition in results['conditions'].values():
        low, high = wilson(condition['correct'], condition['n'])
        condition['accuracy_ci95'] = [float(low), float(high)]

    comparisons = []
    names = list(results['conditions'])
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            a, b = results['conditions'][left], results['conditions'][right]
            comparisons.append({
                'a': left, 'b': right,
                'macro_f1_delta': a['macro_f1'] - b['macro_f1'],
                'mcnemar_exact_p': exact_mcnemar(a['labels'], a['predictions'], b['predictions']),
                'identical_predictions': a['predictions'] == b['predictions'],
            })
    results['paired_comparisons'] = comparisons

    section('[6/6] RESULTS')
    n_test = results['conditions']['image_only']['n']
    log(f'Identical held-out rows: n={n_test}. '
        f'One changed prediction moves accuracy by {1.0 / n_test:.3f}.')
    log('')
    log(f'{"condition":14s} {"macro-F1":>9s} {"accuracy":>9s} {"95% CI":>18s}')
    for name, condition in results['conditions'].items():
        low, high = condition['accuracy_ci95']
        log(f'{name:14s} {condition["macro_f1"]:9.4f} {condition["accuracy"]:9.4f} '
            f'{f"[{low:.3f}, {high:.3f}]":>18s}')
    text_condition = results['conditions']['text_only']
    log(f'\ntext_only over independent template groups: '
        f'{text_condition["group_correct"]}/{text_condition["independent_groups"]} '
        f'group macro-F1={text_condition["group_macro_f1"]:.4f}')
    log('  (row-level text rests on far fewer independent units than n suggests)')
    log('\nper-class F1:')
    for name, condition in results['conditions'].items():
        rendered = '  '.join(f'{k}={v:.3f}' for k, v in condition['per_class_f1'].items())
        log(f'  {name:14s} {rendered}')
    log('\npaired exact McNemar:')
    for row in comparisons:
        note = '  [identical predictions]' if row['identical_predictions'] else ''
        log(f'  {row["a"]:12s} vs {row["b"]:12s} '
            f'd_macro={row["macro_f1_delta"]:+.4f}  p={row["mcnemar_exact_p"]:.4f}{note}')

    output = work_root() / 'modality_comparison_results.json'
    output.write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')
    log(f'\nSaved: {output.resolve()}')


if __name__ == '__main__':
    main()
