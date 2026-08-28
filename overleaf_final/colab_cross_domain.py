#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FarmFederate cross-domain experiment - SINGLE-CELL COLAB / KAGGLE SCRIPT.

Paste this whole file into one cell and run it. Nothing else is needed: no
other project file has to be uploaded.

  Colab   Runtime  -> Change runtime type -> T4 GPU
  Kaggle  Settings -> Accelerator: GPU, and Internet: ON
          (Internet is required once to fetch the pretrained backbone.)

Kaggle inputs
-------------
Attach your data as Kaggle Datasets; the script searches /kaggle/input
recursively and accepts either form of each:

  field   an expanded  real_dataset_sorted/  folder, or  data_final.zip
  studio  an expanded  studio_sorted/        folder, or  teasickness_studio.zip

If the studio data is absent it is rebuilt from Kaggle automatically.
/kaggle/input is read-only, so anything that must be unpacked is written to
/kaggle/working, which is also where results are saved as notebook output.

Question
--------
Does the public studio-photographed tea benchmark (TeaSickness) transfer to
real in-situ field photographs, or does the imaging-domain gap block it?

  A. field_only         train on held-in field images
  B. studio_only        train on studio images only (zero-shot to field)
  C. studio_then_field  studio pretraining, then field fine-tuning

The held-out TEST set is ALWAYS field-only. Studio images are barred from
validation and test by a hard assertion: the two domains are separable on
background alone, so admitting studio data into evaluation would exploit the
domain gap rather than measure it. That is the same failure mode that inflated
the earlier image-only result.

Scope
-----
Three classes, the ones with a defensible studio source:
LEAF_BLIGHT, LEAF_RUST, MOSQUITO_BUG.

LOOPER_CATERPILLARS has no studio source. LEAF_HOPPERS is excluded from the
benchmark: all 8 field images show no leafhopper symptoms (no stippling, tip
burn, curl or bronzing), which is why every vision backbone scored exactly
0.000 on it.

Data
----
Field images come from data_final.zip on Google Drive. The studio bundle is
taken from Drive if present, otherwise rebuilt from Kaggle automatically.
"""
from __future__ import annotations

import hashlib
import json
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
BACKBONE = 'microsoft/swin-tiny-patch4-window7-224'
EPOCHS_FIELD = 15
EPOCHS_STUDIO = 6
BATCH_SIZE = 16
ENCODER_LR = 2e-5
HEAD_LR = 1e-4
SEED = 42

# data_final.zip on Drive (public fallback id used by the main pipeline)
DATA_FINAL_DRIVE_ID = '1PBZJrCGQ8-sxZnKzxMPhA9ySzOZWmMzf'
KAGGLE_TEASICKNESS = 'rohitttdx/tea-sickness-dataset'

CLASSES = ['LEAF_BLIGHT', 'LEAF_RUST', 'MOSQUITO_BUG']
CLASS_TO_IDX = {name: index for index, name in enumerate(CLASSES)}
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

# TeaSickness source class -> (target class, justification, confidence)
STUDIO_MAPPING = {
    'brown_blight': ('LEAF_BLIGHT',
                     'Foliar blight of tea (Colletotrichum spp.); same symptom '
                     'category as the target class.', 'solid'),
    'gray_blight': ('LEAF_BLIGHT',
                    'Foliar blight of tea (Pestalotiopsis spp.); merged with '
                    'brown blight into the single LEAF_BLIGHT target.', 'solid'),
    'helopeltis': ('MOSQUITO_BUG',
                   'Helopeltis theivora IS the tea mosquito bug - same organism, '
                   'different common name. Dark punctate feeding lesions match '
                   'the field MOSQUITO_BUG images.', 'exact'),
    'algal_spot': ('LEAF_RUST',
                   'Tea "red rust" is caused by the alga Cephaleuros, the same '
                   'pathogen genus as algal leaf spot. Verified visually: discrete '
                   'small tan/orange circular lesions, as in field LEAF_RUST.',
                   'defensible-with-caveat'),
}
STUDIO_EXCLUDED = {
    'red_spot': 'Large-area whole-leaf reddening, not the discrete small spots '
                'of field LEAF_RUST. Visually distinct; would contaminate the class.',
    'healthy': 'No corresponding target class in the FarmFederate schema.',
}

def _detect_runtime():
    """Kaggle's markers are the more specific ones, so they take priority.

    Some Kaggle images expose Colab-like modules or paths; misdetecting the
    runtime would send the Drive resolver at a Kaggle input mount.
    """
    in_kaggle = bool(os.environ.get('KAGGLE_KERNEL_RUN_TYPE')) or (
        Path('/kaggle/input').is_dir() and Path('/kaggle/working').is_dir()
    )
    in_colab = False if in_kaggle else (
        'google.colab' in sys.modules or Path('/content').exists()
    )
    return in_colab, in_kaggle


IN_COLAB, IN_KAGGLE = _detect_runtime()
KAGGLE_INPUT = Path(os.environ.get('KAGGLE_INPUT_DIR', '/kaggle/input'))
KAGGLE_WORKING = Path(os.environ.get('KAGGLE_WORKING_DIR', '/kaggle/working'))

if IN_KAGGLE:
    # Kaggle inputs are read-only and $HOME is small. Point every download
    # cache at the writable working area before transformers is imported,
    # otherwise fetching the pretrained backbone fails.
    _cache_root = KAGGLE_WORKING / '.cache'
    os.environ.setdefault('HF_HOME', str(_cache_root / 'huggingface'))
    os.environ.setdefault('TORCH_HOME', str(_cache_root / 'torch'))
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')


def work_root() -> Path:
    """Writable scratch/output directory for the detected runtime."""
    if IN_KAGGLE:
        return KAGGLE_WORKING
    if IN_COLAB:
        return Path('/content')
    return Path('.')


def find_in_kaggle_input(dir_name: str, zip_name: str):
    """Locate an expanded folder or a zip among the attached Kaggle Datasets.

    Returns ``('dir'|'zip', path)`` or ``(None, None)``. Expanded folders win:
    they need no unpacking and can be read in place from /kaggle/input.
    """
    if not KAGGLE_INPUT.is_dir():
        return None, None
    for candidate in sorted(KAGGLE_INPUT.rglob(dir_name)):
        if candidate.is_dir() and any((candidate / c).is_dir() for c in CLASSES):
            return 'dir', candidate
    for candidate in sorted(KAGGLE_INPUT.rglob(zip_name)):
        if candidate.is_file():
            return 'zip', candidate
    return None, None


def log(message):
    print(message, flush=True)


def section(title):
    log('\n' + '=' * 78)
    log(title)
    log('=' * 78)


# ===========================================================================
# 1. ENVIRONMENT
# ===========================================================================
def setup_environment():
    section('[1/6] ENVIRONMENT')
    runtime = 'Kaggle' if IN_KAGGLE else ('Colab' if IN_COLAB else 'local')
    log(f'  runtime: {runtime}')
    if IN_KAGGLE:
        log(f'  inputs (read-only): {KAGGLE_INPUT}')
        log(f'  working/output    : {KAGGLE_WORKING}')
        log(f'  HF cache          : {os.environ.get("HF_HOME")}')

    for module, package in (('transformers', 'transformers'),
                            ('PIL', 'pillow'),
                            ('numpy', 'numpy')):
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
        log('  [WARN] No GPU detected. On CPU this takes hours, not ~25 minutes.')
        if IN_KAGGLE:
            log('         Kaggle: Settings -> Accelerator -> GPU T4 x2.')
        else:
            log('         Colab: Runtime -> Change runtime type -> T4 GPU.')
    if IN_KAGGLE:
        log('  [NOTE] Internet must be ON (Settings -> Internet) so the '
            'pretrained backbone can be downloaded.')
    return torch


def mount_drive():
    """Colab only. Kaggle reads from its attached Inputs instead."""
    if IN_KAGGLE:
        log('  Kaggle runtime - using attached Inputs, no Drive mount.')
        return None
    if not IN_COLAB:
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


# ===========================================================================
# 2. FIELD DATA (data_final.zip from Drive)
# ===========================================================================
def resolve_field_root(drive_root):
    section('[2/6] FIELD DATA (real_dataset_sorted)')

    existing = _find_sorted_root(work_root())
    if existing:
        log(f'  Already extracted: {existing}')
        return existing

    bundle = None

    if IN_KAGGLE:
        kind, found = find_in_kaggle_input('real_dataset_sorted', 'data_final.zip')
        if kind == 'dir':
            # Read straight from the read-only input mount; nothing to unpack.
            log(f'  Kaggle input (expanded): {found}')
            return found
        if kind == 'zip':
            bundle = found
            log(f'  Kaggle input (zip): {bundle}')
        else:
            raise FileNotFoundError(
                'No field data among the attached Kaggle Datasets. Use '
                '"Add Input" to attach a dataset containing either an expanded '
                'real_dataset_sorted/ folder or data_final.zip.'
            )
    else:
        candidates = []
        if drive_root is not None:
            candidates += [
                drive_root / 'FarmFederate' / 'data_final.zip',
                drive_root / 'data_final.zip',
            ]
        candidates += [Path('/content/data_final.zip'), Path('data_final.zip')]
        for candidate in candidates:
            if candidate.is_file():
                bundle = candidate
                log(f'  Found on Drive/local: {bundle}')
                break

        if bundle is None:
            log('  Not found on Drive - downloading the public copy ...')
            try:
                subprocess.check_call(
                    [sys.executable, '-m', 'pip', 'install', '-q', 'gdown'])
                import gdown
                target = work_root() / 'data_final.zip'
                gdown.download(id=DATA_FINAL_DRIVE_ID, output=str(target),
                               quiet=False, fuzzy=True)
                if target.is_file():
                    bundle = target
            except Exception as error:
                log(f'  [ERROR] Download failed: {error}')

        if bundle is None:
            raise FileNotFoundError(
                'data_final.zip not found. Put it at '
                'MyDrive/FarmFederate/data_final.zip or MyDrive/data_final.zip, '
                'or upload it to /content/.'
            )

    destination = work_root() / 'field_data'
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    log(f'  Extracting {bundle.name} -> {destination} ...')
    with zipfile.ZipFile(bundle) as archive:
        archive.extractall(destination)

    sorted_root = _find_sorted_root(destination)
    if sorted_root is None:
        raise FileNotFoundError(f'No real_dataset_sorted/ folder inside {bundle}.')
    log(f'  Field images: {sorted_root}')
    return sorted_root


def _find_sorted_root(root: Path):
    if not root.exists():
        return None
    for candidate in root.rglob('real_dataset_sorted'):
        if candidate.is_dir() and any((candidate / c).is_dir() for c in CLASSES):
            return candidate
    return None


# ===========================================================================
# 3. STUDIO DATA (Drive bundle, else rebuild from Kaggle)
# ===========================================================================
def resolve_studio_root(drive_root):
    section('[3/6] STUDIO DATA (TeaSickness)')

    existing = _find_studio_root(work_root())
    if existing:
        log(f'  Already extracted: {existing}')
        return existing

    candidates = []
    if IN_KAGGLE:
        kind, found = find_in_kaggle_input('studio_sorted', 'teasickness_studio.zip')
        if kind == 'dir':
            log(f'  Kaggle input (expanded): {found}')
            return found
        if kind == 'zip':
            candidates.append(found)
    else:
        if drive_root is not None:
            candidates += [
                drive_root / 'FarmFederate' / 'teasickness_studio.zip',
                drive_root / 'teasickness_studio.zip',
            ]
        candidates += [
            Path('/content/teasickness_studio.zip'),
            Path('teasickness_studio.zip'),
        ]

    for candidate in candidates:
        if candidate.is_file():
            log(f'  Found prebuilt bundle: {candidate}')
            destination = work_root()
            with zipfile.ZipFile(candidate) as archive:
                archive.extractall(destination)
            found = _find_studio_root(destination)
            if found:
                log(f'  Studio images: {found}')
                return found

    log('  No prebuilt bundle - rebuilding from the Kaggle API ...')
    return build_studio_from_kaggle()


def _find_studio_root(root: Path):
    if not root.exists():
        return None
    for candidate in root.rglob('studio_sorted'):
        if candidate.is_dir() and any((candidate / c).is_dir() for c in CLASSES):
            return candidate
    return None


def ensure_kaggle_credentials():
    if (Path.home() / '.kaggle' / 'kaggle.json').is_file():
        return True
    if os.environ.get('KAGGLE_USERNAME') and os.environ.get('KAGGLE_KEY'):
        target = Path.home() / '.kaggle' / 'kaggle.json'
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({
            'username': os.environ['KAGGLE_USERNAME'],
            'key': os.environ['KAGGLE_KEY'],
        }))
        target.chmod(0o600)
        return True
    if IN_COLAB:
        log('  Kaggle credentials needed. Upload kaggle.json now'
            ' (Kaggle -> Account -> Create New API Token):')
        try:
            from google.colab import files
            uploaded = files.upload()
            if 'kaggle.json' in uploaded:
                target = Path.home() / '.kaggle' / 'kaggle.json'
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(uploaded['kaggle.json'])
                target.chmod(0o600)
                return True
        except Exception as error:
            log(f'  [ERROR] Upload failed: {error}')
    elif IN_KAGGLE:
        log('  No Kaggle API credentials in this notebook. Either attach the '
            'studio data as an Input (an expanded studio_sorted/ folder or '
            'teasickness_studio.zip), or add KAGGLE_USERNAME / KAGGLE_KEY '
            'via Add-ons -> Secrets.')
    return False


def average_hash(path: Path, size: int = 12) -> str:
    import numpy as np
    from PIL import Image
    image = Image.open(path).convert('L').resize((size, size), Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32)
    return hashlib.md5((array > array.mean()).tobytes()).hexdigest()[:16]


def build_studio_from_kaggle():
    if not ensure_kaggle_credentials():
        raise RuntimeError(
            'Kaggle credentials unavailable and no teasickness_studio.zip on Drive. '
            'Upload teasickness_studio.zip to MyDrive/FarmFederate/ and re-run.'
        )
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'kaggle'])
    import kaggle

    kaggle.api.authenticate()
    raw = work_root() / 'teasickness_raw'
    raw.mkdir(parents=True, exist_ok=True)
    log(f'  Downloading {KAGGLE_TEASICKNESS} ...')
    kaggle.api.dataset_download_files(KAGGLE_TEASICKNESS, path=str(raw),
                                      unzip=True, quiet=True)

    source_root = None
    for candidate in raw.rglob('*'):
        if candidate.is_dir() and all(
            (candidate / name).is_dir() for name in STUDIO_MAPPING
        ):
            source_root = candidate
            break
    if source_root is None:
        raise FileNotFoundError('Could not locate TeaSickness class folders.')

    output = work_root() / 'teasickness_studio'
    if output.exists():
        shutil.rmtree(output)
    sorted_root = output / 'studio_sorted'
    sorted_root.mkdir(parents=True)

    manifest = {
        'bundle_version': 'teasickness_studio_v1',
        'domain': 'studio',
        'usage_constraint': 'TRAINING ONLY - never validation or test.',
        'source_dataset': f'Kaggle {KAGGLE_TEASICKNESS}',
        'mapping': {}, 'excluded_source_classes': STUDIO_EXCLUDED,
    }
    log('  Mapping classes onto the FarmFederate schema ...')
    for source_class, (target, justification, confidence) in STUDIO_MAPPING.items():
        target_dir = sorted_root / target
        target_dir.mkdir(parents=True, exist_ok=True)
        images = sorted(p for p in (source_root / source_class).iterdir()
                        if p.suffix.lower() in IMAGE_SUFFIXES)
        for image_path in images:
            group = average_hash(image_path)
            stem = image_path.stem.replace(' ', '_').replace('(', '').replace(')', '')
            shutil.copy2(
                image_path,
                target_dir / f'studio__{source_class}__g{group}__{stem}{image_path.suffix.lower()}',
            )
        manifest['mapping'][source_class] = {
            'target_class': target, 'justification': justification,
            'confidence': confidence, 'images': len(images),
        }
        log(f'    {source_class:14s} -> {target:14s} {len(images):5d} images')

    (output / 'mapping_manifest.json').write_text(
        json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return sorted_root


# ===========================================================================
# 4. MODEL - faithful copy of the pipeline's HuggingFaceVisionClassifier
# ===========================================================================
def encoder_hidden_dim(config):
    """Encoder output width. Deliberately prefers transformer output attrs."""
    for attribute in ('hidden_size', 'dim', 'd_model', 'n_embd'):
        if hasattr(config, attribute):
            return int(getattr(config, attribute))
    if getattr(config, 'hidden_sizes', None):
        return int(config.hidden_sizes[-1])
    if hasattr(config, 'hidden_dim'):
        return int(config.hidden_dim)
    if hasattr(config, 'embed_dim'):
        return int(config.embed_dim)
    return 768


def build_vision_classifier(torch, model_name, num_labels, dropout=0.20,
                            label_smoothing=0.05):
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import AutoImageProcessor, AutoModel

    class VisionClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.label_smoothing = float(label_smoothing)
            self.encoder = AutoModel.from_pretrained(model_name)
            try:
                processor = AutoImageProcessor.from_pretrained(model_name, use_fast=False)
            except TypeError:
                processor = AutoImageProcessor.from_pretrained(model_name)
            self.register_buffer('processor_mean', torch.tensor(
                getattr(processor, 'image_mean', [0.5, 0.5, 0.5]),
                dtype=torch.float32).view(1, 3, 1, 1))
            self.register_buffer('processor_std', torch.tensor(
                getattr(processor, 'image_std', [0.5, 0.5, 0.5]),
                dtype=torch.float32).view(1, 3, 1, 1))
            self.feature_dim = encoder_hidden_dim(self.encoder.config)
            self.classifier = nn.Sequential(
                nn.LayerNorm(self.feature_dim),
                nn.Dropout(dropout),
                nn.Linear(self.feature_dim, num_labels),
            )
            nn.init.normal_(self.classifier[-1].weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.classifier[-1].bias)

        def encode_features(self, pixel_values):
            # Dataset tensors use ImageNet normalization; convert back to RGB
            # then apply this checkpoint's own processor statistics.
            imagenet_mean = pixel_values.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            imagenet_std = pixel_values.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            raw = torch.clamp(pixel_values * imagenet_std + imagenet_mean, 0.0, 1.0)
            model_pixels = (raw - self.processor_mean.to(raw.dtype)) / \
                self.processor_std.to(raw.dtype)
            outputs = self.encoder(pixel_values=model_pixels, return_dict=True)
            hidden = outputs.last_hidden_state
            model_type = str(getattr(self.encoder.config, 'model_type', '')).lower()
            # ViT/DeiT classification checkpoints often omit the tanh pooler;
            # use the directly pretrained CLS token instead.
            if hidden.dim() == 3 and model_type in {'vit', 'deit'}:
                return hidden[:, 0, :]
            pooled = getattr(outputs, 'pooler_output', None)
            if pooled is not None and pooled.dim() == 2:
                return pooled
            if hidden.dim() == 4:
                return hidden.mean(dim=(-2, -1))
            if hidden.dim() == 3:
                return hidden.mean(dim=1)
            return hidden

        def forward(self, pixel_values, labels=None):
            features = self.encode_features(pixel_values)
            logits = self.classifier(features)
            loss = None
            if labels is not None:
                target = labels.argmax(dim=-1) if labels.dim() > 1 else labels
                loss = F.cross_entropy(logits, target.long(),
                                       label_smoothing=self.label_smoothing)
            return {'loss': loss, 'logits': logits}

    return VisionClassifier()


def augment_normalized(torch, pixel_values):
    """Training-only view of an ImageNet-normalized image (pipeline recipe)."""
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
    gain = raw.new_tensor([0.95 + 0.10 * random.random() for _ in range(3)]).view(3, 1, 1)
    raw = torch.clamp(raw * brightness * gain + torch.randn_like(raw) * 0.015, 0.0, 1.0)
    return (raw - mean) / std


# ===========================================================================
# 5. STATISTICS
# ===========================================================================
def wilson_interval(correct, total, z=1.959963984540054):
    import math
    if total < 1:
        return 0.0, 1.0
    proportion = correct / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    centre = (proportion + z2 / (2.0 * total)) / denominator
    half = z * math.sqrt(proportion * (1.0 - proportion) / total
                         + z2 / (4.0 * total * total)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def exact_mcnemar(labels, predictions_a, predictions_b):
    import math
    a_only = sum(1 for y, a, b in zip(labels, predictions_a, predictions_b)
                 if a == y and b != y)
    b_only = sum(1 for y, a, b in zip(labels, predictions_a, predictions_b)
                 if a != y and b == y)
    discordant = a_only + b_only
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(a_only, b_only) + 1))
    return float(min(1.0, 2.0 * tail / (2.0 ** discordant)))


# ===========================================================================
# 6. DATA HANDLING
# ===========================================================================
def field_group(path: Path) -> str:
    return re.sub(r'\.rf\.[0-9a-fA-F]+$', '', path.stem).casefold()


def studio_group(path: Path) -> str:
    match = re.search(r'__g([0-9a-f]+)__', path.name)
    return f'studio_{match.group(1)}' if match else f'studio_{path.stem.casefold()}'


def collect(root: Path, domain: str):
    grouper = field_group if domain == 'field' else studio_group
    records = []
    for class_name in CLASSES:
        class_dir = Path(root) / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f'Missing {domain} class folder: {class_dir}')
        for image_path in sorted(class_dir.rglob('*')):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_SUFFIXES:
                records.append({
                    'path': image_path, 'label': CLASS_TO_IDX[class_name],
                    'group': f'{domain}:{grouper(image_path)}', 'domain': domain,
                })
    return records


def grouped_split(records, train_ratio=0.70, val_ratio=0.15, seed=SEED):
    """Split by source group so no photograph straddles a split boundary."""
    by_class = defaultdict(lambda: defaultdict(list))
    for record in records:
        by_class[record['label']][record['group']].append(record)

    train, val, test = [], [], []
    for label in sorted(by_class):
        groups = list(by_class[label])
        random.Random(seed + label * 7919).shuffle(groups)
        if len(groups) < 3:
            raise ValueError(f'Class {CLASSES[label]} has only {len(groups)} groups.')
        n_val = max(1, round(len(groups) * val_ratio))
        n_test = max(1, round(len(groups) * (1.0 - train_ratio - val_ratio)))
        while n_val + n_test >= len(groups):
            if n_test >= n_val and n_test > 1:
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


def assert_guards(train, val, test):
    for name, split in (('validation', val), ('test', test)):
        offenders = [r for r in split if r['domain'] != 'field']
        if offenders:
            raise AssertionError(
                f'{len(offenders)} non-field images reached {name}. '
                'Studio data is training-only by construction.')
    groups = {name: {r['group'] for r in split}
              for name, split in (('train', train), ('val', val), ('test', test))}
    for left, right in (('train', 'val'), ('train', 'test'), ('val', 'test')):
        overlap = groups[left] & groups[right]
        if overlap:
            raise AssertionError(
                f'{len(overlap)} source groups shared between {left} and {right}.')
    log('  GUARDS PASSED: test is field-only; source groups disjoint.')


def make_loader(torch, records, batch_size, augment, shuffle):
    import torchvision.transforms as T
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

    to_tensor = T.Compose([
        T.Resize((224, 224)), T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    class _Dataset(Dataset):
        def __len__(self):
            return len(records)

        def __getitem__(self, index):
            record = records[index]
            pixel_values = to_tensor(Image.open(record['path']).convert('RGB'))
            if augment:
                pixel_values = augment_normalized(torch, pixel_values)
            labels = torch.zeros(len(CLASSES), dtype=torch.float32)
            labels[record['label']] = 1.0
            return {'pixel_values': pixel_values, 'labels': labels}

    dataset = _Dataset()
    if not shuffle:
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)
    counts = Counter(r['label'] for r in records)
    weights = [1.0 / counts[r['label']] for r in records]
    sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double),
                                    num_samples=len(records), replacement=True)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)


# ===========================================================================
# 7. TRAIN / EVALUATE
# ===========================================================================
def evaluate(torch, model, loader, device):
    import numpy as np
    model.eval()
    predictions, labels = [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(pixel_values=batch['pixel_values'].to(device))['logits']
            predictions.append(logits.argmax(dim=-1).cpu())
            labels.append(batch['labels'].argmax(dim=-1))
    predictions = torch.cat(predictions).numpy()
    labels = torch.cat(labels).numpy()

    per_class = []
    for index in range(len(CLASSES)):
        true_positive = int(((predictions == index) & (labels == index)).sum())
        predicted = int((predictions == index).sum())
        actual = int((labels == index).sum())
        denominator = predicted + actual
        per_class.append((2.0 * true_positive / denominator) if denominator else 0.0)

    correct = int((predictions == labels).sum())
    total = int(len(labels))
    return {
        'n': total, 'correct': correct,
        'accuracy': correct / total if total else 0.0,
        'macro_f1': float(np.mean(per_class)),
        'per_class_f1': {CLASSES[i]: float(v) for i, v in enumerate(per_class)},
        'predictions': predictions.tolist(), 'labels': labels.tolist(),
    }


def train(torch, model, train_loader, val_loader, device, epochs, tag):
    """Fine-tune; keep the checkpoint with the best VALIDATION macro-F1."""
    optimizer = torch.optim.AdamW([
        {'params': model.encoder.parameters(), 'lr': ENCODER_LR},
        {'params': model.classifier.parameters(), 'lr': HEAD_LR},
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
            pixels = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            with torch.amp.autocast('cuda', enabled=use_amp):
                loss = model(pixel_values=pixels, labels=labels)['loss']
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

        metrics = evaluate(torch, model, val_loader, device)
        key = (metrics['macro_f1'], metrics['accuracy'])
        marker = ''
        if key > best_key:
            best_key = key
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            marker = '   <- best'
        log(f'    [{tag}] epoch {epoch + 1}/{epochs} '
            f'loss={total_loss / max(1, len(train_loader)):.4f} '
            f'val_macro={metrics["macro_f1"]:.4f} '
            f'val_acc={metrics["accuracy"]:.4f}{marker}')

    model.load_state_dict(best_state)
    return model, best_key[0]


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

    drive_root = mount_drive()
    field_root = resolve_field_root(drive_root)
    studio_root = resolve_studio_root(drive_root)

    section('[4/6] DATA AUDIT')
    field = collect(field_root, 'field')
    studio = collect(studio_root, 'studio')
    for name, records in (('field', field), ('studio', studio)):
        counts = Counter(CLASSES[r['label']] for r in records)
        log(f'  {name:6s} {len(records):5d} images  '
            f'{len({r["group"] for r in records}):5d} source groups   '
            + '  '.join(f'{c}={counts[c]}' for c in CLASSES))

    field_train, field_val, field_test = grouped_split(field)
    assert_guards(field_train, field_val, field_test)
    log(f'  field split: train={len(field_train)} val={len(field_val)} '
        f'test={len(field_test)}')

    val_loader = make_loader(torch, field_val, BATCH_SIZE, False, False)
    test_loader = make_loader(torch, field_test, BATCH_SIZE, False, False)
    field_loader = make_loader(torch, field_train, BATCH_SIZE, True, True)
    studio_loader = make_loader(torch, studio, BATCH_SIZE, True, True)

    results = {
        'question': 'Does studio-photographed public tea data transfer to field images?',
        'classes': CLASSES, 'backbone': BACKBONE, 'seed': SEED,
        'test_is_field_only': True, 'studio_barred_from_eval': True,
        'studio_mapping': {k: v[0] for k, v in STUDIO_MAPPING.items()},
        'studio_excluded': STUDIO_EXCLUDED,
        'field_counts': dict(Counter(CLASSES[r['label']] for r in field)),
        'studio_counts': dict(Counter(CLASSES[r['label']] for r in studio)),
        'split_sizes': {'train': len(field_train), 'val': len(field_val),
                        'test': len(field_test)},
        'conditions': {},
    }

    section('[5/6] TRAINING')
    log('\n--- A. field_only ---')
    model = build_vision_classifier(torch, BACKBONE, len(CLASSES)).to(device)
    model, val_macro = train(torch, model, field_loader, val_loader, device,
                             EPOCHS_FIELD, 'field_only')
    results['conditions']['field_only'] = {
        **evaluate(torch, model, test_loader, device), 'val_macro_f1': val_macro}
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    log('\n--- B. studio_only (zero-shot to field) ---')
    model = build_vision_classifier(torch, BACKBONE, len(CLASSES)).to(device)
    model, val_macro = train(torch, model, studio_loader, val_loader, device,
                             EPOCHS_STUDIO, 'studio_only')
    results['conditions']['studio_only'] = {
        **evaluate(torch, model, test_loader, device), 'val_macro_f1': val_macro}
    studio_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    log('\n--- C. studio_then_field ---')
    model = build_vision_classifier(torch, BACKBONE, len(CLASSES)).to(device)
    model.load_state_dict(studio_state)
    model, val_macro = train(torch, model, field_loader, val_loader, device,
                             EPOCHS_FIELD, 'studio_then_field')
    results['conditions']['studio_then_field'] = {
        **evaluate(torch, model, test_loader, device), 'val_macro_f1': val_macro}
    del model

    for condition in results['conditions'].values():
        low, high = wilson_interval(condition['correct'], condition['n'])
        condition['accuracy_ci95'] = [float(low), float(high)]

    reference = results['conditions']['field_only']['labels']
    comparisons = []
    names = list(results['conditions'])
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            a, b = results['conditions'][left], results['conditions'][right]
            assert a['labels'] == b['labels'] == reference, 'label order drift'
            comparisons.append({
                'a': left, 'b': right,
                'macro_f1_delta': a['macro_f1'] - b['macro_f1'],
                'accuracy_delta': a['accuracy'] - b['accuracy'],
                'mcnemar_exact_p': exact_mcnemar(a['labels'], a['predictions'],
                                                 b['predictions']),
            })
    results['paired_comparisons'] = comparisons

    section('[6/6] RESULTS')
    n_test = results['conditions']['field_only']['n']
    log(f'Held-out test: {n_test} FIELD images. '
        f'One changed prediction moves accuracy by {1.0 / n_test:.3f}.')
    log('')
    log(f'{"condition":22s} {"macro-F1":>9s} {"accuracy":>9s} {"95% CI":>18s}')
    for name, condition in results['conditions'].items():
        low, high = condition['accuracy_ci95']
        log(f'{name:22s} {condition["macro_f1"]:9.4f} {condition["accuracy"]:9.4f} '
            f'{f"[{low:.3f}, {high:.3f}]":>18s}')
    log('\nper-class F1:')
    for name, condition in results['conditions'].items():
        rendered = '  '.join(f'{k}={v:.3f}' for k, v in condition['per_class_f1'].items())
        log(f'  {name:22s} {rendered}')
    log('\npaired exact McNemar:')
    for row in comparisons:
        log(f'  {row["a"]:20s} vs {row["b"]:20s} '
            f'd_macro={row["macro_f1_delta"]:+.4f}  p={row["mcnemar_exact_p"]:.4f}')

    output = work_root() / 'cross_domain_results.json'
    output.write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')
    log(f'\nSaved: {output.resolve()}')
    if IN_KAGGLE:
        log('  (in /kaggle/working - it appears under the notebook Output tab)')
    if drive_root is not None:
        try:
            backup = drive_root / 'FarmFederate' / 'cross_domain_results.json'
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(output, backup)
            log(f'Saved to Drive: {backup}')
        except Exception as error:
            log(f'[WARN] Drive copy failed: {error}')


if __name__ == '__main__':
    main()
