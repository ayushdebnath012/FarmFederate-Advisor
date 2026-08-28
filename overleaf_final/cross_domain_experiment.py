#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cross-domain tea disease experiment: studio-trained vs field-trained.

Question
--------
Does the public studio-photographed tea disease benchmark (TeaSickness) improve
diagnosis on real in-situ field photographs, or does the imaging-domain gap
prevent transfer?

Design
------
Three training conditions, ONE evaluation set:

  A. field_only        train on held-in field images
  B. studio_only       train on studio images only (zero-shot to field)
  C. studio_then_field studio pretraining, then field fine-tuning

The held-out TEST set is ALWAYS field-only. Studio images are structurally
barred from validation and test by an assertion, because a model can separate
the two domains on background alone; letting studio images into evaluation
would exploit the domain gap instead of measuring it.

Scope
-----
Three classes, the ones with a defensible studio source:
LEAF_BLIGHT, LEAF_RUST, MOSQUITO_BUG.

LOOPER_CATERPILLARS has no studio source. LEAF_HOPPERS was removed from the
benchmark entirely: inspection of all 8 images found no leafhopper symptoms
(no stippling, tip burn, curl or bronzing), which is why every vision backbone
scored exactly 0.000 on it. See the accompanying analysis.

Usage
-----
    python cross_domain_experiment.py \
        --field-root  data_final/data_final/real_dataset_sorted \
        --studio-root teasickness_studio/studio_sorted \
        --backbone    microsoft/swin-tiny-patch4-window7-224
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# Classes with both a field and a studio source.
CLASSES = ['LEAF_BLIGHT', 'LEAF_RUST', 'MOSQUITO_BUG']
CLASS_TO_IDX = {name: index for index, name in enumerate(CLASSES)}
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def load_farmfederate(path: Path):
    """Import the pipeline module so the experiment reuses its exact encoder."""
    spec = importlib.util.spec_from_file_location('farmfederate', str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules['farmfederate'] = module
    spec.loader.exec_module(module)
    return module


def field_source_group(path: Path) -> str:
    """Collapse Roboflow export variants of one photograph to a single group."""
    return re.sub(r'\.rf\.[0-9a-fA-F]+$', '', path.stem).casefold()


def studio_source_group(path: Path) -> str:
    """Recover the perceptual group token written by build_studio_bundle.py."""
    match = re.search(r'__g([0-9a-f]+)__', path.name)
    return f'studio_{match.group(1)}' if match else f'studio_{path.stem.casefold()}'


def collect(root: Path, domain: str) -> list:
    """Return [{path, label, group, domain}] for every image under ``root``."""
    root = Path(root)
    grouper = field_source_group if domain == 'field' else studio_source_group
    records = []
    for class_name in CLASSES:
        class_dir = root / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f'Missing {domain} class folder: {class_dir}')
        for image_path in sorted(class_dir.rglob('*')):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_SUFFIXES:
                records.append({
                    'path': image_path,
                    'label': CLASS_TO_IDX[class_name],
                    'group': f'{domain}:{grouper(image_path)}',
                    'domain': domain,
                })
    return records


def grouped_stratified_split(records, train_ratio=0.70, val_ratio=0.15, seed=42):
    """Split by source group, keeping class ratios as close as the groups allow.

    Whole groups move together so no photograph (or near-duplicate export of it)
    can appear on both sides of a split boundary.
    """
    by_class = defaultdict(lambda: defaultdict(list))
    for record in records:
        by_class[record['label']][record['group']].append(record)

    train, val, test = [], [], []
    for label in sorted(by_class):
        groups = list(by_class[label])
        rng = random.Random(seed + label * 7919)
        rng.shuffle(groups)
        if len(groups) < 3:
            raise ValueError(
                f'Class {CLASSES[label]} has only {len(groups)} independent '
                'groups; need at least 3 for a leakage-safe split.'
            )
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


def assert_no_studio_in_eval(val_records, test_records):
    """Hard guard: the domain gap must be measured, never exploited."""
    for split_name, split in (('validation', val_records), ('test', test_records)):
        offenders = [r for r in split if r['domain'] != 'field']
        if offenders:
            raise AssertionError(
                f'{len(offenders)} non-field images reached the {split_name} '
                'split. Studio data is training-only by construction.'
            )


def assert_disjoint_groups(train_records, val_records, test_records):
    train_groups = {r['group'] for r in train_records}
    val_groups = {r['group'] for r in val_records}
    test_groups = {r['group'] for r in test_records}
    for left_name, left, right_name, right in (
        ('train', train_groups, 'val', val_groups),
        ('train', train_groups, 'test', test_groups),
        ('val', val_groups, 'test', test_groups),
    ):
        overlap = left & right
        if overlap:
            raise AssertionError(
                f'{len(overlap)} source groups shared between {left_name} and '
                f'{right_name}: {sorted(overlap)[:5]}'
            )


def build_dataset(ff, records, augment: bool):
    """Wrap records in a torch Dataset producing the pipeline's tensor format."""
    import torch
    import torchvision.transforms as T
    from PIL import Image
    from torch.utils.data import Dataset

    to_tensor = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    class _CrossDomainDataset(Dataset):
        def __init__(self):
            self.records = records
            self.augment = augment

        def __len__(self):
            return len(self.records)

        def __getitem__(self, index):
            record = self.records[index]
            pixel_values = to_tensor(Image.open(record['path']).convert('RGB'))
            if self.augment:
                # Reuse the pipeline's augmentation so the recipe is identical.
                pixel_values = ff._augment_normalized_image_tensor(pixel_values, 1.0)
            labels = torch.zeros(len(CLASSES), dtype=torch.float32)
            labels[record['label']] = 1.0
            return {'pixel_values': pixel_values, 'labels': labels}

    return _CrossDomainDataset()


def class_balanced_loader(ff, dataset, records, batch_size, shuffle=True):
    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler

    if not shuffle:
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)
    counts = Counter(r['label'] for r in records)
    weights = [1.0 / counts[r['label']] for r in records]
    sampler = WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(records),
        replacement=True,
    )
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)


def evaluate(model, loader, device):
    import torch

    model.eval()
    predictions, labels, probabilities = [], [], []
    with torch.no_grad():
        for batch in loader:
            pixels = batch['pixel_values'].to(device)
            logits = model(pixel_values=pixels)['logits']
            probs = torch.softmax(logits, dim=-1)
            predictions.append(logits.argmax(dim=-1).cpu())
            probabilities.append(probs.cpu())
            labels.append(batch['labels'].argmax(dim=-1))
    predictions = torch.cat(predictions).numpy()
    labels = torch.cat(labels).numpy()
    probabilities = torch.cat(probabilities).numpy()

    per_class = []
    for index in range(len(CLASSES)):
        true_positive = int(((predictions == index) & (labels == index)).sum())
        predicted = int((predictions == index).sum())
        actual = int((labels == index).sum())
        denominator = 2 * true_positive + (predicted - true_positive) + (actual - true_positive)
        per_class.append((2.0 * true_positive / denominator) if denominator else 0.0)

    correct = int((predictions == labels).sum())
    total = int(len(labels))
    return {
        'n': total,
        'correct': correct,
        'accuracy': correct / total if total else 0.0,
        'macro_f1': float(np.mean(per_class)),
        'per_class_f1': {CLASSES[i]: float(v) for i, v in enumerate(per_class)},
        'predictions': predictions.tolist(),
        'labels': labels.tolist(),
        'probabilities': probabilities.tolist(),
    }


def train(ff, model, train_loader, val_loader, device, epochs, lr, head_lr, tag):
    """Fine-tune and keep the checkpoint with the best VALIDATION macro-F1."""
    import torch

    optimizer = torch.optim.AdamW(
        [
            {'params': model.encoder.parameters(), 'lr': lr},
            {'params': model.classifier.parameters(), 'lr': head_lr},
        ],
        weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs * max(1, len(train_loader)))
    )
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

        metrics = evaluate(model, val_loader, device)
        key = (metrics['macro_f1'], metrics['accuracy'])
        marker = ''
        if key > best_key:
            best_key = key
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            marker = '  <- best'
        print(
            f'    [{tag}] epoch {epoch + 1}/{epochs} '
            f'loss={total_loss / max(1, len(train_loader)):.4f} '
            f'val_macro={metrics["macro_f1"]:.4f} '
            f'val_acc={metrics["accuracy"]:.4f}{marker}'
        )

    model.load_state_dict(best_state)
    return model, best_key


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--field-root', default='data_final/data_final/real_dataset_sorted')
    parser.add_argument('--studio-root', default='teasickness_studio/studio_sorted')
    parser.add_argument('--pipeline', default='farmfederate.py')
    parser.add_argument('--backbone', default='microsoft/swin-tiny-patch4-window7-224')
    parser.add_argument('--epochs-field', type=int, default=15)
    parser.add_argument('--epochs-studio', type=int, default=6)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--head-lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out', default='cross_domain_results.json')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    ff = load_farmfederate(Path(args.pipeline))
    import torch

    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[Device] {device}')
    if device.type != 'cuda':
        print('[WARN] No CUDA device. This will be very slow on CPU.')

    # ---- data -------------------------------------------------------------
    field = collect(Path(args.field_root), 'field')
    studio = collect(Path(args.studio_root), 'studio')
    print(f'[Data] field  : {len(field)} images, {len({r["group"] for r in field})} source groups')
    print(f'[Data] studio : {len(studio)} images, {len({r["group"] for r in studio})} source groups')
    for name, records in (('field', field), ('studio', studio)):
        counts = Counter(CLASSES[r['label']] for r in records)
        print(f'         {name}: ' + '  '.join(f'{c}={counts[c]}' for c in CLASSES))

    field_train, field_val, field_test = grouped_stratified_split(
        field, seed=args.seed
    )
    assert_no_studio_in_eval(field_val, field_test)
    assert_disjoint_groups(field_train, field_val, field_test)
    print(
        f'[Split] field train={len(field_train)} val={len(field_val)} '
        f'test={len(field_test)}  (test is field-only, group-disjoint)'
    )

    def make_model():
        model = ff.HuggingFaceVisionClassifier(
            args.backbone, num_labels=len(CLASSES), inference_tta=False
        )
        return model.to(device)

    val_loader = class_balanced_loader(
        ff, build_dataset(ff, field_val, False), field_val, args.batch_size, shuffle=False
    )
    test_loader = class_balanced_loader(
        ff, build_dataset(ff, field_test, False), field_test, args.batch_size, shuffle=False
    )
    field_train_loader = class_balanced_loader(
        ff, build_dataset(ff, field_train, True), field_train, args.batch_size
    )
    studio_train_loader = class_balanced_loader(
        ff, build_dataset(ff, studio, True), studio, args.batch_size
    )

    results = {
        'question': 'Does studio-photographed public tea data transfer to field images?',
        'classes': CLASSES,
        'backbone': args.backbone,
        'seed': args.seed,
        'test_is_field_only': True,
        'studio_barred_from_eval': True,
        'field_counts': dict(Counter(CLASSES[r['label']] for r in field)),
        'studio_counts': dict(Counter(CLASSES[r['label']] for r in studio)),
        'split_sizes': {
            'train': len(field_train), 'val': len(field_val), 'test': len(field_test)
        },
        'conditions': {},
    }

    # ---- A: field only ----------------------------------------------------
    print('\n=== A. field_only ===')
    model, val_key = make_model(), None
    model, val_key = train(
        ff, model, field_train_loader, val_loader, device,
        args.epochs_field, args.lr, args.head_lr, 'field_only',
    )
    results['conditions']['field_only'] = {
        **evaluate(model, test_loader, device),
        'val_macro_f1': val_key[0],
    }
    del model

    # ---- B: studio only (zero-shot to field) ------------------------------
    print('\n=== B. studio_only (zero-shot to field) ===')
    model = make_model()
    model, val_key = train(
        ff, model, studio_train_loader, val_loader, device,
        args.epochs_studio, args.lr, args.head_lr, 'studio_only',
    )
    results['conditions']['studio_only'] = {
        **evaluate(model, test_loader, device),
        'val_macro_f1': val_key[0],
    }
    studio_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    del model

    # ---- C: studio pretrain -> field fine-tune ----------------------------
    print('\n=== C. studio_then_field ===')
    model = make_model()
    model.load_state_dict(studio_state)
    model, val_key = train(
        ff, model, field_train_loader, val_loader, device,
        args.epochs_field, args.lr, args.head_lr, 'studio_then_field',
    )
    results['conditions']['studio_then_field'] = {
        **evaluate(model, test_loader, device),
        'val_macro_f1': val_key[0],
    }
    del model

    # ---- statistics -------------------------------------------------------
    for name, condition in results['conditions'].items():
        low, high = ff.wilson_score_interval(condition['correct'], condition['n'])
        condition['accuracy_ci95'] = [float(low), float(high)]

    reference_labels = results['conditions']['field_only']['labels']
    comparisons = []
    names = list(results['conditions'])
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            a = results['conditions'][left]
            b = results['conditions'][right]
            assert a['labels'] == b['labels'] == reference_labels, 'label order drift'
            comparisons.append({
                'a': left, 'b': right,
                'macro_f1_delta': a['macro_f1'] - b['macro_f1'],
                'accuracy_delta': a['accuracy'] - b['accuracy'],
                'mcnemar_exact_p': ff.exact_mcnemar_pvalue(
                    a['labels'], a['predictions'], b['predictions']
                ),
            })
    results['paired_comparisons'] = comparisons

    Path(args.out).write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')

    print('\n' + '=' * 78)
    print(f'CROSS-DOMAIN RESULTS  (field-only test, n={results["conditions"]["field_only"]["n"]})')
    print('=' * 78)
    print(f'{"condition":22s} {"macro-F1":>9s} {"accuracy":>9s} {"95% CI":>18s}')
    for name, condition in results['conditions'].items():
        low, high = condition['accuracy_ci95']
        print(
            f'{name:22s} {condition["macro_f1"]:9.4f} {condition["accuracy"]:9.4f} '
            f'{f"[{low:.3f}, {high:.3f}]":>18s}'
        )
    print('\nper-class F1:')
    for name, condition in results['conditions'].items():
        rendered = '  '.join(f'{k}={v:.3f}' for k, v in condition['per_class_f1'].items())
        print(f'  {name:22s} {rendered}')
    print('\npaired exact McNemar:')
    for row in comparisons:
        print(
            f'  {row["a"]:20s} vs {row["b"]:20s} '
            f'Δmacro={row["macro_f1_delta"]:+.4f}  p={row["mcnemar_exact_p"]:.4f}'
        )
    print(f'\nSaved: {args.out}')


if __name__ == '__main__':
    main()
