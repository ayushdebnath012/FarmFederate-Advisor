#!/usr/bin/env python3
"""Build a domain-tagged studio bundle from the TeaSickness dataset.

The bundle is DELIBERATELY named ``studio_sorted`` rather than
``real_dataset_sorted`` so it can never be mistaken for the 200 in-situ field
photographs by an existing loader. Studio images are a different imaging domain
(detached leaf on a plain backdrop, 256x256) and must only ever enter TRAINING;
validation and test must remain field-only or the domain gap is exploited
instead of measured.

Mapping decisions and their justification are recorded in the manifest so a
reader can audit every class assignment.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

# Source class -> (target class in the FarmFederate schema, justification, confidence)
MAPPING = {
    'brown_blight': (
        'LEAF_BLIGHT',
        'Brown blight is a foliar blight of tea (Colletotrichum spp.); same '
        'symptom category as the target class.',
        'solid',
    ),
    'gray_blight': (
        'LEAF_BLIGHT',
        'Gray blight (Pestalotiopsis spp.) is a foliar blight of tea; merged '
        'with brown blight into the single LEAF_BLIGHT target.',
        'solid',
    ),
    'helopeltis': (
        'MOSQUITO_BUG',
        'Helopeltis theivora IS the tea mosquito bug. Same organism, different '
        'common name. Lesion morphology (dark punctate feeding wounds) matches '
        'the field MOSQUITO_BUG images.',
        'exact',
    ),
    'algal_spot': (
        'LEAF_RUST',
        'Tea "red rust" is caused by the alga Cephaleuros, the same pathogen '
        'genus as algal leaf spot. Visual check confirmed matching morphology: '
        'discrete small tan/orange circular lesions scattered on the blade, as '
        'in the field LEAF_RUST images.',
        'defensible-with-caveat',
    ),
}

# Explicitly excluded, with the reason, so the omission is auditable.
EXCLUDED = {
    'red_spot': (
        'Presents as large-area whole-leaf reddening/browning, not the discrete '
        'small spots seen in field LEAF_RUST. Visually distinct; lumping it in '
        'would inflate the class with a different condition.'
    ),
    'healthy': 'No corresponding target class in the FarmFederate schema.',
}

# Classes in the FarmFederate schema with NO studio source available.
NO_STUDIO_SOURCE = {
    'LOOPER_CATERPILLARS': 'No caterpillar/defoliation class in TeaSickness.',
    'LEAF_HOPPERS': 'Dropped from the benchmark; images show no leafhopper symptoms.',
}


def average_hash(path: Path, size: int = 12) -> str:
    """Perceptual hash used to group near-duplicate leaves into one source."""
    image = Image.open(path).convert('L').resize((size, size), Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32)
    return hashlib.md5((array > array.mean()).tobytes()).hexdigest()[:16]


def build(source_root: Path, output_root: Path) -> dict:
    source_root = Path(source_root)
    output_root = Path(output_root)
    if output_root.exists():
        shutil.rmtree(output_root)
    sorted_root = output_root / 'studio_sorted'
    sorted_root.mkdir(parents=True)

    manifest = {
        'bundle_version': 'teasickness_studio_v1',
        'domain': 'studio',
        'domain_description': (
            'Detached single tea leaf photographed on a plain backdrop at '
            '256x256. NOT the same imaging domain as the 200 in-situ field '
            'photographs.'
        ),
        'usage_constraint': (
            'TRAINING ONLY. Studio images must never enter validation or test; '
            'held-out evaluation stays field-only so the domain gap is measured '
            'rather than exploited.'
        ),
        'source_dataset': 'Kaggle rohitttdx/tea-sickness-dataset (Saikat Dutta tea)',
        'mapping': {},
        'excluded_source_classes': EXCLUDED,
        'schema_classes_without_studio_source': NO_STUDIO_SOURCE,
        'per_target_counts': {},
        'source_groups': {},
    }

    per_target = defaultdict(int)
    per_target_groups = defaultdict(set)

    for source_class, (target, justification, confidence) in MAPPING.items():
        class_dir = source_root / source_class
        if not class_dir.is_dir():
            raise FileNotFoundError(f'Missing source class folder: {class_dir}')
        target_dir = sorted_root / target
        target_dir.mkdir(parents=True, exist_ok=True)

        images = sorted(
            p for p in class_dir.iterdir()
            if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}
        )
        copied = 0
        for image_path in images:
            group = average_hash(image_path)
            # Provenance is preserved in the filename: domain, source class and
            # perceptual group. The pipeline groups on the "g<hash>" token so
            # near-duplicate leaves cannot straddle a split boundary.
            safe_stem = image_path.stem.replace(' ', '_').replace('(', '').replace(')', '')
            out_name = f'studio__{source_class}__g{group}__{safe_stem}{image_path.suffix.lower()}'
            shutil.copy2(image_path, target_dir / out_name)
            per_target_groups[target].add(group)
            copied += 1

        manifest['mapping'][source_class] = {
            'target_class': target,
            'justification': justification,
            'confidence': confidence,
            'images': copied,
        }
        per_target[target] += copied

    manifest['per_target_counts'] = dict(per_target)
    manifest['source_groups'] = {
        target: len(groups) for target, groups in per_target_groups.items()
    }

    (output_root / 'mapping_manifest.json').write_text(
        json.dumps(manifest, indent=2) + '\n', encoding='utf-8'
    )
    return manifest


if __name__ == '__main__':
    import sys

    src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        'dl3/saikai dutta tea/Tea_Leaf_Disease'
    )
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('teasickness_studio')
    result = build(src, out)
    print(json.dumps(
        {k: result[k] for k in
         ('per_target_counts', 'source_groups', 'schema_classes_without_studio_source')},
        indent=2,
    ))
