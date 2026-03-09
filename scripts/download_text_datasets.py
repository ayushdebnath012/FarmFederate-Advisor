"""
Download real text datasets (AG News, PubMed, SQUAD) and save to data/{stress_type}/text.csv
"""
import os
import csv
import sys

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

STRESS_LABELS = ['water_stress', 'nutrient_def', 'pest_risk', 'disease_risk', 'heat_stress']

STRESS_KEYWORDS = {
    'water_stress': {
        'high': ['drought', 'water stress', 'wilting', 'desiccation', 'dehydration', 'water deficit'],
        'medium': ['irrigation', 'moisture', 'dry soil', 'rainfall', 'water scarcity', 'hydraulic'],
        'low': ['water', 'dry', 'thirst', 'arid', 'precipitation'],
    },
    'nutrient_def': {
        'high': ['nutrient deficiency', 'nitrogen deficiency', 'phosphorus deficiency', 'potassium deficiency', 'chlorosis'],
        'medium': ['fertilizer', 'nutrient', 'mineral', 'soil fertility', 'micronutrient', 'macronutrient'],
        'low': ['nitrogen', 'phosphorus', 'potassium', 'iron', 'zinc', 'soil'],
    },
    'pest_risk': {
        'high': ['pest infestation', 'insect damage', 'pest attack', 'herbivore', 'insect pest'],
        'medium': ['aphid', 'beetle', 'caterpillar', 'mite', 'larvae', 'weevil', 'thrips'],
        'low': ['pest', 'insect', 'bug', 'worm', 'fly', 'moth'],
    },
    'disease_risk': {
        'high': ['plant disease', 'fungal infection', 'bacterial infection', 'viral disease', 'pathogen attack'],
        'medium': ['blight', 'mildew', 'rust disease', 'rot', 'lesion', 'necrosis', 'wilt disease'],
        'low': ['disease', 'fungus', 'bacteria', 'virus', 'infection', 'pathogen'],
    },
    'heat_stress': {
        'high': ['heat stress', 'thermal stress', 'heat wave', 'high temperature stress', 'thermal damage'],
        'medium': ['temperature stress', 'sunburn', 'heat injury', 'thermal injury', 'canopy temperature'],
        'low': ['heat', 'temperature', 'hot', 'warm', 'scorch', 'burn'],
    },
}

AG_KEYWORDS = [
    'plant disease', 'crop disease', 'phytopathology', 'plant pathogen',
    'fungal disease', 'bacterial disease', 'viral disease', 'leaf spot',
    'blight', 'rust', 'mildew', 'wilt', 'canker', 'rot', 'scab',
    'anthracnose', 'septoria', 'fusarium', 'botrytis', 'phytophthora',
    'disease diagnosis', 'disease detection', 'disease identification',
    'pathogen detection', 'crop health monitoring', 'disease symptom',
    'irrigation', 'drought', 'fertilizer', 'nutrient', 'pest', 'heat stress',
    'water stress', 'moisture', 'agricultural', 'farming', 'crops',
]

DATASETS_TO_TRY = [
    ('ag_news',                   None, 'train', 'text',    'AGNews'),
    ('ccdv/pubmed-summarization', None, 'train', 'article', 'PubMed'),
    ('squad',                     None, 'train', 'context', 'SQUAD'),
]

# Target per stress class
TARGET_PER_CLASS = 500


def classify_text(text_lower: str, fallback_idx: int) -> str:
    scores = {}
    for stress, groups in STRESS_KEYWORDS.items():
        s = 0
        for kw in groups['high']:
            if kw in text_lower:
                s += 3
        for kw in groups['medium']:
            if kw in text_lower:
                s += 2
        for kw in groups['low']:
            if kw in text_lower:
                s += 1
        scores[stress] = s
    max_score = max(scores.values())
    if max_score > 0:
        return max(scores, key=scores.get)
    return STRESS_LABELS[fallback_idx % len(STRESS_LABELS)]


def main():
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' package not installed. Run: pip install datasets")
        sys.exit(1)

    # buckets: stress -> list of (text, label_idx, source)
    buckets = {s: [] for s in STRESS_LABELS}
    total_scanned = 0
    total_accepted = 0

    for ds_name, ds_config, ds_split, text_col, source_tag in DATASETS_TO_TRY:
        print(f"\n[Dataset] {ds_name} ({source_tag})")
        try:
            if ds_config:
                ds = load_dataset(ds_name, ds_config, split=ds_split, streaming=True)
            else:
                ds = load_dataset(ds_name, split=ds_split, streaming=True)
        except Exception as e:
            print(f"  FAILED to load: {e}")
            continue

        count = 0
        for item in ds:
            text = item.get(text_col, '') or item.get('text', '') or ''
            if not text or len(text) < 50:
                continue

            total_scanned += 1
            text_lower = text.lower()

            if not any(kw in text_lower for kw in AG_KEYWORDS):
                continue

            stress = classify_text(text_lower, count)

            # Skip if this class is already full
            if len(buckets[stress]) >= TARGET_PER_CLASS:
                # Try to place in another non-full class
                for alt in STRESS_LABELS:
                    if len(buckets[alt]) < TARGET_PER_CLASS:
                        stress = alt
                        break
                else:
                    # All classes full
                    break

            text_trunc = text[:300] if len(text) > 300 else text
            label_idx = STRESS_LABELS.index(stress)
            buckets[stress].append((text_trunc, label_idx, f"{ds_name}_{source_tag}"))
            count += 1
            total_accepted += 1

            filled = sum(1 for s in STRESS_LABELS if len(buckets[s]) >= TARGET_PER_CLASS)
            if filled == len(STRESS_LABELS):
                print(f"  All classes full after {count} accepted from this source.")
                break

            if count % 100 == 0:
                status = {s: len(buckets[s]) for s in STRESS_LABELS}
                print(f"  Accepted {count} so far | {status}")

        status = {s: len(buckets[s]) for s in STRESS_LABELS}
        print(f"  Done: accepted {count} from {ds_name} | running totals: {status}")

    # Report
    print(f"\n=== Download complete ===")
    print(f"Total scanned: {total_scanned:,} | Total accepted: {total_accepted:,}")
    for s in STRESS_LABELS:
        print(f"  {s}: {len(buckets[s])} samples")

    # Save to data/{stress_type}/text.csv
    print("\n[Saving] Writing CSV files...")
    for stress in STRESS_LABELS:
        out_path = os.path.join(DATA_DIR, stress, 'text.csv')
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        rows = buckets[stress]
        with open(out_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['text', 'labels', 'label_name', 'source'])
            for text, label_idx, source in rows:
                writer.writerow([text, f"[{label_idx}]", stress, source])
        print(f"  Saved {len(rows)} rows -> {out_path}")

    print("\nDone.")


if __name__ == '__main__':
    main()
