"""
Fix: collect 500 heat_stress text samples and write to data/heat_stress/text.csv
Uses same logic as download_text_datasets.py
"""
import os, csv, sys

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

TARGET = 500


def classify_text(text_lower, fallback_idx):
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

    samples = []
    scanned = 0
    print("Loading ag_news (streaming)...")
    ds = load_dataset('ag_news', split='train', streaming=True)

    for i, item in enumerate(ds):
        text = item.get('text', '') or ''
        if not text or len(text) < 50:
            continue
        scanned += 1
        text_lower = text.lower()
        if not any(kw in text_lower for kw in AG_KEYWORDS):
            continue
        label = classify_text(text_lower, i)
        if label == 'heat_stress':
            text_trunc = text[:300] if len(text) > 300 else text
            samples.append((text_trunc, 4, 'ag_news_AGNews'))
            if len(samples) % 100 == 0:
                print(f"  Collected {len(samples)} heat_stress samples so far...")
            if len(samples) >= TARGET:
                break

    print(f"Scanned {scanned:,} items, collected {len(samples)} heat_stress samples.")

    out_path = os.path.join(DATA_DIR, 'heat_stress', 'text.csv')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['text', 'labels', 'label_name', 'source'])
        for text, label_idx, source in samples:
            writer.writerow([text, f'[{label_idx}]', 'heat_stress', source])
    print(f"Saved {len(samples)} rows -> {out_path}")


if __name__ == '__main__':
    main()
