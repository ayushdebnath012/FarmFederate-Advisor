"""
Utility functions for dataset reporting and offline checks.
These are pure functions and do not perform any network operations.
"""
import os
import glob
import pandas as pd
from typing import Dict


def count_images_in_dir(path: str) -> int:
    exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff')
    total = 0
    for e in exts:
        total += len(glob.glob(os.path.join(path, '**', e), recursive=True))
    return total


def build_text_label_dfs(all_text_df: pd.DataFrame, label_keywords: Dict[str, list], min_per_label: int = 150):
    """Build per-label DataFrames from a combined text DataFrame.
    Returns a dict label -> DataFrame. Synthesizes minimal rows if needed.
    """
    results = {}
    tmp = all_text_df.copy()
    tmp['text_lower'] = tmp['text'].astype(str).str.lower()
    for lbl, kws in label_keywords.items():
        pat = '|'.join(kws)
        matches = tmp[tmp['text_lower'].str.contains(pat, na=False)].drop(columns=['text_lower'])
        if len(matches) >= min_per_label:
            results[lbl] = matches.reset_index(drop=True)
        else:
            # Synthesize additional rows
            need = max(0, min_per_label - len(matches))
            synth_texts = []
            for i in range(need):
                synth_texts.append({
                    'text': f'Synthesized example for {lbl} #{i}',
                    'labels': [[list(range(1))]],
                    'dataset': f'synth_{lbl}'
                })
            synth_df = pd.DataFrame(synth_texts)
            combined = pd.concat([matches.reset_index(drop=True), synth_df], ignore_index=True)
            results[lbl] = combined
    return results


if __name__ == '__main__':
    print('dataset_utils module loaded. Use functions programmatically.')