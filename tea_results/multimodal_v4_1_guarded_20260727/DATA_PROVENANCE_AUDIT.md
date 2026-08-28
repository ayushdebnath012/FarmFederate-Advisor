# Additional Image Data Provenance Audit

## Corrected legacy-folder crosswalk

The repository's legacy classification folder names do not match the corrected
OBB disease taxonomy. Filename-level comparison against all 371 corrected
annotations produced the following one-to-one crosswalk:

| Legacy folder | Corrected class | Original images | Annotation conflicts under legacy name |
|---|---|---:|---:|
| `LEAF_BLIGHT` | `gray_blight` | 62 | 62/62 |
| `LEAF_HOPPERS` | `helopeltis` | 8 | 8/8 |
| `LEAF_RUST` | `algal_leaf_spot` | 35 | 35/35 |
| `LOOPER_CATERPILLARS` | `brown_blight` | 48 | 48/48 |
| `MOSQUITO_BUG` | `red_leaf_spot` | 47 | 47/47 |

The same obsolete semantics are inherited by `Augmented-Dataset`, which has
400 files in each legacy folder.

## Inclusion decision

The augmented files were excluded from the reported training run because:

- there is no augmentation manifest mapping each derived image to its source;
- source images are shared with the grouped OBB dataset;
- without lineage, augmented versions of validation images could enter
  training and create leakage;
- treating the legacy folder names literally would invert the corrected class
  mapping.

The original classification folders add no independent source groups: their
filenames overlap the corrected OBB image set. They therefore do not resolve
the Helopeltis scarcity.

Two untracked `ref_*` images exist per legacy folder, but their origin and
license are not documented, so they were not added automatically.

## Required remediation

Before using `Augmented-Dataset`, reconstruct or regenerate it with a manifest
containing:

1. source image filename and hash;
2. corrected disease class;
3. augmentation parameters and seed;
4. train/validation group inherited from the source image;
5. provenance and license for every non-derived reference image.

This preserves the zero source-image overlap guarantee.
