# ViT + Language-Model Encoder Selection

The architecture decision used validation accuracy only, with macro-F1 as the first tie-break. Candidate test features and predictions were computed only after the candidate configuration had been frozen; the incumbent test artifact was reused unchanged and never entered selection.

## Result

- Selected architecture: `resnet50_compact_transformer` (higher validation accuracy).
- Incumbent validation: 0.8919 accuracy, 0.8541 macro-F1, 0.8692 score.
- ViT + DistilBERT validation: 0.8243 accuracy, 0.6892 macro-F1, 0.7433 score.
- Incumbent locked test: 0.8133 accuracy, 0.6679 macro-F1.
- ViT + DistilBERT locked test: 0.8267 accuracy, 0.8192 macro-F1.
- Paired test discordance (candidate-only/incumbent-only): `11/10`; exact two-sided McNemar p=`1.0000`.

## Selected ViT + DistilBERT candidate

- PCA components per modality: `64`
- Fusion recipe: `concatenation`
- Image scale: `1.0`
- Ridge alpha: `100.0`
- Class-weight power: `0.5`
- Test-time image views: `['identity']`
- Validation configurations checked: `360`

## Audit

- Split: `{'train': 222, 'validation': 74, 'test': 75}` crops
- Source groups: `{'train': 122, 'validation': 40, 'test': 38}`
- Source-group overlap: `{'train_validation': 0, 'train_test': 0, 'validation_test': 0}`
- Sparse-note keep rate: `0.5`
- Test used for selection: `False`
- Both pretrained encoders were loaded from the local model cache and remained frozen.
