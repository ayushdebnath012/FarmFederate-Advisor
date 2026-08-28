# FarmFederate Realistic Multimodal Evaluation

> **Superseded by v6:** The evaluation protocol remains valid, but the
> vision-improved v6 run now provides the recommended checkpoint. See
> `../multimodal_v6_vision_full_20260727/VISION_IMPROVEMENT_REPORT.md`.

## Recommended result

The recommended unguarded model achieved **86.67% accuracy** and **0.7810
macro-F1** on a locked, source-image-grouped test set. It correctly classified
65 of 75 test crops.

A 5,000-resample source-group bootstrap gives:

- accuracy 95% interval: 79.17% to 93.06%;
- macro-F1 95% interval: 0.6619 to 0.8859.

These intervals are wide because the test set contains only 38 independent
source images. The result is useful internal evidence, not a state-of-the-art
claim.

## Why the previous 1.000 result was rejected

The earlier result used a parameter-free confidence guard selected on the same
internal validation set. It copied a confident text-only prediction over the
paired prediction. The captions were generated after the disease label was
known and contain many class-specific lexical patterns, so the guard amplified
a text shortcut rather than demonstrating better multimodal generalization.

The confidence guard is now disabled by default and is absent from the
recommended checkpoint.

## Corrected protocol

- Train: 222 crops from 122 source images.
- Validation: 74 crops from 40 source images.
- Locked test: 75 crops from 38 source images.
- Pairwise source-image overlap: 0.
- Exact image-box/text coverage: 100% in every partition.
- The validation split alone selected epoch 14.
- The locked test was evaluated after checkpoint selection.
- 178 class-exclusive caption tokens were identified from training
  annotations only and masked in train, validation, and test.
- The ImageNet-pretrained ResNet-50 backbone remained frozen.
- No confidence threshold or post-hoc override was tuned on test results.

## Locked-test cross-modal analysis

| Condition | Macro-F1 | Accuracy | NLL | ECE |
|---|---:|---:|---:|---:|
| Correctly paired | 0.7810 | 0.8667 | 0.5539 | 0.2264 |
| Text only | 0.9125 | 0.9600 | 0.2239 | 0.1233 |
| Image only | 0.4747 | 0.5200 | 1.3075 | 0.1238 |
| Mismatched text | 0.3243 | 0.3733 | 1.6512 | 0.1687 |

The paired model drops by 0.4567 macro-F1 when text is deliberately
mismatched, so the modalities interact. However, paired macro-F1 is 0.1314
below text-only macro-F1. This dataset therefore does not yet demonstrate
positive fusion gain.

Text-only accuracy remains an assisted upper bound, not a field deployment
metric: even after token masking, all captions were originally generated from
known labels and retain class-correlated combinations. The paired locked-test
score is the primary reported result, while image-only measures performance
when no trustworthy text is available.

Retrieval diagnostics:

- text-to-image class Recall@1: 0.7067;
- image-to-text class Recall@1: 0.4933;
- paired-versus-rolled cosine margin: +0.0478.

## Per-class paired F1

| Class | F1 |
|---|---:|
| gray_blight | 0.8800 |
| helopeltis | 0.3333 |
| algal_leaf_spot | 0.8571 |
| brown_blight | 0.8718 |
| red_leaf_spot | 0.9630 |

Helopeltis remains the largest uncertainty: there are only nine total crops,
including two in the locked test. More genuine, provenance-checked source
images are required; resampling these nine crops cannot establish robust
generalization.

## Verification

- `tea_train.py` compiles.
- Seven regression checks pass, including three-way group isolation and
  training-only text-shortcut fitting.
- The saved checkpoint reloads with `strict=True`; all keys match.
- Checkpoint metadata records `text_confidence_guard=None`.

## Claim boundary

This report deliberately does not claim category-best performance. A strong
external claim requires independently collected image-text pairs, more
Helopeltis source images, repeated grouped seeds or nested cross-validation,
and calibration on data that was not used for model or threshold selection.
