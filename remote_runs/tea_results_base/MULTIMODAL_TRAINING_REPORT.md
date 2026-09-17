# FarmFederate Multimodal Training Report

- Generated: 2026-09-17T19:51:35.108855+00:00
- Run type: **full_training**
- Device: `cuda`
- Split: source-image grouped, validation fraction `20%`, locked-test fraction `20%`
- Pairwise train/validation/test source-image overlap: `0`
- Exact train pair coverage: `100.0%`
- Exact validation pair coverage: `100.0%`
- Exact test pair coverage: `100.0%`
- Training-fitted target-shortcut tokens masked: `45`

## Locked-test results

| Model | Selected validation F1 | Test macro-F1 | Test accuracy | Test ECE | Federated validation F1 |
|---|---:|---:|---:|---:|---:|
| Text | 0.5925 | 0.4401 | 0.5733 | 0.1092 | 0.5655 |
| Image | 0.5005 | 0.3687 | 0.3867 | 0.2106 | 0.4062 |
| Multimodal | 0.7019 | 0.5243 | 0.6133 | 0.0601 | 0.5864 |

## Locked-test cross-modal analysis

| Condition | Macro-F1 | Accuracy | NLL | ECE |
|---|---:|---:|---:|---:|
| Correctly paired | 0.5243 | 0.6133 | 1.0973 | 0.0601 |
| Text only | 0.3106 | 0.3867 | 1.4589 | 0.1170 |
| Image only | 0.5774 | 0.6133 | 1.2491 | 0.1046 |
| Mismatched text | 0.5215 | 0.6133 | 1.1294 | 0.0970 |

- Fusion gain over best unimodal path: `-0.0531` macro-F1.
- Drop after mismatching the text: `+0.0028` macro-F1.
- Text→image class Recall@1: `0.3067`.
- Image→text class Recall@1: `0.2800`.
- Paired-vs-rolled cosine margin: `+0.0020`.

## Interpretation guardrails

- Model selection uses macro-F1, not micro-F1/accuracy, because the disease classes are imbalanced.
- Exact image/box text pairs are used; same-class random pairing is only a fallback for missing annotations.
- All boxes from a source image remain in exactly one of train, validation, or locked test.
- Validation selects the checkpoint; the locked test is evaluated only after training.
- Class-exclusive caption tokens are learned from training annotations only and masked in every partition.
- The fused head includes a fixed reliability-weighted residual from the two auxiliary modality experts.
- Cross-modal benefit requires paired performance to exceed both unimodal ablations and to degrade under mismatching.
- A category-best claim additionally requires repeated seeds and an external held-out dataset; this report does not infer either.
