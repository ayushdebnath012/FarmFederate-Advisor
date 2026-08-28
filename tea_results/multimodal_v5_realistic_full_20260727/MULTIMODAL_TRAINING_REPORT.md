# FarmFederate Multimodal Training Report

- Generated: 2026-07-26T20:49:38.476862+00:00
- Run type: **full_training**
- Device: `cpu`
- Split: source-image grouped, validation fraction `20%`, locked-test fraction `20%`
- Pairwise train/validation/test source-image overlap: `0`
- Exact train pair coverage: `100.0%`
- Exact validation pair coverage: `100.0%`
- Exact test pair coverage: `100.0%`
- Training-fitted target-shortcut tokens masked: `178`

## Locked-test results

| Model | Selected validation F1 | Test macro-F1 | Test accuracy | Test ECE | Federated validation F1 |
|---|---:|---:|---:|---:|---:|
| Multimodal | 0.8455 | 0.7810 | 0.8667 | 0.2264 | not run |

## Locked-test cross-modal analysis

| Condition | Macro-F1 | Accuracy | NLL | ECE |
|---|---:|---:|---:|---:|
| Correctly paired | 0.7810 | 0.8667 | 0.5539 | 0.2264 |
| Text only | 0.9125 | 0.9600 | 0.2239 | 0.1233 |
| Image only | 0.4747 | 0.5200 | 1.3075 | 0.1238 |
| Mismatched text | 0.3243 | 0.3733 | 1.6512 | 0.1687 |

- Fusion gain over best unimodal path: `-0.1314` macro-F1.
- Drop after mismatching the text: `+0.4567` macro-F1.
- Text→image class Recall@1: `0.7067`.
- Image→text class Recall@1: `0.4933`.
- Paired-vs-rolled cosine margin: `+0.0478`.

## Interpretation guardrails

- Model selection uses macro-F1, not micro-F1/accuracy, because the disease classes are imbalanced.
- Exact image/box text pairs are used; same-class random pairing is only a fallback for missing annotations.
- All boxes from a source image remain in exactly one of train, validation, or locked test.
- Validation selects the checkpoint; the locked test is evaluated only after training.
- Class-exclusive caption tokens are learned from training annotations only and masked in every partition.
- The fused head includes a fixed reliability-weighted residual from the two auxiliary modality experts.
- Cross-modal benefit requires paired performance to exceed both unimodal ablations and to degrade under mismatching.
- A category-best claim additionally requires repeated seeds and an external held-out dataset; this report does not infer either.
