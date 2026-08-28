# FarmFederate Multimodal Training Report

- Generated: 2026-07-26T17:36:19.666537+00:00
- Run type: **smoke_test**
- Device: `cpu`
- Split: source-image grouped, validation fraction `20%`
- Train/validation source-image overlap: `0`
- Exact train pair coverage: `100.0%`
- Exact validation pair coverage: `100.0%`

## Validation results

| Model | Central macro-F1 | Accuracy | ECE | Federated macro-F1 |
|---|---:|---:|---:|---:|
| Multimodal | 0.0741 | 0.2273 | 0.0076 | not run |

## Cross-modal analysis

| Condition | Macro-F1 | Accuracy | NLL | ECE |
|---|---:|---:|---:|---:|
| Correctly paired | 0.0741 | 0.2273 | 1.6104 | 0.0076 |
| Text only | 0.2242 | 0.3636 | 1.5909 | 0.1492 |
| Image only | 0.0741 | 0.2273 | 1.5933 | 0.0094 |
| Mismatched text | 0.0741 | 0.2273 | 1.6192 | 0.0072 |

- Fusion gain over best unimodal path: `-0.1502` macro-F1.
- Drop after mismatching the text: `+0.0000` macro-F1.
- Text→image class Recall@1: `0.1818`.
- Image→text class Recall@1: `0.1818`.
- Paired-vs-rolled cosine margin: `+0.0006`.

## Interpretation guardrails

- Model selection uses macro-F1, not micro-F1/accuracy, because the disease classes are imbalanced.
- Exact image/box text pairs are used; same-class random pairing is only a fallback for missing annotations.
- All boxes from a source image remain in one partition.
- Cross-modal benefit requires paired performance to exceed both unimodal ablations and to degrade under mismatching.
- This was a short smoke run. Its metrics verify the pipeline but do not establish a state-of-the-art claim.
