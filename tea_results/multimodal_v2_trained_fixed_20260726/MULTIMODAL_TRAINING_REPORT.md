# FarmFederate Multimodal Training Report

- Generated: 2026-07-26T17:40:22.763693+00:00
- Run type: **smoke_test**
- Device: `cpu`
- Split: source-image grouped, validation fraction `20%`
- Train/validation source-image overlap: `0`
- Exact train pair coverage: `100.0%`
- Exact validation pair coverage: `100.0%`

## Validation results

| Model | Central macro-F1 | Accuracy | ECE | Federated macro-F1 |
|---|---:|---:|---:|---:|
| Multimodal | 0.7423 | 0.7273 | 0.2239 | not run |

## Cross-modal analysis

| Condition | Macro-F1 | Accuracy | NLL | ECE |
|---|---:|---:|---:|---:|
| Correctly paired | 0.7423 | 0.7273 | 0.8589 | 0.2239 |
| Text only | 0.8489 | 0.8636 | 0.9529 | 0.4312 |
| Image only | 0.3359 | 0.3636 | 1.5078 | 0.1868 |
| Mismatched text | 0.1689 | 0.1364 | 1.5654 | 0.3419 |

- Fusion gain over best unimodal path: `-0.1066` macro-F1.
- Drop after mismatching the text: `+0.5734` macro-F1.
- Text→image class Recall@1: `0.3636`.
- Image→text class Recall@1: `0.3636`.
- Paired-vs-rolled cosine margin: `+0.0446`.

## Interpretation guardrails

- Model selection uses macro-F1, not micro-F1/accuracy, because the disease classes are imbalanced.
- Exact image/box text pairs are used; same-class random pairing is only a fallback for missing annotations.
- All boxes from a source image remain in one partition.
- Cross-modal benefit requires paired performance to exceed both unimodal ablations and to degrade under mismatching.
- This was a short smoke run. Its metrics verify the pipeline but do not establish a state-of-the-art claim.
