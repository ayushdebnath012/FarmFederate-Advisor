# FarmFederate Multimodal Training Report

- Generated: 2026-07-26T17:44:23.807172+00:00
- Run type: **smoke_test**
- Device: `cpu`
- Split: source-image grouped, validation fraction `20%`
- Train/validation source-image overlap: `0`
- Exact train pair coverage: `100.0%`
- Exact validation pair coverage: `100.0%`

## Validation results

| Model | Central macro-F1 | Accuracy | ECE | Federated macro-F1 |
|---|---:|---:|---:|---:|
| Multimodal | 0.9136 | 0.9091 | 0.2846 | not run |

## Cross-modal analysis

| Condition | Macro-F1 | Accuracy | NLL | ECE |
|---|---:|---:|---:|---:|
| Correctly paired | 0.9136 | 0.9091 | 0.5708 | 0.2846 |
| Text only | 0.7985 | 0.8182 | 0.6297 | 0.2311 |
| Image only | 0.2933 | 0.3636 | 1.8746 | 0.2482 |
| Mismatched text | 0.2278 | 0.1818 | 2.4644 | 0.5117 |

- Fusion gain over best unimodal path: `+0.1152` macro-F1.
- Drop after mismatching the text: `+0.6859` macro-F1.
- Text→image class Recall@1: `0.1818`.
- Image→text class Recall@1: `0.3182`.
- Paired-vs-rolled cosine margin: `+0.0212`.

## Interpretation guardrails

- Model selection uses macro-F1, not micro-F1/accuracy, because the disease classes are imbalanced.
- Exact image/box text pairs are used; same-class random pairing is only a fallback for missing annotations.
- All boxes from a source image remain in one partition.
- The fused head includes a fixed reliability-weighted residual from the two auxiliary modality experts.
- Cross-modal benefit requires paired performance to exceed both unimodal ablations and to degrade under mismatching.
- This was a short smoke run. Its metrics verify the pipeline but do not establish a state-of-the-art claim.
