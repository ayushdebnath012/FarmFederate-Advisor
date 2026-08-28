# FarmFederate Multimodal Training Report

- Generated: 2026-07-26T19:00:15.588932+00:00
- Run type: **full_training**
- Device: `cpu`
- Split: source-image grouped, validation fraction `20%`
- Train/validation source-image overlap: `0`
- Exact train pair coverage: `100.0%`
- Exact validation pair coverage: `100.0%`

## Validation results

| Model | Central macro-F1 | Accuracy | ECE | Federated macro-F1 |
|---|---:|---:|---:|---:|
| Multimodal | 0.9259 | 0.9867 | 0.2588 | not run |

## Cross-modal analysis

| Condition | Macro-F1 | Accuracy | NLL | ECE |
|---|---:|---:|---:|---:|
| Correctly paired | 0.9259 | 0.9867 | 0.3475 | 0.2588 |
| Text only | 1.0000 | 1.0000 | 0.0854 | 0.0766 |
| Image only | 0.5257 | 0.6133 | 0.9574 | 0.1044 |
| Mismatched text | 0.3258 | 0.3733 | 1.7676 | 0.2953 |

- Fusion gain over best unimodal path: `-0.0741` macro-F1.
- Drop after mismatching the text: `+0.6001` macro-F1.
- Text→image class Recall@1: `0.5333`.
- Image→text class Recall@1: `0.5867`.
- Paired-vs-rolled cosine margin: `+0.0583`.

## Interpretation guardrails

- Model selection uses macro-F1, not micro-F1/accuracy, because the disease classes are imbalanced.
- Exact image/box text pairs are used; same-class random pairing is only a fallback for missing annotations.
- All boxes from a source image remain in one partition.
- The fused head includes a fixed reliability-weighted residual from the two auxiliary modality experts.
- Cross-modal benefit requires paired performance to exceed both unimodal ablations and to degrade under mismatching.
- A category-best claim additionally requires repeated seeds and an external held-out dataset; this report does not infer either.
