# FarmFederate Multimodal Training Report

- Generated: 2026-07-26T22:05:10.279403+00:00
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
| Multimodal | 0.8965 | 0.8451 | 0.8800 | 0.1848 | not run |

## Locked-test cross-modal analysis

| Condition | Macro-F1 | Accuracy | NLL | ECE |
|---|---:|---:|---:|---:|
| Correctly paired | 0.8451 | 0.8800 | 0.4975 | 0.1848 |
| Text only | 0.8966 | 0.9467 | 0.2245 | 0.1212 |
| Image only | 0.5434 | 0.6533 | 1.1613 | 0.0918 |
| Mismatched text | 0.2890 | 0.3467 | 1.7553 | 0.2398 |

- Fusion gain over best unimodal path: `-0.0515` macro-F1.
- Drop after mismatching the text: `+0.5561` macro-F1.
- Text→image class Recall@1: `0.5733`.
- Image→text class Recall@1: `0.6133`.
- Paired-vs-rolled cosine margin: `+0.0459`.

## Interpretation guardrails

- Model selection uses macro-F1, not micro-F1/accuracy, because the disease classes are imbalanced.
- Exact image/box text pairs are used; same-class random pairing is only a fallback for missing annotations.
- All boxes from a source image remain in exactly one of train, validation, or locked test.
- Validation selects the checkpoint; the locked test is evaluated only after training.
- Class-exclusive caption tokens are learned from training annotations only and masked in every partition.
- The fused head includes a fixed reliability-weighted residual from the two auxiliary modality experts.
- Cross-modal benefit requires paired performance to exceed both unimodal ablations and to degrade under mismatching.
- A category-best claim additionally requires repeated seeds and an external held-out dataset; this report does not infer either.
