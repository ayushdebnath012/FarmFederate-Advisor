# FarmFederate Multimodal Training Report

- Generated: 2026-07-26T21:13:41.445369+00:00
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
| Multimodal | 0.9099 | 0.8753 | 0.9200 | 0.2648 | not run |

## Locked-test cross-modal analysis

| Condition | Macro-F1 | Accuracy | NLL | ECE |
|---|---:|---:|---:|---:|
| Correctly paired | 0.8753 | 0.9200 | 0.5315 | 0.2648 |
| Text only | 0.9049 | 0.9600 | 0.2733 | 0.1463 |
| Image only | 0.5431 | 0.6133 | 1.1828 | 0.1135 |
| Mismatched text | 0.2804 | 0.3333 | 1.7254 | 0.2193 |

- Fusion gain over best unimodal path: `-0.0296` macro-F1.
- Drop after mismatching the text: `+0.5949` macro-F1.
- Text→image class Recall@1: `0.6000`.
- Image→text class Recall@1: `0.6133`.
- Paired-vs-rolled cosine margin: `+0.0413`.

## Interpretation guardrails

- Model selection uses macro-F1, not micro-F1/accuracy, because the disease classes are imbalanced.
- Exact image/box text pairs are used; same-class random pairing is only a fallback for missing annotations.
- All boxes from a source image remain in exactly one of train, validation, or locked test.
- Validation selects the checkpoint; the locked test is evaluated only after training.
- Class-exclusive caption tokens are learned from training annotations only and masked in every partition.
- The fused head includes a fixed reliability-weighted residual from the two auxiliary modality experts.
- Cross-modal benefit requires paired performance to exceed both unimodal ablations and to degrade under mismatching.
- A category-best claim additionally requires repeated seeds and an external held-out dataset; this report does not infer either.
