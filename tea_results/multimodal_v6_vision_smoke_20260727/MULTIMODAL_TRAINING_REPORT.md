# FarmFederate Multimodal Training Report

- Generated: 2026-07-26T21:06:19.606263+00:00
- Run type: **smoke_test**
- Device: `cpu`
- Split: source-image grouped, validation fraction `20%`, locked-test fraction `20%`
- Pairwise train/validation/test source-image overlap: `0`
- Exact train pair coverage: `100.0%`
- Exact validation pair coverage: `100.0%`
- Exact test pair coverage: `100.0%`
- Training-fitted target-shortcut tokens masked: `72`

## Locked-test results

| Model | Selected validation F1 | Test macro-F1 | Test accuracy | Test ECE | Federated validation F1 |
|---|---:|---:|---:|---:|---:|
| Multimodal | 0.2923 | 0.1342 | 0.2273 | 0.0467 | not run |

## Locked-test cross-modal analysis

| Condition | Macro-F1 | Accuracy | NLL | ECE |
|---|---:|---:|---:|---:|
| Correctly paired | 0.1342 | 0.2273 | 1.6167 | 0.0467 |
| Text only | 0.0881 | 0.1364 | 1.6464 | 0.1217 |
| Image only | 0.2082 | 0.2727 | 1.6886 | 0.1129 |
| Mismatched text | 0.1032 | 0.1818 | 1.6134 | 0.0938 |

- Fusion gain over best unimodal path: `-0.0740` macro-F1.
- Drop after mismatching the text: `+0.0311` macro-F1.
- Text→image class Recall@1: `0.1818`.
- Image→text class Recall@1: `0.1818`.
- Paired-vs-rolled cosine margin: `+0.0020`.

## Interpretation guardrails

- Model selection uses macro-F1, not micro-F1/accuracy, because the disease classes are imbalanced.
- Exact image/box text pairs are used; same-class random pairing is only a fallback for missing annotations.
- All boxes from a source image remain in exactly one of train, validation, or locked test.
- Validation selects the checkpoint; the locked test is evaluated only after training.
- Class-exclusive caption tokens are learned from training annotations only and masked in every partition.
- The fused head includes a fixed reliability-weighted residual from the two auxiliary modality experts.
- Cross-modal benefit requires paired performance to exceed both unimodal ablations and to degrade under mismatching.
- This was a short smoke run. Its metrics verify the pipeline but do not establish a state-of-the-art claim.
