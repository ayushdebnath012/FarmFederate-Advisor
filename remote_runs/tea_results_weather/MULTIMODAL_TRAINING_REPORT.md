# FarmFederate Multimodal Training Report

- Generated: 2026-09-17T19:53:02.706255+00:00
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
| Multimodal | 0.6868 | 0.5358 | 0.6400 | 0.0856 | 0.5671 |

## Locked-test cross-modal analysis

| Condition | Macro-F1 | Accuracy | NLL | ECE |
|---|---:|---:|---:|---:|
| Correctly paired | 0.5358 | 0.6400 | 1.0391 | 0.0856 |
| Text only | 0.3891 | 0.4933 | 1.3684 | 0.0799 |
| Image only | 0.5578 | 0.6000 | 1.2534 | 0.0929 |
| Mismatched text | 0.4862 | 0.5733 | 1.1951 | 0.0897 |

- Fusion gain over best unimodal path: `-0.0220` macro-F1.
- Drop after mismatching the text: `+0.0496` macro-F1.
- Text→image class Recall@1: `0.5333`.
- Image→text class Recall@1: `0.4000`.
- Paired-vs-rolled cosine margin: `+0.0058`.

## Interpretation guardrails

- Model selection uses macro-F1, not micro-F1/accuracy, because the disease classes are imbalanced.
- Exact image/box text pairs are used; same-class random pairing is only a fallback for missing annotations.
- All boxes from a source image remain in exactly one of train, validation, or locked test.
- Validation selects the checkpoint; the locked test is evaluated only after training.
- Class-exclusive caption tokens are learned from training annotations only and masked in every partition.
- The fused head includes a fixed reliability-weighted residual from the two auxiliary modality experts.
- Cross-modal benefit requires paired performance to exceed both unimodal ablations and to degrade under mismatching.
- A category-best claim additionally requires repeated seeds and an external held-out dataset; this report does not infer either.
