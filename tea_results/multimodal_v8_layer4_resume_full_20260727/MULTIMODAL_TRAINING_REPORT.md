# FarmFederate Multimodal Training Report

- Generated: 2026-07-26T22:33:22.623359+00:00
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
| Multimodal | 0.8900 | 1.0000 | 1.0000 | 0.2756 | not run |

## Locked-test cross-modal analysis

| Condition | Macro-F1 | Accuracy | NLL | ECE |
|---|---:|---:|---:|---:|
| Correctly paired | 1.0000 | 1.0000 | 0.3502 | 0.2756 |
| Text only | 0.8790 | 0.9600 | 0.2981 | 0.1750 |
| Image only | 0.5650 | 0.6000 | 1.0838 | 0.0584 |
| Mismatched text | 0.2780 | 0.3067 | 1.6926 | 0.3445 |

- Fusion gain over best unimodal path: `+0.1210` macro-F1.
- Drop after mismatching the text: `+0.7220` macro-F1.
- Text→image class Recall@1: `0.6667`.
- Image→text class Recall@1: `0.5733`.
- Paired-vs-rolled cosine margin: `+0.0302`.

## Interpretation guardrails

- Model selection uses macro-F1, not micro-F1/accuracy, because the disease classes are imbalanced.
- Exact image/box text pairs are used; same-class random pairing is only a fallback for missing annotations.
- All boxes from a source image remain in exactly one of train, validation, or locked test.
- Validation selects the checkpoint; the locked test is evaluated only after training.
- Class-exclusive caption tokens are learned from training annotations only and masked in every partition.
- The fused head includes a fixed reliability-weighted residual from the two auxiliary modality experts.
- Cross-modal benefit requires paired performance to exceed both unimodal ablations and to degrade under mismatching.
- A category-best claim additionally requires repeated seeds and an external held-out dataset; this report does not infer either.
