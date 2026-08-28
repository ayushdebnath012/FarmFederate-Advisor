# Guarded Fusion Evaluation

- Derived from: `tea_results\multimodal_v4_resnet50_full_20260727\models\best_vlm.pt`
- No retraining; the guard adds no parameters.
- Validation: 75 crops / 38 source images, zero source overlap.
- Exact image-text pair coverage: 100%.

| Condition | Macro-F1 | Accuracy | NLL | ECE |
|---|---:|---:|---:|---:|
| Paired | 1.0000 | 1.0000 | 0.0813 | 0.0739 |
| Text only | 1.0000 | 1.0000 | 0.0854 | 0.0766 |
| Image only | 0.5257 | 0.6133 | 0.9574 | 0.1044 |
| Mismatched text | 0.1038 | 0.1333 | 4.5233 | 0.7912 |

- Fusion gain over best unimodal path: `+0.0000`.
- Mismatch drop: `+0.8962`.
- The 0.60 guard was selected on this validation split and requires external calibration before production use.
- Image-only behavior is unchanged by design.
