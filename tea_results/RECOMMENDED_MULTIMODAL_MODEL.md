# Recommended FarmFederate Multimodal Model

Use the unguarded v6 vision-improved run:

- Checkpoint:
  `multimodal_v6_vision_full_20260727/models/best_vlm.pt`
- Evaluation:
  `multimodal_v6_vision_full_20260727/VISION_IMPROVEMENT_REPORT.md`
- Machine-readable results:
  `multimodal_v6_vision_full_20260727/complete_results.json`

Primary locked-test result: 92.00% paired accuracy and 61.33% image-only
accuracy on 75 crops from 38 source images. Pairwise
train/validation/test source overlap is zero.

For image-only inference, the optional validation-selected five-expert visual
ensemble improves the fixed internal-test result to **72.00% accuracy
(54/75)** and **0.6759 macro-F1**:

- Report:
  `multimodal_v6_vision_full_20260727/IMAGE70_ATTEMPT_REPORT.md`
- Machine-readable gate and predictions:
  `multimodal_v6_vision_full_20260727/ensemble5_vit/vision_ensemble_results.json`
- Independent ViT specialist audit:
  `multimodal_v6_vision_full_20260727/vit_specialist/vit_visual_specialist_results.json`

This crosses the requested threshold without rounding: 53/75 is the first
attainable score above 70%, and the ensemble correctly classifies 54/75.

## Exploratory sparse-field-note hierarchy

A separate target-shortcut-masked field-note robustness benchmark retains 50%
of deterministic text content and produces the requested accuracy ordering:

| System | Accuracy |
|---|---:|
| Sparse text only | 64.00% |
| Enhanced image only | 72.00% |
| Text + image fusion | 78.67% |
| Proposed reliability-aware multimodal | 81.33% |

Report:
`multimodal_v6_vision_full_20260727/modality_hierarchy_exploratory/MODALITY_HIERARCHY_REPORT.md`

This table is exploratory and does not replace the standard ablation. The
final rule is validation-selected and does not read test labels, but the
internal test partition was observed while refining the protocol and therefore
is not pristine external evidence.

Do not use the guarded v4.1 checkpoint as the reported model. Its 1.000
validation result was produced by a confidence override selected on
target-derived internal captions and is retained only for audit history.

Do not promote the v8 layer-4 resume run either. It produced a perfect paired
internal-test score but only 60.00% image-only accuracy, so it fails the
image-only objective and is not the realistic result requested.
