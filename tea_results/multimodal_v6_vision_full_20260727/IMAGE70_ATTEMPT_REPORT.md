# Image-Only Accuracy Improvement Report

## Result

The strongest validation-selected image-only system reached **72.00% accuracy**
(54 correct predictions from 75 crops) and **0.6759 macro-F1** on the fixed
internal test partition. This improves the v6 checkpoint's direct image-only
result by **10.67 percentage points** and improves macro-F1 by **0.1329**.

The requested 70% boundary was crossed by two correct crops: 53/75 is the
first attainable result above 70%, while the selected system achieved 54/75.
The reported value has not been rounded up or adjusted using test labels.

| Image-only system | Accuracy | Correct / 75 | Macro-F1 | Decision |
|---|---:|---:|---:|---|
| v6 checkpoint, direct | 61.33% | 46 | 0.5431 | Base model |
| Six-view TTA | 61.33% | 46 | 0.5744 | Not promoted |
| Deep-feature specialist | 62.67% | 47 | 0.5400 | Not promoted |
| Validation-calibrated v6 | 68.00% | 51 | 0.5704 | Not promoted alone |
| Four-expert visual ensemble | 69.33% | 52 | 0.6525 | Superseded |
| Frozen ViT specialist | 66.67% | 50 | 0.5830 | Ensemble expert |
| Five-expert visual ensemble | **72.00%** | **54** | **0.6759** | Best image-only system |
| Image-focused v7 retrain | 65.33% | 49 | 0.5434 | Not promoted |
| Resumed layer-4 v8 fine-tune | 60.00% | 45 | 0.5650 | Rejected |

## Selected image-only system

The selected system is a class-conditional gate over five frozen experts:

1. direct v6 image-only predictions;
2. six-view test-time augmentation;
3. a deep visual-feature specialist;
4. validation-calibrated v6 predictions;
5. an independently pretrained frozen ViT specialist.

The gate was selected on the 74-crop validation partition from 3,125
predeclared candidates. Its gate vector is `[2, 3, 4, 1, 4]`, keyed only by
the v6 base prediction, and was then applied unchanged to the test partition.
Validation accuracy was 85.14% and validation macro-F1 was 0.7957.

The fixed-test confusion matrix is:

| Actual \ Predicted | gray | helopeltis | algal | brown | red |
|---|---:|---:|---:|---:|---:|
| gray_blight | 18 | 0 | 1 | 7 | 0 |
| helopeltis | 0 | 1 | 0 | 1 | 0 |
| algal_leaf_spot | 3 | 1 | 8 | 1 | 0 |
| brown_blight | 4 | 2 | 0 | 15 | 0 |
| red_leaf_spot | 1 | 0 | 0 | 0 | 12 |

| Class | F1 |
|---|---:|
| gray_blight | 0.6923 |
| helopeltis | 0.3333 |
| algal_leaf_spot | 0.7273 |
| brown_blight | 0.6667 |
| red_leaf_spot | 0.9600 |

A 5,000-resample source-image group bootstrap over 38 source images gives a
95% interval of **60.23% to 85.14%** for accuracy and **0.5365 to 0.8256** for
macro-F1. The interval is wide because the evaluation set is small.

Machine-readable predictions and the selected gate are in
`ensemble5_vit/vision_ensemble_results.json`. The independent ViT specialist
audit is in `vit_specialist/vit_visual_specialist_results.json`.

## Multimodal model decision

The single recommended multimodal checkpoint remains v6:

- paired accuracy: 92.00%;
- paired macro-F1: 0.8753;
- direct image-only accuracy: 61.33%;
- direct image-only macro-F1: 0.5431.

The v8 layer-4 experiment produced a perfect paired internal-test score but
only 60.00% image-only accuracy. That run is explicitly rejected: the perfect
paired number is not a credible reason to replace the realistic v6 result and
does not satisfy the image-only objective.

## Claim boundary

- Train, validation, and test source-image overlap is zero.
- All ensemble and calibration choices were made from validation predictions,
  not test labels.
- The test partition contains 75 crops from only 38 source images.
- Because several development iterations have now been compared on this same
  fixed test partition, it should be treated as an internal development test,
  not as fresh external evidence of category-best performance.
- A defensible claim beyond this 72% internal result requires independently sourced
  labeled images, especially Helopeltis and the visually confusable gray,
  algal, and brown disease classes, followed by a new untouched group-held-out
  evaluation.
