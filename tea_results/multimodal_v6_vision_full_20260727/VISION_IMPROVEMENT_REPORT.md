# FarmFederate v6 Vision Improvement Report

## Outcome

The v6 visual training changes improved image-only locked-test accuracy from
**52.00% to 61.33%** (+9.33 percentage points) and image-only macro-F1 from
**0.4747 to 0.5431** (+0.0684).

Paired performance improved at the same time:

| Metric | v5 realistic | v6 vision | Change |
|---|---:|---:|---:|
| Paired accuracy | 0.8667 | 0.9200 | +0.0533 |
| Paired macro-F1 | 0.7810 | 0.8753 | +0.0943 |
| Image-only accuracy | 0.5200 | 0.6133 | +0.0933 |
| Image-only macro-F1 | 0.4747 | 0.5431 | +0.0684 |
| Image-only ECE | 0.1238 | 0.1135 | -0.0103 |
| Fusion gap versus text-only | -0.1314 | -0.0296 | +0.1018 |

The v5 and v6 comparisons use the same seed, source-grouped
train/validation/test partitions, target-shortcut masking protocol, and locked
test set.

## Changes

- Cached two reproducible, independently augmented ResNet-50 feature views per
  training crop instead of one.
- Increased image-only multimodal training exposure from 35% to 50%.
- Extended image-only warm-up from two to four epochs.
- Corrected warm-up loss masking so absent text and pair-alignment objectives
  are not optimized during image-only batches.
- Increased the dedicated vision auxiliary weight from 0.75 to 1.25.
- Selected checkpoints with a predeclared validation score containing 65%
  paired macro-F1 and 35% image-only macro-F1.
- Kept the confidence guard disabled.

## Locked-test results

| Condition | Macro-F1 | Accuracy | NLL | ECE |
|---|---:|---:|---:|---:|
| Correctly paired | 0.8753 | 0.9200 | 0.5315 | 0.2648 |
| Text only | 0.9049 | 0.9600 | 0.2733 | 0.1463 |
| Image only | 0.5431 | 0.6133 | 1.1828 | 0.1135 |
| Mismatched text | 0.2804 | 0.3333 | 1.7254 | 0.2193 |

The paired model correctly classified 69 of 75 test crops. Image-only
classified 46 of 75.

A 5,000-resample source-group bootstrap gives:

- paired accuracy 95% interval: 85.71% to 98.22%;
- paired macro-F1 95% interval: 0.7074 to 0.9741;
- image-only accuracy 95% interval: 48.57% to 75.68%;
- image-only macro-F1 95% interval: 0.4493 to 0.6272.

## Image-only per-class F1

| Class | F1 |
|---|---:|
| gray_blight | 0.5909 |
| helopeltis | 0.0000 |
| algal_leaf_spot | 0.5455 |
| brown_blight | 0.5789 |
| red_leaf_spot | 1.0000 |

Helopeltis remains unresolved because only nine total crops exist and only two
are in the locked test. The next defensible image-only gain requires new,
independently sourced Helopeltis images rather than additional copies of the
same samples.

## Verification and claim boundary

- Pairwise train/validation/test source-image overlap: 0.
- Exact image-box/text pairing coverage: 100%.
- Target-shortcut tokens masked using training annotations only: 178.
- Seven regression checks pass.
- The checkpoint reloads with `strict=True`; all keys match.
- Checkpoint metadata records `text_confidence_guard=None`.

This is the recommended repository model, but it is not evidence of
category-best performance. The test set contains only 38 independent source
images, so external validation and repeated grouped seeds remain necessary.

## Subsequent image-only inference improvement

A later validation-selected visual ensemble, including an independent frozen
ViT specialist, raised image-only internal-test accuracy from 61.33% to
**72.00%** and macro-F1 from 0.5431 to **0.6759** without changing the
recommended v6 checkpoint. See `IMAGE70_ATTEMPT_REPORT.md`. This is 54/75
correct and crosses the requested 70% boundary without rounding.
