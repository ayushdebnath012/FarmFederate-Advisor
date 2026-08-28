# Vision-Branch Upgrade Report

## Outcome

The image-only branch improved from **0.2933 to 0.5257 macro-F1**
(+0.2324 absolute, approximately +79% relative). Image-only accuracy improved
from **36.36% to 61.33%**, while ECE decreased from **0.2482 to 0.1044**.

The comparison is directional rather than a controlled statistical comparison:
the baseline was a 55/22-crop smoke run, whereas v4 used the full grouped
296/75-crop split.

| Metric | v3 smoke baseline | v4 full ResNet-50 | Change |
|---|---:|---:|---:|
| Paired macro-F1 | 0.9136 | 0.9259 | +0.0123 |
| Image-only macro-F1 | 0.2933 | 0.5257 | +0.2324 |
| Image-only accuracy | 0.3636 | 0.6133 | +0.2497 |
| Image-only ECE | 0.2482 | 0.1044 | -0.1438 |
| Mismatched-text macro-F1 | 0.2278 | 0.3258 | +0.0980 |
| Paired-vs-mismatch drop | 0.6859 | 0.6001 | -0.0858 |

## Changes

- Replaced the randomly initialized image CNN with a locally cached,
  ImageNet-pretrained ResNet-50 spatial backbone.
- Preserved its 7x7 feature map for bidirectional cross-attention.
- Froze the pretrained backbone and cached fp16 spatial maps once per sample.
- Added two image-only warm-up epochs.
- Increased image-only exposure to 35% of multimodal training batches.
- Weighted the vision auxiliary loss 3x more heavily than the text auxiliary
  loss.
- Retained exact image/box-to-text pairing and source-image-grouped splitting.

## Full-data evidence

- Training crops: 296 from 162 source images.
- Validation crops: 75 from 38 source images.
- Source-image overlap: 0.
- Exact pair coverage: 100% in both partitions.
- Best epoch: 4.
- Paired macro-F1: 0.9259.
- Paired accuracy: 0.9867.
- Image-only macro-F1: 0.5257.
- Text-to-image class Recall@1: 0.5333.
- Image-to-text class Recall@1: 0.5867.

## Remaining visual bottleneck

Image-only per-class F1 is:

| Class | F1 |
|---|---:|
| gray_blight | 0.5926 |
| helopeltis | 0.0000 |
| algal_leaf_spot | 0.5600 |
| brown_blight | 0.5128 |
| red_leaf_spot | 0.9630 |

Helopeltis has only seven grouped training crops and two validation crops.
Synthetic oversampling cannot replace genuine visual diversity, so no
additional repository images were automatically ingested where their label
provenance conflicts with the corrected OBB class mapping.

Text-only reaches 1.0000 macro-F1 because the annotations explicitly describe
class-defining symptoms. Consequently, paired fusion does not beat the
text-only shortcut on this internal split even though it remains highly
sensitive to mismatching. A category-best claim requires external paired data,
more genuine Helopeltis samples, and repeated-seed evaluation.
