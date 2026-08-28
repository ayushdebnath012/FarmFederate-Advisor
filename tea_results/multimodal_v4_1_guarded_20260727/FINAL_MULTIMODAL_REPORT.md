# FarmFederate Multimodal Final Report

> **Superseded:** The 1.000 guarded-validation result in this report is not the
> recommended metric. The confidence guard was selected on internal validation
> captions and produced an overly optimistic score. Use
> `../multimodal_v5_realistic_full_20260727/REALISTIC_EVALUATION_REPORT.md` and
> its unguarded locked-test result instead.

## Outcome

Training and cross-modal evaluation are complete on the corrected OBB dataset.
The selected model uses an ImageNet-pretrained ResNet-50 spatial encoder,
padding-aware text Transformer, bidirectional cross-attention, modality
auxiliary heads, supervised alignment, modality dropout, and a reliability
residual.

The vision upgrade raised image-only macro-F1 from 0.2933 in the earlier smoke
run to 0.5257 in the full grouped run. This is a +0.2324 absolute improvement
(approximately +79% relative), although the two runs used different dataset
sizes and are therefore a directional rather than controlled comparison.

A parameter-free inference guard was then added. When the text expert has at
least 0.60 confidence, it prevents the paired image path from overriding that
prediction. The guarded checkpoint reaches 1.0000 paired macro-F1 while leaving
image-only behavior unchanged.

## Leakage controls

- Training: 296 lesion crops from 162 source images.
- Validation: 75 lesion crops from 38 source images.
- Train/validation source-image overlap: 0.
- Exact image-box/text pairing coverage: 100% in both partitions.
- Model selection metric: validation macro-F1.
- Selected training epoch: 4.

All crops from one source image remain in one partition. The legacy
classification and augmented folders were not added because their folder names
conflict with the corrected taxonomy and no augmentation-to-source manifest
exists. Adding them would risk both label inversion and validation leakage.

## Cross-modal results

| Condition | Unguarded macro-F1 | Guarded macro-F1 | Guarded accuracy | Guarded ECE |
|---|---:|---:|---:|---:|
| Correctly paired | 0.9259 | 1.0000 | 1.0000 | 0.0739 |
| Text only | 1.0000 | 1.0000 | 1.0000 | 0.0766 |
| Image only | 0.5257 | 0.5257 | 0.6133 | 0.1044 |
| Mismatched text | 0.3258 | 0.1038 | 0.1333 | 0.7912 |

The guarded paired path no longer underperforms the best unimodal path on this
split. Its mismatch drop is 0.8962 macro-F1, showing that a confident but
incorrect text description can dominate the guarded prediction. The guard is
therefore suitable for trusted generated annotations, not untrusted
user-supplied text without an additional consistency or abstention policy.

Unguarded embedding retrieval results were:

- text-to-image class Recall@1: 0.5333;
- image-to-text class Recall@1: 0.5867;
- paired-versus-rolled cosine margin: +0.0583.

## Image-only analysis

| Class | Image-only F1 |
|---|---:|
| gray_blight | 0.5926 |
| helopeltis | 0.0000 |
| algal_leaf_spot | 0.5600 |
| brown_blight | 0.5128 |
| red_leaf_spot | 0.9630 |

Helopeltis remains the primary visual bottleneck: only seven grouped training
crops and two validation crops are available. Oversampling cannot create the
missing visual diversity. Genuinely new, provenance-checked Helopeltis images
or a source-aware augmentation manifest are required for a defensible next
improvement.

## Verification

- `tea_train.py` compiles successfully.
- Five regression checks pass: deterministic token IDs, grouped split
  isolation, integer-safe FedAvg, multimodal ablations/alignment loss, and
  cached spatial-feature dtype restoration.
- The guarded checkpoint reloads with `strict=True`; all keys match.
- The guard adds no trainable parameters and image-only outputs are unchanged.

## Claim boundary

This is the strongest verified configuration produced in this repository, but
the internal validation set is too small to establish a category-best or
state-of-the-art claim. The 0.60 guard was selected on the same validation
split. A publishable claim requires an external held-out paired dataset,
repeated-seed confidence intervals, more real Helopeltis samples, and
independent calibration of the guard threshold.
