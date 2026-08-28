# Current Exact-Split Model-Family Baselines

The proposed FarmFederate v6 core was not modified or retrained. All four rows use the same fixed internal-test label sequence of 75 crops from 38 source photographs.

| System | Architecture role | Accuracy | Macro-F1 | Correct |
|---|---|---:|---:|---:|
| Text-only PLM/LLM (Frozen DistilBERT) | distilbert-base-uncased frozen encoder, deterministic 50% sparse-note attention mask, masked-mean 768-d embedding, validation-selected class-weighted ridge | 65.33% | 0.6732 | 49/75 |
| Image-only ViT (Frozen ViT-tiny) | WinKawaks/vit-tiny-patch16-224 frozen CLS embeddings; ridge classifier selected on validation | 66.67% | 0.5830 | 50/75 |
| Image+text VLM (Raw v6 Cross-Attention) | unchanged validation-selected v6 paired bidirectional cross-attention output under deterministic 50% sparse notes; no post-hoc class gate | 77.33% | 0.7155 | 58/75 |
| Proposed FarmFederate (Unchanged) | unchanged v6 ResNet-50 + text Transformer + bidirectional cross-attention core with the existing validation-selected reliability-aware proposed_multimodal router | 81.33% | 0.6679 | 61/75 |

## DistilBERT validation selection

- Frozen encoder: `distilbert-base-uncased` (66,362,880 parameters)
- Sparse-note keep rate: `50%` using the existing deterministic FieldTxt hash rule
- Candidates: `90` ridge/class-weight settings
- Selected representation: `l2`
- Selected ridge alpha: `0.1`
- Selected class-weight power: `1.0`
- Validation accuracy: `67.57%`
- Validation macro-F1: `0.6767`
- Test embeddings were extracted only after these choices were frozen.

## VLM diagnostic

The headline image+text VLM row is the unchanged raw v6 paired cross-attention output, without a post-hoc class gate. The existing validation-selected late-fusion diagnostic reaches `78.67%` accuracy and `0.6472` macro-F1 (59/75). It does not replace the raw VLM baseline.

## Data and compatibility audit

- Split: `222 / 74 / 75` crops and `122 / 40 / 38` source photographs
- Pairwise source-photo overlap: `0`
- Exact text/image pairing coverage: `100%` in all partitions
- Training-fitted blocked lexical shortcuts: `178`
- Validation and fixed-test label arrays were verified against both the genuine ViT specialist and v6 hierarchy artifacts.

## Claim boundary

This is an internal, same-support model-family comparison. The fixed test was observed during earlier protocol development, so these values are not fresh external or category-best evidence. The VLM and proposed rows are the existing exploratory v6 predictions; this experiment does not alter the core architecture.
