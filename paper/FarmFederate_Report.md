# FarmFederate — Short Report
**FarmFederate: Federated LLM, ViT, and VLM for Privacy-Preserving Crop Stress Detection**

---

## What is FarmFederate?

FarmFederate is a federated learning framework for **crop stress detection** — identifying whether a crop is suffering from water stress, nutrient deficiency, pest infestation, disease risk, or heat stress. It is the **first work to define, separately train, and directly compare three federated learning paradigms** for agricultural AI:

| Paradigm | What it uses | Federated F1 | Fed/Cent Ratio |
|---|---|---|---|
| **Federated LLM** | Text (symptom descriptions) | 0.548 | 92.9% |
| **Federated ViT** | Images (crop photos) | 0.659 | 86.1% |
| **Federated VLM** | Text + Image (multimodal) | 0.785 | 92.6% |

---

## The Problem

- Crop stress kills yield — early detection is critical for food security
- Most AI methods are **centralized** (farms must share raw data) and **unimodal** (image only)
- Farms are unwilling to share proprietary data due to privacy and ownership concerns
- Real-world data is distributed across geographically dispersed, heterogeneous farms

---

## The Solution: Three Federated Paradigms

Each paradigm is trained **independently** using FedAvg (3 clients, 8 rounds):

### Paradigm 1 — Federated LLM
- **Input**: Farmer-written symptom text (e.g., *"leaves yellowing, dry soil"*)
- **Models**: DistilBERT, BERT-tiny, RoBERTa-tiny, ALBERT-tiny, MobileBERT
- **Best**: ALBERT-tiny, centralized F1 = 0.590, federated F1 = 0.548
- **Key finding**: Smallest federated gap (7.1%) — most federation-robust paradigm
- **Example**: Text *"brown spots on leaf edges, curling"* → Predicted: **Disease Risk**

### Paradigm 2 — Federated ViT
- **Input**: Crop photographs (224×224 RGB)
- **Models**: ViT-Base, DeiT-tiny, Swin-tiny, ConvNeXT-tiny, EfficientNet
- **Best**: ConvNeXT-tiny, centralized F1 = 0.765, federated F1 = 0.659
- **Key finding**: Highest unimodal accuracy but largest federated gap (13.9%) — visual features are sensitive to non-IID data distributions across farms
- **Example**: Image of wilting, pale-green leaves → Predicted: **Water Stress**

### Paradigm 3 — Federated VLM
- **Input**: Text + Image combined
- **Fusion strategies**: Concatenation, Cross-Attention, Gated, CLIP, Flamingo, BLIP-2, CoCa, Unified-IO
- **Best**: CLIP fusion, centralized F1 = 0.848, federated F1 = 0.785
- **Key finding**: Best absolute accuracy with a federated gap comparable to LLM (7.4%)
- **Example**: Image of spotted leaves + text *"small holes in leaves, sticky residue"* → Predicted: **Pest Risk** (84.8% confidence)

---

## Dataset

- **5 stress classes**: Water Stress, Nutrient Deficiency, Pest Risk, Disease Risk, Heat Stress
- **Text**: AG News + PubMed + SQuAD filtered for agriculture → ~1,089 balanced samples
- **Images**: PlantVillage + Beans (real) + synthetic class-distinctive patterns
- **Rebalancing**: Original 25:1 imbalance (disease vs heat stress) → balanced via capping + oversampling

---

## Anti-Collapse Mechanisms

Class imbalance causes models to predict only 1-2 classes. FarmFederate prevents this with:
1. **Balanced Batch Sampling** — equal class representation per batch
2. **Diversity Loss** — penalizes low-entropy predictions (λ=1.0, 70% threshold)
3. **Early Abort** — stops training after 3 consecutive collapsed epochs

Result: **100% prediction diversity** across all 24 trained models.

---

## Key Results

| Metric | Value |
|---|---|
| Best federated F1 (VLM-CLIP) | 0.785 |
| Best centralized F1 (VLM-CLIP) | 0.848 |
| Average Fed/Cent retention | 90.5% |
| Stress-biased evaluation F1 | 0.989–1.000 |
| Total models trained | 24 |
| Hardware | NVIDIA Tesla T4 |

---

## Why is This Novel?

No prior work has:
1. Defined **Federated LLM**, **Federated ViT**, and **Federated VLM** as three distinct paradigms
2. Trained each paradigm **independently** under identical federated conditions
3. Provided a **cross-paradigm comparison** with per-paradigm privacy-utility trade-off analysis
4. Applied any of these to **multi-class crop stress** (not just binary disease detection)

---

## Practical Use Case

A farmer photographs a diseased crop and types *"yellowing leaves, wilting in the afternoon"*. The FarmFederate Android app sends only the model gradient — not the photo — to the federated server. The VLM fusion predicts **Nutrient Deficiency (78%)** and recommends targeted fertilization. No farm data ever leaves the device.

---

## Conclusion

- **Federated LLM**: Best federation robustness, lowest cost — ideal for text-rich, bandwidth-constrained deployments
- **Federated ViT**: Strong unimodal accuracy, needs better aggregation (FedBN) for non-IID robustness
- **Federated VLM**: Best overall — highest accuracy + strong federation retention — recommended when compute permits
