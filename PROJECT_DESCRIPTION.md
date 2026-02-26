# FarmFederate: Privacy-Preserving Multimodal Federated Learning for Crop Stress Detection

## Project Overview

FarmFederate is a groundbreaking research and production system that combines federated learning with multimodal AI to detect crop stress while preserving farmer privacy. This project represents the **first work to define, train, and directly compare three distinct federated learning paradigms** for agricultural AI: Federated LLM (text-based), Federated ViT (vision-based), and Federated VLM (multimodal vision-language fusion).

## Motivation

The agricultural sector faces a critical challenge: crop stress (water shortage, nutrient deficiency, pest infestation, disease risk, heat stress) devastates yields globally, yet farmers are unwilling to share proprietary crop data with centralized AI systems due to privacy concerns and data ownership issues. Existing solutions require farms to upload raw images and sensitive operational data to cloud servers, creating barriers to adoption.

I identified that:
- Most agricultural AI is **centralized** (requires data sharing) and **unimodal** (image-only)
- Real-world farm data is heterogeneous and geographically distributed
- No prior work had systematically compared text-only, vision-only, and multimodal approaches **under federated constraints**

## Technical Architecture

### Backend Training Pipeline

**Location**: [backend/FarmFederate_Colab_Complete.py](backend/FarmFederate_Colab_Complete.py)

- 395KB single-file Python script designed for Google Colab/Kaggle execution
- Trained **24 distinct models** across three paradigms:
  - **5 LLM variants**: DistilBERT, BERT-tiny, RoBERTa-tiny, ALBERT-tiny, MobileBERT
  - **5 ViT variants**: ViT-Base, DeiT-tiny, Swin-tiny, ConvNeXT-tiny, EfficientNet
  - **8 VLM fusion architectures**: Concatenation, Cross-Attention, Gated, CLIP, Flamingo, BLIP-2, CoCa, Unified-IO

- Implemented **FedAvg** (Federated Averaging) with K=3 non-IID clients, 8 federated rounds
- Hardware: NVIDIA Tesla T4 GPU
- Generated 53 high-resolution visualizations and comprehensive metrics

### Frontend Mobile Application

**Location**: [frontend/](frontend/)

Flutter-based Android app with 14 screens including:
- Real-time crop diagnosis ([multimodal_diagnosis_screen.dart](frontend/lib/screens/multimodal_diagnosis_screen.dart))
- AI chat interface ([ai_chat_screen.dart](frontend/lib/screens/ai_chat_screen.dart))
- Federated learning dashboard ([federated_learning_screen.dart](frontend/lib/screens/federated_learning_screen.dart))
- Model selection and benchmarking ([model_selection_screen.dart](frontend/lib/screens/model_selection_screen.dart), [benchmark_results_screen.dart](frontend/lib/screens/benchmark_results_screen.dart))
- Analytics and visualization ([analytics_screen.dart](frontend/lib/screens/analytics_screen.dart))

**Key Features**:
- Integration with Firebase authentication
- MQTT protocol for IoT sensor data
- Local gradient computation ensures **no raw farm data ever leaves the device**
- Real-time model switching between LLM, ViT, and VLM paradigms

## Dataset Engineering

I constructed a balanced 5-class crop stress dataset addressing severe class imbalance (original 25:1 ratio):

**Text Data**:
- Curated from HuggingFace datasets: AG News, PubMed, and SQuAD
- Filtered for agricultural relevance
- ~1,089 balanced samples across 5 stress classes

**Image Data**:
- Combined PlantVillage and Beans datasets (real crop disease images)
- Synthetic class-distinctive patterns for augmentation
- 224×224 RGB images

**Preprocessing**:
- Applied capping + oversampling to achieve perfect class balance
- Eliminated 25:1 imbalance between disease and heat stress classes

## Novel Anti-Collapse Mechanisms

Class imbalance in federated settings causes catastrophic model collapse (predicting only 1-2 classes). I developed three countermeasures:

1. **Balanced Batch Sampling**: Equal class representation per batch during training
2. **Diversity Loss** (λ=1.0): Penalizes low-entropy predictions when 70% of predictions collapse to a single class
3. **Early Abort**: Halts training after 3 consecutive collapsed epochs to prevent wasted computation

**Result**: Achieved **100% prediction diversity** across all 24 models - every model predicts all 5 stress classes.

## Key Results

### Cross-Paradigm Performance Comparison

| Paradigm | Best Model | Centralized F1 | Federated F1 | Fed/Cent Ratio | Key Insight |
|----------|-----------|----------------|--------------|----------------|-------------|
| **Federated LLM** | ALBERT-tiny | 0.590 | 0.548 | 92.9% | Smallest federated gap - most robust to federation |
| **Federated ViT** | ConvNeXT-tiny | 0.765 | 0.659 | 86.1% | Highest unimodal accuracy but sensitive to non-IID data |
| **Federated VLM** | CLIP Fusion | 0.848 | 0.785 | 92.6% | Best absolute accuracy with strong federation retention |

### Additional Metrics

- **Average federated retention**: 90.5% of centralized performance across all paradigms
- **Stress-biased evaluation F1**: 0.989–1.000 (near-perfect under regional distribution shifts)
- **Privacy guarantee**: Zero raw data transmission (only gradients shared)
- **Total models trained**: 24 (including centralized and federated variants)
- **Training artifacts generated**: 79 files (models, plots, metrics)
- **Package size**: 1.05 GB (24 model checkpoints + visualizations)

## Real-World Use Case

A farmer photographs a diseased crop and types "yellowing leaves, wilting in the afternoon" into the FarmFederate Android app. The app:

1. Processes the image and text **locally on-device**
2. Computes model gradients without transmitting raw data
3. Uses VLM fusion to predict **Nutrient Deficiency (78% confidence)**
4. Recommends targeted fertilization strategies
5. Sends only gradient updates to federated server for model improvement

**Privacy preserved**: The photograph and symptom description never leave the farmer's device.

## Personal Contributions

As the lead developer and researcher, I:

### 1. Conceptualized the Three-Paradigm Framework
- Defined Federated LLM, ViT, and VLM as independent research directions
- Designed controlled experiments for rigorous cross-paradigm comparison
- Established evaluation metrics for privacy-utility trade-off analysis

### 2. Engineered the Complete Training Pipeline
- Built the 395KB monolithic Colab script with automatic dependency management
- Implemented 8 distinct VLM fusion strategies (including CLIP, Flamingo, BLIP-2, CoCa, Unified-IO)
- Developed the anti-collapse mechanisms that prevented model degradation
- Created comprehensive benchmarking suite comparing 45+ SOTA papers (2016-2025)

### 3. Designed and Implemented the Flutter Mobile App
- Created 14 interconnected screens with multimodal input handling (camera, text, IoT sensors)
- Integrated on-device gradient computation for federated training
- Built real-time AI chat with model selection capabilities
- Implemented Firebase authentication and MQTT sensor integration
- Designed analytics dashboard with fl_chart visualizations

### 4. Conducted Comprehensive Benchmarking
- Trained 24 models with systematic hyperparameter tuning
- Generated 53 research-quality visualizations (300 DPI PNG)
- Performed dataset comparison across AG News, PubMed, SQuAD, PlantVillage, and Beans
- Authored IEEE-format research paper ([paper/farmfederate_paper.tex](paper/farmfederate_paper.tex))
- Created detailed technical report ([paper/FarmFederate_Report.md](paper/FarmFederate_Report.md))

### 5. End-to-End Deployment
- Built production-ready Android APK ([FarmFederate.apk](FarmFederate.apk))
- Packaged all 24 trained models (.pt checkpoints, 1.05 GB total)
- Created reproducible research artifacts (plots, metrics, code)
- Prepared presentation materials (PowerPoint, LaTeX slides)

## Why This Matters

FarmFederate demonstrates that **privacy and accuracy are not mutually exclusive** in agricultural AI:

- The **VLM paradigm** achieves 78.5% F1 in federated mode (92.6% of centralized baseline), proving that farmers can benefit from state-of-the-art multimodal AI without surrendering proprietary data

- The **text-only LLM paradigm** shows the smallest federated gap (7.1%), making it ideal for low-bandwidth rural deployments where image transmission is costly

- The **vision-based ViT paradigm** demonstrates the highest unimodal accuracy but reveals that visual features are more sensitive to non-IID data distributions across farms

This work provides the first systematic comparison of these three paradigms under identical federated conditions, giving agricultural AI practitioners clear guidance on which approach to deploy based on their privacy, bandwidth, and accuracy requirements.

## Novel Contributions to Research

No prior work has:

1. **Defined three distinct paradigms**: Federated LLM, Federated ViT, and Federated VLM as separate research directions
2. **Trained each paradigm independently**: Under identical federated conditions (FedAvg, K=3, 8 rounds)
3. **Provided cross-paradigm comparison**: With per-paradigm privacy-utility trade-off analysis
4. **Applied to multi-class crop stress**: Previous work focused on binary disease detection, not 5-class stress classification
5. **Developed anti-collapse mechanisms**: Achieving 100% prediction diversity in federated settings with severe class imbalance

## Technical Stack

### Backend
- **Deep Learning**: PyTorch, torchvision, torchaudio
- **Transformers**: HuggingFace Transformers, sentence-transformers, timm
- **Data Processing**: pandas, NumPy, scikit-learn, Pillow
- **Visualization**: Matplotlib, Seaborn (50+ publication-quality plots)

### Frontend
- **Framework**: Flutter/Dart 3.0+
- **Authentication**: Firebase Core, Firebase Auth
- **Networking**: HTTP, MQTT client
- **Storage**: shared_preferences
- **Visualization**: fl_chart
- **State Management**: Provider
- **Media Handling**: image_picker, file_picker

### Research & Documentation
- **Paper Writing**: LaTeX (IEEE format)
- **Data Analysis**: Jupyter notebooks, pandas
- **Version Control**: Git
- **Cloud Training**: Google Colab (Tesla T4 GPU)

### DevOps
- **Mobile Build**: Android build tools, Gradle
- **Dependency Management**: pip, pub
- **Environment**: flutter_dotenv for configuration

## Project Timeline

**Total Duration**: 3-4 months (Research to Production)

**Phase 1 - Research & Design** (4 weeks):
- Literature review of 45+ SOTA papers
- Dataset curation from HuggingFace and PlantVillage
- Architecture design for three paradigms

**Phase 2 - Model Development** (6 weeks):
- Implementation of 24 model variants
- Federated learning infrastructure (FedAvg)
- Anti-collapse mechanism development

**Phase 3 - Training & Benchmarking** (3 weeks):
- Training on Tesla T4 GPU
- Hyperparameter tuning
- Generation of 53 visualizations

**Phase 4 - Mobile Application** (4 weeks):
- Flutter UI/UX design (14 screens)
- Firebase and MQTT integration
- On-device inference implementation

**Phase 5 - Documentation & Deployment** (2 weeks):
- IEEE research paper writing
- APK compilation and testing
- Repository organization and README creation

## Project Impact

### Academic Impact
- First systematic comparison of Federated LLM, ViT, and VLM paradigms
- Novel anti-collapse mechanisms for federated class-imbalanced learning
- Comprehensive benchmark dataset for future research

### Practical Impact
- Production-ready Android app for farmers
- Privacy-preserving AI deployment without centralized data collection
- Scalable architecture for geographically distributed farms

### Technical Impact
- Reusable federated learning codebase
- 24 pre-trained models available for transfer learning
- Comprehensive visualization suite for federated learning research

## Repository Structure

```
FarmFederate/
├── backend/
│   └── FarmFederate_Colab_Complete.py  # 395KB monolithic training script
├── frontend/
│   ├── lib/
│   │   ├── screens/                     # 14 Flutter screens
│   │   ├── constants.dart               # API endpoints and config
│   │   └── main.dart
│   ├── pubspec.yaml                     # Flutter dependencies
│   └── android/                         # Android build configuration
├── paper/
│   ├── farmfederate_paper.tex          # IEEE format research paper
│   ├── farmfederate_paper.pdf          # Compiled PDF
│   ├── FarmFederate_Report.md          # Technical report
│   ├── FarmFederate_Presentation.pptx  # Presentation slides
│   └── figures/                         # App screenshots and diagrams
├── models/                              # 24 trained model checkpoints (.pt)
├── plots/                               # 53 research visualizations
├── results/                             # Training metrics (JSON/CSV)
├── FarmFederate.apk                    # Production Android APK
└── README.md                            # Project documentation
```

## How to Run

### Backend Training (Google Colab)

```python
# Single-cell execution
# 1. Open Google Colab: https://colab.research.google.com/
# 2. Enable GPU: Runtime → Change runtime type → T4 GPU
# 3. Copy entire FarmFederate_Colab_Complete.py into a cell
# 4. Run the cell

# Configuration modes:
EXECUTION_MODE = 'quick'     # 2-3 min smoke test
EXECUTION_MODE = 'standard'  # 30-60 min full training
EXECUTION_MODE = 'full'      # 60-90 min maximum performance
```

### Frontend Mobile App

```bash
# Prerequisites: Flutter SDK 3.0+
cd frontend
flutter pub get
flutter run --release

# Build APK
flutter build apk --release
```

## Lessons Learned

1. **Federated learning is not just centralized learning + distribution**: The non-IID data distribution across farms creates unique challenges that require specialized mechanisms like diversity loss

2. **Multimodal fusion is crucial for agricultural AI**: VLM achieves 19.5% higher F1 than text-only and 16.5% higher than vision-only, proving that combining farmer observations with visual data is essential

3. **Class imbalance is catastrophic in federated settings**: Without anti-collapse mechanisms, models degrade to predicting only 1-2 classes, making them useless in production

4. **Privacy can be preserved without sacrificing accuracy**: 90.5% average retention shows that farmers don't need to choose between privacy and performance

5. **Text-based models are surprisingly robust to federation**: The 7.1% federated gap for LLM suggests that linguistic features are more consistent across farms than visual features

## Future Work

1. **Federated Batch Normalization (FedBN)**: Improve ViT paradigm's federated retention
2. **Personalized federated learning**: Allow farms to customize models for local crop varieties
3. **Differential privacy**: Add formal privacy guarantees with DP-SGD
4. **Edge deployment**: Optimize models for on-device inference (TFLite, ONNX)
5. **Real-world pilot**: Deploy to actual farms and collect feedback
6. **Temporal modeling**: Incorporate time-series data from IoT sensors
7. **Multi-language support**: Extend LLM paradigm to non-English regions

## Conclusion

FarmFederate bridges the gap between cutting-edge federated learning research and practical agricultural deployment. By systematically comparing three distinct paradigms (LLM, ViT, VLM) under identical federated conditions, this project provides clear guidance for agricultural AI practitioners: use text-only LLM for bandwidth-constrained deployments, vision-only ViT for image-rich scenarios with good data distribution, and multimodal VLM when maximum accuracy is required and compute permits. All three preserve farmer privacy by ensuring raw data never leaves the device.

This project exemplifies my ability to bridge research and production, combining cutting-edge federated learning research with practical mobile deployment while addressing real-world privacy constraints in agriculture.

---

**Author**: Ayush Debnath
**License**: MIT
**Version**: 5.1
**Last Updated**: February 2026
**Contact**: Available via GitHub repository
