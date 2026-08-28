// FarmFederate Constants
// All metrics sourced directly from checkpoint files in farmfederate_results (3)/drive/checkpoints/

import 'dart:io' show Platform;
import 'package:flutter/foundation.dart' show kIsWeb;

// ============================================================
// BACKEND CONFIGURATION
// ============================================================

/// Production backend URL - UPDATE THIS for deployment
const String PRODUCTION_BACKEND = "https://farmfederate-advisor.onrender.com";

/// Local development backend URLs
const String _LOCALHOST = "http://localhost:8000";
const String _ANDROID_EMULATOR = "http://10.0.2.2:8000";
const String _IOS_SIMULATOR = "http://localhost:8000";

/// LAN IP for testing on real devices (update to your machine's IP)
const String _LAN_BACKEND = "http://192.168.1.100:8000";

/// Set to true when deploying to production
const bool IS_PRODUCTION = false;

/// Get the appropriate backend URL based on platform
String getBackendUrl() {
  if (IS_PRODUCTION && PRODUCTION_BACKEND != "https://your-backend-url.com") {
    return PRODUCTION_BACKEND;
  }
  if (kIsWeb) return _LOCALHOST;
  try {
    if (Platform.isAndroid) return _ANDROID_EMULATOR;
    if (Platform.isIOS) return _IOS_SIMULATOR;
  } catch (_) {}
  return _LOCALHOST;
}

/// Default backend URL (computed at runtime)
String get DEFAULT_BACKEND => getBackendUrl();

const String PREDICT_PATH = "/predict";
const String HEALTH_PATH = "/health";
const String MODELS_PATH = "/models";
const String SENSORS_PATH = "/sensors/latest";
const String RAG_PATH = "/rag";
const String RAG_METRICS_PATH = "/rag/metrics";

// ============================================================
// MQTT CONFIGURATION (IoT Sensors)
// ============================================================

const String MQTT_BROKER = "ws://192.168.1.100:9001";
const String MQTT_USERNAME = "";
const String MQTT_PASSWORD = "";
const String SENSOR_TOPIC = "farm/sensors/#";
const String CMD_TOPIC = "farm/cmd/";

// ============================================================
// DATASET CONFIGURATION (Paper Section 3 - IIT Kharagpur tea garden)
// ============================================================

const int TOTAL_FIELD_IMAGES = 200; // field-collected photographs
const int TOTAL_IMAGE_TENSORS = 792; // after minority-class fill
const int TRAINING_SAMPLES = 633; // train split
const int VALIDATION_SAMPLES = 79; // validation split
const int TOTAL_TEXT_SAMPLES = 3000; // synthetic agronomic text (600/class)
const int YOLO_ANNOTATIONS = 371; // YOLO-OBB bounding boxes
const int NUM_CLASSES = 5;

// kept for backward-compat
const int TOTAL_SAMPLES = TOTAL_IMAGE_TENSORS;
const int TEST_SAMPLES = 80;

/// Image parameters
const int IMAGE_SIZE = 224;
const int IMAGE_CHANNELS = 3;

/// Text parameters
const int MAX_SEQUENCE_LENGTH = 128;
const int AVG_SEQUENCE_LENGTH = 45;
const int VOCABULARY_SIZE = 30522; // BERT vocab

/// Training parameters (paper Section 4)
const int BATCH_SIZE = 32;
const double LEARNING_RATE = 2e-4;
const int EPOCHS = 30;
const String OPTIMIZER = "AdamW";
const double WEIGHT_DECAY = 0.01;

/// Federated parameters (K=3 clients, T=8 rounds, E=3 local epochs - paper Section 4)
const int NUM_CLIENTS = 3;
const int FEDERATED_ROUNDS = 8;
const int LOCAL_EPOCHS = 3;
const double NON_IID_ALPHA = 1.0;

// ============================================================
// RAG CONFIGURATION (paper Section 6)
// ============================================================

const int RAG_TRAINING_ROUNDS = 10;
const int RAG_TOP_K = 5;
const int RAG_QUERY_DIM = 128;
const double RAG_EMA_MU = 0.9;
const int RAG_SAMPLES_PER_FARM = 400;
const double RAG_NON_IID_ALPHA = 0.5;

/// RAG evaluation results
const double RAG_MACRO_F1 =
    1.000; // on synthetic template text (trivially easy)
const double RAG_MICRO_F1 = 1.000;
const double RAG_RECALL_AT_5 = 0.129; // honest real-world retrieval signal
const double RAG_MRR = 0.100;
const double RAG_NDCG_AT_5 = 0.172;
const double RAG_KB_COVERAGE = 1.000;
const double RAG_EMBEDDING_DRIFT = 0.156;
const int RAG_KB_SIZE = 15; // 3 documents per class x 5 classes

/// Loss parameters
const double FOCAL_GAMMA = 2.0;
const double DIVERSITY_LAMBDA = 0.1;

// ============================================================
// 5-CLASS TEA DISEASE LABELS (Colab / paper Section 3)
// Matches backend STRESS_LABELS exactly:
//   0=LEAF_BLIGHT  1=LEAF_HOPPERS  2=LEAF_RUST
//   3=LOOPER_CATERPILLARS  4=MOSQUITO_BUG
// ============================================================

const List<String> STRESS_LABELS = [
  'LEAF_BLIGHT',
  'LEAF_HOPPERS',
  'LEAF_RUST',
  'LOOPER_CATERPILLARS',
  'MOSQUITO_BUG',
];

/// Display names shown to the farmer
const Map<String, String> STRESS_DISPLAY_NAMES = {
  'LEAF_BLIGHT': 'Leaf Blight',
  'LEAF_HOPPERS': 'Leaf Hoppers',
  'LEAF_RUST': 'Leaf Rust',
  'LOOPER_CATERPILLARS': 'Looper Caterpillars',
  'MOSQUITO_BUG': 'Mosquito Bug',
};

/// Short labels for chips / charts
const Map<String, String> STRESS_SHORT = {
  'LEAF_BLIGHT': 'Blight',
  'LEAF_HOPPERS': 'Hoppers',
  'LEAF_RUST': 'Rust',
  'LOOPER_CATERPILLARS': 'Loopers',
  'MOSQUITO_BUG': 'Mosquito',
};

/// Scientific descriptions
const Map<String, String> STRESS_DESCRIPTIONS = {
  'LEAF_BLIGHT':
      'Brown necrotic lesions with water-soaked margins on leaf surfaces.',
  'LEAF_HOPPERS':
      'Stippling, tip burn and marginal scorch caused by leafhopper feeding.',
  'LEAF_RUST':
      'Orange-yellow urediniospore pustules on the lower leaf surface.',
  'LOOPER_CATERPILLARS':
      'Skeletonised leaves with characteristic frass from caterpillar feeding.',
  'MOSQUITO_BUG':
      'Corky raised lesions and shoot dieback caused by Helopeltis theivora.',
};

/// Actionable treatment advice per disease
const Map<String, String> STRESS_ADVICE = {
  'LEAF_BLIGHT':
      'Apply copper-based fungicide (copper oxychloride 50 WP). Remove and destroy infected '
          'leaves immediately. Improve drainage and avoid overhead irrigation.',
  'LEAF_HOPPERS':
      'Apply systemic insecticide (imidacloprid 200 SL or thiamethoxam 25 WG). '
          'Prune dense canopy to reduce hopper habitat. Monitor every 2 weeks.',
  'LEAF_RUST':
      'Apply propiconazole 25 EC or hexaconazole 5 SC fungicide. Collect and destroy '
          'fallen infected leaves. Improve inter-row air circulation.',
  'LOOPER_CATERPILLARS':
      'Apply Bacillus thuringiensis (Bt) spray or chlorpyrifos 20 EC. '
          'Inspect plants early morning. Introduce natural predators (parasitic wasps).',
  'MOSQUITO_BUG':
      'Apply endosulfan 35 EC or lambda-cyhalothrin 5 CS. Shade regulation and '
          'removal of weeds that harbour Helopeltis populations.',
};

// ============================================================
// MODEL CONFIGURATIONS
// ============================================================

/// 5 LLM Encoder variants (paper Table 8)
const List<String> LLM_ENCODERS = [
  'bert_tiny',
  'roberta_tiny',
  'albert_tiny',
  'distilbert',
  'mobilebert',
];

/// 5 ViT Encoder variants (paper Table 9)
const List<String> VIT_ENCODERS = [
  'efficientnet',
  'convnext_tiny',
  'deit_tiny',
  'swin_tiny',
  'vit_base',
];

/// 8 VLM Fusion strategies (paper Table 10)
const List<String> VLM_FUSIONS = [
  'clip',
  'blip2',
  'attention',
  'flamingo',
  'unified_io',
  'gated',
  'coca',
  'concat',
];

/// 3 Federated learning paradigms
const List<String> FEDERATED_MODES = [
  'fed_llm',
  'fed_vit',
  'fed_vlm',
];

const int TOTAL_CONFIGURATIONS = 203;

// ============================================================
// BEST CONFIGURATION - from checkpoint files
// Source: farmfederate_results (3)/drive/checkpoints/
// ============================================================

// Best per paradigm (centralized individual benchmark)
const String BEST_LLM = 'bert_tiny'; // Cent F1 = 0.487
const String BEST_VIT = 'efficientnet'; // Cent F1 = 0.911
const String BEST_FUSION = 'clip'; // Cent F1 = 0.949

// Cross-paradigm CENTRALIZED baselines  (centralized_xxx_best.pt)
const double CENT_F1_LLM = 0.427; // centralized_llm_best.pt
const double CENT_F1_VIT = 0.886; // centralized_vit_best.pt
const double CENT_F1_VLM = 0.873; // centralized_vlm_best.pt

// Cross-paradigm FEDERATED F1  (federated_xxx_best.pt)
const double FED_F1_LLM = 0.460; // federated_llm_best.pt
const double FED_F1_VIT = 0.886; // federated_vit_best.pt
const double FED_F1_VLM = 0.861; // federated_vlm_best.pt

// Federation retention rates (Fed / Cent x 100)
const double RETENTION_LLM = 107.7; // 0.460 / 0.427
const double RETENTION_VIT = 100.0; // 0.886 / 0.886
const double RETENTION_VLM = 98.6; // 0.861 / 0.873

// Overall best scores
const double BEST_MACRO_F1_CENTRALIZED = 0.949; // CLIP centralized individual
const double BEST_MACRO_F1_FEDERATED = 0.861; // VLM-CLIP federated
const double BEST_ACCURACY = 0.949;

// Derived summary statistics
const double MULTIMODAL_IMPROVEMENT = 87.2; // ((0.861-0.460)/0.460) x 100
const double FED_CENTRALIZED_RATIO = 102.1; // avg retention across 3 paradigms
const double COMMUNICATION_REDUCTION = 67.0; // % bandwidth reduction

// Model sizes
const int BEST_MODEL_PARAMETERS_M = 16; // CLIP: 15.7M params
const double TRAINING_TIME_HOURS = 4.2;
const int INFERENCE_LATENCY_MS = 89;

// ============================================================
// PER-MODEL CENTRALIZED F1 SCORES (from individual checkpoints)
// ============================================================

const Map<String, double> LLM_CENT_F1 = {
  'bert_tiny': 0.487,
  'roberta_tiny': 0.463,
  'albert_tiny': 0.450,
  'distilbert': 0.433,
  'mobilebert': 0.417,
};

const Map<String, double> VIT_CENT_F1 = {
  'efficientnet': 0.911,
  'convnext_tiny': 0.899,
  'deit_tiny': 0.873,
  'swin_tiny': 0.873,
  'vit_base': 0.861,
};

const Map<String, double> VLM_CENT_F1 = {
  'clip': 0.949,
  'blip2': 0.937,
  'attention': 0.924,
  'flamingo': 0.911,
  'unified_io': 0.911,
  'gated': 0.899,
  'coca': 0.899,
  'concat': 0.886,
};

// ============================================================
// VLM PARAMETER COUNTS (from checkpoints, in millions)
// ============================================================

const Map<String, double> VLM_PARAMS_M = {
  'clip': 15.7,
  'blip2': 15.9,
  'attention': 15.9,
  'flamingo': 16.1,
  'unified_io': 17.2,
  'gated': 15.8,
  'coca': 16.2,
  'concat': 15.6,
};

// ============================================================
// CHECKPOINT FILE SIZES (MB)
// ============================================================

const Map<String, double> MODEL_SIZE_MB = {
  'bert_tiny': 42.4,
  'roberta_tiny': 42.4,
  'albert_tiny': 42.4,
  'distilbert': 42.4,
  'mobilebert': 42.4,
  'efficientnet': 19.0,
  'convnext_tiny': 19.0,
  'deit_tiny': 19.0,
  'swin_tiny': 19.0,
  'vit_base': 19.0,
  'clip': 60.1,
  'blip2': 60.8,
  'attention': 60.6,
  'flamingo': 61.6,
  'unified_io': 65.6,
  'gated': 60.3,
  'coca': 61.8,
  'concat': 59.5,
};

// ============================================================
// FARMER-FRIENDLY MODEL NAMES & DESCRIPTIONS
// ============================================================

const Map<String, String> LLM_FRIENDLY_NAMES = {
  'bert_tiny': 'BERT-tiny',
  'roberta_tiny': 'RoBERTa-tiny',
  'albert_tiny': 'ALBERT-tiny',
  'distilbert': 'DistilBERT',
  'mobilebert': 'MobileBERT',
};

const Map<String, String> LLM_FRIENDLY_DESCRIPTIONS = {
  'bert_tiny':
      'Best LLM - Cent F1 0.487 | top text encoder for tea disease symptom descriptions',
  'roberta_tiny':
      'RoBERTa-tiny - Cent F1 0.463 | robustly trained, strong on varied phrasings',
  'albert_tiny':
      'ALBERT-tiny - Cent F1 0.450 | parameter-sharing reduces memory footprint',
  'distilbert': 'DistilBERT - Cent F1 0.433 | fast 6-layer distilled encoder',
  'mobilebert':
      'MobileBERT - Cent F1 0.417 | mobile-optimised, lowest size (42 MB)',
};

const Map<String, String> VIT_FRIENDLY_NAMES = {
  'efficientnet': 'EfficientNet',
  'convnext_tiny': 'ConvNeXT-tiny',
  'deit_tiny': 'DeiT-tiny',
  'swin_tiny': 'Swin-tiny',
  'vit_base': 'ViT-Base',
};

const Map<String, String> VIT_FRIENDLY_DESCRIPTIONS = {
  'efficientnet':
      'Best ViT - Cent F1 0.911 | top image encoder for tea leaf disease detection',
  'convnext_tiny':
      'ConvNeXT-tiny - Cent F1 0.899 | modern ConvNet; excellent lesion detection',
  'deit_tiny': 'DeiT-tiny - Cent F1 0.873 | data-efficient vision transformer',
  'swin_tiny':
      'Swin-tiny - Cent F1 0.873 | hierarchical transformer; good at scale variation',
  'vit_base':
      'ViT-Base - Cent F1 0.861 | standard patch-based vision transformer',
};

const Map<String, String> FUSION_FRIENDLY_NAMES = {
  'clip': 'CLIP Match',
  'blip2': 'BLIP-2 Deep',
  'attention': 'Cross-Attention',
  'flamingo': 'Flamingo AI',
  'unified_io': 'Unified IO',
  'gated': 'Gated Mix',
  'coca': 'CoCa Link',
  'concat': 'Simple Join',
};

const Map<String, String> FUSION_FRIENDLY_DESCRIPTIONS = {
  'clip':
      'Best VLM - Cent F1 0.949 | CLIP contrastive alignment; jointly aligns tea disease text and image',
  'blip2': 'BLIP-2 - Cent F1 0.937 | Q-Former deep vision-language fusion',
  'attention':
      'Cross-Attention - Cent F1 0.924 | multi-head attention over text and visual tokens',
  'flamingo':
      'Flamingo - Cent F1 0.911 | Perceiver + GatedXAttn cross-modal fusion',
  'unified_io':
      'Unified-IO - Cent F1 0.911 | universal transformer encoder fusion',
  'gated':
      'Gated - Cent F1 0.899 | learnable gate balances text vs. image contribution',
  'coca': 'CoCa - Cent F1 0.899 | dual contrastive-captioning head fusion',
  'concat':
      'Concat - Cent F1 0.886 | simple concatenation of text and visual embeddings',
};

const Map<String, String> FEDERATED_FRIENDLY_NAMES = {
  'fed_llm': 'Federated LLM',
  'fed_vit': 'Federated ViT',
  'fed_vlm': 'Federated VLM',
};

const Map<String, String> FEDERATED_FRIENDLY_DESCRIPTIONS = {
  'fed_llm':
      'Text-only paradigm - BERT-tiny | Fed F1 0.460 | 107.7% retention (exceeds centralized baseline)',
  'fed_vit':
      'Image-only paradigm - EfficientNet | Fed F1 0.886 | 100.0% retention | no accuracy loss from federation',
  'fed_vlm':
      'Text + Image paradigm - CLIP | Fed F1 0.861 | 98.6% retention | best overall federated performance',
};

// ============================================================
// UI CONSTANTS
// ============================================================

const double DEFAULT_PADDING = 16.0;
const double CARD_BORDER_RADIUS = 16.0;
const double BUTTON_BORDER_RADIUS = 12.0;

const int SENSOR_POLL_INTERVAL_SECONDS = 5;
const int API_TIMEOUT_SECONDS = 30;
