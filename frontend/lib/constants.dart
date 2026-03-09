// FarmFederate Constants
// Configuration values matching the paper specifications

import 'dart:io' show Platform;
import 'package:flutter/foundation.dart' show kIsWeb;

// ============================================================
// BACKEND CONFIGURATION
// ============================================================

/// Production backend URL - UPDATE THIS for deployment
/// Set this to your deployed backend URL (e.g., https://api.farmfederate.com)
const String PRODUCTION_BACKEND = "https://farmfederate-advisor.onrender.com";

/// Local development backend URLs
const String _LOCALHOST = "http://localhost:8000";
const String _ANDROID_EMULATOR = "http://10.0.2.2:8000";
const String _IOS_SIMULATOR = "http://localhost:8000";

/// LAN IP for testing on real devices (update to your machine's IP)
const String _LAN_BACKEND = "http://192.168.1.100:8000";

/// Set to true when deploying to production
const bool IS_PRODUCTION = true;

/// Get the appropriate backend URL based on platform
String getBackendUrl() {
  // Use production URL if configured
  if (IS_PRODUCTION && PRODUCTION_BACKEND != "https://your-backend-url.com") {
    return PRODUCTION_BACKEND;
  }

  // Development: auto-detect platform
  if (kIsWeb) {
    return _LOCALHOST;
  }

  try {
    if (Platform.isAndroid) {
      return _ANDROID_EMULATOR; // Change to _LAN_BACKEND for real device testing
    } else if (Platform.isIOS) {
      return _IOS_SIMULATOR; // Change to _LAN_BACKEND for real device testing
    }
  } catch (e) {
    // Platform not available (web)
  }

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
// PAPER CONFIGURATION (Table 12)
// ============================================================

/// Dataset parameters
const int TOTAL_SAMPLES = 5000;
const int TRAINING_SAMPLES = 4000;
const int VALIDATION_SAMPLES = 500;
const int TEST_SAMPLES = 500;
const int NUM_CLASSES = 5;

/// Image parameters
const int IMAGE_SIZE = 224;
const int IMAGE_CHANNELS = 3;

/// Text parameters
const int MAX_SEQUENCE_LENGTH = 128;
const int AVG_SEQUENCE_LENGTH = 45;
const int VOCABULARY_SIZE = 30522; // BERT vocab

/// Training parameters
const int BATCH_SIZE = 32;
const double LEARNING_RATE = 2e-4;
const int EPOCHS = 30;
const String OPTIMIZER = "AdamW";
const double WEIGHT_DECAY = 0.01;

/// Federated parameters (K=3 clients, T=8 rounds, E=3 local epochs — paper §4)
const int NUM_CLIENTS = 3;
const int FEDERATED_ROUNDS = 8;
const int LOCAL_EPOCHS = 3;
const double NON_IID_ALPHA = 1.0;

// ============================================================
// RAG CONFIGURATION (paper §6)
// ============================================================
const int RAG_TRAINING_ROUNDS = 10;
const int RAG_TOP_K = 5;
const int RAG_QUERY_DIM = 128;
const double RAG_EMA_MU = 0.9;
const int RAG_SAMPLES_PER_FARM = 400;
const double RAG_NON_IID_ALPHA = 0.5;

/// RAG evaluation results (from rag_results.json)
const double RAG_MACRO_F1 = 1.000;
const double RAG_MICRO_F1 = 1.000;
const double RAG_RECALL_AT_5 = 0.129;
const double RAG_MRR = 0.100;
const double RAG_NDCG_AT_5 = 0.172;
const double RAG_KB_COVERAGE = 1.000;
const double RAG_EMBEDDING_DRIFT = 0.156;
const int RAG_KB_SIZE = 15; // 15 knowledge-base documents (3 per class)

/// Loss parameters
const double FOCAL_GAMMA = 2.0;
const double DIVERSITY_LAMBDA = 0.1;

// ============================================================
// 5-CLASS STRESS LABELS (Farmer-Friendly)
// ============================================================

const List<String> STRESS_LABELS = [
  'healthy',
  'water_stress',
  'nutrient_def',
  'disease_risk',
  'pest_risk',
];

/// Farmer-friendly display names - simple language everyone understands
const Map<String, String> STRESS_DISPLAY_NAMES = {
  'healthy': 'Healthy Crop',
  'water_stress': 'Needs Water',
  'nutrient_def': 'Needs Fertilizer',
  'disease_risk': 'May Be Sick',
  'pest_risk': 'Bug Problem',
};

/// Detailed descriptions for farmers
const Map<String, String> STRESS_DESCRIPTIONS = {
  'healthy': 'Your crop looks healthy and is growing well',
  'water_stress': 'Your plant shows signs of needing more water',
  'nutrient_def': 'Your plant may need fertilizer or nutrients',
  'disease_risk': 'Your plant may have a disease or infection',
  'pest_risk': 'Your plant may have bugs or insect damage',
};

// ============================================================
// MODEL CONFIGURATIONS
// ============================================================

/// 5 LLM Encoder variants (Table 8)
const List<String> LLM_ENCODERS = [
  'distilbert',
  'bert_tiny',
  'roberta_tiny',
  'albert_tiny',
  'mobilebert',
];

/// 5 ViT Encoder variants (Table 9)
const List<String> VIT_ENCODERS = [
  'vit_base',
  'deit_tiny',
  'swin_tiny',
  'convnext_tiny',
  'efficientnet',
];

/// 8 VLM Fusion strategies (Table 10)
const List<String> VLM_FUSIONS = [
  'concat',
  'attention',
  'gated',
  'clip',
  'flamingo',
  'blip2',
  'coca',
  'unified_io',
];

/// 3 Federated learning modes
const List<String> FEDERATED_MODES = [
  'fed_llm',
  'fed_vit',
  'fed_vlm',
];

/// Total configurations: 5 x 5 x 8 + 3 federated = 203
const int TOTAL_CONFIGURATIONS = 203;

// ============================================================
// BEST CONFIGURATION — from paper cross-paradigm comparison
// ============================================================

// Best per paradigm (paper cross-paradigm comparison)
const String BEST_LLM = 'mobilebert';       // Fed F1 = 0.636, retention 100.8%
const String BEST_VIT = 'vit_base';         // Fed F1 = 0.857, retention 98.9%
const String BEST_FUSION = 'concat';        // Fed F1 = 0.848, retention 97.9%

// Per-paradigm federated F1 scores (best model in each paradigm)
const double FED_F1_LLM = 0.636;
const double FED_F1_VIT = 0.857;
const double FED_F1_VLM = 0.848;

// Per-paradigm centralized F1 scores (cross-paradigm table)
const double CENT_F1_LLM = 0.604;
const double CENT_F1_VIT = 0.825;
const double CENT_F1_VLM = 0.857;

// Per-paradigm federation retention (Fed/Cent %)
const double RETENTION_LLM = 100.8;
const double RETENTION_VIT = 98.9;
const double RETENTION_VLM = 97.9;

const double BEST_MACRO_F1_CENTRALIZED = 0.857;   // VLM-Concat centralized
const double BEST_MACRO_F1_FEDERATED = 0.848;     // VLM-Concat federated
const double BEST_ACCURACY = 0.857;

const double MULTIMODAL_IMPROVEMENT = 33.3; // VLM best (0.848) vs LLM best (0.636) Fed F1
const double FED_CENTRALIZED_RATIO = 99.2;  // avg across 3 paradigms
const double COMMUNICATION_REDUCTION = 67.0; // %

const int BEST_MODEL_PARAMETERS_M = 213;
const double TRAINING_TIME_HOURS = 4.2;
const int INFERENCE_LATENCY_MS = 89;

// ============================================================
// ADVICE MAPPING (Farmer-Friendly)
// ============================================================

const Map<String, String> STRESS_ADVICE = {
  'healthy': 'Great news! Your crop looks healthy. Keep doing what you\'re doing and check back in a few days.',
  'water_stress': 'Your plants need water. Try watering in the early morning when it\'s cooler. Adding mulch around plants can help keep the soil moist longer.',
  'nutrient_def': 'Your plants need more nutrients. Consider adding fertilizer. If the older leaves are turning yellow first, they may need more nitrogen.',
  'disease_risk': 'Your plant may be sick. Remove any damaged leaves right away. Make sure plants have good airflow and aren\'t too crowded. Avoid watering the leaves directly.',
  'pest_risk': 'Check under the leaves for bugs. You can try washing them off with water, or use a mild soap spray. Remove heavily damaged leaves.',
};

// ============================================================
// FARMER-FRIENDLY MODEL NAMES & DESCRIPTIONS
// ============================================================

/// Friendly names for text analysis options (LLM encoders)
const Map<String, String> LLM_FRIENDLY_NAMES = {
  'distilbert': 'DistilBERT',
  'bert_tiny': 'BERT-tiny',
  'roberta_tiny': 'RoBERTa-tiny',
  'albert_tiny': 'ALBERT-tiny',
  'mobilebert': 'MobileBERT',
};

const Map<String, String> LLM_FRIENDLY_DESCRIPTIONS = {
  'distilbert': 'Fast & light text reader - great for quick scans of your crop description',
  'bert_tiny': 'Compact text reader - runs smoothly even on basic phones',
  'roberta_tiny': 'Accurate text reader - strong understanding of symptom descriptions',
  'albert_tiny': 'Compact shared-param reader — F1 0.544 federated',
  'mobilebert': 'Best fed. F1 (0.636) — top LLM, 100.8% retention; marginally exceeds centralized baseline',
};

/// Friendly names for image analysis options (ViT encoders)
const Map<String, String> VIT_FRIENDLY_NAMES = {
  'vit_base': 'ViT-Base',
  'deit_tiny': 'DeiT-tiny',
  'swin_tiny': 'Swin-tiny',
  'convnext_tiny': 'ConvNeXT-tiny',
  'efficientnet': 'EfficientNet',
};

const Map<String, String> VIT_FRIENDLY_DESCRIPTIONS = {
  'vit_base': 'Best fed. F1 (0.857) — top ViT, 98.9% retention; highest unimodal accuracy',
  'deit_tiny': 'Quick photo scanner — F1 0.853 federated, fast and reliable',
  'swin_tiny': 'Sharp-eye scanner — F1 0.853, great at spotting spots & pest marks',
  'convnext_tiny': 'Modern ConvNet scanner — F1 0.848 federated',
  'efficientnet': 'Battery-saver scanner — F1 0.853, uses less power on your phone',
};

/// Friendly names for fusion strategies
const Map<String, String> FUSION_FRIENDLY_NAMES = {
  'concat': 'Simple Join',
  'attention': 'Smart Focus',
  'gated': 'Balanced Mix',
  'clip': 'CLIP Match',
  'flamingo': 'Flamingo AI',
  'blip2': 'BLIP-2 Deep',
  'coca': 'CoCa Link',
  'unified_io': 'Unified IO',
};

const Map<String, String> FUSION_FRIENDLY_DESCRIPTIONS = {
  'concat': 'Best fed. F1 (0.848) — top VLM, 97.9% retention; simple join outperforms all complex fusions',
  'attention': 'Smart Focus — F1 0.788 federated; focuses on most important parts of photo & text',
  'gated': 'Balanced Mix — F1 0.811 federated; smartly weights photo vs text signal',
  'clip': 'CLIP Match — F1 0.797 federated; contrastive alignment of image and text features',
  'flamingo': 'Flamingo AI — F1 0.802 federated; perceiver-based cross-attention',
  'blip2': 'BLIP-2 Deep — F1 0.802 federated; Q-Former combination of photo + text',
  'coca': 'CoCa Link — F1 0.797 federated; dual contrastive-captioning head',
  'unified_io': 'Unified IO — F1 0.783 federated; universal transformer encoder fusion',
};

/// Friendly names for federated learning modes (paper paradigm terminology)
const Map<String, String> FEDERATED_FRIENDLY_NAMES = {
  'fed_llm': 'Federated LLM',
  'fed_vit': 'Federated ViT',
  'fed_vlm': 'Federated VLM',
};

const Map<String, String> FEDERATED_FRIENDLY_DESCRIPTIONS = {
  'fed_llm': 'Text-only paradigm — MobileBERT, Fed F1 = 0.636, 100.8% retention. Marginally exceeds centralized baseline; lightest compute footprint.',
  'fed_vit': 'Image-only paradigm — ViT-Base, Fed F1 = 0.857, 98.9% retention. Highest unimodal accuracy; minimal 1.1% federated gap.',
  'fed_vlm': 'Text + Image paradigm — Concatenation, Fed F1 = 0.848, 97.9% retention. Best multimodal; simple concat outperforms all 7 complex fusions.',
};

// ============================================================
// UI CONSTANTS
// ============================================================

const double DEFAULT_PADDING = 16.0;
const double CARD_BORDER_RADIUS = 16.0;
const double BUTTON_BORDER_RADIUS = 12.0;

const int SENSOR_POLL_INTERVAL_SECONDS = 5;
const int API_TIMEOUT_SECONDS = 30;
