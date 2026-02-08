// FarmFederate Constants
// Configuration values matching the paper specifications

import 'dart:io' show Platform;
import 'package:flutter/foundation.dart' show kIsWeb;

// ============================================================
// BACKEND CONFIGURATION
// ============================================================

/// Production backend URL - UPDATE THIS for deployment
/// Set this to your deployed backend URL (e.g., https://api.farmfederate.com)
const String PRODUCTION_BACKEND = "https://your-backend-url.com";

/// Local development backend URLs
const String _LOCALHOST = "http://localhost:8001";
const String _ANDROID_EMULATOR = "http://10.0.2.2:8001";
const String _IOS_SIMULATOR = "http://localhost:8001";

/// LAN IP for testing on real devices (update to your machine's IP)
const String _LAN_BACKEND = "http://192.168.1.100:8001";

/// Set to true when deploying to production
const bool IS_PRODUCTION = false;

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

/// Federated parameters
const int NUM_CLIENTS = 5;
const int FEDERATED_ROUNDS = 50;
const int LOCAL_EPOCHS = 3;
const double NON_IID_ALPHA = 0.5;

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
  'bert',
  'roberta',
  'distilbert',
  'albert',
  'xlnet',
];

/// 5 ViT Encoder variants (Table 9)
const List<String> VIT_ENCODERS = [
  'vit_base',
  'deit',
  'swin',
  'beit',
  'convnext',
];

/// 8 VLM Fusion strategies (Table 10)
const List<String> VLM_FUSIONS = [
  'concat',
  'cross_attention',
  'gated',
  'clip',
  'flamingo',
  'blip2',
  'coca',
  'unified_io',
];

/// Total configurations: 5 x 5 x 8 = 200
const int TOTAL_CONFIGURATIONS = 200;

// ============================================================
// BEST CONFIGURATION (Table 22)
// ============================================================

const String BEST_LLM = 'roberta';
const String BEST_VIT = 'swin';
const String BEST_FUSION = 'blip2';

const double BEST_MACRO_F1_CENTRALIZED = 0.847;
const double BEST_MACRO_F1_FEDERATED = 0.798;
const double BEST_ACCURACY = 0.874;

const double MULTIMODAL_IMPROVEMENT = 12.3; // % over unimodal
const double SOTA_IMPROVEMENT = 5.7; // % over SOTA
const double FED_CENTRALIZED_RATIO = 94.2; // %
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
// UI CONSTANTS
// ============================================================

const double DEFAULT_PADDING = 16.0;
const double CARD_BORDER_RADIUS = 16.0;
const double BUTTON_BORDER_RADIUS = 12.0;

const int SENSOR_POLL_INTERVAL_SECONDS = 5;
const int API_TIMEOUT_SECONDS = 30;
