import 'dart:io' show Platform;
import 'package:flutter/foundation.dart' show kIsWeb;

const String PRODUCTION_BACKEND = "https://farmfederate-advisor.onrender.com";

const String _LOCALHOST = "http://localhost:8000";
const String _ANDROID_EMULATOR = "http://10.0.2.2:8000";
const String _IOS_SIMULATOR = "http://localhost:8000";

const String _LAN_BACKEND = "http://192.168.1.100:8000";

const bool IS_PRODUCTION = false;

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

String get DEFAULT_BACKEND => getBackendUrl();

const String PREDICT_PATH = "/predict";
const String HEALTH_PATH = "/health";
const String PROFILES_PATH = "/profiles";
const String SENSORS_PATH = "/sensors/latest";
const String LOOKUP_PATH = "/lookup";
const String LOOKUP_METRICS_PATH = "/lookup/metrics";

const String MQTT_BROKER = "ws://192.168.1.100:9001";
const String MQTT_USERNAME = "";
const String MQTT_PASSWORD = "";
const String SENSOR_TOPIC = "farm/sensors/#";
const String CMD_TOPIC = "farm/cmd/";

const int TOTAL_FIELD_IMAGES = 200;
const int TOTAL_IMAGE_TENSORS = 792;
const int REFERENCE_RECORDS = 633;
const int CHECK_RECORDS = 79;
const int TOTAL_TEXT_RECORDS = 3000;
const int YOLO_ANNOTATIONS = 371;
const int NUM_CLASSES = 5;

const int TOTAL_RECORDS = TOTAL_IMAGE_TENSORS;
const int REVIEW_RECORDS = 80;

const int IMAGE_SIZE = 224;
const int IMAGE_CHANNELS = 3;

const int MAX_SEQUENCE_LENGTH = 128;
const int AVG_SEQUENCE_LENGTH = 45;
const int VOCABULARY_SIZE = 30522;

const int BATCH_SIZE = 32;
const double ADJUSTMENT_RATE = 2e-4;
const int PASSES = 30;
const String OPTIMIZER = "AdamW";
const double WEIGHT_DECAY = 0.01;

const int NUM_CLIENTS = 3;
const int NETWORK_ROUNDS = 8;
const int LOCAL_PASSES = 3;
const double NON_IID_ALPHA = 1.0;

const int LOOKUP_SYNC_ROUNDS = 10;
const int LOOKUP_TOP_K = 5;
const int LOOKUP_QUERY_DIM = 128;
const double LOOKUP_EMA_MU = 0.9;
const int LOOKUP_RECORDS_PER_FARM = 400;
const double LOOKUP_NON_IID_ALPHA = 0.5;

const double LOOKUP_MACRO_F1 = 1.000;
const double LOOKUP_MICRO_F1 = 1.000;
const double LOOKUP_RECALL_AT_5 = 0.129;
const double LOOKUP_MRR = 0.100;
const double LOOKUP_NDCG_AT_5 = 0.172;
const double LOOKUP_KB_COVERAGE = 1.000;
const double LOOKUP_INDEX_DRIFT = 0.156;
const int LOOKUP_KB_SIZE = 15;

const double FOCAL_GAMMA = 2.0;
const double DIVERSITY_LAMBDA = 0.1;

const List<String> STRESS_LABELS = [
  'LEAF_BLIGHT',
  'LEAF_HOPPERS',
  'LEAF_RUST',
  'LOOPER_CATERPILLARS',
  'MOSQUITO_BUG',
];

const Map<String, String> STRESS_DISPLAY_NAMES = {
  'LEAF_BLIGHT': 'Leaf Blight',
  'LEAF_HOPPERS': 'Leaf Hoppers',
  'LEAF_RUST': 'Leaf Rust',
  'LOOPER_CATERPILLARS': 'Looper Caterpillars',
  'MOSQUITO_BUG': 'Mosquito Bug',
};

const Map<String, String> STRESS_SHORT = {
  'LEAF_BLIGHT': 'Blight',
  'LEAF_HOPPERS': 'Hoppers',
  'LEAF_RUST': 'Rust',
  'LOOPER_CATERPILLARS': 'Loopers',
  'MOSQUITO_BUG': 'Mosquito',
};

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

const List<String> TEXT_ANALYZERS = [
  'text_core_a',
  'text_core_b',
  'text_core_c',
  'text_core_d',
  'text_core_e',
];

const List<String> IMAGE_ANALYZERS = [
  'image_core_a',
  'image_core_b',
  'image_core_c',
  'image_core_d',
  'image_core_e',
];

const List<String> FUSION_ANALYZERS = [
  'fusion_core_a',
  'fusion_core_b',
  'fusion_core_c',
  'fusion_core_d',
  'fusion_core_e',
  'fusion_core_f',
  'fusion_core_g',
  'fusion_core_h',
];

const List<String> NETWORK_MODES = [
  'network_text',
  'network_image',
  'network_fusion'
];

const int TOTAL_CONFIGURATIONS = 203;

const String BEST_TEXT_ANALYZER = 'text_core_a';
const String BEST_IMAGE_ANALYZER = 'image_core_a';
const String BEST_FUSION = 'fusion_core_a';

const double CENT_F1_TEXT = 0.427;
const double CENT_F1_IMAGE = 0.886;
const double CENT_F1_FUSION = 0.873;

const double NETWORK_F1_TEXT = 0.460;
const double NETWORK_F1_IMAGE = 0.886;
const double NETWORK_F1_FUSION = 0.861;

const double RETENTION_TEXT = 107.7;
const double RETENTION_IMAGE = 100.0;
const double RETENTION_FUSION = 98.6;

const double BEST_MACRO_F1_BASELINE = 0.949;
const double BEST_MACRO_F1_NETWORK = 0.861;
const double BEST_ACCURACY = 0.949;

const double MULTIMODAL_IMPROVEMENT = 87.2;
const double NETWORK_BASELINE_RATIO = 102.1;
const double COMMUNICATION_REDUCTION = 67.0;

const int BEST_PROFILE_PARAMETERS_M = 16;
const double CALIBRATION_TIME_HOURS = 4.2;
const int INFERENCE_LATENCY_MS = 89;

const Map<String, double> TEXT_CENT_F1 = {
  'text_core_a': 0.487,
  'text_core_b': 0.463,
  'text_core_c': 0.450,
  'text_core_d': 0.433,
  'text_core_e': 0.417,
};

const Map<String, double> IMAGE_CENT_F1 = {
  'image_core_a': 0.911,
  'image_core_b': 0.899,
  'image_core_c': 0.873,
  'image_core_d': 0.873,
  'image_core_e': 0.861,
};

const Map<String, double> FUSION_CENT_F1 = {
  'fusion_core_a': 0.949,
  'fusion_core_b': 0.937,
  'fusion_core_c': 0.924,
  'fusion_core_d': 0.911,
  'fusion_core_e': 0.911,
  'fusion_core_f': 0.899,
  'fusion_core_g': 0.899,
  'fusion_core_h': 0.886,
};

const Map<String, double> FUSION_PARAMS_M = {
  'fusion_core_a': 15.7,
  'fusion_core_b': 15.9,
  'fusion_core_c': 15.9,
  'fusion_core_d': 16.1,
  'fusion_core_e': 17.2,
  'fusion_core_f': 15.8,
  'fusion_core_g': 16.2,
  'fusion_core_h': 15.6,
};

const Map<String, double> PROFILE_SIZE_MB = {
  'text_core_a': 42.4,
  'text_core_b': 42.4,
  'text_core_c': 42.4,
  'text_core_d': 42.4,
  'text_core_e': 42.4,
  'image_core_a': 19.0,
  'image_core_b': 19.0,
  'image_core_c': 19.0,
  'image_core_d': 19.0,
  'image_core_e': 19.0,
  'fusion_core_a': 60.1,
  'fusion_core_b': 60.8,
  'fusion_core_c': 60.6,
  'fusion_core_d': 61.6,
  'fusion_core_e': 65.6,
  'fusion_core_f': 60.3,
  'fusion_core_g': 61.8,
  'fusion_core_h': 59.5,
};

const Map<String, String> TEXT_FRIENDLY_NAMES = {
  'text_core_a': 'Text Core A',
  'text_core_b': 'Text Core B',
  'text_core_c': 'Text Core C',
  'text_core_d': 'Text Core D',
  'text_core_e': 'Text Core E',
};

const Map<String, String> TEXT_FRIENDLY_DESCRIPTIONS = {
  'text_core_a':
      'Best Text - Base F1 0.487 | top text profile for tea disease symptom descriptions',
  'text_core_b':
      'Text Core B - Base F1 0.463 | robustly calibrated, strong on varied phrasings',
  'text_core_c':
      'Text Core C - Base F1 0.450 | parameter-sharing reduces memory footprint',
  'text_core_d': 'Text Core D - Base F1 0.433 | fast 6-layer compact profile',
  'text_core_e':
      'Text Core E - Base F1 0.417 | mobile-optimised, lowest size (42 MB)',
};

const Map<String, String> IMAGE_FRIENDLY_NAMES = {
  'image_core_a': 'Image Core A',
  'image_core_b': 'Image Core B',
  'image_core_c': 'Image Core C',
  'image_core_d': 'Image Core D',
  'image_core_e': 'Image Core E',
};

const Map<String, String> IMAGE_FRIENDLY_DESCRIPTIONS = {
  'image_core_a':
      'Best Image - Base F1 0.911 | top image profile for tea leaf disease detection',
  'image_core_b':
      'Image Core B - Base F1 0.899 | modern image profile; excellent lesion detection',
  'image_core_c':
      'Image Core C - Base F1 0.873 | field-efficient image profile',
  'image_core_d':
      'Image Core D - Base F1 0.873 | hierarchical profile; good at scale variation',
  'image_core_e':
      'Image Core E - Base F1 0.861 | standard patch-based image profile',
};

const Map<String, String> FUSION_FRIENDLY_NAMES = {
  'fusion_core_a': 'Fusion Core A',
  'fusion_core_b': 'Fusion Core B',
  'fusion_core_c': 'Fusion Core C',
  'fusion_core_d': 'Fusion Core D',
  'fusion_core_e': 'Fusion Core E',
  'fusion_core_f': 'Gated Mix',
  'fusion_core_g': 'Fusion Core G',
  'fusion_core_h': 'Simple Join',
};

const Map<String, String> FUSION_FRIENDLY_DESCRIPTIONS = {
  'fusion_core_a':
      'Best Fusion - Base F1 0.949 | Fusion Core A paired evidence alignment; jointly aligns tea disease text and image',
  'fusion_core_b': 'Fusion Core B - Base F1 0.937 | rich text-image comparison',
  'fusion_core_c':
      'Fusion Core C - Base F1 0.924 | balanced weighting over text and image cues',
  'fusion_core_d': 'Fusion Core D - Base F1 0.911 | gated cross-source fusion',
  'fusion_core_e': 'Fusion Core E - Base F1 0.911 | universal fusion profile',
  'fusion_core_f':
      'Gated - Base F1 0.899 | adaptive gate balances text vs. image contribution',
  'fusion_core_g':
      'Fusion Core G - Base F1 0.899 | dual evidence-summary fusion',
  'fusion_core_h':
      'Concat - Base F1 0.886 | simple concatenation of text and visual indicators',
};

const Map<String, String> NETWORK_FRIENDLY_NAMES = {
  'network_text': 'Network Text',
  'network_image': 'Network Image',
  'network_fusion': 'Network Fusion',
};

const Map<String, String> NETWORK_FRIENDLY_DESCRIPTIONS = {
  'network_text':
      'Text-only group - Text Core A | Net F1 0.460 | 107.7% retention (exceeds baseline result)',
  'network_image':
      'Image-only group - Image Core A | Net F1 0.886 | 100.0% retention | no accuracy loss in the network run',
  'network_fusion':
      'Text + Image group - Fusion Core A | Net F1 0.861 | 98.6% retention | best overall network performance',
};

const double DEFAULT_PADDING = 16.0;
const double CARD_BORDER_RADIUS = 16.0;
const double BUTTON_BORDER_RADIUS = 12.0;

const int SENSOR_POLL_INTERVAL_SECONDS = 5;
const int API_TIMEOUT_SECONDS = 30;
