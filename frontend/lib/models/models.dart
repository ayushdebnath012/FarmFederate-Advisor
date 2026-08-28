import 'package:flutter/material.dart';
import '../constants.dart';

// ============================================================
// TEA DISEASE CLASS ENUM
// Matches backend STRESS_LABELS exactly (Colab / paper §3)
// 0=LEAF_BLIGHT  1=LEAF_HOPPERS  2=LEAF_RUST
// 3=LOOPER_CATERPILLARS  4=MOSQUITO_BUG
// ============================================================

enum StressClass {
  leafBlight,
  leafHoppers,
  leafRust,
  looperCaterpillars,
  mosquitoBug,
}

extension StressClassExtension on StressClass {
  String get label {
    switch (this) {
      case StressClass.leafBlight:
        return 'LEAF_BLIGHT';
      case StressClass.leafHoppers:
        return 'LEAF_HOPPERS';
      case StressClass.leafRust:
        return 'LEAF_RUST';
      case StressClass.looperCaterpillars:
        return 'LOOPER_CATERPILLARS';
      case StressClass.mosquitoBug:
        return 'MOSQUITO_BUG';
    }
  }

  String get displayName => STRESS_DISPLAY_NAMES[label] ?? label;

  String get shortName => STRESS_SHORT[label] ?? label;

  Color get color {
    switch (this) {
      case StressClass.leafBlight:
        return const Color(0xFF22A822); // green
      case StressClass.leafHoppers:
        return Colors.redAccent;
      case StressClass.leafRust:
        return const Color(0xFF1E90FF); // blue
      case StressClass.looperCaterpillars:
        return Colors.purpleAccent;
      case StressClass.mosquitoBug:
        return const Color(0xFF00CED1); // teal
    }
  }

  IconData get icon {
    switch (this) {
      case StressClass.leafBlight:
        return Icons.blur_on;
      case StressClass.leafHoppers:
        return Icons.bug_report;
      case StressClass.leafRust:
        return Icons.circle;
      case StressClass.looperCaterpillars:
        return Icons.pest_control;
      case StressClass.mosquitoBug:
        return Icons.pest_control_rodent;
    }
  }

  static StressClass fromLabel(String raw) {
    switch (raw.toUpperCase().trim()) {
      case 'LEAF_BLIGHT':
      case 'BLIGHT':
      case 'BROWN_BLIGHT':
        return StressClass.leafBlight;
      case 'LEAF_HOPPERS':
      case 'LEAF_HOPPER':
      case 'HOPPERS':
      case 'LEAFHOPPER':
      case 'TIP_BURN':
        return StressClass.leafHoppers;
      case 'LEAF_RUST':
      case 'RUST':
      case 'ORANGE_PUSTULE':
        return StressClass.leafRust;
      case 'LOOPER_CATERPILLARS':
      case 'LOOPER':
      case 'CATERPILLAR':
      case 'LOOPER_CATERPILLAR':
        return StressClass.looperCaterpillars;
      case 'MOSQUITO_BUG':
      case 'MOSQUITO':
      case 'HELOPELTIS':
        return StressClass.mosquitoBug;
      default:
        // graceful fallback — pick highest-probability class instead of crashing
        return StressClass.leafBlight;
    }
  }
}

// ============================================================
// CLASS SCORE — individual prediction score for one disease
// ============================================================

class ClassScore {
  final StressClass stressClass;
  final double probability;
  final double threshold;
  final bool isActive;

  ClassScore({
    required this.stressClass,
    required this.probability,
    this.threshold = 0.2,
    this.isActive = false,
  });

  double get probabilityPercent => probability * 100;

  factory ClassScore.fromJson(Map<String, dynamic> json) {
    final label = json['label'] as String? ?? 'LEAF_BLIGHT';
    return ClassScore(
      stressClass: StressClassExtension.fromLabel(label),
      probability: (json['prob'] as num?)?.toDouble() ?? 0.0,
      threshold: (json['threshold'] as num?)?.toDouble() ?? 0.2,
      isActive: json['active'] as bool? ?? false,
    );
  }
}

// ============================================================
// MODEL INFO — current active model configuration
// ============================================================

class ModelInfo {
  final String id;
  final String name;
  final String description;
  final double accuracy;
  final String llmEncoder;
  final String vitEncoder;
  final String fusionStrategy;

  ModelInfo({
    required this.id,
    required this.name,
    this.description = '',
    this.accuracy = 0.0,
    this.llmEncoder = '',
    this.vitEncoder = '',
    this.fusionStrategy = '',
  });

  factory ModelInfo.fromJson(Map<String, dynamic> json) {
    return ModelInfo(
      id: json['id'] as String? ?? 'default',
      name: json['name'] as String? ?? 'Tea Disease Analyser',
      description: json['description'] as String? ??
          'Multimodal tea leaf disease classifier',
      accuracy:
          (json['accuracy'] as num?)?.toDouble() ?? BEST_MACRO_F1_CENTRALIZED,
      llmEncoder: json['llm_encoder'] as String? ?? BEST_LLM,
      vitEncoder: json['vit_encoder'] as String? ?? BEST_VIT,
      fusionStrategy: json['fusion_strategy'] as String? ?? BEST_FUSION,
    );
  }
}

// ============================================================
// PREDICTION RESULT — full result from /predict endpoint
// ============================================================

class PredictionResult {
  final String predictedLabel;
  final List<ClassScore> scores;
  final List<ClassScore> activeStresses;
  final String advice;
  final Map<String, double>? uncertainty;
  final String? modelUsed;

  PredictionResult({
    required this.predictedLabel,
    required this.scores,
    required this.activeStresses,
    required this.advice,
    this.uncertainty,
    this.modelUsed,
  });

  // All 5 classes are tea diseases — prediction always represents a disease
  bool get hasStress => true;

  Color get severityColor {
    if (activeStresses.isNotEmpty) {
      return activeStresses.first.stressClass.color;
    }
    return StressClassExtension.fromLabel(predictedLabel).color;
  }

  List<ClassScore> topN(int n) {
    final sorted = List<ClassScore>.from(scores)
      ..sort((a, b) => b.probability.compareTo(a.probability));
    return sorted.take(n).toList();
  }

  factory PredictionResult.fromJson(Map<String, dynamic> json) {
    final result = json['result'] as Map<String, dynamic>? ?? {};

    final allScores = (result['all_scores'] as List? ?? [])
        .map((e) => ClassScore.fromJson(e as Map<String, dynamic>))
        .toList();

    final activeLabels = (result['active_labels'] as List? ?? [])
        .map((e) => ClassScore.fromJson(e as Map<String, dynamic>))
        .toList();

    final activeSet = activeLabels.map((a) => a.stressClass.label).toSet();
    final scoredList = allScores.map((s) {
      if (activeSet.contains(s.stressClass.label)) {
        return ClassScore(
          stressClass: s.stressClass,
          probability: s.probability,
          threshold: s.threshold,
          isActive: true,
        );
      }
      return s;
    }).toList();

    Map<String, double>? unc;
    if (result['uncertainty'] != null) {
      unc = (result['uncertainty'] as Map<String, dynamic>)
          .map((k, v) => MapEntry(k, (v as num).toDouble()));
    }

    final predicted = result['predicted'] as String? ??
        (allScores.isNotEmpty
            ? (List<ClassScore>.from(allScores)
                  ..sort((a, b) => b.probability.compareTo(a.probability)))
                .first
                .stressClass
                .label
            : 'LEAF_BLIGHT');

    final advice = json['advice'] as String? ??
        STRESS_ADVICE[predicted.toUpperCase()] ??
        '';

    return PredictionResult(
      predictedLabel: predicted,
      scores: scoredList.isNotEmpty ? scoredList : allScores,
      activeStresses: scoredList.where((s) => s.isActive).toList(),
      advice: advice,
      uncertainty: unc,
      modelUsed: result['model'] as String?,
    );
  }
}
