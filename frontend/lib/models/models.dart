import 'package:flutter/material.dart';
import '../constants.dart';

// ============================================================
// STRESS CLASS ENUM WITH DISPLAY PROPERTIES
// ============================================================

enum StressClass {
  healthy,
  waterStress,
  nutrientDef,
  diseaseRisk,
  pestRisk,
}

extension StressClassExtension on StressClass {
  String get label {
    switch (this) {
      case StressClass.healthy:
        return 'healthy';
      case StressClass.waterStress:
        return 'water_stress';
      case StressClass.nutrientDef:
        return 'nutrient_def';
      case StressClass.diseaseRisk:
        return 'disease_risk';
      case StressClass.pestRisk:
        return 'pest_risk';
    }
  }

  String get displayName {
    return STRESS_DISPLAY_NAMES[label] ?? label;
  }

  Color get color {
    switch (this) {
      case StressClass.healthy:
        return const Color(0xFF4CAF50);
      case StressClass.waterStress:
        return const Color(0xFFFF9800);
      case StressClass.nutrientDef:
        return const Color(0xFFFFEB3B);
      case StressClass.diseaseRisk:
        return const Color(0xFFF44336);
      case StressClass.pestRisk:
        return const Color(0xFF9C27B0);
    }
  }

  IconData get icon {
    switch (this) {
      case StressClass.healthy:
        return Icons.check_circle;
      case StressClass.waterStress:
        return Icons.water_drop;
      case StressClass.nutrientDef:
        return Icons.grass;
      case StressClass.diseaseRisk:
        return Icons.coronavirus;
      case StressClass.pestRisk:
        return Icons.bug_report;
    }
  }

  static StressClass fromLabel(String label) {
    switch (label.toLowerCase()) {
      case 'healthy':
        return StressClass.healthy;
      case 'water_stress':
      case 'needs water':
        return StressClass.waterStress;
      case 'nutrient_def':
      case 'needs fertilizer':
        return StressClass.nutrientDef;
      case 'disease_risk':
      case 'may be sick':
        return StressClass.diseaseRisk;
      case 'pest_risk':
      case 'bug problem':
        return StressClass.pestRisk;
      default:
        return StressClass.healthy;
    }
  }
}

// ============================================================
// CLASS SCORE - Individual prediction score for a stress class
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
    final label = json['label'] as String? ?? 'healthy';
    return ClassScore(
      stressClass: StressClassExtension.fromLabel(label),
      probability: (json['prob'] as num?)?.toDouble() ?? 0.0,
      threshold: (json['threshold'] as num?)?.toDouble() ?? 0.2,
      isActive: json['active'] as bool? ?? false,
    );
  }
}

// ============================================================
// MODEL INFO - Information about the current active model
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
      id: json['id'] as String? ?? '',
      name: json['name'] as String? ?? 'Unknown Model',
      description: json['description'] as String? ?? '',
      accuracy: (json['accuracy'] as num?)?.toDouble() ?? 0.0,
      llmEncoder: json['llm_encoder'] as String? ?? '',
      vitEncoder: json['vit_encoder'] as String? ?? '',
      fusionStrategy: json['fusion_strategy'] as String? ?? '',
    );
  }
}

// ============================================================
// PREDICTION RESULT - Full result from /predict endpoint
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

  bool get hasStress =>
      predictedLabel.toLowerCase() != 'healthy' || activeStresses.isNotEmpty;

  Color get severityColor {
    if (!hasStress) return const Color(0xFF4CAF50);
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

    // Parse all scores
    final allScores = (result['all_scores'] as List? ?? [])
        .map((e) => ClassScore.fromJson(e as Map<String, dynamic>))
        .toList();

    // Parse active labels
    final activeLabels = (result['active_labels'] as List? ?? [])
        .map((e) => ClassScore.fromJson(e as Map<String, dynamic>))
        .toList();

    // Mark active scores
    final activeSet =
        activeLabels.map((a) => a.stressClass.label).toSet();
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

    // Parse uncertainty if present
    Map<String, double>? unc;
    if (result['uncertainty'] != null) {
      unc = (result['uncertainty'] as Map<String, dynamic>)
          .map((k, v) => MapEntry(k, (v as num).toDouble()));
    }

    // Predicted label
    final predicted = result['predicted'] as String? ??
        (allScores.isNotEmpty
            ? (List<ClassScore>.from(allScores)
                  ..sort(
                      (a, b) => b.probability.compareTo(a.probability)))
                .first
                .stressClass
                .label
            : 'healthy');

    // Advice
    final advice = json['advice'] as String? ??
        STRESS_ADVICE[predicted] ??
        '';

    return PredictionResult(
      predictedLabel: predicted,
      scores: scoredList.isNotEmpty ? scoredList : allScores,
      activeStresses:
          scoredList.where((s) => s.isActive).toList(),
      advice: advice,
      uncertainty: unc,
      modelUsed: result['model'] as String?,
    );
  }
}
