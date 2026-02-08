import 'package:flutter/material.dart';
import '../theme/app_theme.dart';
import '../constants.dart';

/// Benchmark Results Screen
/// Displays model performance metrics from the paper
class BenchmarkResultsScreen extends StatelessWidget {
  final String apiBase;

  const BenchmarkResultsScreen({super.key, required this.apiBase});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppTheme.backgroundDark,
      appBar: AppBar(
        title: const Text('Performance Results'),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            _buildOverviewCard(),
            const SizedBox(height: 16),
            _buildMetricCard(
              'Best Model Accuracy',
              '${(BEST_ACCURACY * 100).toStringAsFixed(1)}%',
              '$BEST_LLM + $BEST_VIT + $BEST_FUSION',
              Icons.check_circle,
              AppTheme.primaryGreen,
            ),
            const SizedBox(height: 12),
            _buildMetricCard(
              'Centralized Macro-F1',
              '${(BEST_MACRO_F1_CENTRALIZED * 100).toStringAsFixed(1)}%',
              'Best single-server performance',
              Icons.analytics,
              AppTheme.accentBlue,
            ),
            const SizedBox(height: 12),
            _buildMetricCard(
              'Federated Macro-F1',
              '${(BEST_MACRO_F1_FEDERATED * 100).toStringAsFixed(1)}%',
              'Privacy-preserving distributed training',
              Icons.security,
              AppTheme.accentCyan,
            ),
            const SizedBox(height: 12),
            _buildMetricCard(
              'Multimodal Improvement',
              '+$MULTIMODAL_IMPROVEMENT%',
              'Over single-mode analysis',
              Icons.trending_up,
              AppTheme.accentOrange,
            ),
            const SizedBox(height: 16),
            _buildDetailsCard(),
          ],
        ),
      ),
    );
  }

  Widget _buildOverviewCard() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        gradient: AppTheme.primaryGradient,
        borderRadius: BorderRadius.circular(16),
      ),
      child: const Column(
        children: [
          Icon(Icons.emoji_events, color: Colors.white, size: 40),
          SizedBox(height: 8),
          Text(
            'FarmFederate Results',
            style: TextStyle(
              color: Colors.white,
              fontWeight: FontWeight.bold,
              fontSize: 20,
            ),
          ),
          SizedBox(height: 4),
          Text(
            '$TOTAL_CONFIGURATIONS model configurations evaluated',
            style: TextStyle(color: Colors.white70),
          ),
        ],
      ),
    );
  }

  Widget _buildMetricCard(
      String title, String value, String subtitle, IconData icon, Color color) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AppTheme.cardDark,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: color.withValues(alpha: 0.3)),
      ),
      child: Row(
        children: [
          Container(
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: color.withValues(alpha: 0.2),
              borderRadius: BorderRadius.circular(12),
            ),
            child: Icon(icon, color: color, size: 28),
          ),
          const SizedBox(width: 16),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(title, style: const TextStyle(color: Colors.white70, fontSize: 13)),
                Text(
                  value,
                  style: TextStyle(
                    color: color,
                    fontWeight: FontWeight.bold,
                    fontSize: 24,
                  ),
                ),
                Text(subtitle, style: const TextStyle(color: Colors.white38, fontSize: 12)),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildDetailsCard() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AppTheme.cardDark,
        borderRadius: BorderRadius.circular(16),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            'Technical Details',
            style: TextStyle(
              color: Colors.white,
              fontWeight: FontWeight.bold,
              fontSize: 16,
            ),
          ),
          const SizedBox(height: 12),
          _buildDetailRow('Model Parameters', '${BEST_MODEL_PARAMETERS_M}M'),
          _buildDetailRow('Training Time', '${TRAINING_TIME_HOURS}h'),
          _buildDetailRow('Inference Latency', '${INFERENCE_LATENCY_MS}ms'),
          _buildDetailRow('Fed/Centralized Ratio', '$FED_CENTRALIZED_RATIO%'),
          _buildDetailRow('Communication Reduction', '$COMMUNICATION_REDUCTION%'),
          _buildDetailRow('SOTA Improvement', '+$SOTA_IMPROVEMENT%'),
          _buildDetailRow('Federated Clients', '$NUM_CLIENTS'),
          _buildDetailRow('Federated Rounds', '$FEDERATED_ROUNDS'),
        ],
      ),
    );
  }

  Widget _buildDetailRow(String label, String value) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(label, style: const TextStyle(color: Colors.white70)),
          Text(
            value,
            style: const TextStyle(
              color: Colors.white,
              fontWeight: FontWeight.bold,
              fontFamily: 'monospace',
            ),
          ),
        ],
      ),
    );
  }
}
