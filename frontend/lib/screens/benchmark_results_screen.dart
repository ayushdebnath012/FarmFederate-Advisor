import 'package:flutter/material.dart';
import '../theme/app_theme.dart';
import '../constants.dart';

/// Benchmark Results Screen
/// Displays the three-paradigm cross-comparison from the paper
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
            _buildParadigmTable(),
            const SizedBox(height: 16),
            _buildLLMTable(),
            const SizedBox(height: 16),
            _buildViTTable(),
            const SizedBox(height: 16),
            _buildVLMTable(),
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
            'FarmFederate — Cross-Paradigm Results',
            style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold, fontSize: 18),
            textAlign: TextAlign.center,
          ),
          SizedBox(height: 4),
          Text(
            'Federated LLM · Federated ViT · Federated VLM\nK=3 clients · T=8 rounds · E=3 local epochs · FedAvg',
            style: TextStyle(color: Colors.white70, fontSize: 12),
            textAlign: TextAlign.center,
          ),
        ],
      ),
    );
  }

  Widget _buildParadigmTable() {
    return _buildTableCard(
      title: 'Cross-Paradigm Comparison',
      subtitle: 'Best model per paradigm',
      color: AppTheme.primaryGreen,
      headers: const ['Paradigm', 'Best Model', 'Fed F1', 'Retention'],
      rows: const [
        ['Fed. LLM', 'ALBERT-tiny', '0.548', '92.9%'],
        ['Fed. ViT', 'ConvNeXT-tiny', '0.659', '86.1%'],
        ['Fed. VLM', 'CLIP fusion', '0.785', '92.6%'],
      ],
      bestRow: 2,
    );
  }

  Widget _buildLLMTable() {
    return _buildTableCard(
      title: 'Federated LLM — All Models',
      subtitle: 'Text paradigm (5 encoders)',
      color: const Color(0xFF4169E1),
      headers: const ['Model', 'Cent F1', 'Fed F1'],
      rows: const [
        ['DistilBERT', '0.571', '0.530'],
        ['BERT-tiny', '0.585', '0.543'],
        ['RoBERTa-tiny', '0.585', '0.541'],
        ['ALBERT-tiny', '0.590', '0.548'],
        ['MobileBERT', '0.535', '0.497'],
      ],
      bestRow: 3,
    );
  }

  Widget _buildViTTable() {
    return _buildTableCard(
      title: 'Federated ViT — All Models',
      subtitle: 'Image paradigm (5 encoders)',
      color: const Color(0xFF228B22),
      headers: const ['Model', 'Cent F1', 'Fed F1'],
      rows: const [
        ['ViT-Base', '0.696', '0.599'],
        ['DeiT-tiny', '0.747', '0.643'],
        ['Swin-tiny', '0.724', '0.623'],
        ['ConvNeXT-tiny', '0.765', '0.659'],
        ['EfficientNet', '0.724', '0.623'],
      ],
      bestRow: 3,
    );
  }

  Widget _buildVLMTable() {
    return _buildTableCard(
      title: 'Federated VLM — All Fusions',
      subtitle: 'Multimodal paradigm (8 fusion strategies)',
      color: const Color(0xFFB22222),
      headers: const ['Fusion', 'Cent F1', 'Fed F1'],
      rows: const [
        ['Concat', '0.797', '0.738'],
        ['Cross-Attention', '0.807', '0.747'],
        ['Gated', '0.779', '0.722'],
        ['CLIP', '0.848', '0.785'],
        ['Flamingo', '0.816', '0.755'],
        ['BLIP-2', '0.834', '0.771'],
        ['CoCa', '0.783', '0.725'],
        ['Unified-IO', '0.802', '0.743'],
      ],
      bestRow: 3,
    );
  }

  Widget _buildTableCard({
    required String title,
    required String subtitle,
    required Color color,
    required List<String> headers,
    required List<List<String>> rows,
    required int bestRow,
  }) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AppTheme.cardDark,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: color.withValues(alpha: 0.3)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(width: 4, height: 20, decoration: BoxDecoration(color: color, borderRadius: BorderRadius.circular(2))),
              const SizedBox(width: 10),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(title, style: const TextStyle(color: Colors.white, fontWeight: FontWeight.bold, fontSize: 15)),
                    Text(subtitle, style: const TextStyle(color: Colors.white38, fontSize: 11)),
                  ],
                ),
              ),
            ],
          ),
          const SizedBox(height: 14),
          // Header row
          Row(
            children: headers.map((h) => Expanded(
              child: Text(h, style: TextStyle(color: color, fontSize: 11, fontWeight: FontWeight.bold)),
            )).toList(),
          ),
          const SizedBox(height: 6),
          Container(height: 1, color: Colors.white12),
          const SizedBox(height: 6),
          // Data rows
          ...rows.asMap().entries.map((entry) {
            final i = entry.key;
            final row = entry.value;
            final isBest = i == bestRow;
            return Container(
              margin: const EdgeInsets.only(bottom: 4),
              padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 5),
              decoration: BoxDecoration(
                color: isBest ? color.withValues(alpha: 0.12) : Colors.transparent,
                borderRadius: BorderRadius.circular(6),
                border: isBest ? Border.all(color: color.withValues(alpha: 0.4)) : null,
              ),
              child: Row(
                children: row.asMap().entries.map((cell) {
                  final isFirst = cell.key == 0;
                  return Expanded(
                    child: Row(
                      children: [
                        Text(
                          cell.value,
                          style: TextStyle(
                            color: isBest ? Colors.white : Colors.white70,
                            fontSize: 12,
                            fontWeight: isBest ? FontWeight.bold : FontWeight.normal,
                          ),
                        ),
                        if (isBest && isFirst) ...[
                          const SizedBox(width: 4),
                          Container(
                            padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 1),
                            decoration: BoxDecoration(
                              color: color.withValues(alpha: 0.3),
                              borderRadius: BorderRadius.circular(3),
                            ),
                            child: Text('BEST', style: TextStyle(color: color, fontSize: 8, fontWeight: FontWeight.bold)),
                          ),
                        ],
                      ],
                    ),
                  );
                }).toList(),
              ),
            );
          }),
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
            'Federated Setup',
            style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold, fontSize: 16),
          ),
          const SizedBox(height: 12),
          _buildDetailRow('Federated Clients (K)', '$NUM_CLIENTS'),
          _buildDetailRow('Federated Rounds (T)', '$FEDERATED_ROUNDS'),
          _buildDetailRow('Local Epochs (E)', '$LOCAL_EPOCHS'),
          _buildDetailRow('Aggregation', 'FedAvg (weighted by client size)'),
          _buildDetailRow('Data Split', 'Non-IID Dirichlet (α=${NON_IID_ALPHA})'),
          _buildDetailRow('Avg Fed/Cent Retention', '$FED_CENTRALIZED_RATIO%'),
          _buildDetailRow('Best Fed F1 (VLM-CLIP)', '${BEST_MACRO_F1_FEDERATED}'),
          _buildDetailRow('Best Cent F1 (VLM-CLIP)', '${BEST_MACRO_F1_CENTRALIZED}'),
          _buildDetailRow('VLM gain over LLM', '+${MULTIMODAL_IMPROVEMENT}% Fed F1'),
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
            style: const TextStyle(color: Colors.white, fontWeight: FontWeight.bold, fontFamily: 'monospace'),
          ),
        ],
      ),
    );
  }
}
