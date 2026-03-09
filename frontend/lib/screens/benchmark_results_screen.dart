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
            _buildRAGCard(),
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
            'Federated LLM · Federated ViT · Federated VLM · Federated RAG\nK=3 clients · T=8 rounds · E=3 local epochs · FedAvg · avg 99.2% retention',
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
        ['Fed. LLM', 'MobileBERT', '0.636', '100.8%'],
        ['Fed. ViT', 'ViT-Base', '0.857', '98.9%'],
        ['Fed. VLM', 'Concatenation', '0.848', '97.9%'],
      ],
      bestRow: 2,
    );
  }

  Widget _buildLLMTable() {
    return _buildTableCard(
      title: 'Federated LLM — All Models',
      subtitle: 'Text paradigm (5 encoders)',
      color: const Color(0xFF4169E1),
      headers: const ['Model', 'Fed F1', 'Diversity'],
      rows: const [
        ['DistilBERT', '0.595', '100%'],
        ['BERT-tiny', '0.604', '100%'],
        ['RoBERTa-tiny', '0.608', '100%'],
        ['ALBERT-tiny', '0.544', '100%'],
        ['MobileBERT', '0.636', '100%'],
      ],
      bestRow: 4,
    );
  }

  Widget _buildViTTable() {
    return _buildTableCard(
      title: 'Federated ViT — All Models',
      subtitle: 'Image paradigm (5 encoders)',
      color: const Color(0xFF228B22),
      headers: const ['Model', 'Fed F1', 'Diversity'],
      rows: const [
        ['ViT-Base', '0.857', '100%'],
        ['DeiT-tiny', '0.853', '100%'],
        ['Swin-tiny', '0.853', '100%'],
        ['ConvNeXT-tiny', '0.848', '100%'],
        ['EfficientNet', '0.853', '100%'],
      ],
      bestRow: 0,
    );
  }

  Widget _buildVLMTable() {
    return _buildTableCard(
      title: 'Federated VLM — All Fusions',
      subtitle: 'Multimodal paradigm (8 fusion strategies)',
      color: const Color(0xFFB22222),
      headers: const ['Fusion', 'Fed F1', 'Diversity'],
      rows: const [
        ['Concatenation', '0.848', '100%'],
        ['Gated', '0.811', '100%'],
        ['Flamingo', '0.802', '100%'],
        ['BLIP-2', '0.802', '100%'],
        ['CLIP', '0.797', '100%'],
        ['CoCa', '0.797', '100%'],
        ['Cross-Attention', '0.788', '100%'],
        ['Unified-IO', '0.783', '100%'],
      ],
      bestRow: 0,
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

  Widget _buildRAGCard() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AppTheme.cardDark,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: const Color(0xFF9C27B0).withValues(alpha: 0.3)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(width: 4, height: 20, decoration: BoxDecoration(color: const Color(0xFF9C27B0), borderRadius: BorderRadius.circular(2))),
              const SizedBox(width: 10),
              const Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text('Federated RAG Advisory', style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold, fontSize: 15)),
                    Text('Privacy-preserving retrieval + treatment generation (10 rounds)', style: TextStyle(color: Colors.white38, fontSize: 11)),
                  ],
                ),
              ),
            ],
          ),
          const SizedBox(height: 14),
          Row(
            children: const [
              Expanded(child: Text('Metric', style: TextStyle(color: Color(0xFF9C27B0), fontSize: 11, fontWeight: FontWeight.bold))),
              Expanded(child: Text('Value', style: TextStyle(color: Color(0xFF9C27B0), fontSize: 11, fontWeight: FontWeight.bold))),
              Expanded(child: Text('Target', style: TextStyle(color: Color(0xFF9C27B0), fontSize: 11, fontWeight: FontWeight.bold))),
            ],
          ),
          const SizedBox(height: 6),
          Container(height: 1, color: Colors.white12),
          const SizedBox(height: 6),
          _buildRAGRow('Macro F1 *', '$RAG_MACRO_F1', '—', highlight: true),
          _buildRAGRow('KB Coverage', '$RAG_KB_COVERAGE', '1.00', highlight: true),
          _buildRAGRow('Recall@5', '$RAG_RECALL_AT_5', '≥0.85', highlight: false),
          _buildRAGRow('MRR', '$RAG_MRR', '≥0.70', highlight: false),
          _buildRAGRow('NDCG@5', '$RAG_NDCG_AT_5', '≥0.70', highlight: false),
          _buildRAGRow('Embedding Drift', '$RAG_EMBEDDING_DRIFT', '< 0.2', highlight: false),
          const SizedBox(height: 10),
          Container(
            padding: const EdgeInsets.all(10),
            decoration: BoxDecoration(
              color: Colors.amber.withValues(alpha: 0.08),
              borderRadius: BorderRadius.circular(8),
              border: Border.all(color: Colors.amber.withValues(alpha: 0.25)),
            ),
            child: const Text(
              '* F1 = 1.0 is on synthetic template text (not real data). Each stress class uses a fixed keyword pattern, making classification trivially easy. The low Recall@5 = 0.129 is the honest real-world retrieval signal — the retriever learns pattern matching, not deep semantic alignment.',
              style: TextStyle(color: Colors.amber, fontSize: 11, height: 1.4),
            ),
          ),
          const SizedBox(height: 6),
          Container(
            padding: const EdgeInsets.all(10),
            decoration: BoxDecoration(
              color: const Color(0xFF9C27B0).withValues(alpha: 0.08),
              borderRadius: BorderRadius.circular(8),
            ),
            child: const Text(
              'Privacy: farm-local FAISS stores never leave the device. Joint training uses InfoNCE + BCE losses. EMA aggregation (μ=0.9) prevents embedding drift.',
              style: TextStyle(color: Colors.white54, fontSize: 11, height: 1.4),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildRAGRow(String metric, String value, String target, {required bool highlight}) {
    return Container(
      margin: const EdgeInsets.only(bottom: 4),
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 5),
      decoration: BoxDecoration(
        color: highlight ? const Color(0xFF9C27B0).withValues(alpha: 0.12) : Colors.transparent,
        borderRadius: BorderRadius.circular(6),
        border: highlight ? Border.all(color: const Color(0xFF9C27B0).withValues(alpha: 0.4)) : null,
      ),
      child: Row(
        children: [
          Expanded(child: Text(metric, style: TextStyle(color: highlight ? Colors.white : Colors.white70, fontSize: 12, fontWeight: highlight ? FontWeight.bold : FontWeight.normal))),
          Expanded(child: Text(value, style: TextStyle(color: highlight ? Colors.white : Colors.white70, fontSize: 12, fontWeight: highlight ? FontWeight.bold : FontWeight.normal))),
          Expanded(child: Text(target, style: const TextStyle(color: Colors.white38, fontSize: 12))),
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
          _buildDetailRow('Best Fed F1 (VLM-Concat)', '$BEST_MACRO_F1_FEDERATED'),
          _buildDetailRow('Best Cent F1 (VLM-Concat)', '$BEST_MACRO_F1_CENTRALIZED'),
          _buildDetailRow('VLM gain over LLM', '+$MULTIMODAL_IMPROVEMENT% Fed F1'),
          _buildDetailRow('RAG KB Coverage', '$RAG_KB_COVERAGE'),
          _buildDetailRow('RAG Recall@5', '$RAG_RECALL_AT_5'),
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
