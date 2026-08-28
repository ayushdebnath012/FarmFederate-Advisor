import 'package:flutter/material.dart';
import '../theme/app_theme.dart';
import '../constants.dart';

class PerformanceResultsScreen extends StatelessWidget {
  final String apiBase;

  const PerformanceResultsScreen({super.key, required this.apiBase});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppTheme.backgroundDark,
      appBar: AppBar(title: const Text('Performance Results')),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            _buildOverviewCard(),
            const SizedBox(height: 16),
            _buildGroupTable(),
            const SizedBox(height: 16),
            _buildTextTable(),
            const SizedBox(height: 16),
            _buildImageTable(),
            const SizedBox(height: 16),
            _buildFusionTable(),
            const SizedBox(height: 16),
            _buildLookupCard(),
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
            'FarmFederate - Tea Disease Results',
            style: TextStyle(
              color: Colors.white,
              fontWeight: FontWeight.bold,
              fontSize: 18,
            ),
            textAlign: TextAlign.center,
          ),
          SizedBox(height: 4),
          Text(
            'Network Text | Network Image | Network Fusion | Network Lookup\nK=3 clients | T=8 rounds | E=3 local passes | secure weighted update | avg 102.1% retention',
            style: TextStyle(color: Colors.white70, fontSize: 12),
            textAlign: TextAlign.center,
          ),
        ],
      ),
    );
  }

  Widget _buildGroupTable() {
    return _buildTableCard(
      title: 'Cross-Group Comparison',
      subtitle: 'Best profile per group',
      color: AppTheme.primaryGreen,
      headers: const ['Group', 'Best Profile', 'Net F1', 'Retention'],
      rows: const [
        ['Net. Text', 'Text Core A', '0.460', '107.7%'],
        ['Net. Image', 'Image Core A', '0.886', '100.0%'],
        ['Net. Fusion', 'Fusion Core A', '0.861', '98.6%'],
      ],
      bestRow: 2,
    );
  }

  Widget _buildTextTable() {
    return _buildTableCard(
      title: 'Text Profiles - Baseline Review',
      subtitle: 'Text group (5 profiles, baseline review)',
      color: const Color(0xFF4169E1),
      headers: const ['Profile', 'Base F1', 'Diversity'],
      rows: const [
        ['Text Core A', '0.487', '100%'],
        ['Text Core B', '0.463', '100%'],
        ['Text Core C', '0.450', '100%'],
        ['Text Core D', '0.433', '100%'],
        ['Text Core E', '0.417', '100%'],
      ],
      bestRow: 0,
    );
  }

  Widget _buildImageTable() {
    return _buildTableCard(
      title: 'Image Profiles - Baseline Review',
      subtitle: 'Image group (5 profiles, baseline review)',
      color: const Color(0xFF228B22),
      headers: const ['Profile', 'Base F1', 'Diversity'],
      rows: const [
        ['Image Core A', '0.911', '100%'],
        ['Image Core B', '0.899', '100%'],
        ['Image Core C', '0.873', '100%'],
        ['Image Core D', '0.873', '100%'],
        ['Image Core E', '0.861', '100%'],
      ],
      bestRow: 0,
    );
  }

  Widget _buildFusionTable() {
    return _buildTableCard(
      title: 'Fusion Strategies - Baseline Review',
      subtitle: 'Multimodal group (8 fusion strategies, baseline review)',
      color: const Color(0xFFB22222),
      headers: const ['Fusion', 'Base F1', 'Params (M)'],
      rows: const [
        ['Fusion Core A', '0.949', '15.7M'],
        ['Fusion Core B', '0.937', '15.9M'],
        ['Fusion Core C', '0.924', '15.9M'],
        ['Fusion Core D', '0.911', '16.1M'],
        ['Fusion Core E', '0.911', '17.2M'],
        ['Gated', '0.899', '15.8M'],
        ['Fusion Core G', '0.899', '16.2M'],
        ['Concat', '0.886', '15.6M'],
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
              Container(
                width: 4,
                height: 20,
                decoration: BoxDecoration(
                  color: color,
                  borderRadius: BorderRadius.circular(2),
                ),
              ),
              const SizedBox(width: 10),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      title,
                      style: const TextStyle(
                        color: Colors.white,
                        fontWeight: FontWeight.bold,
                        fontSize: 15,
                      ),
                    ),
                    Text(
                      subtitle,
                      style: const TextStyle(
                        color: Colors.white38,
                        fontSize: 11,
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
          const SizedBox(height: 14),
          Row(
            children: headers
                .map(
                  (h) => Expanded(
                    child: Text(
                      h,
                      style: TextStyle(
                        color: color,
                        fontSize: 11,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ),
                )
                .toList(),
          ),
          const SizedBox(height: 6),
          Container(height: 1, color: Colors.white12),
          const SizedBox(height: 6),
          ...rows.asMap().entries.map((entry) {
            final i = entry.key;
            final row = entry.value;
            final isBest = i == bestRow;
            return Container(
              margin: const EdgeInsets.only(bottom: 4),
              padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 5),
              decoration: BoxDecoration(
                color:
                    isBest ? color.withValues(alpha: 0.12) : Colors.transparent,
                borderRadius: BorderRadius.circular(6),
                border: isBest
                    ? Border.all(color: color.withValues(alpha: 0.4))
                    : null,
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
                            fontWeight:
                                isBest ? FontWeight.bold : FontWeight.normal,
                          ),
                        ),
                        if (isBest && isFirst) ...[
                          const SizedBox(width: 4),
                          Container(
                            padding: const EdgeInsets.symmetric(
                              horizontal: 4,
                              vertical: 1,
                            ),
                            decoration: BoxDecoration(
                              color: color.withValues(alpha: 0.3),
                              borderRadius: BorderRadius.circular(3),
                            ),
                            child: Text(
                              'BEST',
                              style: TextStyle(
                                color: color,
                                fontSize: 8,
                                fontWeight: FontWeight.bold,
                              ),
                            ),
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

  Widget _buildLookupCard() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AppTheme.cardDark,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(
          color: const Color(0xFF9C27B0).withValues(alpha: 0.3),
        ),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                width: 4,
                height: 20,
                decoration: BoxDecoration(
                  color: const Color(0xFF9C27B0),
                  borderRadius: BorderRadius.circular(2),
                ),
              ),
              const SizedBox(width: 10),
              const Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'Network Lookup Advisory',
                      style: TextStyle(
                        color: Colors.white,
                        fontWeight: FontWeight.bold,
                        fontSize: 15,
                      ),
                    ),
                    Text(
                      'Privacy-preserving retrieval + treatment guidance (10 rounds)',
                      style: TextStyle(color: Colors.white38, fontSize: 11),
                    ),
                  ],
                ),
              ),
            ],
          ),
          const SizedBox(height: 14),
          const Row(
            children: [
              Expanded(
                child: Text(
                  'Metric',
                  style: TextStyle(
                    color: Color(0xFF9C27B0),
                    fontSize: 11,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
              Expanded(
                child: Text(
                  'Value',
                  style: TextStyle(
                    color: Color(0xFF9C27B0),
                    fontSize: 11,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
              Expanded(
                child: Text(
                  'Target',
                  style: TextStyle(
                    color: Color(0xFF9C27B0),
                    fontSize: 11,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 6),
          Container(height: 1, color: Colors.white12),
          const SizedBox(height: 6),
          _buildLookupRow(
            'Macro F1 *',
            '$LOOKUP_MACRO_F1',
            '-',
            highlight: true,
          ),
          _buildLookupRow(
            'KB Coverage',
            '$LOOKUP_KB_COVERAGE',
            '1.00',
            highlight: true,
          ),
          _buildLookupRow(
            'Recall@5',
            '$LOOKUP_RECALL_AT_5',
            '>=0.85',
            highlight: false,
          ),
          _buildLookupRow('MRR', '$LOOKUP_MRR', '>=0.70', highlight: false),
          _buildLookupRow(
            'NDCG@5',
            '$LOOKUP_NDCG_AT_5',
            '>=0.70',
            highlight: false,
          ),
          _buildLookupRow(
            'Index Drift',
            '$LOOKUP_INDEX_DRIFT',
            '< 0.2',
            highlight: false,
          ),
          const SizedBox(height: 10),
          Container(
            padding: const EdgeInsets.all(10),
            decoration: BoxDecoration(
              color: Colors.amber.withValues(alpha: 0.08),
              borderRadius: BorderRadius.circular(8),
              border: Border.all(color: Colors.amber.withValues(alpha: 0.25)),
            ),
            child: const Text(
              '* F1 = 1.0 is on controlled evaluation text, not field text. Each tea disease class uses a fixed keyword pattern, making sorting easy. The low Recall@5 = 0.129 is the honest real-world retrieval signal - the lookup uses pattern matching rather than field context.',
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
              'Privacy: local estate indexes stay on the device. Secure weighted updates keep the shared view stable.',
              style: TextStyle(
                color: Colors.white54,
                fontSize: 11,
                height: 1.4,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildLookupRow(
    String metric,
    String value,
    String target, {
    required bool highlight,
  }) {
    return Container(
      margin: const EdgeInsets.only(bottom: 4),
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 5),
      decoration: BoxDecoration(
        color: highlight
            ? const Color(0xFF9C27B0).withValues(alpha: 0.12)
            : Colors.transparent,
        borderRadius: BorderRadius.circular(6),
        border: highlight
            ? Border.all(color: const Color(0xFF9C27B0).withValues(alpha: 0.4))
            : null,
      ),
      child: Row(
        children: [
          Expanded(
            child: Text(
              metric,
              style: TextStyle(
                color: highlight ? Colors.white : Colors.white70,
                fontSize: 12,
                fontWeight: highlight ? FontWeight.bold : FontWeight.normal,
              ),
            ),
          ),
          Expanded(
            child: Text(
              value,
              style: TextStyle(
                color: highlight ? Colors.white : Colors.white70,
                fontSize: 12,
                fontWeight: highlight ? FontWeight.bold : FontWeight.normal,
              ),
            ),
          ),
          Expanded(
            child: Text(
              target,
              style: const TextStyle(color: Colors.white38, fontSize: 12),
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
            'Network Setup',
            style: TextStyle(
              color: Colors.white,
              fontWeight: FontWeight.bold,
              fontSize: 16,
            ),
          ),
          const SizedBox(height: 12),
          _buildDetailRow('Farm Groups (K)', '$NUM_CLIENTS'),
          _buildDetailRow('Network Rounds (T)', '$NETWORK_ROUNDS'),
          _buildDetailRow('Local Passes (E)', '$LOCAL_PASSES'),
          _buildDetailRow('Aggregation', 'weighted farm update'),
          _buildDetailRow(
            'Data Split',
            'Mixed farm split (alpha=$NON_IID_ALPHA)',
          ),
          _buildDetailRow('Avg Net/Base Retention', '$NETWORK_BASELINE_RATIO%'),
          _buildDetailRow(
            'Best Net F1 (Fusion Core A)',
            '$BEST_MACRO_F1_NETWORK',
          ),
          _buildDetailRow(
            'Best Base F1 (Fusion Core A)',
            '$BEST_MACRO_F1_BASELINE',
          ),
          _buildDetailRow(
            'Fusion gain over Text',
            '+$MULTIMODAL_IMPROVEMENT% Net F1',
          ),
          _buildDetailRow('Lookup KB Coverage', '$LOOKUP_KB_COVERAGE'),
          _buildDetailRow('Lookup Recall@5', '$LOOKUP_RECALL_AT_5'),
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
